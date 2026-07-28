"""Fine-tune an Xeno-canto DS-CNN with isolated DB3V few-shot support.

The support pool is split by region and class for epoch selection.  DB3V
held-out data is deliberately not accepted by this script.  After selecting an
epoch without held-out access, the base model is reloaded and trained on the
complete support pool for that fixed number of epochs.

The ablation policies use explicit names so that the trainable parameter scope
is auditable:

``head_only``
    Train only the final softmax layer.
``bn_head``
    Train all BatchNormalization layers and the final softmax layer.
``bn_head_replay``
    Use the same trainable layers as ``bn_head`` and mix a class-balanced
    Xeno-canto training replay buffer with DB3V support.
``full``
    Train every layer, including BatchNormalization layers.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.utils import to_categorical


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SLICES_PER_RECORDING = 8
POLICIES = (
    "head_only",
    "bn_head",
    "bn_head_replay",
    "full",
)
REPLAY_POLICIES = ("bn_head_replay",)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-dir", type=Path, required=True)
    parser.add_argument("--support-dir", type=Path, required=True)
    parser.add_argument("--xeno-dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="DS_CNN_Model")
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-recordings-per-stratum", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--replay-ratio",
        type=float,
        default=1.0,
        help=(
            "Number of Xeno-canto replay slices per DB3V support slice. "
            "Used only by bn_head_replay."
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(resolved)


def recording_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    recording_index: np.ndarray,
    num_classes: int,
) -> dict[str, float | int]:
    recording_count = int(recording_index.max()) + 1
    sums = np.zeros((recording_count, num_classes), dtype=np.float64)
    counts = np.bincount(recording_index, minlength=recording_count)
    recording_labels = np.full(recording_count, -1, dtype=np.int64)
    np.add.at(sums, recording_index, probabilities)
    for index, label in zip(recording_index, labels, strict=True):
        if recording_labels[index] not in {-1, int(label)}:
            raise ValueError("A recording contains inconsistent labels.")
        recording_labels[index] = int(label)
    if np.any(counts == 0) or np.any(recording_labels < 0):
        raise ValueError("A recording is missing slices or a label.")
    means = sums / counts[:, np.newaxis]
    predictions = means.argmax(axis=1)
    top3 = np.argpartition(means, -3, axis=1)[:, -3:]
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (recording_labels, predictions), 1)
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    true_positive = np.diag(confusion)
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros(num_classes, dtype=np.float64),
        where=support > 0,
    )
    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros(num_classes, dtype=np.float64),
        where=predicted > 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(num_classes, dtype=np.float64),
        where=(precision + recall) > 0,
    )
    return {
        "recordings": recording_count,
        "accuracy": float(np.mean(predictions == recording_labels)),
        "balanced_accuracy": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "top_3_accuracy": float(
            np.mean(np.any(top3 == recording_labels[:, np.newaxis], axis=1))
        ),
    }


def recording_rows(recordings: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            np.arange(
                index * SLICES_PER_RECORDING,
                (index + 1) * SLICES_PER_RECORDING,
            )
            for index in recordings
        ]
    )


def support_split(
    labels: np.ndarray,
    regions: np.ndarray,
    validation_per_stratum: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(labels) % SLICES_PER_RECORDING:
        raise ValueError("Support slices are not divisible into eight-slice recordings.")
    recording_labels = labels.reshape(-1, SLICES_PER_RECORDING)
    recording_regions = regions.reshape(-1, SLICES_PER_RECORDING)
    if np.any(recording_labels != recording_labels[:, :1]):
        raise ValueError("A support recording contains inconsistent class labels.")
    if np.any(recording_regions != recording_regions[:, :1]):
        raise ValueError("A support recording contains inconsistent regions.")
    recording_labels = recording_labels[:, 0]
    recording_regions = recording_regions[:, 0]
    random = np.random.default_rng(seed)
    train: list[int] = []
    validation: list[int] = []
    for region in sorted(np.unique(recording_regions)):
        for label in sorted(np.unique(recording_labels)):
            candidates = np.flatnonzero(
                (recording_regions == region) & (recording_labels == label)
            )
            if len(candidates) <= validation_per_stratum:
                raise ValueError(
                    f"Region {region}, label {label} has only {len(candidates)} recordings."
                )
            random.shuffle(candidates)
            validation.extend(candidates[:validation_per_stratum].tolist())
            train.extend(candidates[validation_per_stratum:].tolist())
    return np.asarray(sorted(train)), np.asarray(sorted(validation))


def configure_trainable(model: tf.keras.Model, policy: str) -> list[str]:
    for layer in model.layers:
        layer.trainable = False
    if policy == "head_only":
        model.layers[-1].trainable = True
        if not isinstance(model.layers[-1], tf.keras.layers.Dense):
            raise ValueError("The final model layer must be the classification Dense layer.")
        return [model.layers[-1].name]
    if policy in {"bn_head", "bn_head_replay"}:
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True
        model.layers[-1].trainable = True
        if not isinstance(model.layers[-1], tf.keras.layers.Dense):
            raise ValueError("The final model layer must be the classification Dense layer.")
        return [layer.name for layer in model.layers if layer.trainable]
    if policy == "full":
        for layer in model.layers:
            layer.trainable = True
        return [layer.name for layer in model.layers if layer.trainable]
    raise ValueError(f"Unsupported strict fine-tuning policy: {policy}")


def balanced_replay_rows(
    labels: np.ndarray,
    requested_rows: int,
    num_classes: int,
    seed: int,
) -> np.ndarray:
    """Select a deterministic, approximately class-balanced replay buffer."""
    if requested_rows < 1:
        return np.empty(0, dtype=np.int64)
    random = np.random.default_rng(seed)
    base, remainder = divmod(requested_rows, num_classes)
    selected: list[np.ndarray] = []
    for label in range(num_classes):
        candidates = np.flatnonzero(labels == label)
        count = base + int(label < remainder)
        if len(candidates) < count:
            raise ValueError(
                f"Xeno class {label} has {len(candidates)} slices, fewer than "
                f"the requested replay count {count}."
            )
        selected.append(random.choice(candidates, size=count, replace=False))
    rows = np.concatenate(selected).astype(np.int64, copy=False)
    random.shuffle(rows)
    return rows


def mix_with_replay(
    support_data: np.ndarray,
    support_labels: np.ndarray,
    xeno_train_data: np.ndarray,
    xeno_train_labels: np.ndarray,
    replay_ratio: float,
    num_classes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    replay_count = int(round(len(support_labels) * replay_ratio))
    replay_rows = balanced_replay_rows(
        xeno_train_labels,
        replay_count,
        num_classes,
        seed,
    )
    replay_data = np.asarray(xeno_train_data[replay_rows], dtype=np.float32)[
        ..., np.newaxis
    ]
    replay_labels = np.asarray(xeno_train_labels[replay_rows], dtype=np.int64)
    mixed_data = np.concatenate((support_data, replay_data))
    mixed_labels = np.concatenate((support_labels, replay_labels))
    random = np.random.default_rng(seed + 1)
    order = random.permutation(len(mixed_labels))
    return mixed_data[order], mixed_labels[order], replay_rows


def predict_recordings(
    model: tf.keras.Model,
    data: np.ndarray,
    labels: np.ndarray,
    recording_index: np.ndarray,
    num_classes: int,
) -> dict[str, float | int]:
    probabilities = model.predict(data, batch_size=128, verbose=0)
    return recording_metrics(labels, probabilities, recording_index, num_classes)


class AdaptationMetrics(Callback):
    def __init__(
        self,
        support_data: np.ndarray,
        support_labels: np.ndarray,
        support_index: np.ndarray,
        xeno_data: np.ndarray,
        xeno_labels: np.ndarray,
        xeno_index: np.ndarray,
        num_classes: int,
        baseline_xeno_f1: float,
    ) -> None:
        super().__init__()
        self.support_data = support_data
        self.support_labels = support_labels
        self.support_index = support_index
        self.xeno_data = xeno_data
        self.xeno_labels = xeno_labels
        self.xeno_index = xeno_index
        self.num_classes = num_classes
        self.baseline_xeno_f1 = baseline_xeno_f1

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        logs = logs if logs is not None else {}
        support = predict_recordings(
            self.model,
            self.support_data,
            self.support_labels,
            self.support_index,
            self.num_classes,
        )
        xeno = predict_recordings(
            self.model,
            self.xeno_data,
            self.xeno_labels,
            self.xeno_index,
            self.num_classes,
        )
        retention = xeno["macro_f1"] / self.baseline_xeno_f1
        score = support["macro_f1"] * min(1.0, max(0.0, retention))
        logs["val_support_recording_macro_f1"] = support["macro_f1"]
        logs["val_xeno_recording_macro_f1"] = xeno["macro_f1"]
        logs["val_xeno_retention"] = retention
        logs["val_adaptation_score"] = score
        print(
            f"Epoch {epoch + 1}: support_macro_f1={support['macro_f1']:.4f}, "
            f"xeno_macro_f1={xeno['macro_f1']:.4f}, retention={retention:.4f}, "
            f"adaptation_score={score:.4f}",
            flush=True,
        )


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


def main() -> None:
    args = arguments()
    if (
        args.learning_rate <= 0
        or args.epochs < 1
        or args.patience < 1
        or args.batch_size < 1
        or args.validation_recordings_per_stratum < 1
        or args.replay_ratio < 0
    ):
        raise ValueError("Learning rate/count arguments must be positive and replay ratio nonnegative.")
    if args.policy in REPLAY_POLICIES and args.replay_ratio <= 0:
        raise ValueError("Replay policies require --replay-ratio greater than zero.")
    tf.keras.utils.set_random_seed(args.seed)

    model_path = args.base_model_dir / f"{args.model_name}.h5"
    base_labels = load_json(args.base_model_dir / f"{args.model_name}.labels.json")
    support_labels_map = load_json(args.support_dir / "label_map.json")
    xeno_labels_map = load_json(args.xeno_dataset_dir / "label_map.json")
    split_manifest = load_json(args.support_dir / "split_manifest.json")
    if base_labels != support_labels_map or base_labels != xeno_labels_map:
        raise ValueError("Base, support, and Xeno label maps differ.")
    num_classes = len(base_labels)
    shots_per_region_class = int(
        split_manifest["requested_shots_per_region_class"]
    )

    support_data = np.load(args.support_dir / "support_data.npy").astype(np.float32)
    support_labels = np.load(args.support_dir / "support_label.npy").astype(np.int64)
    support_regions = np.load(args.support_dir / "support_region.npy").astype(np.int64)
    support_manifest = load_json(args.support_dir / "support_manifest.json")
    if len(support_data) != len(support_labels) or len(support_labels) != len(
        support_regions
    ):
        raise ValueError("Support data, labels, and regions have different lengths.")
    if len(support_manifest) * SLICES_PER_RECORDING != len(support_data):
        raise ValueError("Support manifest and slice counts differ.")
    expected_support_recordings = shots_per_region_class * num_classes * 3
    support_shortfall = expected_support_recordings - len(support_manifest)
    recorded_shortfall = int(
        split_manifest.get("support_shortfall_recordings", support_shortfall)
    )
    if support_shortfall < 0 or recorded_shortfall != support_shortfall:
        raise ValueError(
            "Support recording count is inconsistent with the split manifest: "
            f"expected at most {expected_support_recordings}, found "
            f"{len(support_manifest)}, recorded shortfall {recorded_shortfall}."
        )

    train_recordings, validation_recordings = support_split(
        support_labels,
        support_regions,
        args.validation_recordings_per_stratum,
        args.seed,
    )
    train_rows = recording_rows(train_recordings)
    validation_rows = recording_rows(validation_recordings)
    train_data = support_data[train_rows][..., np.newaxis]
    train_labels = support_labels[train_rows]
    validation_data = support_data[validation_rows][..., np.newaxis]
    validation_labels = support_labels[validation_rows]
    validation_index = np.repeat(
        np.arange(len(validation_recordings)), SLICES_PER_RECORDING
    )
    full_support_data = support_data[..., np.newaxis]
    full_support_index = np.repeat(
        np.arange(len(support_manifest)), SLICES_PER_RECORDING
    )

    xeno_data = np.load(args.xeno_dataset_dir / "validation_data.npy").astype(
        np.float32
    )[..., np.newaxis]
    xeno_labels = np.load(args.xeno_dataset_dir / "validation_label.npy").astype(
        np.int64
    )
    xeno_index = np.load(
        args.xeno_dataset_dir / "validation_recording_index.npy"
    ).astype(np.int64)
    xeno_train_data: np.ndarray | None = None
    xeno_train_labels: np.ndarray | None = None
    selection_replay_rows = np.empty(0, dtype=np.int64)
    final_replay_rows = np.empty(0, dtype=np.int64)
    selection_fit_data = train_data
    selection_fit_labels = train_labels
    final_fit_data = full_support_data
    final_fit_labels = support_labels
    if args.policy in REPLAY_POLICIES:
        xeno_train_data = np.load(
            args.xeno_dataset_dir / "train_data.npy",
            mmap_mode="r",
        )
        xeno_train_labels = np.load(
            args.xeno_dataset_dir / "train_label.npy",
            mmap_mode="r",
        )
        if tuple(xeno_train_data.shape[1:]) != tuple(support_data.shape[1:]):
            raise ValueError("Xeno replay and DB3V support feature shapes differ.")
        selection_fit_data, selection_fit_labels, selection_replay_rows = (
            mix_with_replay(
                train_data,
                train_labels,
                xeno_train_data,
                xeno_train_labels,
                args.replay_ratio,
                num_classes,
                args.seed + 10_000,
            )
        )
        final_fit_data, final_fit_labels, final_replay_rows = mix_with_replay(
            full_support_data,
            support_labels,
            xeno_train_data,
            xeno_train_labels,
            args.replay_ratio,
            num_classes,
            args.seed + 20_000,
        )

    selection_model = tf.keras.models.load_model(model_path, compile=False)
    if tuple(selection_model.input_shape[1:]) != tuple(train_data.shape[1:]):
        raise ValueError(
            f"Model input {selection_model.input_shape} does not match support "
            f"{train_data.shape}."
        )
    if selection_model.layers[-1].activation.__name__ != "softmax":
        raise ValueError("DB3V adaptation requires a single-label softmax model.")
    baseline_support = predict_recordings(
        selection_model,
        validation_data,
        validation_labels,
        validation_index,
        num_classes,
    )
    baseline_xeno = predict_recordings(
        selection_model,
        xeno_data,
        xeno_labels,
        xeno_index,
        num_classes,
    )
    trainable_layers = configure_trainable(selection_model, args.policy)
    compile_model(selection_model, args.learning_rate)
    trainable_parameters = int(
        sum(np.prod(variable.shape) for variable in selection_model.trainable_weights)
    )
    total_parameters = int(selection_model.count_params())
    adaptation_metrics = AdaptationMetrics(
        validation_data,
        validation_labels,
        validation_index,
        xeno_data,
        xeno_labels,
        xeno_index,
        num_classes,
        float(baseline_xeno["macro_f1"]),
    )
    early_stopping = EarlyStopping(
        monitor="val_adaptation_score",
        mode="max",
        patience=args.patience,
        restore_best_weights=True,
    )
    selection_history = selection_model.fit(
        selection_fit_data,
        to_categorical(selection_fit_labels, num_classes),
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(
            validation_data,
            to_categorical(validation_labels, num_classes),
        ),
        callbacks=[adaptation_metrics, early_stopping],
        verbose=2,
    )
    scores = np.asarray(selection_history.history["val_adaptation_score"])
    best_epoch = int(scores.argmax()) + 1
    selected_support = predict_recordings(
        selection_model,
        validation_data,
        validation_labels,
        validation_index,
        num_classes,
    )
    selected_xeno = predict_recordings(
        selection_model,
        xeno_data,
        xeno_labels,
        xeno_index,
        num_classes,
    )
    selected_retention = selected_xeno["macro_f1"] / baseline_xeno["macro_f1"]
    selected_score = selected_support["macro_f1"] * min(
        1.0, max(0.0, selected_retention)
    )

    del selection_model
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(args.seed)
    final_model = tf.keras.models.load_model(model_path, compile=False)
    configure_trainable(final_model, args.policy)
    compile_model(final_model, args.learning_rate)
    final_history = final_model.fit(
        final_fit_data,
        to_categorical(final_fit_labels, num_classes),
        batch_size=args.batch_size,
        epochs=best_epoch,
        verbose=2,
    )
    final_support = predict_recordings(
        final_model,
        full_support_data,
        support_labels,
        full_support_index,
        num_classes,
    )
    final_xeno = predict_recordings(
        final_model,
        xeno_data,
        xeno_labels,
        xeno_index,
        num_classes,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_model.save(args.output_dir / f"{args.model_name}.h5")
    (args.output_dir / f"{args.model_name}.labels.json").write_text(
        json.dumps(base_labels, indent=2), encoding="utf-8"
    )
    history_report = {
        "selection": selection_history.history,
        "final_all_support": final_history.history,
    }
    (args.output_dir / f"{args.model_name}.fewshot_history.json").write_text(
        json.dumps(history_report, indent=2), encoding="utf-8"
    )
    report = {
        "experiment": (
            f"DB3V {shots_per_region_class}-shot regional adaptation from "
            "Xeno-canto DS-CNN"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "heldout_access_during_training": False,
        "base_model": relative_path(model_path),
        "support_dir": relative_path(args.support_dir),
        "xeno_validation_dir": relative_path(args.xeno_dataset_dir),
        "label_map": base_labels,
        "protocol": {
            "support_recordings": len(support_manifest),
            "support_slices": len(support_data),
            "shots_per_region_class": shots_per_region_class,
            "requested_support_recordings": expected_support_recordings,
            "support_shortfall_recordings": support_shortfall,
            "support_recordings_by_region_class": split_manifest.get(
                "support_recordings_by_region_class"
            ),
            "selection_train_recordings": len(train_recordings),
            "selection_validation_recordings": len(validation_recordings),
            "selection_validation_per_region_class": (
                args.validation_recordings_per_stratum
            ),
            "seed": args.seed,
            "selection_validation_paths": [
                support_manifest[index]["path"]
                for index in validation_recordings.tolist()
            ],
            "policy": args.policy,
            "learning_rate": args.learning_rate,
            "maximum_selection_epochs": args.epochs,
            "patience": args.patience,
            "selected_epoch": best_epoch,
            "selection_metric": (
                "support validation recording macro-F1 multiplied by capped "
                "Xeno validation macro-F1 retention"
            ),
            "final_training": (
                "Reload the original Xeno model and train on all "
                f"{len(support_manifest)} support recordings for the selected "
                "fixed epoch count."
            ),
            "trainable_layers": trainable_layers,
            "trainable_parameters": trainable_parameters,
            "total_parameters": total_parameters,
            "batch_normalization_trainable": any(
                isinstance(layer, tf.keras.layers.BatchNormalization)
                and layer.trainable
                for layer in final_model.layers
            ),
            "replay": {
                "enabled": args.policy in REPLAY_POLICIES,
                "source": (
                    relative_path(args.xeno_dataset_dir / "train_data.npy")
                    if args.policy in REPLAY_POLICIES
                    else None
                ),
                "source_partition": (
                    "Xeno-canto baseline training split"
                    if args.policy in REPLAY_POLICIES
                    else None
                ),
                "class_balanced": args.policy in REPLAY_POLICIES,
                "ratio_to_db3v_support_slices": (
                    args.replay_ratio if args.policy in REPLAY_POLICIES else 0.0
                ),
                "selection_replay_slices": len(selection_replay_rows),
                "final_replay_slices": len(final_replay_rows),
                "selection_sampling_seed": (
                    args.seed + 10_000 if args.policy in REPLAY_POLICIES else None
                ),
                "final_sampling_seed": (
                    args.seed + 20_000 if args.policy in REPLAY_POLICIES else None
                ),
                "heldout_or_validation_used_for_replay": False,
            },
        },
        "baseline_before_selection": {
            "support_validation": baseline_support,
            "xeno_validation": baseline_xeno,
        },
        "selected_split_model": {
            "support_validation": selected_support,
            "xeno_validation": selected_xeno,
            "xeno_macro_f1_retention": selected_retention,
            "adaptation_score": selected_score,
        },
        "final_all_support_model": {
            "support_training": final_support,
            "xeno_validation": final_xeno,
            "xeno_macro_f1_retention": (
                final_xeno["macro_f1"] / baseline_xeno["macro_f1"]
            ),
        },
    }
    (args.output_dir / f"{args.model_name}.fewshot.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
