"""Fine-tune an Xeno-canto DS-CNN with isolated BirdSet SSW support.

BirdSet support remains multi-label. Model selection uses a fixed split by
original long recording plus Xeno-canto validation retention; the independent
BirdSet held-out directory is deliberately not accepted by this script.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, EarlyStopping


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RECORDING_PATTERN = re.compile(r"^(.*)_\d+_\d+\.ogg$")
POLICIES = ("head", "last_block", "all")


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
    parser.add_argument("--validation-recordings", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
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
            raise ValueError("A Xeno recording contains inconsistent labels.")
        recording_labels[index] = int(label)
    if np.any(counts == 0) or np.any(recording_labels < 0):
        raise ValueError("A Xeno recording is missing slices or a label.")
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


def clip_metrics(
    clip_labels: np.ndarray,
    probabilities: np.ndarray,
    clip_index: np.ndarray,
) -> dict[str, Any]:
    clip_count = len(clip_labels)
    if clip_index.min() != 0 or clip_index.max() + 1 != clip_count:
        raise ValueError("Clip indices are not contiguous.")
    sums = np.zeros((clip_count, clip_labels.shape[1]), dtype=np.float64)
    counts = np.bincount(clip_index, minlength=clip_count)
    np.add.at(sums, clip_index, probabilities)
    means = sums / counts[:, np.newaxis]
    threshold_predictions = means >= 0.5
    true_positive = np.logical_and(threshold_predictions, clip_labels).sum(axis=0)
    support = clip_labels.sum(axis=0)
    predicted = threshold_predictions.sum(axis=0)
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros_like(support, dtype=np.float64),
        where=support > 0,
    )
    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros_like(support, dtype=np.float64),
        where=predicted > 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(support, dtype=np.float64),
        where=(precision + recall) > 0,
    )
    supported = support > 0
    top1 = means.argmax(axis=1)
    top3 = np.argpartition(means, -3, axis=1)[:, -3:]
    rows = np.arange(clip_count)
    return {
        "clips": clip_count,
        "supported_classes": np.flatnonzero(supported).astype(int).tolist(),
        "positive_clips_by_class": support.astype(int).tolist(),
        "top_1_any_target_accuracy": float(
            np.mean(clip_labels[rows, top1] > 0)
        ),
        "top_3_any_target_accuracy": float(
            np.mean(np.any(np.take_along_axis(clip_labels, top3, axis=1), axis=1))
        ),
        "threshold_subset_accuracy": float(
            np.mean(np.all(threshold_predictions == clip_labels, axis=1))
        ),
        "threshold_supported_macro_f1": float(np.mean(f1[supported])),
    }


def recording_groups(clips: list[dict]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, clip in enumerate(clips):
        match = RECORDING_PATTERN.match(Path(clip["filepath"]).name)
        if match is None:
            raise ValueError(f"Cannot group BirdSet clip {clip['filepath']!r}.")
        groups[match.group(1)].append(index)
    return dict(groups)


def choose_validation_groups(
    groups: dict[str, list[int]],
    clip_labels: np.ndarray,
    count: int,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    names = sorted(groups)
    if count < 1 or count >= len(names):
        raise ValueError("--validation-recordings must leave at least one train group.")
    target_clips = round(len(clip_labels) * 0.2)
    best_score: tuple[int, int] | None = None
    best_names: tuple[str, ...] | None = None
    for candidate in combinations(names, count):
        validation_clips = np.asarray(
            sorted(index for name in candidate for index in groups[name]),
            dtype=np.int64,
        )
        train_clips = np.asarray(
            sorted(index for name in names if name not in candidate for index in groups[name]),
            dtype=np.int64,
        )
        if np.any(clip_labels[train_clips].sum(axis=0) == 0):
            continue
        supported_validation_classes = int(
            np.sum(clip_labels[validation_clips].sum(axis=0) > 0)
        )
        score = (
            supported_validation_classes,
            -abs(len(validation_clips) - target_clips),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_names = candidate
    if best_names is None:
        raise ValueError("No grouped validation split preserves all train classes.")
    validation = np.asarray(
        sorted(index for name in best_names for index in groups[name]), dtype=np.int64
    )
    train = np.asarray(
        sorted(index for name in names if name not in best_names for index in groups[name]),
        dtype=np.int64,
    )
    return list(best_names), train, validation


def subset(
    features: np.ndarray,
    source_clip_index: np.ndarray,
    clip_labels: np.ndarray,
    selected_clips: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = np.flatnonzero(np.isin(source_clip_index, selected_clips))
    remap = np.full(len(clip_labels), -1, dtype=np.int64)
    remap[selected_clips] = np.arange(len(selected_clips))
    local_index = remap[source_clip_index[rows]]
    selected_labels = clip_labels[selected_clips].astype(np.float32)
    slice_labels = selected_labels[local_index]
    return (
        np.asarray(features[rows], dtype=np.float32)[..., np.newaxis],
        slice_labels,
        local_index,
        selected_labels,
    )


def balanced_multilabel_slice_weights(
    clip_labels: np.ndarray, slice_clip_index: np.ndarray
) -> np.ndarray:
    class_counts = clip_labels.sum(axis=0)
    if np.any(class_counts == 0) or np.any(clip_labels.sum(axis=1) == 0):
        raise ValueError("Training clips must cover every class and have a label.")
    inverse = 1.0 / class_counts
    clip_weights = (clip_labels * inverse).sum(axis=1) / clip_labels.sum(axis=1)
    clip_weights /= clip_weights.mean()
    slices_per_clip = np.bincount(slice_clip_index, minlength=len(clip_labels))
    return (
        clip_weights[slice_clip_index] / slices_per_clip[slice_clip_index]
    ).astype(np.float32)


def configure_trainable(model: tf.keras.Model, policy: str) -> list[str]:
    for layer in model.layers:
        layer.trainable = False
    if policy == "head":
        dense_indices = [
            index
            for index, layer in enumerate(model.layers)
            if isinstance(layer, tf.keras.layers.Dense)
        ]
        start = dense_indices[-2]
    elif policy == "last_block":
        block_indices = [
            index
            for index, layer in enumerate(model.layers)
            if isinstance(layer, tf.keras.layers.SeparableConv2D)
        ]
        start = block_indices[-1]
    else:
        start = 0
    for layer in model.layers[start:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
    return [layer.name for layer in model.layers if layer.trainable]


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="binary_accuracy")],
    )


def predict_clips(
    model: tf.keras.Model,
    data: np.ndarray,
    clip_labels: np.ndarray,
    clip_index: np.ndarray,
) -> dict[str, Any]:
    probabilities = model.predict(data, batch_size=128, verbose=0)
    return clip_metrics(clip_labels, probabilities, clip_index)


def predict_xeno(
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
        birdset_data: np.ndarray,
        birdset_labels: np.ndarray,
        birdset_index: np.ndarray,
        xeno_data: np.ndarray,
        xeno_labels: np.ndarray,
        xeno_index: np.ndarray,
        num_classes: int,
        baseline_xeno_f1: float,
    ) -> None:
        super().__init__()
        self.birdset_data = birdset_data
        self.birdset_labels = birdset_labels
        self.birdset_index = birdset_index
        self.xeno_data = xeno_data
        self.xeno_labels = xeno_labels
        self.xeno_index = xeno_index
        self.num_classes = num_classes
        self.baseline_xeno_f1 = baseline_xeno_f1

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        logs = logs if logs is not None else {}
        birdset = predict_clips(
            self.model,
            self.birdset_data,
            self.birdset_labels,
            self.birdset_index,
        )
        xeno = predict_xeno(
            self.model,
            self.xeno_data,
            self.xeno_labels,
            self.xeno_index,
            self.num_classes,
        )
        retention = xeno["macro_f1"] / self.baseline_xeno_f1
        score = birdset["top_1_any_target_accuracy"] * min(
            1.0, max(0.0, retention)
        )
        logs["val_birdset_clip_top1_any_target"] = birdset[
            "top_1_any_target_accuracy"
        ]
        logs["val_birdset_clip_supported_macro_f1"] = birdset[
            "threshold_supported_macro_f1"
        ]
        logs["val_xeno_recording_macro_f1"] = xeno["macro_f1"]
        logs["val_xeno_retention"] = retention
        logs["val_adaptation_score"] = score
        print(
            f"Epoch {epoch + 1}: birdset_top1={birdset['top_1_any_target_accuracy']:.4f}, "
            f"birdset_macro_f1={birdset['threshold_supported_macro_f1']:.4f}, "
            f"xeno_macro_f1={xeno['macro_f1']:.4f}, retention={retention:.4f}, "
            f"adaptation_score={score:.4f}",
            flush=True,
        )


def main() -> None:
    args = arguments()
    if (
        args.learning_rate <= 0
        or args.epochs < 1
        or args.patience < 1
        or args.batch_size < 1
    ):
        raise ValueError("Learning rate and count arguments must be positive.")
    tf.keras.utils.set_random_seed(args.seed)

    model_path = args.base_model_dir / f"{args.model_name}.h5"
    base_labels = load_json(args.base_model_dir / f"{args.model_name}.labels.json")
    support_labels_map = load_json(args.support_dir / "label_map.json")
    xeno_labels_map = load_json(args.xeno_dataset_dir / "label_map.json")
    if base_labels != support_labels_map or base_labels != xeno_labels_map:
        raise ValueError("Base, support, and Xeno label maps differ.")
    num_classes = len(base_labels)

    features = np.load(args.support_dir / "support_data.npy", mmap_mode="r")
    source_clip_index = np.load(args.support_dir / "support_clip_index.npy").astype(
        np.int64
    )
    clip_labels = np.load(
        args.support_dir / "support_clip_multilabel.npy"
    ).astype(np.float32)
    manifest = load_json(args.support_dir / "manifest.json")
    clips = manifest["clips"]
    if len(clips) != len(clip_labels):
        raise ValueError("Support manifest and clip labels differ.")
    if len(features) != len(source_clip_index):
        raise ValueError("Support features and indices differ.")
    if clip_labels.shape[1] != num_classes:
        raise ValueError("BirdSet support label width differs from the base model.")
    groups = recording_groups(clips)
    validation_names, train_clips, validation_clips = choose_validation_groups(
        groups,
        clip_labels,
        args.validation_recordings,
    )
    all_clips = np.arange(len(clip_labels), dtype=np.int64)
    train_data, train_slice_labels, train_index, train_clip_labels = subset(
        features,
        source_clip_index,
        clip_labels,
        train_clips,
    )
    (
        validation_data,
        validation_slice_labels,
        validation_index,
        validation_clip_labels,
    ) = subset(
        features,
        source_clip_index,
        clip_labels,
        validation_clips,
    )
    full_data, full_slice_labels, full_index, full_clip_labels = subset(
        features,
        source_clip_index,
        clip_labels,
        all_clips,
    )
    train_weights = balanced_multilabel_slice_weights(
        train_clip_labels, train_index
    )
    full_weights = balanced_multilabel_slice_weights(full_clip_labels, full_index)

    xeno_data = np.load(args.xeno_dataset_dir / "validation_data.npy").astype(
        np.float32
    )[..., np.newaxis]
    xeno_labels = np.load(args.xeno_dataset_dir / "validation_label.npy").astype(
        np.int64
    )
    xeno_index = np.load(
        args.xeno_dataset_dir / "validation_recording_index.npy"
    ).astype(np.int64)

    selection_model = tf.keras.models.load_model(model_path, compile=False)
    if tuple(selection_model.input_shape[1:]) != tuple(train_data.shape[1:]):
        raise ValueError(
            f"Model input {selection_model.input_shape} does not match "
            f"BirdSet support {train_data.shape}."
        )
    if selection_model.layers[-1].activation.__name__ != "softmax":
        raise ValueError("The Xeno-canto base model must use softmax.")
    baseline_birdset = predict_clips(
        selection_model,
        validation_data,
        validation_clip_labels,
        validation_index,
    )
    baseline_xeno = predict_xeno(
        selection_model,
        xeno_data,
        xeno_labels,
        xeno_index,
        num_classes,
    )

    selection_model.layers[-1].activation = tf.keras.activations.sigmoid
    trainable_layers = configure_trainable(selection_model, args.policy)
    compile_model(selection_model, args.learning_rate)
    trainable_parameters = int(
        sum(np.prod(variable.shape) for variable in selection_model.trainable_weights)
    )
    total_parameters = int(selection_model.count_params())
    adaptation_metrics = AdaptationMetrics(
        validation_data,
        validation_clip_labels,
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
        train_data,
        train_slice_labels,
        sample_weight=train_weights,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(validation_data, validation_slice_labels),
        callbacks=[adaptation_metrics, early_stopping],
        verbose=2,
    )
    scores = np.asarray(selection_history.history["val_adaptation_score"])
    best_epoch = int(scores.argmax()) + 1
    selected_birdset = predict_clips(
        selection_model,
        validation_data,
        validation_clip_labels,
        validation_index,
    )
    selected_xeno = predict_xeno(
        selection_model,
        xeno_data,
        xeno_labels,
        xeno_index,
        num_classes,
    )
    selected_retention = selected_xeno["macro_f1"] / baseline_xeno["macro_f1"]
    selected_score = selected_birdset["top_1_any_target_accuracy"] * min(
        1.0, max(0.0, selected_retention)
    )

    del selection_model
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(args.seed)
    final_model = tf.keras.models.load_model(model_path, compile=False)
    final_model.layers[-1].activation = tf.keras.activations.sigmoid
    configure_trainable(final_model, args.policy)
    compile_model(final_model, args.learning_rate)
    final_history = final_model.fit(
        full_data,
        full_slice_labels,
        sample_weight=full_weights,
        batch_size=args.batch_size,
        epochs=best_epoch,
        verbose=2,
    )
    final_support = predict_clips(
        final_model,
        full_data,
        full_clip_labels,
        full_index,
    )
    final_xeno = predict_xeno(
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
    (args.output_dir / f"{args.model_name}.birdset_fewshot_history.json").write_text(
        json.dumps(
            {
                "selection": selection_history.history,
                "final_all_support": final_history.history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    report = {
        "experiment": "BirdSet SSW grouped five-shot adaptation from Xeno-canto DS-CNN",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "heldout_access_during_training": False,
        "base_model": relative_path(model_path),
        "support_dir": relative_path(args.support_dir),
        "xeno_validation_dir": relative_path(args.xeno_dataset_dir),
        "label_map": base_labels,
        "protocol": {
            "task": "multi-label soundscape adaptation",
            "output_activation": "sigmoid",
            "loss": "binary_crossentropy",
            "support_original_recordings": len(groups),
            "support_clips": len(clip_labels),
            "support_slices": len(features),
            "selection_train_original_recordings": len(groups)
            - len(validation_names),
            "selection_validation_original_recordings": len(validation_names),
            "selection_train_clips": len(train_clips),
            "selection_validation_clips": len(validation_clips),
            "selection_validation_recording_names": validation_names,
            "selection_validation_supported_classes": selected_birdset[
                "supported_classes"
            ],
            "policy": args.policy,
            "learning_rate": args.learning_rate,
            "maximum_selection_epochs": args.epochs,
            "patience": args.patience,
            "selected_epoch": best_epoch,
            "selection_metric": (
                "BirdSet support validation clip top-1 any-target accuracy "
                "multiplied by capped Xeno validation macro-F1 retention"
            ),
            "class_balancing": (
                "Per-clip inverse positive-class-frequency weights, divided "
                "across the five slices of each clip."
            ),
            "final_training": (
                "Reload the original Xeno softmax model, switch the output "
                "activation to sigmoid, and train on all 10 support recordings "
                "for the selected fixed epoch count."
            ),
            "trainable_layers": trainable_layers,
            "trainable_parameters": trainable_parameters,
            "total_parameters": total_parameters,
        },
        "baseline_before_selection": {
            "birdset_support_validation": baseline_birdset,
            "xeno_validation": baseline_xeno,
        },
        "selected_split_model": {
            "birdset_support_validation": selected_birdset,
            "xeno_validation": selected_xeno,
            "xeno_macro_f1_retention": selected_retention,
            "adaptation_score": selected_score,
        },
        "final_all_support_model": {
            "birdset_support_training": final_support,
            "xeno_validation": final_xeno,
            "xeno_macro_f1_retention": (
                final_xeno["macro_f1"] / baseline_xeno["macro_f1"]
            ),
        },
    }
    (args.output_dir / f"{args.model_name}.birdset_fewshot.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
