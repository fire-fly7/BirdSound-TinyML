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
POLICIES = (
    "head",
    "last_block",
    "all",
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
    parser.add_argument("--validation-recordings", type=int, default=3)
    parser.add_argument(
        "--validation-variant",
        type=int,
        help=(
            "Optional zero-based choice among the three highest-quality grouped "
            "validation candidates. Used by the multi-seed runner."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--replay-ratio",
        type=float,
        default=1.0,
        help=(
            "Number of class-balanced Xeno-canto training slices per BirdSet "
            "support slice. Used only by bn_head_replay."
        ),
    )
    parser.add_argument(
        "--print-report",
        action="store_true",
        help="Print the complete JSON report instead of a concise completion summary.",
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
    seed: int,
    variant: int | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    names = sorted(groups)
    if count < 1 or count >= len(names):
        raise ValueError("--validation-recordings must leave at least one train group.")
    target_clips = round(len(clip_labels) * 0.2)
    candidates: list[tuple[tuple[str, ...], int, int]] = []
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
        candidates.append(
            (
                candidate,
                supported_validation_classes,
                abs(len(validation_clips) - target_clips),
            )
        )
    if not candidates:
        raise ValueError("No grouped validation split preserves all train classes.")
    best_supported = max(item[1] for item in candidates)
    coverage_candidates = [
        item for item in candidates if item[1] == best_supported
    ]
    eligible = sorted(
        coverage_candidates,
        key=lambda item: (item[2], item[0]),
    )[:3]
    if variant is None:
        random = np.random.default_rng(seed)
        selected_index = int(random.integers(len(eligible)))
    else:
        if variant < 0:
            raise ValueError("--validation-variant must be nonnegative.")
        selected_index = variant % len(eligible)
    best_names = eligible[selected_index][0]
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


def balanced_replay_rows(
    labels: np.ndarray,
    count: int,
    num_classes: int,
    seed: int,
) -> np.ndarray:
    """Select a deterministic approximately class-balanced Xeno replay buffer."""
    if count < 1:
        return np.empty(0, dtype=np.int64)
    random = np.random.default_rng(seed)
    per_class = count // num_classes
    remainder = count % num_classes
    selected: list[np.ndarray] = []
    for class_id in range(num_classes):
        candidates = np.flatnonzero(labels == class_id)
        requested = per_class + (1 if class_id < remainder else 0)
        if requested > len(candidates):
            raise ValueError(
                f"Xeno class {class_id} has {len(candidates)} slices, fewer than "
                f"the requested replay count {requested}."
            )
        selected.append(random.choice(candidates, size=requested, replace=False))
    rows = np.concatenate(selected)
    random.shuffle(rows)
    return rows.astype(np.int64)


def replay_weights(
    replay_labels: np.ndarray,
    target_mean: float,
    num_classes: int,
) -> np.ndarray:
    counts = np.bincount(replay_labels, minlength=num_classes).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("Replay sampling must include every class.")
    weights = 1.0 / counts[replay_labels]
    weights *= target_mean / weights.mean()
    return weights.astype(np.float32)


def mix_with_replay(
    support_data: np.ndarray,
    support_labels: np.ndarray,
    support_weights: np.ndarray,
    xeno_train_data: np.ndarray,
    xeno_train_labels: np.ndarray,
    replay_ratio: float,
    num_classes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    replay_count = int(round(len(support_labels) * replay_ratio))
    rows = balanced_replay_rows(
        xeno_train_labels,
        replay_count,
        num_classes,
        seed,
    )
    replay_data = np.asarray(xeno_train_data[rows], dtype=np.float32)[
        ..., np.newaxis
    ]
    replay_targets = np.eye(num_classes, dtype=np.float32)[xeno_train_labels[rows]]
    weights = replay_weights(
        xeno_train_labels[rows],
        float(support_weights.mean()),
        num_classes,
    )
    mixed_data = np.concatenate((support_data, replay_data))
    mixed_labels = np.concatenate((support_labels, replay_targets))
    mixed_weights = np.concatenate((support_weights, weights))
    random = np.random.default_rng(seed + 1)
    order = random.permutation(len(mixed_labels))
    return (
        mixed_data[order],
        mixed_labels[order],
        mixed_weights[order],
        rows,
    )


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
        or args.validation_recordings < 1
        or args.replay_ratio < 0
    ):
        raise ValueError(
            "Learning rate/count arguments must be positive and replay ratio nonnegative."
        )
    if args.policy in REPLAY_POLICIES and args.replay_ratio <= 0:
        raise ValueError("Replay policies require --replay-ratio greater than zero.")
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
    split_manifest = load_json(args.support_dir.parent / "split_manifest.json")
    requested_shots = int(split_manifest["requested_shots_per_class"])
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
        args.seed,
        args.validation_variant,
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
    selection_fit_data = train_data
    selection_fit_labels = train_slice_labels
    selection_fit_weights = train_weights
    final_fit_data = full_data
    final_fit_labels = full_slice_labels
    final_fit_weights = full_weights
    selection_replay_rows = np.empty(0, dtype=np.int64)
    final_replay_rows = np.empty(0, dtype=np.int64)
    if args.policy in REPLAY_POLICIES:
        xeno_train_data = np.load(
            args.xeno_dataset_dir / "train_data.npy",
            mmap_mode="r",
        )
        xeno_train_labels = np.load(
            args.xeno_dataset_dir / "train_label.npy",
            mmap_mode="r",
        ).astype(np.int64)
        if tuple(xeno_train_data.shape[1:]) != tuple(features.shape[1:]):
            raise ValueError("Xeno replay and BirdSet support feature shapes differ.")
        (
            selection_fit_data,
            selection_fit_labels,
            selection_fit_weights,
            selection_replay_rows,
        ) = mix_with_replay(
            train_data,
            train_slice_labels,
            train_weights,
            xeno_train_data,
            xeno_train_labels,
            args.replay_ratio,
            num_classes,
            args.seed + 10_000,
        )
        (
            final_fit_data,
            final_fit_labels,
            final_fit_weights,
            final_replay_rows,
        ) = mix_with_replay(
            full_data,
            full_slice_labels,
            full_weights,
            xeno_train_data,
            xeno_train_labels,
            args.replay_ratio,
            num_classes,
            args.seed + 20_000,
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
        selection_fit_data,
        selection_fit_labels,
        sample_weight=selection_fit_weights,
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
        final_fit_data,
        final_fit_labels,
        sample_weight=final_fit_weights,
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
        "experiment": (
            f"BirdSet SSW grouped {requested_shots}-shot adaptation from "
            "Xeno-canto DS-CNN"
        ),
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
            "requested_shots_per_class": requested_shots,
            "support_positive_clips_by_class": split_manifest[
                "support_positive_clips_by_class"
            ],
            "support_shortfall_positive_clips_by_class": split_manifest.get(
                "support_shortfall_positive_clips_by_class",
                np.maximum(
                    requested_shots
                    - np.asarray(
                        split_manifest["support_positive_clips_by_class"],
                        dtype=np.int64,
                    ),
                    0,
                ).tolist(),
            ),
            "selection_train_original_recordings": len(groups)
            - len(validation_names),
            "selection_validation_original_recordings": len(validation_names),
            "selection_train_clips": len(train_clips),
            "selection_validation_clips": len(validation_clips),
            "selection_validation_recording_names": validation_names,
            "selection_validation_variant": args.validation_variant,
            "seed": args.seed,
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
                "ratio_to_birdset_support_slices": (
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
    if args.print_report:
        print(json.dumps(report, indent=2))
    else:
        print(
            json.dumps(
                {
                    "report": relative_path(
                        args.output_dir
                        / f"{args.model_name}.birdset_fewshot.json"
                    ),
                    "requested_shots": requested_shots,
                    "policy": args.policy,
                    "seed": args.seed,
                    "selected_epoch": best_epoch,
                    "selection_adaptation_score": selected_score,
                    "final_xeno_macro_f1": final_xeno["macro_f1"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
