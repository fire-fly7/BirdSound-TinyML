"""Train a multi-label BirdSet alternative-baseline diagnostic model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, EarlyStopping

from construct_model.BC_ResNet import create_model as create_bc_resnet
from construct_model.CNN_Model import create_model as create_cnn
from construct_model.DS_CNN_Model import create_model as create_ds_cnn
from construct_model.MobileNetV2 import create_model as create_mobilenetv2


MODEL_FACTORIES = {
    "BC_ResNet": create_bc_resnet,
    "CNN_Model": create_cnn,
    "DS_CNN_Model": create_ds_cnn,
    "MobileNetV2": create_mobilenetv2,
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_FACTORIES, default="DS_CNN_Model")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=8)
    return parser.parse_args()


def clip_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    clip_index: np.ndarray,
) -> dict[str, float | int]:
    clip_count = int(clip_index.max()) + 1
    sums = np.zeros((clip_count, labels.shape[1]), dtype=np.float64)
    counts = np.bincount(clip_index, minlength=clip_count)
    np.add.at(sums, clip_index, probabilities)
    clip_probabilities = sums / counts[:, np.newaxis]
    clip_labels = np.zeros((clip_count, labels.shape[1]), dtype=np.float32)
    for row, index in zip(labels, clip_index, strict=True):
        if clip_labels[index].any() and not np.array_equal(clip_labels[index], row):
            raise ValueError("A BirdSet clip has inconsistent slice labels.")
        clip_labels[index] = row
    predictions = clip_probabilities >= 0.5
    true_positive = np.logical_and(predictions, clip_labels).sum(axis=0)
    support = clip_labels.sum(axis=0)
    predicted = predictions.sum(axis=0)
    recall = np.divide(true_positive, support, out=np.zeros_like(support), where=support > 0)
    precision = np.divide(
        true_positive, predicted, out=np.zeros_like(support), where=predicted > 0
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(support),
        where=(precision + recall) > 0,
    )
    supported = support > 0
    top1 = clip_probabilities.argmax(axis=1)
    return {
        "clips": clip_count,
        "supported_classes": int(supported.sum()),
        "subset_accuracy": float(np.mean(np.all(predictions == clip_labels, axis=1))),
        "top_1_any_target_accuracy": float(
            np.mean(clip_labels[np.arange(clip_count), top1] > 0)
        ),
        "macro_recall": float(np.mean(recall[supported])),
        "macro_f1": float(np.mean(f1[supported])),
    }


class ClipValidationMetrics(Callback):
    def __init__(self, data: np.ndarray, labels: np.ndarray, clip_index: np.ndarray) -> None:
        super().__init__()
        self.data = data
        self.labels = labels
        self.clip_index = clip_index

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        logs = logs if logs is not None else {}
        probabilities = self.model.predict(self.data, batch_size=128, verbose=0)
        metrics = clip_metrics(self.labels, probabilities, self.clip_index)
        logs["val_clip_macro_f1"] = metrics["macro_f1"]
        logs["val_clip_subset_accuracy"] = metrics["subset_accuracy"]
        logs["val_clip_top_1_any_target_accuracy"] = metrics[
            "top_1_any_target_accuracy"
        ]
        print(
            f"Epoch {epoch + 1}: val_clip_macro_f1={metrics['macro_f1']:.4f}, "
            f"val_clip_top1_any_target={metrics['top_1_any_target_accuracy']:.4f}",
            flush=True,
        )


def main() -> None:
    args = arguments()
    tf.keras.utils.set_random_seed(42)
    train_data = np.load(args.dataset_dir / "train_data.npy").astype(np.float32)
    train_labels = np.load(args.dataset_dir / "train_label.npy").astype(np.float32)
    train_weights = np.load(args.dataset_dir / "train_sample_weight.npy").astype(np.float32)
    validation_data = np.load(args.dataset_dir / "validation_data.npy").astype(np.float32)
    validation_labels = np.load(args.dataset_dir / "validation_label.npy").astype(np.float32)
    validation_index = np.load(
        args.dataset_dir / "validation_recording_index.npy"
    ).astype(np.int64)
    label_map = json.loads((args.dataset_dir / "label_map.json").read_text(encoding="utf-8"))
    if train_labels.shape[1] != len(label_map) or validation_labels.shape[1] != len(label_map):
        raise ValueError("Multi-label width does not match label_map.json.")

    train_data = train_data[..., np.newaxis]
    validation_data = validation_data[..., np.newaxis]
    model = MODEL_FACTORIES[args.model](train_data.shape[1:], len(label_map))
    model.layers[-1].activation = tf.keras.activations.sigmoid
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="binary_accuracy")],
    )
    callback = ClipValidationMetrics(validation_data, validation_labels, validation_index)
    history = model.fit(
        train_data,
        train_labels,
        sample_weight=train_weights,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(validation_data, validation_labels),
        callbacks=[
            callback,
            EarlyStopping(
                monitor="val_clip_macro_f1",
                mode="max",
                patience=args.patience,
                restore_best_weights=True,
            ),
        ],
        verbose=2,
    )
    probabilities = model.predict(validation_data, batch_size=128, verbose=0)
    metrics = clip_metrics(validation_labels, probabilities, validation_index)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    model.save(args.model_dir / f"{args.model}.h5")
    (args.model_dir / f"{args.model}.labels.json").write_text(
        json.dumps(label_map, indent=2), encoding="utf-8"
    )
    (args.model_dir / f"{args.model}.history.json").write_text(
        json.dumps(history.history, indent=2), encoding="utf-8"
    )
    (args.model_dir / f"{args.model}.validation.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
