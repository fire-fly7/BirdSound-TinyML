"""Train an N-class bird-call model from processed Xeno-canto MFCC features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.utils import to_categorical

from construct_model.BC_ResNet import create_model as create_bc_resnet
from construct_model.CNN_Model import create_model as create_cnn
from construct_model.DS_CNN_Model import create_model as create_ds_cnn
from construct_model.MobileNetV2 import create_model as create_mobilenetv2


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = (
    REPOSITORY_ROOT / "dataset_processing" / "output" / "MFCC_dataset_A_8class"
)
DEFAULT_MODEL_DIR = REPOSITORY_ROOT / "src" / "model_train&test" / "TinyML_model_8class"
MODEL_FACTORIES = {
    "BC_ResNet": create_bc_resnet,
    "CNN_Model": create_cnn,
    "DS_CNN_Model": create_ds_cnn,
    "MobileNetV2": create_mobilenetv2,
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_FACTORIES, default="DS_CNN_Model")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=8)
    return parser.parse_args()


def load_data(
    dataset_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    train_data = np.load(dataset_dir / "train_data.npy").astype(np.float32)
    train_labels = np.load(dataset_dir / "train_label.npy").astype(np.int64)
    train_sample_weights = np.load(dataset_dir / "train_sample_weight.npy").astype(np.float32)
    validation_data = np.load(dataset_dir / "validation_data.npy").astype(np.float32)
    validation_labels = np.load(dataset_dir / "validation_label.npy").astype(np.int64)
    validation_recording_indices = np.load(
        dataset_dir / "validation_recording_index.npy"
    ).astype(np.int64)
    label_map = json.loads((dataset_dir / "label_map.json").read_text(encoding="utf-8"))
    return (
        train_data,
        train_labels,
        train_sample_weights,
        validation_data,
        validation_labels,
        validation_recording_indices,
        label_map,
    )


def validate_data(
    train_data: np.ndarray,
    train_labels: np.ndarray,
    train_weights: np.ndarray,
    validation_data: np.ndarray,
    validation_labels: np.ndarray,
    validation_recording_indices: np.ndarray,
    label_map: dict[str, int],
) -> None:
    expected_labels = np.arange(len(label_map))
    if sorted(label_map.values()) != expected_labels.tolist():
        raise ValueError("The label map must contain contiguous class IDs.")
    if train_data.ndim != 3 or train_data.shape[1] != 32:
        raise ValueError(f"Unexpected training feature shape: {train_data.shape}")
    if validation_data.ndim != 3 or validation_data.shape[1] != 32:
        raise ValueError(f"Unexpected validation feature shape: {validation_data.shape}")
    if train_data.shape[1:] != validation_data.shape[1:]:
        raise ValueError("Training and validation feature shapes do not match.")
    if not np.array_equal(np.unique(train_labels), expected_labels):
        raise ValueError("Training labels do not match the label map.")
    if not np.array_equal(np.unique(validation_labels), expected_labels):
        raise ValueError("Validation labels do not match the label map.")
    if len(train_data) != len(train_labels) or len(train_labels) != len(train_weights):
        raise ValueError("Training data, labels, and sample weights must have equal lengths.")
    if len(validation_data) != len(validation_labels) or len(validation_labels) != len(
        validation_recording_indices
    ):
        raise ValueError("Validation data, labels, and recording indices must have equal lengths.")
    if not np.isfinite(train_data).all() or not np.isfinite(validation_data).all():
        raise ValueError("MFCC features contain non-finite values.")
    if np.any(train_weights <= 0) or not np.isfinite(train_weights).all():
        raise ValueError("Training sample weights must be finite and positive.")


def recording_level_metrics(
    labels: np.ndarray, probabilities: np.ndarray, recording_indices: np.ndarray, num_classes: int
) -> dict[str, float]:
    """Average slice probabilities per recording, then calculate balanced metrics."""
    recording_count = int(recording_indices.max()) + 1
    probability_sums = np.zeros((recording_count, num_classes), dtype=np.float64)
    slice_counts = np.zeros(recording_count, dtype=np.int64)
    recording_labels = np.full(recording_count, -1, dtype=np.int64)
    np.add.at(probability_sums, recording_indices, probabilities)
    np.add.at(slice_counts, recording_indices, 1)
    for index, label in zip(recording_indices, labels, strict=True):
        if recording_labels[index] not in {-1, label}:
            raise ValueError("A validation recording has inconsistent labels.")
        recording_labels[index] = label
    if np.any(slice_counts == 0) or np.any(recording_labels < 0):
        raise ValueError("A validation recording has no slices or no label.")

    predictions = (probability_sums / slice_counts[:, np.newaxis]).argmax(axis=1)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (recording_labels, predictions), 1)
    support = confusion.sum(axis=1)
    true_positive = np.diag(confusion)
    recall = np.divide(true_positive, support, out=np.zeros(num_classes), where=support != 0)
    predicted = confusion.sum(axis=0)
    precision = np.divide(true_positive, predicted, out=np.zeros(num_classes), where=predicted != 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(num_classes),
        where=(precision + recall) != 0,
    )
    return {
        "recordings": float(recording_count),
        "accuracy": float(np.mean(predictions == recording_labels)),
        "balanced_accuracy": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
    }


class RecordingValidationMetrics(Callback):
    """Expose recording-level validation metrics to Keras early stopping."""

    def __init__(
        self,
        validation_data: np.ndarray,
        validation_labels: np.ndarray,
        validation_recording_indices: np.ndarray,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.validation_data = validation_data
        self.validation_labels = validation_labels
        self.validation_recording_indices = validation_recording_indices
        self.num_classes = num_classes

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        logs = logs if logs is not None else {}
        probabilities = self.model.predict(self.validation_data, batch_size=128, verbose=0)
        metrics = recording_level_metrics(
            self.validation_labels,
            probabilities,
            self.validation_recording_indices,
            self.num_classes,
        )
        logs["val_recording_accuracy"] = metrics["accuracy"]
        logs["val_recording_balanced_accuracy"] = metrics["balanced_accuracy"]
        logs["val_recording_macro_f1"] = metrics["macro_f1"]
        print(
            f"Epoch {epoch + 1}: val_recording_accuracy={metrics['accuracy']:.4f}, "
            f"val_recording_balanced_accuracy={metrics['balanced_accuracy']:.4f}, "
            f"val_recording_macro_f1={metrics['macro_f1']:.4f}",
            flush=True,
        )


def main() -> None:
    arguments = parse_arguments()
    if arguments.epochs < 1 or arguments.batch_size < 1 or arguments.patience < 1:
        raise ValueError("--epochs, --batch-size, and --patience must be positive.")
    tf.keras.utils.set_random_seed(42)
    (
        train_data,
        train_labels,
        train_weights,
        validation_data,
        validation_labels,
        validation_recording_indices,
        label_map,
    ) = load_data(arguments.dataset_dir)
    validate_data(
        train_data,
        train_labels,
        train_weights,
        validation_data,
        validation_labels,
        validation_recording_indices,
        label_map,
    )

    train_data = train_data[..., np.newaxis]
    validation_data = validation_data[..., np.newaxis]
    num_classes = len(label_map)
    train_targets = to_categorical(train_labels, num_classes)
    validation_targets = to_categorical(validation_labels, num_classes)

    model = MODEL_FACTORIES[arguments.model](train_data.shape[1:], num_classes)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    recording_metrics = RecordingValidationMetrics(
        validation_data,
        validation_labels,
        validation_recording_indices,
        num_classes,
    )
    history = model.fit(
        train_data,
        train_targets,
        sample_weight=train_weights,
        batch_size=arguments.batch_size,
        epochs=arguments.epochs,
        validation_data=(validation_data, validation_targets),
        callbacks=[
            recording_metrics,
            EarlyStopping(
                monitor="val_recording_macro_f1",
                mode="max",
                patience=arguments.patience,
                restore_best_weights=True,
            ),
        ],
        verbose=2,
    )

    probabilities = model.predict(validation_data, batch_size=128, verbose=0)
    final_recording_metrics = recording_level_metrics(
        validation_labels,
        probabilities,
        validation_recording_indices,
        num_classes,
    )
    arguments.model_dir.mkdir(parents=True, exist_ok=True)
    model_path = arguments.model_dir / f"{arguments.model}.h5"
    model.save(model_path)
    (arguments.model_dir / f"{arguments.model}.labels.json").write_text(
        json.dumps(label_map, indent=2), encoding="utf-8"
    )
    (arguments.model_dir / f"{arguments.model}.history.json").write_text(
        json.dumps(history.history, indent=2), encoding="utf-8"
    )
    (arguments.model_dir / f"{arguments.model}.validation.json").write_text(
        json.dumps(final_recording_metrics, indent=2), encoding="utf-8"
    )
    print(f"Saved {arguments.model}: {model_path}")
    print(f"Input shape: {model.input_shape}; output shape: {model.output_shape}")
    print(
        "Best protocol metric (recording macro-F1): "
        f"{max(history.history['val_recording_macro_f1']):.4f}; "
        f"restored model metric: {final_recording_metrics['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
