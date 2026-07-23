"""Evaluate trained bird-call classifiers on the held-out DB3V regions.

DB3V is never used for fitting or model selection.  The script reports both
one-second slice metrics and recording metrics, where the eight slice softmax
vectors from each DB3V WAV are averaged before classification.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = REPOSITORY_ROOT / "dataset_processing" / "output" / "MFCC_dataset_DB3V"
DEFAULT_MODEL_DIR = REPOSITORY_ROOT / "src" / "model_train&test" / "TinyML_model"
MODEL_NAMES = ("BC_ResNet", "CNN_Model", "DS_CNN_Model", "MobileNetV2")
REGIONS = (1, 2, 3)
SLICES_PER_RECORDING = 8
EXPECTED_FEATURE_SHAPE = (32, 13)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=list(MODEL_NAMES),
        help="Models to evaluate. Defaults to every trained model.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        help="Path for the detailed JSON report. Defaults to <model-dir>/DB3V_evaluation.json.",
    )
    return parser.parse_args()


def load_label_map(path: Path) -> dict[str, int]:
    label_map = json.loads(path.read_text(encoding="utf-8"))
    if sorted(label_map.values()) != list(range(len(label_map))):
        raise ValueError(f"Label map must contain contiguous class IDs: {path}")
    return label_map


def class_names(label_map: dict[str, int]) -> list[str]:
    return [name for name, _ in sorted(label_map.items(), key=lambda item: item[1])]


def load_region(dataset_dir: Path, region: int, num_classes: int) -> tuple[np.ndarray, np.ndarray]:
    features = np.load(dataset_dir / f"region_{region}_data.npy", mmap_mode="r")
    labels = np.load(dataset_dir / f"region_{region}_label.npy", mmap_mode="r")
    if features.ndim != 3 or tuple(features.shape[1:]) != EXPECTED_FEATURE_SHAPE:
        raise ValueError(f"Unexpected DB3V feature shape for region {region}: {features.shape}")
    if len(features) != len(labels):
        raise ValueError(f"Feature/label count mismatch in DB3V region {region}.")
    if not np.isfinite(features).all():
        raise ValueError(f"DB3V region {region} contains non-finite feature values.")
    if labels.min() < 0 or labels.max() >= num_classes:
        raise ValueError(f"DB3V region {region} contains an out-of-range class label.")
    if len(labels) % SLICES_PER_RECORDING != 0:
        raise ValueError(f"DB3V region {region} is not divisible into 8-second recordings.")
    grouped_labels = np.asarray(labels).reshape(-1, SLICES_PER_RECORDING)
    if not np.all(grouped_labels == grouped_labels[:, :1]):
        raise ValueError(f"DB3V region {region} does not preserve contiguous recording slices.")
    return features, np.asarray(labels, dtype=np.int64)


def predict_probabilities(model: tf.keras.Model, features: np.ndarray, batch_size: int) -> np.ndarray:
    inputs = np.expand_dims(features, axis=-1)
    return model.predict(inputs, batch_size=batch_size, verbose=0)


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    names: list[str],
) -> dict[str, Any]:
    num_classes = len(names)
    predictions = probabilities.argmax(axis=1)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)

    true_totals = confusion.sum(axis=1)
    predicted_totals = confusion.sum(axis=0)
    true_positives = np.diag(confusion)
    recall = np.divide(true_positives, true_totals, out=np.zeros(num_classes), where=true_totals != 0)
    precision = np.divide(
        true_positives,
        predicted_totals,
        out=np.zeros(num_classes),
        where=predicted_totals != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(num_classes),
        where=(precision + recall) != 0,
    )
    top_3 = np.argpartition(probabilities, -3, axis=1)[:, -3:]
    top_3_accuracy = float(np.mean(np.any(top_3 == labels[:, np.newaxis], axis=1)))

    per_class = {
        name: {
            "label": index,
            "support": int(true_totals[index]),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
        }
        for index, name in enumerate(names)
    }
    return {
        "samples": int(len(labels)),
        "accuracy": float(np.mean(predictions == labels)),
        "balanced_accuracy": float(np.mean(recall)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(np.average(f1, weights=true_totals)),
        "top_3_accuracy": top_3_accuracy,
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
    }


def recording_probabilities(labels: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    grouped_labels = labels.reshape(-1, SLICES_PER_RECORDING)
    grouped_probabilities = probabilities.reshape(-1, SLICES_PER_RECORDING, probabilities.shape[1])
    return grouped_labels[:, 0], grouped_probabilities.mean(axis=1)


def validate_model(model: tf.keras.Model, model_name: str, num_classes: int) -> None:
    if tuple(model.input_shape[1:]) != (*EXPECTED_FEATURE_SHAPE, 1):
        raise ValueError(f"{model_name} has an unexpected input shape: {model.input_shape}")
    if model.output_shape[-1] != num_classes:
        raise ValueError(f"{model_name} has an unexpected output shape: {model.output_shape}")


def evaluate_model(
    model_name: str,
    expected_label_map: dict[str, int],
    batch_size: int,
    dataset_dir: Path,
    model_dir: Path,
) -> dict[str, Any]:
    model_path = model_dir / f"{model_name}.h5"
    model_label_map = load_label_map(model_dir / f"{model_name}.labels.json")
    if model_label_map != expected_label_map:
        raise ValueError(f"{model_name} label map does not match DB3V.")

    model = tf.keras.models.load_model(model_path, compile=False)
    validate_model(model, model_name, len(expected_label_map))
    names = class_names(expected_label_map)
    region_results: dict[str, Any] = {}
    all_labels: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    started_at = time.perf_counter()

    print(f"Evaluating {model_name}...", flush=True)
    for region in REGIONS:
        features, labels = load_region(dataset_dir, region, len(names))
        probabilities = predict_probabilities(model, features, batch_size)
        slice_result = classification_metrics(labels, probabilities, names)
        clip_labels, clip_probabilities = recording_probabilities(labels, probabilities)
        recording_result = classification_metrics(clip_labels, clip_probabilities, names)
        region_results[str(region)] = {
            "recordings": int(len(clip_labels)),
            "slice_level": slice_result,
            "recording_level": recording_result,
        }
        all_labels.append(labels)
        all_probabilities.append(probabilities)
        print(
            f"  Region {region}: recording accuracy={recording_result['accuracy']:.4f}, "
            f"macro-F1={recording_result['macro_f1']:.4f}",
            flush=True,
        )

    pooled_labels = np.concatenate(all_labels)
    pooled_probabilities = np.concatenate(all_probabilities)
    pooled_slice = classification_metrics(pooled_labels, pooled_probabilities, names)
    pooled_recording_labels, pooled_recording_probabilities = recording_probabilities(
        pooled_labels, pooled_probabilities
    )
    pooled_recording = classification_metrics(
        pooled_recording_labels, pooled_recording_probabilities, names
    )
    elapsed_seconds = time.perf_counter() - started_at
    print(
        f"  Pooled: recording accuracy={pooled_recording['accuracy']:.4f}, "
        f"macro-F1={pooled_recording['macro_f1']:.4f}, elapsed={elapsed_seconds:.1f}s",
        flush=True,
    )

    tf.keras.backend.clear_session()
    return {
        "model_path": str(model_path.relative_to(REPOSITORY_ROOT)),
        "input_shape": list(model.input_shape),
        "output_shape": list(model.output_shape),
        "inference_seconds": elapsed_seconds,
        "regions": region_results,
        "pooled": {
            "recordings": int(len(pooled_recording_labels)),
            "slice_level": pooled_slice,
            "recording_level": pooled_recording,
        },
    }


def write_summary_csv(report: dict[str, Any], output_path: Path) -> Path:
    summary_path = output_path.with_name(f"{output_path.stem}_summary.csv")
    columns = [
        "model",
        "scope",
        "level",
        "samples",
        "recordings",
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "top_3_accuracy",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for model_name, model_result in report["models"].items():
            scopes = {"pooled": model_result["pooled"], **model_result["regions"]}
            for scope, scope_result in scopes.items():
                for level in ("slice_level", "recording_level"):
                    metrics = scope_result[level]
                    writer.writerow(
                        {
                            "model": model_name,
                            "scope": scope,
                            "level": level,
                            "samples": metrics["samples"],
                            "recordings": scope_result["recordings"],
                            "accuracy": metrics["accuracy"],
                            "balanced_accuracy": metrics["balanced_accuracy"],
                            "macro_precision": metrics["macro_precision"],
                            "macro_recall": metrics["macro_recall"],
                            "macro_f1": metrics["macro_f1"],
                            "weighted_f1": metrics["weighted_f1"],
                            "top_3_accuracy": metrics["top_3_accuracy"],
                        }
                    )
    return summary_path


def main() -> None:
    arguments = parse_arguments()
    arguments.dataset_dir = arguments.dataset_dir.resolve()
    arguments.model_dir = arguments.model_dir.resolve()
    if arguments.output is not None:
        arguments.output = arguments.output.resolve()
    tf.get_logger().setLevel("ERROR")
    if arguments.batch_size <= 0:
        raise ValueError("Batch size must be positive.")
    label_map = load_label_map(arguments.dataset_dir / "label_map.json")
    report = {
        "dataset": {
            "name": "DB3V",
            "purpose": "Independent regional evaluation only; no DB3V samples were used for model fitting.",
            "feature_protocol": (
                "16 kHz mono audio, non-overlapping one-second slices, 13 MFCCs, "
                "fixed to (32, 13); matches data_sugment_MFCC.py."
            ),
            "slices_per_recording": SLICES_PER_RECORDING,
            "label_map": label_map,
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "models": {},
    }
    for model_name in arguments.models:
        report["models"][model_name] = evaluate_model(
            model_name,
            label_map,
            arguments.batch_size,
            arguments.dataset_dir,
            arguments.model_dir,
        )

    output_path = arguments.output or arguments.model_dir / "DB3V_evaluation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary_path = write_summary_csv(report, output_path)
    print(f"Saved detailed report: {output_path}", flush=True)
    print(f"Saved summary table: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
