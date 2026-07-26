"""Evaluate an eight-class Keras model on multi-label BirdSet SSW soundscapes."""

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


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = (
    REPOSITORY_ROOT
    / "src"
    / "dataset_processing"
    / "output"
    / "MFCC_dataset_BirdSet_SSW_8class"
)
DEFAULT_MODEL_DIR = REPOSITORY_ROOT / "src" / "experiments" / "TinyML_model_8class"
MODEL_NAMES = ("BC_ResNet", "CNN_Model", "DS_CNN_Model", "MobileNetV2")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=["DS_CNN_Model"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def names_for(label_map: dict[str, int]) -> list[str]:
    return [name for name, _ in sorted(label_map.items(), key=lambda item: item[1])]


def multilabel_metrics(labels: np.ndarray, probabilities: np.ndarray, names: list[str]) -> dict:
    predictions = probabilities.argmax(axis=1)
    top3 = np.argpartition(probabilities, -3, axis=1)[:, -3:]
    rows = np.arange(len(labels))
    top1_hit = labels[rows, predictions].astype(bool)
    top3_hit = np.any(np.take_along_axis(labels, top3, axis=1), axis=1)
    per_class: dict[str, Any] = {}
    for class_id, name in enumerate(names):
        positives = labels[:, class_id].astype(bool)
        per_class[name] = {
            "label": class_id,
            "support": int(positives.sum()),
            "top1_recall": (
                float(np.mean(predictions[positives] == class_id))
                if positives.any()
                else None
            ),
        }
    return {
        "samples": int(len(labels)),
        "top1_any_target_accuracy": float(np.mean(top1_hit)),
        "top3_any_target_accuracy": float(np.mean(top3_hit)),
        "per_class": per_class,
    }


def singleton_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    names: list[str],
    mask: np.ndarray,
) -> dict:
    singleton_labels = labels[mask].argmax(axis=1)
    predictions = probabilities[mask].argmax(axis=1)
    confusion = np.zeros((len(names), len(names)), dtype=np.int64)
    np.add.at(confusion, (singleton_labels, predictions), 1)
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    true_positive = np.diag(confusion)
    recall = np.divide(true_positive, support, out=np.zeros(len(names)), where=support != 0)
    precision = np.divide(
        true_positive, predicted, out=np.zeros(len(names)), where=predicted != 0
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(len(names)),
        where=(precision + recall) != 0,
    )
    supported = support > 0
    return {
        "samples": int(mask.sum()),
        "accuracy": float(np.mean(predictions == singleton_labels)) if mask.any() else None,
        "supported_macro_f1": float(np.mean(f1[supported])) if supported.any() else None,
        "supported_classes": [names[index] for index in np.flatnonzero(supported)],
        "confusion_matrix": confusion.tolist(),
    }


def main() -> None:
    arguments = parse_arguments()
    dataset_dir = arguments.dataset_dir.resolve()
    model_dir = arguments.model_dir.resolve()
    label_map = load_json(dataset_dir / "label_map.json")
    names = names_for(label_map)
    features = np.load(dataset_dir / "test_data.npy", mmap_mode="r")
    clip_index = np.load(dataset_dir / "test_clip_index.npy")
    clip_labels = np.load(dataset_dir / "test_clip_multilabel.npy")
    manifest = load_json(dataset_dir / "manifest.json")
    global_singleton_mask = np.asarray(
        [item["is_globally_singleton"] for item in manifest["clips"]], dtype=bool
    )
    if features.ndim != 3 or features.shape[1] != 32:
        raise ValueError(f"Unexpected feature shape: {features.shape}")
    if len(features) != len(clip_index):
        raise ValueError("Feature and clip-index counts differ.")
    if clip_labels.shape[1] != len(label_map):
        raise ValueError("Multi-label width does not match the label map.")
    if clip_index.min() != 0 or clip_index.max() + 1 != len(clip_labels):
        raise ValueError("Clip indices are not contiguous.")
    if len(global_singleton_mask) != len(clip_labels):
        raise ValueError("Manifest and multi-label clip counts differ.")

    slice_labels = clip_labels[clip_index]
    report = {
        "dataset": "BirdSet SSW test_5s",
        "purpose": "Independent multi-label soundscape evaluation only.",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "label_map": label_map,
        "models": {},
    }
    for model_name in arguments.models:
        model_labels = load_json(model_dir / f"{model_name}.labels.json")
        if model_labels != label_map:
            raise ValueError(f"{model_name} label map does not match BirdSet data.")
        model = tf.keras.models.load_model(model_dir / f"{model_name}.h5", compile=False)
        if tuple(model.input_shape[1:]) != (*features.shape[1:], 1):
            raise ValueError(f"Unexpected {model_name} input shape: {model.input_shape}")
        probabilities = model.predict(
            np.expand_dims(features, axis=-1),
            batch_size=arguments.batch_size,
            verbose=0,
        )
        clip_probabilities = np.zeros((len(clip_labels), len(names)), dtype=np.float32)
        clip_counts = np.bincount(clip_index, minlength=len(clip_labels))
        np.add.at(clip_probabilities, clip_index, probabilities)
        clip_probabilities /= clip_counts[:, np.newaxis]
        report["models"][model_name] = {
            "slice_level": multilabel_metrics(slice_labels, probabilities, names),
            "clip_level": multilabel_metrics(clip_labels, clip_probabilities, names),
            "globally_singleton_clip_level": singleton_metrics(
                clip_labels,
                clip_probabilities,
                names,
                global_singleton_mask,
            ),
        }
        print(
            f"{model_name}: clip top-1 any-target="
            f"{report['models'][model_name]['clip_level']['top1_any_target_accuracy']:.4f}"
        )
        tf.keras.backend.clear_session()

    output = arguments.output or model_dir / "BirdSet_SSW_evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved report: {output}")


if __name__ == "__main__":
    main()
