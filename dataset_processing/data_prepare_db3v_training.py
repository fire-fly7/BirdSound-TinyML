"""Build a diagnostic DB3V train/validation split at recording granularity.

This script is only for testing DB3V's suitability as an alternative baseline
dataset. It stratifies by region and class, but both partitions still contain
all three regions and therefore do not measure unseen-region generalization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SLICES_PER_RECORDING = 8


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if not 0 < args.validation_ratio < 1:
        raise ValueError("--validation-ratio must be between zero and one.")
    label_map = json.loads((args.source_dir / "label_map.json").read_text(encoding="utf-8"))
    random = np.random.default_rng(args.seed)
    train_features: list[np.ndarray] = []
    train_labels: list[np.ndarray] = []
    validation_features: list[np.ndarray] = []
    validation_labels: list[np.ndarray] = []
    split_counts: dict[str, dict[str, int]] = {}

    for region in (1, 2, 3):
        features = np.load(args.source_dir / f"region_{region}_data.npy", mmap_mode="r")
        labels = np.load(args.source_dir / f"region_{region}_label.npy")
        if len(labels) % SLICES_PER_RECORDING:
            raise ValueError(f"Region {region} is not divisible into eight-slice recordings.")
        recording_features = features.reshape(
            -1, SLICES_PER_RECORDING, *features.shape[1:]
        )
        recording_labels = labels.reshape(-1, SLICES_PER_RECORDING)[:, 0]
        region_train: list[int] = []
        region_validation: list[int] = []
        for label in range(len(label_map)):
            indices = np.flatnonzero(recording_labels == label)
            random.shuffle(indices)
            validation_count = min(
                len(indices) - 1,
                max(1, round(len(indices) * args.validation_ratio)),
            )
            region_validation.extend(indices[:validation_count].tolist())
            region_train.extend(indices[validation_count:].tolist())
        region_train = sorted(region_train)
        region_validation = sorted(region_validation)
        train_features.append(recording_features[region_train].reshape(-1, *features.shape[1:]))
        train_labels.append(
            np.repeat(recording_labels[region_train], SLICES_PER_RECORDING)
        )
        validation_features.append(
            recording_features[region_validation].reshape(-1, *features.shape[1:])
        )
        validation_labels.append(
            np.repeat(recording_labels[region_validation], SLICES_PER_RECORDING)
        )
        split_counts[str(region)] = {
            "train_recordings": len(region_train),
            "validation_recordings": len(region_validation),
        }

    train_data = np.concatenate(train_features).astype(np.float32)
    train_label = np.concatenate(train_labels).astype(np.int64)
    validation_data = np.concatenate(validation_features).astype(np.float32)
    validation_label = np.concatenate(validation_labels).astype(np.int64)
    train_recording_index = np.repeat(
        np.arange(len(train_data) // SLICES_PER_RECORDING), SLICES_PER_RECORDING
    )
    validation_recording_index = np.repeat(
        np.arange(len(validation_data) // SLICES_PER_RECORDING), SLICES_PER_RECORDING
    )
    recording_labels = train_label[::SLICES_PER_RECORDING]
    class_counts = np.bincount(recording_labels, minlength=len(label_map))
    class_weights = len(recording_labels) / (len(label_map) * class_counts)
    train_weights = (
        class_weights[train_label] / SLICES_PER_RECORDING
    ).astype(np.float32)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "train_data.npy": train_data,
        "train_label.npy": train_label,
        "train_recording_index.npy": train_recording_index,
        "train_sample_weight.npy": train_weights,
        "validation_data.npy": validation_data,
        "validation_label.npy": validation_label,
        "validation_recording_index.npy": validation_recording_index,
    }
    for name, array in arrays.items():
        np.save(args.output_dir / name, array)
    (args.output_dir / "label_map.json").write_text(
        json.dumps(label_map, indent=2), encoding="utf-8"
    )
    report = {
        "dataset": "DB3V diagnostic baseline split",
        "purpose": "Alternative-baseline suitability test, not regional generalization",
        "split_unit": "eight-second recording",
        "stratification": "region and class",
        "validation_ratio": args.validation_ratio,
        "seed": args.seed,
        "feature_shape": list(train_data.shape[1:]),
        "train_recordings": len(train_data) // SLICES_PER_RECORDING,
        "validation_recordings": len(validation_data) // SLICES_PER_RECORDING,
        "regions": split_counts,
        "limitation": (
            "Every region appears in both partitions; validation measures same-region "
            "holdout performance and must not be reported as unseen-region generalization."
        ),
    }
    (args.output_dir / "split_manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
