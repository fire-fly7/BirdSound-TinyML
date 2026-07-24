"""Build the fixed BirdSet alternative-baseline diagnostic split.

BirdSet SSW is multi-label, so this output is compatible with Train_birdset.py,
not the single-label Xeno-canto/DB3V trainer. The split is fixed by original
long recording to reproduce the archived comparison results.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT = Path(__file__).with_name("birdset_baseline_validation_recordings.json")
RECORDING_PATTERN = re.compile(r"^(.*)_\d+_\d+\.ogg$")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    return parser.parse_args()


def recording_groups(clips: list[dict]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, clip in enumerate(clips):
        match = RECORDING_PATTERN.match(Path(clip["filepath"]).name)
        if match is None:
            raise ValueError(f"Cannot group BirdSet clip {clip['filepath']!r}.")
        groups[match.group(1)].append(index)
    return groups


def subset(
    features: np.ndarray,
    clip_index: np.ndarray,
    clip_labels: np.ndarray,
    selected_clips: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = np.asarray(sorted(selected_clips), dtype=np.int64)
    rows = np.flatnonzero(np.isin(clip_index, selected))
    remap = np.full(len(clip_labels), -1, dtype=np.int64)
    remap[selected] = np.arange(len(selected))
    local_index = remap[clip_index[rows]]
    return (
        np.asarray(features[rows], dtype=np.float32),
        np.asarray(clip_labels[clip_index[rows]], dtype=np.float32),
        local_index,
    )


def balanced_multilabel_slice_weights(
    clip_labels: np.ndarray, slice_clip_index: np.ndarray
) -> np.ndarray:
    class_counts = clip_labels.sum(axis=0)
    if np.any(class_counts == 0) or np.any(clip_labels.sum(axis=1) == 0):
        raise ValueError("Training clips must cover every class and have a positive label.")
    inverse = 1.0 / class_counts
    clip_weights = (clip_labels * inverse).sum(axis=1) / clip_labels.sum(axis=1)
    clip_weights /= clip_weights.mean()
    slices_per_clip = np.bincount(slice_clip_index, minlength=len(clip_labels))
    return (clip_weights[slice_clip_index] / slices_per_clip[slice_clip_index]).astype(
        np.float32
    )


def class_counts(labels: np.ndarray, label_map: dict[str, int]) -> dict[str, int]:
    counts = labels.sum(axis=0).astype(int)
    return {
        name: int(counts[class_id])
        for name, class_id in sorted(label_map.items(), key=lambda item: item[1])
    }


def main() -> None:
    args = arguments()
    manifest = json.loads((args.source_dir / "manifest.json").read_text(encoding="utf-8"))
    split = json.loads(args.split_file.read_text(encoding="utf-8"))
    label_map = json.loads((args.source_dir / "label_map.json").read_text(encoding="utf-8"))
    features = np.load(args.source_dir / "test_data.npy", mmap_mode="r")
    source_clip_index = np.load(args.source_dir / "test_clip_index.npy")
    clip_labels = np.load(args.source_dir / "test_clip_multilabel.npy").astype(np.float32)
    groups = recording_groups(manifest["clips"])
    validation_names = set(split["validation_recordings"])
    unknown = validation_names.difference(groups)
    if unknown:
        raise ValueError(f"Unknown validation recordings: {sorted(unknown)}")

    validation_clips = [
        clip for name in sorted(validation_names) for clip in groups[name]
    ]
    train_clips = [
        clip for name in sorted(groups) if name not in validation_names for clip in groups[name]
    ]
    train_data, train_labels, train_index = subset(
        features, source_clip_index, clip_labels, train_clips
    )
    validation_data, validation_labels, validation_index = subset(
        features, source_clip_index, clip_labels, validation_clips
    )
    train_clip_labels = clip_labels[np.asarray(sorted(train_clips))]
    train_weights = balanced_multilabel_slice_weights(train_clip_labels, train_index)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "train_data.npy": train_data,
        "train_label.npy": train_labels,
        "train_recording_index.npy": train_index,
        "train_sample_weight.npy": train_weights,
        "validation_data.npy": validation_data,
        "validation_label.npy": validation_labels,
        "validation_recording_index.npy": validation_index,
    }
    for name, array in arrays.items():
        np.save(args.output_dir / name, array)
    (args.output_dir / "label_map.json").write_text(
        json.dumps(label_map, indent=2), encoding="utf-8"
    )
    report = {
        "dataset": "BirdSet SSW alternative-baseline diagnostic",
        "task": "multi-label soundscape classification",
        "split_unit": "original long recording",
        "feature_shape": list(train_data.shape[1:]),
        "train": {
            "recordings": len(groups) - len(validation_names),
            "clips": len(train_clips),
            "slices": len(train_data),
            "positive_clips_by_class": class_counts(train_clip_labels, label_map),
        },
        "validation": {
            "recordings": len(validation_names),
            "clips": len(validation_clips),
            "slices": len(validation_data),
            "positive_clips_by_class": class_counts(
                clip_labels[np.asarray(sorted(validation_clips))], label_map
            ),
        },
        "limitations": [
            split["limitation"],
            "This diagnostic repurposes BirdSet SSW test_5s for development, so it is not an independent BirdSet test.",
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
