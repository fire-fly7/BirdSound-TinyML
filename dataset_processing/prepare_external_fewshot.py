"""Create disjoint few-shot support and held-out test partitions for external data.

BirdSet is grouped by original long recording. DB3V is stratified by region
and class at the eight-second recording level. The held-out directories keep
the filenames expected by the existing evaluators.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


RECORDING_PATTERN = re.compile(r"^(.*)_\d+_\d+\.ogg$")
SLICES_PER_DB3V_RECORDING = 8


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("birdset", "db3v"))
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def subset_birdset(
    features: np.ndarray,
    source_indices: np.ndarray,
    selected_clips: list[int],
    clip_targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = np.asarray(sorted(selected_clips), dtype=np.int64)
    remap = {old: new for new, old in enumerate(selected.tolist())}
    rows = np.flatnonzero(np.isin(source_indices, selected))
    indices = np.asarray([remap[int(old)] for old in source_indices[rows]], dtype=np.int64)
    return np.asarray(features[rows]), indices, clip_targets[selected]


def split_birdset(source: Path, output: Path, shots: int, seed: int) -> None:
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    clips = manifest["clips"]
    features = np.load(source / "test_data.npy", mmap_mode="r")
    source_indices = np.load(source / "test_clip_index.npy")
    targets = np.load(source / "test_clip_multilabel.npy")
    label_map = json.loads((source / "label_map.json").read_text(encoding="utf-8"))

    groups: dict[str, list[int]] = defaultdict(list)
    for index, clip in enumerate(clips):
        match = RECORDING_PATTERN.match(Path(clip["filepath"]).name)
        if match is None:
            raise ValueError(f"Cannot group BirdSet clip {clip['filepath']!r}.")
        groups[match.group(1)].append(index)
    names = sorted(groups)
    random = np.random.default_rng(seed)
    random.shuffle(names)
    positives = np.zeros(len(label_map), dtype=np.int64)
    support_groups: set[str] = set()
    for name in names:
        group_positive = targets[groups[name]].sum(axis=0)
        if np.any((positives < shots) & (group_positive > 0)):
            support_groups.add(name)
            positives += group_positive.astype(np.int64)
        if np.all(positives >= shots):
            break
    support_clips = [index for name in support_groups for index in groups[name]]
    test_clips = [
        index for name, members in groups.items() if name not in support_groups for index in members
    ]
    support_data, support_indices, support_targets = subset_birdset(
        features, source_indices, support_clips, targets
    )
    test_data, test_indices, test_targets = subset_birdset(
        features, source_indices, test_clips, targets
    )
    support_dir, test_dir = output / "support", output / "test"
    support_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    np.save(support_dir / "support_data.npy", support_data)
    np.save(support_dir / "support_clip_index.npy", support_indices)
    np.save(support_dir / "support_clip_multilabel.npy", support_targets)
    np.save(test_dir / "test_data.npy", test_data)
    np.save(test_dir / "test_clip_index.npy", test_indices)
    np.save(test_dir / "test_clip_multilabel.npy", test_targets)
    for directory in (support_dir, test_dir):
        write_json(directory / "label_map.json", label_map)
    support_manifest = [clips[index] for index in sorted(support_clips)]
    test_manifest = [clips[index] for index in sorted(test_clips)]
    write_json(
        support_dir / "manifest.json",
        {**{key: value for key, value in manifest.items() if key != "clips"}, "clips": support_manifest},
    )
    write_json(
        test_dir / "manifest.json",
        {**{key: value for key, value in manifest.items() if key != "clips"}, "clips": test_manifest},
    )
    audit = {
        "dataset": "BirdSet SSW",
        "split_unit": "original long recording",
        "requested_shots_per_class": shots,
        "support_recordings": len(support_groups),
        "support_clips": len(support_clips),
        "heldout_recordings": len(groups) - len(support_groups),
        "heldout_clips": len(test_clips),
        "support_positive_clips_by_class": positives.tolist(),
        "note": "A grouped support set can exceed the requested shots because recordings contain many clips.",
    }
    write_json(output / "split_manifest.json", audit)
    print(json.dumps(audit, indent=2))


def split_db3v(source: Path, output: Path, shots: int, seed: int) -> None:
    label_map = json.loads((source / "label_map.json").read_text(encoding="utf-8"))
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    random = np.random.default_rng(seed)
    support_features: list[np.ndarray] = []
    support_labels: list[np.ndarray] = []
    support_regions: list[np.ndarray] = []
    support_manifest: list[dict] = []
    heldout_manifest: list[dict] = []
    offset = 0
    output.mkdir(parents=True, exist_ok=True)
    for region in (1, 2, 3):
        features = np.load(source / f"region_{region}_data.npy", mmap_mode="r")
        labels = np.load(source / f"region_{region}_label.npy")
        recording_labels = labels.reshape(-1, SLICES_PER_DB3V_RECORDING)[:, 0]
        selected: set[int] = set()
        for label in range(len(label_map)):
            candidates = np.flatnonzero(recording_labels == label)
            random.shuffle(candidates)
            selected.update(candidates[: min(shots, len(candidates))].tolist())
        support_rows = np.concatenate(
            [
                np.arange(index * SLICES_PER_DB3V_RECORDING, (index + 1) * SLICES_PER_DB3V_RECORDING)
                for index in sorted(selected)
            ]
        )
        heldout_recordings = sorted(set(range(len(recording_labels))) - selected)
        heldout_rows = np.concatenate(
            [
                np.arange(index * SLICES_PER_DB3V_RECORDING, (index + 1) * SLICES_PER_DB3V_RECORDING)
                for index in heldout_recordings
            ]
        )
        region_test_data = np.asarray(features[heldout_rows])
        region_test_labels = labels[heldout_rows]
        np.save(output / f"region_{region}_data.npy", region_test_data)
        np.save(output / f"region_{region}_label.npy", region_test_labels)
        support_features.append(np.asarray(features[support_rows]))
        support_labels.append(labels[support_rows])
        support_regions.append(np.full(len(support_rows), region, dtype=np.int8))
        region_manifest = [item for item in manifest if int(item["region"]) == region]
        support_manifest.extend(region_manifest[index] for index in sorted(selected))
        heldout_manifest.extend(region_manifest[index] for index in heldout_recordings)
        offset += len(region_manifest)
    np.save(output / "support_data.npy", np.concatenate(support_features))
    np.save(output / "support_label.npy", np.concatenate(support_labels))
    np.save(output / "support_region.npy", np.concatenate(support_regions))
    write_json(output / "label_map.json", label_map)
    write_json(output / "support_manifest.json", support_manifest)
    write_json(output / "manifest.json", heldout_manifest)
    audit = {
        "dataset": "DB3V",
        "split_unit": "eight-second recording",
        "stratification": "region and class",
        "requested_shots_per_region_class": shots,
        "support_recordings": len(support_manifest),
        "heldout_recordings": len(heldout_manifest),
    }
    write_json(output / "split_manifest.json", audit)
    print(json.dumps(audit, indent=2))


def main() -> None:
    args = arguments()
    if args.shots < 1:
        raise ValueError("--shots must be positive.")
    if args.dataset == "birdset":
        split_birdset(args.source_dir, args.output_dir, args.shots, args.seed)
    else:
        split_db3v(args.source_dir, args.output_dir, args.shots, args.seed)


if __name__ == "__main__":
    main()
