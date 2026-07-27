"""Define and validate the canonical BirdSet SSW test sample specification.

All current zero-shot, few-shot, FP32, and INT8 comparisons use the test
partition left after removing the grouped 20-shot support set. Because the
5-shot and 10-shot support recordings are nested inside the 20-shot support,
this is the largest common BirdSet partition that is leakage-free for every
adaptation scale.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPOSITORY_ROOT / "src" / "dataset_processing" / "output"
DEFAULT_PROTOCOL_OUTPUT = (
    REPOSITORY_ROOT
    / "src"
    / "experiments"
    / "BirdSet_common_test_8class"
    / "test_protocol.json"
)

FEATURES = ("MFCC", "LogMel", "PCEN")
FEATURE_PREFIX = {"MFCC": "mfcc", "LogMel": "logmel", "PCEN": "pcen"}
CANONICAL_SPLIT_SHOTS = 20
CANONICAL_SPLIT_SEED = 42
CANONICAL_SCOPE = "common_20shot_heldout_197_recordings"
CANONICAL_REPORT_NAME = "BirdSet_common_20shot_heldout_evaluation.json"
EXPECTED_RECORDINGS = 197
EXPECTED_CLIPS = 18_265
EXPECTED_SLICES = 91_325
EXPECTED_SLICES_PER_CLIP = 5
EXPECTED_SAMPLE_RATE = 16_000
EXPECTED_CLASSES = 8
RECORDING_PATTERN = re.compile(r"^(.*)_\d+_\d+\.ogg$")


def _repository_relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


def canonical_dataset_dir(data_root: Path, feature: str) -> Path:
    return birdset_split_dir(data_root, feature, CANONICAL_SPLIT_SHOTS) / "test"


def birdset_split_dir(data_root: Path, feature: str, shots: int) -> Path:
    if feature not in FEATURES:
        raise ValueError(f"Unsupported BirdSet feature: {feature}")
    if shots == 5:
        return data_root / f"{feature}_BirdSet_external_split_8class"
    if shots in (10, 20):
        return data_root / f"{feature}_BirdSet_external_split_{shots}shot_8class"
    raise ValueError(f"Unsupported BirdSet shot scale: {shots}")


def _recording_name(filepath: str) -> str:
    match = RECORDING_PATTERN.match(Path(filepath).name)
    if match is None:
        raise ValueError(f"Unexpected BirdSet clip filename: {filepath!r}")
    return match.group(1)


def _identity_payload(clips: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "filepath": str(clip["filepath"]),
            "labels": [int(value) for value in clip["labels"]],
            "slices": int(clip["slices"]),
        }
        for clip in clips
    ]


def _identity_sha256(clips: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        _identity_payload(clips),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@lru_cache(maxsize=None)
def audit_dataset(
    dataset_dir: Path,
    require_canonical: bool = True,
) -> dict[str, Any]:
    dataset_dir = dataset_dir.resolve()
    manifest = json.loads(
        (dataset_dir / "manifest.json").read_text(encoding="utf-8")
    )
    label_map = json.loads(
        (dataset_dir / "label_map.json").read_text(encoding="utf-8")
    )
    clips = manifest["clips"]
    features = np.load(dataset_dir / "test_data.npy", mmap_mode="r")
    clip_index = np.load(dataset_dir / "test_clip_index.npy", mmap_mode="r")
    clip_labels = np.load(
        dataset_dir / "test_clip_multilabel.npy",
        mmap_mode="r",
    )

    if sorted(label_map.values()) != list(range(EXPECTED_CLASSES)):
        raise ValueError(f"BirdSet label IDs are not contiguous: {dataset_dir}")
    if features.ndim != 3 or features.shape[1] != 32:
        raise ValueError(f"Unexpected BirdSet feature shape: {features.shape}")
    if len(features) != len(clip_index):
        raise ValueError("BirdSet feature and clip-index counts differ.")
    if clip_labels.shape != (len(clips), EXPECTED_CLASSES):
        raise ValueError(
            "BirdSet clip-label shape does not match the manifest and classes."
        )
    if not len(clips):
        raise ValueError("BirdSet test manifest is empty.")
    if int(clip_index.min()) != 0 or int(clip_index.max()) + 1 != len(clips):
        raise ValueError("BirdSet clip indices are not contiguous.")
    clip_counts = np.bincount(
        np.asarray(clip_index, dtype=np.int64),
        minlength=len(clips),
    )
    if not np.all(clip_counts == EXPECTED_SLICES_PER_CLIP):
        raise ValueError("Every BirdSet clip must contain exactly five slices.")

    for index, clip in enumerate(clips):
        expected = np.zeros(EXPECTED_CLASSES, dtype=np.uint8)
        expected[[int(value) for value in clip["labels"]]] = 1
        if not np.array_equal(expected, clip_labels[index]):
            raise ValueError(
                f"BirdSet manifest labels differ from the multi-hot array at {index}."
            )

    recordings = sorted({_recording_name(clip["filepath"]) for clip in clips})
    feature_type = manifest.get("feature_type")
    if feature_type is None and "mfcc_shape" in manifest:
        feature_type = "mfcc"
    if feature_type is None:
        raise ValueError(f"BirdSet feature type is missing: {dataset_dir}")
    sample_spec = {
        "protocol_id": (
            "birdset_ssw_common_20shot_heldout_v1"
            if require_canonical
            else "birdset_ssw_diagnostic_partition"
        ),
        "scope": (
            CANONICAL_SCOPE
            if require_canonical
            else "noncanonical_diagnostic_partition"
        ),
        "dataset": "BirdSet SSW test_5s",
        "source_split": "SSW test_5s shards 1-4",
        "partition_rule": (
            "Seed-42 grouped 20-shot support recordings are excluded. The "
            "nested 5-shot and 10-shot supports are therefore excluded too."
        ),
        "split_unit": "original long recording",
        "split_seed": CANONICAL_SPLIT_SEED,
        "maximum_support_scale_excluded": CANONICAL_SPLIT_SHOTS,
        "recordings": len(recordings),
        "clips_5s": len(clips),
        "slices_1s": int(len(features)),
        "slices_per_clip": EXPECTED_SLICES_PER_CLIP,
        "sample_rate_hz": int(manifest["sample_rate"]),
        "feature_type": str(feature_type),
        "feature_shape": [int(value) for value in features.shape[1:]],
        "classes": len(label_map),
        "globally_singleton_clips": int(
            sum(bool(clip["is_globally_singleton"]) for clip in clips)
        ),
        "sample_identity_sha256": _identity_sha256(clips),
        "dataset_dir": _repository_relative(dataset_dir),
        "known_limitation": (
            "Setophaga_ruticilla has only two target clips in the complete "
            "filtered SSW set; both belong to an excluded support recording, "
            "so this held-out partition has no positive clip for that class."
        ),
    }

    if require_canonical:
        expected = {
            "recordings": EXPECTED_RECORDINGS,
            "clips_5s": EXPECTED_CLIPS,
            "slices_1s": EXPECTED_SLICES,
            "sample_rate_hz": EXPECTED_SAMPLE_RATE,
            "classes": EXPECTED_CLASSES,
        }
        actual = {key: sample_spec[key] for key in expected}
        if actual != expected:
            raise ValueError(
                f"BirdSet dataset is not the canonical common test: "
                f"expected {expected}, got {actual}."
            )
        split_manifest = json.loads(
            (dataset_dir.parent / "split_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        split_expected = {
            "seed": CANONICAL_SPLIT_SEED,
            "requested_shots_per_class": CANONICAL_SPLIT_SHOTS,
            "heldout_recordings": EXPECTED_RECORDINGS,
            "heldout_clips": EXPECTED_CLIPS,
        }
        split_actual = {
            key: int(split_manifest[key]) for key in split_expected
        }
        if split_actual != split_expected:
            raise ValueError(
                f"BirdSet split manifest differs from the canonical protocol: "
                f"expected {split_expected}, got {split_actual}."
            )
        support_manifest = json.loads(
            (dataset_dir.parent / "support" / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        support_recordings = {
            _recording_name(clip["filepath"])
            for clip in support_manifest["clips"]
        }
        overlap = support_recordings.intersection(recordings)
        if overlap:
            raise ValueError(
                f"BirdSet canonical support/test recording leakage: {sorted(overlap)}"
            )
        sample_spec["support_test_recording_overlap"] = 0
    return sample_spec


def validate_all_features(data_root: Path) -> dict[str, Any]:
    feature_specs = {
        feature: audit_dataset(canonical_dataset_dir(data_root, feature))
        for feature in FEATURES
    }
    identities = {
        spec["sample_identity_sha256"] for spec in feature_specs.values()
    }
    if len(identities) != 1:
        raise ValueError("Canonical BirdSet sample identities differ by feature.")
    for feature, spec in feature_specs.items():
        if spec["feature_type"] != FEATURE_PREFIX[feature]:
            raise ValueError(
                f"BirdSet feature metadata mismatch for {feature}: "
                f"{spec['feature_type']}"
            )

    support_recordings: dict[str, dict[int, set[str]]] = {}
    support_summary: dict[str, dict[str, int]] = {}
    for feature in FEATURES:
        common_test_manifest = json.loads(
            (
                canonical_dataset_dir(data_root, feature) / "manifest.json"
            ).read_text(encoding="utf-8")
        )
        common_test_recordings = {
            _recording_name(clip["filepath"])
            for clip in common_test_manifest["clips"]
        }
        support_recordings[feature] = {}
        for shots in (5, 10, 20):
            split_dir = birdset_split_dir(data_root, feature, shots)
            split_manifest = json.loads(
                (split_dir / "split_manifest.json").read_text(encoding="utf-8")
            )
            support_manifest = json.loads(
                (split_dir / "support" / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            recordings = {
                _recording_name(clip["filepath"])
                for clip in support_manifest["clips"]
            }
            if int(split_manifest["seed"]) != CANONICAL_SPLIT_SEED:
                raise ValueError(
                    f"BirdSet {feature} {shots}-shot split seed is not 42."
                )
            if int(split_manifest["requested_shots_per_class"]) != shots:
                raise ValueError(
                    f"BirdSet {feature} split does not match {shots}-shot."
                )
            if int(split_manifest["support_recordings"]) != len(recordings):
                raise ValueError(
                    f"BirdSet {feature} {shots}-shot recording count differs "
                    "between split and support manifests."
                )
            overlap = recordings.intersection(common_test_recordings)
            if overlap:
                raise ValueError(
                    f"BirdSet {feature} {shots}-shot support leaks into the "
                    f"common test: {sorted(overlap)}"
                )
            support_recordings[feature][shots] = recordings
            if feature == FEATURES[0]:
                support_summary[str(shots)] = {
                    "recordings": len(recordings),
                    "clips_5s": len(support_manifest["clips"]),
                }
        if not (
            support_recordings[feature][5]
            <= support_recordings[feature][10]
            <= support_recordings[feature][20]
        ):
            raise ValueError(
                f"BirdSet {feature} grouped support sets are not nested."
            )

    for shots in (5, 10, 20):
        reference_recordings = support_recordings[FEATURES[0]][shots]
        if any(
            support_recordings[feature][shots] != reference_recordings
            for feature in FEATURES[1:]
        ):
            raise ValueError(
                f"BirdSet {shots}-shot support recordings differ by feature."
            )

    reference = feature_specs[FEATURES[0]]
    return {
        "protocol_id": reference["protocol_id"],
        "scope": CANONICAL_SCOPE,
        "status": "validated",
        "selection_reason": (
            "This is the largest fixed BirdSet SSW partition that excludes all "
            "recordings used by the nested grouped 5/10/20-shot support sets."
        ),
        "sample_specification": {
            key: reference[key]
            for key in (
                "dataset",
                "source_split",
                "partition_rule",
                "split_unit",
                "split_seed",
                "maximum_support_scale_excluded",
                "recordings",
                "clips_5s",
                "slices_1s",
                "slices_per_clip",
                "sample_rate_hz",
                "classes",
                "globally_singleton_clips",
                "sample_identity_sha256",
                "support_test_recording_overlap",
                "known_limitation",
            )
        },
        "feature_variants": feature_specs,
        "validation": {
            "cross_feature_test_sample_identities_match": True,
            "cross_feature_support_recording_identities_match": True,
            "support_scales_are_nested": True,
            "support_test_recording_overlap": 0,
            "excluded_support_scales": support_summary,
        },
        "comparison_rule": (
            "Only BirdSet metrics produced on this protocol may appear in the "
            "current cross-chain ranking. Full-211, 5-shot-heldout-201, and "
            "10-shot-heldout-200 reports are historical diagnostics."
        ),
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_PROTOCOL_OUTPUT)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    report = validate_all_features(arguments.data_root.resolve())
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report["sample_specification"], indent=2))
    print(f"Saved protocol: {arguments.output}")


if __name__ == "__main__":
    main()
