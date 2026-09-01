"""Audit strict DB3V few-shot partitions at original source-recording level."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPOSITORY_ROOT / "src" / "dataset_processing" / "output"
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "src"
    / "experiments"
    / "DB3V_fewshot_ablation_multiseed_8class"
    / "split_audit.json"
)
FEATURES = ("MFCC", "LogMel", "PCEN")
SHOTS = (5, 10, 20)
SOURCE_PATTERN = re.compile(r"^(\d+)(?:_\d+)+$")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def split_dir(root: Path, feature: str, shots: int) -> Path:
    suffix = "" if shots == 5 else f"_{shots}shot"
    return root / f"{feature}_DB3V_external_split{suffix}_8class"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def source_id(item: dict[str, Any]) -> str:
    stem = Path(str(item["path"]).replace("\\", "/")).stem
    match = SOURCE_PATTERN.fullmatch(stem)
    if match is None:
        raise ValueError(f"Cannot recover original DB3V source ID from {stem!r}.")
    recovered = match.group(1)
    declared = str(item.get("source_recording_id", recovered))
    if declared != recovered:
        raise ValueError(
            f"Declared source ID {declared!r} differs from path-derived {recovered!r}."
        )
    return recovered


def identity(items: list[dict[str, Any]]) -> list[tuple[str, int, int, str]]:
    result = []
    for item in items:
        path = str(item["path"]).replace("\\", "/")
        if path.startswith("src/"):
            path = path[4:]
        result.append(
            (
            path,
            int(item["label"]),
            int(item["region"]),
            source_id(item),
        )
        )
    return result


def validate_arrays(directory: Path, support: list[dict], heldout: list[dict]) -> None:
    support_rows = sum(int(item["slices"]) for item in support)
    support_data = np.load(directory / "support_data.npy", mmap_mode="r")
    support_labels = np.load(directory / "support_label.npy", mmap_mode="r")
    support_regions = np.load(directory / "support_region.npy", mmap_mode="r")
    if not (len(support_data) == len(support_labels) == len(support_regions) == support_rows):
        raise ValueError(f"Support arrays/manifests differ in {directory}.")
    for region in (1, 2, 3):
        items = [item for item in heldout if int(item["region"]) == region]
        expected = sum(int(item["slices"]) for item in items)
        data = np.load(directory / f"region_{region}_data.npy", mmap_mode="r")
        labels = np.load(directory / f"region_{region}_label.npy", mmap_mode="r")
        if len(data) != expected or len(labels) != expected:
            raise ValueError(f"Region {region} arrays/manifests differ in {directory}.")


def xeno_ids(data_root: Path) -> tuple[set[str], set[str]]:
    manifest = load_json(data_root / "MFCC_dataset_A_8class" / "split_manifest.json")
    train = {str(item["recording_id"]) for item in manifest["train"]}
    validation = {str(item["recording_id"]) for item in manifest["validation"]}
    if train & validation:
        raise ValueError("Xeno-canto training and validation recording IDs overlap.")
    return train, validation


def main() -> None:
    args = arguments()
    data_root = args.data_root.resolve()
    per_split: dict[str, Any] = {}
    identities: dict[tuple[str, int, str], list[tuple[str, int, int, str]]] = {}
    support_sources: dict[tuple[str, int], set[str]] = {}
    heldout_sources: dict[tuple[str, int], set[str]] = {}

    for feature in FEATURES:
        for shots in SHOTS:
            directory = split_dir(data_root, feature, shots)
            protocol = load_json(directory / "split_manifest.json")
            support = load_json(directory / "support_manifest.json")
            heldout = load_json(directory / "manifest.json")
            validate_arrays(directory, support, heldout)
            support_set = {source_id(item) for item in support}
            heldout_set = {source_id(item) for item in heldout}
            overlap = support_set & heldout_set
            if overlap:
                raise ValueError(
                    f"{feature} {shots}-shot leaks {len(overlap)} source recordings."
                )
            if protocol["protocol_id"] != "db3v_original_xc_recording_grouped_v2":
                raise ValueError(f"Unexpected split protocol in {directory}.")
            if int(protocol["support_source_recordings"]) != len(support_set):
                raise ValueError(f"Support source count differs in {directory}.")
            if int(protocol["heldout_source_recordings"]) != len(heldout_set):
                raise ValueError(f"Held-out source count differs in {directory}.")
            if int(protocol["support_test_source_id_overlap"]) != 0:
                raise ValueError(f"Protocol records source leakage in {directory}.")
            identities[(feature, shots, "support")] = identity(support)
            identities[(feature, shots, "heldout")] = identity(heldout)
            support_sources[(feature, shots)] = support_set
            heldout_sources[(feature, shots)] = heldout_set
            per_split[f"{feature}_{shots}shot"] = {
                "support_source_recordings": len(support_set),
                "heldout_source_recordings": len(heldout_set),
                "support_eight_second_recordings": len(support),
                "heldout_eight_second_recordings": len(heldout),
                "support_slices": int(sum(int(item["slices"]) for item in support)),
                "heldout_slices": int(sum(int(item["slices"]) for item in heldout)),
                "support_shortfall_source_recordings": int(
                    protocol["support_shortfall_source_recordings"]
                ),
                "support_heldout_source_overlap": 0,
            }

    for shots in SHOTS:
        reference_support = identities[(FEATURES[0], shots, "support")]
        reference_heldout = identities[(FEATURES[0], shots, "heldout")]
        for feature in FEATURES[1:]:
            if identities[(feature, shots, "support")] != reference_support:
                raise ValueError(f"Cross-feature support identities differ at {shots}-shot.")
            if identities[(feature, shots, "heldout")] != reference_heldout:
                raise ValueError(f"Cross-feature held-out identities differ at {shots}-shot.")

    for feature in FEATURES:
        if not (
            support_sources[(feature, 5)]
            <= support_sources[(feature, 10)]
            <= support_sources[(feature, 20)]
        ):
            raise ValueError(f"{feature} support source sets are not nested.")
        if not (
            heldout_sources[(feature, 20)]
            <= heldout_sources[(feature, 10)]
            <= heldout_sources[(feature, 5)]
        ):
            raise ValueError(f"{feature} held-out source sets are not nested.")

    train_ids, validation_ids = xeno_ids(data_root)
    db3v_ids = support_sources[("MFCC", 20)] | heldout_sources[("MFCC", 20)]
    train_overlap = train_ids & db3v_ids
    validation_overlap = validation_ids & db3v_ids
    if train_overlap or validation_overlap:
        raise ValueError(
            "DB3V source IDs overlap Xeno-canto training/validation IDs: "
            f"train={len(train_overlap)}, validation={len(validation_overlap)}."
        )

    report = {
        "audit": "DB3V original-source-recording split integrity",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "protocol_id": "db3v_original_xc_recording_grouped_v2",
        "features": list(FEATURES),
        "shots": list(SHOTS),
        "checks": {
            "support_heldout_source_overlap_zero": True,
            "support_sets_nested_5_10_20": True,
            "heldout_sets_nested_20_10_5": True,
            "cross_feature_recording_identity_equal": True,
            "array_manifest_lengths_equal": True,
            "xeno_train_db3v_source_overlap": len(train_overlap),
            "xeno_validation_db3v_source_overlap": len(validation_overlap),
        },
        "xeno_recordings": {
            "train": len(train_ids),
            "validation": len(validation_ids),
        },
        "db3v_source_recordings": len(db3v_ids),
        "splits": per_split,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved audit: {output}")


if __name__ == "__main__":
    main()
