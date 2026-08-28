"""Prepare an exact training-test replay package for the STM32 project.

The package has two complementary layers:

* ``tensors/full`` materializes the exact arrays used by the 57 strict-INT8
  desktop evaluations.  These arrays reproduce the reported metrics.
* ``tensors/probe`` contains deterministic, group-preserving subsets for fast
  TFLite-versus-TFLM migration checks.

The raw PCM probe is built only from local Xeno-canto validation and DB3V
held-out recordings.  It is deliberately marked ``source_label``: it verifies
the embedded frontend and end-to-end plumbing, but is not a manually reviewed
independent scientific test set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import wave
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from data_sugment_MFCC import SAMPLE_RATE, features_for_segments

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPOSITORY_ROOT / "src" / "dataset_processing" / "output"
EXPERIMENTS_ROOT = REPOSITORY_ROOT / "src" / "experiments"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "board_replay_testset"
FEATURES = ("MFCC", "LogMel", "PCEN")
FEATURE_KEYS = {"MFCC": "mfcc", "LogMel": "logmel", "PCEN": "pcen"}
REGIONS = (1, 2, 3)
SAMPLES_PER_WINDOW = 16_000
SLICES_PER_DB3V_RECORDING = 8
SLICES_PER_BIRDSET_CLIP = 5
PACKAGE_SCHEMA = 1
PROBE_SEED = 20260729
STRICT_ROOTS = (
    ("zero_shot", "ZeroShot_strict_INT8_quantization_8class"),
    ("db3v_strict_fewshot", "DB3V_strict_INT8_quantization_8class"),
    ("birdset_strict_fewshot", "BirdSet_strict_INT8_quantization_8class"),
)
EXPECTED_LABEL_MAP = {
    "Agelaius_phoeniceus": 0,
    "Cardinalis_cardinalis": 1,
    "Certhia_americana": 2,
    "Corvus_brachyrhynchos": 3,
    "Setophaga_aestiva": 4,
    "Setophaga_ruticilla": 5,
    "Spinus_tristis": 6,
    "Turdus_migratorius": 7,
}
MANIFEST_FIELDS = (
    "sample_id",
    "file",
    "label",
    "species",
    "source_dataset",
    "source_recording_id",
    "start_sample",
    "duration_samples",
    "split",
    "annotation_status",
    "mixed_species",
    "leakage_check",
    "pcm_sha256",
    "wav_sha256",
    "tensor_scope",
    "source_region",
    "source_feature_index",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--materialize",
        choices=("hardlink", "copy", "reference"),
        default="hardlink",
        help=(
            "hardlink avoids duplicating about 3 GB on this machine; copy creates "
            "a standalone tree; reference writes only manifests and probes"
        ),
    )
    parser.add_argument(
        "--skip-pcm-probe",
        action="store_true",
        help="Do not create the local Xeno/DB3V one-second PCM probe.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def relative_to_repository(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def package_relative(path: Path, output_dir: Path) -> str:
    return path.resolve().relative_to(output_dir.resolve()).as_posix()


def sha256_file(path: Path, cache: dict[Path, str]) -> str:
    resolved = path.resolve()
    if resolved in cache:
        return cache[resolved]
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    cache[resolved] = value
    return value


def stable_key(value: str) -> str:
    return hashlib.sha256(f"{PROBE_SEED}:{value}".encode()).hexdigest()


def safe_destination(path: Path, output_dir: Path) -> Path:
    resolved_output = output_dir.resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(resolved_output)
    except ValueError as exc:
        raise ValueError(f"Package destination escapes output root: {path}") from exc
    return resolved


def materialize_file(
    source: Path,
    destination: Path,
    mode: str,
    output_dir: Path,
) -> str:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = safe_destination(destination, output_dir)
    if mode == "reference":
        return "reference"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        try:
            if mode == "hardlink" and os.path.samefile(source, destination):
                return "hardlink"
        except OSError:
            pass
        destination.unlink()
    if mode == "hardlink":
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError:
            shutil.copy2(source, destination)
            return "copy_fallback"
    shutil.copy2(source, destination)
    return "copy"


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(fields))
        writer.writeheader()
        writer.writerows(rows)


def validate_label_maps() -> None:
    paths = [
        DATA_ROOT / f"{feature}_dataset_A_8class" / "label_map.json"
        for feature in FEATURES
    ]
    for path in paths:
        if load_json(path) != EXPECTED_LABEL_MAP:
            raise ValueError(f"Unexpected label map: {path}")


def xeno_dir(feature: str) -> Path:
    return DATA_ROOT / f"{feature}_dataset_A_8class"


def birdset_dir(feature: str) -> Path:
    return DATA_ROOT / f"{feature}_BirdSet_external_split_20shot_8class" / "test"


def db3v_dir(feature: str, common_heldout: bool) -> Path:
    if common_heldout:
        return DATA_ROOT / f"{feature}_DB3V_external_split_20shot_8class"
    return DATA_ROOT / f"{feature}_dataset_DB3V_8class"


def canonical_record_path(value: str) -> str:
    result = value.replace("\\", "/")
    result = result.removeprefix("src/")
    return result


def resolve_raw_path(value: str) -> Path:
    normalized = canonical_record_path(value)
    if normalized.startswith("row_dataset/"):
        return REPOSITORY_ROOT / "src" / normalized
    return REPOSITORY_ROOT / normalized


def source_windows_are_nonzero(
    item: dict[str, Any],
    slice_indices: Iterable[int],
) -> bool:
    source = resolve_raw_path(str(item["path"]))
    with wave.open(str(source), "rb") as wav:
        if (
            wav.getnchannels(),
            wav.getsampwidth(),
            wav.getframerate(),
            wav.getcomptype(),
        ) != (1, 2, SAMPLE_RATE, "NONE"):
            return False
        for slice_index in slice_indices:
            start = int(slice_index) * SAMPLES_PER_WINDOW
            if start + SAMPLES_PER_WINDOW > wav.getnframes():
                return False
            wav.setpos(start)
            samples = np.frombuffer(
                wav.readframes(SAMPLES_PER_WINDOW), dtype="<i2"
            )
            if not np.any(samples):
                return False
    return True


def validate_cross_feature_identity() -> dict[str, Any]:
    report: dict[str, Any] = {}

    xeno_reference = xeno_dir("MFCC")
    reference_labels = np.load(xeno_reference / "validation_label.npy")
    reference_indices = np.load(xeno_reference / "validation_recording_index.npy")
    reference_manifest = load_json(xeno_reference / "split_manifest.json")
    xeno_identity = [
        (
            canonical_record_path(item["path"]),
            int(item["label"]),
            tuple(int(value) for value in item["selected_slice_indices"]),
        )
        for item in reference_manifest["validation"]
    ]
    for feature in FEATURES[1:]:
        directory = xeno_dir(feature)
        if not np.array_equal(
            reference_labels, np.load(directory / "validation_label.npy")
        ):
            raise ValueError(f"Xeno validation labels differ for {feature}.")
        if not np.array_equal(
            reference_indices,
            np.load(directory / "validation_recording_index.npy"),
        ):
            raise ValueError(f"Xeno recording indices differ for {feature}.")
        manifest = load_json(directory / "split_manifest.json")
        identity = [
            (
                canonical_record_path(item["path"]),
                int(item["label"]),
                tuple(int(value) for value in item["selected_slice_indices"]),
            )
            for item in manifest["validation"]
        ]
        if identity != xeno_identity:
            raise ValueError(f"Xeno validation manifest differs for {feature}.")
    report["xeno_validation"] = {
        "slices": len(reference_labels),
        "recordings": len(xeno_identity),
        "cross_feature_identity": True,
    }

    bird_reference = birdset_dir("MFCC")
    reference_clip_index = np.load(bird_reference / "test_clip_index.npy")
    reference_clip_labels = np.load(bird_reference / "test_clip_multilabel.npy")
    reference_clips = load_json(bird_reference / "manifest.json")["clips"]
    bird_identity = [
        (item["filepath"], tuple(int(value) for value in item["labels"]))
        for item in reference_clips
    ]
    for feature in FEATURES[1:]:
        directory = birdset_dir(feature)
        if not np.array_equal(
            reference_clip_index, np.load(directory / "test_clip_index.npy")
        ):
            raise ValueError(f"BirdSet clip indices differ for {feature}.")
        if not np.array_equal(
            reference_clip_labels,
            np.load(directory / "test_clip_multilabel.npy"),
        ):
            raise ValueError(f"BirdSet clip labels differ for {feature}.")
        clips = load_json(directory / "manifest.json")["clips"]
        identity = [
            (item["filepath"], tuple(int(value) for value in item["labels"]))
            for item in clips
        ]
        if identity != bird_identity:
            raise ValueError(f"BirdSet manifest differs for {feature}.")
    report["birdset_common_20shot_heldout"] = {
        "slices": len(reference_clip_index),
        "clips": len(reference_clip_labels),
        "recordings": 197,
        "cross_feature_identity": True,
    }

    for common, scope in (
        (False, "db3v_full"),
        (True, "db3v_common_20shot_heldout"),
    ):
        reference_dir = db3v_dir("MFCC", common)
        reference_manifest = load_json(reference_dir / "manifest.json")
        reference_identity = [
            (
                canonical_record_path(item["path"]),
                int(item["region"]),
                int(item["label"]),
            )
            for item in reference_manifest
        ]
        total_slices = 0
        for region in REGIONS:
            reference_region_labels = np.load(
                reference_dir / f"region_{region}_label.npy"
            )
            total_slices += len(reference_region_labels)
            for feature in FEATURES[1:]:
                labels = np.load(
                    db3v_dir(feature, common) / f"region_{region}_label.npy"
                )
                if not np.array_equal(reference_region_labels, labels):
                    raise ValueError(
                        f"{scope} region {region} labels differ for {feature}."
                    )
        for feature in FEATURES[1:]:
            manifest = load_json(db3v_dir(feature, common) / "manifest.json")
            identity = [
                (
                    canonical_record_path(item["path"]),
                    int(item["region"]),
                    int(item["label"]),
                )
                for item in manifest
            ]
            if identity != reference_identity:
                raise ValueError(f"{scope} manifest differs for {feature}.")
        report[scope] = {
            "slices": int(total_slices),
            "recordings": len(reference_identity),
            "cross_feature_identity": True,
        }
    return report


def add_artifact(
    source: Path,
    destination: Path,
    mode: str,
    output_dir: Path,
    artifacts: list[dict[str, Any]],
    hash_cache: dict[Path, str],
) -> None:
    method = materialize_file(source, destination, mode, output_dir)
    artifacts.append(
        {
            "package_file": package_relative(destination, output_dir),
            "source_file": relative_to_repository(source),
            "bytes": source.stat().st_size,
            "sha256": sha256_file(source, hash_cache),
            "materialization": method,
        }
    )


def materialize_full_tensors(
    output_dir: Path,
    mode: str,
    artifacts: list[dict[str, Any]],
    hash_cache: dict[Path, str],
) -> dict[str, Any]:
    scopes: dict[str, Any] = {
        "xeno_validation": {
            "metric_levels": ["slice", "recording"],
            "group_size": "variable, maximum 8 slices per recording",
            "features": {},
        },
        "birdset_common_20shot_heldout": {
            "metric_levels": ["slice_multilabel", "clip_multilabel", "singleton_clip"],
            "group_size": "5 slices per clip",
            "features": {},
        },
        "db3v_full": {
            "metric_levels": ["slice", "recording", "region", "pooled"],
            "group_size": "8 slices per recording",
            "features": {},
        },
        "db3v_common_20shot_heldout": {
            "metric_levels": ["slice", "recording", "region", "pooled"],
            "group_size": "8 slices per recording",
            "features": {},
        },
    }
    for feature in FEATURES:
        source = xeno_dir(feature)
        destination = output_dir / "tensors" / "full" / "xeno_validation" / feature
        mapping = {
            "data.npy": "validation_data.npy",
            "labels.npy": "validation_label.npy",
            "recording_index.npy": "validation_recording_index.npy",
            "manifest.json": "split_manifest.json",
            "label_map.json": "label_map.json",
        }
        for target_name, source_name in mapping.items():
            add_artifact(
                source / source_name,
                destination / target_name,
                mode if source_name.endswith(".npy") else "copy",
                output_dir,
                artifacts,
                hash_cache,
            )
        data = np.load(source / "validation_data.npy", mmap_mode="r")
        scopes["xeno_validation"]["features"][feature] = {
            "directory": package_relative(destination, output_dir),
            "shape": list(data.shape),
            "dtype": str(data.dtype),
        }

        source = birdset_dir(feature)
        destination = (
            output_dir
            / "tensors"
            / "full"
            / "birdset_common_20shot_heldout"
            / feature
        )
        mapping = {
            "data.npy": "test_data.npy",
            "clip_index.npy": "test_clip_index.npy",
            "clip_multilabel.npy": "test_clip_multilabel.npy",
            "manifest.json": "manifest.json",
            "label_map.json": "label_map.json",
        }
        for target_name, source_name in mapping.items():
            add_artifact(
                source / source_name,
                destination / target_name,
                mode if source_name.endswith(".npy") else "copy",
                output_dir,
                artifacts,
                hash_cache,
            )
        data = np.load(source / "test_data.npy", mmap_mode="r")
        scopes["birdset_common_20shot_heldout"]["features"][feature] = {
            "directory": package_relative(destination, output_dir),
            "shape": list(data.shape),
            "dtype": str(data.dtype),
        }

        for common, scope in (
            (False, "db3v_full"),
            (True, "db3v_common_20shot_heldout"),
        ):
            source = db3v_dir(feature, common)
            destination = output_dir / "tensors" / "full" / scope / feature
            metadata_names = ["manifest.json", "label_map.json"]
            if common:
                metadata_names.append("split_manifest.json")
            for name in metadata_names:
                add_artifact(
                    source / name,
                    destination / name,
                    "copy",
                    output_dir,
                    artifacts,
                    hash_cache,
                )
            shapes: dict[str, list[int]] = {}
            for region in REGIONS:
                for kind in ("data", "label"):
                    name = f"region_{region}_{kind}.npy"
                    add_artifact(
                        source / name,
                        destination / name,
                        mode,
                        output_dir,
                        artifacts,
                        hash_cache,
                    )
                shapes[str(region)] = list(
                    np.load(source / f"region_{region}_data.npy", mmap_mode="r").shape
                )
            scopes[scope]["features"][feature] = {
                "directory": package_relative(destination, output_dir),
                "region_shapes": shapes,
                "dtype": "float32",
            }
    return scopes


def choose_xeno_probe() -> list[dict[str, Any]]:
    directory = xeno_dir("MFCC")
    manifest = load_json(directory / "split_manifest.json")["validation"]
    recording_index = np.load(directory / "validation_recording_index.npy")
    offsets = np.flatnonzero(
        np.r_[True, recording_index[1:] != recording_index[:-1]]
    )
    selected: list[dict[str, Any]] = []
    for label in range(len(EXPECTED_LABEL_MAP)):
        candidates = []
        for record_id, item in enumerate(manifest):
            if int(item["label"]) != label or int(item["retained_slices"]) != 8:
                continue
            if not source_windows_are_nonzero(item, item["selected_slice_indices"]):
                continue
            candidates.append((stable_key(str(item["recording_id"])), record_id, item))
        if not candidates:
            raise ValueError(f"No eight-slice Xeno validation recording for label {label}.")
        _, record_id, item = min(candidates)
        start = int(offsets[record_id])
        indices = list(range(start, start + 8))
        if not np.all(recording_index[indices] == record_id):
            raise ValueError("Xeno manifest and recording indices are misaligned.")
        selected.append(
            {
                **item,
                "recording_index": record_id,
                "source_indices": indices,
            }
        )
    return selected


def indexed_db3v_manifest(common: bool) -> list[dict[str, Any]]:
    manifest = load_json(db3v_dir("MFCC", common) / "manifest.json")
    region_offsets: dict[int, int] = defaultdict(int)
    indexed: list[dict[str, Any]] = []
    for item in manifest:
        region = int(item["region"])
        slices = int(item.get("slices", SLICES_PER_DB3V_RECORDING))
        if slices != SLICES_PER_DB3V_RECORDING:
            raise ValueError(f"Unexpected DB3V slice count: {item}")
        start = region_offsets[region]
        region_offsets[region] += slices
        indexed.append(
            {
                **item,
                "region_feature_start": start,
                "source_indices": list(range(start, start + slices)),
            }
        )
    for region in REGIONS:
        expected = len(
            np.load(
                db3v_dir("MFCC", common) / f"region_{region}_label.npy",
                mmap_mode="r",
            )
        )
        if region_offsets[region] != expected:
            raise ValueError(
                f"DB3V manifest has {region_offsets[region]} slices for region "
                f"{region}, expected {expected}."
            )
    return indexed


def choose_db3v_probe(common: bool) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for item in indexed_db3v_manifest(common):
        grouped[(int(item["region"]), int(item["label"]))].append(item)
    selected = []
    for region in REGIONS:
        for label in range(len(EXPECTED_LABEL_MAP)):
            candidates = grouped[(region, label)]
            if not candidates:
                if common:
                    continue
                raise ValueError(f"No DB3V record for region {region}, label {label}.")
            ordered = sorted(
                candidates, key=lambda item: stable_key(str(item["path"]))
            )
            selected_record = next(
                (
                    item
                    for item in ordered
                    if source_windows_are_nonzero(item, range(8))
                ),
                None,
            )
            if selected_record is None:
                raise ValueError(
                    f"No non-silent DB3V record for region {region}, label {label}."
                )
            selected.append(selected_record)
    return selected


def choose_birdset_probe() -> list[int]:
    clips = load_json(birdset_dir("MFCC") / "manifest.json")["clips"]
    selected: list[int] = []
    for label in range(len(EXPECTED_LABEL_MAP)):
        candidates = [
            index
            for index, clip in enumerate(clips)
            if bool(clip["is_globally_singleton"]) and clip["labels"] == [label]
        ]
        if candidates:
            selected.append(
                min(candidates, key=lambda index: stable_key(clips[index]["filepath"]))
            )
    multi_label = [
        index
        for index, clip in enumerate(clips)
        if len(clip["labels"]) > 1 and index not in selected
    ]
    for index in sorted(multi_label, key=lambda value: stable_key(clips[value]["filepath"])):
        selected.append(index)
        if len(selected) == 16:
            break
    if len(selected) < 16:
        remaining = [index for index in range(len(clips)) if index not in selected]
        for index in sorted(
            remaining, key=lambda value: stable_key(clips[value]["filepath"])
        ):
            selected.append(index)
            if len(selected) == 16:
                break
    if len(selected) != 16:
        raise ValueError("Could not select 16 BirdSet probe clips.")
    return selected


def save_probe_array(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, value)


def write_probe_tensors(
    output_dir: Path,
    xeno_records: list[dict[str, Any]],
    db_full_records: list[dict[str, Any]],
    db_common_records: list[dict[str, Any]],
    bird_clip_indices: list[int],
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    xeno_source_indices = np.asarray(
        [
            record["source_indices"][slice_offset]
            for slice_offset in range(8)
            for record in xeno_records
        ],
        dtype=np.int64,
    )
    xeno_labels = np.tile(np.arange(8, dtype=np.int64), 8)
    xeno_recording_index = np.tile(np.arange(8, dtype=np.int64), (8, 1)).T.reshape(-1)
    # The interleaved order is slice 0 labels 0..7, then slice 1 labels 0..7.
    xeno_recording_index = np.tile(np.arange(8, dtype=np.int64), 8)
    for feature in FEATURES:
        source = np.load(xeno_dir(feature) / "validation_data.npy", mmap_mode="r")
        destination = output_dir / "tensors" / "probe" / "xeno_validation" / feature
        save_probe_array(destination / "data.npy", np.asarray(source[xeno_source_indices]))
        save_probe_array(destination / "labels.npy", xeno_labels)
        save_probe_array(destination / "recording_index.npy", xeno_recording_index)
        save_probe_array(destination / "source_indices.npy", xeno_source_indices)
        shutil.copy2(xeno_dir(feature) / "label_map.json", destination / "label_map.json")
    write_json(
        output_dir / "tensors" / "probe" / "xeno_validation" / "manifest.json",
        {"records": xeno_records, "ordering": "slice_offset_then_label"},
    )
    report["xeno_validation"] = {"slices": 64, "recordings": 8}

    for common, scope, records in (
        (False, "db3v_full", db_full_records),
        (True, "db3v_common_20shot_heldout", db_common_records),
    ):
        ordered_records = {
            (int(item["region"]), int(item["label"])): item for item in records
        }
        record_keys = sorted(ordered_records)
        record_ids = {key: index for index, key in enumerate(record_keys)}
        order = [
            (region, label, slice_offset)
            for region in REGIONS
            for slice_offset in range(8)
            for label in range(8)
            if (region, label) in ordered_records
        ]
        labels = np.asarray([label for _, label, _ in order], dtype=np.int64)
        regions = np.asarray([region for region, _, _ in order], dtype=np.int8)
        recording_index = np.asarray(
            [record_ids[(region, label)] for region, label, _ in order],
            dtype=np.int64,
        )
        source_indices = np.asarray(
            [
                ordered_records[(region, label)]["source_indices"][slice_offset]
                for region, label, slice_offset in order
            ],
            dtype=np.int64,
        )
        for feature in FEATURES:
            parts = []
            for region in REGIONS:
                source = np.load(
                    db3v_dir(feature, common) / f"region_{region}_data.npy",
                    mmap_mode="r",
                )
                region_mask = regions == region
                parts.append(np.asarray(source[source_indices[region_mask]]))
            data = np.concatenate(parts)
            destination = output_dir / "tensors" / "probe" / scope / feature
            save_probe_array(destination / "data.npy", data)
            save_probe_array(destination / "labels.npy", labels)
            save_probe_array(destination / "recording_index.npy", recording_index)
            save_probe_array(destination / "region.npy", regions)
            save_probe_array(destination / "source_region_indices.npy", source_indices)
            shutil.copy2(
                db3v_dir(feature, common) / "label_map.json",
                destination / "label_map.json",
            )
        write_json(
            output_dir / "tensors" / "probe" / scope / "manifest.json",
            {"records": records, "ordering": "region_then_slice_offset_then_label"},
        )
        report[scope] = {
            "slices": len(order),
            "recordings": len(records),
            "missing_region_class_strata": [
                {"region": region, "label": label}
                for region in REGIONS
                for label in range(8)
                if (region, label) not in ordered_records
            ],
        }

    bird_source_indices = np.asarray(
        [
            clip_index * SLICES_PER_BIRDSET_CLIP + offset
            for clip_index in bird_clip_indices
            for offset in range(SLICES_PER_BIRDSET_CLIP)
        ],
        dtype=np.int64,
    )
    bird_clip_index = np.repeat(
        np.arange(len(bird_clip_indices), dtype=np.int64),
        SLICES_PER_BIRDSET_CLIP,
    )
    reference_labels = np.load(
        birdset_dir("MFCC") / "test_clip_multilabel.npy"
    )
    for feature in FEATURES:
        source = np.load(birdset_dir(feature) / "test_data.npy", mmap_mode="r")
        destination = (
            output_dir
            / "tensors"
            / "probe"
            / "birdset_common_20shot_heldout"
            / feature
        )
        save_probe_array(destination / "data.npy", np.asarray(source[bird_source_indices]))
        save_probe_array(destination / "clip_index.npy", bird_clip_index)
        save_probe_array(
            destination / "clip_multilabel.npy",
            reference_labels[bird_clip_indices],
        )
        save_probe_array(
            destination / "source_clip_indices.npy",
            np.asarray(bird_clip_indices, dtype=np.int64),
        )
        shutil.copy2(
            birdset_dir(feature) / "label_map.json",
            destination / "label_map.json",
        )
        source_manifest = load_json(birdset_dir(feature) / "manifest.json")
        write_json(
            destination / "manifest.json",
            {
                **{key: value for key, value in source_manifest.items() if key != "clips"},
                "scope": "deterministic_migration_probe_only",
                "clips": [source_manifest["clips"][index] for index in bird_clip_indices],
            },
        )
    report["birdset_common_20shot_heldout"] = {
        "slices": len(bird_source_indices),
        "clips": len(bird_clip_indices),
        "known_limitation": (
            "The canonical held-out set has no Setophaga_ruticilla positive clip."
        ),
    }
    return report


def sanitize_sample_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def extract_pcm_window(source: Path, start_sample: int, destination: Path) -> bytes:
    with wave.open(str(source), "rb") as wav:
        actual = (
            wav.getnchannels(),
            wav.getsampwidth(),
            wav.getframerate(),
            wav.getcomptype(),
        )
        expected = (1, 2, SAMPLE_RATE, "NONE")
        if actual != expected:
            raise ValueError(f"Unsupported source WAV {source}: {actual}, expected {expected}.")
        if start_sample + SAMPLES_PER_WINDOW > wav.getnframes():
            raise ValueError(f"Window exceeds source WAV: {source}, start={start_sample}")
        wav.setpos(start_sample)
        pcm = wav.readframes(SAMPLES_PER_WINDOW)
    if len(pcm) != SAMPLES_PER_WINDOW * 2:
        raise ValueError(f"Short PCM window read from {source}.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(destination), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(pcm)
    return pcm


def validate_pcm_feature(
    pcm: bytes,
    feature: str,
    expected: np.ndarray,
) -> dict[str, float | int]:
    audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    actual = features_for_segments(audio[np.newaxis, :], FEATURE_KEYS[feature])[0]
    difference = actual.astype(np.float64) - np.asarray(expected, dtype=np.float64)
    return {
        "values": int(difference.size),
        "max_abs": float(np.max(np.abs(difference))),
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
    }


def create_raw_probe(
    output_dir: Path,
    xeno_records: list[dict[str, Any]],
    db_common_records: list[dict[str, Any]],
    hash_cache: dict[Path, str],
) -> dict[str, Any]:
    raw_root = output_dir / "raw_pcm_probe"
    rows: list[dict[str, Any]] = []
    expected_audio_paths: set[Path] = set()
    validation: dict[str, list[dict[str, float | int]]] = {
        feature: [] for feature in FEATURES
    }
    species_by_label = {
        label: species for species, label in EXPECTED_LABEL_MAP.items()
    }

    for slice_offset in range(8):
        for record in xeno_records:
            label = int(record["label"])
            source = resolve_raw_path(str(record["path"]))
            source_index = int(record["source_indices"][slice_offset])
            source_slice = int(record["selected_slice_indices"][slice_offset])
            start_sample = source_slice * SAMPLES_PER_WINDOW
            sample_id = sanitize_sample_id(
                f"xeno_val_{record['recording_id']}_s{source_slice:04d}"
            )
            destination = (
                raw_root / "audio" / species_by_label[label] / f"{sample_id}.wav"
            )
            expected_audio_paths.add(destination.resolve())
            pcm = extract_pcm_window(source, start_sample, destination)
            for feature in FEATURES:
                expected = np.load(
                    xeno_dir(feature) / "validation_data.npy", mmap_mode="r"
                )[source_index]
                validation[feature].append(
                    validate_pcm_feature(pcm, feature, expected)
                )
            rows.append(
                {
                    "sample_id": sample_id,
                    "file": package_relative(destination, raw_root),
                    "label": label,
                    "species": species_by_label[label],
                    "source_dataset": "Xeno-canto_validation",
                    "source_recording_id": str(record["recording_id"]),
                    "start_sample": start_sample,
                    "duration_samples": SAMPLES_PER_WINDOW,
                    "split": "board_test",
                    "annotation_status": "source_label",
                    "mixed_species": "unknown",
                    "leakage_check": "unknown",
                    "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
                    "wav_sha256": sha256_file(destination, hash_cache),
                    "tensor_scope": "xeno_validation",
                    "source_region": "",
                    "source_feature_index": source_index,
                }
            )

    records = {
        (int(record["region"]), int(record["label"])): record
        for record in db_common_records
    }
    for region in REGIONS:
        if any((region, label) not in records for label in range(8)):
            continue
        for slice_offset in range(8):
            for label in range(8):
                record = records[(region, label)]
                source = resolve_raw_path(str(record["path"]))
                source_index = int(record["source_indices"][slice_offset])
                start_sample = slice_offset * SAMPLES_PER_WINDOW
                source_id = Path(canonical_record_path(str(record["path"]))).stem
                sample_id = sanitize_sample_id(
                    f"db3v_common20_r{region}_{source_id}_s{slice_offset:02d}"
                )
                destination = (
                    raw_root / "audio" / species_by_label[label] / f"{sample_id}.wav"
                )
                expected_audio_paths.add(destination.resolve())
                pcm = extract_pcm_window(source, start_sample, destination)
                for feature in FEATURES:
                    expected = np.load(
                        db3v_dir(feature, True) / f"region_{region}_data.npy",
                        mmap_mode="r",
                    )[source_index]
                    validation[feature].append(
                        validate_pcm_feature(pcm, feature, expected)
                    )
                rows.append(
                    {
                        "sample_id": sample_id,
                        "file": package_relative(destination, raw_root),
                        "label": label,
                        "species": species_by_label[label],
                        "source_dataset": "DB3V_common_20shot_heldout",
                        "source_recording_id": f"r{region}:{source_id}",
                        "start_sample": start_sample,
                        "duration_samples": SAMPLES_PER_WINDOW,
                        "split": "board_test",
                        "annotation_status": "source_label",
                        "mixed_species": "unknown",
                        "leakage_check": "unknown",
                        "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
                        "wav_sha256": sha256_file(destination, hash_cache),
                        "tensor_scope": "db3v_common_20shot_heldout",
                        "source_region": region,
                        "source_feature_index": source_index,
                    }
                )
    for start in range(0, len(rows), 8):
        if sorted(int(row["label"]) for row in rows[start : start + 8]) != list(range(8)):
            raise ValueError(f"Raw probe block at row {start} is not label-balanced.")
    audio_root = safe_destination(raw_root / "audio", output_dir)
    for stale_path in audio_root.rglob("*.wav"):
        if stale_path.resolve() not in expected_audio_paths:
            stale_path.unlink()
    write_csv(raw_root / "manifest.csv", rows, MANIFEST_FIELDS)
    write_json(raw_root / "label_map.json", EXPECTED_LABEL_MAP)
    summary = {
        feature: {
            "samples": len(items),
            "max_abs": max(float(item["max_abs"]) for item in items),
            "mae_max": max(float(item["mae"]) for item in items),
            "rmse_max": max(float(item["rmse"]) for item in items),
        }
        for feature, items in validation.items()
    }
    report = {
        "samples": len(rows),
        "blocks_of_eight": len(rows) // 8,
        "wav_specification": {
            "channels": 1,
            "sample_width_bytes": 2,
            "sample_rate_hz": SAMPLE_RATE,
            "frames": SAMPLES_PER_WINDOW,
            "pcm_payload_bytes": SAMPLES_PER_WINDOW * 2,
        },
        "annotation_policy": "source-label",
        "scientific_metrics_valid": False,
        "purpose": (
            "Same-source frontend and end-to-end migration replay; not an "
            "independent manually reviewed benchmark."
        ),
        "feature_reconstruction_against_training_arrays": summary,
    }
    write_json(raw_root / "validation.json", report)
    return report


def collect_chains(
    output_dir: Path,
    artifacts: list[dict[str, Any]],
    hash_cache: dict[Path, str],
) -> list[dict[str, Any]]:
    chains: list[dict[str, Any]] = []
    seen: set[str] = set()
    for family, directory_name in STRICT_ROOTS:
        strict_root = EXPERIMENTS_ROOT / directory_name
        with (strict_root / "summary.csv").open(
            "r", encoding="utf-8", newline=""
        ) as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            chain_id = row["chain_id"]
            if chain_id in seen:
                raise ValueError(f"Duplicate chain ID: {chain_id}")
            seen.add(chain_id)
            model_source = strict_root / "models" / chain_id
            model_destination = output_dir / "models" / chain_id
            tflite_source = model_source / "DS_CNN_Model.int8.tflite"
            metadata_source = model_source / "DS_CNN_Model.int8_metadata.json"
            evaluation_source = model_source / "evaluation.json"
            for source, target in (
                (tflite_source, model_destination / "model.tflite"),
                (metadata_source, model_destination / "metadata.json"),
                (evaluation_source, model_destination / "desktop_evaluation.json"),
            ):
                add_artifact(
                    source,
                    target,
                    "copy",
                    output_dir,
                    artifacts,
                    hash_cache,
                )
            metadata = load_json(metadata_source)
            feature = row["feature"]
            db_scope = (
                "db3v_common_20shot_heldout"
                if family == "db3v_strict_fewshot"
                else "db3v_full"
            )
            chains.append(
                {
                    "chain_id": chain_id,
                    "family": family,
                    "feature": feature,
                    "requested_shots": row.get("requested_shots", "0"),
                    "policy": row.get("policy", "baseline"),
                    "seed": row.get("seed", ""),
                    "activation": metadata["output_activation"],
                    "tflite": package_relative(
                        model_destination / "model.tflite", output_dir
                    ),
                    "tflite_sha256": sha256_file(tflite_source, hash_cache),
                    "metadata": package_relative(
                        model_destination / "metadata.json", output_dir
                    ),
                    "desktop_evaluation": package_relative(
                        model_destination / "desktop_evaluation.json", output_dir
                    ),
                    "xeno_scope": "xeno_validation",
                    "birdset_scope": "birdset_common_20shot_heldout",
                    "db3v_scope": db_scope,
                }
            )
    if len(chains) != 57:
        raise ValueError(f"Expected 57 strict INT8 chains, found {len(chains)}.")
    return chains


def write_runbook(output_dir: Path) -> None:
    text = """# Training-test replay package

This directory replays the exact desktop strict-INT8 evaluation inputs on the
STM32 firmware. It is a migration-equivalence package, not a new independent
generalization benchmark.

## What to compare

1. Run `probe` tensors with `--mode both --tflite ...`. This isolates
   TFLite/LiteRT -> TFLite Micro and float-wire -> native-int8 transport.
2. Run `raw_pcm_probe/manifest.csv` with LED_TEST's `source-label` policy. This
   exercises PCM -> board MFCC/LogMel/PCEN -> TFLM.
3. Run `full` tensors and evaluate their CSV output to reproduce the desktop
   Xeno-canto, BirdSet, and DB3V metric granularities.

The BirdSet held-out set is multi-label and has five slices per clip. Do not
evaluate it with the board raw-audio runner's single-label argmax summary.

The historical strict-INT8 reports used a resized batch of 128 with the default
TFLite runtime. LED_TEST intentionally uses batch 1 with TensorFlow 2.19
`BUILTIN_REF` for LiteRT/TFLM parity. The evaluator therefore treats the
per-sample `reference_raw_int8` saved by LED_TEST as the migration reference and
reports the old batch-128 metric difference separately.

## One tensor sweep

```bash
python3 tools/serial_model_client.py --port /dev/ttyACM0 sweep \
  --input /path/to/board_replay_testset/tensors/probe/xeno_validation/MFCC/data.npy \
  --mode both \
  --tflite /path/to/board_replay_testset/models/zero_shot_mfcc/model.tflite \
  --output /tmp/zero_shot_mfcc_xeno.csv
```

Evaluate the saved board scores from the Model_train checkout:

```bash
python3 src/experiments/evaluate_board_replay.py \
  --package board_replay_testset \
  --chain-id zero_shot_mfcc \
  --tier probe \
  --corpus xeno \
  --predictions /tmp/zero_shot_mfcc_xeno.csv \
  --output /tmp/zero_shot_mfcc_xeno_metrics.json
```

For full DB3V, run one sweep per region and pass
`--predictions 1=region1.csv 2=region2.csv 3=region3.csv`.

## Raw frontend/end-to-end probe

```bash
python3 tools/run_board_benchmark.py validate-testset \
  --testset /path/to/board_replay_testset/raw_pcm_probe \
  --annotation-policy source-label
```

The raw probe retains source labels but has not been manually audited for
single-species audibility, so `scientific_metrics_valid` is intentionally false.

## Automated Linux run

Keep LED_TEST and the unpacked firmware pack in separate directories. The
runner and evaluator are included in this package:

```bash
python3 /srv/board_replay_testset/tools/run_board_replay.py \
  --led-test /srv/LED_TEST \
  --pack /srv/led-firmware-pack \
  --package /srv/board_replay_testset \
  --port /dev/ttyACM0 \
  --tier probe \
  --scope core
```

After the core probe passes, replace `--scope core` with `--scope all`. Use
`--tier full --mode native` only for the final metric reproduction because it
streams every original test tensor for every selected model.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def copy_runtime_tools(output_dir: Path) -> None:
    tools_dir = output_dir / "tools"
    tools_dir.mkdir(parents=True, exist_ok=True)
    for name in ("evaluate_board_replay.py", "run_board_replay.py"):
        shutil.copy2(EXPERIMENTS_ROOT / name, tools_dir / name)
    (tools_dir / "requirements-linux.txt").write_text(
        "numpy\npyserial\ntensorflow-cpu==2.19.0\n",
        encoding="utf-8",
    )


def write_package_hashes(
    output_dir: Path,
    artifacts: list[dict[str, Any]],
    hash_cache: dict[Path, str],
) -> None:
    known = {
        str(row["package_file"]): str(row["sha256"]) for row in artifacts
    }
    roots = [
        output_dir / "models",
        output_dir / "tensors",
        output_dir / "raw_pcm_probe",
        output_dir / "tools",
    ]
    files = [
        output_dir / name
        for name in (
            "README.md",
            "chain_matrix.csv",
            "label_map.json",
            "protocol.json",
            "source_artifacts.csv",
        )
    ]
    for root in roots:
        if root.is_dir():
            files.extend(
                path
                for path in root.rglob("*")
                if path.is_file() and path.name != "led_test_validation.json"
            )
    rows = []
    for path in sorted(set(files), key=lambda value: package_relative(value, output_dir)):
        relative = package_relative(path, output_dir)
        rows.append(
            {
                "file": relative,
                "bytes": path.stat().st_size,
                "sha256": known.get(relative) or sha256_file(path, hash_cache),
            }
        )
    write_csv(
        output_dir / "artifact_hashes.csv",
        rows,
        ("file", "bytes", "sha256"),
    )


def main() -> None:
    arguments = parse_arguments()
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    validate_label_maps()
    identity = validate_cross_feature_identity()
    artifacts: list[dict[str, Any]] = []
    hash_cache: dict[Path, str] = {}
    scopes = materialize_full_tensors(
        output_dir,
        arguments.materialize,
        artifacts,
        hash_cache,
    )
    xeno_records = choose_xeno_probe()
    db_full_records = choose_db3v_probe(False)
    db_common_records = choose_db3v_probe(True)
    bird_clip_indices = choose_birdset_probe()
    probes = write_probe_tensors(
        output_dir,
        xeno_records,
        db_full_records,
        db_common_records,
        bird_clip_indices,
    )
    raw_probe = None
    if not arguments.skip_pcm_probe:
        raw_probe = create_raw_probe(
            output_dir,
            xeno_records,
            db_common_records,
            hash_cache,
        )
    chains = collect_chains(output_dir, artifacts, hash_cache)
    write_csv(
        output_dir / "chain_matrix.csv",
        chains,
        (
            "chain_id",
            "family",
            "feature",
            "requested_shots",
            "policy",
            "seed",
            "activation",
            "tflite",
            "tflite_sha256",
            "metadata",
            "desktop_evaluation",
            "xeno_scope",
            "birdset_scope",
            "db3v_scope",
        ),
    )
    write_csv(
        output_dir / "source_artifacts.csv",
        artifacts,
        (
            "package_file",
            "source_file",
            "bytes",
            "sha256",
            "materialization",
        ),
    )
    protocol = {
        "schema_version": PACKAGE_SCHEMA,
        "purpose": "Exact desktop-to-STM32 training-test migration replay.",
        "model_chains": len(chains),
        "features": list(FEATURES),
        "label_map": EXPECTED_LABEL_MAP,
        "materialization": arguments.materialize,
        "identity_validation": identity,
        "full_scopes": scopes,
        "probe_scopes": probes,
        "raw_pcm_probe": raw_probe,
        "rules": {
            "migration_reference": (
                "TensorFlow 2.19 BUILTIN_REF, batch 1, using the exact model "
                "TFLite and per-sample reference_raw_int8 saved by LED_TEST."
            ),
            "legacy_reference": (
                "Existing strict-INT8 reports used batch 128 with the default "
                "runtime; retain their metric delta as a diagnostic, not an LSB oracle."
            ),
            "full_metrics": (
                "Use complete arrays and the original recording/clip aggregation."
            ),
            "probe_metrics": (
                "Diagnostic only; use output LSB parity as the primary migration check."
            ),
            "raw_metrics": (
                "Source-label diagnostic only; not an independent scientific result."
            ),
            "birdset": (
                "Use multi-hot labels and clip-level averaging; never reduce the "
                "canonical BirdSet test to a single-label raw-audio accuracy."
            ),
        },
    }
    write_json(output_dir / "protocol.json", protocol)
    write_json(output_dir / "label_map.json", EXPECTED_LABEL_MAP)
    write_runbook(output_dir)
    copy_runtime_tools(output_dir)
    write_package_hashes(output_dir, artifacts, hash_cache)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "chains": len(chains),
                "full_scopes": list(scopes),
                "probe_scopes": probes,
                "raw_pcm_samples": raw_probe["samples"] if raw_probe else 0,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
