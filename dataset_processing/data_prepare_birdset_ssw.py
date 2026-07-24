"""Stream BirdSet SSW soundscapes into a selected spectral feature format.

The source SSW 5-second split is multi-label.  Every retained clip contains at
least one of the project's eight target species.  Five non-overlapping
one-second spectral windows are generated per clip without storing the complete
BirdSet archives locally.
"""

from __future__ import annotations

import argparse
import io
import json
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as parquet

from data_sugment_MFCC import FEATURE_CHOICES, features_for_segments


BASE_URL = "https://huggingface.co/datasets/DBD-research-group/BirdSet/resolve/data/SSW"
METADATA_NAME = "SSW_metadata_test_5s.parquet"
SHARD_TEMPLATE = "SSW_test5s_shard_{shard:04d}.tar.gz"
DEFAULT_RAW_DIR = Path("row_dataset") / "BirdSet_SSW"
DEFAULT_OUTPUT_DIR = (
    Path("dataset_processing") / "output" / "MFCC_dataset_BirdSet_SSW_8class"
)
SAMPLE_RATE = 16_000
SAMPLES_PER_SLICE = SAMPLE_RATE
SLICES_PER_CLIP = 5
MFCC_COUNT = 13
MFCC_FRAMES = 32
SHARDS = (1, 2, 3, 4)

SPECIES_TO_EBIRD = {
    "Agelaius_phoeniceus": "rewbla",
    "Cardinalis_cardinalis": "norcar",
    "Certhia_americana": "brncre",
    "Corvus_brachyrhynchos": "amecro",
    "Setophaga_aestiva": "yelwar",
    "Setophaga_ruticilla": "amered",
    "Spinus_tristis": "amegfi",
    "Turdus_migratorius": "amerob",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--feature", choices=FEATURE_CHOICES, default="mfcc")
    parser.add_argument(
        "--label-map",
        type=Path,
        default=Path("dataset_processing")
        / "output"
        / "MFCC_dataset_A_8class"
        / "label_map.json",
    )
    parser.add_argument("--shards", nargs="+", type=int, choices=SHARDS, default=list(SHARDS))
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Download and audit metadata without streaming the audio archives.",
    )
    return parser.parse_args()


def download(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size:
        return destination
    request = urllib.request.Request(url, headers={"User-Agent": "birdset-tinyml/1.0"})
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    temporary.replace(destination)
    return destination


def load_label_map(path: Path) -> dict[str, int]:
    label_map: dict[str, int] = json.loads(path.read_text(encoding="utf-8"))
    expected = set(SPECIES_TO_EBIRD)
    if set(label_map) != expected:
        raise ValueError(
            f"Label map species do not match the expected eight classes: {path}"
        )
    if sorted(label_map.values()) != list(range(len(label_map))):
        raise ValueError(f"Label IDs must be contiguous: {path}")
    return label_map


def load_metadata(metadata_path: Path, label_map: dict[str, int]) -> tuple[dict[str, Any], dict]:
    columns = [
        "filepath",
        "ebird_code_multilabel",
        "license",
        "source",
        "lat",
        "long",
        "local_time",
    ]
    rows = parquet.read_table(metadata_path, columns=columns).to_pylist()
    code_to_label = {
        SPECIES_TO_EBIRD[species]: label for species, label in label_map.items()
    }
    selected: dict[str, Any] = {}
    positive_counts: Counter[str] = Counter()
    singleton_counts: Counter[str] = Counter()
    for row in rows:
        all_codes = sorted(set(row["ebird_code_multilabel"] or []))
        target_codes = sorted(set(all_codes) & set(code_to_label))
        if not target_codes:
            continue
        labels = sorted(code_to_label[code] for code in target_codes)
        basename = Path(row["filepath"]).name
        selected[basename] = {
            "filepath": basename,
            "target_codes": target_codes,
            "all_codes": all_codes,
            "labels": labels,
            "is_target_singleton": len(target_codes) == 1,
            "is_globally_singleton": len(all_codes) == 1,
            "license": row["license"],
            "source": row["source"],
            "lat": row["lat"],
            "long": row["long"],
            "local_time": str(row["local_time"]),
        }
        positive_counts.update(target_codes)
        if len(all_codes) == 1:
            singleton_counts.update(target_codes)
    audit = {
        "metadata_rows": len(rows),
        "selected_clips": len(selected),
        "positive_clip_counts": dict(sorted(positive_counts.items())),
        "globally_singleton_clip_counts": dict(sorted(singleton_counts.items())),
        "note": (
            "Counts are clip-level and multi-label, so positive counts may sum to more "
            "than selected_clips."
        ),
    }
    return selected, audit


def audio_to_features(audio_bytes: bytes, feature_type: str) -> np.ndarray:
    import librosa

    audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE, mono=True)
    required = SLICES_PER_CLIP * SAMPLES_PER_SLICE
    if len(audio) < required:
        audio = np.pad(audio, (0, required - len(audio)))
    segments = audio[:required].reshape(SLICES_PER_CLIP, SAMPLES_PER_SLICE)
    return features_for_segments(segments, feature_type)


def stream_shard(
    shard: int,
    selected: dict[str, Any],
    features: list[np.ndarray],
    clip_labels: list[np.ndarray],
    clip_indices: list[int],
    manifest: list[dict[str, Any]],
    feature_type: str,
) -> int:
    url = f"{BASE_URL}/{SHARD_TEMPLATE.format(shard=shard)}"
    request = urllib.request.Request(url, headers={"User-Agent": "birdset-tinyml/1.0"})
    retained = 0
    print(f"Streaming shard {shard}: {url}", flush=True)
    with urllib.request.urlopen(request, timeout=120) as response:
        with tarfile.open(fileobj=response, mode="r|gz") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                basename = Path(member.name).name
                metadata = selected.get(basename)
                if metadata is None:
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                clip_features = audio_to_features(extracted.read(), feature_type)
                clip_index = len(manifest)
                features.extend(clip_features)
                clip_labels.append(np.asarray(metadata["labels"], dtype=np.int64))
                clip_indices.extend([clip_index] * len(clip_features))
                manifest.append({**metadata, "shard": shard, "slices": len(clip_features)})
                retained += 1
                if retained % 500 == 0:
                    print(f"  retained {retained} clips", flush=True)
    print(f"Shard {shard}: retained {retained} clips.", flush=True)
    return retained


def write_output(
    output_dir: Path,
    label_map: dict[str, int],
    features: list[np.ndarray],
    clip_labels: list[np.ndarray],
    clip_indices: list[int],
    manifest: list[dict[str, Any]],
    audit: dict,
    shards: list[int],
    feature_type: str,
) -> None:
    if not features:
        raise ValueError("No target BirdSet clips were found in the selected shards.")
    output_dir.mkdir(parents=True, exist_ok=True)
    multilabel = np.zeros((len(manifest), len(label_map)), dtype=np.uint8)
    for index, labels in enumerate(clip_labels):
        multilabel[index, labels] = 1
    np.save(output_dir / "test_data.npy", np.stack(features).astype(np.float32))
    np.save(output_dir / "test_clip_index.npy", np.asarray(clip_indices, dtype=np.int64))
    np.save(output_dir / "test_clip_multilabel.npy", multilabel)
    (output_dir / "label_map.json").write_text(
        json.dumps(label_map, indent=2), encoding="utf-8"
    )
    report = {
        "dataset": "BirdSet SSW test_5s",
        "source": BASE_URL,
        "sample_rate": SAMPLE_RATE,
        "slices_per_clip": SLICES_PER_CLIP,
        "feature_type": feature_type,
        "feature_shape": list(np.asarray(features[0]).shape),
        "shards": shards,
        "audit": audit,
        "retained_clips": len(manifest),
        "retained_slices": len(features),
        "clips": manifest,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    arguments = parse_arguments()
    if arguments.output_dir is None:
        prefix = {"mfcc": "MFCC", "logmel": "LogMel", "pcen": "PCEN"}[arguments.feature]
        arguments.output_dir = (
            Path("dataset_processing") / "output" / f"{prefix}_dataset_BirdSet_SSW_8class"
        )
    label_map = load_label_map(arguments.label_map)
    metadata_path = download(f"{BASE_URL}/{METADATA_NAME}", arguments.raw_dir / METADATA_NAME)
    selected, audit = load_metadata(metadata_path, label_map)
    arguments.raw_dir.mkdir(parents=True, exist_ok=True)
    (arguments.raw_dir / "metadata_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2), flush=True)
    if arguments.metadata_only:
        return

    features: list[np.ndarray] = []
    clip_labels: list[np.ndarray] = []
    clip_indices: list[int] = []
    manifest: list[dict[str, Any]] = []
    for shard in sorted(set(arguments.shards)):
        stream_shard(
            shard,
            selected,
            features,
            clip_labels,
            clip_indices,
            manifest,
            arguments.feature,
        )
    if set(arguments.shards) == set(SHARDS) and len(manifest) != len(selected):
        found = {item["filepath"] for item in manifest}
        missing = sorted(set(selected) - found)
        raise ValueError(
            f"Full-shard extraction retained {len(manifest)} of {len(selected)} "
            f"selected clips; first missing files: {missing[:10]}"
        )
    write_output(
        arguments.output_dir,
        label_map,
        features,
        clip_labels,
        clip_indices,
        manifest,
        audit,
        sorted(set(arguments.shards)),
        arguments.feature,
    )
    print(
        f"BirdSet {arguments.feature} data: {len(features)} slices "
        f"from {len(manifest)} clips."
    )
    print(f"Output directory: {arguments.output_dir}")


if __name__ == "__main__":
    main()
