"""Create a leakage-resistant eight-class MFCC dataset from Xeno-canto WAVs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import librosa
import numpy as np
from scipy.fft import dct


EXPECTED_SPECIES = (
    "Agelaius_phoeniceus",
    "Cardinalis_cardinalis",
    "Certhia_americana",
    "Corvus_brachyrhynchos",
    "Setophaga_aestiva",
    "Setophaga_ruticilla",
    "Spinus_tristis",
    "Turdus_migratorius",
)
SAMPLE_RATE = 16_000
SAMPLES_PER_SLICE = SAMPLE_RATE
MFCC_COUNT = 13
MFCC_FRAMES = 32
VALIDATION_RATIO = 0.2
RANDOM_SEED = 42
MAX_SLICES_PER_RECORDING = 8


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=Path("row_dataset") / "row_bird_dataset_A"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset_processing") / "output" / "MFCC_dataset_A",
    )
    parser.add_argument("--validation-ratio", type=float, default=VALIDATION_RATIO)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--max-slices-per-recording", type=int, default=MAX_SLICES_PER_RECORDING
    )
    return parser.parse_args()


def load_metadata(species_dir: Path) -> dict[str, dict[str, Any]]:
    """Index crawler metadata by local WAV basename."""
    metadata_path = species_dir / "metadata.jsonl"
    if not metadata_path.exists():
        return {}
    metadata_by_name: dict[str, dict[str, Any]] = {}
    for line in metadata_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            metadata = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid metadata JSON in {metadata_path}: {error}") from error
        local_name = Path(str(metadata.get("local_file") or "")).name
        if local_name:
            metadata_by_name[local_name] = metadata
    return metadata_by_name


def session_key(metadata: dict[str, Any], recording_id: str) -> str:
    """Return a stable session group, never merging all missing-metadata recordings."""
    crawler_key = str(metadata.get("session_key") or "").strip()
    normalized = crawler_key.casefold().replace(" ", "")
    if crawler_key and normalized not in {"unknown|unknown|unknown|", "unknown"}:
        return crawler_key
    country = str(metadata.get("cnt") or "Unknown").strip()
    location = str(metadata.get("loc") or "Unknown").strip()
    date = str(metadata.get("date") or "Unknown").strip()
    latitude = str(metadata.get("lat") or "").strip()
    longitude = str(metadata.get("lng") or "").strip()
    if any(value not in {"", "Unknown"} for value in (country, location, date, latitude, longitude)):
        return f"{country}|{location}|{date}|{latitude},{longitude}"
    return f"recording:{recording_id}"


def load_recordings(input_dir: Path) -> tuple[dict[str, int], list[dict[str, Any]]]:
    """Load exactly the eligible eight species and their crawler metadata."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    available_species = {path.name for path in input_dir.iterdir() if path.is_dir()}
    expected_species = set(EXPECTED_SPECIES)
    missing_species = expected_species - available_species
    unexpected_species = available_species - expected_species
    if missing_species or unexpected_species:
        details: list[str] = []
        if missing_species:
            details.append(f"missing={sorted(missing_species)}")
        if unexpected_species:
            details.append(f"unexpected={sorted(unexpected_species)}")
        raise ValueError(
            "Input must contain exactly the eight selected species; " + ", ".join(details)
        )

    label_map = {species: label for label, species in enumerate(EXPECTED_SPECIES)}
    recordings: list[dict[str, Any]] = []
    for species in EXPECTED_SPECIES:
        species_dir = input_dir / species
        metadata_by_name = load_metadata(species_dir)
        for audio_path in sorted(species_dir.glob("*.wav")):
            metadata = metadata_by_name.get(audio_path.name, {})
            recording_id = str(metadata.get("id") or audio_path.stem.removeprefix("XC"))
            recordings.append(
                {
                    "path": str(audio_path),
                    "species": species,
                    "label": label_map[species],
                    "recording_id": recording_id,
                    "country": str(metadata.get("cnt") or "Unknown"),
                    "session_key": session_key(metadata, recording_id),
                    "metadata": metadata,
                }
            )
    if not recordings:
        raise ValueError(f"No WAV files found under {input_dir}")
    return label_map, recordings


def group_counts(records: list[dict[str, Any]]) -> Counter[int]:
    return Counter(int(recording["label"]) for recording in records)


def split_by_session(
    recordings: list[dict[str, Any]], validation_ratio: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split complete session groups while approximately preserving class counts."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for recording in recordings:
        grouped[str(recording["session_key"])].append(recording)

    total_counts = group_counts(recordings)
    if min(total_counts.values()) < 2:
        raise ValueError("Every class needs at least two source recordings for a session-level split.")
    targets = {
        label: min(total_counts[label] - 1, max(1, round(total_counts[label] * validation_ratio)))
        for label in total_counts
    }
    group_items = list(grouped.items())
    np.random.default_rng(seed).shuffle(group_items)
    validation_groups: set[str] = set()
    validation_counts: Counter[int] = Counter()

    def error_for(counts: Counter[int]) -> int:
        return sum(abs(targets[label] - counts[label]) for label in targets)

    for key, group in group_items:
        candidate_counts = validation_counts + group_counts(group)
        keeps_training_examples = all(
            candidate_counts[label] < total_counts[label] for label in group_counts(group)
        )
        if keeps_training_examples and error_for(candidate_counts) < error_for(validation_counts):
            validation_groups.add(key)
            validation_counts = candidate_counts

    for label, target in targets.items():
        if validation_counts[label] > 0:
            continue
        candidates = [
            (key, group)
            for key, group in group_items
            if key not in validation_groups
            and any(recording["label"] == label for recording in group)
            and all(
                validation_counts[recording_label] + count < total_counts[recording_label]
                for recording_label, count in group_counts(group).items()
            )
        ]
        if not candidates:
            raise ValueError(f"Could not reserve a validation session for class ID {label}.")
        key, group = min(
            candidates,
            key=lambda item: error_for(validation_counts + group_counts(item[1])),
        )
        validation_groups.add(key)
        validation_counts += group_counts(group)

    validation_records = [
        recording for recording in recordings if str(recording["session_key"]) in validation_groups
    ]
    training_records = [
        recording for recording in recordings if str(recording["session_key"]) not in validation_groups
    ]
    training_counts = group_counts(training_records)
    validation_counts = group_counts(validation_records)
    if set(training_counts) != set(total_counts) or set(validation_counts) != set(total_counts):
        raise ValueError("Session split did not preserve every class in both partitions.")
    training_sessions = {str(recording["session_key"]) for recording in training_records}
    validation_sessions = {str(recording["session_key"]) for recording in validation_records}
    if training_sessions & validation_sessions:
        raise ValueError("Session leakage was detected in the recording split.")
    training_ids = {str(recording["recording_id"]) for recording in training_records}
    validation_ids = {str(recording["recording_id"]) for recording in validation_records}
    if training_ids & validation_ids:
        raise ValueError("Recording leakage was detected in the recording split.")
    return training_records, validation_records


def selected_slice_indices(total_slices: int, recording_id: str, seed: int, maximum: int) -> list[int]:
    """Select at most one deterministic slice from each temporal bin of a recording."""
    if total_slices <= maximum:
        return list(range(total_slices))
    digest = hashlib.sha256(f"{seed}:{recording_id}".encode("utf-8")).digest()
    random = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    boundaries = np.floor(np.linspace(0, total_slices, maximum + 1)).astype(int)
    return [
        int(random.integers(boundaries[index], boundaries[index + 1]))
        for index in range(maximum)
    ]


def fixed_mfcc_frames(mfcc: np.ndarray) -> np.ndarray:
    """Return a fixed-length MFCC frame matrix."""
    mfcc = mfcc[:MFCC_FRAMES]
    if mfcc.shape[0] < MFCC_FRAMES:
        mfcc = np.pad(mfcc, ((0, MFCC_FRAMES - mfcc.shape[0]), (0, 0)))
    return mfcc.astype(np.float32)


def mfcc_for_segments(segments: np.ndarray) -> np.ndarray:
    """Extract per-segment MFCCs with the same semantics as the original pipeline.

    Mel spectra are batched for speed, while dB clipping remains per segment so
    librosa's default ``top_db`` behavior stays equivalent to individual calls.
    """
    mel_spectrograms = librosa.feature.melspectrogram(y=segments, sr=SAMPLE_RATE)
    log_mel_spectrograms = np.stack(
        [librosa.power_to_db(mel_spectrogram) for mel_spectrogram in mel_spectrograms]
    )
    mfccs = dct(log_mel_spectrograms, axis=-2, type=2, norm="ortho")[:, :MFCC_COUNT, :]
    return np.stack([fixed_mfcc_frames(mfcc.T) for mfcc in mfccs])


def extract_mfcc_slices(
    recordings: list[dict[str, Any]], max_slices_per_recording: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Extract capped MFCC slices and retain the source recording index for every sample."""
    features: list[np.ndarray] = []
    labels: list[int] = []
    recording_indices: list[int] = []
    retained_records: list[dict[str, Any]] = []
    for recording in recordings:
        audio, _ = librosa.load(recording["path"], sr=SAMPLE_RATE, mono=True)
        total_slices = len(audio) // SAMPLES_PER_SLICE
        slice_indices = selected_slice_indices(
            total_slices,
            str(recording["recording_id"]),
            seed,
            max_slices_per_recording,
        )
        if not slice_indices:
            continue
        recording_index = len(retained_records)
        segments = np.stack(
            [
                audio[
                    slice_index * SAMPLES_PER_SLICE : (slice_index + 1) * SAMPLES_PER_SLICE
                ]
                for slice_index in slice_indices
            ]
        )
        for mfcc in mfcc_for_segments(segments):
            features.append(mfcc)
            labels.append(int(recording["label"]))
            recording_indices.append(recording_index)
        retained_records.append(
            {
                "path": recording["path"],
                "species": recording["species"],
                "label": int(recording["label"]),
                "recording_id": recording["recording_id"],
                "country": recording["country"],
                "session_key": recording["session_key"],
                "selected_slice_indices": slice_indices,
                "retained_slices": len(slice_indices),
            }
        )
    if not features:
        raise ValueError("No complete one-second slices were available for MFCC extraction.")
    return (
        np.stack(features),
        np.asarray(labels, dtype=np.int64),
        np.asarray(recording_indices, dtype=np.int64),
        retained_records,
    )


def recording_balanced_sample_weights(
    labels: np.ndarray, recording_indices: np.ndarray, retained_records: list[dict[str, Any]], num_classes: int
) -> np.ndarray:
    """Give every recording equal total mass and every class equal total training mass."""
    record_labels = np.asarray([record["label"] for record in retained_records], dtype=np.int64)
    record_slice_counts = np.asarray(
        [record["retained_slices"] for record in retained_records], dtype=np.float32
    )
    class_record_counts = np.bincount(record_labels, minlength=num_classes)
    if np.any(class_record_counts == 0):
        raise ValueError("A training class has no retained recordings.")
    class_weights = len(retained_records) / (num_classes * class_record_counts)
    return (class_weights[labels] / record_slice_counts[recording_indices]).astype(np.float32)


def write_dataset(
    output_dir: Path,
    label_map: dict[str, int],
    train_data: np.ndarray,
    train_labels: np.ndarray,
    train_recording_indices: np.ndarray,
    train_sample_weights: np.ndarray,
    validation_data: np.ndarray,
    validation_labels: np.ndarray,
    validation_recording_indices: np.ndarray,
    train_records: list[dict[str, Any]],
    validation_records: list[dict[str, Any]],
    arguments: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "train_data.npy", train_data)
    np.save(output_dir / "train_label.npy", train_labels)
    np.save(output_dir / "train_recording_index.npy", train_recording_indices)
    np.save(output_dir / "train_sample_weight.npy", train_sample_weights)
    np.save(output_dir / "validation_data.npy", validation_data)
    np.save(output_dir / "validation_label.npy", validation_labels)
    np.save(output_dir / "validation_recording_index.npy", validation_recording_indices)
    for stale_path in (output_dir / "test_data.npy", output_dir / "test_label.npy"):
        stale_path.unlink(missing_ok=True)
    (output_dir / "label_map.json").write_text(json.dumps(label_map, indent=2), encoding="utf-8")
    manifest = {
        "classes": list(EXPECTED_SPECIES),
        "sample_rate": SAMPLE_RATE,
        "mfcc_shape": [MFCC_FRAMES, MFCC_COUNT],
        "validation_ratio": arguments.validation_ratio,
        "seed": arguments.seed,
        "max_slices_per_recording": arguments.max_slices_per_recording,
        "train": train_records,
        "validation": validation_records,
    }
    (output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    arguments = parse_arguments()
    if not 0 < arguments.validation_ratio < 1:
        raise ValueError("--validation-ratio must be between zero and one.")
    if arguments.max_slices_per_recording < 1:
        raise ValueError("--max-slices-per-recording must be at least one.")
    label_map, recordings = load_recordings(arguments.input_dir)
    train_recordings, validation_recordings = split_by_session(
        recordings, arguments.validation_ratio, arguments.seed
    )
    train_data, train_labels, train_indices, retained_train = extract_mfcc_slices(
        train_recordings, arguments.max_slices_per_recording, arguments.seed
    )
    validation_data, validation_labels, validation_indices, retained_validation = extract_mfcc_slices(
        validation_recordings, arguments.max_slices_per_recording, arguments.seed
    )
    expected_labels = np.arange(len(label_map))
    if not np.array_equal(np.unique(train_labels), expected_labels):
        raise ValueError("Training slices do not cover every expected class.")
    if not np.array_equal(np.unique(validation_labels), expected_labels):
        raise ValueError("Validation slices do not cover every expected class.")
    train_weights = recording_balanced_sample_weights(
        train_labels, train_indices, retained_train, len(label_map)
    )
    write_dataset(
        arguments.output_dir,
        label_map,
        train_data,
        train_labels,
        train_indices,
        train_weights,
        validation_data,
        validation_labels,
        validation_indices,
        retained_train,
        retained_validation,
        arguments,
    )
    print(f"Training data: {train_data.shape} from {len(retained_train)} recordings.")
    print(f"Validation data: {validation_data.shape} from {len(retained_validation)} recordings.")
    print(f"Classes: {len(label_map)}")
    print(f"Output directory: {arguments.output_dir}")


if __name__ == "__main__":
    main()
