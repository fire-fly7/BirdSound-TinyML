"""Create independent, region-aware DB3V spectral evaluation datasets.

Only species present in the current Xeno-canto label map are retained, allowing
the DB3V test corpus to remain an independent subset when training uses fewer
than the original ten species. MFCC, Log-Mel, and PCEN use the same feature
definitions as the Xeno-canto comparison pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data_sugment_MFCC import (
    FEATURE_CHOICES,
    MFCC_COUNT,
    MFCC_FRAMES,
    SAMPLE_RATE,
    features_for_segments,
)


INPUT_DIR = Path("row_dataset") / "DB3V" / "extracted" / "data_wav_8s_2"
XENO_LABEL_MAP_PATH = (
    Path("dataset_processing") / "output" / "MFCC_dataset_A_8class" / "label_map.json"
)
OUTPUT_DIR = Path("dataset_processing") / "output" / "MFCC_dataset_DB3V_8class"
SAMPLES_PER_SLICE = SAMPLE_RATE


def extract_feature_slices(audio: np.ndarray, feature_type: str) -> np.ndarray:
    """Extract features using the exact protocol used for Xeno-canto."""
    slices = len(audio) // SAMPLES_PER_SLICE
    if slices == 0:
        width = MFCC_COUNT if feature_type == "mfcc" else 40
        return np.empty((0, MFCC_FRAMES, width), dtype=np.float32)

    segments = audio[: slices * SAMPLES_PER_SLICE].reshape(slices, SAMPLES_PER_SLICE)
    return features_for_segments(segments, feature_type)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", type=int, choices=(1, 2, 3))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--xeno-label-map", type=Path, default=XENO_LABEL_MAP_PATH)
    parser.add_argument("--feature", choices=FEATURE_CHOICES, default="mfcc")
    arguments = parser.parse_args()
    output_dir = arguments.output_dir
    label_map: dict[str, int] = json.loads(arguments.xeno_label_map.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "label_map.json").write_text(json.dumps(label_map, indent=2), encoding="utf-8")
    feature_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    region_parts: list[np.ndarray] = []
    manifest: list[dict[str, str | int]] = []

    region_dirs = sorted(path for path in INPUT_DIR.iterdir() if path.is_dir())
    if arguments.region is not None:
        region_dirs = [path for path in region_dirs if path.name == str(arguments.region)]
    for region_dir in region_dirs:
        region = int(region_dir.name)
        region_features: list[np.ndarray] = []
        region_labels: list[int] = []
        for species_dir in sorted(path for path in region_dir.iterdir() if path.is_dir()):
            species_key = species_dir.name.replace(" ", "_")
            if species_key not in label_map:
                print(f"Region {region}: skipping untrained DB3V species {species_dir.name}.")
                continue
            label = label_map[species_key]
            for audio_path in sorted(species_dir.glob("*.wav")):
                import librosa

                audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
                clip_features = extract_feature_slices(audio, arguments.feature)
                if len(clip_features) == 0:
                    continue
                region_features.extend(clip_features)
                region_labels.extend([label] * len(clip_features))
                manifest.append(
                    {
                        "path": str(audio_path),
                        "region": region,
                        "species": species_key,
                        "label": label,
                        "slices": len(clip_features),
                        "feature_type": arguments.feature,
                    }
                )

        if not region_features:
            raise ValueError(f"No valid WAV files found in DB3V region {region}")
        region_data = np.stack(region_features)
        region_labels_array = np.asarray(region_labels, dtype=np.int64)
        feature_parts.append(region_data)
        label_parts.append(region_labels_array)
        region_parts.append(np.full(len(region_labels_array), region, dtype=np.int8))
        np.save(output_dir / f"region_{region}_data.npy", region_data)
        np.save(output_dir / f"region_{region}_label.npy", region_labels_array)
        print(f"Region {region}: {region_data.shape}")

    if arguments.region is not None:
        return

    test_data = np.concatenate(feature_parts)
    test_labels = np.concatenate(label_parts)
    test_regions = np.concatenate(region_parts)
    np.save(output_dir / "test_data.npy", test_data)
    np.save(output_dir / "test_label.npy", test_labels)
    np.save(output_dir / "test_region.npy", test_regions)
    (output_dir / "label_map.json").write_text(json.dumps(label_map, indent=2), encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Independent DB3V test data: {test_data.shape}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
