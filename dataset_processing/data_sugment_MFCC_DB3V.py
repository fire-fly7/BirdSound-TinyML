"""Create independent, region-aware DB3V MFCC evaluation datasets.

Only species present in the current Xeno-canto label map are retained, allowing
the DB3V test corpus to remain an independent subset when training uses fewer
than the original ten species.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
from scipy.fft import dct


INPUT_DIR = Path("row_dataset") / "DB3V" / "extracted" / "data_wav_8s_2"
XENO_LABEL_MAP_PATH = Path("dataset_processing") / "output" / "MFCC_dataset_A" / "label_map.json"
OUTPUT_DIR = Path("dataset_processing") / "output" / "MFCC_dataset_DB3V"
SAMPLE_RATE = 16_000
SAMPLES_PER_SLICE = SAMPLE_RATE
MFCC_COUNT = 13
MFCC_FRAMES = 32


def fixed_mfcc(mfcc: np.ndarray) -> np.ndarray:
    """Return a fixed (32, 13) MFCC frame window."""
    mfcc = mfcc[:MFCC_FRAMES]
    if mfcc.shape[0] < MFCC_FRAMES:
        mfcc = np.pad(mfcc, ((0, MFCC_FRAMES - mfcc.shape[0]), (0, 0)))
    return mfcc.astype(np.float32)


def extract_mfcc_slices(audio: np.ndarray) -> np.ndarray:
    """Extract MFCCs using the exact one-second protocol used for Xeno-canto.

    Mel spectra are calculated in a batch for efficiency.  The dB conversion
    remains per slice because librosa's default ``top_db`` clipping is
    slice-relative in the Xeno-canto processing script.
    """
    slices = len(audio) // SAMPLES_PER_SLICE
    if slices == 0:
        return np.empty((0, MFCC_FRAMES, MFCC_COUNT), dtype=np.float32)

    segments = audio[: slices * SAMPLES_PER_SLICE].reshape(slices, SAMPLES_PER_SLICE)
    mel_spectrograms = librosa.feature.melspectrogram(y=segments, sr=SAMPLE_RATE)
    log_mel_spectrograms = np.stack(
        [librosa.power_to_db(mel_spectrogram) for mel_spectrogram in mel_spectrograms]
    )
    mfccs = dct(log_mel_spectrograms, axis=-2, type=2, norm="ortho")[:, :MFCC_COUNT, :]
    return np.stack([fixed_mfcc(mfcc.T) for mfcc in mfccs])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", type=int, choices=(1, 2, 3))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--xeno-label-map", type=Path, default=XENO_LABEL_MAP_PATH)
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
                audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
                clip_features = extract_mfcc_slices(audio)
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
