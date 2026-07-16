#!/usr/bin/env python3
"""Generate a deterministic MFCC reference and compare STM32 CSV output."""

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .acoustic_frontend import (
        CONFIG_PATH,
        config_sha256,
        extract_mfcc,
        load_config,
        load_config_data,
        make_dct_matrix,
        make_mel_filterbank,
        make_periodic_hann,
    )
except ImportError:
    from acoustic_frontend import (
        CONFIG_PATH,
        config_sha256,
        extract_mfcc,
        load_config,
        load_config_data,
        make_dct_matrix,
        make_mel_filterbank,
        make_periodic_hann,
    )


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_REFERENCE_DIRECTORY = SCRIPT_DIRECTORY / "output" / "frontend_reference"


def make_reference_pcm(config):
    sample_index = np.arange(config.clip_sample_count, dtype=np.float64)
    time_seconds = sample_index / config.sample_rate
    chirp_phase = 2.0 * np.pi * (
        220.0 * time_seconds + 0.5 * 880.0 * time_seconds**2
    )
    rng = np.random.default_rng(20260717)
    waveform = (
        0.45 * np.sin(2.0 * np.pi * 440.0 * time_seconds)
        + 0.20 * np.sin(2.0 * np.pi * 997.0 * time_seconds + 0.3)
        + 0.15 * np.sin(chirp_phase)
        + 0.01 * rng.standard_normal(config.clip_sample_count)
    )
    return np.clip(
        np.rint(waveform * 32767.0),
        -32768,
        32767,
    ).astype(np.int16)


def generate_reference(config_path, output_directory):
    config_data = load_config_data(config_path)
    config = load_config(config_path)
    config_hash = config_sha256(config_data)
    window = make_periodic_hann(config)
    mel_filterbank = make_mel_filterbank(config)
    dct_matrix = make_dct_matrix(config)
    pcm = make_reference_pcm(config)
    audio = pcm.astype(np.float32) * np.float32(config.pcm_int16_scale)
    mfcc = extract_mfcc(
        audio,
        config=config,
        window=window,
        mel_filterbank=mel_filterbank,
        dct_matrix=dct_matrix,
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    pcm_path = output_directory / "reference_pcm.csv"
    mfcc_path = output_directory / "reference_mfcc.csv"
    metadata_path = output_directory / "reference_metadata.json"

    np.savetxt(pcm_path, pcm, fmt="%d", delimiter=",")
    np.savetxt(mfcc_path, mfcc, fmt="%.9g", delimiter=",")
    metadata_path.write_text(
        json.dumps(
            {
                "frontend_config_sha256": config_hash,
                "pcm_file": pcm_path.name,
                "pcm_dtype": "int16",
                "pcm_sample_count": int(pcm.size),
                "mfcc_file": mfcc_path.name,
                "mfcc_dtype": "float32",
                "mfcc_shape": list(mfcc.shape),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"reference PCM: {pcm_path}")
    print(f"reference MFCC: {mfcc_path}")
    print(f"frontend config sha256: {config_hash}")
    return mfcc


def load_mcu_mfcc(path, expected_shape):
    try:
        values = np.loadtxt(path, delimiter=",", dtype=np.float32)
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read STM32 MFCC CSV {path}: {error}") from error

    if values.size != int(np.prod(expected_shape)):
        raise ValueError(
            f"STM32 MFCC CSV contains {values.size} values; "
            f"expected {int(np.prod(expected_shape))} for shape {expected_shape}."
        )
    return values.reshape(expected_shape)


def compare_mfcc(reference, actual, absolute_tolerance, relative_tolerance):
    difference = np.abs(reference - actual)
    maximum_index = np.unravel_index(np.argmax(difference), difference.shape)
    maximum_error = float(difference[maximum_index])
    mean_error = float(np.mean(difference))
    passed = bool(
        np.allclose(
            reference,
            actual,
            atol=absolute_tolerance,
            rtol=relative_tolerance,
        )
    )

    print(f"maximum absolute error: {maximum_error:.9g} at {maximum_index}")
    print(f"mean absolute error: {mean_error:.9g}")
    print(
        "largest pair: "
        f"python={reference[maximum_index]:.9g}, "
        f"stm32={actual[maximum_index]:.9g}"
    )
    if not passed:
        raise SystemExit(
            "MFCC consistency check failed "
            f"(atol={absolute_tolerance}, rtol={relative_tolerance})."
        )
    print("MFCC consistency check passed.")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic PCM/Python MFCC files and optionally "
            "compare STM32 MFCC CSV output."
        )
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=DEFAULT_REFERENCE_DIRECTORY,
    )
    parser.add_argument(
        "--mcu-mfcc",
        type=Path,
        help="CSV containing 30 x 13 STM32 MFCC values.",
    )
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-4)
    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.atol < 0 or args.rtol < 0:
        raise SystemExit("--atol and --rtol must be non-negative.")

    reference = generate_reference(
        args.config.resolve(),
        args.reference_dir.resolve(),
    )
    if args.mcu_mfcc is not None:
        actual = load_mcu_mfcc(args.mcu_mfcc.resolve(), reference.shape)
        compare_mfcc(reference, actual, args.atol, args.rtol)


if __name__ == "__main__":
    main()
