"""Export a trained bird-call model to TFLite, SavedModel, and a C header."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPOSITORY_ROOT / "src" / "model_train&test" / "TinyML_model"
LITERT_MODEL_DIR = REPOSITORY_ROOT / "src" / "model_train&test" / "LiteRT_model"
DATASET_DIR = REPOSITORY_ROOT / "dataset_processing" / "output" / "MFCC_dataset_A"


def representative_dataset(data_path: Path):
    data = np.expand_dims(np.load(data_path).astype(np.float32), axis=-1)
    sample_count = min(100, len(data))
    for index in np.linspace(0, len(data) - 1, sample_count, dtype=int):
        yield [data[index : index + 1]]


def validate_model(model: tf.keras.Model, model_path: Path) -> None:
    label_path = model_path.with_suffix(".labels.json")
    if not label_path.exists():
        raise FileNotFoundError(f"Missing label map for conversion: {label_path}")
    label_map = json.loads(label_path.read_text(encoding="utf-8"))
    if tuple(model.input_shape[1:]) != (32, 13, 1):
        raise ValueError(f"Unexpected model input shape: {model.input_shape}")
    if model.output_shape[-1] != len(label_map):
        raise ValueError("Model output classes do not match its label map.")


def convert_to_tflite(
    model: tf.keras.Model,
    output_path: Path,
    quantization: str,
    representative_data_path: Path,
) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if quantization == "int8":
        converter.representative_dataset = lambda: representative_dataset(representative_data_path)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(converter.convert())


def convert_tflite_to_header(tflite_path: Path, output_dir: Path) -> Path:
    data = tflite_path.read_bytes()
    array_name = f"{tflite_path.stem.lower()}_data"
    header_guard = f"{array_name.upper()}_H"
    align_macro = f"{array_name.upper()}_ALIGN"
    output_dir.mkdir(parents=True, exist_ok=True)
    header_path = output_dir / f"{array_name}.h"

    with header_path.open("w", encoding="ascii", newline="\n") as header:
        header.write(f"#ifndef {header_guard}\n#define {header_guard}\n\n")
        header.write("#include <stddef.h>\n#include <stdint.h>\n\n")
        header.write(
            f"#if defined(__GNUC__)\n#define {align_macro} __attribute__((aligned(16)))\n"
            f"#else\n#define {align_macro}\n#endif\n\n"
        )
        header.write(f"static {align_macro} const unsigned char {array_name}[] = {{\n")
        for index, byte in enumerate(data):
            if index % 12 == 0:
                header.write("  ")
            header.write(f"0x{byte:02x}, ")
            if index % 12 == 11:
                header.write("\n")
        if len(data) % 12:
            header.write("\n")
        header.write("};\n\n")
        header.write(f"static const unsigned int {array_name}_len = {len(data)};\n\n")
        header.write(f"#endif  // {header_guard}\n")
    return header_path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--model", type=Path, help="Trained H5 model to convert.")
    source.add_argument("--tflite", type=Path, help="Existing TFLite model to convert only to a C header.")
    parser.add_argument("--tflite-output", type=Path)
    parser.add_argument("--header-output-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--quantization", choices=("dynamic", "int8"), default="dynamic")
    parser.add_argument("--representative-data", type=Path, default=DATASET_DIR / "train_data.npy")
    parser.add_argument("--export-savedmodel", action="store_true")
    parser.add_argument("--savedmodel-output", type=Path)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.tflite:
        tflite_path = arguments.tflite
    else:
        model_path = arguments.model or MODEL_DIR / "BC_ResNet.h5"
        model = tf.keras.models.load_model(model_path, compile=False)
        validate_model(model, model_path)
        if arguments.export_savedmodel:
            saved_model_path = arguments.savedmodel_output or LITERT_MODEL_DIR / model_path.stem
            if saved_model_path.exists():
                raise FileExistsError(f"SavedModel output already exists: {saved_model_path}")
            saved_model_path.parent.mkdir(parents=True, exist_ok=True)
            model.export(saved_model_path)
            print(f"SavedModel saved to: {saved_model_path}")
        tflite_path = arguments.tflite_output or model_path.with_suffix(".tflite")
        convert_to_tflite(model, tflite_path, arguments.quantization, arguments.representative_data)
        print(f"TFLite model saved to: {tflite_path}")

    header_path = convert_tflite_to_header(tflite_path, arguments.header_output_dir)
    print(f"C header saved to: {header_path}")


if __name__ == "__main__":
    main()
