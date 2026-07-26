"""Feature-aware strict-INT8 conversion and batched TFLite inference helpers."""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import tensorflow as tf


FEATURE_SHAPES: dict[str, tuple[int, int]] = {
    "MFCC": (32, 13),
    "LogMel": (32, 40),
    "PCEN": (32, 40),
}


@dataclass(frozen=True)
class FeatureInputSpec:
    """Post-feature-extraction interface for one model chain."""

    feature: str
    feature_shape: tuple[int, int]
    source_dtype: str = "float32"
    model_dtype: str = "int8"
    channel_dimension: int = 1

    @classmethod
    def for_feature(cls, feature: str) -> "FeatureInputSpec":
        if feature not in FEATURE_SHAPES:
            raise ValueError(f"Unsupported feature interface: {feature}")
        return cls(feature=feature, feature_shape=FEATURE_SHAPES[feature])

    @property
    def tensor_shape(self) -> tuple[int, int, int]:
        return (*self.feature_shape, self.channel_dimension)

    def validate_array(self, array: np.ndarray, source: Path) -> None:
        if array.ndim != 3 or tuple(array.shape[1:]) != self.feature_shape:
            raise ValueError(
                f"{source} has shape {array.shape}; expected "
                f"(samples, {self.feature_shape[0]}, {self.feature_shape[1]})."
            )
        if array.dtype != np.float32:
            raise ValueError(f"{source} must contain float32 features, got {array.dtype}.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "source_shape": ["batch", *self.feature_shape],
            "source_dtype": self.source_dtype,
            "model_tensor_shape": ["batch", *self.tensor_shape],
            "model_dtype": self.model_dtype,
            "channel_dimension": self.channel_dimension,
            "normalization": (
                "No additional normalization after feature extraction. Quantize with "
                "q=clip(round(x/input_scale)+input_zero_point,-128,127)."
            ),
        }


def _even_indices(length: int, count: int) -> np.ndarray:
    if length < 1 or count < 1:
        raise ValueError("Representative sources and sample counts must be positive.")
    return np.linspace(0, length - 1, min(length, count), dtype=np.int64)


def representative_dataset(
    sources: list[Path],
    spec: FeatureInputSpec,
    total_samples: int,
) -> Iterable[list[np.ndarray]]:
    """Yield deterministic, evenly spaced calibration samples from training data."""

    if not sources:
        raise ValueError("At least one representative source is required.")
    base_count, remainder = divmod(total_samples, len(sources))
    for source_index, source in enumerate(sources):
        requested = base_count + (1 if source_index < remainder else 0)
        data = np.load(source, mmap_mode="r")
        spec.validate_array(data, source)
        for index in _even_indices(len(data), requested):
            yield [
                np.asarray(data[index : index + 1], dtype=np.float32)[..., np.newaxis]
            ]


def _dtype_counts(interpreter: tf.lite.Interpreter) -> dict[str, int]:
    counts = Counter(
        np.dtype(detail["dtype"]).name for detail in interpreter.get_tensor_details()
    )
    return dict(sorted(counts.items()))


def validate_strict_int8_model(
    interpreter: tf.lite.Interpreter,
    spec: FeatureInputSpec,
) -> dict[str, Any]:
    """Reject hybrid models and return a serializable integer-interface description."""

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    if len(input_details) != 1 or len(output_details) != 1:
        raise ValueError("The evaluator requires exactly one model input and one output.")
    input_detail = input_details[0]
    output_detail = output_details[0]
    if input_detail["dtype"] != np.int8 or output_detail["dtype"] != np.int8:
        raise ValueError(
            f"Strict INT8 requires int8 I/O, got "
            f"{input_detail['dtype']} -> {output_detail['dtype']}."
        )
    input_signature = tuple(int(item) for item in input_detail["shape_signature"])
    if input_signature[1:] != spec.tensor_shape:
        raise ValueError(
            f"TFLite input signature {input_signature} does not match "
            f"{spec.feature} {spec.tensor_shape}."
        )
    if output_detail["shape_signature"][-1] != 8:
        raise ValueError(f"Expected eight outputs, got {output_detail['shape_signature']}.")
    dtype_counts = _dtype_counts(interpreter)
    floating = {
        name: count for name, count in dtype_counts.items() if name.startswith("float")
    }
    if floating:
        raise ValueError(f"Hybrid model contains floating-point tensors: {floating}")
    operators = sorted(
        {
            item["op_name"]
            for item in interpreter._get_ops_details()  # pylint: disable=protected-access
        }
    )
    forbidden = sorted({"QUANTIZE", "DEQUANTIZE"}.intersection(operators))
    if forbidden:
        raise ValueError(f"Strict INT8 graph contains conversion operators: {forbidden}")
    input_scale, input_zero_point = input_detail["quantization"]
    output_scale, output_zero_point = output_detail["quantization"]
    if input_scale <= 0 or output_scale <= 0:
        raise ValueError("INT8 input and output must have positive quantization scales.")
    return {
        "strict_int8": True,
        "input": {
            "dtype": "int8",
            "shape_signature": list(input_signature),
            "scale": float(input_scale),
            "zero_point": int(input_zero_point),
        },
        "output": {
            "dtype": "int8",
            "shape_signature": [
                int(item) for item in output_detail["shape_signature"]
            ],
            "scale": float(output_scale),
            "zero_point": int(output_zero_point),
        },
        "tensor_dtype_counts": dtype_counts,
        "floating_point_tensor_count": 0,
        "operators": operators,
        "forbidden_conversion_operators": forbidden,
    }


def convert_strict_int8(
    model_path: Path,
    output_path: Path,
    spec: FeatureInputSpec,
    representative_sources: list[Path],
    representative_samples: int = 256,
) -> dict[str, Any]:
    """Convert one H5 model and verify that the saved artifact is integer-only."""

    model = tf.keras.models.load_model(model_path, compile=False)
    if tuple(model.input_shape[1:]) != spec.tensor_shape:
        raise ValueError(
            f"{model_path} expects {model.input_shape[1:]}, not {spec.tensor_shape}."
        )
    if model.output_shape[-1] != 8:
        raise ValueError(f"{model_path} does not have eight outputs.")
    output_activation = model.layers[-1].activation.__name__
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(
        representative_sources, spec, representative_samples
    )
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    started = time.perf_counter()
    model_content = converter.convert()
    conversion_seconds = time.perf_counter() - started
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(model_content)

    interpreter = tf.lite.Interpreter(model_content=model_content, num_threads=1)
    interpreter.allocate_tensors()
    metadata = validate_strict_int8_model(interpreter, spec)
    metadata.update(
        {
            "model_path": str(model_path),
            "tflite_path": str(output_path),
            "tflite_bytes": len(model_content),
            "keras_h5_bytes": model_path.stat().st_size,
            "output_activation": output_activation,
            "representative_samples_requested": representative_samples,
            "representative_sources": [str(path) for path in representative_sources],
            "conversion_seconds": conversion_seconds,
            "tensorflow_version": tf.__version__,
        }
    )
    tf.keras.backend.clear_session()
    return metadata


class StrictInt8Predictor:
    """Batch float feature arrays through an integer-only TFLite interface."""

    def __init__(
        self,
        model_path: Path,
        spec: FeatureInputSpec,
        batch_size: int = 128,
        num_threads: int = 4,
    ) -> None:
        if batch_size < 1 or num_threads < 1:
            raise ValueError("Batch size and thread count must be positive.")
        self.model_path = model_path
        self.spec = spec
        self.batch_size = batch_size
        self.interpreter = tf.lite.Interpreter(
            model_path=str(model_path), num_threads=num_threads
        )
        initial_input = self.interpreter.get_input_details()[0]
        self.interpreter.resize_tensor_input(
            initial_input["index"],
            [batch_size, *spec.tensor_shape],
            strict=False,
        )
        self.interpreter.allocate_tensors()
        self.interface = validate_strict_int8_model(self.interpreter, spec)
        self.input_detail = self.interpreter.get_input_details()[0]
        self.output_detail = self.interpreter.get_output_details()[0]
        self.input_scale, self.input_zero_point = self.input_detail["quantization"]
        self.output_scale, self.output_zero_point = self.output_detail["quantization"]

    def predict(self, features: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        self.spec.validate_array(features, self.model_path)
        outputs = np.empty((len(features), 8), dtype=np.float32)
        clipped_low = 0
        clipped_high = 0
        value_count = 0
        started = time.perf_counter()
        padded = np.full(
            (self.batch_size, *self.spec.tensor_shape),
            fill_value=np.int8(self.input_zero_point),
            dtype=np.int8,
        )
        for start in range(0, len(features), self.batch_size):
            stop = min(start + self.batch_size, len(features))
            current = np.asarray(features[start:stop], dtype=np.float32)[..., np.newaxis]
            scaled = np.rint(current / self.input_scale + self.input_zero_point)
            clipped_low += int(np.count_nonzero(scaled < -128))
            clipped_high += int(np.count_nonzero(scaled > 127))
            value_count += int(scaled.size)
            quantized = np.clip(scaled, -128, 127).astype(np.int8)
            padded.fill(np.int8(self.input_zero_point))
            padded[: stop - start] = quantized
            self.interpreter.set_tensor(self.input_detail["index"], padded)
            self.interpreter.invoke()
            raw_output = self.interpreter.get_tensor(self.output_detail["index"])[
                : stop - start
            ]
            outputs[start:stop] = (
                raw_output.astype(np.float32) - self.output_zero_point
            ) * self.output_scale
        elapsed = time.perf_counter() - started
        return outputs, {
            "samples": int(len(features)),
            "values": value_count,
            "clipped_low": clipped_low,
            "clipped_high": clipped_high,
            "input_saturation_fraction": (
                (clipped_low + clipped_high) / value_count if value_count else 0.0
            ),
            "inference_seconds": elapsed,
            "samples_per_second": len(features) / elapsed if elapsed else None,
        }


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
