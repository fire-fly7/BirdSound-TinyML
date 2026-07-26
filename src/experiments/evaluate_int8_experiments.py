"""Quantize and evaluate all selected zero-shot and few-shot model chains."""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf

from evaluate_birdset_ssw import (
    multilabel_metrics,
    names_for,
    singleton_metrics,
)
from evaluate_db3v import (
    REGIONS,
    class_names,
    classification_metrics,
    load_label_map,
    load_region,
    recording_probabilities,
)
from int8_inference import (
    FeatureInputSpec,
    StrictInt8Predictor,
    convert_strict_int8,
    write_metadata,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = REPOSITORY_ROOT / "src" / "experiments"
DATA_DIR = REPOSITORY_ROOT / "src" / "dataset_processing" / "output"
OUTPUT_DIR = EXPERIMENTS_DIR / "INT8_quantization_8class"
FEATURES = ("MFCC", "LogMel", "PCEN")
MODEL_NAME = "DS_CNN_Model"


@dataclass(frozen=True)
class Chain:
    chain_id: str
    family: str
    feature: str
    requested_shots: int
    policy: str
    model_dir: Path
    representative_sources: tuple[Path, ...]
    birdset_scope: str
    birdset_dataset_dir: Path
    birdset_fp32_report: Path
    db3v_scope: str
    db3v_dataset_dir: Path
    db3v_fp32_report: Path
    xeno_fp32_report: Path
    xeno_fp32_kind: str

    @property
    def model_path(self) -> Path:
        return self.model_dir / f"{MODEL_NAME}.h5"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--families",
        nargs="+",
        choices=("zero_shot", "db3v_fewshot", "birdset_fewshot"),
        default=("zero_shot", "db3v_fewshot", "birdset_fewshot"),
    )
    parser.add_argument("--chains", nargs="+", help="Optional exact chain IDs.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--representative-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--force-convert", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(resolved)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def db3v_support_dir(feature: str, shots: int) -> Path:
    suffix = "" if shots == 5 else f"_{shots}shot"
    return DATA_DIR / f"{feature}_DB3V_external_split{suffix}_8class"


def build_chains() -> list[Chain]:
    chains: list[Chain] = []
    for feature in FEATURES:
        model_dir = EXPERIMENTS_DIR / "Feature_comparison_8class" / feature
        chains.append(
            Chain(
                chain_id=f"zero_shot_{feature.lower()}",
                family="zero_shot",
                feature=feature,
                requested_shots=0,
                policy="baseline",
                model_dir=model_dir,
                representative_sources=(
                    DATA_DIR / f"{feature}_dataset_A_8class" / "train_data.npy",
                ),
                birdset_scope="heldout_201_recordings",
                birdset_dataset_dir=(
                    DATA_DIR / f"{feature}_BirdSet_external_split_8class" / "test"
                ),
                birdset_fp32_report=(
                    model_dir / "BirdSet_SSW_heldout_evaluation.json"
                ),
                db3v_scope="full_10658_recordings",
                db3v_dataset_dir=DATA_DIR / f"{feature}_dataset_DB3V_8class",
                db3v_fp32_report=model_dir / "DB3V_evaluation.json",
                xeno_fp32_report=model_dir / f"{MODEL_NAME}.validation.json",
                xeno_fp32_kind="validation",
            )
        )

    shot_roots = {
        5: EXPERIMENTS_DIR / "DB3V_fewshot_8class",
        10: EXPERIMENTS_DIR / "DB3V_fewshot_10shot_8class",
        20: EXPERIMENTS_DIR / "DB3V_fewshot_20shot_8class",
    }
    comparison = read_csv(
        EXPERIMENTS_DIR
        / "DB3V_fewshot_comparison_8class"
        / "comparison_summary.csv"
    )
    selected_rows = [
        row for row in comparison if int(row["requested_shots"]) in shot_roots
    ]
    for row in selected_rows:
        feature = row["feature"]
        shots = int(row["requested_shots"])
        policy = row["selected_policy"]
        model_dir = shot_roots[shots] / feature / policy
        chains.append(
            Chain(
                chain_id=f"db3v_{shots}shot_{feature.lower()}_{policy}",
                family="db3v_fewshot",
                feature=feature,
                requested_shots=shots,
                policy=policy,
                model_dir=model_dir,
                representative_sources=(
                    DATA_DIR / f"{feature}_dataset_A_8class" / "train_data.npy",
                    db3v_support_dir(feature, shots) / "support_data.npy",
                ),
                birdset_scope="full_211_recordings",
                birdset_dataset_dir=DATA_DIR
                / f"{feature}_dataset_BirdSet_SSW_8class",
                birdset_fp32_report=model_dir / "BirdSet_SSW_full_evaluation.json",
                db3v_scope="common_20shot_heldout_10197_recordings",
                db3v_dataset_dir=(
                    DATA_DIR / f"{feature}_DB3V_external_split_20shot_8class"
                ),
                db3v_fp32_report=(
                    model_dir / "DB3V_common_20shot_heldout_evaluation.json"
                ),
                xeno_fp32_report=model_dir / f"{MODEL_NAME}.fewshot.json",
                xeno_fp32_kind="fewshot",
            )
        )

    birdset_comparison = read_csv(
        EXPERIMENTS_DIR / "BirdSet_fewshot_8class" / "comparison_summary.csv"
    )
    for row in birdset_comparison:
        feature = row["feature"]
        policy = row["selected_policy"]
        model_dir = EXPERIMENTS_DIR / "BirdSet_fewshot_8class" / feature / policy
        chains.append(
            Chain(
                chain_id=f"birdset_5shot_{feature.lower()}_{policy}",
                family="birdset_fewshot",
                feature=feature,
                requested_shots=5,
                policy=policy,
                model_dir=model_dir,
                representative_sources=(
                    DATA_DIR / f"{feature}_dataset_A_8class" / "train_data.npy",
                    DATA_DIR
                    / f"{feature}_BirdSet_external_split_8class"
                    / "support"
                    / "support_data.npy",
                ),
                birdset_scope="heldout_201_recordings",
                birdset_dataset_dir=(
                    DATA_DIR / f"{feature}_BirdSet_external_split_8class" / "test"
                ),
                birdset_fp32_report=(
                    model_dir / "BirdSet_SSW_heldout_evaluation.json"
                ),
                db3v_scope="full_10658_recordings",
                db3v_dataset_dir=DATA_DIR / f"{feature}_dataset_DB3V_8class",
                db3v_fp32_report=model_dir / "DB3V_full_evaluation.json",
                xeno_fp32_report=(
                    model_dir / f"{MODEL_NAME}.birdset_fewshot.json"
                ),
                xeno_fp32_kind="birdset_fewshot",
            )
        )
    return chains


def aggregate_recordings(
    labels: np.ndarray,
    probabilities: np.ndarray,
    recording_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    recording_count = int(recording_index.max()) + 1
    sums = np.zeros((recording_count, probabilities.shape[1]), dtype=np.float64)
    counts = np.bincount(recording_index, minlength=recording_count)
    recording_labels = np.full(recording_count, -1, dtype=np.int64)
    np.add.at(sums, recording_index, probabilities)
    for index, label in zip(recording_index, labels, strict=True):
        if recording_labels[index] not in {-1, int(label)}:
            raise ValueError("A Xeno recording contains inconsistent labels.")
        recording_labels[index] = int(label)
    if np.any(counts == 0) or np.any(recording_labels < 0):
        raise ValueError("Xeno recording indices are incomplete.")
    return recording_labels, sums / counts[:, np.newaxis]


def evaluate_xeno(
    predictor: StrictInt8Predictor,
    dataset_dir: Path,
    names: list[str],
) -> dict[str, Any]:
    features = np.load(dataset_dir / "validation_data.npy", mmap_mode="r")
    labels = np.load(dataset_dir / "validation_label.npy").astype(np.int64)
    recording_index = np.load(
        dataset_dir / "validation_recording_index.npy"
    ).astype(np.int64)
    probabilities, inference = predictor.predict(features)
    recording_labels, recording_probs = aggregate_recordings(
        labels, probabilities, recording_index
    )
    return {
        "dataset_dir": relative(dataset_dir),
        "inference": inference,
        "slice_level": classification_metrics(labels, probabilities, names),
        "recording_level": classification_metrics(
            recording_labels, recording_probs, names
        ),
    }


def evaluate_birdset(
    predictor: StrictInt8Predictor,
    dataset_dir: Path,
    names: list[str],
) -> dict[str, Any]:
    features = np.load(dataset_dir / "test_data.npy", mmap_mode="r")
    clip_index = np.load(dataset_dir / "test_clip_index.npy").astype(np.int64)
    clip_labels = np.load(dataset_dir / "test_clip_multilabel.npy")
    manifest = load_json(dataset_dir / "manifest.json")
    global_singleton = np.asarray(
        [item["is_globally_singleton"] for item in manifest["clips"]], dtype=bool
    )
    probabilities, inference = predictor.predict(features)
    clip_probabilities = np.zeros(
        (len(clip_labels), len(names)), dtype=np.float64
    )
    clip_counts = np.bincount(clip_index, minlength=len(clip_labels))
    np.add.at(clip_probabilities, clip_index, probabilities)
    clip_probabilities /= clip_counts[:, np.newaxis]
    return {
        "dataset_dir": relative(dataset_dir),
        "inference": inference,
        "slice_level": multilabel_metrics(
            clip_labels[clip_index], probabilities, names
        ),
        "clip_level": multilabel_metrics(
            clip_labels, clip_probabilities, names
        ),
        "globally_singleton_clip_level": singleton_metrics(
            clip_labels,
            clip_probabilities,
            names,
            global_singleton,
        ),
    }


def add_inference_stats(items: list[dict[str, Any]]) -> dict[str, Any]:
    samples = sum(item["samples"] for item in items)
    values = sum(item["values"] for item in items)
    clipped_low = sum(item["clipped_low"] for item in items)
    clipped_high = sum(item["clipped_high"] for item in items)
    elapsed = sum(item["inference_seconds"] for item in items)
    return {
        "samples": samples,
        "values": values,
        "clipped_low": clipped_low,
        "clipped_high": clipped_high,
        "input_saturation_fraction": (
            (clipped_low + clipped_high) / values if values else 0.0
        ),
        "inference_seconds": elapsed,
        "samples_per_second": samples / elapsed if elapsed else None,
    }


def evaluate_db3v(
    predictor: StrictInt8Predictor,
    dataset_dir: Path,
    names: list[str],
    feature_shape: tuple[int, int],
) -> dict[str, Any]:
    regions: dict[str, Any] = {}
    all_labels: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    inference_items: list[dict[str, Any]] = []
    for region in REGIONS:
        features, labels = load_region(
            dataset_dir, region, len(names), feature_shape
        )
        probabilities, inference = predictor.predict(features)
        recording_labels, recording_probs = recording_probabilities(
            labels, probabilities
        )
        regions[str(region)] = {
            "recordings": int(len(recording_labels)),
            "inference": inference,
            "slice_level": classification_metrics(labels, probabilities, names),
            "recording_level": classification_metrics(
                recording_labels, recording_probs, names
            ),
        }
        inference_items.append(inference)
        all_labels.append(labels)
        all_probabilities.append(probabilities)
    pooled_labels = np.concatenate(all_labels)
    pooled_probabilities = np.concatenate(all_probabilities)
    recording_labels, recording_probs = recording_probabilities(
        pooled_labels, pooled_probabilities
    )
    return {
        "dataset_dir": relative(dataset_dir),
        "inference": add_inference_stats(inference_items),
        "regions": regions,
        "pooled": {
            "recordings": int(len(recording_labels)),
            "slice_level": classification_metrics(
                pooled_labels, pooled_probabilities, names
            ),
            "recording_level": classification_metrics(
                recording_labels, recording_probs, names
            ),
        },
    }


def fp32_xeno(chain: Chain) -> dict[str, Any]:
    report = load_json(chain.xeno_fp32_report)
    if chain.xeno_fp32_kind == "validation":
        return report
    return report["final_all_support_model"]["xeno_validation"]


def fp32_birdset(chain: Chain) -> dict[str, float]:
    metrics = load_json(chain.birdset_fp32_report)["models"][MODEL_NAME]
    return {
        "top1_any_target_accuracy": metrics["clip_level"][
            "top1_any_target_accuracy"
        ],
        "top3_any_target_accuracy": metrics["clip_level"][
            "top3_any_target_accuracy"
        ],
        "singleton_supported_macro_f1": metrics[
            "globally_singleton_clip_level"
        ]["supported_macro_f1"],
    }


def fp32_db3v(chain: Chain) -> dict[str, float]:
    metrics = load_json(chain.db3v_fp32_report)["models"][MODEL_NAME]["pooled"][
        "recording_level"
    ]
    return {
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "top3_accuracy": metrics["top_3_accuracy"],
    }


def int8_birdset_summary(result: dict[str, Any]) -> dict[str, float]:
    return {
        "top1_any_target_accuracy": result["clip_level"][
            "top1_any_target_accuracy"
        ],
        "top3_any_target_accuracy": result["clip_level"][
            "top3_any_target_accuracy"
        ],
        "singleton_supported_macro_f1": result[
            "globally_singleton_clip_level"
        ]["supported_macro_f1"],
    }


def int8_db3v_summary(result: dict[str, Any]) -> dict[str, float]:
    metrics = result["pooled"]["recording_level"]
    return {
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "top3_accuracy": metrics["top_3_accuracy"],
    }


def metric_delta(
    fp32: dict[str, float], int8: dict[str, float]
) -> dict[str, float | None]:
    delta: dict[str, float | None] = {}
    for key, value in fp32.items():
        if key not in int8 or not isinstance(value, (int, float)):
            continue
        delta[key] = int8[key] - value
        delta[f"{key}_retention"] = int8[key] / value if value else None
    return delta


def summary_row(report: dict[str, Any]) -> dict[str, Any]:
    chain = report["chain"]
    quantization = report["quantization"]
    fp32 = report["fp32_reference"]
    int8 = report["int8"]
    return {
        "chain_id": chain["chain_id"],
        "family": chain["family"],
        "feature": chain["feature"],
        "requested_shots": chain["requested_shots"],
        "policy": chain["policy"],
        "output_activation": quantization["output_activation"],
        "strict_int8": quantization["strict_int8"],
        "floating_point_tensor_count": quantization[
            "floating_point_tensor_count"
        ],
        "tflite_bytes": quantization["tflite_bytes"],
        "input_scale": quantization["input"]["scale"],
        "input_zero_point": quantization["input"]["zero_point"],
        "output_scale": quantization["output"]["scale"],
        "output_zero_point": quantization["output"]["zero_point"],
        "xeno_fp32_macro_f1": fp32["xeno_validation"]["macro_f1"],
        "xeno_int8_macro_f1": int8["xeno_validation"]["recording_level"][
            "macro_f1"
        ],
        "xeno_macro_f1_delta": report["delta"]["xeno_validation"]["macro_f1"],
        "xeno_macro_f1_retention": report["delta"]["xeno_validation"][
            "macro_f1_retention"
        ],
        "xeno_input_saturation": int8["xeno_validation"]["inference"][
            "input_saturation_fraction"
        ],
        "birdset_scope": chain["birdset_scope"],
        "birdset_fp32_top1": fp32["birdset"]["top1_any_target_accuracy"],
        "birdset_int8_top1": report["int8_summary"]["birdset"][
            "top1_any_target_accuracy"
        ],
        "birdset_top1_delta": report["delta"]["birdset"][
            "top1_any_target_accuracy"
        ],
        "birdset_fp32_top3": fp32["birdset"]["top3_any_target_accuracy"],
        "birdset_int8_top3": report["int8_summary"]["birdset"][
            "top3_any_target_accuracy"
        ],
        "birdset_top3_delta": report["delta"]["birdset"][
            "top3_any_target_accuracy"
        ],
        "birdset_fp32_singleton_macro_f1": fp32["birdset"][
            "singleton_supported_macro_f1"
        ],
        "birdset_int8_singleton_macro_f1": report["int8_summary"]["birdset"][
            "singleton_supported_macro_f1"
        ],
        "birdset_singleton_macro_f1_delta": report["delta"]["birdset"][
            "singleton_supported_macro_f1"
        ],
        "birdset_input_saturation": int8["birdset"]["inference"][
            "input_saturation_fraction"
        ],
        "db3v_scope": chain["db3v_scope"],
        "db3v_fp32_accuracy": fp32["db3v"]["accuracy"],
        "db3v_int8_accuracy": report["int8_summary"]["db3v"]["accuracy"],
        "db3v_accuracy_delta": report["delta"]["db3v"]["accuracy"],
        "db3v_fp32_balanced_accuracy": fp32["db3v"]["balanced_accuracy"],
        "db3v_int8_balanced_accuracy": report["int8_summary"]["db3v"][
            "balanced_accuracy"
        ],
        "db3v_balanced_accuracy_delta": report["delta"]["db3v"][
            "balanced_accuracy"
        ],
        "db3v_fp32_macro_f1": fp32["db3v"]["macro_f1"],
        "db3v_int8_macro_f1": report["int8_summary"]["db3v"]["macro_f1"],
        "db3v_macro_f1_delta": report["delta"]["db3v"]["macro_f1"],
        "db3v_macro_f1_retention": report["delta"]["db3v"][
            "macro_f1_retention"
        ],
        "db3v_fp32_top3": fp32["db3v"]["top3_accuracy"],
        "db3v_int8_top3": report["int8_summary"]["db3v"]["top3_accuracy"],
        "db3v_top3_delta": report["delta"]["db3v"]["top3_accuracy"],
        "db3v_input_saturation": int8["db3v"]["inference"][
            "input_saturation_fraction"
        ],
    }


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_chain(
    chain: Chain,
    output_dir: Path,
    representative_samples: int,
    batch_size: int,
    num_threads: int,
    force_convert: bool,
) -> dict[str, Any]:
    chain_dir = output_dir / "models" / chain.chain_id
    tflite_path = chain_dir / f"{MODEL_NAME}.int8.tflite"
    metadata_path = chain_dir / f"{MODEL_NAME}.int8_metadata.json"
    spec = FeatureInputSpec.for_feature(chain.feature)
    if force_convert or not tflite_path.exists() or not metadata_path.exists():
        metadata = convert_strict_int8(
            chain.model_path,
            tflite_path,
            spec,
            list(chain.representative_sources),
            representative_samples,
        )
        metadata["model_path"] = relative(chain.model_path)
        metadata["tflite_path"] = relative(tflite_path)
        metadata["representative_sources"] = [
            relative(path) for path in chain.representative_sources
        ]
        write_metadata(metadata_path, metadata)
    else:
        metadata = load_json(metadata_path)

    predictor = StrictInt8Predictor(
        tflite_path,
        spec,
        batch_size=batch_size,
        num_threads=num_threads,
    )
    if predictor.interface["floating_point_tensor_count"] != 0:
        raise ValueError(f"{chain.chain_id} is not strict INT8.")
    label_map = load_label_map(chain.model_dir / f"{MODEL_NAME}.labels.json")
    names = class_names(label_map)
    for dataset_dir in (
        DATA_DIR / f"{chain.feature}_dataset_A_8class",
        chain.birdset_dataset_dir,
        chain.db3v_dataset_dir,
    ):
        if load_label_map(dataset_dir / "label_map.json") != label_map:
            raise ValueError(f"Label map mismatch for {dataset_dir}.")

    print(f"  {chain.chain_id}: Xeno validation", flush=True)
    xeno_result = evaluate_xeno(
        predictor,
        DATA_DIR / f"{chain.feature}_dataset_A_8class",
        names,
    )
    print(f"  {chain.chain_id}: BirdSet {chain.birdset_scope}", flush=True)
    birdset_result = evaluate_birdset(
        predictor, chain.birdset_dataset_dir, names
    )
    print(f"  {chain.chain_id}: DB3V {chain.db3v_scope}", flush=True)
    db3v_result = evaluate_db3v(
        predictor,
        chain.db3v_dataset_dir,
        names,
        spec.feature_shape,
    )

    fp32 = {
        "sources": {
            "xeno_validation": relative(chain.xeno_fp32_report),
            "birdset": relative(chain.birdset_fp32_report),
            "db3v": relative(chain.db3v_fp32_report),
        },
        "xeno_validation": fp32_xeno(chain),
        "birdset": fp32_birdset(chain),
        "db3v": fp32_db3v(chain),
    }
    int8_summary = {
        "xeno_validation": {
            key: value
            for key, value in xeno_result["recording_level"].items()
            if isinstance(value, (int, float))
        },
        "birdset": int8_birdset_summary(birdset_result),
        "db3v": int8_db3v_summary(db3v_result),
    }
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "chain": {
            "chain_id": chain.chain_id,
            "family": chain.family,
            "feature": chain.feature,
            "requested_shots": chain.requested_shots,
            "policy": chain.policy,
            "model_path": relative(chain.model_path),
            "birdset_scope": chain.birdset_scope,
            "db3v_scope": chain.db3v_scope,
        },
        "input_interface": spec.as_dict(),
        "quantization": metadata,
        "fp32_reference": fp32,
        "int8_summary": int8_summary,
        "delta": {
            "xeno_validation": metric_delta(
                {"macro_f1": fp32["xeno_validation"]["macro_f1"]},
                {"macro_f1": int8_summary["xeno_validation"]["macro_f1"]},
            ),
            "birdset": metric_delta(fp32["birdset"], int8_summary["birdset"]),
            "db3v": metric_delta(fp32["db3v"], int8_summary["db3v"]),
        },
        "int8": {
            "xeno_validation": xeno_result,
            "birdset": birdset_result,
            "db3v": db3v_result,
        },
    }
    report_path = chain_dir / "evaluation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"  {chain.chain_id}: Xeno F1 "
        f"{fp32['xeno_validation']['macro_f1']:.4f}->"
        f"{int8_summary['xeno_validation']['macro_f1']:.4f}; "
        f"Bird Top1 {fp32['birdset']['top1_any_target_accuracy']:.4f}->"
        f"{int8_summary['birdset']['top1_any_target_accuracy']:.4f}; "
        f"DB3V F1 {fp32['db3v']['macro_f1']:.4f}->"
        f"{int8_summary['db3v']['macro_f1']:.4f}",
        flush=True,
    )
    return report


def main() -> None:
    arguments = parse_arguments()
    if (
        arguments.representative_samples < 1
        or arguments.batch_size < 1
        or arguments.num_threads < 1
    ):
        raise ValueError("Sample, batch, and thread counts must be positive.")
    tf.get_logger().setLevel("ERROR")
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    chains = [
        chain for chain in build_chains() if chain.family in arguments.families
    ]
    if arguments.chains:
        requested = set(arguments.chains)
        chains = [chain for chain in chains if chain.chain_id in requested]
        missing = requested.difference(chain.chain_id for chain in chains)
        if missing:
            raise ValueError(f"Unknown or filtered chain IDs: {sorted(missing)}")
    if not chains:
        raise ValueError("No chains selected.")

    reports: list[dict[str, Any]] = []
    for index, chain in enumerate(chains, start=1):
        print(f"[{index}/{len(chains)}] {chain.chain_id}", flush=True)
        reports.append(
            evaluate_chain(
                chain,
                output_dir,
                arguments.representative_samples,
                arguments.batch_size,
                arguments.num_threads,
                arguments.force_convert,
            )
        )
    rows = [summary_row(report) for report in reports]
    write_summary(output_dir / "summary.csv", rows)
    protocol = {
        "experiment": "Strict INT8 evaluation of selected zero-shot and few-shot chains",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": (
            "Quantize only FP32-selected chains. INT8 held-out results never select "
            "features, policies, shots, epochs, thresholds, or calibration sources."
        ),
        "strict_int8_definition": (
            "TFLITE_BUILTINS_INT8 only, int8 input and output, no floating-point "
            "tensors, and no QUANTIZE/DEQUANTIZE graph operators. Int32 bias and "
            "accumulators are expected integer-kernel behavior."
        ),
        "representative_samples": arguments.representative_samples,
        "calibration_rule": {
            "zero_shot": "Xeno-canto training features only.",
            "db3v_fewshot": (
                "Equal requested sample allocation between Xeno-canto training "
                "features and the matching 5/10/20-shot DB3V support."
            ),
            "birdset_fewshot": (
                "Equal requested sample allocation between Xeno-canto training "
                "features and isolated BirdSet support."
            ),
            "heldout_used_for_calibration": False,
        },
        "batch_size": arguments.batch_size,
        "num_threads": arguments.num_threads,
        "tensorflow_version": tf.__version__,
        "chains": [report["chain"] for report in reports],
    }
    (output_dir / "experiment_protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    interfaces = {
        report["chain"]["chain_id"]: {
            **report["input_interface"],
            "quantization": {
                "input": report["quantization"]["input"],
                "output": report["quantization"]["output"],
            },
        }
        for report in reports
    }
    (output_dir / "input_interfaces.json").write_text(
        json.dumps(interfaces, indent=2), encoding="utf-8"
    )
    print(f"Saved summary: {output_dir / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
