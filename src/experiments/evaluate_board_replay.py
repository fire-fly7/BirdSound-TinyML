"""Evaluate STM32 tensor-sweep CSVs with the desktop experiment granularities."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKAGE = REPOSITORY_ROOT / "board_replay_testset"
REGIONS = (1, 2, 3)
CLASS_COUNT = 8
SLICES_PER_DB3V_RECORDING = 8


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--chain-id", required=True)
    parser.add_argument("--tier", choices=("probe", "full"), default="probe")
    parser.add_argument(
        "--corpus",
        choices=("xeno", "birdset", "db3v"),
        required=True,
    )
    parser.add_argument(
        "--predictions",
        nargs="+",
        required=True,
        help=(
            "serial_model_client.py sweep CSV. Full DB3V requires "
            "1=region1.csv 2=region2.csv 3=region3.csv"
        ),
    )
    parser.add_argument("--mode", choices=("f32", "native"), default="native")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_chain(package: Path, chain_id: str) -> dict[str, str]:
    path = package / "chain_matrix.csv"
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    matches = [row for row in rows if row["chain_id"] == chain_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one chain {chain_id!r} in {path}, found {len(matches)}.")
    return matches[0]


def parse_prediction_arguments(
    values: list[str],
    corpus: str,
    tier: str,
) -> dict[int | None, Path]:
    result: dict[int | None, Path] = {}
    for value in values:
        if "=" in value:
            key, raw_path = value.split("=", 1)
            try:
                region: int | None = int(key)
            except ValueError as exc:
                raise ValueError(f"Invalid prediction region {key!r}.") from exc
        else:
            region = None
            raw_path = value
        if region in result:
            raise ValueError(f"Duplicate prediction input for {region!r}.")
        result[region] = Path(raw_path).resolve()
    if corpus == "db3v" and tier == "full":
        if set(result) != set(REGIONS):
            raise ValueError("Full DB3V needs exactly 1=, 2=, and 3= prediction CSVs.")
    elif set(result) != {None}:
        raise ValueError("This replay scope needs exactly one unprefixed prediction CSV.")
    for path in result.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    return result


def parse_json_vector(value: str, field: str, path: Path, row_number: int) -> np.ndarray:
    try:
        vector = np.asarray(json.loads(value), dtype=np.float64)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field} at {path}:{row_number}.") from exc
    if vector.shape != (CLASS_COUNT,) or not np.isfinite(vector).all():
        raise ValueError(f"Invalid {field} vector at {path}:{row_number}: {vector}.")
    return vector


def load_board_scores(
    path: Path,
    expected_samples: int,
    selected_mode: str,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        all_rows = list(csv.DictReader(stream))
    if not all_rows:
        raise ValueError(f"Prediction CSV is empty: {path}")
    modes = sorted({row.get("mode", "") for row in all_rows})
    rows = [row for row in all_rows if row.get("mode") == selected_mode]
    if len(rows) != expected_samples:
        raise ValueError(
            f"{path} has {len(rows)} {selected_mode} rows, expected {expected_samples}."
        )
    by_index: dict[int, dict[str, str]] = {}
    for row_number, row in enumerate(rows, start=2):
        try:
            index = int(row["sample_index"])
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Invalid sample index at {path}:{row_number}.") from exc
        if index in by_index:
            raise ValueError(f"Duplicate sample index {index} in {path}.")
        by_index[index] = row
    if set(by_index) != set(range(expected_samples)):
        missing = sorted(set(range(expected_samples)) - set(by_index))
        extra = sorted(set(by_index) - set(range(expected_samples)))
        raise ValueError(
            f"{path} does not cover 0..{expected_samples - 1}; "
            f"missing={missing[:10]}, extra={extra[:10]}."
        )
    scores = np.stack(
        [
            parse_json_vector(by_index[index]["scores"], "scores", path, index + 2)
            for index in range(expected_samples)
        ]
    )

    checked = 0
    mismatches = 0
    maximum_lsb = 0
    reference_vectors: list[np.ndarray] = []
    for index in range(expected_samples):
        row = by_index[index]
        reference_value = row.get("reference_raw_int8", "")
        raw_value = row.get("raw_output_int8", "")
        if not reference_value:
            continue
        raw = parse_json_vector(raw_value, "raw_output_int8", path, index + 2).astype(
            np.int16
        )
        reference = parse_json_vector(
            reference_value, "reference_raw_int8", path, index + 2
        ).astype(np.int16)
        reference_vectors.append(reference.astype(np.int8))
        checked += 1
        maximum_lsb = max(maximum_lsb, int(np.max(np.abs(raw - reference))))
        mismatches += int(int(np.argmax(raw)) != int(np.argmax(reference)))

    cross_mode: dict[str, Any] | None = None
    if {"f32", "native"}.issubset(modes):
        grouped: dict[tuple[int, str], dict[str, str]] = {}
        for row in all_rows:
            grouped[(int(row["sample_index"]), row["mode"])] = row
        cross_maximum = 0
        cross_mismatches = 0
        for index in range(expected_samples):
            f32 = parse_json_vector(
                grouped[(index, "f32")]["raw_output_int8"],
                "raw_output_int8",
                path,
                index + 2,
            ).astype(np.int16)
            native = parse_json_vector(
                grouped[(index, "native")]["raw_output_int8"],
                "raw_output_int8",
                path,
                index + 2,
            ).astype(np.int16)
            cross_maximum = max(
                cross_maximum, int(np.max(np.abs(f32 - native)))
            )
            cross_mismatches += int(int(np.argmax(f32)) != int(np.argmax(native)))
        cross_mode = {
            "samples": expected_samples,
            "max_lsb_error": cross_maximum,
            "prediction_mismatches": cross_mismatches,
            "passed": cross_maximum == 0 and cross_mismatches == 0,
        }

    parity = {
        "available_modes": modes,
        "selected_mode": selected_mode,
        "reference_checked_samples": checked,
        "reference_max_lsb_error": maximum_lsb if checked else None,
        "reference_prediction_mismatches": mismatches if checked else None,
        "reference_passed": (
            maximum_lsb <= 1 and mismatches == 0 if checked else None
        ),
        "f32_vs_native": cross_mode,
    }
    if checked not in {0, expected_samples}:
        raise ValueError(
            f"{path} has desktop references for only {checked}/{expected_samples} samples."
        )
    reference_raw = np.stack(reference_vectors) if reference_vectors else None
    return scores, reference_raw, parity


def class_names(label_map: dict[str, int]) -> list[str]:
    return [
        name for name, _ in sorted(label_map.items(), key=lambda item: item[1])
    ]


def classification_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    names: list[str],
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = scores.argmax(axis=1)
    confusion = np.zeros((len(names), len(names)), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    true_totals = confusion.sum(axis=1)
    predicted_totals = confusion.sum(axis=0)
    true_positive = np.diag(confusion)
    recall = np.divide(
        true_positive,
        true_totals,
        out=np.zeros(len(names), dtype=np.float64),
        where=true_totals != 0,
    )
    precision = np.divide(
        true_positive,
        predicted_totals,
        out=np.zeros(len(names), dtype=np.float64),
        where=predicted_totals != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(len(names), dtype=np.float64),
        where=(precision + recall) != 0,
    )
    top3 = np.argpartition(scores, -3, axis=1)[:, -3:]
    return {
        "samples": len(labels),
        "accuracy": float(np.mean(predictions == labels)),
        "balanced_accuracy": float(np.mean(recall)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(np.average(f1, weights=true_totals)),
        "top_3_accuracy": float(
            np.mean(np.any(top3 == labels[:, np.newaxis], axis=1))
        ),
        "confusion_matrix": confusion.tolist(),
        "per_class": {
            name: {
                "label": index,
                "support": int(true_totals[index]),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
            }
            for index, name in enumerate(names)
        },
    }


def aggregate_groups(
    labels: np.ndarray,
    scores: np.ndarray,
    group_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    group_index = np.asarray(group_index, dtype=np.int64)
    unique = np.unique(group_index)
    if not np.array_equal(unique, np.arange(len(unique))):
        raise ValueError("Group indices must be contiguous from zero.")
    sums = np.zeros((len(unique), scores.shape[1]), dtype=np.float64)
    counts = np.bincount(group_index, minlength=len(unique))
    group_labels = np.full(len(unique), -1, dtype=np.int64)
    np.add.at(sums, group_index, scores)
    for index, label in zip(group_index, labels, strict=True):
        if group_labels[index] not in {-1, int(label)}:
            raise ValueError(f"Group {index} has inconsistent single labels.")
        group_labels[index] = int(label)
    if np.any(counts == 0) or np.any(group_labels < 0):
        raise ValueError("Group indices contain gaps.")
    return group_labels, sums / counts[:, np.newaxis]


def multilabel_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    names: list[str],
) -> dict[str, Any]:
    predictions = scores.argmax(axis=1)
    top3 = np.argpartition(scores, -3, axis=1)[:, -3:]
    rows = np.arange(len(labels))
    top1_hit = labels[rows, predictions].astype(bool)
    top3_hit = np.any(np.take_along_axis(labels, top3, axis=1), axis=1)
    return {
        "samples": len(labels),
        "top1_any_target_accuracy": float(np.mean(top1_hit)),
        "top3_any_target_accuracy": float(np.mean(top3_hit)),
        "per_class": {
            name: {
                "label": class_id,
                "support": int(labels[:, class_id].astype(bool).sum()),
                "top1_recall": (
                    float(
                        np.mean(
                            predictions[labels[:, class_id].astype(bool)] == class_id
                        )
                    )
                    if labels[:, class_id].astype(bool).any()
                    else None
                ),
            }
            for class_id, name in enumerate(names)
        },
    }


def singleton_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    names: list[str],
    mask: np.ndarray,
) -> dict[str, Any]:
    singleton_labels = labels[mask].argmax(axis=1)
    predictions = scores[mask].argmax(axis=1)
    confusion = np.zeros((len(names), len(names)), dtype=np.int64)
    np.add.at(confusion, (singleton_labels, predictions), 1)
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    true_positive = np.diag(confusion)
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros(len(names), dtype=np.float64),
        where=support != 0,
    )
    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros(len(names), dtype=np.float64),
        where=predicted != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(len(names), dtype=np.float64),
        where=(precision + recall) != 0,
    )
    supported = support > 0
    return {
        "samples": int(mask.sum()),
        "accuracy": (
            float(np.mean(predictions == singleton_labels)) if mask.any() else None
        ),
        "supported_macro_f1": (
            float(np.mean(f1[supported])) if supported.any() else None
        ),
        "supported_classes": [
            names[index] for index in np.flatnonzero(supported)
        ],
        "confusion_matrix": confusion.tolist(),
    }


def evaluate_xeno(
    directory: Path,
    scores: np.ndarray,
    names: list[str],
) -> dict[str, Any]:
    labels = np.load(directory / "labels.npy")
    recording_index = np.load(directory / "recording_index.npy")
    recording_labels, recording_scores = aggregate_groups(
        labels, scores, recording_index
    )
    return {
        "slice_level": classification_metrics(labels, scores, names),
        "recording_level": classification_metrics(
            recording_labels, recording_scores, names
        ),
    }


def evaluate_birdset(
    directory: Path,
    scores: np.ndarray,
    names: list[str],
) -> dict[str, Any]:
    clip_index = np.load(directory / "clip_index.npy")
    clip_labels = np.load(directory / "clip_multilabel.npy")
    manifest = load_json(directory / "manifest.json")
    clip_scores = np.zeros((len(clip_labels), scores.shape[1]), dtype=np.float64)
    counts = np.bincount(clip_index, minlength=len(clip_labels))
    np.add.at(clip_scores, clip_index, scores)
    if np.any(counts == 0):
        raise ValueError("BirdSet clip indices contain gaps.")
    clip_scores /= counts[:, np.newaxis]
    singleton = np.asarray(
        [bool(item["is_globally_singleton"]) for item in manifest["clips"]],
        dtype=bool,
    )
    return {
        "slice_level": multilabel_metrics(clip_labels[clip_index], scores, names),
        "clip_level": multilabel_metrics(clip_labels, clip_scores, names),
        "globally_singleton_clip_level": singleton_metrics(
            clip_labels, clip_scores, names, singleton
        ),
    }


def db3v_region_result(
    labels: np.ndarray,
    scores: np.ndarray,
    recording_index: np.ndarray,
    names: list[str],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    recording_labels, recording_scores = aggregate_groups(
        labels, scores, recording_index
    )
    result = {
        "recordings": len(recording_labels),
        "slice_level": classification_metrics(labels, scores, names),
        "recording_level": classification_metrics(
            recording_labels, recording_scores, names
        ),
    }
    return result, recording_labels, recording_scores


def evaluate_db3v_probe(
    directory: Path,
    scores: np.ndarray,
    names: list[str],
) -> dict[str, Any]:
    labels = np.load(directory / "labels.npy")
    recording_index = np.load(directory / "recording_index.npy")
    regions = np.load(directory / "region.npy")
    region_results: dict[str, Any] = {}
    pooled_recording_labels: list[np.ndarray] = []
    pooled_recording_scores: list[np.ndarray] = []
    for region in REGIONS:
        mask = regions == region
        _, local_recording_index = np.unique(
            recording_index[mask], return_inverse=True
        )
        result, recording_labels, recording_scores = db3v_region_result(
            labels[mask], scores[mask], local_recording_index, names
        )
        region_results[str(region)] = result
        pooled_recording_labels.append(recording_labels)
        pooled_recording_scores.append(recording_scores)
    pooled_labels = np.concatenate(pooled_recording_labels)
    pooled_scores = np.concatenate(pooled_recording_scores)
    return {
        "regions": region_results,
        "pooled": {
            "recordings": len(pooled_labels),
            "slice_level": classification_metrics(labels, scores, names),
            "recording_level": classification_metrics(
                pooled_labels, pooled_scores, names
            ),
        },
    }


def evaluate_db3v_full(
    directory: Path,
    score_parts: dict[int | None, np.ndarray],
    names: list[str],
) -> dict[str, Any]:
    region_results: dict[str, Any] = {}
    all_labels: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    all_recording_labels: list[np.ndarray] = []
    all_recording_scores: list[np.ndarray] = []
    for region in REGIONS:
        labels = np.load(directory / f"region_{region}_label.npy")
        scores = score_parts[region]
        recording_index = np.repeat(
            np.arange(len(labels) // SLICES_PER_DB3V_RECORDING, dtype=np.int64),
            SLICES_PER_DB3V_RECORDING,
        )
        result, recording_labels, recording_scores = db3v_region_result(
            labels, scores, recording_index, names
        )
        region_results[str(region)] = result
        all_labels.append(labels)
        all_scores.append(scores)
        all_recording_labels.append(recording_labels)
        all_recording_scores.append(recording_scores)
    labels = np.concatenate(all_labels)
    scores = np.concatenate(all_scores)
    recording_labels = np.concatenate(all_recording_labels)
    recording_scores = np.concatenate(all_recording_scores)
    return {
        "regions": region_results,
        "pooled": {
            "recordings": len(recording_labels),
            "slice_level": classification_metrics(labels, scores, names),
            "recording_level": classification_metrics(
                recording_labels, recording_scores, names
            ),
        },
    }


def flatten_numeric(value: Any, prefix: str = "") -> dict[str, float]:
    result: dict[str, float] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten_numeric(child, child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.update(flatten_numeric(child, f"{prefix}[{index}]"))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        result[prefix] = float(value)
    return result


def compare_reference(
    actual: dict[str, Any],
    expected: dict[str, Any],
) -> dict[str, Any]:
    actual_flat = flatten_numeric(actual)
    expected_flat = flatten_numeric(expected)
    shared = sorted(set(actual_flat) & set(expected_flat))
    missing = sorted(set(actual_flat) - set(expected_flat))
    deltas = {key: actual_flat[key] - expected_flat[key] for key in shared}
    mismatched = {
        key: delta for key, delta in deltas.items() if abs(delta) > 1.0e-12
    }
    return {
        "numeric_leaves_compared": len(shared),
        "actual_leaves_missing_from_reference": missing,
        "max_abs_delta": max((abs(value) for value in deltas.values()), default=0.0),
        "mismatched_numeric_leaves": mismatched,
        "passed": not missing and not mismatched,
    }


def reference_metrics(
    package: Path,
    chain: dict[str, str],
    corpus: str,
) -> dict[str, Any]:
    report = load_json(package / chain["desktop_evaluation"])
    if corpus == "xeno":
        value = report["int8"]["xeno_validation"]
        return {
            "slice_level": value["slice_level"],
            "recording_level": value["recording_level"],
        }
    if corpus == "birdset":
        value = report["int8"]["birdset"]
        return {
            "slice_level": value["slice_level"],
            "clip_level": value["clip_level"],
            "globally_singleton_clip_level": value[
                "globally_singleton_clip_level"
            ],
        }
    value = report["int8"]["db3v"]
    return {"regions": value["regions"], "pooled": value["pooled"]}


def evaluate_scope(
    corpus: str,
    tier: str,
    directory: Path,
    score_parts: dict[int | None, np.ndarray],
    names: list[str],
) -> dict[str, Any]:
    if corpus == "xeno":
        return evaluate_xeno(directory, score_parts[None], names)
    if corpus == "birdset":
        return evaluate_birdset(directory, score_parts[None], names)
    if tier == "full":
        return evaluate_db3v_full(directory, score_parts, names)
    return evaluate_db3v_probe(directory, score_parts[None], names)


def main() -> None:
    arguments = parse_arguments()
    package = arguments.package.resolve()
    chain = load_chain(package, arguments.chain_id)
    label_map = load_json(package / "label_map.json")
    names = class_names(label_map)
    prediction_paths = parse_prediction_arguments(
        arguments.predictions, arguments.corpus, arguments.tier
    )
    if arguments.corpus == "xeno":
        scope = "xeno_validation"
    elif arguments.corpus == "birdset":
        scope = "birdset_common_20shot_heldout"
    else:
        scope = chain["db3v_scope"]
    directory = (
        package / "tensors" / arguments.tier / scope / chain["feature"]
    )
    if not directory.is_dir():
        raise FileNotFoundError(directory)

    score_parts: dict[int | None, np.ndarray] = {}
    reference_score_parts: dict[int | None, np.ndarray] = {}
    parity_parts: dict[str, Any] = {}
    metadata = load_json(package / chain["metadata"])
    output_scale = float(metadata["output"]["scale"])
    output_zero_point = int(metadata["output"]["zero_point"])
    if arguments.corpus == "db3v" and arguments.tier == "full":
        for region in REGIONS:
            labels = np.load(directory / f"region_{region}_label.npy", mmap_mode="r")
            scores, reference_raw, parity = load_board_scores(
                prediction_paths[region], len(labels), arguments.mode
            )
            score_parts[region] = scores
            if reference_raw is not None:
                reference_score_parts[region] = (
                    reference_raw.astype(np.float32) - output_zero_point
                ) * output_scale
            parity_parts[str(region)] = parity
    else:
        if arguments.corpus == "xeno":
            expected_samples = len(np.load(directory / "labels.npy", mmap_mode="r"))
        elif arguments.corpus == "birdset":
            expected_samples = len(np.load(directory / "data.npy", mmap_mode="r"))
        else:
            expected_samples = len(np.load(directory / "labels.npy", mmap_mode="r"))
        scores, reference_raw, parity = load_board_scores(
            prediction_paths[None], expected_samples, arguments.mode
        )
        score_parts[None] = scores
        if reference_raw is not None:
            reference_score_parts[None] = (
                reference_raw.astype(np.float32) - output_zero_point
            ) * output_scale
        parity_parts["all"] = parity

    metrics = evaluate_scope(
        arguments.corpus,
        arguments.tier,
        directory,
        score_parts,
        names,
    )

    wire_reference_metrics = None
    wire_reference_comparison = None
    if len(reference_score_parts) == len(score_parts):
        wire_reference_metrics = evaluate_scope(
            arguments.corpus,
            arguments.tier,
            directory,
            reference_score_parts,
            names,
        )
        wire_reference_comparison = compare_reference(
            metrics, wire_reference_metrics
        )
    legacy_reference_comparison = None
    if arguments.tier == "full":
        legacy_reference_comparison = compare_reference(
            metrics,
            reference_metrics(package, chain, arguments.corpus),
        )
    output = {
        "chain_id": arguments.chain_id,
        "family": chain["family"],
        "feature": chain["feature"],
        "activation": chain["activation"],
        "tier": arguments.tier,
        "corpus": arguments.corpus,
        "scope": scope,
        "mode": arguments.mode,
        "prediction_files": {
            str(key): str(value) for key, value in prediction_paths.items()
        },
        "parity": parity_parts,
        "metrics": metrics,
        "desktop_builtin_ref_metrics": wire_reference_metrics,
        "desktop_builtin_ref_comparison": wire_reference_comparison,
        "legacy_batch128_default_runtime_comparison": legacy_reference_comparison,
        "interpretation": (
            "Migration equivalence is judged against the per-sample pinned "
            "BUILTIN_REF values embedded in the sweep CSV. The legacy strict-INT8 "
            "report used batch-128 default-runtime inference and is diagnostic only."
        ),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
