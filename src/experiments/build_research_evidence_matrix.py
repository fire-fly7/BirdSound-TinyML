"""Build a machine-readable checklist for the project's required evidence."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "src" / "experiments"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPERIMENTS / "research_evidence_matrix.json",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def strict_fewshot(name: str) -> dict[str, Any]:
    directory = EXPERIMENTS / name
    runs_path = directory / "runs.csv"
    aggregate_path = directory / "aggregate.csv"
    protocol_path = directory / "experiment_protocol.json"
    runs = load_csv(runs_path)
    aggregate = load_csv(aggregate_path)
    seeds = sorted({int(row["seed"]) for row in runs})
    shots = sorted({int(row["requested_shots"]) for row in runs})
    policies = sorted({row["policy"] for row in runs})
    features = sorted({row["feature"] for row in runs})
    complete = (
        len(runs) == 108
        and len(aggregate) == 36
        and seeds == [42, 123, 2026]
        and shots == [5, 10, 20]
        and len(features) == 3
        and len(policies) == 4
        and all(row.get("complete", "").lower() == "true" for row in aggregate)
        and all(row.get("final_xeno_macro_f1", "") for row in runs)
    )
    return {
        "status": "complete" if complete else "incomplete",
        "runs": len(runs),
        "aggregate_groups": len(aggregate),
        "features": features,
        "shots": shots,
        "policies": policies,
        "seeds": seeds,
        "xeno_forgetting_recorded_for_every_run": all(
            row.get("final_xeno_macro_f1", "")
            and row.get("xeno_macro_f1_change", "")
            and row.get("xeno_macro_f1_retention", "")
            for row in runs
        ),
        "sources": [relative(runs_path), relative(aggregate_path), relative(protocol_path)],
    }


def int8_family(name: str, expected: int) -> dict[str, Any]:
    directory = EXPERIMENTS / name
    summary_path = directory / "summary.csv"
    protocol_path = directory / "experiment_protocol.json"
    rows = load_csv(summary_path)
    paired = all(
        row.get("xeno_fp32_macro_f1", "")
        and row.get("xeno_int8_macro_f1", "")
        and row.get("birdset_fp32_top1", "")
        and row.get("birdset_int8_top1", "")
        and row.get("db3v_fp32_macro_f1", "")
        and row.get("db3v_int8_macro_f1", "")
        for row in rows
    )
    strict = all(
        row.get("strict_int8", "").lower() == "true"
        and int(row.get("floating_point_tensor_count", "-1")) == 0
        for row in rows
    )
    return {
        "status": "complete" if len(rows) == expected and paired and strict else "incomplete",
        "models": len(rows),
        "models_expected": expected,
        "same_model_same_test_fp32_int8_pairs": paired,
        "strict_integer_graphs": strict,
        "sources": [relative(summary_path), relative(protocol_path)],
    }


def main() -> None:
    args = arguments()
    split_path = (
        EXPERIMENTS
        / "DB3V_fewshot_ablation_multiseed_8class"
        / "split_audit.json"
    )
    split_audit = load_json(split_path)
    comparison_path = EXPERIMENTS / "Feature_comparison_8class" / "comparison_summary.csv"
    birdset_path = (
        EXPERIMENTS
        / "Feature_comparison_8class"
        / "birdset_common_20shot_heldout_summary.csv"
    )
    comparison = load_csv(comparison_path)
    birdset = load_csv(birdset_path)
    zero_complete = (
        len(comparison) == 3
        and len(birdset) == 3
        and all(row.get("db3v_source_recording_macro_f1", "") for row in comparison)
        and bool(split_audit.get("passed"))
    )

    deployment_path = EXPERIMENTS / "Board_replay_8class" / "deployment_evidence.json"
    deployment = load_json(deployment_path)
    historical = deployment["historical_complete_run"]
    refresh = deployment["current_refresh_status"]
    embedded = {
        "historical_output_parity": {
            "status": "complete"
            if historical["prediction_mismatches"] == 0
            and historical["maximum_output_error_lsb"] == 0
            else "failed",
            "comparisons": historical[
                "desktop_tflite_vs_tflm_output_comparisons"
            ],
            "model_pack_is_current": False,
        },
        "historical_latency": {
            "status": "complete",
            "model_pack_is_current": False,
        },
        "flash_and_ram": {"status": "complete"},
        "current_feature_parity": refresh["feature_tensor_parity"],
        "current_output_parity_and_latency": refresh[
            "refreshed_model_output_parity_and_latency"
        ],
        "power": refresh["power"],
        "source": relative(deployment_path),
    }

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_evidence": {
            "cross_dataset_generalization_without_adaptation": {
                "status": "complete" if zero_complete else "incomplete",
                "features": [row["feature"] for row in comparison],
                "xeno_validation_recordings": 480,
                "db3v_original_source_recordings": int(
                    comparison[0]["db3v_source_recordings"]
                ),
                "birdset_common_test_rows": len(birdset),
                "db3v_split_audit_passed": bool(split_audit.get("passed")),
                "sources": [
                    relative(comparison_path),
                    relative(birdset_path),
                    relative(split_path),
                ],
            },
            "few_shot_adaptation": {
                "db3v": strict_fewshot(
                    "DB3V_fewshot_ablation_multiseed_8class"
                ),
                "birdset": strict_fewshot(
                    "BirdSet_fewshot_ablation_multiseed_8class"
                ),
            },
            "int8_stability": {
                "zero_shot": int8_family(
                    "ZeroShot_strict_INT8_quantization_8class", 3
                ),
                "db3v_selected_fewshot": int8_family(
                    "DB3V_strict_INT8_quantization_8class", 27
                ),
                "birdset_selected_fewshot": int8_family(
                    "BirdSet_strict_INT8_quantization_8class", 27
                ),
            },
            "embedded_deployment": embedded,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Saved research evidence matrix: {args.output}")


if __name__ == "__main__":
    main()
