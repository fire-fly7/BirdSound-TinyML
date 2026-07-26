"""Build traceable CSV summaries for the BirdSet few-shot experiment."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


EXPERIMENTS_DIR = Path(__file__).resolve().parent
FEATURES = ("MFCC", "LogMel", "PCEN")
POLICIES = ("head", "last_block", "all")
MODEL_NAME = "DS_CNN_Model"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows generated for {path}.")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def candidate_report(root: Path, feature: str, policy: str) -> dict[str, Any]:
    path = root / feature / policy / f"{MODEL_NAME}.birdset_fewshot.json"
    report = load_json(path)
    if report["heldout_access_during_training"] is not False:
        raise ValueError(f"Held-out leakage flag is not false in {path}.")
    return report


def birdset_metrics(path: Path) -> dict[str, float]:
    metrics = load_json(path)["models"][MODEL_NAME]
    clip = metrics["clip_level"]
    singleton = metrics["globally_singleton_clip_level"]
    return {
        "top1": clip["top1_any_target_accuracy"],
        "top3": clip["top3_any_target_accuracy"],
        "singleton_macro_f1": singleton["supported_macro_f1"],
    }


def db3v_metrics(path: Path) -> dict[str, float]:
    metrics = load_json(path)["models"][MODEL_NAME]["pooled"]["recording_level"]
    return {
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "top3": metrics["top_3_accuracy"],
    }


def main() -> None:
    root = EXPERIMENTS_DIR / "BirdSet_fewshot_8class"
    reports: dict[tuple[str, str], dict[str, Any]] = {}
    selected: dict[str, str] = {}

    for feature in FEATURES:
        for policy in POLICIES:
            reports[(feature, policy)] = candidate_report(root, feature, policy)
        selected[feature] = max(
            POLICIES,
            key=lambda policy: reports[(feature, policy)]["selected_split_model"][
                "adaptation_score"
            ],
        )

    selection_rows: list[dict[str, Any]] = []
    for feature in FEATURES:
        for policy in POLICIES:
            report = reports[(feature, policy)]
            protocol = report["protocol"]
            split = report["selected_split_model"]
            support = split["birdset_support_validation"]
            final = report["final_all_support_model"]
            selection_rows.append(
                {
                    "feature": feature,
                    "policy": policy,
                    "learning_rate": protocol["learning_rate"],
                    "selected_epoch": protocol["selected_epoch"],
                    "validation_clips": support["clips"],
                    "validation_supported_classes": len(
                        support["supported_classes"]
                    ),
                    "validation_top1_any_target": support[
                        "top_1_any_target_accuracy"
                    ],
                    "validation_top3_any_target": support[
                        "top_3_any_target_accuracy"
                    ],
                    "validation_threshold_supported_macro_f1": support[
                        "threshold_supported_macro_f1"
                    ],
                    "selection_xeno_macro_f1": split["xeno_validation"][
                        "macro_f1"
                    ],
                    "selection_xeno_retention": split[
                        "xeno_macro_f1_retention"
                    ],
                    "adaptation_score": split["adaptation_score"],
                    "final_xeno_macro_f1": final["xeno_validation"]["macro_f1"],
                    "final_xeno_retention": final["xeno_macro_f1_retention"],
                    "selected": policy == selected[feature],
                }
            )
    write_csv(root / "selection_summary.csv", selection_rows)

    comparison_rows: list[dict[str, Any]] = []
    baseline_root = EXPERIMENTS_DIR / "Feature_comparison_8class"
    for feature in FEATURES:
        policy = selected[feature]
        report = reports[(feature, policy)]
        protocol = report["protocol"]
        final = report["final_all_support_model"]
        baseline_birdset = birdset_metrics(
            baseline_root / feature / "BirdSet_SSW_heldout_evaluation.json"
        )
        adapted_birdset = birdset_metrics(
            root / feature / policy / "BirdSet_SSW_heldout_evaluation.json"
        )
        baseline_db3v = db3v_metrics(
            baseline_root / feature / "DB3V_evaluation.json"
        )
        adapted_db3v = db3v_metrics(
            root / feature / policy / "DB3V_full_evaluation.json"
        )
        baseline_xeno = report["baseline_before_selection"]["xeno_validation"][
            "macro_f1"
        ]
        adapted_xeno = final["xeno_validation"]["macro_f1"]
        comparison_rows.append(
            {
                "feature": feature,
                "selected_policy": policy,
                "selected_epoch": protocol["selected_epoch"],
                "birdset_baseline_top1": baseline_birdset["top1"],
                "birdset_adapted_top1": adapted_birdset["top1"],
                "birdset_top1_delta": (
                    adapted_birdset["top1"] - baseline_birdset["top1"]
                ),
                "birdset_baseline_top3": baseline_birdset["top3"],
                "birdset_adapted_top3": adapted_birdset["top3"],
                "birdset_top3_delta": (
                    adapted_birdset["top3"] - baseline_birdset["top3"]
                ),
                "birdset_baseline_singleton_macro_f1": baseline_birdset[
                    "singleton_macro_f1"
                ],
                "birdset_adapted_singleton_macro_f1": adapted_birdset[
                    "singleton_macro_f1"
                ],
                "birdset_singleton_macro_f1_delta": (
                    adapted_birdset["singleton_macro_f1"]
                    - baseline_birdset["singleton_macro_f1"]
                ),
                "xeno_baseline_macro_f1": baseline_xeno,
                "xeno_adapted_macro_f1": adapted_xeno,
                "xeno_macro_f1_delta": adapted_xeno - baseline_xeno,
                "xeno_macro_f1_retention": final["xeno_macro_f1_retention"],
                "db3v_baseline_accuracy": baseline_db3v["accuracy"],
                "db3v_adapted_accuracy": adapted_db3v["accuracy"],
                "db3v_baseline_balanced_accuracy": baseline_db3v[
                    "balanced_accuracy"
                ],
                "db3v_adapted_balanced_accuracy": adapted_db3v[
                    "balanced_accuracy"
                ],
                "db3v_baseline_macro_f1": baseline_db3v["macro_f1"],
                "db3v_adapted_macro_f1": adapted_db3v["macro_f1"],
                "db3v_macro_f1_delta": (
                    adapted_db3v["macro_f1"] - baseline_db3v["macro_f1"]
                ),
                "db3v_baseline_top3": baseline_db3v["top3"],
                "db3v_adapted_top3": adapted_db3v["top3"],
            }
        )
    write_csv(root / "comparison_summary.csv", comparison_rows)
    print(f"Selected policies: {selected}")
    print(f"Saved {root / 'selection_summary.csv'}")
    print(f"Saved {root / 'comparison_summary.csv'}")


if __name__ == "__main__":
    main()
