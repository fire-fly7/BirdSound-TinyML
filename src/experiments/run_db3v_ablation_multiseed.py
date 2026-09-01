"""Run and summarize the strict DB3V few-shot fine-tuning ablation.

This driver keeps the DB3V support and held-out partitions fixed, repeats the
support selection/training procedure with multiple random seeds, and reports
sample mean and sample standard deviation across seeds.  DB3V held-out results
are produced only after each model has been trained; they are never read by
``fine_tune_db3v.py`` or used for epoch/policy selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from birdset_test_protocol import (
    CANONICAL_REPORT_NAME,
    canonical_dataset_dir,
    validate_all_features,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = REPOSITORY_ROOT / "src" / "experiments"
DATASETS_DIR = REPOSITORY_ROOT / "src" / "dataset_processing" / "output"
FEATURES = ("MFCC", "LogMel", "PCEN")
SHOTS = (5, 10, 20)
POLICIES = ("head_only", "bn_head", "bn_head_replay", "full")
SEEDS = (42, 123, 2026)
LEARNING_RATES = {
    "head_only": 3e-4,
    "bn_head": 1e-4,
    "bn_head_replay": 1e-4,
    "full": 3e-5,
}
VALIDATION_RECORDINGS_PER_STRATUM = {5: 1, 10: 2, 20: 4}
POLICY_DEFINITIONS = {
    "head_only": "Only the final softmax Dense layer is trainable.",
    "bn_head": "All BatchNormalization layers and the final softmax Dense layer are trainable.",
    "bn_head_replay": (
        "BN+Head with a class-balanced Xeno-canto training replay buffer at a "
        "1:1 replay-to-DB3V-slice ratio."
    ),
    "full": "Every layer is trainable, including all BatchNormalization layers.",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", nargs="+", choices=FEATURES, default=list(FEATURES))
    parser.add_argument("--shots", nargs="+", type=int, choices=SHOTS, default=list(SHOTS))
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=POLICIES,
        default=list(POLICIES),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXPERIMENTS_DIR / "DB3V_fewshot_ablation_multiseed_8class",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--replay-ratio", type=float, default=1.0)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun a seed even when its fine-tuning and DB3V reports already exist.",
    )
    parser.add_argument(
        "--force-evaluation",
        action="store_true",
        help="Regenerate DB3V reports without retraining completed models.",
    )
    parser.add_argument(
        "--force-cross-domain",
        action="store_true",
        help="Regenerate BirdSet reports for the policies selected after summarization.",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Do not train or evaluate; rebuild summaries from existing reports.",
    )
    return parser.parse_args()


def support_dir(feature: str, shots: int) -> Path:
    suffix = "" if shots == 5 else f"_{shots}shot"
    return DATASETS_DIR / f"{feature}_DB3V_external_split{suffix}_8class"


def output_dir(root: Path, feature: str, shots: int, policy: str, seed: int) -> Path:
    return root / feature / f"{shots}shot" / policy / f"seed_{seed}"


def run(command: list[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)


def train_and_evaluate(args: argparse.Namespace) -> None:
    fine_tune_script = EXPERIMENTS_DIR / "fine_tune_db3v.py"
    evaluate_script = EXPERIMENTS_DIR / "evaluate_db3v.py"
    for feature in args.features:
        base_model_dir = EXPERIMENTS_DIR / "Feature_comparison_8class" / feature
        xeno_dir = DATASETS_DIR / f"{feature}_dataset_A_8class"
        common_heldout_dir = support_dir(feature, 20)
        for shots in args.shots:
            db3v_support_dir = support_dir(feature, shots)
            for policy in args.policies:
                for seed in args.seeds:
                    destination = output_dir(
                        args.output_dir,
                        feature,
                        shots,
                        policy,
                        seed,
                    )
                    fine_tune_report = destination / "DS_CNN_Model.fewshot.json"
                    evaluation_report = (
                        destination / "DB3V_common_20shot_heldout_evaluation.json"
                    )
                    if args.force or not fine_tune_report.exists():
                        run(
                            [
                                sys.executable,
                                str(fine_tune_script),
                                "--base-model-dir",
                                str(base_model_dir),
                                "--support-dir",
                                str(db3v_support_dir),
                                "--xeno-dataset-dir",
                                str(xeno_dir),
                                "--output-dir",
                                str(destination),
                                "--policy",
                                policy,
                                "--learning-rate",
                                str(LEARNING_RATES[policy]),
                                "--epochs",
                                str(args.epochs),
                                "--patience",
                                str(args.patience),
                                "--batch-size",
                                str(args.batch_size),
                                "--validation-recordings-per-stratum",
                                str(VALIDATION_RECORDINGS_PER_STRATUM[shots]),
                                "--seed",
                                str(seed),
                                "--replay-ratio",
                                str(args.replay_ratio),
                            ]
                        )
                    else:
                        print(f"Using existing fine-tune report: {fine_tune_report}", flush=True)
                    if args.force or args.force_evaluation or not evaluation_report.exists():
                        run(
                            [
                                sys.executable,
                                str(evaluate_script),
                                "--models",
                                "DS_CNN_Model",
                                "--dataset-dir",
                                str(common_heldout_dir),
                                "--model-dir",
                                str(destination),
                                "--output",
                                str(evaluation_report),
                            ]
                        )
                    else:
                        print(f"Using existing DB3V report: {evaluation_report}", flush=True)


def selected_policy_map(
    aggregate_rows: list[dict[str, Any]],
) -> dict[tuple[str, int], str]:
    return {
        (str(row["feature"]), int(row["requested_shots"])): str(row["policy"])
        for row in aggregate_rows
        if bool(row["selected_by_mean_adaptation_score"])
    }


def evaluate_selected_birdset(
    args: argparse.Namespace,
    run_rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
) -> None:
    """Cross-test the selected DB3V policies without changing model selection."""
    validate_all_features(DATASETS_DIR)
    evaluate_script = EXPERIMENTS_DIR / "evaluate_birdset_ssw.py"
    selected = selected_policy_map(aggregate_rows)
    for row in run_rows:
        feature = str(row["feature"])
        shots = int(row["requested_shots"])
        policy = str(row["policy"])
        if selected.get((feature, shots)) != policy:
            continue
        destination = output_dir(
            args.output_dir,
            feature,
            shots,
            policy,
            int(row["seed"]),
        )
        report = destination / CANONICAL_REPORT_NAME
        if (
            args.force
            or args.force_evaluation
            or args.force_cross_domain
            or not report.exists()
        ):
            run(
                [
                    sys.executable,
                    str(evaluate_script),
                    "--models",
                    "DS_CNN_Model",
                    "--dataset-dir",
                    str(canonical_dataset_dir(DATASETS_DIR, feature)),
                    "--model-dir",
                    str(destination),
                    "--output",
                    str(report),
                ]
            )
        else:
            print(f"Using existing BirdSet cross-test report: {report}", flush=True)


def summarize_selected_cross_domain(
    args: argparse.Namespace,
    run_rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
) -> None:
    selected = selected_policy_map(aggregate_rows)
    cross_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for row in run_rows:
        feature = str(row["feature"])
        shots = int(row["requested_shots"])
        policy = str(row["policy"])
        if selected.get((feature, shots)) != policy:
            continue
        destination = output_dir(
            args.output_dir,
            feature,
            shots,
            policy,
            int(row["seed"]),
        )
        report_path = destination / CANONICAL_REPORT_NAME
        if not report_path.exists():
            missing.append(str(report_path.relative_to(REPOSITORY_ROOT)))
            continue
        evaluation = load_json(report_path)["models"]["DS_CNN_Model"]
        clip = evaluation["clip_level"]
        singleton = evaluation["globally_singleton_clip_level"]
        cross_rows.append(
            {
                "feature": feature,
                "requested_shots": shots,
                "policy": policy,
                "seed": int(row["seed"]),
                "xeno_macro_f1": float(row["final_xeno_macro_f1"]),
                "xeno_macro_f1_retention": float(row["xeno_macro_f1_retention"]),
                "db3v_source_recording_macro_f1": float(row["db3v_macro_f1"]),
                "db3v_source_recording_accuracy": float(row["db3v_accuracy"]),
                "birdset_clips": int(clip["samples"]),
                "birdset_top1_any_target": float(
                    clip["top1_any_target_accuracy"]
                ),
                "birdset_top3_any_target": float(
                    clip["top3_any_target_accuracy"]
                ),
                "birdset_singleton_clips": int(singleton["samples"]),
                "birdset_singleton_accuracy": float(singleton["accuracy"]),
                "birdset_singleton_supported_macro_f1": float(
                    singleton["supported_macro_f1"]
                ),
            }
        )
    if missing:
        print(
            "Warning: selected DB3V chains missing BirdSet cross-test reports:\n  "
            + "\n  ".join(missing),
            file=sys.stderr,
        )
    if not cross_rows:
        return
    write_csv(args.output_dir / "selected_cross_domain_runs.csv", cross_rows)
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in cross_rows:
        groups[(row["feature"], row["requested_shots"], row["policy"])].append(row)
    metrics = (
        "xeno_macro_f1",
        "xeno_macro_f1_retention",
        "db3v_source_recording_macro_f1",
        "db3v_source_recording_accuracy",
        "birdset_top1_any_target",
        "birdset_top3_any_target",
        "birdset_singleton_accuracy",
        "birdset_singleton_supported_macro_f1",
    )
    rows: list[dict[str, Any]] = []
    for (feature, shots, policy), values in sorted(groups.items()):
        summary: dict[str, Any] = {
            "feature": feature,
            "requested_shots": shots,
            "policy": policy,
            "n_seeds": len(values),
            "seeds": "|".join(
                str(item["seed"]) for item in sorted(values, key=lambda x: x["seed"])
            ),
            "birdset_clips": values[0]["birdset_clips"],
            "birdset_singleton_clips": values[0]["birdset_singleton_clips"],
        }
        for metric in metrics:
            metric_mean, metric_std = metric_stats(
                [float(item[metric]) for item in values]
            )
            summary[f"{metric}_mean"] = metric_mean
            summary[f"{metric}_std"] = metric_std
        rows.append(summary)
    write_csv(args.output_dir / "selected_cross_domain_aggregate.csv", rows)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_stats(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("Cannot summarize an empty metric list.")
    return mean(values), stdev(values) if len(values) > 1 else 0.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path}.")
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    run_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for feature in args.features:
        for shots in args.shots:
            for policy in args.policies:
                for seed in args.seeds:
                    destination = output_dir(
                        args.output_dir,
                        feature,
                        shots,
                        policy,
                        seed,
                    )
                    fine_tune_path = destination / "DS_CNN_Model.fewshot.json"
                    evaluation_path = (
                        destination / "DB3V_common_20shot_heldout_evaluation.json"
                    )
                    if not fine_tune_path.exists() or not evaluation_path.exists():
                        missing.append(str(destination.relative_to(REPOSITORY_ROOT)))
                        continue
                    fine_tune = load_json(fine_tune_path)
                    evaluation = load_json(evaluation_path)
                    protocol = fine_tune["protocol"]
                    baseline_xeno = fine_tune["baseline_before_selection"][
                        "xeno_validation"
                    ]["macro_f1"]
                    final_xeno = fine_tune["final_all_support_model"][
                        "xeno_validation"
                    ]["macro_f1"]
                    final_retention = fine_tune["final_all_support_model"][
                        "xeno_macro_f1_retention"
                    ]
                    selected = fine_tune["selected_split_model"]
                    db3v = evaluation["models"]["DS_CNN_Model"]["pooled"][
                        "source_recording_level"
                    ]
                    run_rows.append(
                        {
                            "feature": feature,
                            "requested_shots": shots,
                            "actual_support_recordings": protocol[
                                "support_recordings"
                            ],
                            "actual_support_source_recordings": protocol[
                                "support_source_recordings"
                            ],
                            "policy": policy,
                            "seed": seed,
                            "selected_epoch": protocol["selected_epoch"],
                            "trainable_parameters": protocol[
                                "trainable_parameters"
                            ],
                            "total_parameters": protocol["total_parameters"],
                            "batch_normalization_trainable": protocol[
                                "batch_normalization_trainable"
                            ],
                            "replay_enabled": protocol["replay"]["enabled"],
                            "selection_support_macro_f1": selected[
                                "support_validation"
                            ]["macro_f1"],
                            "selection_xeno_macro_f1": selected["xeno_validation"][
                                "macro_f1"
                            ],
                            "selection_adaptation_score": selected[
                                "adaptation_score"
                            ],
                            "final_xeno_macro_f1": final_xeno,
                            "xeno_macro_f1_change": final_xeno - baseline_xeno,
                            "xeno_macro_f1_retention": final_retention,
                            "common_db3v_heldout_recordings": evaluation["models"][
                                "DS_CNN_Model"
                            ]["pooled"]["recordings"],
                            "common_db3v_heldout_source_recordings": evaluation[
                                "models"
                            ]["DS_CNN_Model"]["pooled"]["source_recordings"],
                            "db3v_accuracy": db3v["accuracy"],
                            "db3v_balanced_accuracy": db3v["balanced_accuracy"],
                            "db3v_macro_f1": db3v["macro_f1"],
                            "db3v_top_3_accuracy": db3v["top_3_accuracy"],
                        }
                    )
    if missing:
        print(
            "Warning: incomplete run directories:\n  " + "\n  ".join(missing),
            file=sys.stderr,
        )
    if not run_rows:
        raise RuntimeError("No complete fine-tuning/evaluation reports were found.")

    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        groups[(row["feature"], row["requested_shots"], row["policy"])].append(row)
    aggregate_rows: list[dict[str, Any]] = []
    metrics = (
        "selected_epoch",
        "selection_support_macro_f1",
        "selection_xeno_macro_f1",
        "selection_adaptation_score",
        "final_xeno_macro_f1",
        "xeno_macro_f1_change",
        "xeno_macro_f1_retention",
        "db3v_accuracy",
        "db3v_balanced_accuracy",
        "db3v_macro_f1",
        "db3v_top_3_accuracy",
    )
    for (feature, shots, policy), rows in sorted(groups.items()):
        aggregate: dict[str, Any] = {
            "feature": feature,
            "requested_shots": shots,
            "actual_support_recordings": rows[0]["actual_support_recordings"],
            "actual_support_source_recordings": rows[0][
                "actual_support_source_recordings"
            ],
            "policy": policy,
            "n_seeds": len(rows),
            "expected_seeds": len(args.seeds),
            "complete": len(rows) == len(args.seeds),
            "seeds": "|".join(str(row["seed"]) for row in sorted(rows, key=lambda x: x["seed"])),
            "trainable_parameters": rows[0]["trainable_parameters"],
            "total_parameters": rows[0]["total_parameters"],
        }
        for metric in metrics:
            metric_mean, metric_std = metric_stats(
                [float(row[metric]) for row in rows]
            )
            aggregate[f"{metric}_mean"] = metric_mean
            aggregate[f"{metric}_std"] = metric_std
        aggregate_rows.append(aggregate)

    selection_groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in aggregate_rows:
        selection_groups[(row["feature"], row["requested_shots"])].append(row)
    for rows in selection_groups.values():
        eligible = [row for row in rows if row["complete"]]
        selected_policy = (
            max(eligible, key=lambda row: row["selection_adaptation_score_mean"])[
                "policy"
            ]
            if eligible
            else None
        )
        for row in rows:
            row["selected_by_mean_adaptation_score"] = (
                row["policy"] == selected_policy
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "runs.csv", run_rows)
    write_csv(args.output_dir / "aggregate.csv", aggregate_rows)
    protocol = {
        "experiment": "Strict DB3V few-shot fine-tuning ablation with multiple seeds",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "features": args.features,
        "requested_shots_per_region_class": args.shots,
        "policies": {policy: POLICY_DEFINITIONS[policy] for policy in args.policies},
        "seeds": args.seeds,
        "random_seed_scope": (
            "Fixed source-grouped external support/held-out recordings; seed changes "
            "the source-grouped internal support train/validation split, replay sampling, batch order, and "
            "TensorFlow stochastic operations."
        ),
        "selection": (
            "Policy/epoch selection uses support validation original-source-recording macro-F1 "
            "multiplied by capped Xeno-canto validation macro-F1 retention."
        ),
        "heldout_access_during_training_or_selection": False,
        "db3v_final_test": (
            "Every run is evaluated after training on the feature-matched common "
            "20-shot source-grouped held-out set. Primary metrics aggregate all "
            "eight-second chunks by original Xeno-canto recording ID and are reported but never used "
            "by the training script or selected_by_mean_adaptation_score."
        ),
        "xeno_forgetting_test": (
            "Final Xeno-canto validation macro-F1, absolute change from the base "
            "model, and retention ratio are recorded for every run."
        ),
        "birdset_cross_domain_test": (
            "After policy selection, every seed of the selected policy for each "
            "feature/shot pair is evaluated on the canonical common BirdSet held-out "
            "partition. BirdSet metrics never participate in policy or epoch selection."
        ),
        "standard_deviation": "Sample standard deviation (statistics.stdev, ddof=1).",
        "replay_ratio": args.replay_ratio,
        "maximum_epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "runs_expected": (
            len(args.features)
            * len(args.shots)
            * len(args.policies)
            * len(args.seeds)
        ),
        "runs_complete": len(run_rows),
    }
    (args.output_dir / "experiment_protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return run_rows, aggregate_rows


def main() -> None:
    args = arguments()
    args.output_dir = args.output_dir.resolve()
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        raise ValueError("Epoch, patience and batch-size values must be positive.")
    if args.replay_ratio <= 0:
        raise ValueError("--replay-ratio must be greater than zero.")
    if not args.summarize_only:
        train_and_evaluate(args)
    run_rows, aggregate_rows = summarize(args)
    if not args.summarize_only:
        evaluate_selected_birdset(args, run_rows, aggregate_rows)
    summarize_selected_cross_domain(args, run_rows, aggregate_rows)
    print(
        f"Saved {len(run_rows)} seed runs and {len(aggregate_rows)} aggregate rows "
        f"to {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
