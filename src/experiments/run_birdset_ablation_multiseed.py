"""Run and summarize strict grouped BirdSet few-shot adaptation experiments.

The external support partitions remain fixed across seeds. Each seed changes
only the internal support train/validation grouping, replay sampling, batch
order, and TensorFlow stochastic operations. The common BirdSet held-out set
and full DB3V set are evaluated only after training and never participate in
epoch or policy selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
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
VALIDATION_RECORDINGS = {5: 3, 10: 3, 20: 3}
POLICY_DEFINITIONS = {
    "head_only": "Only the final sigmoid Dense layer is trainable.",
    "bn_head": (
        "All BatchNormalization layers and the final sigmoid Dense layer are "
        "trainable."
    ),
    "bn_head_replay": (
        "BN+Head with a class-balanced Xeno-canto training replay buffer at a "
        "1:1 replay-to-BirdSet-slice ratio."
    ),
    "full": "Every layer is trainable, including all BatchNormalization layers.",
}
RECORDING_PATTERN = re.compile(r"^(.*)_\d+_\d+\.ogg$")


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
        default=EXPERIMENTS_DIR / "BirdSet_fewshot_ablation_multiseed_8class",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--replay-ratio", type=float, default=1.0)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun training even when a complete fine-tune report exists.",
    )
    parser.add_argument(
        "--force-evaluation",
        action="store_true",
        help="Regenerate BirdSet and DB3V reports without retraining.",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Rebuild summaries from existing reports without training.",
    )
    return parser.parse_args()


def split_root(feature: str, shots: int) -> Path:
    suffix = "" if shots == 5 else f"_{shots}shot"
    return DATASETS_DIR / f"{feature}_BirdSet_external_split{suffix}_8class"


def output_dir(root: Path, feature: str, shots: int, policy: str, seed: int) -> Path:
    return root / feature / f"{shots}shot" / policy / f"seed_{seed}"


def run(command: list[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)


def train_and_evaluate(args: argparse.Namespace) -> None:
    validate_all_features(DATASETS_DIR)
    fine_tune_script = EXPERIMENTS_DIR / "fine_tune_birdset.py"
    evaluate_birdset_script = EXPERIMENTS_DIR / "evaluate_birdset_ssw.py"
    evaluate_db3v_script = EXPERIMENTS_DIR / "evaluate_db3v.py"
    for feature in args.features:
        base_model_dir = EXPERIMENTS_DIR / "Feature_comparison_8class" / feature
        xeno_dir = DATASETS_DIR / f"{feature}_dataset_A_8class"
        common_birdset_heldout = canonical_dataset_dir(DATASETS_DIR, feature)
        full_db3v = DATASETS_DIR / f"{feature}_dataset_DB3V_8class"
        for shots in args.shots:
            support = split_root(feature, shots) / "support"
            for policy in args.policies:
                for seed in args.seeds:
                    validation_variant = args.seeds.index(seed)
                    destination = output_dir(
                        args.output_dir,
                        feature,
                        shots,
                        policy,
                        seed,
                    )
                    fine_tune_report = (
                        destination / "DS_CNN_Model.birdset_fewshot.json"
                    )
                    birdset_report = destination / CANONICAL_REPORT_NAME
                    db3v_report = destination / "DB3V_full_evaluation.json"
                    if args.force or not fine_tune_report.exists():
                        run(
                            [
                                sys.executable,
                                str(fine_tune_script),
                                "--base-model-dir",
                                str(base_model_dir),
                                "--support-dir",
                                str(support),
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
                                "--validation-recordings",
                                str(VALIDATION_RECORDINGS[shots]),
                                "--validation-variant",
                                str(validation_variant),
                                "--seed",
                                str(seed),
                                "--replay-ratio",
                                str(args.replay_ratio),
                            ]
                        )
                    else:
                        print(
                            f"Using existing fine-tune report: {fine_tune_report}",
                            flush=True,
                        )
                    if (
                        args.force
                        or args.force_evaluation
                        or not birdset_report.exists()
                    ):
                        run(
                            [
                                sys.executable,
                                str(evaluate_birdset_script),
                                "--models",
                                "DS_CNN_Model",
                                "--dataset-dir",
                                str(common_birdset_heldout),
                                "--model-dir",
                                str(destination),
                                "--output",
                                str(birdset_report),
                            ]
                        )
                    else:
                        print(
                            f"Using existing BirdSet report: {birdset_report}",
                            flush=True,
                        )
                    if (
                        args.force
                        or args.force_evaluation
                        or not db3v_report.exists()
                    ):
                        run(
                            [
                                sys.executable,
                                str(evaluate_db3v_script),
                                "--models",
                                "DS_CNN_Model",
                                "--dataset-dir",
                                str(full_db3v),
                                "--model-dir",
                                str(destination),
                                "--output",
                                str(db3v_report),
                            ]
                        )
                    else:
                        print(
                            f"Using existing DB3V report: {db3v_report}",
                            flush=True,
                        )


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def recording_name(filepath: str) -> str:
    match = RECORDING_PATTERN.match(filepath)
    if match is None:
        raise ValueError(f"Unexpected BirdSet clip filename: {filepath}")
    return match.group(1)


def partition_identity(feature: str, shots: int, partition: str) -> list[tuple]:
    manifest = load_json(split_root(feature, shots) / partition / "manifest.json")
    return [
        (
            clip["filepath"],
            tuple(int(value) for value in clip["labels"]),
            int(clip["slices"]),
        )
        for clip in manifest["clips"]
    ]


def verify_split_integrity(
    features: list[str],
    shots_values: list[int],
) -> dict[str, Any]:
    reference_feature = features[0]
    support_sets: dict[int, set[tuple]] = {}
    leakage_by_shot: dict[str, int] = {}
    for shots in shots_values:
        reference_support = partition_identity(reference_feature, shots, "support")
        reference_test = partition_identity(reference_feature, shots, "test")
        for feature in features[1:]:
            if partition_identity(feature, shots, "support") != reference_support:
                raise ValueError(
                    f"BirdSet {shots}-shot support identities differ between "
                    f"{reference_feature} and {feature}."
                )
            if partition_identity(feature, shots, "test") != reference_test:
                raise ValueError(
                    f"BirdSet {shots}-shot test identities differ between "
                    f"{reference_feature} and {feature}."
                )
        support_recordings = {
            recording_name(str(item[0])) for item in reference_support
        }
        test_recordings = {recording_name(str(item[0])) for item in reference_test}
        overlap = support_recordings & test_recordings
        leakage_by_shot[str(shots)] = len(overlap)
        if overlap:
            raise ValueError(
                f"BirdSet {shots}-shot support/test recording leakage: "
                f"{sorted(overlap)}"
            )
        support_sets[shots] = set(reference_support)
    ordered_shots = sorted(shots_values)
    for smaller, larger in zip(ordered_shots, ordered_shots[1:]):
        if not support_sets[smaller].issubset(support_sets[larger]):
            raise ValueError(
                f"BirdSet grouped support is not nested: {smaller}-shot is not "
                f"a subset of {larger}-shot."
            )
    return {
        "feature_split_clip_identity_verified": True,
        "features_compared": features,
        "support_test_recording_overlap_by_shot": leakage_by_shot,
        "nested_support_clip_identity_verified": len(ordered_shots) > 1,
        "nested_shot_order": ordered_shots,
    }


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


def summarize(
    args: argparse.Namespace,
    split_integrity: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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
                    fine_tune_path = (
                        destination / "DS_CNN_Model.birdset_fewshot.json"
                    )
                    birdset_path = (
                        destination
                        / "BirdSet_common_20shot_heldout_evaluation.json"
                    )
                    db3v_path = destination / "DB3V_full_evaluation.json"
                    if (
                        not fine_tune_path.exists()
                        or not birdset_path.exists()
                        or not db3v_path.exists()
                    ):
                        missing.append(str(destination.relative_to(REPOSITORY_ROOT)))
                        continue
                    fine_tune = load_json(fine_tune_path)
                    birdset_evaluation = load_json(birdset_path)
                    db3v_evaluation = load_json(db3v_path)
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
                    support_validation = selected["birdset_support_validation"]
                    birdset = birdset_evaluation["models"]["DS_CNN_Model"]
                    birdset_clip = birdset["clip_level"]
                    birdset_singleton = birdset["globally_singleton_clip_level"]
                    db3v = db3v_evaluation["models"]["DS_CNN_Model"]["pooled"][
                        "recording_level"
                    ]
                    run_rows.append(
                        {
                            "feature": feature,
                            "requested_shots": shots,
                            "actual_support_recordings": protocol[
                                "support_original_recordings"
                            ],
                            "actual_support_clips": protocol["support_clips"],
                            "support_shortfall_by_class": "|".join(
                                str(value)
                                for value in protocol[
                                    "support_shortfall_positive_clips_by_class"
                                ]
                            ),
                            "policy": policy,
                            "seed": seed,
                            "selected_epoch": protocol["selected_epoch"],
                            "selection_validation_recordings": protocol[
                                "selection_validation_original_recordings"
                            ],
                            "selection_validation_recording_names": "|".join(
                                protocol["selection_validation_recording_names"]
                            ),
                            "trainable_parameters": protocol[
                                "trainable_parameters"
                            ],
                            "total_parameters": protocol["total_parameters"],
                            "batch_normalization_trainable": protocol[
                                "batch_normalization_trainable"
                            ],
                            "replay_enabled": protocol["replay"]["enabled"],
                            "selection_birdset_top1": support_validation[
                                "top_1_any_target_accuracy"
                            ],
                            "selection_birdset_top3": support_validation[
                                "top_3_any_target_accuracy"
                            ],
                            "selection_birdset_supported_macro_f1": (
                                support_validation[
                                    "threshold_supported_macro_f1"
                                ]
                            ),
                            "selection_xeno_macro_f1": selected[
                                "xeno_validation"
                            ]["macro_f1"],
                            "selection_adaptation_score": selected[
                                "adaptation_score"
                            ],
                            "final_xeno_macro_f1": final_xeno,
                            "xeno_macro_f1_change": final_xeno - baseline_xeno,
                            "xeno_macro_f1_retention": final_retention,
                            "common_birdset_heldout_clips": birdset_clip[
                                "samples"
                            ],
                            "birdset_top1": birdset_clip[
                                "top1_any_target_accuracy"
                            ],
                            "birdset_top3": birdset_clip[
                                "top3_any_target_accuracy"
                            ],
                            "birdset_singleton_samples": birdset_singleton[
                                "samples"
                            ],
                            "birdset_singleton_accuracy": birdset_singleton[
                                "accuracy"
                            ],
                            "birdset_singleton_macro_f1": birdset_singleton[
                                "supported_macro_f1"
                            ],
                            "db3v_recordings": db3v_evaluation["models"][
                                "DS_CNN_Model"
                            ]["pooled"]["recordings"],
                            "db3v_accuracy": db3v["accuracy"],
                            "db3v_balanced_accuracy": db3v[
                                "balanced_accuracy"
                            ],
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
        "selection_birdset_top1",
        "selection_birdset_top3",
        "selection_birdset_supported_macro_f1",
        "selection_xeno_macro_f1",
        "selection_adaptation_score",
        "final_xeno_macro_f1",
        "xeno_macro_f1_change",
        "xeno_macro_f1_retention",
        "birdset_top1",
        "birdset_top3",
        "birdset_singleton_accuracy",
        "birdset_singleton_macro_f1",
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
            "actual_support_clips": rows[0]["actual_support_clips"],
            "support_shortfall_by_class": rows[0]["support_shortfall_by_class"],
            "policy": policy,
            "n_seeds": len(rows),
            "expected_seeds": len(args.seeds),
            "complete": len(rows) == len(args.seeds),
            "seeds": "|".join(
                str(row["seed"]) for row in sorted(rows, key=lambda item: item["seed"])
            ),
            "trainable_parameters": rows[0]["trainable_parameters"],
            "total_parameters": rows[0]["total_parameters"],
            "common_birdset_heldout_clips": rows[0][
                "common_birdset_heldout_clips"
            ],
            "birdset_singleton_samples": rows[0]["birdset_singleton_samples"],
            "db3v_recordings": rows[0]["db3v_recordings"],
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
    split_summaries = {}
    for shots in args.shots:
        manifest = load_json(split_root(args.features[0], shots) / "split_manifest.json")
        split_summaries[str(shots)] = manifest
    protocol = {
        "experiment": (
            "Strict grouped BirdSet few-shot fine-tuning ablation with multiple seeds"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_sources": {
            "birdset_dataset": (
                "https://huggingface.co/datasets/DBD-research-group/BirdSet"
            ),
            "birdset_code": "https://github.com/DBD-research-group/BirdSet",
            "birdset_paper": (
                "https://proceedings.iclr.cc/paper_files/paper/2025/"
                "hash/484d254ff80e99d543159440a06db0de-Abstract-Conference.html"
            ),
            "ssw_audio": "https://zenodo.org/records/7079380",
            "db3v_cross_test": "https://doi.org/10.5281/zenodo.11544734",
        },
        "features": args.features,
        "requested_positive_clips_per_class": args.shots,
        "shot_definition": (
            "Requested positive five-second clips per class while preserving complete "
            "original long recordings. Actual support can exceed the request; a "
            "rare class with only two available positive clips is never duplicated."
        ),
        "split_summaries": split_summaries,
        "split_integrity": split_integrity,
        "common_birdset_heldout": (
            "The feature-matched 20-shot split test partition is used for every "
            "5/10/20-shot run."
        ),
        "policies": {policy: POLICY_DEFINITIONS[policy] for policy in args.policies},
        "seeds": args.seeds,
        "random_seed_scope": (
            "Fixed external support/common-held-out recordings; seed changes the "
            "assigned one of the three highest-quality internal grouped validation "
            "variants and changes replay sampling, batch order, and TensorFlow "
            "stochastic operations."
        ),
        "selection": (
            "Policy/epoch selection uses support validation clip top-1 any-target "
            "accuracy multiplied by capped Xeno-canto validation macro-F1 retention."
        ),
        "heldout_access_during_training_or_selection": False,
        "birdset_final_test": (
            "Every run is evaluated after training on the feature-matched common "
            "20-shot BirdSet held-out set. These metrics never select policies."
        ),
        "db3v_cross_domain_test": (
            "Every run is evaluated after training on full DB3V, which never "
            "participates in BirdSet adaptation or policy selection."
        ),
        "xeno_forgetting_test": (
            "Final Xeno-canto validation macro-F1, absolute change from the base "
            "model, and retention ratio are recorded for every run."
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
    split_integrity = verify_split_integrity(args.features, args.shots)
    if not args.summarize_only:
        train_and_evaluate(args)
    run_rows, aggregate_rows = summarize(args, split_integrity)
    print(
        f"Saved {len(run_rows)} seed runs and {len(aggregate_rows)} aggregate rows "
        f"to {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
