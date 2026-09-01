"""Summarize externally measured board-current traces for deployment experiments.

The STM32 serial protocol reports latency but cannot measure supply current.  This
tool consumes a timestamped current trace exported by an ammeter/power analyzer
and a separate interval manifest synchronized with a GPIO marker or the analyzer
clock.  It never derives power from ST-Link target-voltage readings alone.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--intervals", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument(
        "--voltage-v",
        type=float,
        help="Fixed supply voltage; required when samples.csv has no voltage_v column.",
    )
    parser.add_argument(
        "--baseline-current-ma",
        type=float,
        help="Override idle baseline. Otherwise all phase=idle intervals define it.",
    )
    parser.add_argument("--minimum-samples", type=int, default=10)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: str, field: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return number


def load_samples(path: Path, fixed_voltage: float | None) -> list[tuple[float, float, float]]:
    rows = read_csv(path)
    required = {"timestamp_s", "current_ma"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"samples CSV is missing columns: {sorted(missing)}")
    if "voltage_v" not in rows[0] and fixed_voltage is None:
        raise ValueError("samples CSV has no voltage_v; pass --voltage-v")
    samples: list[tuple[float, float, float]] = []
    previous = -math.inf
    for index, row in enumerate(rows, start=2):
        timestamp = finite(row["timestamp_s"], f"timestamp_s at row {index}")
        current = finite(row["current_ma"], f"current_ma at row {index}")
        voltage = (
            finite(row["voltage_v"], f"voltage_v at row {index}")
            if "voltage_v" in row and row["voltage_v"].strip()
            else fixed_voltage
        )
        if voltage is None or voltage <= 0:
            raise ValueError(f"voltage must be positive at row {index}")
        if current < 0:
            raise ValueError(f"current must be non-negative at row {index}")
        if timestamp <= previous:
            raise ValueError("sample timestamps must be strictly increasing")
        samples.append((timestamp, current, voltage))
        previous = timestamp
    if len(samples) < 2:
        raise ValueError("at least two power samples are required")
    return samples


def load_intervals(path: Path) -> list[dict[str, Any]]:
    rows = read_csv(path)
    required = {"chain_id", "phase", "start_s", "end_s", "iterations"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"interval CSV is missing columns: {sorted(missing)}")
    intervals: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=2):
        start = finite(row["start_s"], f"start_s at row {index}")
        end = finite(row["end_s"], f"end_s at row {index}")
        iterations = int(row["iterations"])
        if end <= start:
            raise ValueError(f"end_s must exceed start_s at row {index}")
        if iterations <= 0:
            raise ValueError(f"iterations must be positive at row {index}")
        chain_id = row["chain_id"].strip()
        phase = row["phase"].strip().lower()
        if not chain_id or not phase:
            raise ValueError(f"chain_id and phase are required at row {index}")
        intervals.append(
            {
                "chain_id": chain_id,
                "phase": phase,
                "start_s": start,
                "end_s": end,
                "iterations": iterations,
            }
        )
    return intervals


def interpolate(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
    timestamp: float,
) -> tuple[float, float, float]:
    fraction = (timestamp - left[0]) / (right[0] - left[0])
    return (
        timestamp,
        left[1] + fraction * (right[1] - left[1]),
        left[2] + fraction * (right[2] - left[2]),
    )


def clipped_samples(
    samples: list[tuple[float, float, float]], start: float, end: float
) -> list[tuple[float, float, float]]:
    if start < samples[0][0] or end > samples[-1][0]:
        raise ValueError(
            f"interval [{start}, {end}] lies outside trace "
            f"[{samples[0][0]}, {samples[-1][0]}]"
        )
    inside = [sample for sample in samples if start <= sample[0] <= end]
    for boundary, insert_at in ((start, 0), (end, -1)):
        if any(sample[0] == boundary for sample in inside):
            continue
        for left, right in zip(samples, samples[1:]):
            if left[0] < boundary < right[0]:
                point = interpolate(left, right, boundary)
                if insert_at == 0:
                    inside.insert(0, point)
                else:
                    inside.append(point)
                break
    return inside


def trapezoid(points: list[tuple[float, float]]) -> float:
    return sum(
        (right_t - left_t) * (left_y + right_y) / 2.0
        for (left_t, left_y), (right_t, right_y) in zip(points, points[1:])
    )


def summarize_interval(
    samples: list[tuple[float, float, float]], interval: dict[str, Any], minimum: int
) -> dict[str, Any]:
    selected = clipped_samples(samples, interval["start_s"], interval["end_s"])
    if len(selected) < minimum:
        raise ValueError(
            f"{interval['chain_id']}/{interval['phase']} has {len(selected)} samples; "
            f"minimum is {minimum}"
        )
    duration = interval["end_s"] - interval["start_s"]
    current_integral = trapezoid([(row[0], row[1]) for row in selected])
    voltage_integral = trapezoid([(row[0], row[2]) for row in selected])
    energy_mj = trapezoid([(row[0], row[1] * row[2]) for row in selected])
    return {
        **interval,
        "sample_count": len(selected),
        "duration_s": duration,
        "mean_current_ma": current_integral / duration,
        "mean_voltage_v": voltage_integral / duration,
        "mean_power_mw": energy_mj / duration,
        "gross_energy_mj": energy_mj,
        "gross_energy_per_iteration_mj": energy_mj / interval["iterations"],
    }


def mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def main() -> None:
    arguments = parse_arguments()
    if arguments.minimum_samples < 2:
        raise ValueError("--minimum-samples must be at least 2")
    if arguments.voltage_v is not None and arguments.voltage_v <= 0:
        raise ValueError("--voltage-v must be positive")
    if arguments.baseline_current_ma is not None and arguments.baseline_current_ma < 0:
        raise ValueError("--baseline-current-ma must be non-negative")

    samples = load_samples(arguments.samples, arguments.voltage_v)
    intervals = load_intervals(arguments.intervals)
    results = [
        summarize_interval(samples, interval, arguments.minimum_samples)
        for interval in intervals
    ]

    idle_results = [row for row in results if row["phase"] == "idle"]
    if arguments.baseline_current_ma is not None:
        baseline_current = arguments.baseline_current_ma
        baseline_source = "command_line"
    elif idle_results:
        baseline_current = sum(
            row["mean_current_ma"] * row["duration_s"] for row in idle_results
        ) / sum(row["duration_s"] for row in idle_results)
        baseline_source = "phase_idle_intervals_duration_weighted"
    else:
        baseline_current = None
        baseline_source = "unavailable"

    for row in results:
        if row["phase"] == "idle" or baseline_current is None:
            row["dynamic_energy_mj"] = None
            row["dynamic_energy_per_iteration_mj"] = None
            continue
        baseline_energy = (
            baseline_current * row["mean_voltage_v"] * row["duration_s"]
        )
        dynamic = row["gross_energy_mj"] - baseline_energy
        row["dynamic_energy_mj"] = dynamic
        row["dynamic_energy_per_iteration_mj"] = dynamic / row["iterations"]

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        if row["phase"] != "idle":
            grouped[(row["chain_id"], row["phase"])].append(row)
    aggregate: list[dict[str, Any]] = []
    for (chain_id, phase), rows in sorted(grouped.items()):
        item: dict[str, Any] = {
            "chain_id": chain_id,
            "phase": phase,
            "runs": len(rows),
            "gross_energy_per_iteration_mj": mean_std(
                [row["gross_energy_per_iteration_mj"] for row in rows]
            ),
            "mean_power_mw": mean_std([row["mean_power_mw"] for row in rows]),
        }
        dynamic_values = [
            row["dynamic_energy_per_iteration_mj"]
            for row in rows
            if row["dynamic_energy_per_iteration_mj"] is not None
        ]
        item["dynamic_energy_per_iteration_mj"] = (
            mean_std(dynamic_values) if dynamic_values else None
        )
        aggregate.append(item)

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "measurement_scope": "external_supply_current_trace",
        "samples_sha256": sha256_file(arguments.samples),
        "intervals_sha256": sha256_file(arguments.intervals),
        "sample_count": len(samples),
        "trace_start_s": samples[0][0],
        "trace_end_s": samples[-1][0],
        "baseline_current_ma": baseline_current,
        "baseline_source": baseline_source,
        "intervals": results,
        "aggregate": aggregate,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    if arguments.output_csv:
        arguments.output_csv.parent.mkdir(parents=True, exist_ok=True)
        fields = list(results[0])
        with arguments.output_csv.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(results)
    print(
        f"Saved {len(results)} intervals and {len(aggregate)} aggregate rows to "
        f"{arguments.output}"
    )


if __name__ == "__main__":
    main()
