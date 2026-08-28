"""Flash LED_TEST models and replay the desktop test tensors on STM32.

Run this script on the Linux board-development host.  It keeps the Model_train
and LED_TEST repositories separate and communicates with LED_TEST only through
its documented tools and an unpacked firmware package.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKAGE = REPOSITORY_ROOT / "board_replay_testset"
EXPECTED_LABELS = (
    "Agelaius_phoeniceus",
    "Cardinalis_cardinalis",
    "Certhia_americana",
    "Corvus_brachyrhynchos",
    "Setophaga_aestiva",
    "Setophaga_ruticilla",
    "Spinus_tristis",
    "Turdus_migratorius",
)
REGIONS = (1, 2, 3)


class ReplayError(RuntimeError):
    """Raised when board replay cannot continue safely."""


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--led-test", required=True, type=Path)
    parser.add_argument("--pack", required=True, type=Path)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--timeout", type=float, default=12.0)
    parser.add_argument("--serial-wait", type=float, default=30.0)
    parser.add_argument("--openocd", default="openocd")
    parser.add_argument("--flash-config", type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--tier", choices=("probe", "full"), default="probe")
    parser.add_argument(
        "--corpus",
        action="append",
        choices=("xeno", "birdset", "db3v"),
        default=[],
        help="May be repeated; defaults to all three corpora.",
    )
    parser.add_argument("--scope", choices=("core", "all"), default="core")
    parser.add_argument(
        "--chain",
        action="append",
        default=[],
        help="Exact chain_id; may be repeated.",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "f32", "native", "both"),
        default="auto",
        help="auto uses both for probe and native for full.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def resolve_layout(arguments: argparse.Namespace) -> dict[str, Path]:
    led_test = arguments.led_test.resolve()
    pack = arguments.pack.resolve()
    package = arguments.package.resolve()
    serial_client = led_test / "tools" / "serial_model_client.py"
    flash_tool = led_test / "tools" / "flash_experiment.py"
    repository_evaluator = (
        REPOSITORY_ROOT / "src" / "experiments" / "evaluate_board_replay.py"
    )
    packaged_evaluator = package / "tools" / "evaluate_board_replay.py"
    evaluator = (
        repository_evaluator
        if repository_evaluator.is_file()
        else packaged_evaluator
    )
    required = (
        led_test / ".git",
        pack / "INDEX.csv",
        package / "chain_matrix.csv",
        serial_client,
        flash_tool,
        evaluator,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise ReplayError(f"Required paths do not exist: {missing}")
    results = (
        arguments.results.resolve()
        if arguments.results
        else package / "board_replay_results" / arguments.tier
    )
    return {
        "led_test": led_test,
        "pack": pack,
        "package": package,
        "serial_client": serial_client,
        "flash_tool": flash_tool,
        "evaluator": evaluator,
        "results": results,
    }


def select_chains(
    arguments: argparse.Namespace,
    layout: dict[str, Path],
) -> list[dict[str, str]]:
    replay_rows = load_csv(layout["package"] / "chain_matrix.csv")
    pack_rows = load_csv(layout["pack"] / "INDEX.csv")
    pack_by_chain = {row["chain_id"]: row for row in pack_rows}
    requested = set(arguments.chain)
    unknown = requested - {row["chain_id"] for row in replay_rows}
    if unknown:
        raise ReplayError(f"Unknown replay chain IDs: {sorted(unknown)}")
    selected: list[dict[str, str]] = []
    for replay in replay_rows:
        chain_id = replay["chain_id"]
        if requested and chain_id not in requested:
            continue
        pack = pack_by_chain.get(chain_id)
        if pack is None:
            raise ReplayError(f"Firmware pack is missing chain {chain_id}.")
        if (
            not requested
            and arguments.scope == "core"
            and pack.get("core_experiment") != "True"
        ):
            continue
        if pack.get("feature", "").upper() != replay["feature"].upper():
            raise ReplayError(f"Feature mismatch for {chain_id}.")
        if pack.get("activation", "").lower() != replay["activation"].lower():
            raise ReplayError(f"Activation mismatch for {chain_id}.")
        if pack.get("model_sha256") != replay["tflite_sha256"]:
            raise ReplayError(f"INDEX.csv model hash mismatch for {chain_id}.")
        firmware = (layout["pack"] / pack["hex_path"]).resolve()
        try:
            firmware.relative_to(layout["pack"])
        except ValueError as exc:
            raise ReplayError(f"Firmware path escapes pack for {chain_id}.") from exc
        if not firmware.is_file():
            raise ReplayError(f"Firmware does not exist: {firmware}")
        packaged_tflite = firmware.parent / "model.tflite"
        replay_tflite = layout["package"] / replay["tflite"]
        if not packaged_tflite.is_file():
            raise ReplayError(f"Packaged TFLite does not exist: {packaged_tflite}")
        if sha256_file(packaged_tflite) != replay["tflite_sha256"]:
            raise ReplayError(f"Packaged TFLite hash mismatch for {chain_id}.")
        if sha256_file(replay_tflite) != replay["tflite_sha256"]:
            raise ReplayError(f"Replay TFLite hash mismatch for {chain_id}.")
        selected.append(
            {
                **replay,
                "hex_path": str(firmware),
                "pack_tflite": str(packaged_tflite),
            }
        )
    if not selected:
        raise ReplayError("Model selection is empty.")
    return selected


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        stream.write(f"$ {' '.join(command)}\n")
        stream.flush()
        result = subprocess.run(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        raise ReplayError(
            f"Command failed with exit code {result.returncode}; see {log_path}"
        )


def wait_for_info(
    command: list[str],
    wait_seconds: float,
    timeout: float,
    log_path: Path,
) -> dict[str, Any]:
    deadline = time.monotonic() + wait_seconds
    attempts: list[str] = []
    while time.monotonic() < deadline:
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=max(timeout, 1.0),
                check=False,
            )
            attempts.append(
                f"$ {' '.join(command)}\n{result.stdout}\n{result.stderr}\n"
            )
            if result.returncode == 0:
                value = json.loads(result.stdout)
                if isinstance(value, dict):
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    log_path.write_text("\n".join(attempts), encoding="utf-8")
                    return value
        except (subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
            attempts.append(f"$ {' '.join(command)}\n{exc}\n")
        time.sleep(0.5)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(attempts), encoding="utf-8")
    raise ReplayError(f"Board INFO did not become valid within {wait_seconds}s.")


def serial_prefix(
    layout: dict[str, Path],
    arguments: argparse.Namespace,
) -> list[str]:
    return [
        sys.executable,
        str(layout["serial_client"]),
        "--port",
        arguments.port,
        "--baud",
        str(arguments.baud),
        "--timeout",
        str(arguments.timeout),
    ]


def validate_info(info: dict[str, Any], chain: dict[str, str]) -> None:
    if info.get("model_status") != 0:
        raise ReplayError(f"Board model_status is {info.get('model_status')!r}.")
    if info.get("model_sha256") != chain["tflite_sha256"]:
        raise ReplayError(
            f"Board model hash {info.get('model_sha256')!r} does not match "
            f"{chain['chain_id']}."
        )
    if str(info.get("feature", "")).upper() != chain["feature"].upper():
        raise ReplayError(f"Board feature mismatch for {chain['chain_id']}.")
    if str(info.get("activation", "")).lower() != chain["activation"].lower():
        raise ReplayError(f"Board activation mismatch for {chain['chain_id']}.")
    if tuple(info.get("labels", ())) != EXPECTED_LABELS:
        raise ReplayError(f"Board label order mismatch for {chain['chain_id']}.")


def tensor_inputs(
    package: Path,
    tier: str,
    chain: dict[str, str],
    corpus: str,
) -> list[tuple[str, Path]]:
    feature = chain["feature"]
    if corpus == "xeno":
        directory = package / "tensors" / tier / "xeno_validation" / feature
        return [("all", directory / "data.npy")]
    if corpus == "birdset":
        directory = (
            package
            / "tensors"
            / tier
            / "birdset_common_20shot_heldout"
            / feature
        )
        return [("all", directory / "data.npy")]
    directory = package / "tensors" / tier / chain["db3v_scope"] / feature
    if tier == "probe":
        return [("all", directory / "data.npy")]
    return [
        (str(region), directory / f"region_{region}_data.npy")
        for region in REGIONS
    ]


def evaluate_command(
    layout: dict[str, Path],
    arguments: argparse.Namespace,
    chain: dict[str, str],
    corpus: str,
    predictions: list[tuple[str, Path]],
    output: Path,
    mode: str,
) -> list[str]:
    values = []
    for key, path in predictions:
        values.append(str(path) if key == "all" else f"{key}={path}")
    return [
        sys.executable,
        str(layout["evaluator"]),
        "--package",
        str(layout["package"]),
        "--chain-id",
        chain["chain_id"],
        "--tier",
        arguments.tier,
        "--corpus",
        corpus,
        "--mode",
        "native" if mode == "both" else mode,
        "--predictions",
        *values,
        "--output",
        str(output),
    ]


def run_chain(
    arguments: argparse.Namespace,
    layout: dict[str, Path],
    chain: dict[str, str],
    corpora: list[str],
    mode: str,
) -> dict[str, Any]:
    chain_id = chain["chain_id"]
    chain_dir = layout["results"] / "models" / chain_id
    status_path = chain_dir / "status.json"
    if arguments.resume and status_path.is_file():
        existing = load_json(status_path)
        if existing.get("status") == "complete":
            return existing
    status: dict[str, Any] = {
        "chain_id": chain_id,
        "status": "running",
        "tier": arguments.tier,
        "started_at": utc_now(),
        "corpora": {},
    }
    write_json(status_path, status)
    try:
        flash_config = (
            arguments.flash_config.resolve()
            if arguments.flash_config
            else layout["pack"] / "flash.cfg"
        )
        flash_command = [
            sys.executable,
            str(layout["flash_tool"]),
            chain["hex_path"],
            "--config",
            str(flash_config),
            "--openocd",
            arguments.openocd,
        ]
        run_logged(flash_command, chain_dir / "flash.log")

        info_command = [*serial_prefix(layout, arguments), "info"]
        info = wait_for_info(
            info_command,
            arguments.serial_wait,
            arguments.timeout,
            chain_dir / "info.log",
        )
        validate_info(info, chain)
        write_json(chain_dir / "info.json", info)

        tflite = layout["package"] / chain["tflite"]
        parity_command = [
            *serial_prefix(layout, arguments),
            "smoke",
            "--mode",
            "both",
            "--tflite",
            str(tflite),
            "--max-lsb-error",
            "1",
            "--output",
            str(chain_dir / "parity_predictions.csv"),
        ]
        run_logged(parity_command, chain_dir / "parity.log")

        for corpus in corpora:
            corpus_dir = chain_dir / corpus
            inputs = tensor_inputs(
                layout["package"], arguments.tier, chain, corpus
            )
            predictions: list[tuple[str, Path]] = []
            for key, input_path in inputs:
                if not input_path.is_file():
                    raise ReplayError(f"Replay tensor does not exist: {input_path}")
                output = corpus_dir / f"predictions_{key}.csv"
                predictions.append((key, output))
                if arguments.resume and output.is_file():
                    continue
                sweep_command = [
                    *serial_prefix(layout, arguments),
                    "sweep",
                    "--input",
                    str(input_path),
                    "--mode",
                    mode,
                    "--tflite",
                    str(tflite),
                    "--max-lsb-error",
                    "1",
                    "--output",
                    str(output),
                ]
                run_logged(
                    sweep_command,
                    corpus_dir / f"sweep_{key}.log",
                )
            metrics_path = corpus_dir / "metrics.json"
            evaluation = evaluate_command(
                layout,
                arguments,
                chain,
                corpus,
                predictions,
                metrics_path,
                mode,
            )
            run_logged(evaluation, corpus_dir / "evaluate.log")
            status["corpora"][corpus] = {
                "status": "complete",
                "metrics": str(metrics_path),
            }
            write_json(status_path, status)
        status.update({"status": "complete", "completed_at": utc_now()})
        write_json(status_path, status)
        return status
    except (
        OSError,
        ReplayError,
        subprocess.SubprocessError,
        json.JSONDecodeError,
    ) as exc:
        status.update(
            {
                "status": "failed",
                "completed_at": utc_now(),
                "error": str(exc),
            }
        )
        write_json(status_path, status)
        return status


def write_summary(path: Path, statuses: list[dict[str, Any]]) -> None:
    rows = [
        {
            "chain_id": status["chain_id"],
            "status": status["status"],
            "tier": status.get("tier", ""),
            "completed_corpora": "|".join(
                sorted(
                    corpus
                    for corpus, value in status.get("corpora", {}).items()
                    if value.get("status") == "complete"
                )
            ),
            "error": status.get("error", ""),
        }
        for status in statuses
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    arguments = parse_arguments()
    if arguments.baud <= 0 or arguments.timeout <= 0 or arguments.serial_wait <= 0:
        raise ReplayError("Baud, timeout, and serial-wait must be positive.")
    layout = resolve_layout(arguments)
    chains = select_chains(arguments, layout)
    corpora = arguments.corpus or ["xeno", "birdset", "db3v"]
    mode = (
        ("both" if arguments.tier == "probe" else "native")
        if arguments.mode == "auto"
        else arguments.mode
    )
    plan = {
        "created_at": utc_now(),
        "tier": arguments.tier,
        "mode": mode,
        "corpora": corpora,
        "port": arguments.port,
        "models": [chain["chain_id"] for chain in chains],
        "model_count": len(chains),
        "package": str(layout["package"]),
        "firmware_pack": str(layout["pack"]),
        "led_test": str(layout["led_test"]),
        "results": str(layout["results"]),
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    if arguments.dry_run:
        return 0
    layout["results"].mkdir(parents=True, exist_ok=True)
    write_json(layout["results"] / "run_plan.json", plan)
    statuses = []
    for index, chain in enumerate(chains, start=1):
        print(f"[{index}/{len(chains)}] {chain['chain_id']}", flush=True)
        status = run_chain(arguments, layout, chain, corpora, mode)
        statuses.append(status)
        write_summary(layout["results"] / "summary.csv", statuses)
        if status["status"] != "complete" and not arguments.continue_on_error:
            return 1
    return 0 if all(status["status"] == "complete" for status in statuses) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReplayError as error:
        print(f"board replay error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
