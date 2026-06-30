#!/usr/bin/env python
"""Benchmark the end-to-end flow allocation and disruption scripts."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ALLOCATION_SCRIPT = REPO_ROOT / "scripts" / "flow_model" / "flow_allocation.py"
DISRUPTION_SCRIPT = REPO_ROOT / "scripts" / "flow_model" / "flow_disruptions.py"
DEFAULT_CONFIG = REPO_ROOT / "config.example.json"

CSV_FIELDS = (
    "timestamp_utc",
    "config",
    "repeat",
    "script",
    "elapsed_seconds",
    "return_code",
    "results_path",
    "od_flows_rows",
    "network_edge_total_flows_rows",
    "unassigned_od_flows_rows",
    "flow_disruption_losses_rows",
)


class BenchmarkError(RuntimeError):
    """Raised when a benchmarked script exits unsuccessfully."""


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    start = time.perf_counter()

    try:
        for config_path in args.config:
            for repeat in range(1, args.repeats + 1):
                results = benchmark_config(
                    config_path.resolve(),
                    repeat=repeat,
                    keep_workdirs=args.keep_workdirs,
                )
                all_results.extend(results)
                print_results(results)
    finally:
        if all_results:
            append_csv(output_dir / "flow_script_benchmark.csv", all_results)
            write_json(output_dir / "flow_script_benchmark_latest.json", all_results)

    total_elapsed = time.perf_counter() - start
    if args.max_seconds is not None and total_elapsed > args.max_seconds:
        raise BenchmarkError(
            f"Benchmark took {total_elapsed:.3f}s, exceeding "
            f"--max-seconds={args.max_seconds:.3f}s"
        )

    print(f"Total benchmark time: {total_elapsed:.3f}s")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark flow allocation and disruption scripts."
    )
    parser.add_argument(
        "--config",
        action="append",
        type=Path,
        default=None,
        help=(
            "Config file to benchmark. May be provided multiple times. "
            "Defaults to config.example.json."
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of times to run each config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results"),
        help="Directory for benchmark CSV and latest JSON outputs.",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Fail if total benchmark runtime exceeds this many seconds.",
    )
    parser.add_argument(
        "--keep-workdirs",
        action="store_true",
        help="Keep temporary benchmark work directories for debugging.",
    )
    args = parser.parse_args()
    if args.config is None:
        args.config = [DEFAULT_CONFIG]
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    return args


def benchmark_config(
    config_path: Path,
    *,
    repeat: int,
    keep_workdirs: bool,
) -> list[dict[str, str | int | float]]:
    workdir = Path(tempfile.mkdtemp(prefix="transport-flow-benchmark-"))
    should_cleanup = not keep_workdirs

    try:
        temp_config, results_path = prepare_config(config_path, workdir)
        timestamp = datetime.now(timezone.utc).isoformat()
        rows = []
        for script_name, script_path in (
            ("flow_allocation", ALLOCATION_SCRIPT),
            ("flow_disruptions", DISRUPTION_SCRIPT),
        ):
            run = run_script(script_path, temp_config)
            row = {
                "timestamp_utc": timestamp,
                "config": str(config_path),
                "repeat": repeat,
                "script": script_name,
                "elapsed_seconds": round(run["elapsed_seconds"], 6),
                "return_code": run["return_code"],
                "results_path": str(results_path),
                **count_outputs(results_path),
            }
            rows.append(row)
            if run["return_code"] != 0:
                raise BenchmarkError(
                    f"{script_name} failed for {config_path} "
                    f"with return code {run['return_code']}\n"
                    f"stdout:\n{run['stdout']}\n"
                    f"stderr:\n{run['stderr']}"
                )
        if keep_workdirs:
            print(f"Kept benchmark workdir: {workdir}")
        return rows
    finally:
        if should_cleanup:
            shutil.rmtree(workdir)


def prepare_config(config_path: Path, workdir: Path) -> tuple[Path, Path]:
    with config_path.open("r", encoding="utf-8-sig") as config_fh:
        config = json.load(config_fh)

    paths = config.setdefault("paths", {})
    data_path = copy_input_path(paths.get("data"), "data", workdir)
    results_path = workdir / "results"
    results_path.mkdir(parents=True, exist_ok=True)

    paths["data"] = str(data_path)
    paths["results"] = str(results_path)

    for optional_key in ("incoming_data", "figures"):
        if optional_key not in paths:
            continue
        optional_path = resolve_config_path(paths[optional_key])
        target = workdir / optional_key
        if optional_path.exists():
            copy_path(optional_path, target)
        else:
            target.mkdir(parents=True, exist_ok=True)
        paths[optional_key] = str(target)

    temp_config = workdir / "config.json"
    with temp_config.open("w", encoding="utf-8") as config_fh:
        json.dump(config, config_fh, indent=2)
    return temp_config, results_path


def copy_input_path(path_value: str | None, key: str, workdir: Path) -> Path:
    if path_value is None:
        raise BenchmarkError(f"Config is missing paths.{key}")
    source = resolve_config_path(path_value)
    if not source.exists():
        raise BenchmarkError(f"Configured paths.{key} does not exist: {source}")
    target = workdir / key
    copy_path(source, target)
    return target


def resolve_config_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def copy_path(source: Path, target: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, target)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def run_script(script_path: Path, config_path: Path) -> dict[str, str | int | float]:
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, str(script_path), str(config_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - started
    return {
        "elapsed_seconds": elapsed,
        "return_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def count_outputs(results_path: Path) -> dict[str, int | str]:
    files = {
        "od_flows_rows": results_path / "flow_od_paths" / "od_flows.csv",
        "network_edge_total_flows_rows": (
            results_path / "flow_od_paths" / "network_edge_total_flows.csv"
        ),
        "unassigned_od_flows_rows": (
            results_path / "flow_od_paths" / "unassigned_od_flows.csv"
        ),
        "flow_disruption_losses_rows": (
            results_path / "flow_disruptions" / "flow_disruption_losses.csv"
        ),
    }
    return {field: count_csv_rows(path) for field, path in files.items()}


def count_csv_rows(path: Path) -> int | str:
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8-sig", newline="") as csv_fh:
        reader = csv.reader(csv_fh)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def append_csv(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as csv_fh:
        writer = csv.DictWriter(csv_fh, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    with path.open("w", encoding="utf-8") as json_fh:
        json.dump(rows, json_fh, indent=2)


def print_results(rows: list[dict[str, str | int | float]]) -> None:
    for row in rows:
        print(
            f"{row['script']} config={row['config']} repeat={row['repeat']} "
            f"elapsed={row['elapsed_seconds']:.6f}s return={row['return_code']}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
