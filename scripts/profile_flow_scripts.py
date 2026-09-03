#!/usr/bin/env python
"""Profile flow scripts with py-spy flamegraphs."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config.west_yorkshire.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "profile_results"
ALLOCATION_SCRIPT = REPO_ROOT / "scripts" / "flow_model" / "flow_allocation.py"
DISRUPTIONS_SCRIPT = REPO_ROOT / "scripts" / "flow_model" / "flow_disruptions.py"


class ProfileError(RuntimeError):
    """Raised when profiling cannot run."""


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    py_spy = shutil.which("py-spy")
    if py_spy is None:
        raise ProfileError("py-spy is not available on PATH; run through Pixi")

    if args.script in ("allocation", "all"):
        profile_script(
            py_spy=py_spy,
            script_path=ALLOCATION_SCRIPT,
            config_path=args.config.resolve(),
            output_path=output_dir / "flow_allocation.svg",
            rate=args.rate,
            duration=args.duration,
        )

    if args.script in ("disruptions", "all"):
        if args.script == "disruptions" and not args.skip_setup_allocation:
            run_setup_allocation(args.config.resolve())
        profile_script(
            py_spy=py_spy,
            script_path=DISRUPTIONS_SCRIPT,
            config_path=args.config.resolve(),
            output_path=output_dir / "flow_disruptions.svg",
            rate=args.rate,
            duration=args.duration,
        )

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write py-spy SVG flamegraphs for flow allocation/disruption scripts."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Config JSON to pass to profiled scripts. Default: {DEFAULT_CONFIG}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for SVG flamegraphs. Default: {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--script",
        choices=("allocation", "disruptions", "all"),
        default="all",
        help="Which flow script to profile. Default: all.",
    )
    parser.add_argument(
        "--rate",
        type=int,
        default=10,
        help="py-spy samples per second. Default: 10.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Seconds to sample each profiled script. Default: 30.",
    )
    parser.add_argument(
        "--skip-setup-allocation",
        action="store_true",
        help=(
            "When profiling only disruptions, do not run allocation first to "
            "prepare flow_od_paths inputs."
        ),
    )
    args = parser.parse_args()
    if args.rate < 1:
        parser.error("--rate must be at least 1")
    if args.duration < 1:
        parser.error("--duration must be at least 1")
    if not args.config.exists():
        parser.error(f"Config does not exist: {args.config}")
    return args


def profile_script(
    *,
    py_spy: str,
    script_path: Path,
    config_path: Path,
    output_path: Path,
    rate: int,
    duration: int,
) -> None:
    command = [
        py_spy,
        "record",
        "--format",
        "flamegraph",
        "--rate",
        str(rate),
        "--duration",
        str(duration),
        "--output",
        str(output_path),
        "--",
        sys.executable,
        str(script_path),
        str(config_path),
    ]
    print(f"Writing flamegraph: {output_path}", flush=True)
    run(
        command,
        successful_output_path=output_path,
        tolerate_no_child_exit=True,
    )


def run_setup_allocation(config_path: Path) -> None:
    command = [
        sys.executable,
        str(ALLOCATION_SCRIPT),
        str(config_path),
    ]
    print("Running allocation first to prepare disruption inputs", flush=True)
    run(command)


def run(
    command: list[str],
    *,
    successful_output_path: Path | None = None,
    tolerate_no_child_exit: bool = False,
) -> None:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    if completed.returncode != 0:
        if (
            tolerate_no_child_exit
            and successful_output_path is not None
            and successful_output_path.exists()
            and successful_output_path.stat().st_size > 0
            and "No child process" in f"{completed.stdout}\n{completed.stderr}"
        ):
            print(
                "py-spy returned a non-zero code after writing the flamegraph; "
                "continuing because the output file exists",
                flush=True,
            )
            return
        raise ProfileError(
            f"Command failed with return code {completed.returncode}: {command}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
