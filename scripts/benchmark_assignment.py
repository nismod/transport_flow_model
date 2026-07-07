#!/usr/bin/env python
"""Benchmark assignment methods: relative gap, wall time and peak RSS.

For equilibrium methods wall time is meaningless without solution quality,
so every case records the relative gap (see
:mod:`transport_flow_model.convergence`) alongside timings. Each
(instance, method, threads, repeat) case runs in a fresh subprocess so peak
RSS is measured per case.

Outputs (under ``--output-dir``, default ``benchmark_results/assignment``):

- ``assignment_benchmark_<timestamp>.parquet`` (and ``latest.parquet``):
  one row per case with the gap trajectory.
- ``summary.json``: per-case medians, usable as a regression baseline.
- ``report.md`` and per-instance ``<instance>_gap_vs_time.png`` plots.

Pass ``--baseline summary.json`` to fail (exit 1) on >``--max-slowdown``
wall-time regression or on relative-gap regression at a fixed iteration
budget. ``--handicap`` artificially scales measured wall time and exists so
CI can test that the regression check itself works.

This harness complements — not replaces — the existing performance layers:
``pixi run extension-bench`` (Criterion micro-benchmarks of the Rust core),
``pixi run benchmark-scripts`` (end-to-end legacy script timing) and
``pixi run profile-flow-scripts`` (py-spy flamegraphs).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Instance suites: ``small`` is vendored and safe for per-PR CI; ``large``
#: instances are downloaded into the dataset cache on first use.
SUITES = {
    "small": ("siouxfalls",),
    "large": ("anaheim", "barcelona", "chicago-sketch"),
}

#: Baseline/summary schema version, bumped on incompatible changes.
SUMMARY_VERSION = 1


class BenchmarkError(RuntimeError):
    """Raised when a benchmarked case fails."""


def main() -> int:
    args = parse_args()
    if args.run_case is not None:
        print(json.dumps(run_case(json.loads(args.run_case))))
        return 0

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    instances = list(args.instance or SUITES[args.suite])
    methods = list(args.method or default_methods())

    measurements = []
    for instance in instances:
        for method in methods:
            for repeat in range(1, args.repeats + 1):
                case = {
                    "instance": instance,
                    "method": method,
                    "threads": args.threads,
                    "iterations": args.iterations,
                    "repeat": repeat,
                    "handicap": args.handicap,
                }
                measurement = run_case_subprocess(case)
                if measurement.get("skipped"):
                    print(f"{instance}/{method}: skipped ({measurement['skipped']})")
                    break
                measurements.append(measurement)
                print(
                    f"{instance}/{method} threads={args.threads} repeat={repeat}: "
                    f"wall={measurement['wall_time_s']:.3f}s "
                    f"gap={_format_gap(measurement['relative_gap'])} "
                    f"peak_rss={measurement['peak_rss_bytes'] / 2**20:.0f}MiB"
                )

    if not measurements:
        raise BenchmarkError("No cases ran; all were skipped")

    summary = summarize(measurements)
    write_outputs(measurements, summary, output_dir)
    print(f"\n{render_report(summary)}")
    print(f"Results written to {output_dir}")

    if args.write_baseline is not None:
        args.write_baseline.parent.mkdir(parents=True, exist_ok=True)
        args.write_baseline.write_text(json.dumps(summary, indent=2))
        print(f"Baseline written to {args.write_baseline}")

    if args.baseline is not None:
        baseline = json.loads(args.baseline.read_text())
        violations = compare_to_baseline(
            baseline,
            summary,
            max_slowdown=args.max_slowdown,
            max_gap_ratio=args.max_gap_ratio,
        )
        if violations:
            print("\nPerformance regression check FAILED:", file=sys.stderr)
            for violation in violations:
                print(f"  - {violation}", file=sys.stderr)
            return 1
        print(f"\nPerformance regression check passed against {args.baseline}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark assignment methods (relative gap, time, memory)."
    )
    parser.add_argument(
        "--suite",
        choices=sorted(SUITES),
        default="small",
        help="Instance suite to run when --instance is not given.",
    )
    parser.add_argument(
        "--instance",
        action="append",
        help="Dataset name (repeatable); see transport_flow_model.datasets.",
    )
    parser.add_argument(
        "--method",
        action="append",
        help="Assignment method (repeatable). Defaults to all registered "
        "methods; unimplemented ones are skipped.",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Iteration budget forwarded to iterative methods (max_iterations).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "benchmark_results/assignment"
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        help="summary.json to compare against; exit 1 on regression.",
    )
    parser.add_argument(
        "--write-baseline", type=Path, help="Write this run's summary.json here."
    )
    parser.add_argument(
        "--max-slowdown",
        type=float,
        default=1.2,
        help="Fail --baseline check when median wall time exceeds this ratio.",
    )
    parser.add_argument(
        "--max-gap-ratio",
        type=float,
        default=1.05,
        help="Fail --baseline check when relative gap exceeds this ratio.",
    )
    parser.add_argument(
        "--handicap",
        type=float,
        default=1.0,
        help="Scale measured wall time (for testing the regression check).",
    )
    parser.add_argument("--run-case", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.handicap <= 0:
        parser.error("--handicap must be positive")
    return args


def default_methods() -> list[str]:
    from transport_flow_model.assignment import METHODS

    return sorted(METHODS)


def run_case_subprocess(case: dict) -> dict:
    """Run one benchmark case in a fresh interpreter (clean peak RSS)."""
    env = dict(os.environ)
    threads = str(case["threads"])
    env["OMP_NUM_THREADS"] = threads
    env["RAYON_NUM_THREADS"] = threads
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--run-case", json.dumps(case)],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
        check=False,
    )
    if completed.returncode != 0:
        raise BenchmarkError(
            f"Case {case} failed with return code {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    try:
        return json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise BenchmarkError(
            f"Case {case} produced no measurement JSON\nstdout:\n{completed.stdout}"
        ) from error


def run_case(case: dict) -> dict:
    """Measure one (instance, method) case; runs inside the subprocess."""
    import resource

    from transport_flow_model import Demand, Network, assign, datasets, relative_gap
    from transport_flow_model._version import __version__
    from transport_flow_model import core

    instance = datasets.load_tntp(case["instance"])
    network = Network.from_tntp(instance)
    demand = Demand.from_tntp(instance)
    best_known = datasets.BEST_KNOWN.get(case["instance"])
    distance_cost = best_known.distance_cost if best_known is not None else 0.0

    options = (
        {} if case["method"] == "sequential" else {"max_iterations": case["iterations"]}
    )
    try:
        result = assign(network, demand, method=case["method"], **options)
    except NotImplementedError as error:
        return {"skipped": str(error)}
    wall_time_s = result.provenance.wall_time_s * case.get("handicap", 1.0)

    # Untimed post-hoc quality measure; infeasible (partially unassigned)
    # solutions have no defined gap.
    try:
        gap = relative_gap(network, demand, result, distance_cost=distance_cost)
    except ValueError:
        gap = None
    gap_trajectory = [g for g in result.gap_history] or (
        [gap] if gap is not None else []
    )
    # Per-iteration timings are not recorded by backends (yet): spread the
    # trajectory uniformly over the wall time.
    n = len(gap_trajectory)
    time_trajectory = [wall_time_s * (i + 1) / n for i in range(n)]

    ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_bytes = ru_maxrss if sys.platform == "darwin" else ru_maxrss * 1024
    unassigned = result.unassigned["value"]
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "instance": case["instance"],
        "method": case["method"],
        "threads": case["threads"],
        "repeat": case["repeat"],
        "iterations": result.provenance.iterations,
        "relative_gap": gap,
        "gap_trajectory": gap_trajectory,
        "time_trajectory": time_trajectory,
        "wall_time_s": wall_time_s,
        "peak_rss_bytes": int(peak_rss_bytes),
        "n_nodes": network.n_nodes,
        "n_links": network.n_links,
        "n_od_pairs": demand.n_pairs,
        "total_demand": demand.total,
        "unassigned_demand": float(
            sum(unassigned.to_pylist()) if len(unassigned) else 0.0
        ),
        "package_version": __version__,
        "core_version": core.version(),
    }


def case_key(measurement: dict) -> str:
    return (
        f"{measurement['instance']}/{measurement['method']}"
        f"/threads={measurement['threads']}"
    )


def summarize(measurements: list[dict]) -> dict:
    """Per-case medians over repeats, in the baseline/summary format."""
    cases: dict[str, dict] = {}
    for key in sorted({case_key(m) for m in measurements}):
        repeats = [m for m in measurements if case_key(m) == key]
        gaps = [m["relative_gap"] for m in repeats if m["relative_gap"] is not None]
        cases[key] = {
            "instance": repeats[0]["instance"],
            "method": repeats[0]["method"],
            "threads": repeats[0]["threads"],
            "iterations": max(m["iterations"] for m in repeats),
            "wall_time_s": statistics.median(m["wall_time_s"] for m in repeats),
            "relative_gap": statistics.median(gaps) if gaps else None,
            "peak_rss_bytes": max(m["peak_rss_bytes"] for m in repeats),
            "repeats": len(repeats),
        }
    return {
        "version": SUMMARY_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cases": cases,
    }


def compare_to_baseline(
    baseline: dict,
    current: dict,
    *,
    max_slowdown: float = 1.2,
    max_gap_ratio: float = 1.05,
    gap_epsilon: float = 1e-9,
) -> list[str]:
    """Regression violations of ``current`` against ``baseline`` (empty = pass).

    Fails on median wall time above ``max_slowdown`` times the baseline, and
    on relative gap above ``max_gap_ratio`` times the baseline gap (plus
    ``gap_epsilon`` absolute headroom for gaps at numerical zero). Cases new
    in ``current`` pass; cases missing from ``current`` fail.
    """
    violations = []
    for key, base in baseline.get("cases", {}).items():
        case = current.get("cases", {}).get(key)
        if case is None:
            violations.append(f"{key}: present in baseline but missing from this run")
            continue
        slowdown = case["wall_time_s"] / base["wall_time_s"]
        if slowdown > max_slowdown:
            violations.append(
                f"{key}: {slowdown:.2f}x slower "
                f"({base['wall_time_s']:.4f}s -> {case['wall_time_s']:.4f}s, "
                f"threshold {max_slowdown:.2f}x)"
            )
        base_gap, gap = base.get("relative_gap"), case.get("relative_gap")
        if base_gap is not None and (
            gap is None or gap > base_gap * max_gap_ratio + gap_epsilon
        ):
            violations.append(
                f"{key}: relative gap regressed "
                f"({_format_gap(base_gap)} -> {_format_gap(gap)} "
                f"at {case['iterations']} iterations, "
                f"threshold {max_gap_ratio:.2f}x)"
            )
    return violations


def write_outputs(measurements: list[dict], summary: dict, output_dir: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist(measurements)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    pq.write_table(table, output_dir / f"assignment_benchmark_{stamp}.parquet")
    pq.write_table(table, output_dir / "latest.parquet")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "report.md").write_text(render_report(summary) + "\n")
    plot_gap_vs_time(measurements, output_dir)


def render_report(summary: dict) -> str:
    """Markdown table comparing methods/backends per instance."""
    lines = [
        "| instance | method | threads | iterations | relative gap "
        "| wall time (s) | peak RSS (MiB) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in summary["cases"].values():
        lines.append(
            f"| {case['instance']} | {case['method']} | {case['threads']} "
            f"| {case['iterations']} | {_format_gap(case['relative_gap'])} "
            f"| {case['wall_time_s']:.4f} "
            f"| {case['peak_rss_bytes'] / 2**20:.0f} |"
        )
    return "\n".join(lines)


def plot_gap_vs_time(measurements: list[dict], output_dir: Path) -> None:
    try:
        import matplotlib
    except ImportError:
        print("matplotlib not installed; skipping gap-vs-time plots")
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for instance in sorted({m["instance"] for m in measurements}):
        figure, axes = plt.subplots(figsize=(7, 4.5))
        for measurement in measurements:
            if measurement["instance"] != instance or measurement["repeat"] != 1:
                continue
            gaps = [max(g, 1e-16) for g in measurement["gap_trajectory"]]
            if not gaps:
                continue
            axes.plot(
                measurement["time_trajectory"],
                gaps,
                marker="o",
                label=f"{measurement['method']} (threads={measurement['threads']})",
            )
        axes.set_yscale("log")
        axes.axhline(1e-4, color="grey", linestyle="--", linewidth=1)
        axes.text(
            0.02,
            1.2e-4,
            "1e-4 (Boyce et al. 2004)",
            color="grey",
            fontsize=8,
            transform=axes.get_yaxis_transform(),
        )
        axes.set_xlabel("wall time (s)")
        axes.set_ylabel("relative gap")
        axes.set_title(f"Convergence: {instance}")
        axes.legend()
        figure.tight_layout()
        figure.savefig(output_dir / f"{instance}_gap_vs_time.png", dpi=150)
        plt.close(figure)


def _format_gap(gap: float | None) -> str:
    return "n/a" if gap is None else f"{gap:.3e}"


if __name__ == "__main__":
    raise SystemExit(main())
