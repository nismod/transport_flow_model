"""The assignment benchmark harness must catch performance regressions.

Covers the regression-check logic directly and end-to-end: a run with an
artificially introduced 2x slowdown (``--handicap 2``) must fail against a
freshly written baseline.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "benchmark_assignment.py"

spec = importlib.util.spec_from_file_location("benchmark_assignment", SCRIPT)
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)


def summary(**overrides):
    case = {
        "instance": "siouxfalls",
        "method": "sequential",
        "threads": 1,
        "iterations": 50,
        "wall_time_s": 1.0,
        "relative_gap": 1e-4,
        "peak_rss_bytes": 100 * 2**20,
        "repeats": 3,
        **overrides,
    }
    return {"version": 1, "cases": {"siouxfalls/sequential/threads=1": case}}


def test_passes_within_thresholds():
    baseline = summary()
    current = summary(wall_time_s=1.1, relative_gap=1.01e-4)
    assert harness.compare_to_baseline(baseline, current) == []


def test_catches_slowdown():
    violations = harness.compare_to_baseline(summary(), summary(wall_time_s=2.0))
    assert len(violations) == 1
    assert "2.00x slower" in violations[0]


def test_catches_gap_regression():
    violations = harness.compare_to_baseline(summary(), summary(relative_gap=2e-4))
    assert len(violations) == 1
    assert "gap regressed" in violations[0]


def test_catches_gap_becoming_undefined():
    violations = harness.compare_to_baseline(summary(), summary(relative_gap=None))
    assert len(violations) == 1
    assert "gap regressed" in violations[0]


def test_zero_gap_baseline_allows_numerical_noise():
    baseline = summary(relative_gap=0.0)
    assert harness.compare_to_baseline(baseline, summary(relative_gap=1e-12)) == []
    violations = harness.compare_to_baseline(baseline, summary(relative_gap=1e-3))
    assert len(violations) == 1


def test_catches_missing_case():
    current = {"version": 1, "cases": {}}
    violations = harness.compare_to_baseline(summary(), current)
    assert len(violations) == 1
    assert "missing" in violations[0]


def test_new_case_passes():
    baseline = {"version": 1, "cases": {}}
    assert harness.compare_to_baseline(baseline, summary()) == []


def test_ci_catches_artificial_2x_slowdown(tmp_path):
    """End-to-end acceptance: baseline run, then a 2x-handicapped run fails."""

    def run(*arguments):
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--repeats",
                "1",
                "--method",
                "sequential",
                "--output-dir",
                str(tmp_path / "out"),
                *arguments,
            ],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            check=False,
        )

    baseline_path = tmp_path / "baseline.json"
    recorded = run("--write-baseline", str(baseline_path))
    assert recorded.returncode == 0, recorded.stderr
    assert json.loads(baseline_path.read_text())["cases"]

    slowed = run("--baseline", str(baseline_path), "--handicap", "2")
    assert slowed.returncode == 1
    assert "slower" in slowed.stderr

    # An honest re-run with headroom for timer noise passes.
    rerun = run("--baseline", str(baseline_path), "--max-slowdown", "3")
    assert rerun.returncode == 0, rerun.stderr
