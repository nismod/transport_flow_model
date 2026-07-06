"""End-to-end regression: the flow scripts reproduce the expected example
results exactly when driven through the public API.

The fixtures in tests/data/example_results were produced by the original
script-shaped implementations on the example dataset, so this also pins
the byte-for-byte compatibility of the API drivers with the legacy
pipeline."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts" / "flow_model"
RESULT_FILES = (
    "flow_od_paths/network_edge_total_flows.csv",
    "flow_od_paths/od_flows.csv",
    "flow_od_paths/unassigned_od_flows.csv",
    "flow_disruptions/flow_disruption_losses.csv",
)


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("example_run")
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "paths": {
                    "data": str(REPO_ROOT / "example" / "processed_data"),
                    "results": str(tmp_path / "results"),
                }
            }
        )
    )
    for script in ("flow_allocation.py", "flow_disruptions.py"):
        subprocess.run(
            [sys.executable, str(SCRIPTS / script), str(config_path)],
            check=True,
            cwd=REPO_ROOT,
            capture_output=True,
        )
    return tmp_path / "results"


@pytest.mark.parametrize("result_file", RESULT_FILES)
def test_scripts_reproduce_committed_results(results, result_file):
    expected = (
        REPO_ROOT / "tests" / "data" / "example_results" / result_file
    ).read_text()
    actual = (results / result_file).read_text()
    assert actual == expected
