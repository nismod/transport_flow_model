import json

import pandas as pd
import pytest

from transport_flow_model import RunConfig, load_config
from transport_flow_model.config import AssignmentConfig


@pytest.fixture
def config_file(tmp_path):
    data_dir = tmp_path / "data"
    (data_dir / "network").mkdir(parents=True)
    (data_dir / "od").mkdir()
    (data_dir / "damages").mkdir()
    pd.DataFrame(
        {
            "from_id": ["A", "B"],
            "to_id": ["B", "C"],
            "id": ["E1", "E2"],
            "flow_capacity": [10, 10],
            "gcost_usd_per_ton": [1.0, 2.0],
            "length_m": [100, 200],
            "time_hr": [0.1, 0.2],
        }
    ).to_csv(data_dir / "network" / "network.csv", index=False)
    pd.DataFrame(
        {
            "origin_id": ["A"],
            "destination_id": ["C"],
            "tons": [5.0],
            "industry_A": [5.0],
        }
    ).to_csv(data_dir / "od" / "od.csv", index=False)
    pd.DataFrame({"edge_id": ["E1"]}).to_csv(
        data_dir / "damages" / "failure_set.csv", index=False
    )

    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "comment": "ignored",
                "paths": {
                    "data": str(data_dir),
                    "results": str(tmp_path / "results"),
                },
            }
        )
    )
    return path


def test_from_json_with_defaults(config_file):
    config = RunConfig.from_json(config_file)
    assert config.assignment == AssignmentConfig()
    assert config.assignment.method == "sequential"
    assert config.network.path.name == "network.csv"


def test_load_network(config_file):
    network = RunConfig.from_json(config_file).load_network()
    assert network.n_links == 2
    assert network.to_table().column_names[:3] == [
        "edge_from",
        "edge_to",
        "edge_id",
    ]


def test_load_demand(config_file):
    demand = RunConfig.from_json(config_file).load_demand()
    assert demand.n_pairs == 1
    assert demand.total == 5.0


def test_load_scenarios(config_file):
    scenarios = RunConfig.from_json(config_file).load_scenarios()
    assert [s.id for s in scenarios] == ["E1"]
    assert scenarios[0].removed_links == ("E1",)


def test_column_override(config_file, tmp_path):
    raw = json.loads(config_file.read_text())
    raw["demand"] = {
        "path": "od/od.csv",
        "columns": {
            "origin_id": "origin_id",
            "destination_id": "destination_id",
            "industry_A": "value",
        },
    }
    override = tmp_path / "override.json"
    override.write_text(json.dumps(raw))
    demand = RunConfig.from_json(override).load_demand()
    assert demand.total == 5.0


def test_repository_configs_validate():
    for name in ("config.example.json", "config.west_yorkshire.json"):
        config = RunConfig.from_json(name)
        assert config.paths.data.exists()


def test_load_config_deprecated(config_file):
    with pytest.warns(DeprecationWarning, match="RunConfig.from_json"):
        config = load_config(config_file)
    assert "paths" in config
