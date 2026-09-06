import json

import pandas as pd
import pytest

from transport_flow_model import (
    Conical,
    ConvergenceWarning,
    Demand,
    Network,
    RunConfig,
    assign,
    load_config,
)
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


# --- AssignmentConfig.options() ---------------------------------------------


@pytest.fixture()
def two_link_network():
    """Two BPR links between the same OD pair, asymmetric enough that
    equilibrating them is not trivial: MSA needs more than its default
    50 passes to reach its default 1e-4 target on this network."""
    return Network.from_dataframe(
        pd.DataFrame(
            {
                "edge_from": [1, 1],
                "edge_to": [2, 2],
                "edge_id": [0, 1],
                "cost": [1.0, 2.0],
                "capacity": [50.0, 150.0],
                "alpha": [0.15, 0.15],
                "beta": [4.0, 4.0],
            }
        )
    )


@pytest.fixture()
def two_link_demand():
    return Demand.from_dataframe(
        pd.DataFrame({"origin_id": [1], "destination_id": [2], "value": [500.0]})
    )


def test_options_includes_capacity_constrained_and_directed_for_sequential():
    config = AssignmentConfig()
    assert config.options() == {"capacity_constrained": True, "directed": True}


def test_options_drops_capacity_constrained_for_msa():
    # Regression test: _assign_msa declares neither `capacity_constrained`
    # nor **kwargs, so passing it used to raise TypeError from assign().
    config = AssignmentConfig(method="msa")
    assert config.options() == {"directed": True}


def test_options_raises_on_an_option_the_method_cannot_accept(two_link_network):
    """Dropping a *default* is housekeeping; dropping an instruction is not.

    Silently ignoring a `cost_function` the config asked for would assign
    the run with BPR and leave nothing in the results to say the requested
    curve never reached the backend.
    """
    asked_for_a_curve = AssignmentConfig.model_validate(
        {"method": "sequential", "cost_function": {"name": "conical"}}
    )
    with pytest.raises(ValueError, match="does not accept 'cost_function'"):
        asked_for_a_curve.options(two_link_network)

    explicit = AssignmentConfig.model_validate(
        {"method": "msa", "capacity_constrained": True}
    )
    with pytest.raises(ValueError, match="does not accept 'capacity_constrained'"):
        explicit.options(two_link_network)

    # The same value left at its default is dropped without complaint --
    # that is the whole point of reading the backend's signature.
    assert AssignmentConfig(method="msa").options() == {"directed": True}


def test_msa_config_runs_end_to_end(two_link_network, two_link_demand):
    config = AssignmentConfig(method="msa")
    with pytest.warns(ConvergenceWarning):
        result = assign(
            two_link_network,
            two_link_demand,
            config.method,
            **config.options(two_link_network),
        )
    assert result.link_flows.num_rows == two_link_network.n_links


def test_method_options_are_passed_through_unfiltered(
    two_link_network, two_link_demand
):
    config = AssignmentConfig(method="msa", method_options={"max_iterations": 2})
    with pytest.warns(ConvergenceWarning):
        result = assign(
            two_link_network,
            two_link_demand,
            config.method,
            **config.options(two_link_network),
        )
    assert result.provenance.iterations == 2


def test_cost_function_config_matches_building_it_by_hand(
    two_link_network, two_link_demand
):
    # "sequential" does not accept a cost_function (it has no notion of
    # congestion), so this needs a method that does.
    config = AssignmentConfig(
        method="msa", cost_function={"name": "conical", "alpha": 5.0}
    )
    options = config.options(two_link_network)
    assert options["cost_function"].travel_time([10.0, 10.0]) == pytest.approx(
        Conical.from_network(two_link_network, alpha=5.0).travel_time([10.0, 10.0])
    )


def test_cost_function_config_without_a_network_raises():
    config = AssignmentConfig(cost_function={"name": "bpr"})
    with pytest.raises(ValueError, match="network"):
        config.options()
