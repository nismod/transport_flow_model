from dataclasses import is_dataclass

import pandas as pd

from transport_flow_model import model
from transport_flow_model.model import OD, Network, ODFlows


def _edge_flow_map(network_flows):
    df = network_flows.to_dataframe()
    return df.set_index("edge_id")["flow"].to_dict()


def _assert_frame_equal_up_to_order(actual, expected, sort_by):
    actual = actual.sort_values(sort_by).reset_index(drop=True)
    expected = expected.sort_values(sort_by).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected)


def test_disruption_result_is_named_dataclass_with_public_fields():
    assert is_dataclass(model.DisruptionResult)
    assert set(model.DisruptionResult.__dataclass_fields__) == {
        "rerouted_flows",
        "network_flows",
        "isolated_od",
        "losses",
    }


def test_disruption_isolates_single_path_od_when_failed_edge_removes_access():
    network = Network(
        pd.DataFrame(
            {
                "edge_from": ["A", "B"],
                "edge_to": ["B", "C"],
                "edge_id": ["AB", "BC"],
                "cost": [1, 1],
                "capacity": [100, 100],
            }
        )
    )
    existing_flows = ODFlows(
        pd.DataFrame(
            {
                "origin_id": ["A"],
                "destination_id": ["C"],
                "flow": [10],
                "edge_path": [["AB", "BC"]],
                "cost": [2],
            }
        )
    )

    result = network.disrupt(existing_flows, ["AB"], directed=True)

    assert result.rerouted_flows.to_dataframe().empty
    _assert_frame_equal_up_to_order(
        result.isolated_od.to_dataframe(),
        pd.DataFrame(
            {
                "origin_id": ["A"],
                "destination_id": ["C"],
                "flow": [10],
            }
        ),
        ["origin_id", "destination_id"],
    )
    assert isinstance(result.losses, OD)


def test_disruption_reroutes_failed_short_path_to_remaining_path_and_records_loss():
    network = Network(
        pd.DataFrame(
            {
                "edge_from": ["A", "B", "A"],
                "edge_to": ["B", "C", "C"],
                "edge_id": ["AB", "BC", "AC"],
                "cost": [1, 1, 5],
                "capacity": [100, 100, 100],
            }
        )
    )
    existing_flows = ODFlows(
        pd.DataFrame(
            {
                "origin_id": ["A"],
                "destination_id": ["C"],
                "flow": [10],
                "edge_path": [["AB", "BC"]],
                "cost": [2],
            }
        )
    )

    result = network.disrupt(existing_flows, ["AB"], directed=True)

    _assert_frame_equal_up_to_order(
        result.rerouted_flows.to_dataframe(),
        pd.DataFrame(
            {
                "origin_id": ["A"],
                "destination_id": ["C"],
                "flow": [10],
                "edge_path": [["AC"]],
                "cost": [5],
            }
        ),
        ["origin_id", "destination_id"],
    )
    assert result.isolated_od.to_dataframe().empty
    _assert_frame_equal_up_to_order(
        result.losses.to_dataframe(),
        pd.DataFrame(
            {
                "origin_id": ["A"],
                "destination_id": ["C"],
                "flow": [10],
                "initial_cost": [2],
                "disrupted_cost": [5],
                "rerouting_loss": [3],
            }
        ),
        ["origin_id", "destination_id"],
    )
    assert _edge_flow_map(result.network_flows) == {"AB": 0, "BC": 0, "AC": 10}


def test_disruption_network_flows_keep_unaffected_od_paths():
    network = Network(
        pd.DataFrame(
            {
                "edge_from": ["A", "B", "A", "C"],
                "edge_to": ["B", "C", "C", "D"],
                "edge_id": ["AB", "BC", "AC", "CD"],
                "cost": [1, 1, 5, 1],
                "capacity": [100, 100, 100, 100],
            }
        )
    )
    existing_flows = ODFlows(
        pd.DataFrame(
            {
                "origin_id": ["A", "C"],
                "destination_id": ["C", "D"],
                "flow": [10, 4],
                "edge_path": [["AB", "BC"], ["CD"]],
                "cost": [2, 1],
            }
        )
    )

    result = network.disrupt(existing_flows, ["AB"], directed=True)

    assert _edge_flow_map(result.network_flows) == {
        "AB": 0,
        "BC": 0,
        "AC": 10,
        "CD": 4,
    }


def test_disruption_network_flows_keep_all_paths_when_failure_affects_no_od():
    network = Network(
        pd.DataFrame(
            {
                "edge_from": ["A", "C"],
                "edge_to": ["B", "D"],
                "edge_id": ["AB", "CD"],
                "cost": [1, 1],
                "capacity": [100, 100],
            }
        )
    )
    existing_flows = ODFlows(
        pd.DataFrame(
            {
                "origin_id": ["C"],
                "destination_id": ["D"],
                "flow": [4],
                "edge_path": [["CD"]],
                "cost": [1],
            }
        )
    )

    result = network.disrupt(existing_flows, ["AB"], directed=True)

    assert result.rerouted_flows.to_dataframe().empty
    assert _edge_flow_map(result.network_flows) == {"AB": 0, "CD": 4}


def test_capacity_constrained_disruption_isolates_residual_flow_after_rerouting():
    network = Network(
        pd.DataFrame(
            {
                "edge_from": ["A", "B", "A", "D"],
                "edge_to": ["B", "C", "D", "C"],
                "edge_id": ["AB", "BC", "AD", "DC"],
                "cost": [1, 1, 3, 3],
                "capacity": [15, 15, 10, 10],
            }
        )
    )
    existing_flows = ODFlows(
        pd.DataFrame(
            {
                "origin_id": ["A"],
                "destination_id": ["C"],
                "flow": [15],
                "edge_path": [["AB", "BC"]],
                "cost": [2],
            }
        )
    )

    result = network.disrupt(existing_flows, ["AB"], directed=True)

    _assert_frame_equal_up_to_order(
        result.rerouted_flows.to_dataframe(),
        pd.DataFrame(
            {
                "origin_id": ["A"],
                "destination_id": ["C"],
                "flow": [10],
                "edge_path": [["AD", "DC"]],
                "cost": [6],
            }
        ),
        ["origin_id", "destination_id"],
    )
    _assert_frame_equal_up_to_order(
        result.isolated_od.to_dataframe(),
        pd.DataFrame(
            {
                "origin_id": ["A"],
                "destination_id": ["C"],
                "flow": [5],
            }
        ),
        ["origin_id", "destination_id"],
    )
    _assert_frame_equal_up_to_order(
        result.losses.to_dataframe(),
        pd.DataFrame(
            {
                "origin_id": ["A"],
                "destination_id": ["C"],
                "flow": [10],
                "initial_cost": [2],
                "disrupted_cost": [6],
                "rerouting_loss": [4],
            }
        ),
        ["origin_id", "destination_id"],
    )
    assert _edge_flow_map(result.network_flows) == {
        "AB": 0,
        "BC": 0,
        "AD": 10,
        "DC": 10,
    }
