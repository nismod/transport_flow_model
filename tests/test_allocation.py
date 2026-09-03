from dataclasses import is_dataclass

import pandas as pd

from transport_flow_model import model
from transport_flow_model.model import OD, Network, NetworkFlows, ODFlows


def _edge_flow_map(network_flows):
    df = network_flows.to_dataframe()
    return df.set_index("edge_id")["flow"].to_dict()


def _assert_frame_equal_up_to_order(actual, expected, sort_by):
    actual = actual.sort_values(sort_by).reset_index(drop=True)
    expected = expected.sort_values(sort_by).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected)


def test_allocation_result_is_named_dataclass_with_public_fields():
    assert is_dataclass(model.AllocationResult)
    assert set(model.AllocationResult.__dataclass_fields__) == {
        "od_flows",
        "network_flows",
        "unassigned_od",
    }


def test_allocate_single_od_to_least_cost_path_and_network_flows():
    network = Network(
        pd.DataFrame(
            {
                "edge_from": ["A", "A", "C"],
                "edge_to": ["B", "C", "B"],
                "edge_id": ["AB", "AC", "CB"],
                "cost": [10, 3, 5],
                "capacity": [100, 100, 100],
            }
        )
    )
    od = OD(
        pd.DataFrame(
            {
                "origin_id": ["A"],
                "destination_id": ["B"],
                "flow": [7],
            }
        )
    )

    result = network.allocate(od, directed=True)

    od_flows = result.od_flows.to_dataframe()
    assert len(od_flows) == 1
    assert od_flows.loc[0, "origin_id"] == "A"
    assert od_flows.loc[0, "destination_id"] == "B"
    assert od_flows.loc[0, "flow"] == 7
    assert od_flows.loc[0, "edge_path"] == ["AC", "CB"]
    assert od_flows.loc[0, "cost"] == 8
    assert result.unassigned_od.to_dataframe().empty
    assert _edge_flow_map(result.network_flows) == {"AB": 0, "AC": 7, "CB": 7}


def test_allocate_multiple_od_pairs_aggregates_shared_edge_without_capacity_limits():
    network = Network(
        pd.DataFrame(
            {
                "edge_from": ["A", "B", "A", "C", "B"],
                "edge_to": ["B", "C", "C", "D", "D"],
                "edge_id": ["AB", "BC", "AC", "CD", "BD"],
                "cost": [1, 2, 5, 1, 10],
                "capacity": [100, 100, 100, 100, 100],
            }
        )
    )
    od = OD(
        pd.DataFrame(
            {
                "origin_id": ["A", "B"],
                "destination_id": ["C", "D"],
                "flow": [10, 6],
            }
        )
    )

    result = network.allocate(od, directed=True)

    expected = pd.DataFrame(
        {
            "origin_id": ["A", "B"],
            "destination_id": ["C", "D"],
            "flow": [10, 6],
            "edge_path": [["AB", "BC"], ["BC", "CD"]],
            "cost": [3, 3],
        }
    )
    _assert_frame_equal_up_to_order(
        result.od_flows.to_dataframe(), expected, ["origin_id", "destination_id"]
    )
    assert result.unassigned_od.to_dataframe().empty
    assert _edge_flow_map(result.network_flows) == {
        "AB": 10,
        "BC": 16,
        "AC": 0,
        "CD": 6,
        "BD": 0,
    }


def test_capacity_constrained_allocation_shares_bottleneck_fairly():
    network = Network(
        pd.DataFrame(
            {
                "edge_from": ["A", "B", "C"],
                "edge_to": ["C", "C", "D"],
                "edge_id": ["AC", "BC", "CD"],
                "cost": [1, 1, 1],
                "capacity": [100, 100, 10],
            }
        )
    )
    od = OD(
        pd.DataFrame(
            {
                "origin_id": ["A", "B"],
                "destination_id": ["D", "D"],
                "flow": [10, 10],
            }
        )
    )

    result = network.allocate(od, capacity_constrained=True, directed=True)

    expected_od_flows = pd.DataFrame(
        {
            "origin_id": ["A", "B"],
            "destination_id": ["D", "D"],
            "flow": [5, 5],
            "edge_path": [["AC", "CD"], ["BC", "CD"]],
            "cost": [2, 2],
        }
    )
    expected_unassigned = pd.DataFrame(
        {
            "origin_id": ["A", "B"],
            "destination_id": ["D", "D"],
            "flow": [5, 5],
        }
    )

    _assert_frame_equal_up_to_order(
        result.od_flows.to_dataframe(),
        expected_od_flows,
        ["origin_id", "destination_id"],
    )
    _assert_frame_equal_up_to_order(
        result.unassigned_od.to_dataframe(),
        expected_unassigned,
        ["origin_id", "destination_id"],
    )
    assert _edge_flow_map(result.network_flows) == {"AC": 5, "BC": 5, "CD": 10}


def test_capacity_constrained_allocation_records_single_flow_residual():
    network = Network(
        pd.DataFrame(
            {
                "edge_from": ["A"],
                "edge_to": ["B"],
                "edge_id": ["AB"],
                "cost": [1],
                "capacity": [6],
            }
        )
    )
    od = OD(
        pd.DataFrame(
            {
                "origin_id": ["A"],
                "destination_id": ["B"],
                "flow": [10],
            }
        )
    )

    result = network.allocate(od, capacity_constrained=True, directed=True)

    expected_od_flows = pd.DataFrame(
        {
            "origin_id": ["A"],
            "destination_id": ["B"],
            "flow": [6],
            "edge_path": [["AB"]],
            "cost": [1],
        }
    )
    expected_unassigned = pd.DataFrame(
        {
            "origin_id": ["A"],
            "destination_id": ["B"],
            "flow": [4],
        }
    )
    _assert_frame_equal_up_to_order(
        result.od_flows.to_dataframe(),
        expected_od_flows,
        ["origin_id", "destination_id"],
    )
    _assert_frame_equal_up_to_order(
        result.unassigned_od.to_dataframe(),
        expected_unassigned,
        ["origin_id", "destination_id"],
    )
    assert _edge_flow_map(result.network_flows) == {"AB": 6}


def test_capacity_constrained_allocation_network_flows_include_existing_loads():
    network = Network(
        pd.DataFrame(
            {
                "edge_from": ["A"],
                "edge_to": ["B"],
                "edge_id": ["AB"],
                "cost": [1],
                "capacity": [10],
                "flow": [8],
            }
        )
    )
    od = OD(
        pd.DataFrame(
            {
                "origin_id": ["A"],
                "destination_id": ["B"],
                "flow": [2],
            }
        )
    )

    result = network.allocate(od, capacity_constrained=True, directed=True)

    assert _edge_flow_map(result.network_flows) == {"AB": 10}


def test_network_flows_from_network_and_od_flows_includes_unused_edges():
    network = Network(
        pd.DataFrame(
            {
                "edge_from": ["A", "B", "C"],
                "edge_to": ["B", "C", "D"],
                "edge_id": ["AB", "BC", "CD"],
                "cost": [1, 1, 1],
            }
        )
    )
    od_flows = ODFlows(
        pd.DataFrame(
            {
                "origin_id": ["A", "B"],
                "destination_id": ["C", "C"],
                "flow": [4, 3],
                "edge_path": [["AB", "BC"], ["BC"]],
                "cost": [2, 1],
            }
        )
    )

    network_flows = NetworkFlows.from_network_and_od_flows(network, od_flows)

    assert _edge_flow_map(network_flows) == {"AB": 4, "BC": 7, "CD": 0}


def test_network_flows_from_network_and_od_flows_adds_existing_loads():
    network = Network(
        pd.DataFrame(
            {
                "edge_from": ["A", "B"],
                "edge_to": ["B", "C"],
                "edge_id": ["AB", "BC"],
                "cost": [1, 1],
                "flow": [8, 0],
            }
        )
    )
    od_flows = ODFlows(
        pd.DataFrame(
            {
                "origin_id": ["A"],
                "destination_id": ["C"],
                "flow": [2],
                "edge_path": [["AB", "BC"]],
                "cost": [2],
            }
        )
    )

    network_flows = NetworkFlows.from_network_and_od_flows(network, od_flows)

    assert _edge_flow_map(network_flows) == {"AB": 10, "BC": 2}
