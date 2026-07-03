import pandas as pd
import pyarrow as pa
import pytest

from transport_flow_model import core


def test_extension_allocate():
    network = pd.DataFrame(
        {
            "edge_from": [0, 0, 2],
            "edge_to": [1, 2, 1],
            "edge_id": [0, 1, 2],
            "cost": [10.0, 3.0, 5.0],
            "capacity": [100.0, 100.0, 100.0],
        }
    )
    od = pd.DataFrame(
        {
            "origin_id": [0],
            "destination_id": [1],
            "flow": [7.0],
        }
    )

    result = core.allocate(network, od, directed=True)
    od_flows = result["od_flows"].to_pandas()
    network_flows = result["network_flows"].to_pandas()

    assert list(od_flows.loc[0, "edge_path"]) == [1, 2]
    assert od_flows.loc[0, "cost"] == 8.0
    assert network_flows.set_index("edge_id").loc[1, "flow"] == 7.0


def test_extension_disrupt():
    network = pd.DataFrame(
        {
            "edge_from": [0, 1, 0],
            "edge_to": [1, 2, 2],
            "edge_id": [0, 1, 2],
            "cost": [1.0, 1.0, 5.0],
            "capacity": [100.0, 100.0, 100.0],
        }
    )
    od_flows = pd.DataFrame(
        {
            "origin_id": [0],
            "destination_id": [2],
            "flow": [10.0],
            "edge_path": [[0, 1]],
            "cost": [2.0],
        }
    )

    result = core.disrupt(network, od_flows, [0], directed=True)
    rerouted_flows = result["rerouted_flows"].to_pandas()
    losses = result["losses"].to_pandas()

    assert list(rerouted_flows.loc[0, "edge_path"]) == [2]
    assert rerouted_flows.loc[0, "cost"] == 5.0
    assert losses.loc[0, "rerouting_loss"] == 3.0


def test_extension_allocate_string_ids():
    network = pd.DataFrame(
        {
            "edge_from": ["A", "A", "C"],
            "edge_to": ["B", "C", "B"],
            "edge_id": ["AB", "AC", "CB"],
            "cost": [10.0, 3.0, 5.0],
            "capacity": [100.0, 100.0, 100.0],
        }
    )
    od = pd.DataFrame(
        {
            "origin_id": ["A"],
            "destination_id": ["B"],
            "flow": [7.0],
        }
    )

    result = core.allocate(network, od, directed=True)
    od_flows = result["od_flows"].to_pandas()
    network_flows = result["network_flows"].to_pandas()

    assert od_flows.loc[0, "origin_id"] == "A"
    assert od_flows.loc[0, "destination_id"] == "B"
    assert list(od_flows.loc[0, "edge_path"]) == ["AC", "CB"]
    assert od_flows.loc[0, "cost"] == 8.0
    assert network_flows.set_index("edge_id").loc["AC", "flow"] == 7.0


def test_extension_disrupt_string_failed_edges():
    network = pd.DataFrame(
        {
            "edge_from": ["A", "B", "A"],
            "edge_to": ["B", "C", "C"],
            "edge_id": ["AB", "BC", "AC"],
            "cost": [1.0, 1.0, 5.0],
            "capacity": [100.0, 100.0, 100.0],
        }
    )
    od_flows = pd.DataFrame(
        {
            "origin_id": ["A"],
            "destination_id": ["C"],
            "flow": [10.0],
            "edge_path": [["AB", "BC"]],
            "cost": [2.0],
        }
    )

    result = core.disrupt(network, od_flows, ["AB"], directed=True)
    rerouted_flows = result["rerouted_flows"].to_pandas()
    losses = result["losses"].to_pandas()

    assert list(rerouted_flows.loc[0, "edge_path"]) == ["AC"]
    assert rerouted_flows.loc[0, "cost"] == 5.0
    assert losses.loc[0, "rerouting_loss"] == 3.0


def test_extension_allocate_unsigned_integer_ids():
    network = pa.table(
        {
            "edge_from": pa.array([10, 10, 12], type=pa.uint32()),
            "edge_to": pa.array([11, 12, 11], type=pa.uint32()),
            "edge_id": pa.array([100, 101, 102], type=pa.uint32()),
            "cost": pa.array([10.0, 3.0, 5.0]),
            "capacity": pa.array([100.0, 100.0, 100.0]),
        }
    )
    od = pa.table(
        {
            "origin_id": pa.array([10], type=pa.uint32()),
            "destination_id": pa.array([11], type=pa.uint32()),
            "flow": pa.array([7.0]),
        }
    )

    result = core.allocate(network, od, directed=True)
    od_flows = result["od_flows"].to_pandas()

    assert od_flows.loc[0, "origin_id"] == 10
    assert od_flows.loc[0, "destination_id"] == 11
    assert list(od_flows.loc[0, "edge_path"]) == [101, 102]


def test_extension_rejects_duplicate_edge_ids():
    network = pd.DataFrame(
        {
            "edge_from": ["A", "B"],
            "edge_to": ["B", "C"],
            "edge_id": ["AB", "AB"],
        }
    )
    od = pd.DataFrame({"origin_id": ["A"], "destination_id": ["C"], "flow": [1.0]})

    with pytest.raises(ValueError, match="duplicate edge_id"):
        core.allocate(network, od)


def test_extension_rejects_null_ids():
    network = pd.DataFrame(
        {
            "edge_from": ["A", None],
            "edge_to": ["B", "C"],
            "edge_id": ["AB", "BC"],
        }
    )
    od = pd.DataFrame({"origin_id": ["A"], "destination_id": ["C"], "flow": [1.0]})

    with pytest.raises(ValueError, match="cannot contain null id values"):
        core.allocate(network, od)


def test_extension_rejects_mismatched_id_types():
    network = pd.DataFrame(
        {
            "edge_from": ["A"],
            "edge_to": ["B"],
            "edge_id": ["AB"],
        }
    )
    od = pd.DataFrame({"origin_id": [1], "destination_id": [2], "flow": [1.0]})

    with pytest.raises(ValueError, match="origin_id id type"):
        core.allocate(network, od)


def test_extension_rejects_unknown_edge_path_ids():
    network = pd.DataFrame(
        {
            "edge_from": ["A", "B"],
            "edge_to": ["B", "C"],
            "edge_id": ["AB", "BC"],
        }
    )
    od_flows = pd.DataFrame(
        {
            "origin_id": ["A"],
            "destination_id": ["C"],
            "flow": [1.0],
            "edge_path": [["AB", "missing"]],
            "cost": [2.0],
        }
    )

    with pytest.raises(ValueError, match="unknown edge_id"):
        core.disrupt(network, od_flows, ["AB"])
