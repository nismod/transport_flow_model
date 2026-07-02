import pandas as pd
import pytest

from transport_flow_model import rust


def test_rust_extension_availability_probe_returns_bool():
    assert isinstance(rust.is_available(), bool)


@pytest.mark.skipif(not rust.is_available(), reason="Rust extension is not built")
def test_rust_allocate_arrow_smoke():
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

    result = rust.allocate_arrow(network, od, directed=True)
    od_flows = result["od_flows"].to_pandas()
    network_flows = result["network_flows"].to_pandas()

    assert list(od_flows.loc[0, "edge_path"]) == ["AC", "CB"]
    assert od_flows.loc[0, "cost"] == 8.0
    assert network_flows.set_index("edge_id").loc["AC", "flow"] == 7.0


@pytest.mark.skipif(not rust.is_available(), reason="Rust extension is not built")
def test_rust_disrupt_arrow_smoke():
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

    result = rust.disrupt_arrow(network, od_flows, ["AB"], directed=True)
    rerouted_flows = result["rerouted_flows"].to_pandas()
    losses = result["losses"].to_pandas()

    assert list(rerouted_flows.loc[0, "edge_path"]) == ["AC"]
    assert rerouted_flows.loc[0, "cost"] == 5.0
    assert losses.loc[0, "rerouting_loss"] == 3.0
