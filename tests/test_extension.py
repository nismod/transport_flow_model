import pandas as pd

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
