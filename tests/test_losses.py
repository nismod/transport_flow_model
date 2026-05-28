import pandas as pd

from transport_flow_model.model import OD, ODFlows


def test_losses_from_flows_groups_costs_by_od_pair():
    initial = ODFlows(
        pd.DataFrame(
            {
                "origin_id": ["A", "A", "B"],
                "destination_id": ["C", "C", "D"],
                "flow": [4, 6, 2],
                "edge_path": [["AC1"], ["AC2"], ["BD"]],
                "cost": [2, 3, 7],
            }
        )
    )
    disrupted = ODFlows(
        pd.DataFrame(
            {
                "origin_id": ["A", "A", "B"],
                "destination_id": ["C", "C", "D"],
                "flow": [4, 6, 2],
                # assume AC2 and BD were disrupted with alternate paths available
                "edge_path": [
                    ["AC1"],
                    ["AE", "EC"],
                    ["BE", "ED"],
                ],
                "cost": [2, 5, 12],
            }
        )
    )

    losses = OD.losses_from_flows(initial, disrupted)

    expected = pd.DataFrame(
        {
            "origin_id": ["A", "B"],
            "destination_id": ["C", "D"],
            "flow": [10, 2],
            "initial_cost": [5, 7],
            "disrupted_cost": [7, 12],
            "rerouting_loss": [2, 5],
        }
    )
    actual = losses.to_dataframe()
    assert set(expected.columns).issubset(actual.columns)
    actual = actual.loc[:, expected.columns].sort_values(
        ["origin_id", "destination_id"]
    )
    actual = actual.reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
