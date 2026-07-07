"""Relative gap: exact on hand-built networks, ~zero on published equilibria."""

import numpy as np
import pandas as pd
import pytest

from transport_flow_model import (
    Demand,
    Network,
    assign,
    datasets,
    link_costs,
    relative_gap,
)


@pytest.fixture(scope="module")
def siouxfalls():
    instance = datasets.load_tntp("siouxfalls")
    return Network.from_tntp(instance), Demand.from_tntp(instance)


@pytest.fixture()
def parallel_network():
    """Two parallel fixed-cost links between nodes 1 and 2."""
    return Network.from_dataframe(
        pd.DataFrame(
            {
                "edge_from": [1, 1],
                "edge_to": [2, 2],
                "edge_id": [0, 1],
                "cost": [1.0, 2.0],
            }
        )
    )


@pytest.fixture()
def unit_demand():
    return Demand.from_dataframe(
        pd.DataFrame({"origin_id": [1], "destination_id": [2], "value": [10.0]})
    )


def test_gap_zero_on_least_cost_flows(parallel_network, unit_demand):
    assert relative_gap(parallel_network, unit_demand, [10.0, 0.0]) == 0.0


def test_gap_exact_on_costlier_flows(parallel_network, unit_demand):
    # TSTT = 10 * 2 = 20, SPTT = 10 * 1 = 10, gap = (20 - 10) / 10
    assert relative_gap(parallel_network, unit_demand, [0.0, 10.0]) == 1.0


def test_gap_accepts_assignment_result(parallel_network, unit_demand):
    result = assign(parallel_network, unit_demand, method="sequential")
    assert relative_gap(parallel_network, unit_demand, result) == 0.0
    assert relative_gap(parallel_network, unit_demand, result.link_flows) == 0.0


def test_unreachable_destination_raises(parallel_network):
    demand = Demand.from_dataframe(
        pd.DataFrame({"origin_id": [2], "destination_id": [1], "value": [1.0]})
    )
    with pytest.raises(ValueError, match="unreachable"):
        relative_gap(parallel_network, demand, [0.0, 0.0])


def test_zero_demand_raises(parallel_network):
    demand = Demand.from_dataframe(
        pd.DataFrame({"origin_id": [1], "destination_id": [2], "value": [0.0]})
    )
    with pytest.raises(ValueError, match="zero"):
        relative_gap(parallel_network, demand, [0.0, 0.0])


def test_wrong_flow_length_raises(parallel_network, unit_demand):
    with pytest.raises(ValueError, match="shape"):
        relative_gap(parallel_network, unit_demand, [1.0])


def test_bpr_link_costs_match_published_costs(siouxfalls):
    network, _ = siouxfalls
    published = datasets.best_known_flows("siouxfalls")
    merged = network.to_dataframe().merge(
        published, on=["edge_from", "edge_to"], suffixes=("_free", "")
    )
    costs = link_costs(network, merged["flow"].to_numpy())
    np.testing.assert_allclose(costs, merged["cost"].to_numpy(), rtol=1e-12)


def test_published_siouxfalls_flows_are_at_equilibrium(siouxfalls):
    network, demand = siouxfalls
    published = datasets.best_known_flows("siouxfalls")
    merged = network.to_dataframe().merge(
        published, on=["edge_from", "edge_to"], suffixes=("_free", "")
    )
    gap = relative_gap(network, demand, merged["flow"].to_numpy())
    assert abs(gap) < 1e-10


def test_all_or_nothing_flows_are_far_from_equilibrium(siouxfalls):
    network, demand = siouxfalls
    result = assign(network, demand, method="sequential")
    gap = relative_gap(network, demand, result)
    assert gap > 0.1
