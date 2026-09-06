"""Relative gap: exact on hand-built networks, ~zero on published equilibria."""

import numpy as np
import pandas as pd
import pytest

from transport_flow_model import (
    BPR,
    Conical,
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


def test_link_costs_with_no_cost_function_matches_todays_bpr_result(siouxfalls):
    network, _ = siouxfalls
    flows = np.arange(network.n_links, dtype="float64")
    np.testing.assert_array_equal(
        link_costs(network, flows), BPR.from_network(network).travel_time(flows)
    )


def test_link_costs_accepts_an_explicit_cost_function(siouxfalls):
    network, _ = siouxfalls
    flows = np.arange(network.n_links, dtype="float64")
    cost_function = Conical.from_network(network)
    np.testing.assert_array_equal(
        link_costs(network, flows, cost_function=cost_function),
        cost_function.travel_time(flows),
    )


def test_relative_gap_with_no_cost_function_matches_todays_bpr_result(
    parallel_network, unit_demand
):
    flows = [4.0, 6.0]
    gap_default = relative_gap(parallel_network, unit_demand, flows)
    gap_explicit = relative_gap(
        parallel_network,
        unit_demand,
        flows,
        cost_function=BPR.from_network(parallel_network),
    )
    assert gap_default == gap_explicit


def test_relative_gap_accepts_an_explicit_cost_function():
    # A Conical cost function on a network that also carries alpha/beta:
    # the network *would* build a different (BPR) result if `cost_function`
    # were silently ignored and the default recomputed instead, so this
    # only passes if `cost_function` is genuinely used end to end.
    network = Network.from_dataframe(
        pd.DataFrame(
            {
                "edge_from": [1, 1],
                "edge_to": [2, 2],
                "edge_id": [0, 1],
                "cost": [1.0, 1.0],
                "capacity": [100.0, 10.0],
                "alpha": [0.15, 0.15],
                "beta": [4.0, 4.0],
            }
        )
    )
    demand = Demand.from_dataframe(
        pd.DataFrame({"origin_id": [1], "destination_id": [2], "value": [10.0]})
    )
    cost_function = Conical.from_network(network)
    flows = [6.0, 4.0]
    t = cost_function.travel_time(np.array(flows))
    total_cost = float(np.dot(t, flows))
    min_cost = float(demand.total * min(t))
    expected = (total_cost - min_cost) / min_cost
    assert relative_gap(
        network, demand, flows, cost_function=cost_function
    ) == pytest.approx(expected)


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
