"""Link cost functions: BPR identities and the Beckmann objective.

The acceptance criterion of ws2-01 is
``test_beckmann_objective_matches_published_value``: the Beckmann objective
of the published best-known flows must reproduce
``datasets.BEST_KNOWN[name].objective`` to 1e-6 relative.
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from transport_flow_model import BPR, Network, beckmann_objective, datasets, link_costs


def published(name):
    """A network and its published best-known link flows, in link order.

    Flows are merged on ``edge_from``/``edge_to`` as
    ``tests/test_convergence.py`` does, after mapping split centroid nodes
    back to their TNTP zone ids — anaheim splits centroids, so 59 of its 914
    links would otherwise fail to match the published flow file.
    """
    instance = datasets.load_tntp(name)
    network = Network.from_tntp(instance)
    frame = network.to_dataframe()
    frame["edge_from"] = instance.zone_ids(frame["edge_from"])
    frame["edge_to"] = instance.zone_ids(frame["edge_to"])
    merged = frame.merge(
        datasets.best_known_flows(name),
        on=["edge_from", "edge_to"],
        suffixes=("_free", ""),
    )
    assert len(merged) == network.n_links
    return network, merged["flow"].to_numpy()


@pytest.fixture(scope="module")
def siouxfalls():
    return published("siouxfalls")


@pytest.fixture()
def fixed_cost_network():
    """No alpha/beta/capacity: costs do not depend on flow."""
    return Network.from_dataframe(
        pd.DataFrame(
            {
                "edge_from": [1, 1],
                "edge_to": [2, 2],
                "edge_id": [0, 1],
                "cost": [1.0, 2.5],
            }
        )
    )


@pytest.fixture()
def edge_case_network():
    """One ``beta == 0`` link, one zero-capacity link, one ordinary link."""
    return Network.from_dataframe(
        pd.DataFrame(
            {
                "edge_from": [1, 1, 1],
                "edge_to": [2, 2, 2],
                "edge_id": [0, 1, 2],
                "cost": [1.0, 2.0, 3.0],
                "capacity": [100.0, 0.0, 200.0],
                "alpha": [0.15, 0.15, 0.15],
                "beta": [0.0, 4.0, 4.0],
                "length": [1.0, 2.0, 3.0],
            }
        )
    )


@pytest.mark.parametrize("name", ["siouxfalls", "anaheim", "chicago-sketch"])
def test_beckmann_objective_matches_published_value(name):
    network, flows = published(name)
    best_known = datasets.BEST_KNOWN[name]
    objective = beckmann_objective(
        network, flows, distance_cost=best_known.distance_cost
    )
    assert objective == pytest.approx(best_known.objective, rel=1e-6)


def test_beckmann_objective_accepts_a_cost_function(siouxfalls):
    network, flows = siouxfalls
    cost_function = BPR.from_network(network)
    assert beckmann_objective(network, flows, cost_function=cost_function) == (
        beckmann_objective(network, flows)
    )


def test_beckmann_objective_accepts_a_flow_table(siouxfalls):
    network, flows = siouxfalls
    table = network.to_table().append_column("flow", pa.array(flows))
    assert beckmann_objective(network, table) == beckmann_objective(network, flows)


def test_derivative_matches_finite_difference(siouxfalls):
    network, flows = siouxfalls
    cost_function = BPR.from_network(network)
    step = 1e-4 * np.maximum(flows, 1.0)
    difference = (
        cost_function.travel_time(flows + step)
        - cost_function.travel_time(flows - step)
    ) / (2.0 * step)
    np.testing.assert_allclose(cost_function.derivative(flows), difference, rtol=1e-6)


def test_integral_matches_numerical_quadrature(siouxfalls):
    network, flows = siouxfalls
    cost_function = BPR.from_network(network)
    exact = cost_function.integral(flows)
    grid = np.linspace(0.0, 1.0, 20_001)
    for link in (0, 5, 17, 42, 75):
        # One link repeated over the quadrature grid, so a single vectorised
        # call evaluates t(w) at every w.
        along_grid = BPR(
            free_flow=np.full(grid.size, cost_function.free_flow[link]),
            capacity=np.full(grid.size, cost_function.capacity[link]),
            alpha=np.full(grid.size, cost_function.alpha[link]),
            beta=np.full(grid.size, cost_function.beta[link]),
            distance_term=np.full(grid.size, cost_function.distance_term[link]),
        )
        w = grid * flows[link]
        quadrature = np.trapezoid(along_grid.travel_time(w), w)
        assert quadrature == pytest.approx(exact[link], rel=1e-6)


def test_fixed_cost_network_is_flow_independent(fixed_cost_network):
    cost_function = BPR.from_network(fixed_cost_network)
    free_flow = np.array([1.0, 2.5])
    flows = np.array([0.0, 1234.5])
    assert np.array_equal(cost_function.travel_time(flows), free_flow)
    assert np.array_equal(cost_function.integral(flows), free_flow * flows)
    assert np.array_equal(cost_function.derivative(flows), np.zeros(2))


def test_zero_beta_and_zero_capacity_links(edge_case_network):
    cost_function = BPR.from_network(edge_case_network)
    flows = np.array([50.0, 50.0, 50.0])
    # beta == 0: constant cost * (1 + alpha), whatever the flow.
    # capacity == 0: ratio is taken as zero, so the link stays at free flow.
    np.testing.assert_allclose(
        cost_function.travel_time(flows),
        [1.0 * 1.15, 2.0, 3.0 * (1.0 + 0.15 * (50.0 / 200.0) ** 4)],
    )
    np.testing.assert_allclose(
        cost_function.integral(flows),
        [
            1.0 * 50.0 * 1.15,
            2.0 * 50.0,
            3.0 * 50.0 * (1.0 + 0.15 / 5.0 * (50.0 / 200.0) ** 4),
        ],
    )
    # beta == 0 has zero derivative; zero capacity has no derivative at all.
    derivative = cost_function.derivative(flows)
    assert derivative[0] == 0.0
    assert derivative[1] == 0.0
    assert derivative[2] > 0.0
    assert np.array_equal(cost_function.travel_time(0), [1.15, 2.0, 3.0])


def test_derivative_is_finite_at_zero_flow_for_small_beta():
    network = Network.from_dataframe(
        pd.DataFrame(
            {
                "edge_from": [1],
                "edge_to": [2],
                "edge_id": [0],
                "cost": [1.0],
                "capacity": [100.0],
                "alpha": [0.15],
                "beta": [0.5],
            }
        )
    )
    derivative = BPR.from_network(network).derivative(0.0)
    assert np.array_equal(derivative, [0.0])
    assert np.isfinite(BPR.from_network(network).derivative([1e-12])).all()


def test_scalar_flow_broadcasts(siouxfalls):
    network, _ = siouxfalls
    cost_function = BPR.from_network(network)
    free_flow = network.attribute("cost").to_numpy(zero_copy_only=False)
    np.testing.assert_array_equal(cost_function.travel_time(0), free_flow)
    np.testing.assert_array_equal(cost_function.travel_time(0.0), free_flow)
    assert np.array_equal(cost_function.integral(0), np.zeros(network.n_links))


def test_distance_term_adds_generalized_cost(edge_case_network):
    plain = BPR.from_network(edge_case_network)
    priced = BPR.from_network(edge_case_network, distance_cost=0.04)
    flows = np.array([10.0, 20.0, 30.0])
    length = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        priced.travel_time(flows), plain.travel_time(flows) + 0.04 * length
    )
    np.testing.assert_allclose(
        priced.integral(flows), plain.integral(flows) + 0.04 * length * flows
    )
    np.testing.assert_allclose(priced.derivative(flows), plain.derivative(flows))


def test_link_costs_delegates_to_bpr(siouxfalls):
    network, flows = siouxfalls
    assert np.array_equal(
        link_costs(network, flows), BPR.from_network(network).travel_time(flows)
    )
