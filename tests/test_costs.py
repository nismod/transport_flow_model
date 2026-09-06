"""Link cost functions: BPR identities and the Beckmann objective.

The acceptance criterion of ws2-01 is
``test_beckmann_objective_matches_published_value``: the Beckmann objective
of the published best-known flows must reproduce
``datasets.BEST_KNOWN[name].objective`` to 1e-6 relative.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from transport_flow_model import BPR, Network, beckmann_objective, datasets, link_costs
from transport_flow_model.costs import (
    COST_FUNCTIONS,
    Conical,
    CostFunction,
    SpeedFlow,
    build_cost_function,
)

DATA = Path(__file__).parent / "data"


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


def test_bpr_and_conical_satisfy_the_cost_function_protocol(siouxfalls):
    network, _ = siouxfalls
    assert isinstance(BPR.from_network(network), CostFunction)
    assert isinstance(Conical.from_network(network), CostFunction)


# --- Cost function registry -------------------------------------------------


def test_build_cost_function_bpr_matches_from_network(siouxfalls):
    network, flows = siouxfalls
    built = build_cost_function("bpr", network, distance_cost=0.04)
    expected = BPR.from_network(network, distance_cost=0.04)
    np.testing.assert_array_equal(built.travel_time(flows), expected.travel_time(flows))


def test_build_cost_function_conical_forwards_params(siouxfalls):
    network, flows = siouxfalls
    built = build_cost_function("conical", network, alpha=5.0)
    expected = Conical.from_network(network, alpha=5.0)
    np.testing.assert_array_equal(built.travel_time(flows), expected.travel_time(flows))


def test_build_cost_function_unknown_name_raises(siouxfalls):
    network, _ = siouxfalls
    with pytest.raises(ValueError, match="nonexistent") as excinfo:
        build_cost_function("nonexistent", network)
    # The message names every registered function.
    for name in COST_FUNCTIONS:
        assert name in str(excinfo.value)


def test_build_cost_function_rejects_an_unknown_parameter(siouxfalls):
    network, _ = siouxfalls
    with pytest.raises(TypeError):
        build_cost_function("bpr", network, not_a_real_parameter=1.0)
    with pytest.raises(TypeError):
        build_cost_function("conical", network, not_a_real_parameter=1.0)


def test_build_cost_function_speed_flow_requires_curves(speed_flow_network):
    # Distinct from SpeedFlow.from_table's own "curves must be a pyarrow
    # Table..." TypeError (also raised on a bad curves value): this is
    # build_cost_function's own check for a missing 'curves' keyword.
    with pytest.raises(TypeError, match="requires a 'curves' keyword"):
        build_cost_function("speed_flow", speed_flow_network)


def test_build_cost_function_speed_flow_matches_from_table(
    speed_flow_curves, speed_flow_network
):
    built = build_cost_function(
        "speed_flow", speed_flow_network, curves=speed_flow_curves, min_speed=8.0
    )
    expected = SpeedFlow.from_table(
        speed_flow_curves, speed_flow_network, min_speed=8.0
    )
    flows = np.array([600.0, 400.0])
    np.testing.assert_array_equal(built.travel_time(flows), expected.travel_time(flows))


# --- Conical (Spiess 1990) -------------------------------------------------


def test_conical_c_anchors_at_zero_and_full_capacity():
    # A single link with free_flow=1 so travel_time(x) reads off C(v)
    # directly. C(0) is 1.0000000000000002, not exactly 1 -- see ws2-01.
    network = Network.from_dataframe(
        pd.DataFrame(
            {
                "edge_from": [1],
                "edge_to": [2],
                "edge_id": [0],
                "cost": [1.0],
                "capacity": [1500.0],
            }
        )
    )
    cost_function = Conical.from_network(network, alpha=4.0)
    assert cost_function.travel_time(0.0)[0] == pytest.approx(1.0)
    assert cost_function.travel_time([1500.0])[0] == pytest.approx(2.0)


def test_conical_integral_matches_numerical_quadrature(siouxfalls):
    network, flows = siouxfalls
    cost_function = Conical.from_network(network)
    exact = cost_function.integral(flows)
    grid = np.linspace(0.0, 1.0, 20_001)
    for link in (0, 5, 17, 42, 75):
        along_grid = Conical(
            free_flow=np.full(grid.size, cost_function.free_flow[link]),
            capacity=np.full(grid.size, cost_function.capacity[link]),
            alpha=np.full(grid.size, cost_function.alpha[link]),
            distance_term=np.full(grid.size, cost_function.distance_term[link]),
        )
        w = grid * flows[link]
        quadrature = np.trapezoid(along_grid.travel_time(w), w)
        assert quadrature == pytest.approx(exact[link], rel=1e-6)


def test_conical_derivative_matches_finite_difference(siouxfalls):
    network, flows = siouxfalls
    cost_function = Conical.from_network(network)
    step = 1e-4 * np.maximum(flows, 1.0)
    difference = (
        cost_function.travel_time(flows + step)
        - cost_function.travel_time(flows - step)
    ) / (2.0 * step)
    np.testing.assert_allclose(cost_function.derivative(flows), difference, rtol=1e-6)


def test_conical_alpha_must_exceed_one(siouxfalls):
    network, _ = siouxfalls
    with pytest.raises(ValueError, match="alpha"):
        Conical.from_network(network, alpha=1.0)
    with pytest.raises(ValueError, match="alpha"):
        Conical.from_network(network, alpha=0.5)
    # Per-link array: one bad value among many good ones still raises.
    n = network.n_links
    mixed = np.full(n, 4.0)
    mixed[0] = 1.0
    with pytest.raises(ValueError, match="alpha"):
        Conical.from_network(network, alpha=mixed)


def test_conical_from_network_does_not_read_alpha_link_attribute():
    # A TNTP-shaped network with a BPR `alpha` attribute of 0.15, which
    # would make every alpha <= 1 if it were read as the conical alpha.
    network = Network.from_dataframe(
        pd.DataFrame(
            {
                "edge_from": [1],
                "edge_to": [2],
                "edge_id": [0],
                "cost": [1.0],
                "capacity": [1500.0],
                "alpha": [0.15],
                "beta": [4.0],
            }
        )
    )
    cost_function = Conical.from_network(network)
    np.testing.assert_array_equal(cost_function.alpha, [4.0])


def test_conical_capacity_le_zero_is_uncongested():
    network = Network.from_dataframe(
        pd.DataFrame(
            {
                "edge_from": [1, 1],
                "edge_to": [2, 2],
                "edge_id": [0, 1],
                "cost": [1.0, 3.0],
                "capacity": [0.0, 200.0],
            }
        )
    )
    cost_function = Conical.from_network(network, alpha=4.0)
    flows = np.array([50.0, 50.0])
    constant = cost_function.travel_time(np.array([0.0, 0.0]))[0]
    # The zero-capacity link's cost does not depend on flow at all.
    np.testing.assert_allclose(cost_function.travel_time(flows)[0], constant)
    np.testing.assert_allclose(cost_function.travel_time([1e6, 50.0])[0], constant)
    np.testing.assert_allclose(cost_function.integral(flows)[0], constant * 50.0)
    assert cost_function.derivative(flows)[0] == 0.0
    assert cost_function.derivative(flows)[1] > 0.0


# --- SpeedFlow (DfT-style piecewise-linear speed-flow curve) ---------------


@pytest.fixture()
def single_link_speed_flow():
    """One synthetic curve: 4 breakpoints, falling speed, one link.

    Flow-per-lane 0/300/600/900, speed 100/90/40/10; length 5, 2 lanes, a
    floor of 8. Numbers are invented for this test, not a real curve.
    """
    return SpeedFlow(
        length=np.array([5.0]),
        lanes=np.array([2.0]),
        flow=np.array([[0.0, 300.0, 600.0, 900.0]]),
        speed=np.array([[100.0, 90.0, 40.0, 10.0]]),
        min_speed=8.0,
    )


@pytest.fixture()
def multi_link_speed_flow():
    """The same synthetic curve applied to three links with different
    lengths and lane counts, for the vectorised (quadrature/derivative)
    tests."""
    return SpeedFlow(
        length=np.array([5.0, 5.0, 2.0]),
        lanes=np.array([2.0, 2.0, 1.0]),
        flow=np.tile([0.0, 300.0, 600.0, 900.0], (3, 1)),
        speed=np.tile([100.0, 90.0, 40.0, 10.0], (3, 1)),
        min_speed=8.0,
    )


def test_speed_flow_satisfies_the_cost_function_protocol(single_link_speed_flow):
    assert isinstance(single_link_speed_flow, CostFunction)


def test_speed_flow_integral_matches_numerical_quadrature(multi_link_speed_flow):
    flows = np.array([50.0, 1200.0, 3000.0])
    exact = multi_link_speed_flow.integral(flows)
    grid = np.linspace(0.0, 1.0, 20_001)
    for link in range(multi_link_speed_flow.n_links):
        along_grid = SpeedFlow(
            length=np.full(grid.size, multi_link_speed_flow.length[link]),
            lanes=np.full(grid.size, multi_link_speed_flow.lanes[link]),
            flow=np.tile(multi_link_speed_flow.flow[link], (grid.size, 1)),
            speed=np.tile(multi_link_speed_flow.speed[link], (grid.size, 1)),
            min_speed=multi_link_speed_flow.min_speed,
        )
        w = grid * flows[link]
        quadrature = np.trapezoid(along_grid.travel_time(w), w)
        assert quadrature == pytest.approx(exact[link], rel=1e-6)


def test_speed_flow_derivative_matches_finite_difference(multi_link_speed_flow):
    # Flows chosen away from exact breakpoints (x = 0, 600, 1200, 1800 for
    # the lanes=2 links; x = 0, 300, 600, 900 for the lanes=1 link): the
    # curve has a kink at each one, where a central difference straddling
    # it does not agree with either one-sided derivative.
    flows = np.array([50.0, 1000.0, 700.0])
    step = 1e-4 * np.maximum(flows, 1.0)
    difference = (
        multi_link_speed_flow.travel_time(flows + step)
        - multi_link_speed_flow.travel_time(flows - step)
    ) / (2.0 * step)
    np.testing.assert_allclose(
        multi_link_speed_flow.derivative(flows), difference, rtol=1e-6
    )


def test_speed_flow_clamps_speed_at_the_floor(single_link_speed_flow):
    # q = 3000/2 = 1500, far past the last breakpoint (900): the raw
    # extrapolated line has gone negative, so travel_time is pinned to
    # length / min_speed and the marginal cost of more flow is zero.
    flows = np.array([3000.0])
    np.testing.assert_allclose(single_link_speed_flow.travel_time(flows), [5.0 / 8.0])
    assert single_link_speed_flow.derivative(flows)[0] == 0.0


def test_speed_flow_extrapolates_the_final_segment_beyond_the_last_breakpoint(
    single_link_speed_flow,
):
    # q = 1820/2 = 910, past the last breakpoint (900) but not yet clamped:
    # speed continues along the (900, 10) -> (600, 40) segment's slope,
    # -0.1, giving 10 - 0.1*10 = 9, not the floor of 8.
    flows = np.array([1820.0])
    np.testing.assert_allclose(single_link_speed_flow.travel_time(flows), [5.0 / 9.0])
    assert single_link_speed_flow.derivative(flows)[0] > 0.0


def test_speed_flow_travel_time_and_integral_are_continuous_at_a_breakpoint(
    single_link_speed_flow,
):
    # x = 1200 puts q = 600 exactly on the third breakpoint; values just
    # below and just above it (different segments) must still agree.
    eps = 1e-6
    below = single_link_speed_flow.travel_time(np.array([1200.0 - eps]))
    above = single_link_speed_flow.travel_time(np.array([1200.0 + eps]))
    np.testing.assert_allclose(below, above, atol=1e-6)
    below_integral = single_link_speed_flow.integral(np.array([1200.0 - eps]))
    above_integral = single_link_speed_flow.integral(np.array([1200.0 + eps]))
    np.testing.assert_allclose(below_integral, above_integral, atol=1e-6)


def test_speed_flow_requires_at_least_two_breakpoints():
    with pytest.raises(ValueError, match="at least two breakpoints"):
        SpeedFlow(
            length=np.array([5.0]),
            lanes=np.array([2.0]),
            flow=np.array([[0.0]]),
            speed=np.array([[100.0]]),
        )


def test_speed_flow_requires_strictly_increasing_flow():
    with pytest.raises(ValueError, match="strictly increasing"):
        SpeedFlow(
            length=np.array([5.0]),
            lanes=np.array([2.0]),
            flow=np.array([[0.0, 300.0, 300.0]]),
            speed=np.array([[100.0, 90.0, 40.0]]),
        )


def test_speed_flow_requires_positive_speeds():
    with pytest.raises(ValueError, match="positive"):
        SpeedFlow(
            length=np.array([5.0]),
            lanes=np.array([2.0]),
            flow=np.array([[0.0, 300.0, 600.0]]),
            speed=np.array([[100.0, 0.0, 40.0]]),
        )


def test_speed_flow_requires_at_least_one_lane():
    with pytest.raises(ValueError, match="lanes"):
        SpeedFlow(
            length=np.array([5.0]),
            lanes=np.array([0.0]),
            flow=np.array([[0.0, 300.0, 600.0]]),
            speed=np.array([[100.0, 90.0, 40.0]]),
        )


def test_speed_flow_requires_a_positive_min_speed():
    with pytest.raises(ValueError, match="min_speed"):
        SpeedFlow(
            length=np.array([5.0]),
            lanes=np.array([2.0]),
            flow=np.array([[0.0, 300.0, 600.0]]),
            speed=np.array([[100.0, 90.0, 40.0]]),
            min_speed=0.0,
        )


@pytest.fixture()
def speed_flow_curves():
    """The synthetic fixture curve table (tests/data), not real TAG data."""
    return pd.read_csv(DATA / "synthetic_speed_flow_curves.csv", comment="#")


@pytest.fixture()
def speed_flow_network():
    """Two links referencing the two curves in ``speed_flow_curves``."""
    return Network.from_dataframe(
        pd.DataFrame(
            {
                "edge_from": [1, 1],
                "edge_to": [2, 2],
                "edge_id": [0, 1],
                "length": [5.0, 5.0],
                "lanes": [2.0, 1.0],
                "link_type": [1, 2],
            }
        )
    )


def test_speed_flow_from_table_assigns_one_curve_per_link(
    speed_flow_curves, speed_flow_network
):
    cost_function = SpeedFlow.from_table(
        speed_flow_curves, speed_flow_network, min_speed=8.0
    )
    # Link 0 (curve 1) at q=300: exactly the second breakpoint, speed 90.
    # Link 1 (curve 2) at q=400: exactly its second breakpoint, speed 50.
    np.testing.assert_allclose(
        cost_function.travel_time(np.array([600.0, 400.0])),
        [5.0 / 90.0, 5.0 / 50.0],
    )
    # Curve 2 (3 breakpoints) is padded to curve 1's width (4) by
    # continuing its own last segment's slope, staying positive.
    assert cost_function.flow.shape == (2, 4)
    assert np.all(cost_function.speed > 0)


def test_speed_flow_padding_a_short_curve_changes_nothing(
    speed_flow_curves, speed_flow_network
):
    """The invariant the whole padding scheme rests on.

    ``from_table`` makes curves of different lengths rectangular by adding
    breakpoints on the line the shorter curve's last segment is already
    extrapolated along. That is only safe if the padded curve computes
    exactly what the unpadded one would, so assert it rather than trusting
    the argument: the shorter fixture curve (id 2, three breakpoints) is
    padded to four here.
    """
    padded = SpeedFlow.from_table(speed_flow_curves, speed_flow_network, min_speed=8.0)
    curve = speed_flow_curves[speed_flow_curves["curve_id"] == 2].sort_values(
        "flow_per_lane"
    )
    unpadded = SpeedFlow(
        length=np.array([5.0]),
        lanes=np.array([1.0]),
        flow=curve["flow_per_lane"].to_numpy(dtype="float64")[None, :],
        speed=curve["speed"].to_numpy(dtype="float64")[None, :],
        min_speed=8.0,
    )
    assert padded.flow.shape[1] == unpadded.flow.shape[1] + 1
    # Link 1 of the network is the one carrying curve 2. Sweep from below
    # the first breakpoint, across every segment, past the last breakpoint
    # and into the clamp.
    for x in np.linspace(0.0, 3000.0, 121):
        one = np.array([x])
        two = np.array([0.0, x])
        assert padded.travel_time(two)[1] == pytest.approx(
            unpadded.travel_time(one)[0], rel=1e-12
        )
        assert padded.integral(two)[1] == pytest.approx(
            unpadded.integral(one)[0], rel=1e-9
        )
        assert padded.derivative(two)[1] == pytest.approx(
            unpadded.derivative(one)[0], rel=1e-12
        )


def test_speed_flow_extrapolates_backwards_below_the_first_breakpoint():
    """A curve whose first breakpoint is not at zero flow still has a speed
    at zero flow: the first segment's line is continued backwards, the
    mirror of what happens beyond the last breakpoint."""
    cost_function = SpeedFlow(
        length=np.array([10.0]),
        lanes=np.array([2.0]),
        flow=np.array([[200.0, 500.0, 800.0]]),
        speed=np.array([[70.0, 60.0, 30.0]]),
        min_speed=5.0,
    )
    # Slope of the first segment is (60 - 70) / (500 - 200) = -1/30, so at
    # q = 0 the line reads 70 + 200/30 = 76.666..., not 70.
    assert cost_function.travel_time(0.0)[0] == pytest.approx(10.0 / (70.0 + 200 / 30))
    # And the integral over that region is the one the curve implies, not
    # the one a constant-speed extension would give.
    x = 200.0  # q = 100, still below the first breakpoint
    w = np.linspace(0.0, x, 200_001)
    along_grid = SpeedFlow(
        length=np.full(w.size, 10.0),
        lanes=np.full(w.size, 2.0),
        flow=np.tile([200.0, 500.0, 800.0], (w.size, 1)),
        speed=np.tile([70.0, 60.0, 30.0], (w.size, 1)),
        min_speed=5.0,
    )
    quadrature = np.trapezoid(along_grid.travel_time(w), w)
    assert cost_function.integral(np.array([x]))[0] == pytest.approx(
        quadrature, rel=1e-6
    )


def test_speed_flow_from_table_missing_curve_id_raises(speed_flow_curves):
    network = Network.from_dataframe(
        pd.DataFrame(
            {
                "edge_from": [1],
                "edge_to": [2],
                "edge_id": [0],
                "length": [5.0],
                "lanes": [2.0],
                "link_type": [999],
            }
        )
    )
    with pytest.raises(ValueError, match="999"):
        SpeedFlow.from_table(speed_flow_curves, network)


def test_speed_flow_from_table_missing_columns_raises(speed_flow_network):
    bad_table = pd.DataFrame({"curve_id": [1], "flow_per_lane": [0.0]})
    with pytest.raises(ValueError, match="speed"):
        SpeedFlow.from_table(bad_table, speed_flow_network)


def test_speed_flow_from_table_accepts_a_pyarrow_table(speed_flow_network):
    curves = pa.table(
        {
            "curve_id": [1, 1, 2, 2],
            "flow_per_lane": [0.0, 900.0, 0.0, 900.0],
            "speed": [100.0, 10.0, 100.0, 10.0],
        }
    )
    cost_function = SpeedFlow.from_table(curves, speed_flow_network)
    assert cost_function.n_links == 2
