"""MSA (ws2-02): convergence, the free gap, dtypes and the option surface.

The acceptance criterion of ws2-02 is that MSA reaches a 1e-3 relative gap
on SiouxFalls with a fully populated result object
(``test_msa_reaches_target_gap_on_siouxfalls``). The load-bearing test of
*how* it gets there is ``test_free_gap_matches_independent_relative_gap``:
MSA reads its gap off the all-or-nothing pass it has to do anyway, and that
test proves the shortcut is exact rather than approximately right.
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from transport_flow_model import (
    BPR,
    Demand,
    Network,
    assign,
    beckmann_objective,
    datasets,
    relative_gap,
)
from transport_flow_model.assignment import _msa_iterates

#: MSA converges like 1/k, so 1e-3 on SiouxFalls needs several hundred
#: passes — far more than the default 50. See ``test_msa_convergence_rate``.
SIOUXFALLS_BUDGET = 1000


@pytest.fixture(scope="module")
def siouxfalls():
    instance = datasets.load_tntp("siouxfalls")
    return Network.from_tntp(instance), Demand.from_tntp(instance)


@pytest.fixture(scope="module")
def converged(siouxfalls):
    network, demand = siouxfalls
    return assign(
        network, demand, "msa", target_gap=1e-3, max_iterations=SIOUXFALLS_BUDGET
    )


@pytest.fixture()
def split_network():
    """Two parallel congestible links 1->2, plus an unreachable 3->4."""
    return Network.from_dataframe(
        pd.DataFrame(
            {
                "edge_from": [1, 1, 3],
                "edge_to": [2, 2, 4],
                "edge_id": [0, 1, 2],
                "cost": [1.0, 1.2, 1.0],
                "capacity": [50.0, 50.0, 50.0],
                "alpha": [0.15, 0.15, 0.15],
                "beta": [4.0, 4.0, 4.0],
            }
        )
    )


@pytest.fixture()
def split_demand():
    """One assignable pair and one whose destination has no path."""
    return Demand.from_dataframe(
        pd.DataFrame(
            {
                "origin_id": [1, 1],
                "destination_id": [2, 4],
                "value": [100.0, 5.0],
            }
        )
    )


# --- 1. it converges --------------------------------------------------------


def test_msa_reaches_target_gap_on_siouxfalls(converged, record_property):
    """ws2-02 acceptance: 1e-3 on SiouxFalls, result fully populated."""
    provenance = converged.provenance
    record_property("iterations_to_1e-3", provenance.iterations)
    record_property("final_gap", provenance.relative_gap)
    assert provenance.relative_gap <= 1e-3
    assert provenance.iterations <= SIOUXFALLS_BUDGET
    assert converged.link_flows.num_rows == 76
    assert converged.skims.num_rows == 528
    assert converged.skims["cost"].null_count == 0
    assert converged.unassigned.num_rows == 0


def test_msa_convergence_rate(siouxfalls, record_property):
    """The gap falls like 1/k, so the 50-iteration default is nowhere near.

    Pinned because it is the reason every SiouxFalls test here passes an
    explicit ``max_iterations``: MSA is the slow baseline, not the method
    of choice.
    """
    network, demand = siouxfalls
    result = assign(network, demand, "msa", target_gap=0.0, max_iterations=200)
    gaps = np.array(result.gap_history)
    record_property("gap_at_50", gaps[48])
    record_property("gap_at_200", gaps[-1])
    assert gaps[48] > 1e-3  # 50 passes is not enough, by an order of magnitude
    # Halving the gap takes roughly twice as many passes.
    assert gaps[98] / gaps[198] == pytest.approx(2.0, rel=0.1)


def test_gap_history_trends_down_without_being_monotone(converged):
    """The gap falls over the run, but not step by step.

    MSA averages in a fresh all-or-nothing solution each pass, and that
    solution jumps between competing routes, so the measured gap oscillates
    — on SiouxFalls it rises on roughly a third of the tail steps. Only the
    trend converges, so only the trend is asserted; the rate itself is
    pinned by :func:`test_msa_convergence_rate`.
    """
    gaps = np.array(converged.gap_history)
    assert gaps[-1] < gaps[0]
    assert gaps[-1] < gaps[0] / 100.0

    # Smoothed, the trend is strictly down: every window of 50 passes has a
    # lower mean gap than the one before it.
    windows = gaps[: len(gaps) // 50 * 50].reshape(-1, 50).mean(axis=1)
    assert np.all(np.diff(windows) < 0)

    # Not monotone step by step, which is why the window is needed at all.
    assert np.any(np.diff(gaps[10:]) > 0)


# --- 2. the free gap is exact ----------------------------------------------


def test_free_gap_matches_independent_relative_gap(siouxfalls):
    """Every gap MSA reads off its next AON pass equals an independent one.

    MSA never calls :func:`relative_gap`. It takes the shortest-path travel
    time term from the all-or-nothing load ``y`` it performs anyway:
    ``gap = dot(t, x) / dot(t, y) - 1`` at the *current* costs ``t =
    t(x)``. This test recomputes each reported gap from scratch — congested
    costs, a batched skim over every OD pair, the ratio — and requires the
    two to agree to 1e-10 absolute. It fails if the numerator uses anything
    but the same ``t`` as the denominator.
    """
    network, demand = siouxfalls
    iterates = list(_msa_iterates(network, demand, max_iterations=8, target_gap=0.0))

    assert [it.k for it in iterates] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert iterates[0].gap is None
    for previous, current in zip(iterates, iterates[1:]):
        # current.gap was measured at the costs of, and belongs to, the
        # iterate entering pass current.k — that is previous.flows.
        independent = relative_gap(network, demand, previous.flows)
        assert current.gap == pytest.approx(independent, abs=1e-10)


def test_reported_gap_is_the_gap_of_the_returned_flows(converged, siouxfalls):
    """Stopping on the target skips the averaging, so the two coincide."""
    network, demand = siouxfalls
    independent = relative_gap(network, demand, converged)
    assert converged.provenance.relative_gap == pytest.approx(independent, abs=1e-12)


# --- 3. the objective -------------------------------------------------------


def test_beckmann_objective_approaches_the_published_value(
    siouxfalls, converged, record_property
):
    """Within 1% of the published Beckmann objective, and below the AON's.

    Measured relative error at a 1e-3 gap is 1.7e-3, so the 1e-2 threshold
    keeps a factor of six of headroom. Tightening it further would make the
    test a pin on MSA's exact iterate rather than on it solving the problem.
    """
    network, _ = siouxfalls
    published = datasets.BEST_KNOWN["siouxfalls"].objective
    objective = beckmann_objective(network, converged)
    record_property("objective_relative_error", abs(objective - published) / published)
    assert objective == pytest.approx(published, rel=1e-2)
    assert objective < beckmann_objective(
        network, assign(network, _demand(siouxfalls), "msa", max_iterations=1)
    )


def _demand(siouxfalls):
    return siouxfalls[1]


# --- 4. link flows stay float64 --------------------------------------------


def test_link_flows_are_float64(siouxfalls, converged):
    """SiouxFalls demand is integral, so an AON-only run would coerce to int64.

    ADR-0001: an iterative method calls ``link_flows_table(..., coerce=False)``
    so the dtype does not flip with the iteration count (m0-04).
    """
    network, demand = siouxfalls
    assert converged.link_flows["flow"].type == pa.float64()
    single = assign(network, demand, "msa", max_iterations=1)
    assert single.link_flows["flow"].type == pa.float64()
    # The premise: one all-or-nothing pass really does produce integral flows.
    flows = single.link_flows["flow"].to_numpy(zero_copy_only=False)
    assert np.all(flows == np.round(flows))
    assert assign(network, demand, "sequential").link_flows["flow"].type == pa.int64()


# --- 5. provenance ----------------------------------------------------------


def test_provenance_is_populated_and_consistent(converged):
    provenance = converged.provenance
    assert provenance.method == "msa"
    assert provenance.options == {
        "target_gap": 1e-3,
        "max_iterations": SIOUXFALLS_BUDGET,
    }
    assert provenance.iterations > 1
    assert len(converged.gap_history) == provenance.iterations - 1
    assert converged.gap_history[-1] == provenance.relative_gap
    assert provenance.wall_time_s > 0
    assert all(isinstance(gap, float) for gap in converged.gap_history)


def test_paths_are_never_returned(converged):
    assert converged.paths is None


def test_include_paths_raises(siouxfalls):
    network, demand = siouxfalls
    with pytest.raises(NotImplementedError, match="m0-13"):
        assign(network, demand, "msa", include_paths=True, max_iterations=2)


# --- 6. options -------------------------------------------------------------


def test_single_iteration_is_all_or_nothing(siouxfalls):
    network, demand = siouxfalls
    single = assign(network, demand, "msa", max_iterations=1)
    sequential = assign(network, demand, "sequential")
    assert single.provenance.relative_gap is None
    assert single.gap_history == ()
    assert single.provenance.iterations == 1
    np.testing.assert_allclose(
        single.link_flows["flow"].to_numpy(zero_copy_only=False),
        sequential.link_flows["flow"].to_numpy(zero_copy_only=False),
    )


def test_time_limit_stops_after_the_first_gap(siouxfalls):
    """A zero budget still runs the pass that produces a gap to report."""
    network, demand = siouxfalls
    result = assign(network, demand, "msa", time_limit_s=0.0, max_iterations=50)
    assert result.provenance.iterations == 2
    assert len(result.gap_history) == 1
    # Stopped before averaging, so the flows are still the AON load.
    single = assign(network, demand, "msa", max_iterations=1)
    np.testing.assert_array_equal(
        result.link_flows["flow"].to_numpy(zero_copy_only=False),
        single.link_flows["flow"].to_numpy(zero_copy_only=False),
    )


def test_explicit_cost_function_matches_the_default(siouxfalls):
    network, demand = siouxfalls
    default = assign(network, demand, "msa", max_iterations=12, target_gap=0.0)
    explicit = assign(
        network,
        demand,
        "msa",
        max_iterations=12,
        target_gap=0.0,
        cost_function=BPR.from_network(network),
    )
    assert explicit.gap_history == default.gap_history
    np.testing.assert_array_equal(
        explicit.link_flows["flow"].to_numpy(zero_copy_only=False),
        default.link_flows["flow"].to_numpy(zero_copy_only=False),
    )


def test_unknown_option_raises(siouxfalls):
    network, demand = siouxfalls
    with pytest.raises(TypeError):
        assign(network, demand, "msa", step_size=0.5)


def test_distance_cost_reaches_the_cost_function():
    """chicago-sketch prices distance at 0.04; that must change the run."""
    instance = datasets.load_tntp("chicago-sketch")
    network, demand = Network.from_tntp(instance), Demand.from_tntp(instance)
    priced = assign(
        network, demand, "msa", max_iterations=3, target_gap=0.0, distance_cost=0.04
    )
    plain = assign(network, demand, "msa", max_iterations=3, target_gap=0.0)
    assert np.isfinite(priced.provenance.relative_gap)
    assert len(priced.gap_history) == 2
    assert priced.gap_history != plain.gap_history


# --- 7. unreachable demand --------------------------------------------------


def test_unreachable_demand_does_not_raise(split_network, split_demand):
    """The gap is over assigned demand only; ``relative_gap`` would raise."""
    result = assign(split_network, split_demand, "msa", max_iterations=20)
    unassigned = result.unassigned.to_pydict()
    assert unassigned["origin_id"] == [1]
    assert unassigned["destination_id"] == [4]
    assert unassigned["value"] == [5]
    assert np.isfinite(result.provenance.relative_gap)
    assert result.provenance.relative_gap >= 0.0
    # The unreachable pair still gets a (null-cost) skim row.
    assert result.skims.num_rows == 2
    assert result.skims["cost"].null_count == 1
    with pytest.raises(ValueError, match="unreachable"):
        relative_gap(split_network, split_demand, result)


def test_unreachable_demand_still_equilibrates(split_network, split_demand):
    """100 units split over two parallel links until their costs match."""
    result = assign(split_network, split_demand, "msa", max_iterations=400)
    flows = result.link_flows["flow"].to_numpy(zero_copy_only=False)
    costs = BPR.from_network(split_network).travel_time(flows)
    assert flows[0] + flows[1] == pytest.approx(100.0)
    assert flows[2] == 0.0
    assert costs[0] == pytest.approx(costs[1], rel=1e-2)
