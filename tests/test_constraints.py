"""Tests for constraining and validating OD matrices against aggregate data."""

import numpy as np
import pytest

from transport_flow_model import (
    aggregate_od,
    common_part_of_commuters,
    od_fit,
    radiation_flows,
    scale_to_aggregate,
)

GROUPS = ["a", "a", "b", "b", "c"]


@pytest.fixture
def prior():
    positions = np.array([0.0, 1.0, 5.0, 6.0, 10.0])
    d = np.abs(positions[:, None] - positions[None, :])
    return radiation_flows(d, [100, 50, 80, 20, 60], destination_mass=[10, 60, 40, 40, 30])


@pytest.fixture
def targets():
    # Rows sum to the production of each group: a=150, b=100, c=60.
    # c has one zone, so it can have no internal flow.
    return np.array([[90.0, 50.0, 10.0], [20.0, 60.0, 20.0], [10.0, 50.0, 0.0]])


def test_aggregate_od_sums_groups(prior):
    matrix, labels = aggregate_od(prior, GROUPS)
    assert labels.tolist() == ["a", "b", "c"]
    assert matrix.sum() == pytest.approx(prior.sum())
    assert matrix[0, 1] == pytest.approx(prior[np.ix_([0, 1], [2, 3])].sum())


def test_block_scaling_matches_targets_exactly(prior, targets):
    scaled = scale_to_aggregate(prior, GROUPS, targets, method="block")
    np.testing.assert_allclose(aggregate_od(scaled, GROUPS)[0], targets)
    # The pattern within a block is kept
    ratio = scaled[np.ix_([0, 1], [2, 3])] / prior[np.ix_([0, 1], [2, 3])]
    np.testing.assert_allclose(ratio, ratio[0, 0])


def test_ipf_matches_targets_and_production(prior, targets):
    production = prior.sum(axis=1)
    scaled = scale_to_aggregate(prior, GROUPS, targets, method="ipf", tolerance=1e-9, max_iterations=500)
    np.testing.assert_allclose(scaled.sum(axis=1), production, rtol=1e-6)
    np.testing.assert_allclose(aggregate_od(scaled, GROUPS)[0], targets, rtol=1e-5)


def test_ipf_makes_targets_consistent_with_production(prior, targets):
    # Targets in different units (e.g. a census year with more workers) are
    # rescaled per origin group to the production.
    scaled = scale_to_aggregate(prior, GROUPS, targets * 2, method="ipf", tolerance=1e-9, max_iterations=500)
    np.testing.assert_allclose(scaled.sum(), prior.sum(), rtol=1e-6)
    np.testing.assert_allclose(aggregate_od(scaled, GROUPS)[0], targets, rtol=1e-5)


def test_unreachable_target_pairs_stay_zero():
    flows = np.array([[0.0, 1.0], [0.0, 0.0]])
    scaled = scale_to_aggregate(flows, ["a", "b"], np.array([[0.0, 2.0], [3.0, 0.0]]), method="block")
    assert scaled[1, 0] == 0
    assert scaled[0, 1] == 2


def test_invalid_inputs(prior, targets):
    with pytest.raises(ValueError):
        scale_to_aggregate(prior, GROUPS, targets, method="bad")
    with pytest.raises(ValueError):
        scale_to_aggregate(prior, GROUPS, targets[:2, :2])
    with pytest.raises(ValueError):
        aggregate_od(prior, GROUPS, labels=["a", "b"])


def test_common_part_of_commuters():
    m = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert common_part_of_commuters(m, m) == pytest.approx(1.0)
    assert common_part_of_commuters(m, np.zeros((2, 2))) == pytest.approx(0.0)


def test_od_fit(targets):
    metrics, table = od_fit(targets * 1.1, targets, ["a", "b", "c"])
    assert metrics["cpc"] == pytest.approx(2 / 2.1)
    assert metrics["observed_internal_share"] == pytest.approx(150 / 310)
    assert len(table) == 9
    assert set(table.columns) == {"origin", "destination", "modelled", "observed", "difference", "ratio"}
