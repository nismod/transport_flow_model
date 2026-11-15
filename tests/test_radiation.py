"""Tests for the radiation model."""

import numpy as np
import pandas as pd
import pytest

from transport_flow_model.model import Network
from transport_flow_model.radiation import (
    RadiationModel,
    radiation_flows,
    radiation_probabilities,
)


@pytest.fixture
def simple_network():
    """Create a simple 5-node network for testing.

    Network structure:
    A === B --- C
          |
          D --- E

    Edges with cost 1.0 between each connection.
    """
    network_data = pd.DataFrame(
        {
            "edge_from": ["A", "B", "B", "B", "D"],
            "edge_to": ["B", "C", "D", "A", "E"],
            "edge_id": ["AB", "BC", "BD", "BA", "DE"],
            "cost": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    return Network(network_data)


@pytest.fixture
def zones_with_mapping():
    """Create synthetic zones and mapping for testing."""
    zones = pd.DataFrame(
        {
            "zone_id": [1, 2, 3, 4],
            "population": [1000, 2000, 1500, 1200],
        }
    )
    mapping = pd.DataFrame(
        {
            "zone_id": [1, 2, 3, 4],
            "node_id": ["A", "B", "C", "D"],
        }
    )
    return zones, mapping


def test_radiation_model_initialization(simple_network):
    """Test that RadiationModel initializes correctly."""
    rad = RadiationModel(simple_network)
    assert rad.network is simple_network


def test_generate_returns_dataframe(simple_network, zones_with_mapping):
    """Test that generate() returns a DataFrame with correct columns."""
    zones, mapping = zones_with_mapping
    rad = RadiationModel(simple_network)

    result = rad.generate(
        zones=zones,
        zone_id_column="zone_id",
        zone_to_node_mapping=mapping,
        relevance_column="population",
        distance_threshold=5.0,
    )

    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {"origin", "destination", "probability"}


def test_probabilities_in_valid_range(simple_network, zones_with_mapping):
    """Test that all probabilities are in [0, 1]."""
    zones, mapping = zones_with_mapping
    rad = RadiationModel(simple_network)

    result = rad.generate(
        zones=zones,
        zone_id_column="zone_id",
        zone_to_node_mapping=mapping,
        relevance_column="population",
        distance_threshold=5.0,
    )

    assert (result["probability"] >= 0.0).all()
    assert (result["probability"] <= 1.0).all()


def test_no_self_flows(simple_network, zones_with_mapping):
    """Test that there are no flows from a zone to itself."""
    zones, mapping = zones_with_mapping
    rad = RadiationModel(simple_network)

    result = rad.generate(
        zones=zones,
        zone_id_column="zone_id",
        zone_to_node_mapping=mapping,
        relevance_column="population",
        distance_threshold=5.0,
    )

    # No rows should have origin == destination
    assert (result["origin"] != result["destination"]).all()


def test_distance_threshold_effect(simple_network, zones_with_mapping):
    """Test that increasing distance threshold increases number of OD pairs."""
    zones, mapping = zones_with_mapping
    rad = RadiationModel(simple_network)

    # Small threshold
    result_small = rad.generate(
        zones=zones,
        zone_id_column="zone_id",
        zone_to_node_mapping=mapping,
        relevance_column="population",
        distance_threshold=1.5,
    )

    # Large threshold
    result_large = rad.generate(
        zones=zones,
        zone_id_column="zone_id",
        zone_to_node_mapping=mapping,
        relevance_column="population",
        distance_threshold=10.0,
    )

    assert len(result_large) >= len(result_small)


def test_error_on_missing_column(simple_network, zones_with_mapping):
    """Test that error is raised for missing zone_id_column."""
    zones, mapping = zones_with_mapping
    rad = RadiationModel(simple_network)

    with pytest.raises(ValueError, match="not found in zones DataFrame"):
        rad.generate(
            zones=zones,
            zone_id_column="nonexistent",
            zone_to_node_mapping=mapping,
            relevance_column="population",
            distance_threshold=5.0,
        )


def test_error_on_missing_relevance_column(simple_network, zones_with_mapping):
    """Test that error is raised for missing relevance_column."""
    zones, mapping = zones_with_mapping
    rad = RadiationModel(simple_network)

    with pytest.raises(ValueError, match="not found in zones DataFrame"):
        rad.generate(
            zones=zones,
            zone_id_column="zone_id",
            zone_to_node_mapping=mapping,
            relevance_column="nonexistent",
            distance_threshold=5.0,
        )


def test_error_on_missing_mapping_zone_id(simple_network, zones_with_mapping):
    """Test that error is raised if mapping lacks zone_id column."""
    zones, mapping = zones_with_mapping
    bad_mapping = mapping.drop(columns=["zone_id"])
    rad = RadiationModel(simple_network)

    with pytest.raises(ValueError, match="zone_to_node_mapping must have"):
        rad.generate(
            zones=zones,
            zone_id_column="zone_id",
            zone_to_node_mapping=bad_mapping,
            relevance_column="population",
            distance_threshold=5.0,
        )


def test_error_on_unmapped_zone(simple_network, zones_with_mapping):
    """Test that error is raised if mapping references zone not in zones."""
    zones, mapping = zones_with_mapping
    bad_mapping = mapping.copy()
    bad_mapping.loc[bad_mapping["zone_id"] == 1, "zone_id"] = 999
    rad = RadiationModel(simple_network)

    with pytest.raises(ValueError, match="Zones in mapping not found"):
        rad.generate(
            zones=zones,
            zone_id_column="zone_id",
            zone_to_node_mapping=bad_mapping,
            relevance_column="population",
            distance_threshold=5.0,
        )


def test_error_on_unmapped_node(simple_network, zones_with_mapping):
    """Test that error is raised if mapping references node not in network."""
    zones, mapping = zones_with_mapping
    bad_mapping = mapping.copy()
    bad_mapping.loc[bad_mapping["zone_id"] == 1, "node_id"] = "NONEXISTENT"
    rad = RadiationModel(simple_network)

    with pytest.raises(ValueError, match="Nodes in mapping not found"):
        rad.generate(
            zones=zones,
            zone_id_column="zone_id",
            zone_to_node_mapping=bad_mapping,
            relevance_column="population",
            distance_threshold=5.0,
        )


def test_error_on_insufficient_zones():
    """Test that error is raised if fewer than 2 zones provided."""
    network_data = pd.DataFrame(
        {
            "edge_from": ["A", "B"],
            "edge_to": ["B", "A"],
            "edge_id": ["AB", "BA"],
            "cost": [1.0, 1.0],
        }
    )
    network = Network(network_data)

    zones = pd.DataFrame({"zone_id": [1], "population": [1000]})
    mapping = pd.DataFrame({"zone_id": [1], "node_id": ["A"]})

    rad = RadiationModel(network)

    with pytest.raises(ValueError, match="Need at least 2 zones"):
        rad.generate(
            zones=zones,
            zone_id_column="zone_id",
            zone_to_node_mapping=mapping,
            relevance_column="population",
            distance_threshold=5.0,
        )


def test_empty_result_on_very_small_threshold(simple_network, zones_with_mapping):
    """Test that empty DataFrame is returned when threshold is very small."""
    zones, mapping = zones_with_mapping
    rad = RadiationModel(simple_network)

    result = rad.generate(
        zones=zones,
        zone_id_column="zone_id",
        zone_to_node_mapping=mapping,
        relevance_column="population",
        distance_threshold=0.1,  # Very small threshold
    )

    # Should return empty DataFrame with correct columns
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {"origin", "destination", "probability"}
    assert len(result) == 0


def test_radiation_formula_consistency(simple_network, zones_with_mapping):
    """Test that probabilities are calculated consistently."""
    zones, mapping = zones_with_mapping
    rad = RadiationModel(simple_network)

    result = rad.generate(
        zones=zones,
        zone_id_column="zone_id",
        zone_to_node_mapping=mapping,
        relevance_column="population",
        distance_threshold=5.0,
    )

    # Running again should give identical results
    result2 = rad.generate(
        zones=zones,
        zone_id_column="zone_id",
        zone_to_node_mapping=mapping,
        relevance_column="population",
        distance_threshold=5.0,
    )

    pd.testing.assert_frame_equal(result, result2)


def test_bidirectional_network(simple_network, zones_with_mapping):
    """Test that network is treated as bidirectional."""
    zones, mapping = zones_with_mapping
    rad = RadiationModel(simple_network)

    result = rad.generate(
        zones=zones,
        zone_id_column="zone_id",
        zone_to_node_mapping=mapping,
        relevance_column="population",
        distance_threshold=5.0,
    )

    # Should have some bidirectional flows
    assert len(result) > 0
    any_both = False
    # Check that there are flows in both directions for some pairs
    for _, row in result.iterrows():
        reverse = result[
            (result["origin"] == row["destination"])
            & (result["destination"] == row["origin"])
        ]
        # Not all pairs need to have reverse flows (due to distance threshold)
        # but we should verify the model is working
        if len(reverse):
            any_both = True
    assert any_both


def test_custom_column_names(simple_network):
    """Test that model works with custom column names."""
    zones = pd.DataFrame(
        {
            "loc_id": [1, 2, 3, 4],
            "jobs": [1000, 2000, 1500, 1200],
        }
    )
    mapping = pd.DataFrame(
        {
            "zone_id": [1, 2, 3, 4],
            "node_id": ["A", "B", "C", "D"],
        }
    )

    rad = RadiationModel(simple_network)
    result = rad.generate(
        zones=zones,
        zone_id_column="loc_id",
        zone_to_node_mapping=mapping,
        relevance_column="jobs",
        distance_threshold=5.0,
    )

    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {"origin", "destination", "probability"}


# --- Intervening opportunities, separate masses, flows, performance ---------


def _line(positions):
    positions = np.asarray(positions, dtype=float)
    return np.abs(positions[:, None] - positions[None, :])


def test_intervening_opportunities_are_zones_closer_than_destination():
    p = radiation_probabilities(_line([0, 1, 2, 3]), np.ones(4), normalisation="none")
    # From zone 0: s_01 = 0, s_02 = 1, s_03 = 2
    np.testing.assert_allclose(p[0, 1:], [1 / 2, 1 / 6, 1 / 12])
    assert p[0, 0] == 0


def test_equidistant_zones_intervene_for_each_other():
    d = np.array([[0, 1, 1], [1, 0, 2], [1, 2, 0]], dtype=float)
    p = radiation_probabilities(d, np.ones(3), normalisation="none")
    np.testing.assert_allclose(p[0, 1:], [1 / 6, 1 / 6])


def test_intervening_opportunities_use_destination_mass():
    d = _line([0, 1, 2])
    origin = np.array([1.0, 1.0, 1.0])
    jobs = np.array([0.0, 4.0, 1.0])
    p = radiation_probabilities(d, origin, jobs, normalisation="none")
    # From zone 0: s_02 = jobs at zone 1 = 4
    np.testing.assert_allclose(p[0, 2], (1 * 1) / ((1 + 4) * (1 + 1 + 4)))


def test_row_normalisation_distributes_all_production():
    d = _line([0, 1, 2, 10])
    t = radiation_flows(
        d, [100, 50, 10, 5], destination_mass=[1, 5, 5, 1], max_impedance=5
    )
    np.testing.assert_allclose(t.sum(axis=1), [100, 50, 10, 0])
    assert t[0, 3] == 0


def test_min_flow_keeps_origin_totals():
    d = _line(np.arange(10))
    t = radiation_flows(d, np.full(10, 100.0), min_flow=5.0)
    np.testing.assert_allclose(t.sum(axis=1), 100.0)
    assert ((t == 0) | (t >= 5.0)).all()


def test_unreachable_pairs_have_zero_probability():
    d = np.array([[0, np.inf], [np.inf, 0]])
    assert radiation_probabilities(d, np.ones(2)).sum() == 0


def test_invalid_inputs():
    with pytest.raises(ValueError):
        radiation_probabilities(np.zeros((2, 2)), np.ones(2), normalisation="bad")
    with pytest.raises(ValueError):
        radiation_probabilities(np.zeros((2, 3)), np.ones(2))
    with pytest.raises(ValueError):
        radiation_probabilities(np.zeros((2, 2)), np.array([1.0, -1.0]))


def test_generate_with_destination_relevance(simple_network):
    zones = pd.DataFrame(
        {"zone_id": [1, 2, 3], "workers": [100, 0, 0], "jobs": [0, 10, 10]}
    )
    mapping = pd.DataFrame({"zone_id": [1, 2, 3], "node_id": ["A", "B", "C"]})
    result = RadiationModel(simple_network).generate(
        zones,
        "zone_id",
        mapping,
        "workers",
        10.0,
        destination_relevance_column="jobs",
        normalisation="row",
    )
    from_one = result[result.origin == 1].set_index("destination")["probability"]
    assert from_one.sum() == pytest.approx(1.0)
    # B is closer than C, so it is preferred
    assert from_one[2] > from_one[3]
    # Zones with no origin mass produce no trips
    assert set(result.origin) == {1}


def test_flows_returns_demand_ready_table(simple_network):
    zones = pd.DataFrame(
        {"zone_id": [1, 2, 3, 4], "workers": [100, 50, 0, 0], "jobs": [10, 0, 40, 20]}
    )
    mapping = pd.DataFrame({"zone_id": [1, 2, 3, 4], "node_id": ["A", "B", "C", "E"]})
    flows = RadiationModel(simple_network).flows(
        zones, "zone_id", mapping, "workers", "jobs", 10.0
    )
    assert set(flows.columns) == {
        "origin_id",
        "destination_id",
        "value",
        "origin_node_id",
        "destination_node_id",
    }
    totals = flows.groupby("origin_id")["value"].sum()
    assert totals[1] == pytest.approx(100)
    assert totals[2] == pytest.approx(50)


def test_directed_impedance():
    network = Network(
        pd.DataFrame(
            {
                "edge_from": ["A", "B"],
                "edge_to": ["B", "C"],
                "edge_id": ["AB", "BC"],
                "cost": [1.0, 2.0],
            }
        )
    )
    rad = RadiationModel(network)
    undirected = rad.impedance_matrix(["A", "C"])
    directed = rad.impedance_matrix(["A", "C"], directed=True)
    assert undirected[1, 0] == 3.0
    assert directed[0, 1] == 3.0
    assert np.isinf(directed[1, 0])


def test_performance_thousands_of_zones():
    """2,500 zones (about Mauritius at H3 resolution 8) in a few seconds."""
    import time

    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 50, size=(2500, 2))
    d = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
    start = time.perf_counter()
    t = radiation_flows(
        d, rng.uniform(0, 500, 2500), destination_mass=rng.uniform(0, 500, 2500)
    )
    assert time.perf_counter() - start < 30
    assert t.shape == (2500, 2500)


def test_include_self_uses_diagonal_impedance():
    d = _line([0, 1, 2])
    np.fill_diagonal(d, 0.5)  # intrazonal impedance, closer than any other zone
    m = np.array([2.0, 1.0, 1.0])
    n = np.array([3.0, 1.0, 1.0])
    p = radiation_probabilities(d, m, n, include_self=True, normalisation="none")
    # No zone closer than the intrazonal impedance: P_ii = n_i / (m_i + n_i)
    assert p[0, 0] == pytest.approx(3 / 5)
    # The origin's own mass is not an intervening opportunity for zone 1
    assert p[0, 1] == pytest.approx(2 * 1 / ((2 + 0) * (2 + 1 + 0)))
    # Without include_self the diagonal is ignored
    assert radiation_probabilities(d, m, n, normalisation="none")[0, 0] == 0


def test_include_self_keeps_production_with_row_normalisation():
    d = _line([0, 1, 2, 3])
    np.fill_diagonal(d, 0.3)
    t = radiation_flows(
        d, [100, 50, 10, 0], destination_mass=[10, 60, 80, 10], include_self=True
    )
    np.testing.assert_allclose(t.sum(axis=1), [100, 50, 10, 0])
    assert (np.diag(t)[:3] > 0).all()
