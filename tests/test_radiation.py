"""Tests for the radiation model."""

import pandas as pd
import pytest

from transport_flow_model.model import Network
from transport_flow_model.radiation import RadiationModel


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
