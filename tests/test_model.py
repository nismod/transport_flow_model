import json

import pandas as pd
import pytest

from transport_flow_model.model import Network, OD, ODFlows


def test_network_from_csv_renames(tmp_path):
    csv_path = tmp_path / "network.csv"
    pd.DataFrame(
        {
            "from_id": ["A"],
            "to_id": ["B"],
            "id": ["E1"],
            "flow_capacity": [100],
            "gcost_usd_per_ton": [10.5],
            "length_m": [123],
        }
    ).to_csv(csv_path, index=False)

    network = Network.from_csv(
        csv_path,
        {
            "from_id": "edge_from",
            "to_id": "edge_to",
            "id": "edge_id",
            "flow_capacity": "capacity",
            "gcost_usd_per_ton": "cost",
        },
    )

    df = network.to_dataframe()
    assert list(df.columns) == [
        "edge_from",
        "edge_to",
        "edge_id",
        "capacity",
        "cost",
        # "length_m", - must include in dict to preserve
    ]
    assert df.loc[0, "edge_id"] == "E1"


def test_od_from_csv_requires_required_target_mappings(tmp_path):
    csv_path = tmp_path / "od.csv"
    pd.DataFrame(
        {
            "origin_id": ["A"],
            "destination_id": ["B"],
            "tons": [10],
        }
    ).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="required columns"):
        OD.from_csv(
            csv_path,
            {
                "origin_id": "origin_id",
                "destination_id": "destination_id",
                # missing flow mapping
            },
        )


def test_from_csv_fails_when_mapped_source_column_missing(tmp_path):
    csv_path = tmp_path / "od.csv"
    pd.DataFrame(
        {
            "origin_id": ["A"],
            "destination_id": ["B"],
            "tons": [10],
        }
    ).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Missing expected columns"):
        OD.from_csv(
            csv_path,
            {
                "origin_id": "origin_id",
                "destination_id": "destination_id",
                "typo_or_missing_column": "flow",
            },
        )


def test_data_property_returns_copy():
    od = OD(
        pd.DataFrame(
            {
                "origin_id": ["A"],
                "destination_id": ["B"],
                "flow": [10],
            }
        )
    )

    copy_df = od.data
    copy_df.loc[0, "flow"] = 999

    assert od.to_dataframe().loc[0, "flow"] == 10


def test_odflows_from_dataframes_and_to_csv(tmp_path):
    flow_routes = pd.DataFrame(
        {
            "origin_id": ["A", "B"],
            "destination_id": ["B", "C"],
            "flow": [10, 5],
            "edge_path": [["AB"], ["BA", "AC"]],
        }
    )

    od_flows = ODFlows(flow_routes)
    df = od_flows.to_dataframe()

    assert len(df) == 2
    assert df["flow"].sum() == 15

    csv_path = tmp_path / "od_flows.csv"
    od_flows.to_csv(csv_path)

    expected = flow_routes
    actual = pd.read_csv(csv_path)
    assert len(actual) == 2
    assert list(actual.columns) == [
        "origin_id",
        "destination_id",
        "flow",
        "edge_path",
    ]
    # fix quotes and load with JSON parser to recover list
    actual.edge_path = actual.edge_path.map(lambda s: json.loads(s.replace("'", '"')))
    pd.testing.assert_frame_equal(actual, expected)


# Tests for OD I/O methods
class TestODIO:
    @pytest.fixture
    def od_data(self):
        return OD(
            pd.DataFrame(
                {
                    "origin_id": ["A", "B", "C"],
                    "destination_id": ["B", "C", "A"],
                    "flow": [10.5, 20.0, 15.3],
                }
            )
        )

    def test_od_to_csv_and_from_csv(self, od_data, tmp_path):
        """Test OD to_csv and from_csv round-trip."""
        csv_path = tmp_path / "od.csv"
        od_data.to_csv(csv_path)

        # Read back with column mapping
        od_reloaded = OD.from_csv(
            csv_path,
            {
                "origin_id": "origin_id",
                "destination_id": "destination_id",
                "flow": "flow",
            },
        )

        pd.testing.assert_frame_equal(
            od_data.to_dataframe(), od_reloaded.to_dataframe()
        )

    def test_od_to_parquet_and_from_parquet(self, od_data, tmp_path):
        """Test OD to_parquet and from_parquet round-trip."""
        parquet_path = tmp_path / "od.parquet"
        od_data.to_parquet(parquet_path)

        # Read back with column mapping
        od_reloaded = OD.from_parquet(
            parquet_path,
            {
                "origin_id": "origin_id",
                "destination_id": "destination_id",
                "flow": "flow",
            },
        )

        pd.testing.assert_frame_equal(
            od_data.to_dataframe().sort_index(axis=1),
            od_reloaded.to_dataframe().sort_index(axis=1),
        )

    def test_od_to_file_geojson(self, od_data, tmp_path):
        """Test OD to_file with GeoJSON format."""
        geojson_path = tmp_path / "od.geojson"
        od_data.to_file(geojson_path)

        # Verify file exists
        assert geojson_path.exists()

    def test_od_from_file_geojson(self, od_data, tmp_path):
        """Test OD from_file with GeoJSON format."""
        # First write data to GeoJSON
        geojson_path = tmp_path / "od.geojson"
        od_data.to_file(geojson_path)

        # Read back with column mapping
        od_reloaded = OD.from_file(
            geojson_path,
            {
                "origin_id": "origin_id",
                "destination_id": "destination_id",
                "flow": "flow",
            },
        )

        # Compare data (note: from_file drops geometry column)
        # Also need to handle column order differences
        pd.testing.assert_frame_equal(
            od_data.to_dataframe().sort_index(axis=1),
            od_reloaded.to_dataframe().sort_index(axis=1),
        )

    def test_od_from_parquet_with_column_mapping(self, tmp_path):
        """Test OD from_parquet with column mapping."""
        # Create parquet with different column names
        parquet_path = tmp_path / "od_renamed.parquet"
        pd.DataFrame(
            {
                "src": ["A", "B"],
                "dst": ["B", "C"],
                "vol": [10.0, 20.0],
            }
        ).to_parquet(parquet_path)

        od = OD.from_parquet(
            parquet_path,
            {
                "src": "origin_id",
                "dst": "destination_id",
                "vol": "flow",
            },
        )

        df = od.to_dataframe()
        assert set(df.columns) == {"origin_id", "destination_id", "flow"}
        assert len(df) == 2


# Tests for Network I/O methods
class TestNetworkIO:
    @pytest.fixture
    def network_data(self):
        return Network(
            pd.DataFrame(
                {
                    "edge_from": ["A", "B", "C"],
                    "edge_to": ["B", "C", "A"],
                    "edge_id": ["E1", "E2", "E3"],
                    "capacity": [100, 150, 80],
                    "cost": [10.5, 20.0, 15.3],
                }
            )
        )

    def test_network_to_csv_and_from_csv(self, network_data, tmp_path):
        """Test Network to_csv and from_csv round-trip."""
        csv_path = tmp_path / "network.csv"
        network_data.to_csv(csv_path)

        network_reloaded = Network.from_csv(
            csv_path,
            {
                "edge_from": "edge_from",
                "edge_to": "edge_to",
                "edge_id": "edge_id",
                "capacity": "capacity",
                "cost": "cost",
            },
        )

        pd.testing.assert_frame_equal(
            network_data.to_dataframe(), network_reloaded.to_dataframe()
        )

    def test_network_to_parquet_and_from_parquet(self, network_data, tmp_path):
        """Test Network to_parquet and from_parquet round-trip."""
        parquet_path = tmp_path / "network.parquet"
        network_data.to_parquet(parquet_path)

        network_reloaded = Network.from_parquet(
            parquet_path,
            {
                "edge_from": "edge_from",
                "edge_to": "edge_to",
                "edge_id": "edge_id",
                "capacity": "capacity",
                "cost": "cost",
            },
        )

        pd.testing.assert_frame_equal(
            network_data.to_dataframe().sort_index(axis=1),
            network_reloaded.to_dataframe().sort_index(axis=1),
        )

    def test_network_to_file_geojson(self, network_data, tmp_path):
        """Test Network to_file with GeoJSON format."""
        geojson_path = tmp_path / "network.geojson"
        network_data.to_file(geojson_path)

        # Verify file exists
        assert geojson_path.exists()

    def test_network_from_file_geojson(self, network_data, tmp_path):
        """Test Network from_file with GeoJSON format."""
        # First write data to GeoJSON
        geojson_path = tmp_path / "network.geojson"
        network_data.to_file(geojson_path)

        # Read back with column mapping
        network_reloaded = Network.from_file(
            geojson_path,
            {
                "edge_from": "edge_from",
                "edge_to": "edge_to",
                "edge_id": "edge_id",
                "capacity": "capacity",
                "cost": "cost",
            },
        )

        pd.testing.assert_frame_equal(
            network_data.to_dataframe().sort_index(axis=1),
            network_reloaded.to_dataframe().sort_index(axis=1),
            check_dtype=False,  # GeoJSON may change int64 to int32
        )

    def test_network_from_file_with_column_mapping(self, tmp_path):
        """Test Network from_file with column mapping."""
        # Create a GeoJSON-like file with different column names
        pd.DataFrame(
            {
                "from_node": ["A", "B"],
                "to_node": ["B", "C"],
                "link_id": ["L1", "L2"],
                "cap": [100, 150],
                "expense": [10.0, 20.0],
            }
        ).to_csv(tmp_path / "temp.csv", index=False)

        network = Network.from_csv(
            tmp_path / "temp.csv",
            {
                "from_node": "edge_from",
                "to_node": "edge_to",
                "link_id": "edge_id",
                "cap": "capacity",
                "expense": "cost",
            },
        )

        df = network.to_dataframe()
        assert list(df.columns) == [
            "edge_from",
            "edge_to",
            "edge_id",
            "capacity",
            "cost",
        ]
        assert len(df) == 2


# Tests for ODFlows I/O methods
class TestODFlowsIO:
    @pytest.fixture
    def od_flows_data(self):
        return ODFlows(
            pd.DataFrame(
                {
                    "origin_id": ["A", "B", "C"],
                    "destination_id": ["B", "C", "A"],
                    "flow": [10.5, 20.0, 15.3],
                    "edge_path": [["E1"], ["E2", "E3"], ["E1", "E2"]],
                }
            )
        )

    def test_odflows_to_csv(self, od_flows_data, tmp_path):
        """Test ODFlows to_csv."""
        csv_path = tmp_path / "od_flows.csv"
        od_flows_data.to_csv(csv_path)

        # Read back and verify structure
        df = pd.read_csv(csv_path)
        assert len(df) == 3
        assert set(df.columns) == {"origin_id", "destination_id", "flow", "edge_path"}

    def test_odflows_to_parquet(self, od_flows_data, tmp_path):
        """Test ODFlows to_parquet."""
        parquet_path = tmp_path / "od_flows.parquet"
        od_flows_data.to_parquet(parquet_path)

        # Read back with pandas
        df = pd.read_parquet(parquet_path)
        assert len(df) == 3
        assert set(df.columns) == {"origin_id", "destination_id", "flow", "edge_path"}

    def test_odflows_to_file_geojson(self, od_flows_data, tmp_path):
        """Test ODFlows to_file with GeoJSON format."""
        geojson_path = tmp_path / "od_flows.geojson"
        od_flows_data.to_file(geojson_path)

        # Verify file exists
        assert geojson_path.exists()


# Tests for NetworkFlows I/O methods
class TestNetworkFlowsIO:
    @pytest.fixture
    def network_flows_data(self):
        from transport_flow_model.model import NetworkFlows

        return NetworkFlows(
            pd.DataFrame(
                {
                    "edge_id": ["E1", "E2", "E3"],
                    "edge_from": ["A", "B", "C"],
                    "edge_to": ["B", "C", "A"],
                    "flow": [35.8, 35.3, 10.5],
                    "capacity": [100, 150, 80],
                }
            )
        )

    def test_networkflows_to_csv(self, network_flows_data, tmp_path):
        """Test NetworkFlows to_csv."""
        csv_path = tmp_path / "network_flows.csv"
        network_flows_data.to_csv(csv_path)

        # Read back and verify structure
        df = pd.read_csv(csv_path)
        assert len(df) == 3
        assert "edge_id" in df.columns
        assert "flow" in df.columns

    def test_networkflows_to_parquet(self, network_flows_data, tmp_path):
        """Test NetworkFlows to_parquet."""
        parquet_path = tmp_path / "network_flows.parquet"
        network_flows_data.to_parquet(parquet_path)

        # Read back with pandas
        df = pd.read_parquet(parquet_path)
        assert len(df) == 3
        assert "edge_id" in df.columns
        assert "flow" in df.columns

    def test_networkflows_to_file_geojson(self, network_flows_data, tmp_path):
        """Test NetworkFlows to_file with GeoJSON format."""
        geojson_path = tmp_path / "network_flows.geojson"
        network_flows_data.to_file(geojson_path)

        # Verify file exists
        assert geojson_path.exists()
