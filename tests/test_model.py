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
