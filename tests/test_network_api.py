import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from transport_flow_model import Network, datasets


@pytest.fixture
def links():
    return pd.DataFrame(
        {
            "edge_from": ["A", "C", "B", "B"],
            "edge_to": ["C", "B", "D", "D"],
            "edge_id": ["XX", "YY", "ZZ", "AA"],
            "capacity": [100, 100, 50, 100],
            "cost": [20, 10, 5, 50],
        }
    )


def test_construction_and_counts(links):
    network = Network(links)
    assert network.n_links == 4
    assert network.n_nodes == 4
    assert network.link_ids.to_pylist() == ["XX", "YY", "ZZ", "AA"]
    assert network.node_ids.to_pylist() == ["A", "C", "B", "D"]


def test_construction_from_table(links):
    network = Network(pa.Table.from_pandas(links))
    assert network.n_links == 4


def test_column_mapping():
    source = pd.DataFrame(
        {
            "from_id": ["A"],
            "to_id": ["B"],
            "id": ["E1"],
            "flow_capacity": [10],
        }
    )
    network = Network(
        source,
        columns={
            "from_id": "edge_from",
            "to_id": "edge_to",
            "id": "edge_id",
            "flow_capacity": "capacity",
        },
    )
    assert network.to_table().column_names == [
        "edge_from",
        "edge_to",
        "edge_id",
        "capacity",
    ]


def test_missing_columns_raise(links):
    with pytest.raises(ValueError, match="required link columns"):
        Network(links.drop(columns=["edge_id"]))


def test_duplicate_edge_ids_raise(links):
    links.loc[1, "edge_id"] = "XX"
    with pytest.raises(ValueError, match="unique"):
        Network(links)


def test_geometry_column_dropped(links):
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import LineString

    geo_links = gpd.GeoDataFrame(
        links, geometry=[LineString([(0, 0), (1, 1)])] * len(links)
    )
    network = Network.from_dataframe(geo_links)
    assert "geometry" not in network.to_table().column_names


def test_csr_adjacency(links):
    network = Network(links)
    csr = network.csr
    # node order of first appearance: A=0, C=1, B=2, D=3
    np.testing.assert_array_equal(csr.indptr, [0, 1, 2, 4, 4])
    # A -> C via XX (link 0); C -> B via YY (link 1); B -> D via ZZ, AA
    np.testing.assert_array_equal(csr.heads, [1, 2, 3, 3])
    np.testing.assert_array_equal(csr.links, [0, 1, 2, 3])
    with pytest.raises(ValueError):
        csr.indptr[0] = 99  # arrays are read-only


def test_node_and_link_index(links):
    network = Network(links)
    np.testing.assert_array_equal(network.node_index(["D", "A", "nope"]), [3, 0, -1])
    np.testing.assert_array_equal(network.link_index(["ZZ", "nope"]), [2, -1])


def test_attribute_access(links):
    network = Network(links)
    assert network.attribute("cost").to_pylist() == [20, 10, 5, 50]
    with pytest.raises(KeyError):
        network.attribute("nope")


def test_table_is_shared_not_copied(links):
    network = Network(links)
    assert network.to_table() is network.to_table()


def test_from_tntp_instance():
    instance = datasets.load_tntp("siouxfalls")
    network = Network.from_tntp(instance)
    assert network.n_links == instance.n_links
    assert "cost" in network.to_table().column_names
    assert "capacity" in network.to_table().column_names
