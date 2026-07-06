import textwrap

import pytest

import transport_flow_model as tfm
from transport_flow_model.io import NETWORK_COLUMNS, read_tntp, read_tntp_flows


@pytest.fixture(scope="module")
def siouxfalls_paths():
    return tfm.datasets.fetch("siouxfalls")


@pytest.fixture(scope="module")
def siouxfalls(siouxfalls_paths):
    return read_tntp(siouxfalls_paths["net"], siouxfalls_paths["trips"])


def test_siouxfalls_metadata(siouxfalls):
    assert siouxfalls.n_zones == 24
    assert siouxfalls.n_nodes == 24
    assert siouxfalls.n_links == 76
    assert siouxfalls.first_thru_node == 1
    assert siouxfalls.centroid_offset is None


def test_siouxfalls_network(siouxfalls):
    network = siouxfalls.network.to_dataframe()
    assert list(network.columns) == list(NETWORK_COLUMNS)
    assert len(network) == 76
    assert network["edge_from"].dtype == "int64"
    assert network["edge_to"].dtype == "int64"
    assert network["edge_id"].is_unique
    # first link: 1 -> 2, capacity 25900.20064, free-flow time 6, BPR 0.15/4
    first = network.iloc[0]
    assert first["edge_from"] == 1
    assert first["edge_to"] == 2
    assert first["capacity"] == pytest.approx(25900.20064)
    assert first["cost"] == 6
    assert first["alpha"] == pytest.approx(0.15)
    assert first["beta"] == 4


def test_siouxfalls_demand_matches_published_total(siouxfalls):
    od = siouxfalls.od.to_dataframe()
    # <TOTAL OD FLOW> 360600.0 in SiouxFalls_trips.tntp
    assert od["flow"].sum() == pytest.approx(360600.0)
    assert len(od) == 528  # nonzero OD pairs
    assert (od["flow"] > 0).all()
    assert od["origin_id"].between(1, 24).all()
    assert od["destination_id"].between(1, 24).all()


def test_siouxfalls_keep_zero_flows(siouxfalls_paths):
    instance = read_tntp(
        siouxfalls_paths["net"], siouxfalls_paths["trips"], keep_zero_flows=True
    )
    assert len(instance.od.to_dataframe()) == 24 * 24


def test_siouxfalls_best_known_flows(siouxfalls_paths):
    flows = read_tntp_flows(siouxfalls_paths["flow"])
    assert list(flows.columns) == ["edge_from", "edge_to", "flow", "cost"]
    assert len(flows) == 76
    # published equilibrium costs satisfy the BPR function of the net file
    assert flows.loc[0, "flow"] == pytest.approx(4494.6576464564205)


def test_siouxfalls_allocates_all_demand(siouxfalls):
    result = siouxfalls.network.allocate(siouxfalls.od)
    assert result.unassigned_od.to_dataframe().empty
    od_flows = result.od_flows.to_dataframe()
    assert od_flows["flow"].sum() == pytest.approx(360600.0)


CENTROID_NET = textwrap.dedent(
    """\
    <NUMBER OF ZONES> 2
    <NUMBER OF NODES> 4
    <FIRST THRU NODE> 3
    <NUMBER OF LINKS> 6
    <END OF METADATA>

    ~ init term capacity length fft b power speed toll type ;
    1 3 100 1 1 0.15 4 0 0 1 ;
    3 1 100 1 1 0.15 4 0 0 1 ;
    2 3 100 1 1 0.15 4 0 0 1 ;
    3 2 100 1 1 0.15 4 0 0 1 ;
    3 4 100 1 1 0.15 4 0 0 1 ;
    4 3 100 1 1 0.15 4 0 0 1 ;
    """
)

CENTROID_TRIPS = textwrap.dedent(
    """\
    <NUMBER OF ZONES> 2
    <TOTAL OD FLOW> 15.0
    <END OF METADATA>

    Origin 1
    2 : 10.0;
    Origin 2
    1 : 5.0;
    """
)


@pytest.fixture()
def centroid_instance(tmp_path):
    net = tmp_path / "toy_net.tntp"
    trips = tmp_path / "toy_trips.tntp"
    net.write_text(CENTROID_NET)
    trips.write_text(CENTROID_TRIPS)
    return read_tntp(net, trips)


def test_first_thru_node_splits_centroids(centroid_instance):
    instance = centroid_instance
    assert instance.centroid_offset == 4
    network = instance.network.to_dataframe()
    # links out of zones keep their tail id; links into zones are remapped so
    # no path can pass through a centroid
    assert set(network["edge_from"]) == {1, 2, 3, 4}
    assert set(network["edge_to"]) == {3, 4, 1 + 4, 2 + 4}
    # remapped centroid heads have no outgoing links
    assert not set(network["edge_to"]) & set(network["edge_from"]) & {5, 6}


def test_first_thru_node_remaps_destinations(centroid_instance):
    od = centroid_instance.od.to_dataframe()
    assert od["origin_id"].tolist() == [1, 2]
    assert od["destination_id"].tolist() == [2 + 4, 1 + 4]
    assert centroid_instance.zone_ids(od["destination_id"]).tolist() == [2, 1]


def test_first_thru_node_allocation_avoids_centroids(centroid_instance):
    instance = centroid_instance
    result = instance.network.allocate(instance.od)
    assert result.unassigned_od.to_dataframe().empty
    network = instance.network.to_dataframe()
    edge_nodes = network.set_index("edge_id")[["edge_from", "edge_to"]]
    for row in result.od_flows.to_dataframe().itertuples(index=False):
        interior = [
            edge_nodes.loc[edge_id, "edge_from"] for edge_id in row.edge_path[1:]
        ]
        assert all(node >= instance.first_thru_node for node in interior)


def test_split_centroids_can_be_disabled(tmp_path):
    net = tmp_path / "toy_net.tntp"
    net.write_text(CENTROID_NET)
    instance = read_tntp(net, split_centroids=False)
    assert instance.centroid_offset is None
    network = instance.network.to_dataframe()
    assert set(network["edge_to"]) == {1, 2, 3, 4}


def test_read_tntp_without_trips(siouxfalls_paths):
    instance = read_tntp(siouxfalls_paths["net"])
    assert instance.od is None
    assert len(instance.network.to_dataframe()) == 76
