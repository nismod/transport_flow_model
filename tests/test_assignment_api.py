import pandas as pd
import pytest

from transport_flow_model import Demand, Network, assign
from transport_flow_model.model import OD
from transport_flow_model.model import Network as LegacyNetwork


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


@pytest.fixture
def od():
    return pd.DataFrame(
        {
            "origin_id": ["A", "A", "B"],
            "destination_id": ["B", "C", "D"],
            "value": [30.0, 90.0, 100.0],
        }
    )


def test_sequential_matches_legacy_allocate(links, od):
    result = assign(
        Network(links),
        Demand(od),
        "sequential",
        include_paths=True,
        capacity_constrained=True,
        directed=True,
    )
    legacy = LegacyNetwork(links).allocate(
        OD(od.rename(columns={"value": "flow"})),
        capacity_constrained=True,
        directed=True,
    )

    pd.testing.assert_frame_equal(
        result.paths.to_pandas().assign(edge_path=lambda d: d.edge_path.map(list)),
        legacy.od_flows.to_dataframe(),
    )
    pd.testing.assert_frame_equal(
        result.unassigned.to_pandas().rename(columns={"value": "flow"}),
        legacy.unassigned_od.to_dataframe(),
    )
    pd.testing.assert_frame_equal(
        result.link_flows.to_pandas(),
        legacy.network_flows.to_dataframe(),
    )


def test_link_flows_cover_all_links(links, od):
    result = assign(Network(links), Demand(od), capacity_constrained=True)
    link_flows = result.link_flows.to_pandas()
    assert list(link_flows.edge_id) == ["XX", "YY", "ZZ", "AA"]
    assert list(link_flows.flow) == [100, 25, 50, 50]


def test_skims_flow_weighted_cost(links, od):
    result = assign(Network(links), Demand(od), capacity_constrained=True)
    skims = result.skims.to_pandas().set_index(["origin_id", "destination_id"])
    # B->D splits 50 units at cost 5 and 50 at cost 50: weighted mean 27.5
    assert skims.loc[("B", "D"), "cost"] == pytest.approx(27.5)
    assert skims.loc[("A", "C"), "cost"] == pytest.approx(20.0)


def test_paths_omitted_by_default(links, od):
    result = assign(Network(links), Demand(od))
    assert result.paths is None


def test_provenance_recorded(links, od):
    result = assign(Network(links), Demand(od), seed=42, capacity_constrained=True)
    provenance = result.provenance
    assert provenance.method == "sequential"
    assert provenance.options == {"capacity_constrained": True}
    assert provenance.iterations == 1
    assert provenance.seed == 42
    assert provenance.wall_time_s > 0
    assert provenance.package_version
    assert provenance.core_version
    assert result.gap_history == ()


def test_unknown_method_raises(links, od):
    with pytest.raises(ValueError, match="Unknown assignment method"):
        assign(Network(links), Demand(od), "nope")


@pytest.mark.parametrize("method", ["msa", "fw", "bfw", "staq"])
def test_planned_methods_not_implemented(links, od, method):
    with pytest.raises(NotImplementedError, match=method):
        assign(Network(links), Demand(od), method)


def test_uncapacitated_assignment(links, od):
    result = assign(Network(links), Demand(od), include_paths=True)
    # without capacity constraints everything goes on least-cost paths
    assert result.unassigned.num_rows == 0
    flows = dict(
        zip(
            result.link_flows["edge_id"].to_pylist(),
            result.link_flows["flow"].to_pylist(),
        )
    )
    assert flows["ZZ"] == 100
    assert flows["AA"] == 0
