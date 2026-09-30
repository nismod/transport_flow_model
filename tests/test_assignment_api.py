# %%
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


def test_congestion_assignment_penalizes_saturated_route():
    network = Network(
        pd.DataFrame(
            {
                "edge_from": ["A", "A"],
                "edge_to": ["B", "B"],
                "edge_id": ["fast", "slow"],
                "length_mile": [1.0, 1.0],
                "initial_flow_speeds": [60.0, 40.0],
                "min_flow_speeds": [10.0, 10.0],
                "breakpoint_flows": [0.0, 0.0],
                "combined_label": ["A_single", "A_single"],
                "average_toll_cost": [0.0, 0.0],
                "capacity": [8_000.0, 50_000.0],
            }
        )
    )
    demand = Demand(
        pd.DataFrame({"origin_id": ["A"], "destination_id": ["B"], "value": [24_000.0]})
    )

    result = assign(
        network,
        demand,
        "congestion",
        include_paths=True,
        capacity_constrained=True,
        max_iterations=10,
    )

    flows = dict(
        zip(
            result.link_flows["edge_id"].to_pylist(),
            result.link_flows["flow"].to_pylist(),
        )
    )
    assert flows["fast"] == pytest.approx(8_000.0)
    assert flows["slow"] > 0
    assert sum(flows.values()) == pytest.approx(24_000.0)
    assert result.unassigned.num_rows == 0
    assert result.provenance.iterations == 2


def test_congestion_adjustment_ratio_uses_total_edge_demand():
    network = Network(
        pd.DataFrame(
            {
                "edge_from": ["A", "B", "C"],
                "edge_to": ["B", "C", "D"],
                "edge_id": ["AB", "BC", "CD"],
                "length_mile": [1.0, 1.0, 1.0],
                "initial_flow_speeds": [60.0, 60.0, 60.0],
                "min_flow_speeds": [10.0, 10.0, 10.0],
                "breakpoint_flows": [1_000.0, 1_000.0, 1_000.0],
                "combined_label": ["M", "M", "M"],
                "average_toll_cost": [0.0, 0.0, 0.0],
                "capacity": [1_000.0, 60.0, 1_000.0],
            }
        )
    )
    demand = Demand(
        pd.DataFrame(
            {
                "origin_id": ["A", "B"],
                "destination_id": ["D", "D"],
                "value": [80.0, 40.0],
            }
        )
    )

    result = assign(
        network,
        demand,
        "congestion",
        include_paths=True,
        max_iterations=5,
    )

    paths = result.paths.to_pandas().set_index("origin_id")
    link_flows = dict(
        zip(
            result.link_flows["edge_id"].to_pylist(),
            result.link_flows["flow"].to_pylist(),
        )
    )
    unassigned = dict(
        zip(
            result.unassigned["origin_id"].to_pylist(),
            result.unassigned["value"].to_pylist(),
        )
    )

    assert paths.loc["A", "flow"] == pytest.approx(40.0)
    assert paths.loc["B", "flow"] == pytest.approx(20.0)
    assert link_flows["BC"] == pytest.approx(60.0)
    assert unassigned == {"A": 40.0, "B": 20.0}


def test_congestion_assignment_uses_vehicle_specific_costs():
    network = Network(
        pd.DataFrame(
            {
                "edge_from": ["A"],
                "edge_to": ["B"],
                "edge_id": ["AB"],
                "length_mile": [1.0],
                "initial_flow_speeds": [60.0],
                "min_flow_speeds": [10.0],
                "breakpoint_flows": [100.0],
                "combined_label": ["M"],
                "average_toll_cost": [0.0],
            }
        )
    )
    demand = Demand(
        pd.DataFrame({"origin_id": ["A"], "destination_id": ["B"], "value": [1.0]})
    )

    car = assign(
        network,
        demand,
        "congestion",
        include_paths=True,
        capacity_constrained=False,
        max_iterations=1,
        vehicle_type="car",
    )
    lgv = assign(
        network,
        demand,
        "congestion",
        include_paths=True,
        capacity_constrained=False,
        max_iterations=1,
        vehicle_type="lgv",
    )
    psv = assign(
        network,
        demand,
        "congestion",
        include_paths=True,
        capacity_constrained=False,
        max_iterations=1,
        vehicle_type="psv",
    )

    assert car.skims["cost"][0].as_py() != pytest.approx(lgv.skims["cost"][0].as_py())
    assert psv.skims["cost"][0].as_py() == pytest.approx(13.15 / 60.0)


def test_congestion_assignment_respects_link_capacity():
    network = Network(
        pd.DataFrame(
            {
                "edge_from": ["A"],
                "edge_to": ["B"],
                "edge_id": ["AB"],
                "length_mile": [1.0],
                "initial_flow_speeds": [60.0],
                "min_flow_speeds": [10.0],
                "breakpoint_flows": [100.0],
                "combined_label": ["A_single"],
                "average_toll_cost": [0.0],
                "capacity": [3.0],
            }
        )
    )
    demand = Demand(
        pd.DataFrame({"origin_id": ["A"], "destination_id": ["B"], "value": [5.0]})
    )

    result = assign(network, demand, "congestion", max_iterations=5)

    assert result.link_flows["flow"].to_pylist() == [3]
    assert result.unassigned["value"].to_pylist() == [2]


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
