import math

import pandas as pd
import pytest

from transport_flow_model import (
    Demand,
    LinkDelta,
    Network,
    Scenario,
    assign,
    disrupt,
)
from transport_flow_model.model import Network as LegacyNetwork
from transport_flow_model.model import ODFlows


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


def test_link_delta_validation():
    with pytest.raises(ValueError, match="exactly one"):
        LinkDelta("XX", "cost")
    with pytest.raises(ValueError, match="exactly one"):
        LinkDelta("XX", "cost", value=1.0, scale=2.0)
    assert LinkDelta("XX", "cost", value=math.inf).is_removal
    assert not LinkDelta("XX", "cost", value=10.0).is_removal
    assert not LinkDelta("XX", "capacity", scale=0.5).is_removal


def test_scenario_remove_links():
    scenario = Scenario.remove_links("s1", ["XX", "YY"], hazard="flood")
    assert scenario.removed_links == ("XX", "YY")
    assert scenario.metadata == {"hazard": "flood"}


def test_disrupt_matches_legacy(links, od):
    network = Network(links)
    demand = Demand(od)
    base = assign(network, demand, include_paths=True, capacity_constrained=True)
    results = disrupt(
        network,
        [Scenario.remove_links("ZZ", ["ZZ"])],
        base=base,
        capacity_constrained=True,
    )
    (result,) = results.results

    legacy_paths = base.paths.to_pandas().assign(
        edge_path=lambda d: d.edge_path.map(list)
    )
    legacy_network = LegacyNetwork(base.link_flows.to_pandas())
    legacy = legacy_network.disrupt(
        ODFlows(legacy_paths),
        failed_edges=["ZZ"],
        capacity_constrained=True,
        directed=True,
    )

    pd.testing.assert_frame_equal(
        result.rerouted.to_pandas().assign(edge_path=lambda d: d.edge_path.map(list)),
        legacy.rerouted_flows.to_dataframe(),
    )
    pd.testing.assert_frame_equal(
        result.isolated.to_pandas().rename(columns={"value": "flow"}),
        legacy.isolated_od.to_dataframe(),
    )
    pd.testing.assert_frame_equal(
        result.losses.to_pandas(),
        legacy.losses.to_dataframe(),
    )


def test_disrupt_computes_base_from_demand(links, od):
    results = disrupt(
        Network(links),
        [Scenario.remove_links("ZZ", ["ZZ"])],
        Demand(od),
        capacity_constrained=True,
    )
    assert results.base.paths is not None
    assert len(results.results) == 1


def test_skip_unaffected_scenarios(links, od):
    links.loc[3, "capacity"] = 0  # nothing can use AA
    results = disrupt(
        Network(links),
        [
            Scenario.remove_links("AA", ["AA"]),
            Scenario.remove_links("ZZ", ["ZZ"]),
        ],
        Demand(od),
        capacity_constrained=True,
    )
    assert [s.id for s in results.skipped] == ["AA"]
    assert [r.scenario.id for r in results.results] == ["ZZ"]


def test_no_skip_when_disabled(links, od):
    links.loc[3, "capacity"] = 0
    results = disrupt(
        Network(links),
        [Scenario.remove_links("AA", ["AA"])],
        Demand(od),
        skip_unaffected=False,
        capacity_constrained=True,
    )
    assert results.skipped == ()
    assert len(results.results) == 1
    assert results.results[0].rerouted.num_rows == 0


def test_summary_table(links, od):
    results = disrupt(
        Network(links),
        [
            Scenario.remove_links("XX", ["XX"]),
            Scenario.remove_links("ZZ", ["ZZ"]),
        ],
        Demand(od),
        capacity_constrained=True,
    )
    summary = results.summary().to_pandas().set_index("scenario_id")
    # removing XX isolates everything from A; removing ZZ reroutes B->D via AA
    assert summary.loc["XX", "isolated_flow"] == pytest.approx(100.0)
    assert summary.loc["ZZ", "rerouted_flow"] == pytest.approx(50.0)
    assert summary.loc["ZZ", "rerouting_loss"] == pytest.approx(45.0)


def test_non_removal_deltas_not_implemented(links, od):
    scenario = Scenario("s1", (LinkDelta("XX", "capacity", scale=0.5),))
    with pytest.raises(NotImplementedError, match="non-removal"):
        disrupt(Network(links), [scenario], Demand(od))


def test_requires_base_or_demand(links):
    with pytest.raises(ValueError, match="base result or demand"):
        disrupt(Network(links), [Scenario.remove_links("XX", ["XX"])])
