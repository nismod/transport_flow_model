"""Disruption scenarios that combine link removal with link cost changes."""

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


@pytest.fixture
def links():
    # Baseline A -> D uses AB, BD (cost 2). Detours: A-C-D (cost 4) and the
    # direct AD (cost 5). C -> D uses CD (cost 2).
    return pd.DataFrame(
        {
            "edge_from": ["A", "B", "A", "C", "A"],
            "edge_to": ["B", "D", "C", "D", "D"],
            "edge_id": ["AB", "BD", "AC", "CD", "AD"],
            "cost": [1, 1, 2, 2, 5],
            "capacity": [100, 100, 100, 100, 100],
        }
    )


@pytest.fixture
def od():
    return pd.DataFrame(
        {
            "origin_id": ["A", "C"],
            "destination_id": ["D", "D"],
            "value": [10.0, 3.0],
        }
    )


def _run(links, od, scenarios, **kwargs):
    network = Network(links)
    base = assign(network, Demand(od), include_paths=True, capacity_constrained=True)
    return disrupt(network, scenarios, base=base, capacity_constrained=True, **kwargs)


def _records(table, sort=("origin_id", "destination_id")):
    frame = table.to_pandas()
    if "edge_path" in frame:
        frame["edge_path"] = frame["edge_path"].map(list)
    return frame.sort_values(list(sort)).to_dict("records")


def _link_flows(result):
    return result.link_flows.to_pandas().set_index("edge_id")["flow"].to_dict()


def test_baseline_paths(links, od):
    base = assign(Network(links), Demand(od), include_paths=True)
    assert _records(base.paths) == [
        {
            "origin_id": "A",
            "destination_id": "D",
            "flow": 10,
            "edge_path": ["AB", "BD"],
            "cost": 2,
        },
        {
            "origin_id": "C",
            "destination_id": "D",
            "flow": 3,
            "edge_path": ["CD"],
            "cost": 2,
        },
    ]


def test_removal_only_unchanged(links, od):
    (result,) = _run(links, od, [Scenario.remove_links("s", ["BD"])])
    assert _records(result.rerouted) == [
        {
            "origin_id": "A",
            "destination_id": "D",
            "flow": 10,
            "edge_path": ["AC", "CD"],
            "cost": 4,
        }
    ]
    assert result.repriced.num_rows == 0
    assert result.repriced.column_names == [
        "origin_id",
        "destination_id",
        "flow",
        "edge_path",
        "cost",
    ]


def test_removal_with_cost_change_changes_detour(links, od):
    # CD becomes expensive, so the detour for A -> D is the direct AD link;
    # C -> D keeps its path on CD at the new cost.
    scenario = Scenario.from_changes("s", remove=["BD"], cost={"CD": 10})
    results = _run(links, od, [scenario])
    (result,) = results.results

    assert _records(result.rerouted) == [
        {
            "origin_id": "A",
            "destination_id": "D",
            "flow": 10,
            "edge_path": ["AD"],
            "cost": 5,
        }
    ]
    assert _records(result.repriced) == [
        {
            "origin_id": "C",
            "destination_id": "D",
            "flow": 3,
            "edge_path": ["CD"],
            "cost": 10,
        }
    ]
    assert result.isolated.num_rows == 0
    assert _records(result.losses) == [
        {
            "origin_id": "A",
            "destination_id": "D",
            "flow": 10,
            "initial_cost": 2,
            "disrupted_cost": 5,
            "rerouting_loss": 3,
        },
        {
            "origin_id": "C",
            "destination_id": "D",
            "flow": 3,
            "initial_cost": 2,
            "disrupted_cost": 10,
            "rerouting_loss": 8,
        },
    ]
    assert _link_flows(result) == {"AB": 0, "BD": 0, "AC": 0, "CD": 3, "AD": 10}
    # The scenario link table carries the changed cost.
    costs = result.link_flows.to_pandas().set_index("edge_id")["cost"].to_dict()
    assert costs == {"AB": 1, "BD": 1, "AC": 2, "CD": 10, "AD": 5}

    summary = results.summary().to_pylist()
    assert summary == [
        {
            "scenario_id": "s",
            "rerouted_flow": 10.0,
            "isolated_flow": 0.0,
            "rerouting_loss": 11.0,
            "repriced_flow": 3.0,
        }
    ]


def test_initial_cost_is_baseline_path_cost(links, od):
    # AB (on the A -> D baseline path) gets more expensive and BD is removed:
    # the flow is rerouted (removal wins over repricing) and its initial cost
    # stays the baseline path cost 2, not one recomputed from changed costs.
    scenario = Scenario.from_changes("s", remove=["BD"], cost_scale={"AB": 3})
    (result,) = _run(links, od, [scenario])
    assert result.repriced.num_rows == 0
    assert _records(result.losses) == [
        {
            "origin_id": "A",
            "destination_id": "D",
            "flow": 10,
            "initial_cost": 2,
            "disrupted_cost": 4,
            "rerouting_loss": 2,
        }
    ]


def test_cost_only_scenario_reprices_kept_paths(links, od):
    results = _run(links, od, [Scenario.from_changes("s", cost_scale={"CD": 2})])
    (result,) = results.results
    assert result.rerouted.num_rows == 0
    assert result.isolated.num_rows == 0
    assert _records(result.repriced) == [
        {
            "origin_id": "C",
            "destination_id": "D",
            "flow": 3,
            "edge_path": ["CD"],
            "cost": 4,
        }
    ]
    assert _records(result.losses) == [
        {
            "origin_id": "C",
            "destination_id": "D",
            "flow": 3,
            "initial_cost": 2,
            "disrupted_cost": 4,
            "rerouting_loss": 2,
        }
    ]
    # Baseline flows are unchanged.
    assert _link_flows(result) == {"AB": 10, "BD": 10, "AC": 0, "CD": 3, "AD": 0}
    assert results.summary().to_pylist()[0]["repriced_flow"] == 3.0


def test_reroute_changed_moves_flow_off_expensive_link(links, od):
    scenario = Scenario.from_changes("s", cost={"BD": 10})

    (kept,) = _run(links, od, [scenario])
    assert _records(kept.repriced) == [
        {
            "origin_id": "A",
            "destination_id": "D",
            "flow": 10,
            "edge_path": ["AB", "BD"],
            "cost": 11,
        }
    ]
    assert kept.losses.to_pylist()[0]["rerouting_loss"] == 9
    assert _link_flows(kept)["BD"] == 10

    results = _run(links, od, [scenario], reroute_changed=True)
    (moved,) = results.results
    assert moved.repriced.num_rows == 0
    assert _records(moved.rerouted) == [
        {
            "origin_id": "A",
            "destination_id": "D",
            "flow": 10,
            "edge_path": ["AC", "CD"],
            "cost": 4,
        }
    ]
    assert _records(moved.losses) == [
        {
            "origin_id": "A",
            "destination_id": "D",
            "flow": 10,
            "initial_cost": 2,
            "disrupted_cost": 4,
            "rerouting_loss": 2,
        }
    ]
    assert _link_flows(moved) == {"AB": 0, "BD": 0, "AC": 10, "CD": 13, "AD": 0}
    assert results.summary().to_pylist()[0]["repriced_flow"] == 0.0


def test_reroute_changed_keeps_link_available(links, od):
    # BD gets dearer but its path is still the cheapest: the rerouted flow
    # goes back onto it.
    scenario = Scenario.from_changes("s", cost={"BD": 2})
    (result,) = _run(links, od, [scenario], reroute_changed=True)
    assert _records(result.rerouted)[0]["edge_path"] == ["AB", "BD"]
    assert _records(result.rerouted)[0]["cost"] == 3
    assert _link_flows(result)["BD"] == 10


def test_scale_matches_value(links, od):
    by_value = _run(
        links, od, [Scenario.from_changes("s", remove=["BD"], cost={"CD": 4})]
    )
    by_scale = _run(
        links, od, [Scenario.from_changes("s", remove=["BD"], cost_scale={"CD": 2})]
    )
    (value_result,) = by_value.results
    (scale_result,) = by_scale.results
    for name in ("rerouted", "repriced", "losses", "link_flows", "isolated"):
        assert getattr(value_result, name).equals(getattr(scale_result, name)), name
    # CD at 4 makes A-C-D (cost 6) dearer than AD (cost 5).
    assert _records(value_result.rerouted)[0]["edge_path"] == ["AD"]


def test_link_delta_constructors_equivalent():
    scenario = Scenario.from_changes(
        "s", remove=["BD"], cost={"CD": 4}, cost_scale={"AC": 0.5}, hazard="flood"
    )
    assert scenario.deltas == (
        LinkDelta("BD", "cost", value=math.inf),
        LinkDelta("CD", "cost", value=4.0),
        LinkDelta("AC", "cost", scale=0.5),
    )
    assert scenario.removed_links == ("BD",)
    assert scenario.changed_links == ("CD", "AC")
    assert scenario.metadata == {"hazard": "flood"}


def test_capacity_constrained_with_kept_flows(links, od):
    # CD has capacity 6, 3 of it held by the kept (repriced) C -> D flow, so
    # only 3 of the rerouted A -> D flow fits on A-C-D; the rest takes AD.
    links = links.assign(
        cost=[1, 1, 2, 2, 8],
        capacity=[100, 100, 100, 6, 100],
    )
    scenario = Scenario.from_changes("s", remove=["BD"], cost={"CD": 3})
    (result,) = _run(links, od, [scenario])
    assert _records(result.rerouted, sort=("cost",)) == [
        {
            "origin_id": "A",
            "destination_id": "D",
            "flow": 3,
            "edge_path": ["AC", "CD"],
            "cost": 5,
        },
        {
            "origin_id": "A",
            "destination_id": "D",
            "flow": 7,
            "edge_path": ["AD"],
            "cost": 8,
        },
    ]
    assert _records(result.repriced)[0]["flow"] == 3
    assert _records(result.repriced)[0]["cost"] == 3
    assert result.isolated.num_rows == 0
    assert _link_flows(result) == {"AB": 0, "BD": 0, "AC": 3, "CD": 6, "AD": 7}


def test_capacity_constrained_reroute_changed_respects_capacity(links, od):
    # The A -> D flow (released by removing BD) and the C -> D flow (released
    # because CD's cost changed) compete for CD's capacity of 6.
    links = links.assign(capacity=[100, 100, 100, 6, 100], cost=[1, 1, 2, 2, 50])
    scenario = Scenario.from_changes("s", remove=["BD"], cost_scale={"CD": 1.5})
    (result,) = _run(links, od, [scenario], reroute_changed=True)
    flows = _link_flows(result)
    assert flows["CD"] <= 6 + 1e-9
    total = result.rerouted.to_pandas()["flow"].sum()
    isolated = result.isolated.to_pandas()["value"].sum()
    assert total + isolated == pytest.approx(13.0)
    assert result.repriced.num_rows == 0
    assert total < 13.0  # CD cannot carry everything


def test_unconstrained_capacity_ignored(links, od):
    links = links.assign(capacity=[100, 100, 100, 1, 100])
    network = Network(links)
    base = assign(network, Demand(od), include_paths=True)
    (result,) = disrupt(
        network,
        [Scenario.from_changes("s", remove=["BD"], cost={"AD": 3})],
        base=base,
        capacity_constrained=False,
    ).results
    assert _records(result.rerouted)[0]["edge_path"] == ["AD"]
    assert _records(result.rerouted)[0]["cost"] == 3


def test_skip_unaffected_considers_changed_links(links, od):
    scenarios = [
        Scenario.from_changes("no-flow", cost={"AD": 1}),
        Scenario.from_changes("flow", cost={"CD": 1}),
        Scenario.from_changes("no-flow-removed", remove=["AD"], cost={"AC": 9}),
        Scenario.from_changes("unknown", cost={"XX": 1}),
    ]
    results = _run(links, od, scenarios)
    assert [s.id for s in results.skipped] == ["no-flow", "no-flow-removed", "unknown"]
    assert [r.scenario.id for r in results.results] == ["flow"]

    results = _run(links, od, scenarios, skip_unaffected=False)
    assert len(results.results) == 4
    by_id = {r.scenario.id: r for r in results.results}
    assert by_id["no-flow"].repriced.num_rows == 0
    assert by_id["no-flow"].losses.num_rows == 0
    # A link that got cheaper: kept paths with a negative loss.
    assert by_id["flow"].losses.to_pylist()[0]["rerouting_loss"] == -1
    summary = results.summary().to_pandas().set_index("scenario_id")
    assert summary.loc["no-flow", "repriced_flow"] == 0
    assert summary.loc["flow", "repriced_flow"] == 3


@pytest.mark.parametrize(
    "kwargs",
    [
        {"value": -1.0},
        {"value": math.nan},
        {"value": -math.inf},
        {"scale": -0.5},
        {"scale": math.inf},
        {"scale": math.nan},
    ],
)
def test_invalid_cost_deltas(kwargs):
    with pytest.raises(ValueError, match="cost"):
        LinkDelta("CD", "cost", **kwargs)


def test_conflicting_deltas():
    with pytest.raises(ValueError, match="both removes"):
        Scenario.from_changes("s", remove=["CD"], cost={"CD": 1})
    with pytest.raises(ValueError, match="more than one cost change"):
        Scenario.from_changes("s", cost={"CD": 1}, cost_scale={"CD": 2})


def test_unsupported_attribute_and_id_type(links, od):
    with pytest.raises(NotImplementedError, match="capacity"):
        _run(
            links,
            od,
            [Scenario("s", (LinkDelta("CD", "capacity", scale=0.5),))],
        )
    with pytest.raises(ValueError, match="edge_id type"):
        _run(links, od, [Scenario.from_changes("s", cost={1: 2.0})])


def test_invalid_scenario_fails_before_evaluating(links, od):
    scenarios = [
        Scenario.remove_links("ok", ["BD"]),
        Scenario("bad", (LinkDelta("CD", "capacity", value=1.0),)),
    ]
    with pytest.raises(NotImplementedError):
        _run(links, od, scenarios)


def test_network_without_cost_column(od):
    # The core defaults missing costs to 1; scale applies to that default.
    links = pd.DataFrame(
        {
            "edge_from": ["A", "B", "A"],
            "edge_to": ["B", "D", "D"],
            "edge_id": ["AB", "BD", "AD"],
        }
    )
    od = od.iloc[:1]
    network = Network(links)
    base = assign(network, Demand(od), include_paths=True)
    assert base.paths.to_pylist()[0]["edge_path"] == ["AD"]
    (result,) = disrupt(
        network, [Scenario.from_changes("s", cost_scale={"AD": 4})], base=base
    ).results
    assert result.repriced.to_pylist()[0]["cost"] == 4
    (moved,) = disrupt(
        network,
        [Scenario.from_changes("s", cost_scale={"AD": 4})],
        base=base,
        reroute_changed=True,
    ).results
    assert moved.rerouted.to_pylist()[0]["edge_path"] == ["AB", "BD"]
    assert moved.losses.to_pylist()[0]["rerouting_loss"] == 1
