import pandas as pd
import pyarrow as pa
import pytest

from transport_flow_model import core


def test_extension_allocate():
    network = pd.DataFrame(
        {
            "edge_from": [0, 0, 2],
            "edge_to": [1, 2, 1],
            "edge_id": [0, 1, 2],
            "cost": [10.0, 3.0, 5.0],
            "capacity": [100.0, 100.0, 100.0],
        }
    )
    od = pd.DataFrame(
        {
            "origin_id": [0],
            "destination_id": [1],
            "flow": [7.0],
        }
    )

    result = core.allocate(network, od, directed=True)
    od_flows = result["od_flows"].to_pandas()
    network_flows = result["network_flows"].to_pandas()

    assert list(od_flows.loc[0, "edge_path"]) == [1, 2]
    assert od_flows.loc[0, "cost"] == 8.0
    assert network_flows.set_index("edge_id").loc[1, "flow"] == 7.0


def test_extension_disrupt():
    network = pd.DataFrame(
        {
            "edge_from": [0, 1, 0],
            "edge_to": [1, 2, 2],
            "edge_id": [0, 1, 2],
            "cost": [1.0, 1.0, 5.0],
            "capacity": [100.0, 100.0, 100.0],
        }
    )
    od_flows = pd.DataFrame(
        {
            "origin_id": [0],
            "destination_id": [2],
            "flow": [10.0],
            "edge_path": [[0, 1]],
            "cost": [2.0],
        }
    )

    result = core.disrupt(network, od_flows, [0], directed=True)
    rerouted_flows = result["rerouted_flows"].to_pandas()
    losses = result["losses"].to_pandas()

    assert list(rerouted_flows.loc[0, "edge_path"]) == [2]
    assert rerouted_flows.loc[0, "cost"] == 5.0
    assert losses.loc[0, "rerouting_loss"] == 3.0


def test_extension_allocate_string_ids():
    network = pd.DataFrame(
        {
            "edge_from": ["A", "A", "C"],
            "edge_to": ["B", "C", "B"],
            "edge_id": ["AB", "AC", "CB"],
            "cost": [10.0, 3.0, 5.0],
            "capacity": [100.0, 100.0, 100.0],
        }
    )
    od = pd.DataFrame(
        {
            "origin_id": ["A"],
            "destination_id": ["B"],
            "flow": [7.0],
        }
    )

    result = core.allocate(network, od, directed=True)
    od_flows = result["od_flows"].to_pandas()
    network_flows = result["network_flows"].to_pandas()

    assert od_flows.loc[0, "origin_id"] == "A"
    assert od_flows.loc[0, "destination_id"] == "B"
    assert list(od_flows.loc[0, "edge_path"]) == ["AC", "CB"]
    assert od_flows.loc[0, "cost"] == 8.0
    assert network_flows.set_index("edge_id").loc["AC", "flow"] == 7.0


def test_extension_disrupt_string_failed_edges():
    network = pd.DataFrame(
        {
            "edge_from": ["A", "B", "A"],
            "edge_to": ["B", "C", "C"],
            "edge_id": ["AB", "BC", "AC"],
            "cost": [1.0, 1.0, 5.0],
            "capacity": [100.0, 100.0, 100.0],
        }
    )
    od_flows = pd.DataFrame(
        {
            "origin_id": ["A"],
            "destination_id": ["C"],
            "flow": [10.0],
            "edge_path": [["AB", "BC"]],
            "cost": [2.0],
        }
    )

    result = core.disrupt(network, od_flows, ["AB"], directed=True)
    rerouted_flows = result["rerouted_flows"].to_pandas()
    losses = result["losses"].to_pandas()

    assert list(rerouted_flows.loc[0, "edge_path"]) == ["AC"]
    assert rerouted_flows.loc[0, "cost"] == 5.0
    assert losses.loc[0, "rerouting_loss"] == 3.0


def test_extension_allocate_unsigned_integer_ids():
    network = pa.table(
        {
            "edge_from": pa.array([10, 10, 12], type=pa.uint32()),
            "edge_to": pa.array([11, 12, 11], type=pa.uint32()),
            "edge_id": pa.array([100, 101, 102], type=pa.uint32()),
            "cost": pa.array([10.0, 3.0, 5.0]),
            "capacity": pa.array([100.0, 100.0, 100.0]),
        }
    )
    od = pa.table(
        {
            "origin_id": pa.array([10], type=pa.uint32()),
            "destination_id": pa.array([11], type=pa.uint32()),
            "flow": pa.array([7.0]),
        }
    )

    result = core.allocate(network, od, directed=True)
    od_flows = result["od_flows"].to_pandas()

    assert od_flows.loc[0, "origin_id"] == 10
    assert od_flows.loc[0, "destination_id"] == 11
    assert list(od_flows.loc[0, "edge_path"]) == [101, 102]


def test_extension_rejects_duplicate_edge_ids():
    network = pd.DataFrame(
        {
            "edge_from": ["A", "B"],
            "edge_to": ["B", "C"],
            "edge_id": ["AB", "AB"],
        }
    )
    od = pd.DataFrame({"origin_id": ["A"], "destination_id": ["C"], "flow": [1.0]})

    with pytest.raises(ValueError, match="duplicate edge_id"):
        core.allocate(network, od)


def test_extension_rejects_null_ids():
    network = pd.DataFrame(
        {
            "edge_from": ["A", None],
            "edge_to": ["B", "C"],
            "edge_id": ["AB", "BC"],
        }
    )
    od = pd.DataFrame({"origin_id": ["A"], "destination_id": ["C"], "flow": [1.0]})

    with pytest.raises(ValueError, match="cannot contain null id values"):
        core.allocate(network, od)


def test_extension_rejects_mismatched_id_types():
    network = pd.DataFrame(
        {
            "edge_from": ["A"],
            "edge_to": ["B"],
            "edge_id": ["AB"],
        }
    )
    od = pd.DataFrame({"origin_id": [1], "destination_id": [2], "flow": [1.0]})

    with pytest.raises(ValueError, match="origin_id id type"):
        core.allocate(network, od)


def test_extension_rejects_unknown_edge_path_ids():
    network = pd.DataFrame(
        {
            "edge_from": ["A", "B"],
            "edge_to": ["B", "C"],
            "edge_id": ["AB", "BC"],
        }
    )
    od_flows = pd.DataFrame(
        {
            "origin_id": ["A"],
            "destination_id": ["C"],
            "flow": [1.0],
            "edge_path": [["AB", "missing"]],
            "cost": [2.0],
        }
    )

    with pytest.raises(ValueError, match="unknown edge_id"):
        core.disrupt(network, od_flows, ["AB"])


def test_prepared_network_reports_shape():
    network = pd.DataFrame(
        {
            "edge_from": ["A", "B"],
            "edge_to": ["B", "C"],
            "edge_id": ["AB", "BC"],
        }
    )
    prepared = core.prepare(network)
    assert prepared.n_links == 2
    assert prepared.n_nodes == 3
    assert "n_links=2" in repr(prepared)


def test_prepared_network_matches_free_functions():
    network = pd.DataFrame(
        {
            "edge_from": ["A", "C", "B", "B"],
            "edge_to": ["C", "B", "D", "D"],
            "edge_id": ["XX", "YY", "ZZ", "AA"],
            "capacity": [100, 100, 50, 100],
            "cost": [20, 10, 5, 50],
        }
    )
    od = pd.DataFrame(
        {
            "origin_id": ["A", "A", "B"],
            "destination_id": ["B", "C", "D"],
            "flow": [30.0, 90.0, 100.0],
        }
    )
    prepared = core.prepare(network)
    for _ in range(2):  # reuse must not change the answer
        assert prepared.allocate(od, capacity_constrained=True) == core.allocate(
            network, od, capacity_constrained=True
        )


def test_prepared_network_does_not_leak_unknown_demand_ids():
    """Ids named only by demand must not persist into the next call.

    A prepared network interns identifiers once; demand may name nodes the
    network does not have, and those are interned so unassigned rows can be
    labelled with them. Without a rollback they would accumulate.
    """
    network = pd.DataFrame(
        {
            "edge_from": ["A"],
            "edge_to": ["B"],
            "edge_id": ["AB"],
        }
    )
    unknown = pd.DataFrame({"origin_id": ["A"], "destination_id": ["Z"], "flow": [1.0]})
    known = pd.DataFrame({"origin_id": ["A"], "destination_id": ["B"], "flow": [1.0]})

    prepared = core.prepare(network)
    assert prepared.n_nodes == 2

    unreachable = prepared.allocate(unknown)
    assert unreachable["unassigned_od"].num_rows == 1
    assert unreachable["unassigned_od"]["destination_id"].to_pylist() == ["Z"]
    assert prepared.n_nodes == 2, "unknown demand id leaked into the network"

    # A later call is unaffected by the earlier one.
    assert prepared.allocate(known) == core.allocate(network, known)
    assert prepared.allocate(unknown)["unassigned_od"][
        "destination_id"
    ].to_pylist() == ["Z"]


def test_prepared_network_rejects_the_same_bad_input_as_allocate():
    network = pd.DataFrame(
        {
            "edge_from": ["A", "B"],
            "edge_to": ["B", "C"],
            "edge_id": ["AB", "AB"],
        }
    )
    with pytest.raises(ValueError, match="duplicate edge_id"):
        core.prepare(network)

    prepared = core.prepare(
        pd.DataFrame({"edge_from": ["A"], "edge_to": ["B"], "edge_id": ["AB"]})
    )
    with pytest.raises(ValueError, match="origin_id id type"):
        prepared.allocate(
            pd.DataFrame({"origin_id": [1], "destination_id": [2], "flow": [1.0]})
        )
    # the failed call leaves the handle usable
    assert prepared.n_nodes == 2


def test_skim_matches_a_per_origin_shortest_path_loop():
    """The batched skim must agree with the loop it replaces, nulls included."""
    network = pd.DataFrame(
        {
            "edge_from": [0, 2, 1, 3],
            "edge_to": [2, 1, 3, 1],
            "edge_id": [0, 1, 2, 3],
            "cost": [20.0, 10.0, 5.0, 50.0],
        }
    )
    pairs = pd.DataFrame(
        {
            "origin_id": [0, 0, 0, 1, 3],
            "destination_id": [1, 2, 3, 0, 1],
        }
    )
    skims = core.skim(network, pairs)

    expected = []
    for origin, destination in zip(pairs.origin_id, pairs.destination_id):
        reached = core.shortest_paths_from(network, int(origin))
        costs = dict(zip(reached["node_id"].to_pylist(), reached["cost"].to_pylist()))
        expected.append(costs.get(int(destination)))

    assert skims["cost"].to_pylist() == expected
    assert skims["origin_id"].to_pylist() == list(pairs.origin_id)
    assert skims["destination_id"].to_pylist() == list(pairs.destination_id)
    assert None in expected, "fixture should include an unreachable pair"


def test_skim_builds_one_tree_per_origin():
    """Repeated origins are answered from the same tree, in input order."""
    network = pd.DataFrame(
        {
            "edge_from": ["A", "B"],
            "edge_to": ["B", "C"],
            "edge_id": ["AB", "BC"],
            "cost": [1.0, 2.0],
        }
    )
    pairs = pd.DataFrame(
        {
            "origin_id": ["A", "A", "A"],
            "destination_id": ["C", "B", "C"],
        }
    )
    assert core.skim(network, pairs)["cost"].to_pylist() == [3.0, 1.0, 3.0]


def test_skim_ignores_extra_columns():
    """A demand table can be passed straight in."""
    network = pd.DataFrame(
        {
            "edge_from": ["A"],
            "edge_to": ["B"],
            "edge_id": ["AB"],
            "cost": [7.0],
        }
    )
    demand = pd.DataFrame(
        {"origin_id": ["A"], "destination_id": ["B"], "value": [123.0]}
    )
    assert core.skim(network, demand)["cost"].to_pylist() == [7.0]


def test_skim_undirected_uses_links_both_ways():
    network = pd.DataFrame(
        {
            "edge_from": ["A"],
            "edge_to": ["B"],
            "edge_id": ["AB"],
            "cost": [7.0],
        }
    )
    pairs = pd.DataFrame({"origin_id": ["B"], "destination_id": ["A"]})
    assert core.skim(network, pairs, directed=True)["cost"].to_pylist() == [None]
    assert core.skim(network, pairs, directed=False)["cost"].to_pylist() == [7.0]


def test_prepared_disruption_matches_the_unprepared_call():
    network = pd.DataFrame(
        {
            "edge_from": ["A", "B", "A"],
            "edge_to": ["B", "C", "C"],
            "edge_id": ["AB", "BC", "AC"],
            "cost": [1.0, 1.0, 5.0],
            "capacity": [100.0, 100.0, 100.0],
        }
    )
    od_flows = pd.DataFrame(
        {
            "origin_id": ["A"],
            "destination_id": ["C"],
            "flow": [10.0],
            "edge_path": [["AB", "BC"]],
            "cost": [2.0],
        }
    )
    prepared = core.prepare_disruption(network, od_flows)
    for failed in (["AB"], [], ["AB", "BC"]):
        assert prepared.scenario(failed) == core.disrupt(network, od_flows, failed)


def test_prepared_disruption_is_reusable_across_scenarios():
    """Each scenario is independent of the ones before it."""
    network = pd.DataFrame(
        {
            "edge_from": ["A", "B", "A"],
            "edge_to": ["B", "C", "C"],
            "edge_id": ["AB", "BC", "AC"],
            "cost": [1.0, 1.0, 5.0],
            "capacity": [100.0, 100.0, 100.0],
        }
    )
    od_flows = pd.DataFrame(
        {
            "origin_id": ["A"],
            "destination_id": ["C"],
            "flow": [10.0],
            "edge_path": [["AB", "BC"]],
            "cost": [2.0],
        }
    )
    prepared = core.prepare_disruption(network, od_flows)

    # An unaffected scenario in between must not disturb the answers.
    first = prepared.scenario(["AB"])
    prepared.scenario(["AC"])
    assert prepared.scenario(["AB"]) == first

    # AB gone, so the flow reroutes onto the direct link at cost 5.
    assert first["rerouted_flows"]["cost"].to_pylist() == [5.0]
    assert first["rerouted_flows"]["edge_path"].to_pylist() == [["AC"]]


def test_prepared_disruption_rejects_unknown_edge_path_ids():
    network = pd.DataFrame(
        {
            "edge_from": ["A", "B"],
            "edge_to": ["B", "C"],
            "edge_id": ["AB", "BC"],
        }
    )
    od_flows = pd.DataFrame(
        {
            "origin_id": ["A"],
            "destination_id": ["C"],
            "flow": [1.0],
            "edge_path": [["AB", "missing"]],
            "cost": [2.0],
        }
    )
    with pytest.raises(ValueError, match="unknown edge_id"):
        core.prepare_disruption(network, od_flows)


def _fan_out_network():
    """One origin, three destinations, each reachable by a cheap and a dear link."""
    return pd.DataFrame(
        {
            "edge_from": ["O", "O", "O", "O", "O"],
            "edge_to": ["A", "A", "B", "C", "C"],
            "edge_id": ["OA", "OA2", "OB", "OC", "OC2"],
            "cost": [1.0, 5.0, 1.0, 1.0, 5.0],
            "capacity": [100.0] * 5,
        }
    )


def _fan_out_paths():
    return pd.DataFrame(
        {
            "origin_id": ["O", "O", "O"],
            "destination_id": ["A", "B", "C"],
            "flow": [10.0, 10.0, 10.0],
            "edge_path": [["OA"], ["OB"], ["OC"]],
            "cost": [1.0, 1.0, 1.0],
        }
    )


def test_affected_flows_keep_path_table_row_order():
    """Failed links are gathered in path-table order, not in failure order.

    The affected flows reach the rerouting step in this order and it shows
    up in the output, so gathering link-by-link would permute the result.
    """
    prepared = core.prepare_disruption(_fan_out_network(), _fan_out_paths())
    # OC is row 2 and OA is row 0: naming them in this order must not
    # reorder the output.
    rerouted = prepared.scenario(["OC", "OA"])["rerouted_flows"]
    assert rerouted["destination_id"].to_pylist() == ["A", "C"]
    assert rerouted["edge_path"].to_pylist() == [["OA2"], ["OC2"]]


def test_path_using_two_failed_links_is_affected_once():
    """A path reachable from two failed links must not be counted twice."""
    network = pd.DataFrame(
        {
            "edge_from": ["A", "B", "A"],
            "edge_to": ["B", "C", "C"],
            "edge_id": ["AB", "BC", "AC"],
            "cost": [1.0, 1.0, 5.0],
            "capacity": [100.0, 100.0, 100.0],
        }
    )
    od_flows = pd.DataFrame(
        {
            "origin_id": ["A"],
            "destination_id": ["C"],
            "flow": [10.0],
            "edge_path": [["AB", "BC"]],
            "cost": [2.0],
        }
    )
    prepared = core.prepare_disruption(network, od_flows)

    both = prepared.scenario(["AB", "BC"])
    assert both["rerouted_flows"].num_rows == 1
    assert both["rerouted_flows"]["flow"].to_pylist() == [10.0]
    assert both["losses"].num_rows == 1
    # Failing either link alone reroutes the same single path.
    assert both == prepared.scenario(["AB"])


def test_scenario_on_a_link_no_path_uses_is_a_no_op():
    prepared = core.prepare_disruption(_fan_out_network(), _fan_out_paths())
    unused = prepared.scenario(["OA2"])
    assert unused["rerouted_flows"].num_rows == 0
    assert unused["isolated_od"].num_rows == 0
    assert unused == prepared.scenario([])
