Disruptions
===========

``Network.disrupt`` reroutes OD flows whose existing ``edge_path`` includes a
failed edge. The result separates rerouted flows, isolated demand, updated edge
totals, and rerouting losses.

Isolated demand
---------------

If a failed edge removes the only available path, the affected demand appears in
``isolated_od`` and no rerouted flow is returned.

>>> import pandas as pd
>>> from transport_flow_model.model import Network, ODFlows
>>> network = Network(
...     pd.DataFrame(
...         {
...             "edge_from": ["A", "B"],
...             "edge_to": ["B", "C"],
...             "edge_id": ["AB", "BC"],
...             "cost": [1, 1],
...             "capacity": [100, 100],
...         }
...     )
... )
>>> existing_flows = ODFlows(
...     pd.DataFrame(
...         {
...             "origin_id": ["A"],
...             "destination_id": ["C"],
...             "flow": [10],
...             "edge_path": [["AB", "BC"]],
...             "cost": [2],
...         }
...     )
... )
>>> isolated = network.disrupt(existing_flows, ["AB"], directed=True)
>>> isolated.rerouted_flows.to_dataframe().empty
True
>>> isolated.isolated_od.to_dataframe().to_dict("records")
[{'origin_id': 'A', 'destination_id': 'C', 'flow': 10}]

Rerouting to an alternative path
--------------------------------

If an alternative path exists, the affected flow is rerouted. Comparing the
original path cost with the disrupted path cost gives rerouting losses as
``result.losses``.

>>> reroute_network = Network(
...     pd.DataFrame(
...         {
...             "edge_from": ["A", "B", "A"],
...             "edge_to": ["B", "C", "C"],
...             "edge_id": ["AB", "BC", "AC"],
...             "cost": [1, 1, 5],
...             "capacity": [100, 100, 100],
...         }
...     )
... )
>>> result = reroute_network.disrupt(existing_flows, ["AB"], directed=True)
>>> result.rerouted_flows.to_dataframe().to_dict("records")
[{'origin_id': 'A', 'destination_id': 'C', 'flow': 10, 'edge_path': ['AC'], 'cost': 5}]
>>> result.losses.to_dataframe().to_dict("records")
[{'origin_id': 'A', 'destination_id': 'C', 'flow': 10, 'initial_cost': 2, 'disrupted_cost': 5, 'rerouting_loss': 3}]
>>> result.network_flows.to_dataframe().set_index("edge_id")["flow"].to_dict()
{'AB': 0, 'BC': 0, 'AC': 10}

Unaffected flows stay on the network
------------------------------------

Only OD flows whose path includes a failed edge are rerouted. Other OD flows
remain in the returned ``network_flows`` totals.

>>> partial_network = Network(
...     pd.DataFrame(
...         {
...             "edge_from": ["A", "B", "A", "C"],
...             "edge_to": ["B", "C", "C", "D"],
...             "edge_id": ["AB", "BC", "AC", "CD"],
...             "cost": [1, 1, 5, 1],
...             "capacity": [100, 100, 100, 100],
...         }
...     )
... )
>>> partial_flows = ODFlows(
...     pd.DataFrame(
...         {
...             "origin_id": ["A", "C"],
...             "destination_id": ["C", "D"],
...             "flow": [10, 4],
...             "edge_path": [["AB", "BC"], ["CD"]],
...             "cost": [2, 1],
...         }
...     )
... )
>>> partial = partial_network.disrupt(partial_flows, ["AB"], directed=True)
>>> partial.network_flows.to_dataframe().set_index("edge_id")["flow"].to_dict()
{'AB': 0, 'BC': 0, 'AC': 10, 'CD': 4}

Capacity-constrained disruption
-------------------------------

Disruption rerouting is capacity constrained by default. If the alternate path
has only enough capacity for part of the affected flow, any residual demand is
isolated.

>>> capacity_network = Network(
...     pd.DataFrame(
...         {
...             "edge_from": ["A", "B", "A", "D"],
...             "edge_to": ["B", "C", "D", "C"],
...             "edge_id": ["AB", "BC", "AD", "DC"],
...             "cost": [1, 1, 3, 3],
...             "capacity": [15, 15, 10, 10],
...         }
...     )
... )
>>> capacity_flows = ODFlows(
...     pd.DataFrame(
...         {
...             "origin_id": ["A"],
...             "destination_id": ["C"],
...             "flow": [15],
...             "edge_path": [["AB", "BC"]],
...             "cost": [2],
...         }
...     )
... )
>>> capacity_result = capacity_network.disrupt(
...     capacity_flows, ["AB"], directed=True
... )
>>> capacity_result.rerouted_flows.to_dataframe().to_dict("records")
[{'origin_id': 'A', 'destination_id': 'C', 'flow': 10, 'edge_path': ['AD', 'DC'], 'cost': 6}]
>>> capacity_result.isolated_od.to_dataframe().to_dict("records")
[{'origin_id': 'A', 'destination_id': 'C', 'flow': 5}]
>>> capacity_result.network_flows.to_dataframe().set_index("edge_id")["flow"].to_dict()
{'AB': 0, 'BC': 0, 'AD': 10, 'DC': 10}

Scenarios with link cost changes
--------------------------------

The scenario API (:class:`~transport_flow_model.Scenario` and
:func:`~transport_flow_model.disrupt`) also models links that stay open but
cost more (or less), for example buses on a diversion. ``Scenario.from_changes``
combines removed links (``remove``), new absolute costs (``cost``) and factors
on the baseline cost (``cost_scale``).

- Flows whose baseline path uses a removed link are rerouted using the
  scenario's changed costs.
- Flows whose baseline path uses a link whose cost changed, but no removed
  link, keep their path by default. They are listed in
  ``ScenarioResult.repriced`` with their new path cost (the baseline path cost
  plus the cost changes along it) and are included in ``losses``. Pass
  ``reroute_changed=True`` to reroute them as well.
- ``initial_cost`` in ``losses`` is always the baseline path cost.

Here ``A -> D`` uses ``AB, BD`` and ``C -> D`` uses ``CD``. Removing ``BD``
would send ``A -> D`` along ``AC, CD`` (cost 4), but the scenario also makes
``CD`` cost 10, so the cheapest detour is the direct link ``AD`` (cost 5).
``C -> D`` stays on ``CD`` at the new cost.

>>> import transport_flow_model as tfm
>>> links = pd.DataFrame(
...     {
...         "edge_from": ["A", "B", "A", "C", "A"],
...         "edge_to": ["B", "D", "C", "D", "D"],
...         "edge_id": ["AB", "BD", "AC", "CD", "AD"],
...         "cost": [1, 1, 2, 2, 5],
...         "capacity": [100, 100, 100, 100, 100],
...     }
... )
>>> demand = tfm.Demand(
...     pd.DataFrame(
...         {"origin_id": ["A", "C"], "destination_id": ["D", "D"], "value": [10, 3]}
...     )
... )
>>> scenario = tfm.Scenario.from_changes("diversion", remove=["BD"], cost={"CD": 10})
>>> results = tfm.disrupt(
...     tfm.Network(links), [scenario], demand, capacity_constrained=True
... )
>>> (result,) = results.results
>>> result.rerouted.to_pylist()
[{'origin_id': 'A', 'destination_id': 'D', 'flow': 10, 'edge_path': ['AD'], 'cost': 5}]
>>> result.repriced.to_pylist()
[{'origin_id': 'C', 'destination_id': 'D', 'flow': 3, 'edge_path': ['CD'], 'cost': 10}]
>>> result.losses.select(["origin_id", "initial_cost", "disrupted_cost", "rerouting_loss"]).to_pylist()
[{'origin_id': 'A', 'initial_cost': 2, 'disrupted_cost': 5, 'rerouting_loss': 3}, {'origin_id': 'C', 'initial_cost': 2, 'disrupted_cost': 10, 'rerouting_loss': 8}]
>>> results.summary().to_pylist()
[{'scenario_id': 'diversion', 'rerouted_flow': 10.0, 'isolated_flow': 0.0, 'rerouting_loss': 11.0, 'repriced_flow': 3.0}]

With ``reroute_changed=True``, flows on links whose cost changed are rerouted
too; the changed links stay available. Making ``BD`` cost 10 then moves
``A -> D`` onto ``AC, CD``:

>>> dearer = tfm.Scenario.from_changes("dearer", cost={"BD": 10})
>>> (moved,) = tfm.disrupt(
...     tfm.Network(links), [dearer], demand, reroute_changed=True,
...     capacity_constrained=True,
... ).results
>>> moved.rerouted.select(["origin_id", "edge_path", "cost"]).to_pylist()
[{'origin_id': 'A', 'edge_path': ['AC', 'CD'], 'cost': 4}]
>>> moved.repriced.num_rows
0

With ``skip_unaffected=True`` (the default), a scenario is skipped when none of
its removed or changed links carries baseline flow. Deltas on attributes other
than ``cost`` (such as ``capacity``) raise ``NotImplementedError``.
