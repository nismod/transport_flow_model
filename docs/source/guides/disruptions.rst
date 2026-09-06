Disruptions
===========

:func:`~transport_flow_model.disrupt` takes a baseline assignment and a set
of :class:`~transport_flow_model.Scenario` objects, and for each scenario
reroutes the baseline flows whose path used a removed link. Flows that did
not use one are left alone — that is what makes a scenario sweep affordable.

Each :class:`~transport_flow_model.ScenarioResult` separates rerouted flow,
isolated demand, updated link totals, and the rerouting loss per OD pair.
The baseline must carry paths, so assign it with ``include_paths=True``.

Isolated demand
---------------

If a failed link removes the only available path, the affected demand
appears in ``isolated`` and no rerouted flow is returned.

>>> import pandas as pd
>>> from transport_flow_model import Network, Demand, Scenario, assign, disrupt
>>> network = Network.from_dataframe(
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
>>> demand = Demand.from_dataframe(
...     pd.DataFrame(
...         {
...             "origin_id": ["A"],
...             "destination_id": ["C"],
...             "value": [10],
...         }
...     )
... )
>>> base = assign(network, demand, "sequential", include_paths=True)
>>> results = disrupt(network, [Scenario.remove_links("AB_fails", ["AB"])], base=base)
>>> isolated = results.results[0]
>>> isolated.rerouted.num_rows
0
>>> isolated.isolated.to_pandas().to_dict("records")
[{'origin_id': 'A', 'destination_id': 'C', 'value': 10}]

Rerouting to an alternative path
--------------------------------

If an alternative path exists, the affected flow is rerouted. Comparing the
original path cost with the disrupted one gives the rerouting loss.

>>> reroute_network = Network.from_dataframe(
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
>>> base = assign(reroute_network, demand, "sequential", include_paths=True)
>>> results = disrupt(
...     reroute_network, [Scenario.remove_links("AB_fails", ["AB"])], base=base
... )
>>> result = results.results[0]
>>> [
...     {**row, "edge_path": list(row["edge_path"])}
...     for row in result.rerouted.to_pandas().to_dict("records")
... ]
[{'origin_id': 'A', 'destination_id': 'C', 'flow': 10, 'edge_path': ['AC'], 'cost': 5}]
>>> result.losses.to_pandas().to_dict("records")
[{'origin_id': 'A', 'destination_id': 'C', 'flow': 10, 'initial_cost': 2, 'disrupted_cost': 5, 'rerouting_loss': 3}]
>>> result.link_flows.to_pandas().set_index("edge_id")["flow"].to_dict()
{'AB': 0, 'BC': 0, 'AC': 10}

Unaffected flows stay on the network
------------------------------------

Only flows whose path includes a failed link are rerouted. Others remain in
the returned link totals.

>>> partial_network = Network.from_dataframe(
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
>>> partial_demand = Demand.from_dataframe(
...     pd.DataFrame(
...         {
...             "origin_id": ["A", "C"],
...             "destination_id": ["C", "D"],
...             "value": [10, 4],
...         }
...     )
... )
>>> base = assign(partial_network, partial_demand, "sequential", include_paths=True)
>>> partial = disrupt(
...     partial_network, [Scenario.remove_links("AB_fails", ["AB"])], base=base
... ).results[0]
>>> partial.link_flows.to_pandas().set_index("edge_id")["flow"].to_dict()
{'AB': 0, 'BC': 0, 'AC': 10, 'CD': 4}

A scenario whose links carried no baseline flow at all cannot change
anything, so it is skipped rather than evaluated, and named in ``skipped``.

>>> results = disrupt(
...     partial_network, [Scenario.remove_links("CD_fails", ["AC"])], base=base
... )
>>> results.results, [scenario.id for scenario in results.skipped]
((), ['CD_fails'])

Capacity-constrained rerouting
------------------------------

Rerouting is capacity constrained by default. If the alternative path has
room for only part of the affected flow, the residual demand is isolated.

>>> capacity_network = Network.from_dataframe(
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
>>> capacity_demand = Demand.from_dataframe(
...     pd.DataFrame(
...         {
...             "origin_id": ["A"],
...             "destination_id": ["C"],
...             "value": [15],
...         }
...     )
... )
>>> base = assign(capacity_network, capacity_demand, "sequential", include_paths=True)
>>> capacity_result = disrupt(
...     capacity_network, [Scenario.remove_links("AB_fails", ["AB"])], base=base
... ).results[0]
>>> [
...     {**row, "edge_path": list(row["edge_path"])}
...     for row in capacity_result.rerouted.to_pandas().to_dict("records")
... ]
[{'origin_id': 'A', 'destination_id': 'C', 'flow': 10, 'edge_path': ['AD', 'DC'], 'cost': 6}]
>>> capacity_result.isolated.to_pandas().to_dict("records")
[{'origin_id': 'A', 'destination_id': 'C', 'value': 5}]
>>> capacity_result.link_flows.to_pandas().set_index("edge_id")["flow"].to_dict()
{'AB': 0, 'BC': 0, 'AD': 10, 'DC': 10}

Comparing scenarios is the subject of :doc:`losses`.
