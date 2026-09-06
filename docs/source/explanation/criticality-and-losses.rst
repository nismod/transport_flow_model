Criticality and Rerouting Losses
================================

A rerouting loss is what a disruption costs the flows that survive it: for
each OD pair, its disrupted path cost minus its baseline path cost, over the
flow that took it. Demand with no remaining path at all is not a loss — it
is *isolated*, counted separately, because no finite cost describes it.

Per-OD losses
-------------

Two OD pairs share the ``A -> B -> C`` corridor; one continues to ``D``.

>>> import pandas as pd
>>> from transport_flow_model import Network, Demand, Scenario, assign, disrupt
>>> network = Network.from_dataframe(
...     pd.DataFrame(
...         {
...             "edge_from": ["A", "B", "A", "C", "A"],
...             "edge_to": ["B", "C", "C", "D", "D"],
...             "edge_id": ["AB", "BC", "AC", "CD", "AD"],
...             "cost": [1, 1, 5, 1, 9],
...             "capacity": [100, 100, 100, 100, 100],
...         }
...     )
... )
>>> demand = Demand.from_dataframe(
...     pd.DataFrame(
...         {
...             "origin_id": ["A", "A"],
...             "destination_id": ["C", "D"],
...             "value": [10, 4],
...         }
...     )
... )
>>> base = assign(network, demand, "sequential", include_paths=True)
>>> results = disrupt(
...     network,
...     [
...         Scenario.remove_links("AB_fails", ["AB"]),
...         Scenario.remove_links("CD_fails", ["CD"]),
...     ],
...     base=base,
... )

Losing ``AB`` pushes both pairs onto costlier routes. ``losses`` reports one
row per affected OD pair, with the costs it is the difference of.

>>> results.results[0].losses.to_pandas().to_dict("records")
[{'origin_id': 'A', 'destination_id': 'C', 'flow': 10, 'initial_cost': 2, 'disrupted_cost': 5, 'rerouting_loss': 3}, {'origin_id': 'A', 'destination_id': 'D', 'flow': 4, 'initial_cost': 3, 'disrupted_cost': 6, 'rerouting_loss': 3}]

Losing ``CD`` affects only the pair that used it, but hurts it more: the
detour to ``D`` costs ``9`` against a baseline of ``3``.

>>> results.results[1].losses.to_pandas().to_dict("records")
[{'origin_id': 'A', 'destination_id': 'D', 'flow': 4, 'initial_cost': 3, 'disrupted_cost': 9, 'rerouting_loss': 6}]

Ranking scenarios
-----------------

``summary()`` aggregates every scenario into one table — the usual starting
point for a criticality ranking.

>>> results.summary().to_pandas().to_dict("records")
[{'scenario_id': 'AB_fails', 'rerouted_flow': 14.0, 'isolated_flow': 0.0, 'rerouting_loss': 6.0}, {'scenario_id': 'CD_fails', 'rerouted_flow': 4.0, 'isolated_flow': 0.0, 'rerouting_loss': 6.0}]

Note that these two scenarios tie on ``rerouting_loss`` while disturbing
very different amounts of traffic — 14 units against 4. Which link is
"more critical" is a question about the ranking criterion, not one the
model answers for you: the three columns are reported separately so you can
choose. ``isolated_flow`` in particular does not belong in a cost total, since
demand that can no longer reach its destination has no rerouting cost at all.

Losses compare against the baseline you passed in, so that baseline decides
what the numbers mean. A single all-or-nothing pass, as used here, prices
detours at free-flow cost and takes no account of the congestion the
rerouted traffic itself creates. For a baseline where no traveller can
improve their own route, see :doc:`equilibrium`.
