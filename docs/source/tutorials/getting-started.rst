Getting Started
===============

In this tutorial you will assign traffic to a real benchmark network, find
out how far that first answer is from equilibrium, converge it, and then
break the busiest road to see what the disruption costs.

Everything here runs on data that ships with the package — nothing is
downloaded, and every result is reproducible.

Step 1: load a network
----------------------

SiouxFalls is the standard small benchmark for traffic assignment: 24 nodes,
76 links, and a published equilibrium solution to check yourself against.

>>> from transport_flow_model import Network, Demand, datasets
>>> instance = datasets.load_tntp("siouxfalls")
>>> network = Network.from_tntp(instance)
>>> demand = Demand.from_tntp(instance)
>>> network.n_nodes, network.n_links
(24, 76)

The demand is 528 origin-destination pairs — almost every node to every
other — carrying 360,600 trips in total.

>>> demand.n_pairs, demand.total
(528, 360600.0)

Step 2: assign the demand
-------------------------

:func:`~transport_flow_model.assign` routes each OD pair over the network.
Start with ``"sequential"``, which puts each pair on its cheapest path in
turn. Ask for ``include_paths`` — you will need them in step 5.

>>> from transport_flow_model import assign
>>> loaded = assign(network, demand, "sequential", include_paths=True)

``link_flows`` is the network's link table with a ``flow`` column added.
Sort it to find where the traffic went:

>>> flows = loaded.link_flows.to_pandas()
>>> flows.nlargest(3, "flow")[["edge_from", "edge_to", "flow"]].to_dict("records")
[{'edge_from': 10, 'edge_to': 16, 'flow': 28200}, {'edge_from': 16, 'edge_to': 10, 'flow': 28100}, {'edge_from': 16, 'edge_to': 17, 'flow': 26700}]

The corridor between nodes 10 and 16 is the busiest in the network, in both
directions.

Step 3: find out how wrong that is
----------------------------------

That assignment routed everyone by *free-flow* cost — as if the roads were
empty. They are not: 28,200 vehicles were just put on one link. The
*relative gap* measures how far the result is from a state where nobody
could do better by switching route.

>>> from transport_flow_model import relative_gap
>>> f"{relative_gap(network, demand, loaded):.2f}"
'8.82'

Zero would mean equilibrium. 8.82 means total travel time is nearly ten
times what it would be if everyone took the route that is actually cheapest
at these flows. The first answer is a long way off.

Step 4: converge to equilibrium
-------------------------------

``"msa"`` fixes this by repeating the assignment at congested costs and
averaging the results, until the gap is small enough. Ask for a gap of
``1e-3`` and give it room to get there:

>>> equilibrium = assign(
...     network, demand, "msa", target_gap=1e-3, max_iterations=2000
... )
>>> equilibrium.provenance.iterations
743
>>> f"{equilibrium.provenance.relative_gap:.3e}"
'9.963e-04'

743 passes to move four orders of magnitude. Now compare the busiest links
with step 2:

>>> settled = equilibrium.link_flows.to_pandas()
>>> settled.nlargest(3, "flow")[["edge_from", "edge_to", "flow"]].round(0).to_dict("records")
[{'edge_from': 15, 'edge_to': 10, 'flow': 23210.0}, {'edge_from': 10, 'edge_to': 15, 'flow': 23139.0}, {'edge_from': 10, 'edge_to': 9, 'flow': 21833.0}]

A different corridor is busiest. The 10–16 link that dominated step 2 has
lost more than half its traffic:

>>> f"{settled.query('edge_from == 10 and edge_to == 16')['flow'].iloc[0]:.0f}"
'11047'

28,200 down to 11,047. Once congestion is priced in, traffic spreads onto
alternatives the free-flow assignment ignored — and the answer to "which
road carries the most" changes. That is the reason equilibrium matters:
step 2 did not just get the magnitudes wrong, it identified the wrong link.

You can check the result independently. User equilibrium minimizes the
Beckmann objective, so a converged run should land just above the published
best-known value:

>>> from transport_flow_model import beckmann_objective
>>> published = datasets.BEST_KNOWN["siouxfalls"].objective
>>> f"{100 * (beckmann_objective(network, equilibrium) / published - 1):.3f}%"
'0.167%'

Step 5: break the busiest road
------------------------------

A :class:`~transport_flow_model.Scenario` removes links;
:func:`~transport_flow_model.disrupt` reroutes the flows that used them.
Take the busiest link from step 2 and remove it.

Use the *sequential* result as the baseline, not the equilibrium one:
rerouting needs to know which paths used the failed link, and an
equilibrium — being an average of many assignments — has no single path per
OD pair.

>>> from transport_flow_model import Scenario, disrupt
>>> busiest = int(flows.nlargest(1, "flow").iloc[0]["edge_id"])
>>> results = disrupt(
...     network, [Scenario.remove_links("lose_10_16", [busiest])], base=loaded
... )
>>> summary = results.summary().to_pandas().round(0).to_dict("records")
>>> summary
[{'scenario_id': 'lose_10_16', 'rerouted_flow': 4916.0, 'isolated_flow': 23284.0, 'rerouting_loss': 21.0}]

Of the 28,200 trips that used that link, about 4,900 found another route —
and 23,284 could not. They are *isolated*: with rerouting capacity-limited
by default, the alternatives were already full. Isolated demand is reported
separately from rerouting loss precisely because no travel-time figure
describes it; a trip that cannot be made has no detour cost.

Where to go next
----------------

You have run the whole pipeline. From here:

- :doc:`../how-to/converge-an-assignment` — budgets, time limits, and
  choosing a volume-delay curve
- :doc:`../how-to/evaluate-disruptions` — scenario sets, isolated demand and
  capacity-constrained rerouting
- :doc:`../explanation/equilibrium` — what the gap means and why MSA is the
  baseline rather than the recommendation
- :doc:`../reference/api` — the full API
