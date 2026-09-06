Estimate OD Demand
==================

Generate an origin-destination matrix from location counts — population,
employment — when you have no OD survey data. For how the model works and
why it needs no calibration, see :doc:`../explanation/radiation-model`.

Generate OD probabilities
-------------------------

You need a network with distances on its ``cost`` attribute, a table of
zones with a relevance measure, and a mapping from zones to network nodes.

>>> import pandas as pd
>>> from transport_flow_model import Network, RadiationModel
>>> network = Network.from_dataframe(
...     pd.DataFrame(
...         {
...             "edge_from": ["A", "B", "B"],
...             "edge_to": ["B", "C", "A"],
...             "edge_id": ["AB", "BC", "BA"],
...             "cost": [5.0, 3.0, 5.0],
...         }
...     )
... )
>>> zones = pd.DataFrame({"zone_id": [1, 2, 3], "population": [1000, 2000, 1500]})
>>> mapping = pd.DataFrame({"zone_id": [1, 2, 3], "node_id": ["A", "B", "C"]})
>>> model = RadiationModel(network=network)
>>> probabilities = model.generate(
...     zones=zones,
...     zone_id_column="zone_id",
...     zone_to_node_mapping=mapping,
...     relevance_column="population",
...     distance_threshold=10.0,
... )
>>> probabilities.columns.tolist()
['origin', 'destination', 'probability']
>>> probabilities.round(4).to_dict("records")
[{'origin': 1, 'destination': 2, 'probability': 0.2286}, {'origin': 1, 'destination': 3, 'probability': 0.1429}, {'origin': 2, 'destination': 1, 'probability': 0.2286}, {'origin': 2, 'destination': 3, 'probability': 0.4}, {'origin': 3, 'destination': 1, 'probability': 0.1429}, {'origin': 3, 'destination': 2, 'probability': 0.4}]

Each zone must map to exactly one node, and every ``node_id`` must exist in
the network — the model raises if one does not.

Set the distance threshold
--------------------------

``distance_threshold`` bounds the network distance within which destinations
are considered. Raising it admits more intervening opportunities, which
spreads probability towards nearer destinations; lowering it drops distant
pairs from the result entirely.

>>> near = model.generate(
...     zones=zones,
...     zone_id_column="zone_id",
...     zone_to_node_mapping=mapping,
...     relevance_column="population",
...     distance_threshold=6.0,
... )
>>> len(near), len(probabilities)
(4, 6)

At a threshold of 6 the A–C pairs (8 km apart via B) drop out, leaving four.

Use a different relevance measure
---------------------------------

Any positive numeric column works — employment for commuting, retail floor
space for shopping trips. Pass its name as ``relevance_column``.

>>> zones_jobs = pd.DataFrame(
...     {"zone_id": [1, 2, 3], "population": [1000, 2000, 1500], "jobs": [50, 900, 120]}
... )
>>> by_jobs = model.generate(
...     zones=zones_jobs,
...     zone_id_column="zone_id",
...     zone_to_node_mapping=mapping,
...     relevance_column="jobs",
...     distance_threshold=10.0,
... )
>>> by_jobs.round(4).to_dict("records")[:2]
[{'origin': 1, 'destination': 2, 'probability': 0.2595}, {'origin': 1, 'destination': 3, 'probability': 0.0062}]

Zone 3 barely attracts anyone now: it has plenty of residents but few jobs,
and zone 2 lies between it and zone 1 absorbing the demand.

Convert probabilities to a Demand
---------------------------------

The model returns probabilities keyed by **zone id**. Assignment needs
**node ids** and a ``value`` column, so map them across before building a
:class:`~transport_flow_model.Demand` — this step is easy to miss, and
skipping it fails with a dtype mismatch rather than a helpful message.

>>> from transport_flow_model import Demand
>>> zone_to_node = dict(zip(mapping["zone_id"], mapping["node_id"]))
>>> od = pd.DataFrame(
...     {
...         "origin_id": probabilities["origin"].map(zone_to_node),
...         "destination_id": probabilities["destination"].map(zone_to_node),
...         "value": probabilities["probability"] * 1000,
...     }
... )
>>> demand = Demand.from_dataframe(od)
>>> demand.n_pairs, round(demand.total, 1)
(6, 1542.9)

Scale by whatever total the probabilities should carry — here 1000 trips per
origin. For a travel-to-work matrix, multiply each origin's probabilities by
its resident workforce instead of a flat constant.

Assign the estimated demand
---------------------------

From here it is an ordinary assignment (see :doc:`assign-demand`):

>>> from transport_flow_model import assign
>>> result = assign(network, demand, "sequential")
>>> result.link_flows.to_pandas()[["edge_id", "flow"]].round(1).to_dict("records")
[{'edge_id': 'AB', 'flow': 371.4}, {'edge_id': 'BC', 'flow': 542.9}, {'edge_id': 'BA', 'flow': 228.6}]

Two pairs are unassigned here, because this toy network has no ``C -> A`` or
``C -> B`` route — the radiation model proposes demand between any two
zones, and it is the network that decides whether the trip is possible.

>>> result.unassigned.num_rows
2

Always check ``unassigned`` on estimated demand: a matrix generated from
zone counts has no guarantee that every pair it proposes is connected.
