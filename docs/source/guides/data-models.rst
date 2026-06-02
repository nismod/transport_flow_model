Data Models
===========

``tfm`` defines a few key objects:

- ``Network`` stores nodes and edges
- ``OD`` stores origin-destination flows
- ``ODFlows`` stores flows allocated to paths over a network (for each ``OD``
  pair, the flow might use one or more paths to route over the network)
- ``NetworkFlows`` stores flows allocated to the network in aggregate (for each
  node or edge in a ``Network``, multiple ``OD`` pairs might be allocated to it)

Network edges
-------------

Create a ``Network`` from a DataFrame with ``edge_from``, ``edge_to``, and
``edge_id`` columns. Optional columns such as ``cost`` and ``capacity`` are used
later to allocate flows.

>>> import pandas as pd
>>> from transport_flow_model.model import Network, NetworkFlows, OD, ODFlows
>>> network = Network(
...     pd.DataFrame(
...         {
...             "edge_from": ["A", "B", "C"],
...             "edge_to": ["B", "C", "D"],
...             "edge_id": ["AB", "BC", "CD"],
...             "cost": [1, 1, 1],
...         }
...     )
... )
>>> network.to_dataframe()[["edge_from", "edge_to", "edge_id"]].to_dict("records")
[{'edge_from': 'A', 'edge_to': 'B', 'edge_id': 'AB'}, {'edge_from': 'B', 'edge_to': 'C', 'edge_id': 'BC'}, {'edge_from': 'C', 'edge_to': 'D', 'edge_id': 'CD'}]

OD demand
---------

An ``OD`` object represents demand for flows between origin and destination nodes.

>>> od = OD(
...     pd.DataFrame(
...         {
...             "origin_id": ["A"],
...             "destination_id": ["C"],
...             "flow": [10],
...         }
...     )
... )
>>> od.to_dataframe().to_dict("records")
[{'origin_id': 'A', 'destination_id': 'C', 'flow': 10}]


Allocated flows
---------------

An ``ODFlows`` object stores the paths used for each allocated OD flow.
The ``edge_path`` values are lists of edge IDs.

>>> od_flows = ODFlows(
...     pd.DataFrame(
...         {
...             "origin_id": ["A", "B"],
...             "destination_id": ["B", "C"],
...             "flow": [10, 5],
...             "edge_path": [["AB"], ["BA", "AC"]],
...         }
...     )
... )
>>> od_flows.to_dataframe()["edge_path"].tolist()
[['AB'], ['BA', 'AC']]

Edge totals
-----------

``NetworkFlows.from_network_and_od_flows`` aggregates path flows onto
network edges. Unused edges are have zero flow.

>>> network_flows = NetworkFlows.from_network_and_od_flows(network, od_flows)
>>> network_flows.to_dataframe().set_index("edge_id")["flow"].to_dict()
{'AB': 10, 'BC': 0, 'CD': 0}

CSV input
---------

CSV inputs can use project-specific column names. ``from_csv`` normalizes them
to the model schema using a ``column_map`` dictionary.

>>> import tempfile
>>> with tempfile.TemporaryDirectory() as tmpdir:
...     csv_path = f"{tmpdir}/network.csv"
...     pd.DataFrame(
...         {
...             "from_id": ["A"],
...             "to_id": ["B"],
...             "id": ["E1"],
...             "flow_capacity": [100],
...             "gcost_usd_per_ton": [10.5],
...         }
...     ).to_csv(csv_path, index=False)
...     loaded = Network.from_csv(
...         csv_path,
...         {
...             "from_id": "edge_from",
...             "to_id": "edge_to",
...             "id": "edge_id",
...             "flow_capacity": "capacity",
...             "gcost_usd_per_ton": "cost",
...         },
...     )
...     loaded.to_dataframe().to_dict("records")
[{'edge_from': 'A', 'edge_to': 'B', 'edge_id': 'E1', 'capacity': 100, 'cost': 10.5}]
