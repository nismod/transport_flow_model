Assign Demand to Least-Cost Paths
=================================

:func:`~transport_flow_model.assign` with ``method="sequential"`` puts each
OD pair's demand on its least-cost path, one pair at a time. The result
holds the aggregate link flows, a cost per OD pair, any demand that could
not be assigned, and — when asked for — the paths themselves.

This is a single all-or-nothing pass: it takes no account of the congestion
the assigned flow itself causes. For a solution where no traveller can
improve their own cost by switching route, see :doc:`../explanation/equilibrium`.

Single OD pair
--------------

This network has a direct ``A -> B`` edge with cost ``10`` and an indirect
``A -> C -> B`` path with total cost ``8``. Allocation chooses the lower-cost
indirect path.

>>> import pandas as pd
>>> from transport_flow_model import Network, Demand, assign
>>> network = Network.from_dataframe(
...     pd.DataFrame(
...         {
...             "edge_from": ["A", "A", "C"],
...             "edge_to": ["B", "C", "B"],
...             "edge_id": ["AB", "AC", "CB"],
...             "cost": [10, 3, 5],
...             "capacity": [100, 100, 100],
...         }
...     )
... )
>>> demand = Demand.from_dataframe(
...     pd.DataFrame(
...         {
...             "origin_id": ["A"],
...             "destination_id": ["B"],
...             "value": [7],
...         }
...     )
... )
>>> result = assign(network, demand, "sequential", include_paths=True)

``result.paths`` records the path, its cost, and the flow assigned to it.

>>> [
...     {**row, "edge_path": list(row["edge_path"])}
...     for row in result.paths.to_pandas().to_dict("records")
... ]
[{'origin_id': 'A', 'destination_id': 'B', 'flow': 7, 'edge_path': ['AC', 'CB'], 'cost': 8}]

``result.link_flows`` aggregates those path flows onto every link. The direct
edge is retained with zero flow because it was unused — the table always
covers the whole network, in network link order.

>>> result.link_flows.to_pandas().set_index("edge_id")["flow"].to_dict()
{'AB': 0, 'AC': 7, 'CB': 7}

``result.skims`` gives the cost of reaching each destination, and all demand
was assigned, so ``unassigned`` is empty.

>>> result.skims.to_pandas().to_dict("records")
[{'origin_id': 'A', 'destination_id': 'B', 'cost': 8.0}]
>>> result.unassigned.num_rows
0
