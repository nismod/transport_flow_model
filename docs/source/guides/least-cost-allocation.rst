Least-Cost Allocation
=====================

``Network.allocate`` assigns OD demand to least-cost paths over the
network. The result contains allocated OD paths, aggregate edge flows, and any
unassigned OD demand.

Single OD pair
--------------

This network has a direct ``A -> B`` edge with cost ``10`` and an indirect
``A -> C -> B`` path with total cost ``8``. Allocation chooses the lower-cost
indirect path.

>>> import pandas as pd
>>> from transport_flow_model.model import Network, OD
>>> network = Network(
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
>>> od = OD(
...     pd.DataFrame(
...         {
...             "origin_id": ["A"],
...             "destination_id": ["B"],
...             "flow": [7],
...         }
...     )
... )
>>> result = network.allocate(od, directed=True)

``result.od_flows`` records the path, path cost, and assigned flow.

>>> result.od_flows.to_dataframe().to_dict("records")
[{'origin_id': 'A', 'destination_id': 'B', 'flow': 7, 'edge_path': ['AC', 'CB'], 'cost': 8}]

``result.network_flows`` aggregates those path flows onto every edge. The direct
edge is retained with zero flow because it was unused.

>>> result.network_flows.to_dataframe().set_index("edge_id")["flow"].to_dict()
{'AB': 0, 'AC': 7, 'CB': 7}

All demand was assigned, so ``unassigned_od`` is empty.

>>> result.unassigned_od.to_dataframe().empty
True
