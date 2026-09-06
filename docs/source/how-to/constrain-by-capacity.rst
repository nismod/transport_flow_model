Constrain Assignment by Capacity
================================

When several OD pairs are assigned, ``link_flows`` aggregates their flows on
shared links.

Capacity-constrained assignment
(``assign(..., capacity_constrained=True)``) limits each assignment to the
link capacity still available and reports residual demand in
``unassigned``.

Shared links
------------

The two OD pairs below both use edge ``BC`` on their least-cost path.

>>> import pandas as pd
>>> from transport_flow_model import Network, Demand, assign
>>> network = Network.from_dataframe(
...     pd.DataFrame(
...         {
...             "edge_from": ["A", "B", "A", "C", "B"],
...             "edge_to": ["B", "C", "C", "D", "D"],
...             "edge_id": ["AB", "BC", "AC", "CD", "BD"],
...             "cost": [1, 2, 5, 1, 10],
...             "capacity": [100, 100, 100, 100, 100],
...         }
...     )
... )
>>> demand = Demand.from_dataframe(
...     pd.DataFrame(
...         {
...             "origin_id": ["A", "B"],
...             "destination_id": ["C", "D"],
...             "value": [10, 6],
...         }
...     )
... )
>>> result = assign(network, demand, "sequential", include_paths=True)
>>> [
...     {**row, "edge_path": list(row["edge_path"])}
...     for row in result.paths.to_pandas().to_dict("records")
... ]
[{'origin_id': 'A', 'destination_id': 'C', 'flow': 10, 'edge_path': ['AB', 'BC'], 'cost': 3}, {'origin_id': 'B', 'destination_id': 'D', 'flow': 6, 'edge_path': ['BC', 'CD'], 'cost': 3}]
>>> result.link_flows.to_pandas().set_index("edge_id")["flow"].to_dict()
{'AB': 10, 'BC': 16, 'AC': 0, 'CD': 6, 'BD': 0}

Fair bottleneck sharing
-----------------------

With capacity constraints enabled, flows that need the same bottleneck link
share its available capacity proportionally. Here both OD pairs request
``10`` units through ``CD``, but ``CD`` has capacity ``10``. Each pair
receives ``5`` and leaves ``5`` unassigned.

>>> bottleneck_network = Network.from_dataframe(
...     pd.DataFrame(
...         {
...             "edge_from": ["A", "B", "C"],
...             "edge_to": ["C", "C", "D"],
...             "edge_id": ["AC", "BC", "CD"],
...             "cost": [1, 1, 1],
...             "capacity": [100, 100, 10],
...         }
...     )
... )
>>> bottleneck_demand = Demand.from_dataframe(
...     pd.DataFrame(
...         {
...             "origin_id": ["A", "B"],
...             "destination_id": ["D", "D"],
...             "value": [10, 10],
...         }
...     )
... )
>>> constrained = assign(
...     bottleneck_network,
...     bottleneck_demand,
...     "sequential",
...     capacity_constrained=True,
...     include_paths=True,
... )
>>> [
...     {**row, "edge_path": list(row["edge_path"])}
...     for row in constrained.paths.to_pandas()
...     .sort_values(["origin_id", "destination_id"])
...     .to_dict("records")
... ]
[{'origin_id': 'A', 'destination_id': 'D', 'flow': 5, 'edge_path': ['AC', 'CD'], 'cost': 2}, {'origin_id': 'B', 'destination_id': 'D', 'flow': 5, 'edge_path': ['BC', 'CD'], 'cost': 2}]
>>> constrained.unassigned.to_pandas().sort_values(
...     ["origin_id", "destination_id"]
... ).to_dict("records")
[{'origin_id': 'A', 'destination_id': 'D', 'value': 5}, {'origin_id': 'B', 'destination_id': 'D', 'value': 5}]
>>> constrained.link_flows.to_pandas().set_index("edge_id")["flow"].to_dict()
{'AC': 5, 'BC': 5, 'CD': 10}

This is a heuristic, and an order-dependent one: demand is taken pair by
pair, so which pair gets the last of a scarce link depends on the order the
pairs arrive in. It answers "how much of this demand fits", not "where would
traffic settle" — that is :doc:`../explanation/equilibrium`.

Existing link loads
-------------------

If the network already carries a ``flow`` column, capacity-constrained
assignment treats that flow as occupied capacity. Returned link totals
include both the existing load and the newly assigned demand.

>>> loaded_network = Network.from_dataframe(
...     pd.DataFrame(
...         {
...             "edge_from": ["A"],
...             "edge_to": ["B"],
...             "edge_id": ["AB"],
...             "cost": [1],
...             "capacity": [10],
...             "flow": [8],
...         }
...     )
... )
>>> extra_demand = Demand.from_dataframe(
...     pd.DataFrame(
...         {
...             "origin_id": ["A"],
...             "destination_id": ["B"],
...             "value": [2],
...         }
...     )
... )
>>> loaded = assign(
...     loaded_network, extra_demand, "sequential", capacity_constrained=True
... )
>>> loaded.link_flows.to_pandas().set_index("edge_id")["flow"].to_dict()
{'AB': 10}
