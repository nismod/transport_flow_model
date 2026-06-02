Multiple Flows And Capacity
===========================

When several OD pairs are allocated, ``NetworkFlows`` aggregates their flows on
shared edges.

Capacity-constrained allocation limits assignments to available edge capacity
and records residual demand in ``unassigned_od``.

Shared edges
------------

The two OD pairs below both use edge ``BC`` on their least-cost path.

>>> import pandas as pd
>>> from transport_flow_model.model import Network, NetworkFlows, OD, ODFlows
>>> network = Network(
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
>>> od = OD(
...     pd.DataFrame(
...         {
...             "origin_id": ["A", "B"],
...             "destination_id": ["C", "D"],
...             "flow": [10, 6],
...         }
...     )
... )
>>> result = network.allocate(od, directed=True)
>>> result.od_flows.to_dataframe().to_dict("records")
[{'origin_id': 'A', 'destination_id': 'C', 'flow': 10, 'edge_path': ['AB', 'BC'], 'cost': 3}, {'origin_id': 'B', 'destination_id': 'D', 'flow': 6, 'edge_path': ['BC', 'CD'], 'cost': 3}]
>>> result.network_flows.to_dataframe().set_index("edge_id")["flow"].to_dict()
{'AB': 10, 'BC': 16, 'AC': 0, 'CD': 6, 'BD': 0}

Fair bottleneck sharing
-----------------------

With capacity constraints enabled, flows that require the same bottleneck edge
share its available capacity proportionally. Here both OD pairs request ``10``
units through ``CD``, but ``CD`` only has capacity ``10``. Each pair receives
``5`` and leaves ``5`` unassigned.

>>> bottleneck_network = Network(
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
>>> bottleneck_od = OD(
...     pd.DataFrame(
...         {
...             "origin_id": ["A", "B"],
...             "destination_id": ["D", "D"],
...             "flow": [10, 10],
...         }
...     )
... )
>>> constrained = bottleneck_network.allocate(
...     bottleneck_od, capacity_constrained=True, directed=True
... )
>>> constrained.od_flows.to_dataframe().sort_values(
...     ["origin_id", "destination_id"]
... ).to_dict("records")
[{'origin_id': 'A', 'destination_id': 'D', 'flow': 5, 'edge_path': ['AC', 'CD'], 'cost': 2}, {'origin_id': 'B', 'destination_id': 'D', 'flow': 5, 'edge_path': ['BC', 'CD'], 'cost': 2}]
>>> constrained.unassigned_od.to_dataframe().sort_values(
...     ["origin_id", "destination_id"]
... ).to_dict("records")
[{'origin_id': 'A', 'destination_id': 'D', 'flow': 5}, {'origin_id': 'B', 'destination_id': 'D', 'flow': 5}]
>>> constrained.network_flows.to_dataframe().set_index("edge_id")["flow"].to_dict()
{'AC': 5, 'BC': 5, 'CD': 10}

Existing edge loads
-------------------

If the network already has a ``flow`` column, capacity-constrained allocation
treats that flow as occupied capacity. Returned edge totals include both the
existing load and newly assigned OD flows.

>>> loaded_network = Network(
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
>>> extra_od = OD(
...     pd.DataFrame(
...         {
...             "origin_id": ["A"],
...             "destination_id": ["B"],
...             "flow": [2],
...         }
...     )
... )
>>> loaded = loaded_network.allocate(
...     extra_od, capacity_constrained=True, directed=True
... )
>>> loaded.network_flows.to_dataframe().set_index("edge_id")["flow"].to_dict()
{'AB': 10}

The same additive behavior is available directly through
``NetworkFlows.from_network_and_od_flows``.

>>> od_flows = ODFlows(
...     pd.DataFrame(
...         {
...             "origin_id": ["A"],
...             "destination_id": ["B"],
...             "flow": [2],
...             "edge_path": [["AB"]],
...             "cost": [1],
...         }
...     )
... )
>>> NetworkFlows.from_network_and_od_flows(
...     loaded_network, od_flows
... ).to_dataframe().set_index("edge_id")["flow"].to_dict()
{'AB': 10}
