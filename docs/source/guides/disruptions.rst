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
