Losses
======

Rerouting losses compare initial allocated paths with disrupted allocated paths
for each OD pair. The loss value is the difference between summed disrupted
cost and summed initial cost.

Grouped path costs
------------------

``OD.losses_from_flows`` groups rows by ``origin_id`` and ``destination_id``.
This is useful when one OD pair has multiple allocated rows.

>>> import pandas as pd
>>> from transport_flow_model.model import OD, ODFlows
>>> initial = ODFlows(
...     pd.DataFrame(
...         {
...             "origin_id": ["A", "A", "B"],
...             "destination_id": ["C", "C", "D"],
...             "flow": [4, 6, 2],
...             "edge_path": [["AC1"], ["AC2"], ["BD"]],
...             "cost": [2, 3, 7],
...         }
...     )
... )
>>> disrupted = ODFlows(
...     pd.DataFrame(
...         {
...             "origin_id": ["A", "A", "B"],
...             "destination_id": ["C", "C", "D"],
...             "flow": [4, 6, 2],
...             "edge_path": [["AC1"], ["AE", "EC"], ["BE", "ED"]],
...             "cost": [2, 5, 12],
...         }
...     )
... )
>>> losses = OD.losses_from_flows(initial, disrupted)
>>> losses.to_dataframe().sort_values(
...     ["origin_id", "destination_id"]
... ).to_dict("records")
[{'origin_id': 'A', 'destination_id': 'C', 'flow': 10, 'initial_cost': 5, 'disrupted_cost': 7, 'rerouting_loss': 2}, {'origin_id': 'B', 'destination_id': 'D', 'flow': 2, 'initial_cost': 7, 'disrupted_cost': 12, 'rerouting_loss': 5}]

For ``A -> C``, the initial costs ``2`` and ``3`` sum to ``5``. The disrupted
costs ``2`` and ``5`` sum to ``7``, so ``rerouting_loss`` is ``2``.
