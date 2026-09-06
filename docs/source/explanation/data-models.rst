The Data Model
==============

Four objects carry everything through a run:

- :class:`~transport_flow_model.Network` stores the links and their
  attributes, and the node ids implied by them
- :class:`~transport_flow_model.Demand` stores origin-destination demand,
  one row per OD pair
- :class:`~transport_flow_model.AssignmentResult` stores what
  :func:`~transport_flow_model.assign` produced: link flows, OD skims,
  unassigned demand, and optionally the paths taken
- :class:`~transport_flow_model.Provenance`, attached to every result,
  records how it was produced

``Network`` and ``Demand`` are immutable. Every table in and out is a
:class:`pyarrow.Table`; ``to_dataframe()`` gives you a pandas copy when that
is more convenient.

Network links
-------------

Build a ``Network`` from a DataFrame with ``edge_from``, ``edge_to`` and
``edge_id`` columns. Optional columns such as ``cost`` and ``capacity`` are
used later when demand is assigned.

>>> import pandas as pd
>>> from transport_flow_model import Network, Demand, assign
>>> network = Network.from_dataframe(
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

Nodes are not declared separately: they are factorized from the link
endpoints, so this three-link chain has four of them.

>>> network.n_links, network.n_nodes
(3, 4)

OD demand
---------

A ``Demand`` holds one row per origin-destination pair, in
``origin_id``/``destination_id``/``value`` form. Ids must be node ids of the
network it will be assigned to.

>>> demand = Demand.from_dataframe(
...     pd.DataFrame(
...         {
...             "origin_id": ["A"],
...             "destination_id": ["C"],
...             "value": [10],
...         }
...     )
... )
>>> demand.to_dataframe().to_dict("records")
[{'origin_id': 'A', 'destination_id': 'C', 'value': 10.0}]
>>> demand.n_pairs, demand.total
(1, 10.0)

Assignment results
------------------

:func:`~transport_flow_model.assign` returns an
:class:`~transport_flow_model.AssignmentResult`. ``link_flows`` is the
network's own link table with a ``flow`` column appended, so it stays in
network link order and keeps the attributes you built the network with.

>>> result = assign(network, demand, "sequential")
>>> result.link_flows.to_pandas().set_index("edge_id")["flow"].to_dict()
{'AB': 10, 'BC': 10, 'CD': 0}

``skims`` gives the cost of reaching each destination, and ``unassigned``
the demand that could not be routed at all — empty here, since ``C`` is
reachable from ``A``.

>>> result.skims.to_pandas().to_dict("records")
[{'origin_id': 'A', 'destination_id': 'C', 'cost': 2.0}]
>>> result.unassigned.num_rows
0

``paths`` is populated only when you ask for it, because on a large problem
it is far bigger than the flows. Each row is one path, with ``edge_path``
the list of link ids it uses.

>>> detailed = assign(network, demand, "sequential", include_paths=True)
>>> detailed.paths.to_pandas()["edge_path"].apply(list).tolist()
[['AB', 'BC']]

Provenance
----------

Every result carries the method, the options it was given, and enough
metadata to trace a number back to the code that produced it. Equilibrium
methods also fill in ``iterations`` and ``relative_gap``; a single
all-or-nothing pass has no gap to report.

>>> result.provenance.method
'sequential'
>>> result.provenance.iterations
1
>>> result.provenance.relative_gap is None
True

Reading tables from disk
------------------------

Input files rarely use these column names. Rename on the way in — or let a
JSON config do it for you, which is what
:class:`~transport_flow_model.RunConfig` is for (see
:doc:`../reference/configuration`).

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
...     links = pd.read_csv(csv_path).rename(
...         columns={
...             "from_id": "edge_from",
...             "to_id": "edge_to",
...             "id": "edge_id",
...             "flow_capacity": "capacity",
...             "gcost_usd_per_ton": "cost",
...         }
...     )
...     Network.from_dataframe(links).to_dataframe().to_dict("records")
[{'edge_from': 'A', 'edge_to': 'B', 'edge_id': 'E1', 'capacity': 100, 'cost': 10.5}]
