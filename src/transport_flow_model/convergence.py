"""Solution-quality metrics for traffic assignment.

The standard convergence measure for (user-equilibrium) assignment is the
*relative gap*: the excess of total system travel time over the total
shortest-path travel time at current link costs,

.. math::

    \\mathrm{gap} = \\frac{\\sum_a t_a(x_a) x_a - \\sum_{od} d_{od} c^{min}_{od}}
                        {\\sum_{od} d_{od} c^{min}_{od}}

Boyce, Ralevic-Dekic & Bar-Gera (2004, J. Transportation Engineering 130(1))
argue gaps of 1e-4 or better are needed before flow differences between
scenarios are trustworthy.

Link travel times follow the BPR convention used by the TNTP datasets
(see :data:`transport_flow_model.datasets.BEST_KNOWN`):
``cost * (1 + alpha * (x / capacity)^beta) + distance_cost * length``.
Networks without ``alpha``/``beta``/``capacity`` attributes are treated as
fixed-cost (flow-independent) networks.

Cost of evaluating the gap
--------------------------

If you are writing an iterative method, budget for this: **evaluating the
relative gap currently costs about three times an all-or-nothing pass**, so
checking convergence on every iteration is roughly 75-80% of the iteration,
not a rounding error. Measured with ``scripts/profile_gap_cost.py``
(median of 3-5 repeats, single-threaded):

===============  =====  =======  =========  =========  ==========  ==============
instance         links  origins  AON (s)    gap (s)    gap / AON   gap share
===============  =====  =======  =========  =========  ==========  ==============
siouxfalls          76       24     0.0017     0.0051       3.0x          75%
anaheim            914       38     0.0057     0.0219       3.8x          79%
chicago-sketch    2950      386     0.1631     0.4683       2.9x          74%
===============  =====  =======  =========  =========  ==========  ==============

The reason is not the shortest-path search itself. :func:`relative_gap`
loops in Python over unique origins calling
:func:`transport_flow_model.core.shortest_paths_from`, and that extension
entry point re-reads the link table and rebuilds the graph on *every* call.
The trees alone account for 0.28s of chicago-sketch's 0.47s — more than a
whole ``core.allocate`` pass over the same 386 origins, which builds the
graph once.

Practical consequences until that is fixed (see
``issues/m0-11-fuse-gap-evaluation-into-aon.md``):

- Evaluate the gap every *k* iterations, or only once a cheaper inner
  criterion suggests convergence, rather than unconditionally every
  iteration.
- Report the gap trajectory in ``gap_history`` at whatever cadence you
  chose, and say so in the method's docstring — the benchmark harness
  spreads the trajectory uniformly over wall time.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from transport_flow_model import core
from transport_flow_model.demand import Demand
from transport_flow_model.network import Network


def link_costs(
    network: Network,
    flows: Any,
    *,
    distance_cost: float = 0.0,
) -> np.ndarray:
    """Congested link travel times ``t_a(x_a)``, in network link order.

    Parameters
    ----------
    network : Network
        Must carry a ``cost`` link attribute (free-flow time). If
        ``alpha``, ``beta`` and ``capacity`` attributes are present the BPR
        volume-delay function is applied; otherwise costs are flow-independent.
    flows : array-like
        Link flows ``x_a`` in network link order.
    distance_cost : float
        Generalized-cost weight on the ``length`` attribute (e.g. 0.04 for
        the published chicago-sketch solution).
    """
    x = _as_float_array(flows, network.n_links, "flows")
    t = _as_float_array(network.attribute("cost"), network.n_links, "cost")
    names = network.to_table().column_names
    if {"alpha", "beta", "capacity"} <= set(names):
        alpha = _as_float_array(network.attribute("alpha"), network.n_links, "alpha")
        beta = _as_float_array(network.attribute("beta"), network.n_links, "beta")
        capacity = _as_float_array(
            network.attribute("capacity"), network.n_links, "capacity"
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(capacity > 0, x / capacity, 0.0)
        t = t * (1.0 + alpha * np.power(ratio, beta))
    if distance_cost:
        t = t + distance_cost * _as_float_array(
            network.attribute("length"), network.n_links, "length"
        )
    return t


def relative_gap(
    network: Network,
    demand: Demand,
    flows: Any,
    *,
    distance_cost: float = 0.0,
    directed: bool = True,
) -> float:
    """Relative gap of a link-flow solution against shortest-path costs.

    ``flows`` may be an array of link flows in network link order, an
    :class:`AssignmentResult`, or a table with a ``flow`` column in network
    link order. Zero at user equilibrium; Boyce et al. (2004) recommend
    converging below 1e-4.

    Raises :class:`ValueError` if any demanded destination is unreachable at
    congested costs, or if total shortest-path travel time is zero.
    """
    x = _flow_array(network, flows)
    t = link_costs(network, x, distance_cost=distance_cost)
    total_cost = float(np.dot(t, x))

    links = network.to_table()
    congested = links.set_column(
        links.column_names.index("cost"), "cost", pa.array(t, type=pa.float64())
    )
    od = demand.to_table()
    origins = od["origin_id"].to_numpy(zero_copy_only=False)
    destinations = od["destination_id"].to_numpy(zero_copy_only=False)
    values = od["value"].to_numpy(zero_copy_only=False).astype("float64")

    min_cost_total = 0.0
    unreachable = 0
    for origin in np.unique(origins):
        reached = core.shortest_paths_from(congested, int(origin), directed=directed)
        selection = origins == origin
        positions = pc.index_in(
            pa.array(destinations[selection], type=reached["node_id"].type),
            reached["node_id"].combine_chunks(),
        )
        costs = pc.take(reached["cost"], positions).to_numpy(zero_copy_only=False)
        missing = np.isnan(costs)
        unreachable += int(missing.sum())
        min_cost_total += float(np.dot(values[selection][~missing], costs[~missing]))
    if unreachable:
        raise ValueError(
            f"{unreachable} OD pairs have unreachable destinations at congested "
            "costs; relative gap is undefined for infeasible demand"
        )
    if min_cost_total <= 0.0:
        raise ValueError("Total shortest-path travel time is zero or negative")
    return (total_cost - min_cost_total) / min_cost_total


def _flow_array(network: Network, flows: Any) -> np.ndarray:
    link_flows = getattr(flows, "link_flows", None)
    if link_flows is not None:  # AssignmentResult
        flows = link_flows
    if isinstance(flows, (pa.Table, pa.RecordBatch)):
        if "flow" not in flows.schema.names:
            raise ValueError("Flow table has no 'flow' column")
        flows = flows["flow"]
    return _as_float_array(flows, network.n_links, "flows")


def _as_float_array(values: Any, n: int, name: str) -> np.ndarray:
    if isinstance(values, (pa.Array, pa.ChunkedArray)):
        array = values.to_numpy(zero_copy_only=False)
    else:
        array = np.asarray(values)
    array = array.astype("float64", copy=False)
    if array.shape != (n,):
        raise ValueError(f"Expected {name} to have shape ({n},), got {array.shape}")
    return array
