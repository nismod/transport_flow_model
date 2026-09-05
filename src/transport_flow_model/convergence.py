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

Link travel times come from :mod:`transport_flow_model.costs`, whose cost
functions satisfy the :class:`~transport_flow_model.costs.CostFunction`
protocol. :func:`link_costs` evaluates the default one,
:class:`~transport_flow_model.costs.BPR`, built from the network's link
attributes: ``cost * (1 + alpha * (x / capacity)^beta) + distance_cost *
length``, the convention used by the TNTP datasets (see
:data:`transport_flow_model.datasets.BEST_KNOWN`). Networks without
``alpha``/``beta``/``capacity`` attributes are treated as fixed-cost
(flow-independent) networks.

Cost of evaluating the gap
--------------------------

If you are writing an iterative method, budget for this. Evaluating the
relative gap asks for one shortest-path cost per OD pair, which
:func:`relative_gap` gets from a single
:func:`transport_flow_model.core.skim` call: the network is parsed once and
one tree is built per distinct origin. That costs a fraction of an
all-or-nothing pass, so checking convergence every iteration is affordable.
Measured with ``scripts/profile_gap_cost.py`` (``pixi run
profile-gap-cost``; median of 5 repeats, single-threaded):

===============  =====  =======  =========  =========  ==========  ==========
instance         links  origins  AON (s)    gap (s)    gap / AON   gap share
===============  =====  =======  =========  =========  ==========  ==========
siouxfalls          76       24     0.0013     0.0005       0.35x        26%
anaheim            914       38     0.0036     0.0020       0.55x        36%
chicago-sketch    2950      386     0.1132     0.0457       0.40x        29%
===============  =====  =======  =========  =========  ==========  ==========

It has not always been this way, and the reason is worth knowing before you
add another call into the extension. :func:`relative_gap` used to loop in
Python over unique origins calling
:func:`transport_flow_model.core.shortest_paths_from`, and *every* call to
that entry point re-reads the link table and rebuilds the graph — 85% of a
call on chicago-sketch is parsing, not searching. The gap cost 0.47s there,
three times an all-or-nothing pass. Asking for all the pairs in one call
made it ten times faster.

So: **batch your calls across the boundary.** A per-origin or per-scenario
Python loop around a ``core`` function pays to rebuild the whole network
every iteration. Where a batch is not natural, parse once with
``core.prepare(links)`` and call methods on the result.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyarrow as pa

from transport_flow_model import core
from transport_flow_model.costs import BPR, _as_float_array, _flow_array
from transport_flow_model.demand import Demand
from transport_flow_model.network import Network


class ConvergenceWarning(UserWarning):
    """An iterative method stopped before reaching its target gap.

    Raised as a warning rather than an error because the result is still
    usable — a partly converged flow pattern is the right answer to a
    screening run — but it is not what was asked for, and the difference
    matters: Boyce et al. (2004) put the threshold for trusting flow
    differences between scenarios at a gap of 1e-4.

    Silence it with :func:`warnings.simplefilter`, or avoid it by raising
    ``max_iterations`` or relaxing ``target_gap``.
    """


def link_costs(
    network: Network,
    flows: Any,
    *,
    distance_cost: float = 0.0,
) -> np.ndarray:
    """Congested link travel times ``t_a(x_a)``, in network link order.

    Evaluates :class:`transport_flow_model.costs.BPR` built from the
    network's link attributes; use that class directly if you also need the
    integral or the derivative.

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
    return BPR.from_network(network, distance_cost=distance_cost).travel_time(x)


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
    skims = core.skim(
        congested,
        od.select(["origin_id", "destination_id"]),
        directed=directed,
    )

    unreachable = skims["cost"].null_count
    if unreachable:
        raise ValueError(
            f"{unreachable} OD pairs have unreachable destinations at congested "
            "costs; relative gap is undefined for infeasible demand"
        )
    values = od["value"].to_numpy(zero_copy_only=False).astype("float64")
    costs = skims["cost"].to_numpy(zero_copy_only=False)
    min_cost_total = float(np.dot(values, costs))
    if min_cost_total <= 0.0:
        raise ValueError("Total shortest-path travel time is zero or negative")
    return (total_cost - min_cost_total) / min_cost_total
