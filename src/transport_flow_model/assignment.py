"""Flow assignment: ``assign(network, demand, method=...)``.

Backends are registered by name so that new assignment methods (MSA,
Frank-Wolfe, STAQ, ...) can be added without changing the calling code.
Results are :class:`pyarrow.Table` values throughout, and every
:class:`AssignmentResult` carries :class:`Provenance` metadata so runs are
comparable and reproducible.

Implemented methods
-------------------

``"sequential"``
    All-or-nothing sequential loading, optionally capacity-constrained
    (``capacity_constrained``, ``directed``). One pass, no equilibrium.

``"msa"``
    Method of successive averages: a user-equilibrium method that averages
    repeated all-or-nothing loads at congested costs with step ``1/k``.
    Options: ``max_iterations``, ``target_gap``, ``time_limit_s``,
    ``cost_function``, ``distance_cost``, ``directed``. It reports a gap
    trajectory, gets each gap free from the next all-or-nothing pass, and
    never returns per-OD paths.

``"fw"``, ``"bfw"``, ``"staq"``
    Reserved names, registered to stubs that raise
    :class:`NotImplementedError` naming the workplan issue. They are filled
    in, never renamed; see ``docs/adr/0001-assignment-methods-are-registered-backends.md``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Mapping, NamedTuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from transport_flow_model import core
from transport_flow_model.costs import BPR, CostFunction
from transport_flow_model.demand import Demand
from transport_flow_model.network import Network

#: Columns of the per-path table (``AssignmentResult.paths``).
PATH_COLUMNS = ("origin_id", "destination_id", "flow", "edge_path", "cost")
#: Tolerance when deciding whether a float column holds integral values.
INTEGRAL_EPSILON = 1.0e-9


@dataclass(frozen=True)
class Provenance:
    """How an assignment result was produced."""

    method: str
    options: Mapping[str, Any] = field(default_factory=dict)
    iterations: int = 0
    relative_gap: float | None = None
    wall_time_s: float | None = None
    seed: int | None = None
    package_version: str = ""
    core_version: str = ""


@dataclass(frozen=True)
class AssignmentResult:
    """Result of assigning demand to a network.

    All tables are :class:`pyarrow.Table`:

    - ``link_flows``: the network link table plus a ``flow`` column, in
      network link order.
    - ``skims``: per OD pair, the flow-weighted mean ``cost`` over its
      assigned paths.
    - ``unassigned``: OD demand (``origin_id``, ``destination_id``,
      ``value``) that could not be assigned (disconnected or out of
      capacity).
    - ``paths``: per-path detail (``origin_id``, ``destination_id``,
      ``flow``, ``edge_path``, ``cost``); only populated when
      ``assign(..., include_paths=True)``.
    """

    link_flows: pa.Table
    skims: pa.Table
    unassigned: pa.Table
    provenance: Provenance
    paths: pa.Table | None = None
    gap_history: tuple[float, ...] = ()

    @classmethod
    def from_tables(
        cls,
        link_flows: pa.Table,
        *,
        paths: pa.Table | None = None,
        skims: pa.Table | None = None,
        unassigned: pa.Table | None = None,
        provenance: Provenance | None = None,
    ) -> AssignmentResult:
        """Wrap previously saved tables (e.g. reloaded from CSV/Parquet)."""
        if skims is None and paths is not None:
            skims = _skims_from_paths(paths)
        return cls(
            link_flows=link_flows,
            skims=skims if skims is not None else _empty_skims(),
            unassigned=(unassigned if unassigned is not None else _empty_unassigned()),
            provenance=provenance or _provenance("loaded", {}, 0, None, None, None),
            paths=paths,
        )


AssignmentBackend = Callable[..., dict]

#: Registered assignment backends, keyed by method name.
METHODS: dict[str, AssignmentBackend] = {}


def register_method(name: str) -> Callable[[AssignmentBackend], AssignmentBackend]:
    """Register an assignment backend under ``name`` for :func:`assign`."""

    def decorator(backend: AssignmentBackend) -> AssignmentBackend:
        METHODS[name] = backend
        return backend

    return decorator


def assign(
    network: Network,
    demand: Demand,
    method: str = "sequential",
    *,
    include_paths: bool = False,
    seed: int | None = None,
    **options: Any,
) -> AssignmentResult:
    """Assign OD demand to least-cost network paths.

    Parameters
    ----------
    network : Network
        Immutable network; must carry a ``cost`` link attribute (and
        ``capacity`` for capacity-constrained methods).
    demand : Demand
        OD demand; ids must be network node ids.
    method : str
        Assignment method name; see :data:`METHODS` for what is available.
        ``"sequential"`` and ``"msa"`` are implemented.
    include_paths : bool
        Also return per-path detail (costs memory on large problems). An
        equilibrium method has no single path per OD pair and rejects it.
    seed : int, optional
        Random seed, recorded in provenance; deterministic methods ignore it.
    **options
        Method-specific options, declared by the backend:

        - ``"sequential"``: ``capacity_constrained``, ``directed``.
        - ``"msa"``: ``max_iterations``, ``target_gap``, ``time_limit_s``,
          ``cost_function``, ``distance_cost``, ``directed``.

        An option the chosen method does not declare raises
        :class:`TypeError`.
    """
    try:
        backend = METHODS[method]
    except KeyError:
        raise ValueError(
            f"Unknown assignment method {method!r}; "
            f"available: {', '.join(sorted(METHODS))}"
        ) from None
    start = time.perf_counter()
    output = backend(network, demand, include_paths=include_paths, **options)
    wall_time_s = time.perf_counter() - start
    return AssignmentResult(
        link_flows=output["link_flows"],
        skims=output["skims"],
        unassigned=output["unassigned"],
        paths=output.get("paths"),
        gap_history=tuple(output.get("gap_history", ())),
        provenance=_provenance(
            method,
            options,
            output.get("iterations", 0),
            output.get("relative_gap"),
            wall_time_s,
            seed,
        ),
    )


@register_method("sequential")
def _assign_sequential(
    network: Network,
    demand: Demand,
    *,
    include_paths: bool = False,
    capacity_constrained: bool = False,
    directed: bool = True,
) -> dict:
    """All-or-nothing sequential loading, optionally capacity-constrained.

    Each OD pair in turn is assigned to its least-cost path; with
    ``capacity_constrained=True``, flows are limited by residual link
    capacity and split across successively costlier paths (order-dependent
    heuristic).
    """
    od = demand.to_table()
    core_demand = od.select(["origin_id", "destination_id", "value"]).rename_columns(
        ["origin_id", "destination_id", "flow"]
    )
    result = core.allocate(
        network.to_table(),
        core_demand,
        capacity_constrained=capacity_constrained,
        directed=directed,
    )
    paths = coerce_integral(result["od_flows"].select(list(PATH_COLUMNS)))
    unassigned = coerce_integral(
        result["unassigned_od"].select(["origin_id", "destination_id", "flow"])
    ).rename_columns(["origin_id", "destination_id", "value"])
    return {
        "link_flows": link_flows_table(network.to_table(), result["network_flows"]),
        "skims": _skims_from_paths(paths),
        "unassigned": unassigned,
        "paths": paths if include_paths else None,
        "iterations": 1,
        "relative_gap": None,
    }


class _MSAIterate(NamedTuple):
    """One method-of-successive-averages iteration.

    Attributes
    ----------
    k : int
        Iteration number, equal to the number of all-or-nothing passes
        performed so far.
    flows : numpy.ndarray
        Current link flows, ``float64`` in network link order.
    gap : float or None
        Relative gap measured during pass ``k``, which is the gap of the
        iterate *entering* the pass — that is, of ``flows`` as it stood
        before this pass averaged into it. ``None`` for ``k == 1``, where
        no second load exists to measure against. When the loop stops on
        the gap (or on the time limit) no averaging happens, so the last
        iterate's ``gap`` is exactly the gap of its ``flows``.
    """

    k: int
    flows: np.ndarray
    gap: float | None


class _MSAContext:
    """Inputs of an MSA run, parsed once and reused by every iteration.

    Holds the link table, a :class:`~transport_flow_model.core.PreparedNetwork`
    built from it, the OD table in the column layout ``core.allocate``
    wants, and the cost function. :meth:`aon` is one all-or-nothing load at
    a given set of link costs.
    """

    __slots__ = ("links", "prepared", "od", "cost_function", "directed", "unassigned")

    def __init__(
        self,
        network: Network,
        demand: Demand,
        *,
        cost_function: CostFunction | None = None,
        distance_cost: float = 0.0,
        directed: bool = True,
    ):
        self.links = network.to_table()
        self.prepared = core.prepare(self.links)
        self.od = (
            demand.to_table()
            .select(["origin_id", "destination_id", "value"])
            .rename_columns(["origin_id", "destination_id", "flow"])
        )
        self.cost_function = cost_function or BPR.from_network(
            network, distance_cost=distance_cost
        )
        self.directed = directed
        #: Demand that no path can carry; set by the first :meth:`aon` call.
        self.unassigned: pa.Table | None = None

    def aon(self, costs: np.ndarray) -> np.ndarray:
        """All-or-nothing link loads at ``costs``, in network link order.

        Records ``unassigned`` from the first call only: reachability
        depends on the topology, not on the costs, so it never changes.
        """
        self.prepared.set_costs(costs)
        result = self.prepared.allocate(
            self.od, capacity_constrained=False, directed=self.directed
        )
        if self.unassigned is None:
            self.unassigned = coerce_integral(
                result["unassigned_od"].select(["origin_id", "destination_id", "flow"])
            ).rename_columns(["origin_id", "destination_id", "value"])
        flows = link_flows_table(self.links, result["network_flows"], coerce=False)
        return flows["flow"].to_numpy(zero_copy_only=False).astype("float64")

    def skims(self, costs: np.ndarray) -> pa.Table:
        """Least-cost travel time per OD pair at ``costs``."""
        self.prepared.set_costs(costs)
        return self.prepared.skim(
            self.od.select(["origin_id", "destination_id"]), directed=self.directed
        )


def _msa_iterates(
    network: Network,
    demand: Demand,
    *,
    max_iterations: int = 50,
    target_gap: float = 1.0e-4,
    time_limit_s: float | None = None,
    cost_function: CostFunction | None = None,
    distance_cost: float = 0.0,
    directed: bool = True,
    context: _MSAContext | None = None,
) -> Iterator[_MSAIterate]:
    """Yield one :class:`_MSAIterate` per all-or-nothing pass.

    The loop the ``"msa"`` backend runs, exposed so that tests can check
    each reported gap against an independent
    :func:`~transport_flow_model.relative_gap` evaluation of the iterate it
    belongs to. Pass a ``context`` to keep hold of the prepared network and
    the unassigned demand after the loop finishes; one is built if you do
    not.

    Raises :class:`ValueError` if the shortest-path travel time of an
    all-or-nothing load is not positive, as
    :func:`~transport_flow_model.relative_gap` does.
    """
    if max_iterations < 1:
        raise ValueError(f"max_iterations must be at least 1, got {max_iterations}")
    if context is None:
        context = _MSAContext(
            network,
            demand,
            cost_function=cost_function,
            distance_cost=distance_cost,
            directed=directed,
        )
    start = time.perf_counter()
    travel_time = context.cost_function.travel_time

    x = context.aon(travel_time(0))
    yield _MSAIterate(1, x, None)

    for k in range(2, max_iterations + 1):
        t = travel_time(x)
        y = context.aon(t)
        # The all-or-nothing load y puts every OD pair's demand on its
        # min-cost path at costs t, so dot(t, y) is exactly the
        # shortest-path travel time term of the relative gap at flows x.
        shortest_path_total = float(np.dot(t, y))
        if shortest_path_total <= 0.0:
            raise ValueError("Total shortest-path travel time is zero or negative")
        gap = float(np.dot(t, x)) / shortest_path_total - 1.0
        out_of_time = (
            time_limit_s is not None and time.perf_counter() - start >= time_limit_s
        )
        if gap <= target_gap or out_of_time:
            yield _MSAIterate(k, x, gap)
            return
        x = x + (y - x) / k
        yield _MSAIterate(k, x, gap)


@register_method("msa")
def _assign_msa(
    network: Network,
    demand: Demand,
    *,
    include_paths: bool = False,
    max_iterations: int = 50,
    target_gap: float = 1.0e-4,
    time_limit_s: float | None = None,
    cost_function: CostFunction | None = None,
    distance_cost: float = 0.0,
    directed: bool = True,
) -> dict:
    """Method of successive averages: user equilibrium by averaged AON loads.

    Each iteration loads all demand on the least-cost paths at the current
    congested costs (an all-or-nothing pass) and averages that load into the
    running solution with step ``1/k``. The averaging is what makes it
    converge; it converges slowly near the optimum, which is why it is the
    baseline other equilibrium methods are measured against rather than the
    method of choice.

    Parameters
    ----------
    max_iterations : int
        All-or-nothing passes to perform at most.
    target_gap : float
        Stop once the relative gap is at or below this. Boyce et al. (2004)
        recommend 1e-4 before trusting flow differences between scenarios.
    time_limit_s : float, optional
        Wall-clock budget. Checked after each gap evaluation, so at least
        two passes run when it is set.
    cost_function : CostFunction, optional
        Volume-delay function; defaults to
        :meth:`~transport_flow_model.BPR.from_network`.
    distance_cost : float
        Generalized-cost weight on the ``length`` attribute, passed to that
        default and unused when ``cost_function`` is given.
    directed : bool
        Treat links as one-way.

    Notes
    -----
    **The gap is free.** Iteration ``k`` needs an all-or-nothing load ``y``
    at the current costs ``t`` anyway, and that load puts every OD pair's
    demand on its min-cost path — so ``dot(t, y)`` *is* the shortest-path
    travel time term of the relative gap at the current flows ``x``, and the
    gap is ``dot(t, x) / dot(t, y) - 1``. No extra skim, and no separate
    :func:`~transport_flow_model.relative_gap` call, is needed to decide
    when to stop.

    That gap belongs to the iterate entering the pass. When the loop stops
    because the target was reached (or the time limit expired) the averaging
    is skipped, so the reported ``relative_gap`` is exactly the gap of the
    returned flows. When instead the iteration budget runs out, the last
    pass does average, and the reported gap is that of the previous iterate
    — an upper bound on the returned one in practice, but not a measurement
    of it.

    When some demand is unassigned because its destination is unreachable,
    the gap is over the **assigned** demand only: unassigned demand loads no
    links, so it appears in neither ``dot(t, x)`` nor ``dot(t, y)``.
    :func:`~transport_flow_model.relative_gap` would instead raise on such a
    network, since a pair with no path has no shortest-path cost.

    ``paths`` is always ``None``. An equilibrium is an average of many
    all-or-nothing solutions and has no single path per OD pair, so
    ``include_paths=True`` raises :class:`NotImplementedError` rather than
    returning the last pass's paths, which would be a different (and much
    worse) solution than the flows beside them.
    """
    if include_paths:
        raise NotImplementedError(
            "msa does not produce per-OD paths; an equilibrium has no single "
            "path per OD pair (issues/m0-13)"
        )
    context = _MSAContext(
        network,
        demand,
        cost_function=cost_function,
        distance_cost=distance_cost,
        directed=directed,
    )
    gap_history: list[float] = []
    iterate = None
    for iterate in _msa_iterates(
        network,
        demand,
        max_iterations=max_iterations,
        target_gap=target_gap,
        time_limit_s=time_limit_s,
        directed=directed,
        context=context,
    ):
        if iterate.gap is not None:
            gap_history.append(iterate.gap)
    assert iterate is not None  # _msa_iterates always yields at least once

    flows = iterate.flows
    assert context.unassigned is not None  # set by the first all-or-nothing pass
    return {
        "link_flows": link_flows_table(
            context.links,
            pa.table(
                {
                    "edge_id": context.links["edge_id"],
                    "flow": pa.array(flows, type=pa.float64()),
                }
            ),
            coerce=False,
        ),
        "skims": context.skims(context.cost_function.travel_time(flows)),
        "unassigned": context.unassigned,
        "paths": None,
        "gap_history": tuple(gap_history),
        "iterations": iterate.k,
        "relative_gap": iterate.gap,
    }


def _not_implemented(name: str, issue: str) -> AssignmentBackend:
    def backend(*args: Any, **kwargs: Any) -> dict:
        raise NotImplementedError(
            f"Assignment method {name!r} is planned but not implemented yet "
            f"(workplan {issue})"
        )

    return backend


METHODS["fw"] = _not_implemented("fw", "ws2-03")
METHODS["bfw"] = _not_implemented("bfw", "ws2-03")
METHODS["staq"] = _not_implemented("staq", "ws2-06")


def link_flows_table(
    links: pa.Table, network_flows: pa.Table, *, coerce: bool = True
) -> pa.Table:
    """Attach per-link ``flow`` to a link table, preserving link order.

    Parameters
    ----------
    links : pyarrow.Table
        Link table to attach flows to; any existing ``flow`` column is
        replaced.
    network_flows : pyarrow.Table
        Per-link flows keyed by ``edge_id``; links not present get zero.
    coerce : bool
        Apply :func:`coerce_integral` to the result (the default, for
        legacy parity). **Iterative methods must pass ``coerce=False``**:
        an all-or-nothing first iteration on integral demand produces
        integral flows and would be cast to ``int64``, while later
        averaged iterations produce fractional flows and stay ``float64``,
        so the output dtype would otherwise depend on the iteration count
        and on the input data. The core extension always emits ``flow`` as
        ``float64``, so ``coerce=False`` is stably ``float64``.
    """
    positions = pc.index_in(links["edge_id"], network_flows["edge_id"].combine_chunks())
    flows = pc.fill_null(pc.take(network_flows["flow"], positions), 0)
    if "flow" in links.column_names:
        links = links.drop_columns(["flow"])
    table = links.append_column("flow", flows)
    return coerce_integral(table) if coerce else table


def coerce_integral(
    table: pa.Table, columns: tuple[str, ...] = ("flow", "cost")
) -> pa.Table:
    """Cast float columns holding only integral values to int64.

    Keeps result dtypes stable across methods and runs so that outputs
    compare cleanly (matches the legacy behaviour of
    ``transport_flow_model.model``).
    """
    if table.num_rows == 0:
        return table
    for name in columns:
        if name not in table.column_names:
            continue
        column = table[name]
        if not pa.types.is_floating(column.type) or column.null_count:
            continue
        if not pc.all(pc.is_finite(column)).as_py():
            continue
        rounded = pc.round(column)
        integral = pc.all(
            pc.less_equal(pc.abs(pc.subtract(column, rounded)), INTEGRAL_EPSILON)
        ).as_py()
        if integral:
            table = table.set_column(
                table.column_names.index(name),
                name,
                pc.cast(rounded, pa.int64()),
            )
    return table


def _skims_from_paths(paths: pa.Table) -> pa.Table:
    """Flow-weighted mean path cost per OD pair."""
    if paths.num_rows == 0:
        return _empty_skims(paths)
    weighted = paths.append_column(
        "_flow_cost",
        pc.multiply(
            pc.cast(paths["flow"], pa.float64()),
            pc.cast(paths["cost"], pa.float64()),
        ),
    )
    grouped = weighted.group_by(
        ["origin_id", "destination_id"], use_threads=False
    ).aggregate([("flow", "sum"), ("_flow_cost", "sum")])
    cost = pc.divide(
        grouped["_flow_cost_sum"], pc.cast(grouped["flow_sum"], pa.float64())
    )
    return pa.table(
        {
            "origin_id": grouped["origin_id"],
            "destination_id": grouped["destination_id"],
            "cost": cost,
        }
    )


def _empty_skims(like: pa.Table | None = None) -> pa.Table:
    id_type = like["origin_id"].type if like is not None else pa.int64()
    return pa.table(
        {
            "origin_id": pa.array([], type=id_type),
            "destination_id": pa.array([], type=id_type),
            "cost": pa.array([], type=pa.float64()),
        }
    )


def _empty_unassigned() -> pa.Table:
    return pa.table(
        {
            "origin_id": pa.array([], type=pa.int64()),
            "destination_id": pa.array([], type=pa.int64()),
            "value": pa.array([], type=pa.float64()),
        }
    )


def _provenance(
    method: str,
    options: Mapping[str, Any],
    iterations: int,
    relative_gap: float | None,
    wall_time_s: float | None,
    seed: int | None,
) -> Provenance:
    return Provenance(
        method=method,
        options=dict(options),
        iterations=iterations,
        relative_gap=relative_gap,
        wall_time_s=wall_time_s,
        seed=seed,
        package_version=_package_version(),
        core_version=core.version(),
    )


def _package_version() -> str:
    try:
        from transport_flow_model._version import __version__
    except ImportError:
        return "unknown"
    return __version__
