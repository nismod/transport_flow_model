"""Flow assignment: ``assign(network, demand, method=...)``.

Backends are registered by name so that new assignment methods (MSA,
Frank-Wolfe, STAQ, ...) can be added without changing the calling code.
Results are :class:`pyarrow.Table` values throughout, and every
:class:`AssignmentResult` carries :class:`Provenance` metadata so runs are
comparable and reproducible.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import pyarrow as pa
import pyarrow.compute as pc

from transport_flow_model import core
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
    include_paths : bool
        Also return per-path detail (costs memory on large problems).
    seed : int, optional
        Random seed, recorded in provenance; deterministic methods ignore it.
    **options
        Method-specific options (e.g. ``capacity_constrained``, ``directed``
        for the sequential method).
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


def _not_implemented(name: str, issue: str) -> AssignmentBackend:
    def backend(*args: Any, **kwargs: Any) -> dict:
        raise NotImplementedError(
            f"Assignment method {name!r} is planned but not implemented yet "
            f"(workplan {issue})"
        )

    return backend


METHODS["msa"] = _not_implemented("msa", "ws2-02")
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
