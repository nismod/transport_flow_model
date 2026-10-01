"""Flow assignment: ``assign(network, demand, method=...)``.

Backends are registered by name so that new assignment methods (MSA,
Frank-Wolfe, STAQ, ...) can be added without changing the calling code.
Results are :class:`pyarrow.Table` values throughout, and every
:class:`AssignmentResult` carries :class:`Provenance` metadata so runs are
comparable and reproducible.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from transport_flow_model import core
from transport_flow_model.demand import Demand
from transport_flow_model.network import Network

#: Columns of the per-path table (``AssignmentResult.paths``).
PATH_COLUMNS = ("origin_id", "destination_id", "flow", "edge_path", "cost")
#: Tolerance when deciding whether a float column holds integral values.
INTEGRAL_EPSILON = 1.0e-9
LEGACY_MILE_TO_KM = 1.60934
LEGACY_FUEL_PRICE_GBP_PER_LITRE = 1.4
LEGACY_COST_EPSILON = 1.0e-6
LEGACY_CAPACITY_EPSILON = 1.0e-9
LEGACY_SATURATED_EDGE_PENALTY = 1.0e18

LEGACY_VALUE_OF_TIME = {
    "car": 21.41,
    "lgv": 16.87,
    "ogv": 20.35,
    "psv": 13.15,
    "rail": 38.28,
}
LEGACY_FUEL_LITRE_PER_KM = {
    "car": (0.46693, 0.08257, -0.00088, 0.000006),
    "lgv": (0.40392, 0.15318, -0.00467, 0.000033),
    "ogv": (4.18094, 0.21864, -0.00298, 0.000023),
    "psv": (3.36019, 0.29525, -0.00321, 0.000024),
}
LEGACY_NON_FUEL_PENCE_PER_KM = {
    "car": (6.885, 188.486),
    "lgv": (10.001, 65.321),
    "ogv": (13.709, 535.418),
    "psv": (42.233, 962.974),
}
LEGACY_SPEED_REDUCTION = {
    "M": 0.033,
    "A_dual": 0.033,
    "A_single": 0.05,
    "B": 0.05,
    "B_dual": 0.05,
    "B_single": 0.05,
}


def _legacy_congestion_costs(
    links: pa.Table, flows: np.ndarray, vehicle_type: str
) -> np.ndarray:
    """Calculate legacy speed, time, operating, and toll costs per link."""
    if vehicle_type not in LEGACY_VALUE_OF_TIME:
        supported = ", ".join(LEGACY_VALUE_OF_TIME)
        raise ValueError(f"vehicle_type must be one of: {supported}")

    required_columns = {
        "length_mile",
        "initial_flow_speeds",
        "min_flow_speeds",
        "breakpoint_flows",
        "combined_label",
    }
    missing = sorted(required_columns - set(links.column_names))
    if missing:
        raise ValueError(
            "Legacy congestion assignment requires link columns: "
            f"{', '.join(missing)}"
        )

    def float_column(name: str) -> np.ndarray:
        return pc.cast(links[name], pa.float64()).to_numpy(zero_copy_only=False)

    labels = links["combined_label"].to_pylist()
    reduction_factors = np.fromiter(
        (LEGACY_SPEED_REDUCTION.get(label, 0.0) for label in labels),
        dtype="float64",
        count=links.num_rows,
    )
    initial_speeds = float_column("initial_flow_speeds")
    minimum_speeds = float_column("min_flow_speeds")
    breakpoints = float_column("breakpoint_flows")
    flow_per_hour = np.asarray(flows, dtype="float64") / 24.0
    excess_flow = np.maximum(flow_per_hour - breakpoints, 0.0)
    speeds = np.maximum(
        initial_speeds - reduction_factors * excess_flow, minimum_speeds
    )

    distance_miles = float_column("length_mile")
    distance_km = distance_miles * LEGACY_MILE_TO_KM
    time_hours = distance_miles / np.maximum(speeds, LEGACY_COST_EPSILON)
    speed_kmph = distance_km / np.maximum(time_hours, LEGACY_COST_EPSILON)

    occupancy = 1.06 if vehicle_type == "car" else 1.0
    time_cost = time_hours * LEGACY_VALUE_OF_TIME[vehicle_type] * occupancy
    operating_cost = np.zeros(links.num_rows, dtype="float64")
    if vehicle_type in ("car", "lgv", "ogv"):
        a, b, c, d = LEGACY_FUEL_LITRE_PER_KM[vehicle_type]
        fuel_litres_per_km = (
            a / np.maximum(speed_kmph, LEGACY_COST_EPSILON)
            + b
            + c * speed_kmph
            + d * speed_kmph**2
        )
        fuel_cost_per_km = fuel_litres_per_km * LEGACY_FUEL_PRICE_GBP_PER_LITRE
        non_fuel_a, non_fuel_b = LEGACY_NON_FUEL_PENCE_PER_KM[vehicle_type]
        non_fuel_cost_per_km = (
            non_fuel_a + non_fuel_b / np.maximum(speed_kmph, LEGACY_COST_EPSILON)
        ) / 100.0
        operating_cost = (fuel_cost_per_km + non_fuel_cost_per_km) * distance_km

    if "average_toll_cost" in links.column_names:
        toll_cost = float_column("average_toll_cost")
    else:
        toll_cost = np.zeros(links.num_rows, dtype="float64")
    return time_cost + operating_cost + toll_cost


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


@register_method("congestion")
def _assign_congestion(
    network: Network,
    demand: Demand,
    *,
    include_paths: bool = False,
    capacity_constrained: bool = True,
    directed: bool = False,
    max_iterations: int = 5,
    vehicle_type: str = "car",
) -> dict:
    """Iteratively load remaining demand with legacy capacity penalties.

    Each round assigns all remaining OD demand without capacity constraints, then
    scales each OD flow by the tightest remaining-capacity ratio along its path.
    Near-saturated links are penalized before costs and routes are recalculated for
    the residual demand. Links need the legacy speed (mph) and length fields (miles).

    Citation:
    Li, Y., Pant, R., Russell, T., Thomas, F., Hall, J. W., Oldham, P., ... & Young, P. J.
    (2026). Stress-testing road network resilience using counterfactual flood events
    (1953 - 2024) in Great Britain. Transportation Research Part D: Transport and Environment,
    155, 105292.
    """
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise TypeError("max_iterations must be a positive integer")
    if max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if vehicle_type not in LEGACY_VALUE_OF_TIME:
        supported = ", ".join(LEGACY_VALUE_OF_TIME)
        raise ValueError(f"vehicle_type must be one of: {supported}")

    links = network.to_table()
    od = demand.to_table()
    origin_ids = od["origin_id"].to_pylist()
    destination_ids = od["destination_id"].to_pylist()
    demand_values = pc.cast(od["value"], pa.float64()).to_numpy(zero_copy_only=False)

    # Combine duplicate OD rows so each pair has one remaining-demand balance.
    total_by_pair: dict[tuple[Any, Any], float] = {}
    for origin, destination, value in zip(origin_ids, destination_ids, demand_values):
        pair = (origin, destination)
        total_by_pair[pair] = total_by_pair.get(pair, 0.0) + float(value)
    remaining_by_pair = total_by_pair.copy()

    # Carry any existing link loads forward and align capacities to link order.
    if "flow" in links.column_names:
        cumulative_flows = pc.fill_null(
            pc.cast(links["flow"], pa.float64()), 0.0
        ).to_numpy(zero_copy_only=False)
    else:
        cumulative_flows = np.zeros(links.num_rows, dtype="float64")

    if capacity_constrained and "capacity" in links.column_names:
        capacities = pc.fill_null(
            pc.cast(links["capacity"], pa.float64()), np.inf
        ).to_numpy(zero_copy_only=False)
        capacities = np.where(
            np.isfinite(capacities), np.maximum(capacities, 0.0), np.inf
        )
    else:
        capacities = np.full(links.num_rows, np.inf, dtype="float64")
    edge_positions = {
        edge_id: index for index, edge_id in enumerate(links["edge_id"].to_pylist())
    }

    assigned_by_pair: dict[tuple[Any, Any], float] = {}
    skim_totals: dict[tuple[Any, Any], list[float]] = {}
    path_tables = []
    unreachable_pairs: set[tuple[Any, Any]] = set()
    iterations_run = 0

    for _ in range(max_iterations):
        # Route only demand that still has a feasible destination.
        batch_pairs = [
            pair
            for pair, flow in remaining_by_pair.items()
            if flow > LEGACY_CAPACITY_EPSILON and pair not in unreachable_pairs
        ]
        if not batch_pairs:
            break

        batch = pa.table(
            {
                "origin_id": pa.array(
                    [pair[0] for pair in batch_pairs], type=od["origin_id"].type
                ),
                "destination_id": pa.array(
                    [pair[1] for pair in batch_pairs],
                    type=od["destination_id"].type,
                ),
                "flow": pa.array(
                    [remaining_by_pair[pair] for pair in batch_pairs],
                    type=pa.float64(),
                ),
            }
        )

        costs = _legacy_congestion_costs(links, cumulative_flows, vehicle_type)
        remaining_capacity = np.maximum(capacities - cumulative_flows, 0.0)
        finite_capacity = np.isfinite(capacities)
        utilization_remaining = np.ones(links.num_rows, dtype="float64")
        np.divide(
            remaining_capacity,
            capacities,
            out=utilization_remaining,
            where=finite_capacity & (capacities > 0.0),
        )
        saturated_edges = finite_capacity & (
            (remaining_capacity < 1.0) | (utilization_remaining < 0.01)
        )
        # Make nearly exhausted links unattractive in the next shortest-path search.
        costs[saturated_edges] = LEGACY_SATURATED_EDGE_PENALTY
        if "cost" in links.column_names:
            working_links = links.set_column(
                links.column_names.index("cost"),
                "cost",
                pa.array(costs, type=pa.float64()),
            )
        else:
            working_links = links.append_column(
                "cost", pa.array(costs, type=pa.float64())
            )
        if "flow" in working_links.column_names:
            working_links = working_links.set_column(
                working_links.column_names.index("flow"),
                "flow",
                pa.array(cumulative_flows, type=pa.float64()),
            )
        else:
            working_links = working_links.append_column(
                "flow", pa.array(cumulative_flows, type=pa.float64())
            )

        # First find the best routes without capacity limits; scale them below.
        allocation = core.allocate(
            working_links,
            batch,
            capacity_constrained=False,
            directed=directed,
        )
        iterations_run += 1

        allocated = allocation["od_flows"]
        for origin, destination in zip(
            allocation["unassigned_od"]["origin_id"].to_pylist(),
            allocation["unassigned_od"]["destination_id"].to_pylist(),
        ):
            unreachable_pairs.add((origin, destination))

        proposed_by_edge = np.zeros(links.num_rows, dtype="float64")
        # The ratio denominator is total proposed flow across all OD pairs per edge.
        for edge_path, flow in zip(
            allocated["edge_path"].to_pylist(), allocated["flow"].to_pylist()
        ):
            for edge_id in edge_path:
                proposed_by_edge[edge_positions[edge_id]] += float(flow)

        accepted_by_edge = np.zeros(links.num_rows, dtype="float64")
        accepted_flows = []
        accepted_rows = []
        for row_index, (origin, destination, flow, edge_path, cost) in enumerate(
            zip(
                allocated["origin_id"].to_pylist(),
                allocated["destination_id"].to_pylist(),
                allocated["flow"].to_pylist(),
                allocated["edge_path"].to_pylist(),
                allocated["cost"].to_pylist(),
            )
        ):
            flow = float(flow)
            adjustment_ratio = 1.0
            if capacity_constrained:
                # An OD path is limited by its most constrained edge.
                for edge_id in edge_path:
                    edge_index = edge_positions[edge_id]
                    proposed = proposed_by_edge[edge_index]
                    if proposed > LEGACY_CAPACITY_EPSILON:
                        edge_ratio = min(remaining_capacity[edge_index] / proposed, 1.0)
                        adjustment_ratio = min(adjustment_ratio, max(edge_ratio, 0.0))

            accepted_flow = flow * adjustment_ratio
            if accepted_flow <= LEGACY_CAPACITY_EPSILON:
                continue

            pair = (origin, destination)
            assigned_by_pair[pair] = assigned_by_pair.get(pair, 0.0) + accepted_flow
            remaining_by_pair[pair] = max(remaining_by_pair[pair] - accepted_flow, 0.0)
            totals = skim_totals.setdefault(pair, [0.0, 0.0])
            totals[0] += accepted_flow
            totals[1] += accepted_flow * float(cost)
            accepted_flows.append(accepted_flow)
            accepted_rows.append(row_index)
            for edge_id in edge_path:
                accepted_by_edge[edge_positions[edge_id]] += accepted_flow

        # Keep path-level output consistent with the capacity-adjusted flow.
        if include_paths and accepted_rows:
            accepted_paths = allocated.take(pa.array(accepted_rows, type=pa.int64()))
            accepted_paths = accepted_paths.set_column(
                accepted_paths.column_names.index("flow"),
                "flow",
                pa.array(accepted_flows, type=pa.float64()),
            )
            path_tables.append(accepted_paths.select(list(PATH_COLUMNS)))

        if accepted_by_edge.sum() <= LEGACY_CAPACITY_EPSILON:
            break

        # Recompute costs and routes next round using these accepted loads.
        cumulative_flows += accepted_by_edge

    # Any demand left after the iteration limit or capacity allocation is unassigned.
    unassigned_pairs = [
        (pair, flow)
        for pair, flow in remaining_by_pair.items()
        if flow > LEGACY_CAPACITY_EPSILON
    ]
    unassigned = pa.table(
        {
            "origin_id": pa.array(
                [pair[0] for pair, _ in unassigned_pairs],
                type=od["origin_id"].type,
            ),
            "destination_id": pa.array(
                [pair[1] for pair, _ in unassigned_pairs],
                type=od["destination_id"].type,
            ),
            "value": pa.array(
                [flow for _, flow in unassigned_pairs], type=pa.float64()
            ),
        }
    )
    unassigned = coerce_integral(unassigned, columns=("value",))

    skim_pairs = list(skim_totals)
    skims = pa.table(
        {
            "origin_id": pa.array(
                [pair[0] for pair in skim_pairs], type=od["origin_id"].type
            ),
            "destination_id": pa.array(
                [pair[1] for pair in skim_pairs],
                type=od["destination_id"].type,
            ),
            "cost": pa.array(
                [skim_totals[pair][1] / skim_totals[pair][0] for pair in skim_pairs],
                type=pa.float64(),
            ),
        }
    )
    skims = coerce_integral(skims, columns=("cost",))

    if path_tables:
        paths = coerce_integral(pa.concat_tables(path_tables))
    else:
        paths = pa.table(
            {
                "origin_id": pa.array([], type=od["origin_id"].type),
                "destination_id": pa.array([], type=od["destination_id"].type),
                "flow": pa.array([], type=pa.float64()),
                "edge_path": pa.array([], type=pa.list_(links["edge_id"].type)),
                "cost": pa.array([], type=pa.float64()),
            }
        )

    final_network_flows = pa.table(
        {
            "edge_id": links["edge_id"],
            "flow": pa.array(cumulative_flows, type=pa.float64()),
        }
    )
    return {
        "link_flows": link_flows_table(links, final_network_flows),
        "skims": skims,
        "unassigned": unassigned,
        "paths": paths if include_paths else None,
        "iterations": iterations_run,
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


def link_flows_table(links: pa.Table, network_flows: pa.Table) -> pa.Table:
    """Attach per-link ``flow`` to a link table, preserving link order."""
    positions = pc.index_in(links["edge_id"], network_flows["edge_id"].combine_chunks())
    flows = pc.fill_null(pc.take(network_flows["flow"], positions), 0)
    if "flow" in links.column_names:
        links = links.drop_columns(["flow"])
    return coerce_integral(links.append_column("flow", flows))


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
