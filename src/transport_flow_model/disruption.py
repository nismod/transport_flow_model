"""Disruption scenarios: ``disrupt(network, scenarios, ...)``.

A :class:`Scenario` is a sparse set of link attribute deltas on fixed
topology (see workplan ws4-01). The v0 engine supports link removal
(``cost`` set to infinity) and finite link cost changes (an absolute
``value`` or a ``scale`` factor on ``cost``), alone or combined. Deltas on
other attributes (e.g. ``capacity``) are reserved for later engines and
raise :class:`NotImplementedError`.

How a scenario is evaluated against the baseline paths:

- flows whose baseline path uses a removed link are rerouted on the network
  without the removed links, using the scenario's (changed) link costs;
  demand that cannot be rerouted is isolated;
- flows whose path uses a link whose cost changed (and no removed link) keep
  their path by default and are reported in ``ScenarioResult.repriced``
  with their new path cost; with ``disrupt(..., reroute_changed=True)``
  they are rerouted as well;
- all other flows keep their baseline paths, even if a link elsewhere
  became cheaper.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from transport_flow_model import core
from transport_flow_model.assignment import (
    PATH_COLUMNS,
    AssignmentResult,
    assign,
    coerce_integral,
    link_flows_table,
)
from transport_flow_model.demand import Demand
from transport_flow_model.network import Network

#: Columns of per-OD loss tables (``ScenarioResult.losses``).
LOSS_COLUMNS = (
    "origin_id",
    "destination_id",
    "flow",
    "initial_cost",
    "disrupted_cost",
    "rerouting_loss",
)


@dataclass(frozen=True)
class LinkDelta:
    """A change to one link attribute: an absolute ``value`` or a ``scale``
    factor (exactly one of the two).

    For ``attribute="cost"``, ``value=math.inf`` removes the link; any other
    ``value`` must be finite and non-negative and replaces the link cost; a
    ``scale`` must be finite and non-negative and multiplies the baseline
    link cost.
    """

    link_id: Any
    attribute: str = "cost"
    value: float | None = None
    scale: float | None = None

    def __post_init__(self):
        if (self.value is None) == (self.scale is None):
            raise ValueError("LinkDelta requires exactly one of value or scale")
        if self.attribute != "cost":
            return
        if self.value is not None:
            value = float(self.value)
            if math.isnan(value) or value < 0:
                raise ValueError(
                    f"LinkDelta cost value for link {self.link_id!r} must be "
                    f"non-negative (or math.inf to remove the link), got {self.value!r}"
                )
        else:
            scale = float(self.scale)
            if not math.isfinite(scale) or scale < 0:
                raise ValueError(
                    f"LinkDelta cost scale for link {self.link_id!r} must be "
                    f"finite and non-negative, got {self.scale!r}"
                )

    @property
    def is_removal(self) -> bool:
        """True if this delta removes the link (infinite cost)."""
        return (
            self.attribute == "cost"
            and self.value is not None
            and math.isinf(self.value)
        )

    @property
    def is_cost_change(self) -> bool:
        """True if this delta changes the link cost to a finite value."""
        return self.attribute == "cost" and not self.is_removal


@dataclass(frozen=True)
class Scenario:
    """A disruption scenario: sparse link attribute deltas plus metadata
    (e.g. hazard id, return period, probability)."""

    id: str
    deltas: tuple[LinkDelta, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        removed = set(self.removed_links)
        changed = set()
        for delta in self.deltas:
            if not delta.is_cost_change:
                continue
            if delta.link_id in removed:
                raise ValueError(
                    f"Scenario {self.id!r} both removes link {delta.link_id!r} "
                    "and changes its cost"
                )
            if delta.link_id in changed:
                raise ValueError(
                    f"Scenario {self.id!r} has more than one cost change for "
                    f"link {delta.link_id!r}"
                )
            changed.add(delta.link_id)

    @classmethod
    def remove_links(
        cls, id: str, link_ids: Iterable[Any], **metadata: Any
    ) -> Scenario:
        """A scenario that removes the given links from the network."""
        return cls(
            id=id,
            deltas=tuple(
                LinkDelta(link_id, "cost", value=math.inf) for link_id in link_ids
            ),
            metadata=metadata,
        )

    @classmethod
    def from_changes(
        cls,
        id: str,
        *,
        remove: Iterable[Any] = (),
        cost: Mapping[Any, float] | None = None,
        cost_scale: Mapping[Any, float] | None = None,
        **metadata: Any,
    ) -> Scenario:
        """A scenario that removes links and changes the cost of others.

        ``remove`` lists links to remove, ``cost`` maps link ids to a new
        (absolute) cost and ``cost_scale`` maps link ids to a factor on the
        baseline cost. A link may appear in only one of the three. Further
        keyword arguments are kept as scenario metadata.
        """
        deltas = [LinkDelta(link_id, "cost", value=math.inf) for link_id in remove]
        deltas += [
            LinkDelta(link_id, "cost", value=float(value))
            for link_id, value in (cost or {}).items()
        ]
        deltas += [
            LinkDelta(link_id, "cost", scale=float(scale))
            for link_id, scale in (cost_scale or {}).items()
        ]
        return cls(id=id, deltas=tuple(deltas), metadata=metadata)

    @property
    def removed_links(self) -> tuple[Any, ...]:
        """Links removed by this scenario."""
        return tuple(delta.link_id for delta in self.deltas if delta.is_removal)

    @property
    def changed_links(self) -> tuple[Any, ...]:
        """Links whose cost this scenario changes (but does not remove)."""
        return tuple(delta.link_id for delta in self.deltas if delta.is_cost_change)


@dataclass(frozen=True)
class ScenarioResult:
    """Result of one scenario (all tables are :class:`pyarrow.Table`).

    - ``rerouted``: new paths (``origin_id``, ``destination_id``, ``flow``,
      ``edge_path``, ``cost``) for flows whose baseline path used a removed
      link (and, with ``reroute_changed=True``, a link whose cost changed).
      ``cost`` uses the scenario's link costs.
    - ``isolated``: demand that could not be rerouted (``origin_id``,
      ``destination_id``, ``value``).
    - ``link_flows``: the scenario link table (with changed costs) and the
      resulting ``flow``: kept and repriced baseline flows plus rerouted
      flows.
    - ``losses``: per-OD ``flow``, ``initial_cost`` (summed baseline path
      costs), ``disrupted_cost`` (summed scenario path costs) and
      ``rerouting_loss`` (their difference), over rerouted and repriced
      flows.
    - ``repriced``: baseline paths kept although they use a link whose cost
      changed (columns as ``rerouted``), with ``cost`` the baseline path cost
      plus the cost changes of the changed links on the path. Empty when
      ``reroute_changed=True`` or the scenario only removes links.
    """

    scenario: Scenario
    rerouted: pa.Table
    isolated: pa.Table
    link_flows: pa.Table
    losses: pa.Table
    repriced: pa.Table | None = None


@dataclass(frozen=True)
class DisruptionResults:
    """Results across scenarios.

    ``results`` holds one :class:`ScenarioResult` per evaluated scenario;
    ``skipped`` names scenarios that were not evaluated because no baseline
    flow used their disrupted links.
    """

    base: AssignmentResult
    results: tuple[ScenarioResult, ...]
    skipped: tuple[Scenario, ...] = ()
    wall_time_s: float | None = None

    def __iter__(self):
        return iter(self.results)

    def summary(self) -> pa.Table:
        """Per-scenario aggregate: rerouted and isolated flow, total
        rerouting loss (from per-OD initial vs disrupted cost, including
        repriced flows) and repriced flow (kept on paths whose cost
        changed)."""
        rows = {
            "scenario_id": [],
            "rerouted_flow": [],
            "isolated_flow": [],
            "rerouting_loss": [],
            "repriced_flow": [],
        }
        for result in self.results:
            rows["scenario_id"].append(result.scenario.id)
            rows["rerouted_flow"].append(_total(result.rerouted, "flow"))
            rows["isolated_flow"].append(_total(result.isolated, "value"))
            rows["rerouting_loss"].append(_total(result.losses, "rerouting_loss"))
            rows["repriced_flow"].append(_total(result.repriced, "flow"))
        return pa.table(rows)


def disrupt(
    network: Network,
    scenarios: Sequence[Scenario],
    demand: Demand | None = None,
    *,
    base: AssignmentResult | None = None,
    method: str = "sequential",
    skip_unaffected: bool = True,
    reroute_changed: bool = False,
    **options: Any,
) -> DisruptionResults:
    """Evaluate disruption scenarios against a baseline assignment.

    The baseline is either given (``base``, which must include ``paths``)
    or computed by running ``assign(network, demand, method,
    include_paths=True, **options)``. For each scenario, flows whose
    baseline paths use a removed link are rerouted on the disrupted
    network, with the scenario's changed link costs; demand that cannot be
    rerouted is isolated.

    Flows whose baseline path uses a link whose cost changed (and no
    removed link) keep their path and are reported in
    ``ScenarioResult.repriced`` and ``losses`` with the new path cost. With
    ``reroute_changed=True`` they are rerouted instead, together with the
    flows on removed links (links with changed costs stay available).
    Initial costs in ``losses`` are always the baseline path costs.

    Link ids in a scenario that are not in the network are ignored.

    With ``skip_unaffected`` (default), scenarios whose removed or changed
    links carry no baseline flow are skipped and listed in
    ``DisruptionResults.skipped``.
    """
    start = time.perf_counter()
    if base is None:
        if demand is None:
            raise ValueError("disrupt requires either a base result or demand")
        base = assign(network, demand, method, include_paths=True, **options)
    if base.paths is None:
        raise ValueError("base assignment must include paths (include_paths=True)")

    links = _links_with_base_flow(network, base)
    # Resolve every scenario first so that invalid scenarios fail early.
    changes = [(scenario, _resolve_changes(scenario, links)) for scenario in scenarios]
    results = []
    skipped = []
    for scenario, (removed, new_costs) in changes:
        touched = removed + _edge_ids_at(links, new_costs)
        if skip_unaffected and not _affects_flow(links, touched):
            skipped.append(scenario)
            continue
        results.append(
            _evaluate(
                links,
                base.paths,
                scenario,
                removed,
                new_costs,
                reroute_changed,
                options,
            )
        )
    return DisruptionResults(
        base=base,
        results=tuple(results),
        skipped=tuple(skipped),
        wall_time_s=time.perf_counter() - start,
    )


def _evaluate(
    links: pa.Table,
    paths: pa.Table,
    scenario: Scenario,
    removed: list[Any],
    new_costs: Mapping[int, float],
    reroute_changed: bool,
    options: Mapping[str, Any],
) -> ScenarioResult:
    scenario_links = _with_costs(links, new_costs)
    changed = _edge_ids_at(links, new_costs)
    result = core.disrupt(
        scenario_links,
        paths,
        removed,
        capacity_constrained=bool(options.get("capacity_constrained", True)),
        directed=bool(options.get("directed", True)),
        reroute_edges=changed if reroute_changed and changed else None,
    )
    rerouted = coerce_integral(result["rerouted_flows"].select(list(PATH_COLUMNS)))
    isolated = coerce_integral(
        result["isolated_od"].select(["origin_id", "destination_id", "flow"])
    ).rename_columns(["origin_id", "destination_id", "value"])
    losses = result["losses"].select(list(LOSS_COLUMNS))
    if new_costs and not reroute_changed:
        repriced, initial_cost = _repriced_paths(links, paths, removed, new_costs)
    else:
        repriced, initial_cost = paths.select(list(PATH_COLUMNS)).slice(0, 0), None
    if repriced.num_rows:
        losses = _add_repriced_losses(losses, repriced, initial_cost)
    losses = coerce_integral(
        losses,
        columns=("flow", "initial_cost", "disrupted_cost", "rerouting_loss"),
    )
    return ScenarioResult(
        scenario=scenario,
        rerouted=rerouted,
        isolated=isolated,
        link_flows=link_flows_table(scenario_links, result["network_flows"]),
        losses=losses,
        repriced=coerce_integral(repriced),
    )


def _resolve_changes(
    scenario: Scenario, links: pa.Table
) -> tuple[list[Any], dict[int, float]]:
    """Removed link ids and new costs by link position in ``links``.

    Cost changes on link ids that are not in ``links`` are ignored, as are
    unknown removed links (by the core).
    """
    unsupported = [delta for delta in scenario.deltas if delta.attribute != "cost"]
    if unsupported:
        raise NotImplementedError(
            f"Scenario {scenario.id!r} has non-removal link deltas on "
            f"attributes other than cost (e.g. {unsupported[0]!r}); the v0 "
            "engine only supports link removal and cost changes (workplan ws4-01)"
        )
    removed = list(scenario.removed_links)
    deltas = [delta for delta in scenario.deltas if delta.is_cost_change]
    if not deltas:
        return removed, {}
    edge_ids = links["edge_id"]
    try:
        ids = pa.array([delta.link_id for delta in deltas], type=edge_ids.type)
    except (pa.ArrowInvalid, pa.ArrowTypeError, TypeError, ValueError) as error:
        raise ValueError(
            f"Scenario {scenario.id!r}: link ids must match the network "
            f"edge_id type ({edge_ids.type})"
        ) from error
    positions = pc.index_in(ids, edge_ids).to_pylist()
    base_costs = _base_costs(links)
    new_costs: dict[int, float] = {}
    for delta, position in zip(deltas, positions):
        if position is None:
            continue
        if delta.value is not None:
            new_costs[position] = float(delta.value)
        else:
            new_costs[position] = float(base_costs[position]) * float(delta.scale)
    return removed, new_costs


def _base_costs(links: pa.Table) -> np.ndarray:
    """Baseline link costs (the core's default of 1 if there is no column)."""
    if "cost" not in links.column_names:
        return np.ones(links.num_rows)
    return np.asarray(
        pc.cast(links["cost"], pa.float64()).to_numpy(zero_copy_only=False),
        dtype=float,
    )


def _with_costs(links: pa.Table, new_costs: Mapping[int, float]) -> pa.Table:
    """``links`` with the cost of the links at the given positions replaced."""
    if not new_costs:
        return links
    costs = _base_costs(links).copy()
    for position, cost in new_costs.items():
        costs[position] = cost
    column = pa.array(costs, type=pa.float64())
    if "cost" in links.column_names:
        return links.set_column(links.column_names.index("cost"), "cost", column)
    return links.append_column("cost", column)


def _edge_ids_at(links: pa.Table, positions: Iterable[int]) -> list[Any]:
    return links["edge_id"].take(pa.array(sorted(positions), pa.int64())).to_pylist()


def _repriced_paths(
    links: pa.Table,
    paths: pa.Table,
    removed: list[Any],
    new_costs: Mapping[int, float],
) -> tuple[pa.Table, np.ndarray]:
    """Baseline paths that use a changed link but no removed link, with
    ``cost`` the baseline cost plus the cost changes along the path; also
    returns their baseline costs."""
    n_paths = paths.num_rows
    if n_paths == 0:
        return paths.select(list(PATH_COLUMNS)), np.zeros(0)
    edge_paths = paths["edge_path"].combine_chunks()
    path_links = pc.list_flatten(edge_paths)
    parents = np.asarray(pc.list_parent_indices(edge_paths).to_numpy(), dtype=np.int64)
    positions = pc.index_in(path_links, links["edge_id"])
    positions = np.asarray(
        pc.fill_null(positions, -1).to_numpy(zero_copy_only=False), dtype=np.int64
    )
    known = positions >= 0
    parents, positions = parents[known], positions[known]

    base_costs = _base_costs(links)
    cost_change = np.zeros(links.num_rows)
    is_changed = np.zeros(links.num_rows)
    for position, cost in new_costs.items():
        cost_change[position] = cost - base_costs[position]
        is_changed[position] = 1.0
    is_removed = np.zeros(links.num_rows)
    if removed:
        try:
            removed_ids = pa.array(removed, type=links["edge_id"].type)
        except (pa.ArrowInvalid, pa.ArrowTypeError, TypeError, ValueError):
            removed_ids = None  # the core has already rejected mismatched ids
        if removed_ids is not None:
            mask = pc.is_in(links["edge_id"], value_set=removed_ids)
            is_removed = np.asarray(mask.to_numpy(zero_copy_only=False), dtype=float)

    def per_path(weights: np.ndarray) -> np.ndarray:
        return np.bincount(parents, weights=weights[positions], minlength=n_paths)

    keep = (per_path(is_changed) > 0) & ~(per_path(is_removed) > 0)
    delta = per_path(cost_change)[keep]
    kept = paths.filter(pa.array(keep)).select(list(PATH_COLUMNS))
    initial_cost = np.asarray(
        pc.cast(kept["cost"], pa.float64()).to_numpy(zero_copy_only=False), dtype=float
    )
    kept = kept.set_column(
        kept.column_names.index("cost"),
        "cost",
        pa.array(initial_cost + delta, type=pa.float64()),
    )
    return kept, initial_cost


def _add_repriced_losses(
    losses: pa.Table, repriced: pa.Table, initial_cost: np.ndarray
) -> pa.Table:
    """Add repriced paths to per-OD losses, summing costs per OD pair as for
    rerouted flows. OD pairs keep the order of ``losses``, then of
    ``repriced``."""
    n_losses = losses.num_rows
    origin_type = losses.schema.field("origin_id").type
    destination_type = losses.schema.field("destination_id").type
    rows = pa.table(
        {
            "origin_id": pa.concat_arrays(
                [
                    losses["origin_id"].combine_chunks().cast(origin_type),
                    repriced["origin_id"].combine_chunks().cast(origin_type),
                ]
            ),
            "destination_id": pa.concat_arrays(
                [
                    losses["destination_id"].combine_chunks().cast(destination_type),
                    repriced["destination_id"].combine_chunks().cast(destination_type),
                ]
            ),
            "flow": _float_concat(losses["flow"], repriced["flow"]),
            "initial_cost": _float_concat(
                losses["initial_cost"], pa.array(initial_cost)
            ),
            "disrupted_cost": _float_concat(losses["disrupted_cost"], repriced["cost"]),
            "order": pa.array(np.arange(n_losses + repriced.num_rows)),
        }
    )
    grouped = (
        rows.group_by(["origin_id", "destination_id"], use_threads=False)
        .aggregate(
            [
                ("flow", "sum"),
                ("initial_cost", "sum"),
                ("disrupted_cost", "sum"),
                ("order", "min"),
            ]
        )
        .sort_by("order_min")
    )
    initial = grouped["initial_cost_sum"]
    disrupted = grouped["disrupted_cost_sum"]
    return pa.table(
        {
            "origin_id": grouped["origin_id"],
            "destination_id": grouped["destination_id"],
            "flow": grouped["flow_sum"],
            "initial_cost": initial,
            "disrupted_cost": disrupted,
            "rerouting_loss": pc.subtract(disrupted, initial),
        }
    )


def _float_concat(*columns: pa.Array | pa.ChunkedArray) -> pa.Array:
    arrays = []
    for column in columns:
        if isinstance(column, pa.ChunkedArray):
            column = column.combine_chunks()
        arrays.append(pc.cast(column, pa.float64()))
    return pa.concat_arrays(arrays)


def _links_with_base_flow(network: Network, base: AssignmentResult) -> pa.Table:
    """Network link table with the baseline ``flow`` column attached."""
    links = network.to_table()
    if "flow" in links.column_names:
        return links
    return link_flows_table(links, base.link_flows.select(["edge_id", "flow"]))


def _affects_flow(links: pa.Table, removed: list[Any]) -> bool:
    if not removed:
        return False
    mask = pc.is_in(
        links["edge_id"], value_set=pa.array(removed, type=links["edge_id"].type)
    )
    affected = pc.sum(pc.filter(links["flow"], mask)).as_py()
    return bool(affected and affected > 0)


def _total(table: pa.Table | None, column: str) -> float:
    if table is None or table.num_rows == 0:
        return 0.0
    return float(pc.sum(pc.cast(table[column], pa.float64())).as_py() or 0.0)
