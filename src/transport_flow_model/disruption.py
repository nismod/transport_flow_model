"""Disruption scenarios: ``disrupt(network, scenarios, ...)``.

A :class:`Scenario` is a sparse set of link attribute deltas on fixed
topology (see workplan ws4-01). The v0 engine supports link removal
(``cost`` set to infinity); other attribute deltas are reserved for later
engines and raise :class:`NotImplementedError`.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.compute as pc

from transport_flow_model import core
from transport_flow_model.assignment import (
    AssignmentResult,
    PATH_COLUMNS,
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
    factor (exactly one of the two)."""

    link_id: Any
    attribute: str = "cost"
    value: float | None = None
    scale: float | None = None

    def __post_init__(self):
        if (self.value is None) == (self.scale is None):
            raise ValueError("LinkDelta requires exactly one of value or scale")

    @property
    def is_removal(self) -> bool:
        """True if this delta removes the link (infinite cost)."""
        return (
            self.attribute == "cost"
            and self.value is not None
            and math.isinf(self.value)
        )


@dataclass(frozen=True)
class Scenario:
    """A disruption scenario: sparse link attribute deltas plus metadata
    (e.g. hazard id, return period, probability)."""

    id: str
    deltas: tuple[LinkDelta, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

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

    @property
    def removed_links(self) -> tuple[Any, ...]:
        return tuple(delta.link_id for delta in self.deltas if delta.is_removal)


@dataclass(frozen=True)
class ScenarioResult:
    """Result of one scenario: rerouted path detail, isolated demand,
    resulting link flows, and per-OD losses (all :class:`pyarrow.Table`)."""

    scenario: Scenario
    rerouted: pa.Table
    isolated: pa.Table
    link_flows: pa.Table
    losses: pa.Table


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
        """Per-scenario aggregate: affected, rerouted and isolated flow,
        and total rerouting loss (from per-OD initial vs disrupted cost)."""
        rows = {
            "scenario_id": [],
            "rerouted_flow": [],
            "isolated_flow": [],
            "rerouting_loss": [],
        }
        for result in self.results:
            rows["scenario_id"].append(result.scenario.id)
            rows["rerouted_flow"].append(_total(result.rerouted, "flow"))
            rows["isolated_flow"].append(_total(result.isolated, "value"))
            rows["rerouting_loss"].append(_total(result.losses, "rerouting_loss"))
        return pa.table(rows)


def disrupt(
    network: Network,
    scenarios: Sequence[Scenario],
    demand: Demand | None = None,
    *,
    base: AssignmentResult | None = None,
    method: str = "sequential",
    skip_unaffected: bool = True,
    **options: Any,
) -> DisruptionResults:
    """Evaluate disruption scenarios against a baseline assignment.

    The baseline is either given (``base``, which must include ``paths``)
    or computed by running ``assign(network, demand, method,
    include_paths=True, **options)``. For each scenario, flows whose
    baseline paths use a removed link are rerouted on the disrupted
    network; demand that cannot be rerouted is isolated.

    With ``skip_unaffected`` (default), scenarios whose removed links carry
    no baseline flow are skipped and listed in ``DisruptionResults.skipped``.
    """
    start = time.perf_counter()
    if base is None:
        if demand is None:
            raise ValueError("disrupt requires either a base result or demand")
        base = assign(network, demand, method, include_paths=True, **options)
    if base.paths is None:
        raise ValueError("base assignment must include paths (include_paths=True)")

    links = _links_with_base_flow(network, base)
    results = []
    skipped = []
    for scenario in scenarios:
        removed = _removed_links(scenario)
        if skip_unaffected and not _affects_flow(links, removed):
            skipped.append(scenario)
            continue
        results.append(_evaluate(links, base.paths, scenario, removed, options))
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
    options: Mapping[str, Any],
) -> ScenarioResult:
    result = core.disrupt(
        links,
        paths,
        removed,
        capacity_constrained=bool(options.get("capacity_constrained", True)),
        directed=bool(options.get("directed", True)),
    )
    rerouted = coerce_integral(result["rerouted_flows"].select(list(PATH_COLUMNS)))
    isolated = coerce_integral(
        result["isolated_od"].select(["origin_id", "destination_id", "flow"])
    ).rename_columns(["origin_id", "destination_id", "value"])
    losses = coerce_integral(
        result["losses"].select(list(LOSS_COLUMNS)),
        columns=("flow", "initial_cost", "disrupted_cost", "rerouting_loss"),
    )
    return ScenarioResult(
        scenario=scenario,
        rerouted=rerouted,
        isolated=isolated,
        link_flows=link_flows_table(links, result["network_flows"]),
        losses=losses,
    )


def _removed_links(scenario: Scenario) -> list[Any]:
    unsupported = [delta for delta in scenario.deltas if not delta.is_removal]
    if unsupported:
        raise NotImplementedError(
            f"Scenario {scenario.id!r} has non-removal link deltas "
            f"(e.g. {unsupported[0]!r}); the v0 engine only supports link "
            "removal (workplan ws4-01)"
        )
    return list(scenario.removed_links)


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


def _total(table: pa.Table, column: str) -> float:
    if table.num_rows == 0:
        return 0.0
    return float(pc.sum(pc.cast(table[column], pa.float64())).as_py() or 0.0)
