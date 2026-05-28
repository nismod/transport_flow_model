"""Core model classes"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from heapq import heappop, heappush
from math import inf, isfinite
from pathlib import Path

import pandas as pd


FLOW_COLUMNS = ("origin_id", "destination_id", "flow")
OD_FLOW_COLUMNS = ("origin_id", "destination_id", "flow", "edge_path", "cost")
LOSS_COLUMNS = (
    "origin_id",
    "destination_id",
    "flow",
    "initial_cost",
    "disrupted_cost",
    "rerouting_loss",
)
CAPACITY_EPSILON = 1.0e-9


def _coerce_integral_numeric_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Use integer dtypes for integral numeric results to keep comparisons stable."""
    numeric_columns = {
        "flow",
        "cost",
        "initial_cost",
        "disrupted_cost",
        "rerouting_loss",
    }
    for column in numeric_columns.intersection(data.columns):
        if data.empty:
            continue
        values = pd.to_numeric(data[column], errors="coerce")
        if values.isna().any():
            continue
        if all(isfinite(v) and abs(v - round(v)) <= CAPACITY_EPSILON for v in values):
            data[column] = values.round().astype("int64")
        else:
            data[column] = values
    return data


def _frame_from_records(records: list[dict], columns: tuple[str, ...]) -> pd.DataFrame:
    data = pd.DataFrame.from_records(records, columns=list(columns))
    return _coerce_integral_numeric_columns(data)


def _dataframe_delegate(instance, name):
    data = instance.__dict__.get("_data")
    if data is not None and hasattr(data, name):
        return getattr(data, name)
    raise AttributeError(
        f"{instance.__class__.__name__!r} object has no attribute {name!r}"
    )


@dataclass
class AllocationResult:
    od_flows: ODFlows
    network_flows: NetworkFlows
    unassigned_od: OD


@dataclass
class DisruptionResult:
    rerouted_flows: ODFlows
    network_flows: NetworkFlows
    isolated_od: OD
    losses: OD


class OD:
    """Origin-destination flows.

    - sparse matrix representation
    - output of methods to estimate OD from spatial interactions
    - input to Network flow allocation
    """

    REQUIRED_COLUMNS = ("origin_id", "destination_id", "flow")

    def __init__(self, data: pd.DataFrame):
        missing = [c for c in self.REQUIRED_COLUMNS if c not in data.columns]
        if missing:
            raise ValueError(
                f"Missing required columns for {self.__class__.__name__}: {missing}"
            )
        self._data = data.copy()

    def __getattr__(self, name):
        return _dataframe_delegate(self, name)

    @property
    def data(self) -> pd.DataFrame:
        """Return a defensive copy of the normalized tabular data."""
        return self._data.copy()

    def to_dataframe(self, copy=True) -> pd.DataFrame:
        """Return the normalized tabular data."""
        if copy:
            return self._data.copy()
        else:
            return self._data

    @classmethod
    def from_csv(cls, path: str | Path, column_map: dict[str, str]) -> OD:
        """Load a CSV and rename source columns to this class' schema."""
        target_columns = set(column_map.values())
        to_rename = set(column_map.keys())
        missing_columns = [
            col for col in cls.REQUIRED_COLUMNS if col not in target_columns
        ]
        if missing_columns:
            raise ValueError(
                "column_map must include mappings for required columns "
                f"{missing_columns}"
            )

        try:
            data = pd.read_csv(path, usecols=to_rename)
        except ValueError as e:
            msg = e.args[0].replace(
                "Usecols do not match columns, columns expected but not found: ", ""
            )
            raise ValueError(f"Missing expected columns: {msg}") from e
        data = data.rename(columns=column_map)

        return cls(data)

    @classmethod
    def losses_from_flows(cls, initial: ODFlows, disrupted: ODFlows) -> OD:
        """Aggregate per-OD rerouting losses from initial and disrupted paths."""
        initial_data = initial.to_dataframe()
        disrupted_data = disrupted.to_dataframe()

        if disrupted_data.empty:
            return cls(pd.DataFrame(columns=list(LOSS_COLUMNS)))

        initial_costs = (
            initial_data.groupby(["origin_id", "destination_id"], as_index=False)[
                "cost"
            ]
            .sum()
            .rename(columns={"cost": "initial_cost"})
        )
        disrupted_costs = (
            disrupted_data.groupby(["origin_id", "destination_id"], as_index=False)
            .agg({"flow": "sum", "cost": "sum"})
            .rename(columns={"cost": "disrupted_cost"})
        )
        losses = disrupted_costs.merge(
            initial_costs,
            on=["origin_id", "destination_id"],
            how="left",
        )
        losses["initial_cost"] = losses["initial_cost"].fillna(0)
        losses["rerouting_loss"] = (
            losses["disrupted_cost"] - losses["initial_cost"]
        )
        losses = losses.loc[:, list(LOSS_COLUMNS)]
        return cls(_coerce_integral_numeric_columns(losses))


class Network:
    """Base network model class.

    - graph representation
    - works with methods to allocate OD flows to the network, outputting ODFlows
    """

    REQUIRED_COLUMNS = ("edge_from", "edge_to", "edge_id")
    OPTIONAL_COLUMNS = ("capacity", "cost", "flow")

    def __init__(self, data: pd.DataFrame):
        missing = [c for c in self.REQUIRED_COLUMNS if c not in data.columns]
        if missing:
            raise ValueError(
                f"Missing required columns for {self.__class__.__name__}: {missing}"
            )
        self._data = data.copy()

    def __getattr__(self, name):
        return _dataframe_delegate(self, name)

    def to_dataframe(self, copy=True) -> pd.DataFrame:
        """Return the normalized tabular data."""
        if copy:
            return self._data.copy()
        else:
            return self._data

    @classmethod
    def from_csv(cls, path: str | Path, column_map: dict[str, str]) -> Network:
        """Load a CSV and rename source columns to this class' schema."""
        target_columns = set(column_map.values())
        to_rename = set(column_map.keys())
        missing_columns = [
            col for col in cls.REQUIRED_COLUMNS if col not in target_columns
        ]
        if missing_columns:
            raise ValueError(
                "column_map must include mappings for required columns "
                f"{missing_columns}"
            )

        data = pd.read_csv(path, usecols=to_rename)
        data = data.rename(columns=column_map)

        return cls(data)

    def allocate(
        self,
        od: OD,
        *,
        capacity_constrained: bool = False,
        directed: bool = True,
    ) -> AllocationResult:
        """Allocate OD flows to least-cost paths on this network."""
        if capacity_constrained:
            return self._allocate_capacity_constrained(od, directed=directed)
        return self._allocate_unconstrained(od, directed=directed)

    def disrupt(
        self,
        od_flows: ODFlows,
        failed_edges: list[str],
        *,
        capacity_constrained: bool = True,
        directed: bool = True,
    ) -> DisruptionResult:
        """Reroute flows whose existing paths include any failed edge."""
        failed_edge_set = set(failed_edges)
        flow_data = od_flows.to_dataframe()
        affected_mask = flow_data["edge_path"].map(
            lambda path: bool(failed_edge_set.intersection(path))
        )
        affected_flows = flow_data[affected_mask].copy()
        affected_od_flows = ODFlows(affected_flows)
        post_disruption_network_data = self._with_flows_removed_from_affected_paths(
            od_flows,
            affected_od_flows,
        )
        post_disruption_network_data.loc[
            post_disruption_network_data["edge_id"].isin(failed_edge_set),
            "flow",
        ] = 0

        if affected_flows.empty:
            rerouted_flows = ODFlows(pd.DataFrame(columns=list(OD_FLOW_COLUMNS)))
            isolated_od = OD(pd.DataFrame(columns=list(FLOW_COLUMNS)))
            losses = OD(pd.DataFrame(columns=list(LOSS_COLUMNS)))
            network_flows = NetworkFlows(post_disruption_network_data)
            return DisruptionResult(
                rerouted_flows=rerouted_flows,
                network_flows=network_flows,
                isolated_od=isolated_od,
                losses=losses,
            )

        if capacity_constrained:
            reroute_network_data = post_disruption_network_data
        else:
            reroute_network_data = self.to_dataframe()
        reroute_network_data = reroute_network_data[
            ~reroute_network_data["edge_id"].isin(failed_edge_set)
        ].reset_index(drop=True)

        reroute_network = Network(reroute_network_data)
        affected_od = OD(affected_flows.loc[:, list(FLOW_COLUMNS)])
        allocation = reroute_network.allocate(
            affected_od,
            capacity_constrained=capacity_constrained,
            directed=directed,
        )
        network_flows = NetworkFlows.from_network_and_od_flows(
            Network(post_disruption_network_data),
            allocation.od_flows,
        )
        losses = OD.losses_from_flows(affected_od_flows, allocation.od_flows)

        return DisruptionResult(
            rerouted_flows=allocation.od_flows,
            network_flows=network_flows,
            isolated_od=allocation.unassigned_od,
            losses=losses,
        )

    def _allocate_unconstrained(self, od: OD, *, directed: bool) -> AllocationResult:
        network_data = self.to_dataframe()
        flow_rows = []
        unassigned_rows = []

        for row in od.to_dataframe().itertuples(index=False):
            path, cost = _shortest_path(
                network_data,
                row.origin_id,
                row.destination_id,
                directed=directed,
            )
            if path is None:
                unassigned_rows.append(
                    {
                        "origin_id": row.origin_id,
                        "destination_id": row.destination_id,
                        "flow": row.flow,
                    }
                )
                continue
            flow_rows.append(
                {
                    "origin_id": row.origin_id,
                    "destination_id": row.destination_id,
                    "flow": row.flow,
                    "edge_path": path,
                    "cost": cost,
                }
            )

        od_flows = ODFlows(_frame_from_records(flow_rows, OD_FLOW_COLUMNS))
        unassigned_od = OD(_frame_from_records(unassigned_rows, FLOW_COLUMNS))
        network_flows = NetworkFlows.from_network_and_od_flows(self, od_flows)
        return AllocationResult(
            od_flows=od_flows,
            network_flows=network_flows,
            unassigned_od=unassigned_od,
        )

    def _allocate_capacity_constrained(
        self, od: OD, *, directed: bool
    ) -> AllocationResult:
        network_data = self.to_dataframe()
        residual_capacity = _initial_residual_capacity(network_data)
        pending = od.to_dataframe().loc[:, list(FLOW_COLUMNS)].copy()
        allocated_rows = []
        unassigned_rows = []

        while not pending.empty:
            route_rows = []
            next_pending = []

            for row in pending.itertuples(index=False):
                path, cost = _shortest_path(
                    network_data,
                    row.origin_id,
                    row.destination_id,
                    directed=directed,
                    residual_capacity=residual_capacity,
                )
                if path is None:
                    unassigned_rows.append(
                        {
                            "origin_id": row.origin_id,
                            "destination_id": row.destination_id,
                            "flow": row.flow,
                        }
                    )
                    continue
                route_rows.append(
                    {
                        "origin_id": row.origin_id,
                        "destination_id": row.destination_id,
                        "flow": row.flow,
                        "edge_path": path,
                        "cost": cost,
                    }
                )

            if not route_rows:
                break

            requested_by_edge = defaultdict(float)
            for route in route_rows:
                for edge_id in route["edge_path"]:
                    requested_by_edge[edge_id] += route["flow"]

            assigned_this_round = 0.0
            round_allocations = []
            for route in route_rows:
                requested_flow = route["flow"]
                assigned_flow = requested_flow
                for edge_id in route["edge_path"]:
                    requested_on_edge = requested_by_edge[edge_id]
                    available = residual_capacity.get(edge_id, 0.0)
                    if requested_on_edge > available + CAPACITY_EPSILON:
                        assigned_flow = min(
                            assigned_flow,
                            requested_flow * available / requested_on_edge,
                        )

                if assigned_flow > CAPACITY_EPSILON:
                    assigned_route = dict(route)
                    assigned_route["flow"] = assigned_flow
                    allocated_rows.append(assigned_route)
                    round_allocations.append(assigned_route)
                    assigned_this_round += assigned_flow

                residual_flow = requested_flow - assigned_flow
                if residual_flow > CAPACITY_EPSILON:
                    next_pending.append(
                        {
                            "origin_id": route["origin_id"],
                            "destination_id": route["destination_id"],
                            "flow": residual_flow,
                        }
                    )

            for route in round_allocations:
                for edge_id in route["edge_path"]:
                    residual_capacity[edge_id] -= route["flow"]
                    if residual_capacity[edge_id] < CAPACITY_EPSILON:
                        residual_capacity[edge_id] = 0.0

            if assigned_this_round <= CAPACITY_EPSILON:
                unassigned_rows.extend(next_pending)
                break

            pending = _frame_from_records(next_pending, FLOW_COLUMNS)

        od_flows = ODFlows(_frame_from_records(allocated_rows, OD_FLOW_COLUMNS))
        unassigned_od = OD(
            _aggregate_od(_frame_from_records(unassigned_rows, FLOW_COLUMNS))
        )
        network_flows = NetworkFlows.from_network_and_od_flows(self, od_flows)
        return AllocationResult(
            od_flows=od_flows,
            network_flows=network_flows,
            unassigned_od=unassigned_od,
        )

    def _with_flows_removed_from_affected_paths(
        self,
        od_flows: ODFlows,
        affected_flows: ODFlows,
    ) -> pd.DataFrame:
        network_data = self.to_dataframe()
        current_flows = _flow_by_edge(od_flows)
        affected_by_edge = _flow_by_edge(affected_flows)

        if "flow" in network_data.columns:
            base_flow = pd.to_numeric(network_data["flow"], errors="coerce").fillna(0)
        else:
            base_flow = network_data["edge_id"].map(current_flows).fillna(0)

        network_data["flow"] = [
            max(float(flow) - affected_by_edge.get(edge_id, 0.0), 0.0)
            for edge_id, flow in zip(network_data["edge_id"], base_flow, strict=False)
        ]
        return _coerce_integral_numeric_columns(network_data)


class ODFlows:
    """Origin-destination flow paths

    - full path representation of allocated flows for each OD
    - may include multiple paths for a single OD pair with different capacity
      allocation
    """

    REQUIRED_COLUMNS = ("origin_id", "destination_id", "flow", "edge_path")

    def __init__(self, data: pd.DataFrame):
        missing = [c for c in self.REQUIRED_COLUMNS if c not in data.columns]
        if missing:
            raise ValueError(
                f"Missing required columns for {self.__class__.__name__}: {missing}"
            )
        self._data = data.copy()

    def __getattr__(self, name):
        return _dataframe_delegate(self, name)

    def to_dataframe(self, copy=True) -> pd.DataFrame:
        """Return OD flow paths data."""
        if copy:
            return self._data.copy()
        else:
            return self._data

    def to_csv(self, path: str | Path, index=False):
        """Write OD flow paths to CSV."""
        self._data.to_csv(path, index=index)


class NetworkFlows:
    """Aggregate flows on network

    - could consider this as Network with calculated attributes
      as the result of an allocation
    - multiple attributes may include total allocation, speed, cost, ...
    - can be constructed from a Network and ODFlows
    """

    REQUIRED_COLUMNS = ("edge_id", "flow")

    def __init__(self, data: pd.DataFrame):
        missing = [c for c in self.REQUIRED_COLUMNS if c not in data.columns]
        if missing:
            raise ValueError(
                f"Missing required columns for {self.__class__.__name__}: {missing}"
            )
        self._data = data.copy()

    def __getattr__(self, name):
        return _dataframe_delegate(self, name)

    def to_dataframe(self, copy=True) -> pd.DataFrame:
        """Return aggregate network edge flows."""
        if copy:
            return self._data.copy()
        else:
            return self._data

    @classmethod
    def from_network_and_od_flows(
        cls,
        network: Network,
        od_flows: ODFlows,
    ) -> NetworkFlows:
        network_data = network.to_dataframe()
        if "flow" in network_data.columns:
            base_flows = pd.to_numeric(network_data["flow"], errors="coerce").fillna(0)
        else:
            base_flows = 0
        edge_flows = _flow_by_edge(od_flows)
        network_data["flow"] = (
            base_flows + network_data["edge_id"].map(edge_flows).fillna(0)
        )
        return cls(_coerce_integral_numeric_columns(network_data))


def _aggregate_od(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame(columns=list(FLOW_COLUMNS))
    aggregated = data.groupby(["origin_id", "destination_id"], as_index=False)[
        "flow"
    ].sum()
    return _coerce_integral_numeric_columns(aggregated)


def _flow_by_edge(od_flows: ODFlows) -> dict[str, float]:
    edge_flows = defaultdict(float)
    for row in od_flows.to_dataframe().itertuples(index=False):
        for edge_id in row.edge_path:
            edge_flows[edge_id] += row.flow
    return dict(edge_flows)


def _initial_residual_capacity(network_data: pd.DataFrame) -> dict[str, float]:
    residual_capacity = {}
    has_capacity = "capacity" in network_data.columns
    has_flow = "flow" in network_data.columns

    for row in network_data.itertuples(index=False):
        edge_id = row.edge_id
        capacity = getattr(row, "capacity") if has_capacity else inf
        flow = getattr(row, "flow") if has_flow else 0
        residual_capacity[edge_id] = max(float(capacity) - float(flow), 0.0)

    return residual_capacity


def _shortest_path(
    network_data: pd.DataFrame,
    origin_id,
    destination_id,
    *,
    directed: bool,
    residual_capacity: dict[str, float] | None = None,
) -> tuple[list[str], float] | tuple[None, None]:
    if origin_id == destination_id:
        return [], 0

    adjacency = defaultdict(list)
    has_cost = "cost" in network_data.columns
    sequence = 0
    for row in network_data.itertuples(index=False):
        edge_id = row.edge_id
        if (
            residual_capacity is not None
            and residual_capacity.get(edge_id, 0.0) <= CAPACITY_EPSILON
        ):
            continue
        edge_cost = getattr(row, "cost") if has_cost else 1
        adjacency[row.edge_from].append((row.edge_to, edge_id, float(edge_cost), sequence))
        sequence += 1
        if not directed:
            adjacency[row.edge_to].append(
                (row.edge_from, edge_id, float(edge_cost), sequence)
            )
            sequence += 1

    heap = [(0.0, 0, origin_id, [])]
    best_cost = {origin_id: 0.0}
    counter = 1

    while heap:
        cost, _, node_id, edge_path = heappop(heap)
        if node_id == destination_id:
            return edge_path, cost
        if cost > best_cost.get(node_id, inf) + CAPACITY_EPSILON:
            continue
        for next_node, edge_id, edge_cost, order in adjacency.get(node_id, []):
            next_cost = cost + edge_cost
            if next_cost + CAPACITY_EPSILON < best_cost.get(next_node, inf):
                best_cost[next_node] = next_cost
                heappush(
                    heap,
                    (next_cost, order + counter, next_node, edge_path + [edge_id]),
                )
                counter += 1

    return None, None
