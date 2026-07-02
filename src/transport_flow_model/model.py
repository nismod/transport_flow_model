"""Core model classes"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

import geopandas as gpd
import pandas as pd

import transport_flow_model.core as core

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
    def from_file(cls, path: str | Path, column_map: dict[str, str], layer=None) -> OD:
        """Load a file (GeoJSON, Shapefile, etc.) and rename source columns to this class' schema."""
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

        data = gpd.read_file(path, layer=layer, usecols=to_rename)
        # Drop geometry column if present (for non-spatial data)
        if "geometry" in data.columns:
            data = data.drop(columns=["geometry"])
        data = data.rename(columns=column_map)
        data = pd.DataFrame(data)  # Convert from GeoDataFrame to DataFrame

        return cls(data)

    @classmethod
    def from_parquet(cls, path: str | Path, column_map: dict[str, str]) -> OD:
        """Load a Parquet file and rename source columns to this class' schema."""
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
            data = pd.read_parquet(path, columns=list(to_rename))
        except ValueError as e:
            msg = e.args[0].replace(
                "Usecols do not match columns, columns expected but not found: ", ""
            )
            raise ValueError(f"Missing expected columns: {msg}") from e
        data = data.rename(columns=column_map)

        return cls(data)

    def to_csv(self, path: str | Path, index=False):
        """Write OD data to CSV."""
        self._data.to_csv(path, index=index)

    def to_parquet(self, path: str | Path, index=False):
        """Write OD data to Parquet format."""
        self._data.to_parquet(path, index=index)

    def to_file(self, path: str | Path, layer=None, **kwargs):
        """Write OD data to a file using geopandas (GeoJSON, Shapefile, etc.)."""
        gdf = gpd.GeoDataFrame(self._data)
        gdf.to_file(path, layer=layer, **kwargs)

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
        losses["rerouting_loss"] = losses["disrupted_cost"] - losses["initial_cost"]
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

    def to_csv(self, path: str | Path, index=False):
        """Write network data to CSV."""
        self._data.to_csv(path, index=index)

    def to_parquet(self, path: str | Path, index=False):
        """Write network data to Parquet format."""
        self._data.to_parquet(path, index=index)

    def to_file(self, path: str | Path, layer=None, **kwargs):
        """Write network data to a file using geopandas (GeoJSON, Shapefile, etc.)."""
        gdf = gpd.GeoDataFrame(self._data)
        gdf.to_file(path, layer=layer, **kwargs)

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

    @classmethod
    def from_file(
        cls, path: str | Path, column_map: dict[str, str], layer=None
    ) -> Network:
        """Load a file (GeoJSON, Shapefile, etc.) and rename source columns to this class' schema."""
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

        data = gpd.read_file(path, layer=layer, usecols=to_rename)
        # Drop geometry column if present (for non-spatial data)
        if "geometry" in data.columns:
            data = data.drop(columns=["geometry"])
        data = data.rename(columns=column_map)
        data = pd.DataFrame(data)  # Convert from GeoDataFrame to DataFrame

        return cls(data)

    @classmethod
    def from_parquet(cls, path: str | Path, column_map: dict[str, str]) -> Network:
        """Load a Parquet file and rename source columns to this class' schema."""
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
            data = pd.read_parquet(path, columns=list(to_rename))
        except ValueError as e:
            msg = e.args[0].replace(
                "Usecols do not match columns, columns expected but not found: ", ""
            )
            raise ValueError(f"Missing expected columns: {msg}") from e
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
        return self._allocate(
            od,
            capacity_constrained=capacity_constrained,
            directed=directed,
        )

    def disrupt(
        self,
        od_flows: ODFlows,
        failed_edges: list[str],
        *,
        capacity_constrained: bool = True,
        directed: bool = True,
    ) -> DisruptionResult:
        """Reroute flows whose existing paths include any failed edge."""
        return self._disrupt(
            od_flows,
            failed_edges,
            capacity_constrained=capacity_constrained,
            directed=directed,
        )

    def _allocate(
        self,
        od: OD,
        *,
        capacity_constrained: bool,
        directed: bool,
    ) -> AllocationResult:
        network_data = self.to_dataframe()
        od_data = od.to_dataframe()
        normalized = _normalize(network_data, od_data)
        if normalized is None:
            return None
        network_input, od_input, node_id_map, edge_id_map, edge_value_to_id = normalized

        result = core.allocate(
            network_input,
            od_input,
            capacity_constrained=capacity_constrained,
            directed=directed,
        )
        od_flows = _od_flows_to_dataframe(
            result["od_flows"].to_pandas(),
            node_id_map=node_id_map,
            edge_id_map=edge_id_map,
        )
        unassigned_od = _od_to_dataframe(
            result["unassigned_od"].to_pandas(),
            node_id_map=node_id_map,
        )
        network_flows = _network_flows_from_result(
            network_data,
            result["network_flows"].to_pandas(),
            edge_value_to_id=edge_value_to_id,
        )

        return AllocationResult(
            od_flows=ODFlows(od_flows),
            network_flows=NetworkFlows(network_flows),
            unassigned_od=OD(unassigned_od),
        )

    def _disrupt(
        self,
        od_flows: ODFlows,
        failed_edges: list[str],
        *,
        capacity_constrained: bool,
        directed: bool,
    ) -> DisruptionResult:

        network_data = self.to_dataframe()
        od_flow_data = od_flows.to_dataframe()
        normalized = _normalize(network_data, od_flow_data)
        if normalized is None:
            return None
        network_input, od_flows_input, node_id_map, edge_id_map, edge_value_to_id = (
            normalized
        )
        try:
            od_flows_input["edge_path"] = od_flows_input["edge_path"].map(
                lambda path: [edge_value_to_id[edge_id] for edge_id in path]
            )
            failed_edge_ids = [
                edge_value_to_id[edge_id]
                for edge_id in failed_edges
                if edge_id in edge_value_to_id
            ]
        except TypeError:
            return None

        result = core.disrupt(
            network_input,
            od_flows_input,
            failed_edge_ids,
            capacity_constrained=capacity_constrained,
            directed=directed,
        )
        rerouted_flows = _od_flows_to_dataframe(
            result["rerouted_flows"].to_pandas(),
            node_id_map=node_id_map,
            edge_id_map=edge_id_map,
        )
        isolated_od = _od_to_dataframe(
            result["isolated_od"].to_pandas(),
            node_id_map=node_id_map,
        )
        losses = _losses_to_dataframe(
            result["losses"].to_pandas(),
            node_id_map=node_id_map,
        )
        network_flows = _network_flows_from_result(
            network_data,
            result["network_flows"].to_pandas(),
            edge_value_to_id=edge_value_to_id,
        )

        return DisruptionResult(
            rerouted_flows=ODFlows(rerouted_flows),
            network_flows=NetworkFlows(network_flows),
            isolated_od=OD(isolated_od),
            losses=OD(losses),
        )


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

    def to_parquet(self, path: str | Path, index=False):
        """Write OD flow paths to Parquet format."""
        self._data.to_parquet(path, index=index)

    def to_file(self, path: str | Path, layer=None, **kwargs):
        """Write OD flow paths to a file using geopandas (GeoJSON, Shapefile, etc.)."""
        gdf = gpd.GeoDataFrame(self._data)
        gdf.to_file(path, layer=layer, **kwargs)


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
        network_data["flow"] = base_flows + network_data["edge_id"].map(
            edge_flows
        ).fillna(0)
        return cls(_coerce_integral_numeric_columns(network_data))

    def to_csv(self, path: str | Path, index=False):
        """Write network flows to CSV."""
        self._data.to_csv(path, index=index)

    def to_parquet(self, path: str | Path, index=False):
        """Write network flows to Parquet format."""
        self._data.to_parquet(path, index=index)

    def to_file(self, path: str | Path, layer=None, **kwargs):
        """Write network flows to a file using geopandas (GeoJSON, Shapefile, etc.)."""
        gdf = gpd.GeoDataFrame(self._data)
        gdf.to_file(path, layer=layer, **kwargs)


def _normalize(
    network_data: pd.DataFrame,
    flow_data: pd.DataFrame,
) -> (
    tuple[
        pd.DataFrame,
        pd.DataFrame,
        list[object],
        list[object],
        dict[object, int],
    ]
    | None
):
    node_value_to_id, node_id_map = _indexed_id_map(
        network_data["edge_from"],
        network_data["edge_to"],
        flow_data["origin_id"],
        flow_data["destination_id"],
    )
    edge_value_to_id, edge_id_map = _unique_indexed_id_map(network_data["edge_id"])
    if node_value_to_id is None or edge_value_to_id is None:
        return None

    network_input = network_data.copy()
    network_input["edge_from"] = network_input["edge_from"].map(node_value_to_id)
    network_input["edge_to"] = network_input["edge_to"].map(node_value_to_id)
    network_input["edge_id"] = network_input["edge_id"].map(edge_value_to_id)

    flow_input = flow_data.copy()
    flow_input["origin_id"] = flow_input["origin_id"].map(node_value_to_id)
    flow_input["destination_id"] = flow_input["destination_id"].map(node_value_to_id)

    return network_input, flow_input, node_id_map, edge_id_map, edge_value_to_id


def _indexed_id_map(
    *columns: pd.Series,
) -> tuple[dict[object, int] | None, list[object] | None]:
    value_to_id = {}
    id_to_value = []
    for column in columns:
        for value in column:
            try:
                if value in value_to_id:
                    continue
                value_to_id[value] = len(id_to_value)
            except TypeError:
                return None, None
            id_to_value.append(value)
    return value_to_id, id_to_value


def _unique_indexed_id_map(
    column: pd.Series,
) -> tuple[dict[object, int] | None, list[object] | None]:
    value_to_id = {}
    id_to_value = []
    for value in column:
        try:
            if value in value_to_id:
                return None, None
            value_to_id[value] = len(id_to_value)
        except TypeError:
            return None, None
        id_to_value.append(value)
    return value_to_id, id_to_value


def _od_flows_to_dataframe(
    data: pd.DataFrame,
    *,
    node_id_map: list[object],
    edge_id_map: list[object],
) -> pd.DataFrame:
    data = data.loc[:, list(OD_FLOW_COLUMNS)].copy()
    data["origin_id"] = data["origin_id"].map(lambda value: node_id_map[int(value)])
    data["destination_id"] = data["destination_id"].map(
        lambda value: node_id_map[int(value)]
    )
    data["edge_path"] = data["edge_path"].map(
        lambda path: [edge_id_map[int(edge_id)] for edge_id in path]
    )
    return _coerce_integral_numeric_columns(data)


def _od_to_dataframe(
    data: pd.DataFrame,
    *,
    node_id_map: list[object],
) -> pd.DataFrame:
    data = data.loc[:, list(FLOW_COLUMNS)].copy()
    data["origin_id"] = data["origin_id"].map(lambda value: node_id_map[int(value)])
    data["destination_id"] = data["destination_id"].map(
        lambda value: node_id_map[int(value)]
    )
    return _coerce_integral_numeric_columns(data)


def _losses_to_dataframe(
    data: pd.DataFrame,
    *,
    node_id_map: list[object],
) -> pd.DataFrame:
    data = data.loc[:, list(LOSS_COLUMNS)].copy()
    data["origin_id"] = data["origin_id"].map(lambda value: node_id_map[int(value)])
    data["destination_id"] = data["destination_id"].map(
        lambda value: node_id_map[int(value)]
    )
    return _coerce_integral_numeric_columns(data)


def _network_flows_from_result(
    network_data: pd.DataFrame,
    network_flows: pd.DataFrame,
    *,
    edge_value_to_id: dict[object, int],
) -> pd.DataFrame:
    flow_by_edge = dict(
        zip(
            network_flows["edge_id"].map(int),
            network_flows["flow"],
            strict=False,
        )
    )
    data = network_data.copy()
    data["flow"] = data["edge_id"].map(
        lambda edge_id: flow_by_edge.get(edge_value_to_id[edge_id], 0)
    )
    return _coerce_integral_numeric_columns(data)


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
