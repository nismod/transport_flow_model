"""Core model classes"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

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
        if _is_integral_numeric_column(values):
            data[column] = values.round().astype("int64")
        else:
            data[column] = values
    return data


def _is_integral_numeric_column(values: pd.Series) -> bool:
    column = pa.array(values, from_pandas=True)
    if column.null_count:
        return False

    rounded = pc.round(column)
    difference = pc.abs(pc.subtract(column, rounded))
    close_to_integer = pc.less_equal(difference, CAPACITY_EPSILON)
    return pc.all(pc.is_finite(column)).as_py() and pc.all(close_to_integer).as_py()


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


def _validate_column_map(
    column_map: dict[str, str], required_columns: tuple[str, ...]
) -> None:
    """Raise if column_map doesn't provide all required target column names."""
    missing = [c for c in required_columns if c not in column_map.values()]
    if missing:
        raise ValueError(
            f"column_map must include mappings for required columns {missing}"
        )


class _TabularData:
    """Base class for tabular data wrappers with column validation and I/O."""

    REQUIRED_COLUMNS: tuple[str, ...] = ()

    def __init__(self, data: pd.DataFrame):
        missing = [c for c in self.REQUIRED_COLUMNS if c not in data.columns]
        if missing:
            raise ValueError(
                f"Missing required columns for {self.__class__.__name__}: {missing}"
            )
        self._data = data

    def to_dataframe(self, copy=True) -> pd.DataFrame:
        """Return the tabular data as a pandas DataFrame."""
        return self._data.copy() if copy else self._data

    def to_csv(self, path: str | Path, index=False):
        """Write data to CSV."""
        self._data.to_csv(path, index=index)

    def to_parquet(self, path: str | Path, index=False):
        """Write data to Parquet format."""
        self._data.to_parquet(path, index=index)

    def to_file(self, path: str | Path, layer=None, **kwargs):
        """Write data to a file using geopandas (GeoJSON, Shapefile, etc.)."""
        gpd.GeoDataFrame(self._data).to_file(path, layer=layer, **kwargs)

    @classmethod
    def from_csv(cls, path: str | Path, column_map: dict[str, str]):
        """Load a CSV and rename source columns to this class' schema."""
        _validate_column_map(column_map, cls.REQUIRED_COLUMNS)
        try:
            data = pd.read_csv(path, usecols=list(column_map))
        except ValueError as e:
            msg = e.args[0].replace(
                "Usecols do not match columns, columns expected but not found: ", ""
            )
            raise ValueError(f"Missing expected columns: {msg}") from e
        return cls(data.rename(columns=column_map))

    @classmethod
    def from_file(cls, path: str | Path, column_map: dict[str, str], layer=None):
        """Load a file (GeoJSON, Shapefile, etc.) and rename source columns to this class' schema."""
        _validate_column_map(column_map, cls.REQUIRED_COLUMNS)
        data = gpd.read_file(path, layer=layer, columns=list(column_map))
        if "geometry" in data.columns:
            data = data.drop(columns=["geometry"])
        return cls(pd.DataFrame(data).rename(columns=column_map))

    @classmethod
    def from_parquet(cls, path: str | Path, column_map: dict[str, str]):
        """Load a Parquet file and rename source columns to this class' schema."""
        _validate_column_map(column_map, cls.REQUIRED_COLUMNS)
        try:
            data = pd.read_parquet(path, columns=list(column_map))
        except ValueError as e:
            msg = e.args[0].replace(
                "Usecols do not match columns, columns expected but not found: ", ""
            )
            raise ValueError(f"Missing expected columns: {msg}") from e
        return cls(data.rename(columns=column_map))


class OD(_TabularData):
    """Origin-destination flows.

    - sparse matrix representation
    - output of methods to estimate OD from spatial interactions
    - input to Network flow allocation
    """

    REQUIRED_COLUMNS = ("origin_id", "destination_id", "flow")

    @property
    def data(self) -> pd.DataFrame:
        """Return a defensive copy of the normalized tabular data."""
        return self._data.copy()


class Network(_TabularData):
    """Base network model class.

    - graph representation
    - works with methods to allocate OD flows to the network, outputting ODFlows
    """

    REQUIRED_COLUMNS = ("edge_from", "edge_to", "edge_id")
    OPTIONAL_COLUMNS = ("capacity", "cost", "flow")

    def allocate(
        self,
        od: OD,
        *,
        capacity_constrained: bool = False,
        directed: bool = True,
    ) -> AllocationResult:
        """Allocate OD flows to least-cost paths on this network."""

        result = core.allocate(
            self._data,
            od.to_dataframe(copy=False),
            capacity_constrained=capacity_constrained,
            directed=directed,
        )
        od_flows = _od_flows_to_dataframe(result["od_flows"].to_pandas())
        unassigned_od = _od_to_dataframe(result["unassigned_od"].to_pandas())
        network_flows = _network_flows_from_result(
            self._data,
            result["network_flows"].to_pandas(),
        )

        return AllocationResult(
            od_flows=ODFlows(od_flows),
            network_flows=NetworkFlows(network_flows),
            unassigned_od=OD(unassigned_od),
        )

    def disrupt(
        self,
        od_flows: ODFlows,
        failed_edges: list[object],
        *,
        capacity_constrained: bool = True,
        directed: bool = True,
    ) -> DisruptionResult:
        """Reroute flows whose existing paths include any failed edge."""
        result = core.disrupt(
            self._data,
            od_flows.to_dataframe(copy=False),
            failed_edges,
            capacity_constrained=capacity_constrained,
            directed=directed,
        )
        rerouted_flows = _od_flows_to_dataframe(result["rerouted_flows"].to_pandas())
        isolated_od = _od_to_dataframe(result["isolated_od"].to_pandas())
        losses = _losses_to_dataframe(result["losses"].to_pandas())
        network_flows = _network_flows_from_result(
            self._data,
            result["network_flows"].to_pandas(),
        )

        return DisruptionResult(
            rerouted_flows=ODFlows(rerouted_flows),
            network_flows=NetworkFlows(network_flows),
            isolated_od=OD(isolated_od),
            losses=OD(losses),
        )


class ODFlows(_TabularData):
    """Origin-destination flow paths

    - full path representation of allocated flows for each OD
    - may include multiple paths for a single OD pair with different capacity
      allocation
    """

    REQUIRED_COLUMNS = ("origin_id", "destination_id", "flow", "edge_path")


class NetworkFlows(_TabularData):
    """Aggregate flows on network

    - could consider this as Network with calculated attributes
      as the result of an allocation
    - multiple attributes may include total allocation, speed, cost, ...
    - can be constructed from a Network and ODFlows
    """

    REQUIRED_COLUMNS = ("edge_id", "flow")

    @classmethod
    def from_network_and_od_flows(
        cls,
        network: Network,
        od_flows: ODFlows,
    ) -> NetworkFlows:
        network_data = network.to_dataframe(copy=False)
        if "flow" in network_data.columns:
            base_flows = pd.to_numeric(network_data["flow"], errors="coerce").fillna(0)
        else:
            base_flows = 0
        edge_flows = _flow_by_edge(od_flows)
        network_data = network_data.copy()
        network_data["flow"] = base_flows + network_data["edge_id"].map(
            edge_flows
        ).fillna(0)
        return cls(_coerce_integral_numeric_columns(network_data))


def compute_losses(initial: ODFlows, disrupted: ODFlows) -> OD:
    """Aggregate per-OD rerouting losses from initial and disrupted paths."""
    initial_data = initial.to_dataframe()
    disrupted_data = disrupted.to_dataframe()

    if disrupted_data.empty:
        return OD(pd.DataFrame(columns=list(LOSS_COLUMNS)))

    initial_costs = (
        initial_data.groupby(["origin_id", "destination_id"], as_index=False)["cost"]
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
    return OD(_coerce_integral_numeric_columns(losses))


def _od_flows_to_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    data = data.loc[:, list(OD_FLOW_COLUMNS)].copy()
    data["edge_path"] = data["edge_path"].map(list)
    return _coerce_integral_numeric_columns(data)


def _od_to_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    data = data.loc[:, list(FLOW_COLUMNS)].copy()
    return _coerce_integral_numeric_columns(data)


def _losses_to_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    data = data.loc[:, list(LOSS_COLUMNS)].copy()
    return _coerce_integral_numeric_columns(data)


def _network_flows_from_result(
    network_data: pd.DataFrame,
    network_flows: pd.DataFrame,
) -> pd.DataFrame:
    flow_by_edge = dict(
        zip(network_flows["edge_id"], network_flows["flow"], strict=False)
    )
    data = network_data.copy()
    data["flow"] = data["edge_id"].map(lambda edge_id: flow_by_edge.get(edge_id, 0))
    return _coerce_integral_numeric_columns(data)


def _flow_by_edge(od_flows: ODFlows) -> dict[str, float]:
    edge_flows = defaultdict(float)
    for row in od_flows.to_dataframe().itertuples(index=False):
        for edge_id in row.edge_path:
            edge_flows[edge_id] += row.flow
    return dict(edge_flows)
