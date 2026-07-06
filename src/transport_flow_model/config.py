"""Configuration for transport flow model runs.

JSON configs are a thin, validated layer that maps 1:1 onto the API:
:class:`RunConfig` names the input tables and their column mappings, and
its ``load_*`` methods return the corresponding API objects
(:class:`~transport_flow_model.network.Network`,
:class:`~transport_flow_model.demand.Demand`, scenario lists).

Relative paths are interpreted relative to the current working directory,
matching the behaviour of the original scripts.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from transport_flow_model.demand import Demand
from transport_flow_model.disruption import Scenario
from transport_flow_model.network import Network


class PathsConfig(BaseModel):
    """Data directories for a model run."""

    model_config = ConfigDict(extra="ignore")

    data: Path
    results: Path
    incoming_data: Path | None = None
    figures: Path | None = None


class TableConfig(BaseModel):
    """A tabular input: a path (relative to ``paths.data``) and a mapping
    of source column names to canonical API column names."""

    model_config = ConfigDict(extra="forbid")

    path: Path
    columns: dict[str, str]

    def read(self, data_dir: Path) -> pd.DataFrame:
        path = self.path if self.path.is_absolute() else data_dir / self.path
        if path.suffix == ".parquet":
            data = pd.read_parquet(path, columns=list(self.columns))
        else:
            data = pd.read_csv(path, usecols=list(self.columns))
        return data.rename(columns=self.columns)


class NetworkConfig(TableConfig):
    """Network link table; defaults match this repository's processed-data
    conventions."""

    path: Path = Path("network/network.csv")
    columns: dict[str, str] = Field(
        default_factory=lambda: {
            "from_id": "edge_from",
            "to_id": "edge_to",
            "id": "edge_id",
            "flow_capacity": "capacity",
            "gcost_usd_per_ton": "cost",
            "length_m": "length_m",
            "time_hr": "time_hr",
        }
    )


class DemandConfig(TableConfig):
    """OD demand table; defaults match this repository's processed-data
    conventions."""

    path: Path = Path("od/od.csv")
    columns: dict[str, str] = Field(
        default_factory=lambda: {
            "origin_id": "origin_id",
            "destination_id": "destination_id",
            "tons": "value",
        }
    )


class ScenariosConfig(BaseModel):
    """Disruption scenarios as a table of link ids to remove, one
    single-link scenario per row."""

    model_config = ConfigDict(extra="forbid")

    path: Path = Path("damages/failure_set.csv")
    id_column: str = "edge_id"


class AssignmentConfig(BaseModel):
    """Assignment method and options, passed to
    :func:`transport_flow_model.assignment.assign`."""

    model_config = ConfigDict(extra="forbid")

    method: str = "sequential"
    capacity_constrained: bool = True
    directed: bool = True

    def options(self) -> dict:
        return {
            "capacity_constrained": self.capacity_constrained,
            "directed": self.directed,
        }


class RunConfig(BaseModel):
    """A complete model run configuration."""

    model_config = ConfigDict(extra="ignore")

    paths: PathsConfig
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    demand: DemandConfig = Field(default_factory=DemandConfig)
    scenarios: ScenariosConfig = Field(default_factory=ScenariosConfig)
    assignment: AssignmentConfig = Field(default_factory=AssignmentConfig)

    @classmethod
    def from_json(cls, path: str | Path) -> RunConfig:
        """Load and validate a JSON config file."""
        with open(path) as handle:
            return cls.model_validate(json.load(handle))

    def load_network(self) -> Network:
        """Read the network table and build a :class:`Network`."""
        return Network(self.network.read(self.paths.data))

    def load_demand(self) -> Demand:
        """Read the OD table and build a :class:`Demand`."""
        return Demand(self.demand.read(self.paths.data))

    def load_scenarios(self) -> list[Scenario]:
        """Read the failure set as single-link removal scenarios."""
        path = self.scenarios.path
        if not path.is_absolute():
            path = self.paths.data / path
        failures = pd.read_csv(path)
        return [
            Scenario.remove_links(str(link_id), [link_id])
            for link_id in failures[self.scenarios.id_column]
        ]


def load_config(config_path=None):
    """Load configuration from a JSON file as a plain dict.

    .. deprecated:: 0.2.0
        Use :meth:`RunConfig.from_json` instead; this helper will be
        removed in 0.4.0.
    """
    warnings.warn(
        "load_config is deprecated and will be removed in 0.4.0; "
        "use RunConfig.from_json",
        DeprecationWarning,
        stacklevel=2,
    )
    if config_path is None:
        config_path = "./config.json"

    with open(config_path, "r") as config_fh:
        config = json.load(config_fh)
    return config
