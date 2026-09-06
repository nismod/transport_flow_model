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

import inspect
import json
import warnings
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from transport_flow_model.assignment import METHODS
from transport_flow_model.costs import build_cost_function
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


#: Backend parameters that are not caller-selectable options: ADR-0001 has
#: :func:`~transport_flow_model.assign` supply all three itself.
_NOT_OPTIONS = frozenset({"network", "demand", "include_paths"})


class AssignmentConfig(BaseModel):
    """Assignment method and options, passed to
    :func:`transport_flow_model.assignment.assign`."""

    model_config = ConfigDict(extra="forbid")

    method: str = "sequential"
    capacity_constrained: bool = True
    directed: bool = True
    cost_function: dict | None = None
    method_options: dict = Field(default_factory=dict)

    def options(self, network: Network | None = None) -> dict:
        """Keyword arguments for ``assign(..., method, **options())``.

        ``capacity_constrained`` and ``directed`` are included only when the
        backend registered under ``method`` (see
        :data:`transport_flow_model.assignment.METHODS`) actually declares
        them as parameters — read off the backend's own signature with
        :func:`inspect.signature` rather than a hardcoded per-method list,
        so a new or changed backend is picked up automatically. This is
        safe here specifically because backends are plain functions with
        explicit keyword parameters (see ADR-0001): nothing wraps them in a
        generic ``(*args, **kwargs)``, so their signature *is* the option
        contract, not a guess at one. A ``method`` not (yet) registered is
        not an error here — every option is included unfiltered, and
        :func:`~transport_flow_model.assign` raises when it looks the name
        up itself.

        ``cost_function`` — a ``{"name": ..., **params}`` dict — is built
        with :func:`~transport_flow_model.costs.build_cost_function`
        against ``network`` and included the same way, so passing
        ``network`` in is only required when ``cost_function`` is set.

        Only a field left at its default is dropped that way. A field the
        config *sets* is an instruction, so a method that cannot accept it
        raises :class:`ValueError` here rather than running as if it had
        not been asked for. That distinction matters most for
        ``cost_function``: silently dropping it would assign the run with
        BPR while the config asked for another curve, and nothing in the
        results would say so.

        ``method_options`` (free-form, e.g. MSA's ``max_iterations``) are
        passed straight through, unfiltered: they are exactly what the
        caller wants that backend to receive, and an unknown or misspelled
        one should raise from the backend call, not be silently dropped
        here.
        """
        candidates: dict = {
            "capacity_constrained": self.capacity_constrained,
            "directed": self.directed,
        }
        if self.cost_function is not None:
            if network is None:
                raise ValueError(
                    "AssignmentConfig.cost_function is set; options() needs "
                    "a network to build it"
                )
            name = self.cost_function["name"]
            params = {k: v for k, v in self.cost_function.items() if k != "name"}
            candidates["cost_function"] = build_cost_function(name, network, **params)

        backend = METHODS.get(self.method)
        if backend is not None:
            accepted = inspect.signature(backend).parameters
            unsupported = sorted(
                name
                for name in candidates
                if name not in accepted and name in self.model_fields_set
            )
            if unsupported:
                # ADR-0001: a backend is called as ``backend(network,
                # demand, include_paths=..., **options)``, so its options
                # are its parameters less those three.
                options = sorted(set(accepted) - _NOT_OPTIONS)
                raise ValueError(
                    f"Assignment method {self.method!r} does not accept "
                    f"{', '.join(repr(name) for name in unsupported)}, which "
                    f"this config sets; its options are "
                    f"{', '.join(options)}"
                )
            candidates = {k: v for k, v in candidates.items() if k in accepted}
        return {**candidates, **self.method_options}


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
