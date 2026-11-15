"""Transport flow model: network assignment and disruption analysis."""

from . import datasets, io
from .assignment import AssignmentResult, Provenance, assign
from .config import RunConfig, load_config
from .constraints import (
    aggregate_od,
    calibrate_radiation,
    common_part_of_commuters,
    od_fit,
    scale_to_aggregate,
)
from .convergence import link_costs, relative_gap
from .demand import Demand
from .disruption import (
    DisruptionResults,
    LinkDelta,
    Scenario,
    ScenarioResult,
    disrupt,
)
from .model import compute_losses
from .network import Network
from .radiation import RadiationModel, radiation_flows, radiation_probabilities

try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("nismod-tfm")
except PackageNotFoundError:  # pragma: no cover - package not installed
    __version__ = "unknown"

__all__ = [
    "AssignmentResult",
    "Demand",
    "DisruptionResults",
    "LinkDelta",
    "Network",
    "Provenance",
    "RadiationModel",
    "RunConfig",
    "Scenario",
    "ScenarioResult",
    "__version__",
    "aggregate_od",
    "assign",
    "calibrate_radiation",
    "common_part_of_commuters",
    "compute_losses",
    "datasets",
    "disrupt",
    "io",
    "link_costs",
    "load_config",
    "od_fit",
    "radiation_flows",
    "radiation_probabilities",
    "relative_gap",
    "scale_to_aggregate",
]
