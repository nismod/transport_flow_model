"""Transport flow model: network assignment and disruption analysis."""

from .assignment import AssignmentResult, Provenance, assign
from .config import RunConfig, load_config
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
from .radiation import RadiationModel
from . import datasets, io

try:
    from ._version import __version__
except ImportError:  # pragma: no cover - package not built/installed
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
    "assign",
    "compute_losses",
    "datasets",
    "disrupt",
    "io",
    "link_costs",
    "load_config",
    "relative_gap",
]
