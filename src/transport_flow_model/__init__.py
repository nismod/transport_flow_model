"""Transport flow model: network assignment and disruption analysis.

The public v0 API is the set of names exported here. Legacy classes remain
importable from :mod:`transport_flow_model.model` but are deprecated; see
the changelog and the versioning policy in the documentation.
"""

from .assignment import AssignmentResult, Provenance, assign
from .config import RunConfig, load_config
from .convergence import ConvergenceWarning, link_costs, relative_gap
from .costs import BPR, Conical, SpeedFlow, beckmann_objective
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
    "BPR",
    "Conical",
    "ConvergenceWarning",
    "Demand",
    "DisruptionResults",
    "LinkDelta",
    "Network",
    "Provenance",
    "RadiationModel",
    "RunConfig",
    "Scenario",
    "ScenarioResult",
    "SpeedFlow",
    "__version__",
    "assign",
    "beckmann_objective",
    "compute_losses",
    "datasets",
    "disrupt",
    "io",
    "link_costs",
    "load_config",
    "relative_gap",
]
