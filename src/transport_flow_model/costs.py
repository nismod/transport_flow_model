"""Link cost (volume-delay) functions and the Beckmann objective.

A *cost function* turns link flows into link travel times. Every cost
function satisfies the :class:`CostFunction` protocol: it is vectorised over
links in network link order and provides three quantities,

- ``travel_time(x)`` — :math:`t_a(x_a)`, what assignment and
  :func:`~transport_flow_model.relative_gap` need;
- ``integral(x)`` — :math:`\\int_0^{x_a} t_a(w)\\,\\mathrm{d}w`, the per-link
  term of the Beckmann objective that user equilibrium minimizes;
- ``derivative(x)`` — :math:`\\mathrm{d}t_a/\\mathrm{d}x_a`, for Newton steps
  in bush-based methods and for gradient-based calibration.

:class:`BPR` is the only implementation today. It is the convention used by
the TNTP datasets (see
:data:`transport_flow_model.datasets.BEST_KNOWN`):

.. math::

    t_a(x_a) = t^0_a \\left(1 + \\alpha_a (x_a / c_a)^{\\beta_a}\\right)
               + \\gamma \\, \\ell_a

with a free-flow time :math:`t^0_a` (the ``cost`` link attribute), capacity
:math:`c_a`, per-link :math:`\\alpha_a`, :math:`\\beta_a`, and an optional
additive distance term :math:`\\gamma \\ell_a` (``distance_cost`` times the
``length`` attribute) for generalized cost. Its integral is

.. math::

    \\int_0^{x} t_a(w)\\,\\mathrm{d}w =
        t^0_a x \\left(1 + \\frac{\\alpha_a}{\\beta_a + 1}
                        (x / c_a)^{\\beta_a}\\right) + \\gamma \\ell_a x

which is also right at :math:`\\beta_a = 0`, where the published convention
makes the link a constant :math:`t^0_a (1 + \\alpha_a)`.

Networks without all three of ``alpha``, ``beta`` and ``capacity`` are
fixed-cost: :meth:`BPR.from_network` sets ``alpha`` to zero, so
``travel_time`` returns the free-flow time exactly.

Conical (Spiess 1990) and DfT-style piecewise-linear speed-flow curves, and
mirrored implementations in the Rust core, are still open in ws2-01.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pyarrow as pa

from transport_flow_model.network import Network


@runtime_checkable
class CostFunction(Protocol):
    """Link cost function: travel time, its integral and its derivative.

    Each method takes link flows ``x`` in network link order — a float64
    array of length ``n_links``, or a scalar broadcast to that length — and
    returns a float64 array of the same length.
    """

    def travel_time(self, x: Any) -> np.ndarray:
        """Link travel times ``t_a(x_a)``."""
        ...

    def integral(self, x: Any) -> np.ndarray:
        """Per-link Beckmann terms ``∫_0^x t_a(w) dw``."""
        ...

    def derivative(self, x: Any) -> np.ndarray:
        """Marginal travel times ``dt_a/dx_a``."""
        ...


@dataclass(frozen=True)
class BPR:
    """The Bureau of Public Roads volume-delay function.

    Parameters are per-link float64 arrays in network link order; build one
    from a :class:`~transport_flow_model.Network` with :meth:`from_network`.

    Attributes
    ----------
    free_flow : numpy.ndarray
        Free-flow travel time ``t0`` per link (the ``cost`` attribute).
    capacity : numpy.ndarray
        Link capacity. A link with ``capacity <= 0`` is treated as
        uncongested: its flow/capacity ratio is taken as zero, as
        :func:`~transport_flow_model.link_costs` has always done.
    alpha, beta : numpy.ndarray
        BPR shape parameters. ``alpha == 0`` makes a link fixed-cost, and
        ``travel_time`` then returns ``free_flow`` exactly.
    distance_term : numpy.ndarray
        Additive generalized-cost term ``distance_cost * length`` per link;
        zeros when distance is not priced.
    """

    free_flow: np.ndarray
    capacity: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    distance_term: np.ndarray

    @classmethod
    def from_network(cls, network: Network, *, distance_cost: float = 0.0) -> BPR:
        """Read BPR parameters from a network's link attributes.

        ``cost`` (free-flow travel time) is required. ``alpha``, ``beta``
        and ``capacity`` are read only when **all three** are present;
        otherwise the network is fixed-cost and ``alpha`` is zero.
        ``length`` is read only when ``distance_cost`` is nonzero.
        """
        n = network.n_links
        free_flow = _as_float_array(network.attribute("cost"), n, "cost")
        names = set(network.to_table().column_names)
        if {"alpha", "beta", "capacity"} <= names:
            alpha = _as_float_array(network.attribute("alpha"), n, "alpha")
            beta = _as_float_array(network.attribute("beta"), n, "beta")
            capacity = _as_float_array(network.attribute("capacity"), n, "capacity")
        else:
            alpha = np.zeros(n)
            beta = np.zeros(n)
            capacity = np.zeros(n)
        if distance_cost:
            length = _as_float_array(network.attribute("length"), n, "length")
            distance_term = distance_cost * length
        else:
            distance_term = np.zeros(n)
        return cls(
            free_flow=free_flow,
            capacity=capacity,
            alpha=alpha,
            beta=beta,
            distance_term=distance_term,
        )

    @property
    def n_links(self) -> int:
        return len(self.free_flow)

    def travel_time(self, x: Any) -> np.ndarray:
        """``t0 * (1 + alpha * (x / capacity)^beta) + distance_term``."""
        x = self._flows(x)
        return (
            self.free_flow * (1.0 + self.alpha * np.power(self._ratio(x), self.beta))
            + self.distance_term
        )

    def integral(self, x: Any) -> np.ndarray:
        """``t0 * x * (1 + alpha / (beta + 1) * (x / capacity)^beta)``.

        Plus ``distance_term * x``. Correct at ``beta == 0`` too, where the
        link is a constant ``t0 * (1 + alpha)``.
        """
        x = self._flows(x)
        ratio = self._ratio(x)
        return (
            self.free_flow
            * x
            * (1.0 + self.alpha / (self.beta + 1.0) * np.power(ratio, self.beta))
            + self.distance_term * x
        )

    def derivative(self, x: Any) -> np.ndarray:
        """``t0 * alpha * beta * (x / capacity)^(beta - 1) / capacity``.

        Zero where ``capacity <= 0``. For ``beta < 1`` the true derivative is
        unbounded at ``x == 0``; zero is returned there rather than ``inf``
        or ``nan``, so a Newton step on an empty link is a no-op.
        """
        x = self._flows(x)
        ratio = self._ratio(x)
        usable = self.capacity > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = (
                self.free_flow
                * self.alpha
                * self.beta
                * np.power(ratio, self.beta - 1.0)
                / np.where(usable, self.capacity, 1.0)
            )
        singular = (x <= 0.0) & (self.beta < 1.0)
        return np.where(usable & ~singular, slope, 0.0)

    def _ratio(self, x: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.capacity > 0, x / self.capacity, 0.0)

    def _flows(self, x: Any) -> np.ndarray:
        if np.isscalar(x) or (isinstance(x, np.ndarray) and x.ndim == 0):
            return np.full(self.n_links, float(x), dtype="float64")
        return _as_float_array(x, self.n_links, "flows")


def beckmann_objective(
    network: Network,
    flows: Any,
    *,
    cost_function: CostFunction | None = None,
    distance_cost: float = 0.0,
) -> float:
    """Beckmann objective ``sum_a ∫_0^{x_a} t_a(w) dw`` of a link-flow solution.

    User equilibrium is the flow pattern that minimizes this, so it is the
    natural check against a published solution: see
    :data:`transport_flow_model.datasets.BEST_KNOWN`.

    ``flows`` may be an array of link flows in network link order, an
    :class:`~transport_flow_model.AssignmentResult`, or a table with a
    ``flow`` column in network link order.

    ``cost_function`` defaults to :meth:`BPR.from_network`; ``distance_cost``
    is passed to that default and is unused when a cost function is given.
    """
    x = _flow_array(network, flows)
    if cost_function is None:
        cost_function = BPR.from_network(network, distance_cost=distance_cost)
    return float(np.sum(cost_function.integral(x)))


def _flow_array(network: Network, flows: Any) -> np.ndarray:
    link_flows = getattr(flows, "link_flows", None)
    if link_flows is not None:  # AssignmentResult
        flows = link_flows
    if isinstance(flows, (pa.Table, pa.RecordBatch)):
        if "flow" not in flows.schema.names:
            raise ValueError("Flow table has no 'flow' column")
        flows = flows["flow"]
    return _as_float_array(flows, network.n_links, "flows")


def _as_float_array(values: Any, n: int, name: str) -> np.ndarray:
    if isinstance(values, (pa.Array, pa.ChunkedArray)):
        array = values.to_numpy(zero_copy_only=False)
    else:
        array = np.asarray(values)
    array = array.astype("float64", copy=False)
    if array.shape != (n,):
        raise ValueError(f"Expected {name} to have shape ({n},), got {array.shape}")
    return array
