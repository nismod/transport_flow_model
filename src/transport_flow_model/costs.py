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

:class:`Conical` is the Spiess (1990) conical volume-delay function, an
alternative to BPR that stays finite and increasing at every flow (BPR's
``beta``-power curve is vertical at ``x = capacity`` for ``beta > 1``, which
is awkward for Newton-style bush methods). With :math:`v_a = x_a / c_a` and
per-link shape parameter :math:`\\alpha_a > 1`,

.. math::

    \\beta_a &= \\frac{2\\alpha_a - 1}{2\\alpha_a - 2} \\\\
    C_a(v_a) &= 2 + \\sqrt{\\alpha_a^2 (1 - v_a)^2 + \\beta_a^2}
                 - \\alpha_a (1 - v_a) - \\beta_a \\\\
    t_a(x_a) &= t^0_a\\, C_a(v_a) + \\gamma \\ell_a

:class:`SpeedFlow` is a piecewise-linear speed-against-flow-per-lane curve,
the shape a DfT TAG speed-flow table takes: speed falls linearly between
flow-per-lane breakpoints, clamped below at a floor speed so travel time
stays finite.

Mirrored implementations of all three in the Rust core, golden-tested
against these, are still open in ws2-01, as is selecting a cost function
from :func:`~transport_flow_model.assign`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd
import pyarrow as pa

from transport_flow_model.network import Network, _as_table


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


@dataclass(frozen=True)
class Conical:
    """The Spiess (1990) conical volume-delay function.

    Parameters are per-link float64 arrays in network link order; build one
    from a :class:`~transport_flow_model.Network` with :meth:`from_network`.

    Attributes
    ----------
    free_flow : numpy.ndarray
        Free-flow travel time ``t0`` per link (the ``cost`` attribute).
    capacity : numpy.ndarray
        Link capacity. A link with ``capacity <= 0`` is treated as
        uncongested, the same convention :class:`BPR` uses: its
        flow/capacity ratio is taken as zero, so ``travel_time`` is the
        (near-free-flow) constant ``C(0)`` regardless of ``x``.
    alpha : numpy.ndarray
        Per-link curvature parameter, required to be strictly greater than
        1. It plays the role BPR's ``beta`` plays — how sharply the curve
        bends as flow approaches capacity — and is unrelated to BPR's
        ``alpha`` link attribute despite the name collision (see
        :meth:`from_network`). ``beta = (2*alpha - 1) / (2*alpha - 2)`` is
        derived from it, not stored.
    distance_term : numpy.ndarray
        Additive generalized-cost term ``distance_cost * length`` per link;
        zeros when distance is not priced.
    """

    free_flow: np.ndarray
    capacity: np.ndarray
    alpha: np.ndarray
    distance_term: np.ndarray

    @classmethod
    def from_network(
        cls, network: Network, *, alpha: Any = 4.0, distance_cost: float = 0.0
    ) -> Conical:
        """Read Conical parameters from a network's link attributes.

        ``cost`` (free-flow travel time) and ``capacity`` are required.

        ``alpha`` is **not** read from a network link attribute: on every
        TNTP network, a link attribute named ``alpha`` already exists and
        holds BPR's shape parameter (conventionally 0.15), which is not a
        conical alpha and would silently produce nonsense if reused here.
        Pass it as a keyword argument instead — a scalar applied to every
        link, or a per-link array. The default, ``4.0``, is a reasonable
        curve shape and nothing more: it is not a calibration for any
        network, unlike BPR's ``alpha``/``beta`` link attributes, which
        typically *are* published calibrations.

        ``length`` is read only when ``distance_cost`` is nonzero.

        Raises
        ------
        ValueError
            If any element of ``alpha`` is not strictly greater than 1:
            ``beta`` is undefined at ``alpha == 1`` (division by zero) and
            ``C`` stops being an increasing function of flow below it.
        """
        n = network.n_links
        free_flow = _as_float_array(network.attribute("cost"), n, "cost")
        capacity = _as_float_array(network.attribute("capacity"), n, "capacity")
        if np.isscalar(alpha):
            alpha_array = np.full(n, float(alpha), dtype="float64")
        else:
            alpha_array = _as_float_array(alpha, n, "alpha")
        if np.any(alpha_array <= 1.0):
            raise ValueError(
                "Conical alpha must be > 1: beta = (2*alpha - 1) / (2*alpha "
                "- 2) is undefined at alpha == 1, and C(v) is not an "
                "increasing function of flow below it"
            )
        if distance_cost:
            length = _as_float_array(network.attribute("length"), n, "length")
            distance_term = distance_cost * length
        else:
            distance_term = np.zeros(n)
        return cls(
            free_flow=free_flow,
            capacity=capacity,
            alpha=alpha_array,
            distance_term=distance_term,
        )

    @property
    def n_links(self) -> int:
        return len(self.free_flow)

    def travel_time(self, x: Any) -> np.ndarray:
        """``t0 * C(v) + distance_term`` with ``v = x / capacity``."""
        x = self._flows(x)
        v = self._ratio(x)
        return self.free_flow * self._c(v) + self.distance_term

    def integral(self, x: Any) -> np.ndarray:
        """``t0 * capacity * [(2-beta)*v - alpha*(v - v^2/2) + F(1) - F(1-v)]``.

        Plus ``distance_term * x``, where ``F(s) = (s/2) * sqrt(alpha^2 s^2
        + beta^2) + (beta^2 / (2*alpha)) * asinh(alpha*s/beta)``. Verified
        against ``numpy.trapezoid`` quadrature of ``travel_time`` to a
        relative 7e-15 to 3e-12 at ``alpha=4, t0=7, capacity=1500``.

        A ``capacity <= 0`` link does not follow this formula — its ratio is
        pinned at zero for every ``x``, so it is a straight line, ``(t0 *
        C(0) + distance_term) * x``, not the curve the formula describes.
        """
        x = self._flows(x)
        usable = self.capacity > 0
        capacity = np.where(usable, self.capacity, 1.0)
        v = np.where(usable, x / capacity, 0.0)
        alpha = self.alpha
        beta = self._beta()
        bracket = (
            (2.0 - beta) * v
            - alpha * (v - v * v / 2.0)
            + self._f(np.ones_like(v))
            - self._f(1.0 - v)
        )
        congested = self.free_flow * capacity * bracket + self.distance_term * x
        uncongested = (
            self.free_flow * self._c(np.zeros_like(v)) + self.distance_term
        ) * x
        return np.where(usable, congested, uncongested)

    def derivative(self, x: Any) -> np.ndarray:
        """``(t0 / capacity) * (alpha - alpha^2*(1-v) / sqrt(alpha^2*(1-v)^2 + beta^2))``.

        Zero where ``capacity <= 0``, the same convention :class:`BPR` uses.
        """
        x = self._flows(x)
        v = self._ratio(x)
        alpha = self.alpha
        beta = self._beta()
        usable = self.capacity > 0
        denom = np.sqrt(alpha * alpha * (1.0 - v) ** 2 + beta * beta)
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = (self.free_flow / np.where(usable, self.capacity, 1.0)) * (
                alpha - alpha * alpha * (1.0 - v) / denom
            )
        return np.where(usable, slope, 0.0)

    def _beta(self) -> np.ndarray:
        return (2.0 * self.alpha - 1.0) / (2.0 * self.alpha - 2.0)

    def _c(self, v: np.ndarray) -> np.ndarray:
        alpha = self.alpha
        beta = self._beta()
        return (
            2.0
            + np.sqrt(alpha * alpha * (1.0 - v) ** 2 + beta * beta)
            - alpha * (1.0 - v)
            - beta
        )

    def _f(self, s: np.ndarray) -> np.ndarray:
        alpha = self.alpha
        beta = self._beta()
        return (s / 2.0) * np.sqrt(alpha * alpha * s * s + beta * beta) + (
            beta * beta / (2.0 * alpha)
        ) * np.arcsinh(alpha * s / beta)

    def _ratio(self, x: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.capacity > 0, x / self.capacity, 0.0)

    def _flows(self, x: Any) -> np.ndarray:
        if np.isscalar(x) or (isinstance(x, np.ndarray) and x.ndim == 0):
            return np.full(self.n_links, float(x), dtype="float64")
        return _as_float_array(x, self.n_links, "flows")


@dataclass(frozen=True)
class SpeedFlow:
    """A piecewise-linear speed-against-flow-per-lane volume-delay function.

    The shape a DfT TAG speed-flow curve takes: speed falls linearly
    between flow-per-lane breakpoints, and ``t(x) = length / speed(x /
    lanes)``. Build one from a shared curve table and a
    :class:`~transport_flow_model.Network` with :meth:`from_table`, or
    construct directly with breakpoints already assigned per link.

    Every link's curve is stored as one row of ``flow`` and one row of
    ``speed``, breakpoint values in increasing flow-per-lane order. Curves
    of different lengths (from different DfT road classes, say) are made
    rectangular by :meth:`from_table`, which pads a shorter curve by
    continuing its final segment's slope — exactly the extrapolation
    :meth:`travel_time` already applies beyond a curve's last real
    breakpoint, so the padding is invisible to every method below.

    Below the first breakpoint and at or beyond the last, the nearest
    segment's line is extrapolated rather than held constant — so a curve
    whose first flow-per-lane breakpoint is not zero still has a
    well-defined speed at zero flow.

    **Units are entirely the caller's responsibility and are not checked.**
    ``flow`` must be in the same flow-per-lane-per-hour unit that ``x /
    lanes`` produces from assignment flows, and ``speed`` must be in
    ``length``'s unit per that same hour. A network in vehicles/hour with
    lengths in km and a curve table in mph and vehicles/15-minutes will not
    raise an error; it will silently produce the wrong travel time.

    Attributes
    ----------
    length : numpy.ndarray
        Link length per link, in the same distance unit as ``speed``.
    lanes : numpy.ndarray
        Lane count per link, at least 1. Flow is divided by this to get the
        per-lane flow the curve is defined against.
    flow : numpy.ndarray
        Flow-per-lane breakpoints, shape ``(n_links, n_breakpoints)``,
        strictly increasing along each row.
    speed : numpy.ndarray
        Speed breakpoints, shape ``(n_links, n_breakpoints)``, matching
        ``flow``. Every value must be positive.
    min_speed : float
        A floor speed below which the piecewise-linear curve is clamped, so
        that travel time (``length / speed``) and its integral stay finite
        as flow grows without bound. Must be positive; its unit must match
        ``speed``'s. Below the clamp, ``derivative`` returns zero: marginal
        flow added past the point where a link is already crawling at its
        floor speed adds no marginal *time* in this model, only more delay
        implicitly absorbed by the floor. Defaults to 5.0, an arbitrary
        gridlock-speed placeholder in whatever unit the curve uses — pick a
        value in the caller's own unit, not this default.
    """

    length: np.ndarray
    lanes: np.ndarray
    flow: np.ndarray
    speed: np.ndarray
    min_speed: float = 5.0

    def __post_init__(self) -> None:
        n = len(self.length)
        if self.lanes.shape != (n,):
            raise ValueError(
                f"Expected lanes to have shape ({n},), got {self.lanes.shape}"
            )
        if self.flow.ndim != 2 or self.flow.shape[0] != n:
            raise ValueError(
                f"Expected flow to have shape ({n}, n_breakpoints), got "
                f"{self.flow.shape}"
            )
        if self.speed.shape != self.flow.shape:
            raise ValueError(
                f"Expected speed to have the same shape as flow "
                f"{self.flow.shape}, got {self.speed.shape}"
            )
        if self.flow.shape[1] < 2:
            raise ValueError("Each curve needs at least two breakpoints")
        if not np.all(np.diff(self.flow, axis=1) > 0):
            raise ValueError("flow breakpoints must be strictly increasing")
        if not np.all(self.speed > 0):
            raise ValueError("speed breakpoints must be positive")
        if not np.all(self.lanes >= 1):
            raise ValueError("lanes must be >= 1")
        if not self.min_speed > 0:
            raise ValueError("min_speed must be positive")

    @classmethod
    def from_table(
        cls,
        curves: pa.Table | pa.RecordBatch | pd.DataFrame,
        network: Network,
        *,
        curve_column: str = "link_type",
        min_speed: float = 5.0,
    ) -> SpeedFlow:
        """Build from a tidy curve table and a network's link attributes.

        ``curves`` has one row per breakpoint, with columns ``curve_id``,
        ``flow_per_lane`` and ``speed``; it accepts the same input types
        :class:`~transport_flow_model.Network` builders do (a pyarrow
        ``Table``/``RecordBatch`` or a pandas ``DataFrame``). Each distinct
        ``curve_id`` becomes one curve, sorted by ``flow_per_lane``.

        Each network link is assigned the curve whose ``curve_id`` equals
        the link's ``curve_column`` attribute (default ``"link_type"``, the
        TNTP road-class code). ``length`` and ``lanes`` are read from the
        network's own attributes of those names.

        Raises
        ------
        ValueError
            If a link's ``curve_column`` value does not match any
            ``curve_id`` in ``curves`` — the error names every missing id.
        """
        table = _curves_table(curves)
        required = {"curve_id", "flow_per_lane", "speed"}
        missing_columns = required - set(table.column_names)
        if missing_columns:
            raise ValueError(
                f"Curve table is missing columns: {sorted(missing_columns)}"
            )
        breakpoints: dict[Any, list[tuple[float, float]]] = {}
        for curve_id, flow_per_lane, speed in zip(
            table["curve_id"].to_pylist(),
            table["flow_per_lane"].to_pylist(),
            table["speed"].to_pylist(),
        ):
            breakpoints.setdefault(curve_id, []).append(
                (float(flow_per_lane), float(speed))
            )
        for curve_id, points in breakpoints.items():
            points.sort(key=lambda point: point[0])

        n = network.n_links
        length = _as_float_array(network.attribute("length"), n, "length")
        lanes = _as_float_array(network.attribute("lanes"), n, "lanes")
        curve_ids = network.attribute(curve_column).to_pylist()

        missing_ids = sorted(
            {curve_id for curve_id in curve_ids if curve_id not in breakpoints},
            key=repr,
        )
        if missing_ids:
            raise ValueError(
                f"Curve table has no curve id(s) {missing_ids!r} referenced "
                f"by the network's {curve_column!r} attribute"
            )

        width = max(len(points) for points in breakpoints.values())
        flow = np.empty((n, width), dtype="float64")
        speed_array = np.empty((n, width), dtype="float64")
        padded: dict[Any, tuple[np.ndarray, np.ndarray]] = {}
        for curve_id, points in breakpoints.items():
            padded[curve_id] = _pad_breakpoints(points, width)
        for i, curve_id in enumerate(curve_ids):
            flow[i], speed_array[i] = padded[curve_id]

        return cls(
            length=length,
            lanes=lanes,
            flow=flow,
            speed=speed_array,
            min_speed=min_speed,
        )

    @property
    def n_links(self) -> int:
        return len(self.length)

    def travel_time(self, x: Any) -> np.ndarray:
        """``length / max(speed(x / lanes), min_speed)``."""
        x = self._flows(x)
        q = x / self.lanes
        a, b = self._segment_coefficients(q)
        speed = np.maximum(a + b * q, self.min_speed)
        return self.length / speed

    def integral(self, x: Any) -> np.ndarray:
        """``∫_0^x length / max(speed(w / lanes), min_speed) dw``.

        Computed exactly, segment by segment: whole segments up to the one
        containing ``x / lanes`` contribute their full antiderivative
        (``length * lanes / b * ln(a + b*q)`` where the segment's speed is
        ``a + b*q``, or ``length * lanes * q / a`` where ``b == 0``), the
        segment containing it contributes only its partial term, and any
        part of any segment at or below ``min_speed`` contributes the
        linear ``length * lanes * width / min_speed`` instead — so the
        result stays finite even where flow pushes the raw curve to zero or
        negative speed.
        """
        x = self._flows(x)
        q = x / self.lanes
        n_segments = self.flow.shape[1] - 1
        total = np.zeros(self.n_links)
        for j in range(n_segments):
            lo = np.zeros(self.n_links) if j == 0 else self.flow[:, j]
            hi = (
                np.full(self.n_links, np.inf)
                if j == n_segments - 1
                else self.flow[:, j + 1]
            )
            a, b = self._segment(j)
            total += self._segment_integral(lo, hi, q, a, b)
        return total

    def derivative(self, x: Any) -> np.ndarray:
        """``-length * b / (lanes * speed(x / lanes)^2)``.

        Zero wherever the raw piecewise-linear speed is at or below
        ``min_speed``: past that point, more flow changes nothing (the link
        is already clamped to its floor speed), so there is no marginal
        travel time to report.
        """
        x = self._flows(x)
        q = x / self.lanes
        a, b = self._segment_coefficients(q)
        raw_speed = a + b * q
        speed = np.maximum(raw_speed, self.min_speed)
        clamped = raw_speed <= self.min_speed
        return np.where(clamped, 0.0, -self.length * b / (self.lanes * speed**2))

    def _segment(self, j: int) -> tuple[np.ndarray, np.ndarray]:
        """Affine coefficients ``(a, b)`` of segment ``j``, ``speed = a + b*flow``."""
        flow_lo, flow_hi = self.flow[:, j], self.flow[:, j + 1]
        speed_lo, speed_hi = self.speed[:, j], self.speed[:, j + 1]
        b = (speed_hi - speed_lo) / (flow_hi - flow_lo)
        a = speed_lo - b * flow_lo
        return a, b

    def _segment_index(self, q: np.ndarray) -> np.ndarray:
        """Index of the segment containing ``q``, clipped to extrapolate at both ends."""
        n_segments = self.flow.shape[1] - 1
        interior = np.sum(self.flow[:, 1:] <= q[:, None], axis=1)
        return np.clip(interior, 0, n_segments - 1)

    def _segment_coefficients(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        idx = self._segment_index(q)
        flow_lo = np.take_along_axis(self.flow[:, :-1], idx[:, None], axis=1)[:, 0]
        flow_hi = np.take_along_axis(self.flow[:, 1:], idx[:, None], axis=1)[:, 0]
        speed_lo = np.take_along_axis(self.speed[:, :-1], idx[:, None], axis=1)[:, 0]
        speed_hi = np.take_along_axis(self.speed[:, 1:], idx[:, None], axis=1)[:, 0]
        b = (speed_hi - speed_lo) / (flow_hi - flow_lo)
        a = speed_lo - b * flow_lo
        return a, b

    def _segment_integral(
        self,
        lo: np.ndarray,
        hi: np.ndarray,
        q: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
    ) -> np.ndarray:
        """Contribution of one segment, clipped to ``[lo, min(hi, q)]``, clamp-aware."""
        hi = np.minimum(hi, q)
        width = np.maximum(hi - lo, 0.0)
        min_speed = self.min_speed
        with np.errstate(divide="ignore", invalid="ignore"):
            crossing = np.where(b != 0, (min_speed - a) / np.where(b != 0, b, 1.0), 0.0)
        crossing = np.clip(crossing, lo, hi)
        positive_slope = b > 0
        negative_slope = b < 0
        unclamped_lo = np.where(positive_slope, crossing, lo)
        unclamped_hi = np.where(negative_slope, crossing, hi)
        flat = ~positive_slope & ~negative_slope
        flat_above_floor = flat & (a > min_speed)
        unclamped_lo = np.where(flat, np.where(flat_above_floor, lo, hi), unclamped_lo)
        unclamped_hi = np.where(flat, hi, unclamped_hi)
        unclamped_width = np.maximum(unclamped_hi - unclamped_lo, 0.0)
        clamped_width = np.maximum(width - unclamped_width, 0.0)

        value_lo = np.maximum(a + b * unclamped_lo, min_speed)
        value_hi = np.maximum(a + b * unclamped_hi, min_speed)
        with np.errstate(divide="ignore", invalid="ignore"):
            unclamped_term = np.where(
                b != 0,
                (np.log(value_hi) - np.log(value_lo)) / np.where(b != 0, b, 1.0),
                unclamped_width / np.where(a != 0, a, 1.0),
            )
        factor = self.length * self.lanes
        contribution = factor * (unclamped_term + clamped_width / min_speed)
        return np.where(width > 0, contribution, 0.0)

    def _flows(self, x: Any) -> np.ndarray:
        if np.isscalar(x) or (isinstance(x, np.ndarray) and x.ndim == 0):
            return np.full(self.n_links, float(x), dtype="float64")
        return _as_float_array(x, self.n_links, "flows")


def _pad_breakpoints(
    points: list[tuple[float, float]], width: int
) -> tuple[np.ndarray, np.ndarray]:
    """Extend a curve's breakpoints to ``width`` by continuing its last slope.

    The added points lie exactly on the line the final real segment would
    already be extrapolated along beyond the curve's last breakpoint, so
    padding a shorter curve to match a longer one changes nothing about
    what any :class:`SpeedFlow` method computes for it: the pair of points
    that defines that line's segment is arbitrary, so long as it is two
    distinct points on it.

    The padding step (initially the last real segment's own width) is
    halved as many times as needed to keep every padded speed positive: a
    curve whose last real segment falls steeply enough would otherwise walk
    past zero speed within a single real segment's width of padding, which
    :class:`SpeedFlow` would then (correctly, but unhelpfully) reject as an
    invalid breakpoint.
    """
    if len(points) < 2:
        raise ValueError("Each curve needs at least two breakpoints")
    flows = [point[0] for point in points]
    speeds = [point[1] for point in points]
    step = flows[-1] - flows[-2]
    slope = (speeds[-1] - speeds[-2]) / step
    while len(flows) < width:
        while speeds[-1] + slope * step <= 0.0:
            step /= 2.0
        flows.append(flows[-1] + step)
        speeds.append(speeds[-1] + slope * step)
    return np.array(flows, dtype="float64"), np.array(speeds, dtype="float64")


def _curves_table(curves: pa.Table | pa.RecordBatch | pd.DataFrame) -> pa.Table:
    """Normalize a curve table, accepting what a network table accepts."""
    try:
        return _as_table(curves)
    except TypeError:
        raise TypeError(
            "curves must be a pyarrow Table or RecordBatch, or a pandas DataFrame"
        ) from None


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
