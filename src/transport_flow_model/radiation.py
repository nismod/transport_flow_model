"""Radiation model for OD estimation using network impedance.

The radiation model (Simini et al. 2012) predicts the probability that a trip
starting in zone ``i`` ends in zone ``j``::

    P_ij = m_i * n_j / ((m_i + s_ij) * (m_i + n_j + s_ij))

where

- ``m_i`` is the origin mass (opportunities at the origin),
- ``n_j`` is the destination mass (opportunities at the destination),
- ``s_ij`` is the destination mass of every other zone ``k`` that is at least
  as close to ``i`` as ``j`` is (``d_ik <= d_ij``, excluding ``i`` and ``j``):
  the *intervening opportunities*.

Origin and destination masses can differ, e.g. employed residents at the
origin and jobs at the destination for journeys to work. They should then be
in comparable units (e.g. jobs scaled to the same total as workers), since
``m_i`` and ``s_ij`` are added together.

Flows are ``T_ij = T_i * P_ij``, where ``T_i`` is the trip production at
``i`` (e.g. employed residents).

The functions :func:`radiation_probabilities` and :func:`radiation_flows`
work on a dense zone-by-zone impedance matrix and are vectorised per origin
(``O(Z^2 log Z)`` for ``Z`` zones). :class:`RadiationModel` computes that
matrix from a :class:`~transport_flow_model.Network` with igraph's Dijkstra.
"""

from __future__ import annotations

from collections.abc import Sequence

import igraph
import numpy as np
import pandas as pd

#: Normalisation options for :func:`radiation_probabilities`.
NORMALISATIONS = ("finite", "row", "none")


def radiation_probabilities(
    impedance: np.ndarray,
    origin_mass: np.ndarray,
    destination_mass: np.ndarray | None = None,
    *,
    max_impedance: float = np.inf,
    normalisation: str = "finite",
    include_self: bool = False,
) -> np.ndarray:
    """Dense ``Z x Z`` matrix of radiation model probabilities ``P_ij``.

    Parameters
    ----------
    impedance : numpy.ndarray
        ``Z x Z`` travel impedance (distance, time or generalised cost) from
        zone ``i`` (row) to zone ``j`` (column). ``inf`` or NaN means
        unreachable. The diagonal is ignored: a zone is never its own
        destination.
    origin_mass : numpy.ndarray
        Length-``Z`` origin mass ``m_i``.
    destination_mass : numpy.ndarray, optional
        Length-``Z`` destination mass ``n_j``. Defaults to ``origin_mass``
        (the classic single-mass model).
    max_impedance : float
        Destinations with impedance above this get zero probability.
    normalisation : {"finite", "row", "none"}
        ``"finite"`` multiplies by ``1 / (1 - m_i / M)`` with ``M`` the total
        destination mass (Simini's finite-size correction). ``"row"`` rescales
        each origin's probabilities to sum to 1 over its reachable
        destinations, so all production is distributed. ``"none"`` applies
        the formula as is. ``"finite"`` assumes a zone is not its own
        destination, so it cannot be combined with ``include_self=True``.
    include_self : bool
        Allow the origin zone as a destination (trips that stay within a
        zone), at the impedance on the diagonal of ``impedance``, which must
        then be finite (e.g. a mean intrazonal travel time). The origin's own
        destination mass is never counted as an intervening opportunity, so
        with no zone closer than the intrazonal impedance,
        ``P_ii = n_i / (m_i + n_i)``. By default (``False``) the diagonal is
        ignored, as in the original model. Requires ``normalisation="row"``
        or ``"none"``: the ``"finite"`` correction is derived for a model
        without trips within a zone, and raises :class:`ValueError` here.

    Returns
    -------
    numpy.ndarray
        ``Z x Z`` array, zero on the diagonal and for unreachable pairs.

    Examples
    --------
    Four zones on a line, one unit of mass each. From zone 0, zone 2 has one
    intervening opportunity (zone 1) and zone 3 has two.

    >>> import numpy as np
    >>> d = np.abs(np.subtract.outer([0, 1, 2, 3], [0, 1, 2, 3])).astype(float)
    >>> p = radiation_probabilities(d, np.ones(4), normalisation="none")
    >>> np.round(p[0], 4).tolist()
    [0.0, 0.5, 0.1667, 0.0833]
    """
    if normalisation not in NORMALISATIONS:
        raise ValueError(
            f"normalisation must be one of {NORMALISATIONS}, got {normalisation!r}"
        )
    if include_self and normalisation == "finite":
        raise ValueError(
            'normalisation="finite" cannot be used with include_self=True: the '
            "finite-size correction 1 / (1 - m_i / M) assumes a zone is not its "
            'own destination; use normalisation="row" or "none"'
        )
    d = np.array(impedance, dtype=float)
    m = np.asarray(origin_mass, dtype=float)
    n = m if destination_mass is None else np.asarray(destination_mass, dtype=float)
    z = d.shape[0]
    if d.shape != (z, z):
        raise ValueError(f"impedance must be square, got shape {d.shape}")
    if m.shape != (z,) or n.shape != (z,):
        raise ValueError(f"masses must have length {z}")
    if (m < 0).any() or (n < 0).any():
        raise ValueError("masses must be non-negative")

    d[np.isnan(d)] = np.inf
    if not include_self:
        np.fill_diagonal(d, np.inf)
    probabilities = np.zeros((z, z))

    for i in range(z):
        row = d[i]
        order = np.argsort(row, kind="stable")
        sorted_d = row[order]
        sorted_n = n[order]
        # Intervening mass: destination mass within radius d_ij (ties
        # included), less j itself and never counting the origin.
        intervening = np.where(order == i, 0.0, sorted_n)
        cumulative = np.cumsum(intervening)
        last_tie = np.searchsorted(sorted_d, sorted_d, side="right") - 1
        s = cumulative[last_tie] - intervening
        reachable = np.isfinite(sorted_d) & (sorted_d <= max_impedance)
        numerator = m[i] * sorted_n
        denominator = (m[i] + s) * (m[i] + sorted_n + s)
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.where(reachable & (denominator > 0), numerator / denominator, 0.0)
        probabilities[i, order] = p

    if normalisation == "finite":
        total = n.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            factor = np.where(total - m > 0, total / (total - m), 0.0)
        probabilities *= factor[:, None]
        np.clip(probabilities, 0.0, 1.0, out=probabilities)
    elif normalisation == "row":
        sums = probabilities.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            probabilities = np.where(sums > 0, probabilities / sums, 0.0)
    return probabilities


def radiation_flows(
    impedance: np.ndarray,
    production: np.ndarray,
    origin_mass: np.ndarray | None = None,
    destination_mass: np.ndarray | None = None,
    *,
    max_impedance: float = np.inf,
    normalisation: str = "row",
    min_flow: float = 0.0,
    include_self: bool = False,
) -> np.ndarray:
    """Dense ``Z x Z`` matrix of flows ``T_ij = T_i * P_ij``.

    Parameters
    ----------
    production : numpy.ndarray
        Length-``Z`` trip production ``T_i`` (e.g. employed residents).
    origin_mass : numpy.ndarray, optional
        ``m_i``; defaults to ``production``.
    destination_mass : numpy.ndarray, optional
        ``n_j``; defaults to ``origin_mass``.
    min_flow : float
        Flows below this are dropped and the dropped total is spread back
        over each origin's remaining destinations (origin totals kept).

    Other parameters are as for :func:`radiation_probabilities`. The default
    ``normalisation="row"`` distributes all production.

    Examples
    --------
    >>> import numpy as np
    >>> d = np.abs(np.subtract.outer([0, 1, 2], [0, 1, 2])).astype(float)
    >>> t = radiation_flows(d, [100, 0, 0], [1, 1, 1], [0, 3, 1])
    >>> t.sum(axis=1).tolist()
    [100.0, 0.0, 0.0]
    """
    production = np.asarray(production, dtype=float)
    origin_mass = production if origin_mass is None else origin_mass
    p = radiation_probabilities(
        impedance,
        origin_mass,
        destination_mass,
        max_impedance=max_impedance,
        normalisation=normalisation,
        include_self=include_self,
    )
    flows = production[:, None] * p
    if min_flow > 0:
        flows = _prune(flows, min_flow)
    return flows


def _prune(flows: np.ndarray, min_flow: float) -> np.ndarray:
    totals = flows.sum(axis=1, keepdims=True)
    kept = np.where(flows >= min_flow, flows, 0.0)
    kept_totals = kept.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(kept_totals > 0, totals / kept_totals, 0.0)
    return kept * scale


def matrix_to_frame(
    matrix: np.ndarray,
    zone_ids: Sequence,
    value_name: str,
    origin_name: str = "origin",
    destination_name: str = "destination",
    min_value: float = 0.0,
) -> pd.DataFrame:
    """Long table of the entries of ``matrix`` greater than ``min_value``."""
    zone_ids = np.asarray(zone_ids)
    o, d = np.nonzero(matrix > min_value)
    return pd.DataFrame(
        {
            origin_name: zone_ids[o],
            destination_name: zone_ids[d],
            value_name: matrix[o, d],
        }
    )


class RadiationModel:
    """Radiation model for origin-destination flow estimation on a network.

    The impedance between zones is the shortest-path ``cost`` between their
    network nodes (see :meth:`impedance_matrix`).
    """

    CAPACITY_EPSILON = 1.0e-9

    def __init__(self, network):
        """Initialize radiation model with a network.

        Parameters
        ----------
        network : Network
            Network object with edge topology and cost attributes. Links with
            no ``cost`` (or a missing value) get cost 1.
        """
        self.network = network

    def impedance_matrix(
        self,
        nodes: Sequence,
        *,
        directed: bool = False,
        chunk_size: int = 256,
    ) -> np.ndarray:
        """Shortest-path cost between every pair of the given network nodes.

        Parameters
        ----------
        nodes : sequence
            Network node ids, one per zone (repeats allowed).
        directed : bool
            Respect link direction. The default (``False``) treats every
            link as two-way, as earlier versions did.
        chunk_size : int
            Number of source nodes per Dijkstra batch, to bound memory.

        Returns
        -------
        numpy.ndarray
            ``len(nodes) x len(nodes)`` matrix; ``inf`` where unreachable.
        """
        links = self.network.to_dataframe()
        node_index = {
            node: i
            for i, node in enumerate(
                pd.unique(pd.concat([links["edge_from"], links["edge_to"]]))
            )
        }
        missing = {node for node in nodes if node not in node_index}
        if missing:
            raise ValueError(f"Nodes in mapping not found in network: {missing}")
        if "cost" in links.columns:
            cost = pd.to_numeric(links["cost"], errors="coerce").fillna(1.0)
        else:
            cost = pd.Series(1.0, index=links.index)
        graph = igraph.Graph(
            n=len(node_index),
            edges=list(
                zip(
                    links["edge_from"].map(node_index), links["edge_to"].map(node_index)
                )
            ),
            directed=directed,
        )
        graph.es["cost"] = cost.to_numpy(dtype=float)
        idx = [node_index[node] for node in nodes]
        unique_idx, inverse = np.unique(idx, return_inverse=True)
        mode = "out" if directed else "all"
        blocks = [
            np.asarray(
                graph.distances(
                    source=unique_idx[start : start + chunk_size].tolist(),
                    target=unique_idx.tolist(),
                    weights="cost",
                    mode=mode,
                ),
                dtype=float,
            )
            for start in range(0, len(unique_idx), chunk_size)
        ]
        unique_matrix = np.vstack(blocks)
        return unique_matrix[np.ix_(inverse, inverse)]

    def _zone_nodes(self, zones, zone_id_column, zone_to_node_mapping, columns):
        """Validate inputs; return zones (sorted by id) and their node ids."""
        if zone_id_column not in zones.columns:
            raise ValueError(f"Column '{zone_id_column}' not found in zones DataFrame")
        for column in columns:
            if column not in zones.columns:
                raise ValueError(f"Column '{column}' not found in zones DataFrame")
        if "zone_id" not in zone_to_node_mapping.columns:
            raise ValueError("zone_to_node_mapping must have 'zone_id' column")
        if "node_id" not in zone_to_node_mapping.columns:
            raise ValueError("zone_to_node_mapping must have 'node_id' column")

        zones = zones.rename(columns={zone_id_column: "zone_id"})
        missing = set(zone_to_node_mapping["zone_id"]) - set(zones["zone_id"])
        if missing:
            raise ValueError(
                f"Zones in mapping not found in zones DataFrame: {missing}"
            )
        zone_to_node = dict(
            zip(zone_to_node_mapping["zone_id"], zone_to_node_mapping["node_id"])
        )
        unmapped = set(zones["zone_id"]) - set(zone_to_node)
        if unmapped:
            raise ValueError(f"Zones with no node in zone_to_node_mapping: {unmapped}")
        zones = zones.sort_values("zone_id", kind="mergesort").reset_index(drop=True)
        if len(zones) < 2:
            raise ValueError("Need at least 2 zones to generate OD matrix")
        return zones, [zone_to_node[zone] for zone in zones["zone_id"]]

    def generate(
        self,
        zones,
        zone_id_column,
        zone_to_node_mapping,
        relevance_column,
        distance_threshold,
        *,
        destination_relevance_column=None,
        normalisation="finite",
        directed=False,
    ):
        """Generate the OD probability table.

        Parameters
        ----------
        zones : pandas.DataFrame
            Zone data with ``zone_id_column`` and the relevance column(s).
        zone_id_column : str
            Column name for zone identifiers in ``zones``.
        zone_to_node_mapping : pandas.DataFrame
            ``zone_id`` and ``node_id`` columns: the network node of each zone.
        relevance_column : str
            Origin mass ``m_i`` (and destination mass, unless
            ``destination_relevance_column`` is given).
        distance_threshold : float
            Maximum network impedance from origin to destination.
        destination_relevance_column : str, optional
            Destination mass ``n_j`` (e.g. jobs), if different from the origin
            mass. Intervening opportunities ``s_ij`` are summed over this mass.
        normalisation : {"finite", "row", "none"}
            See :func:`radiation_probabilities`.
        directed : bool
            Respect link direction when computing impedance.

        Returns
        -------
        pandas.DataFrame
            Columns ``[origin, destination, probability]`` (zone ids), for
            pairs with probability above ``CAPACITY_EPSILON``.
        """
        destination_column = destination_relevance_column or relevance_column
        zones, nodes = self._zone_nodes(
            zones,
            zone_id_column,
            zone_to_node_mapping,
            [relevance_column, destination_column],
        )
        impedance = self.impedance_matrix(nodes, directed=directed)
        p = radiation_probabilities(
            impedance,
            zones[relevance_column].to_numpy(dtype=float),
            zones[destination_column].to_numpy(dtype=float),
            max_impedance=distance_threshold,
            normalisation=normalisation,
        )
        result = matrix_to_frame(
            p, zones["zone_id"], "probability", min_value=self.CAPACITY_EPSILON
        )
        return result.reset_index(drop=True)

    def flows(
        self,
        zones,
        zone_id_column,
        zone_to_node_mapping,
        production_column,
        destination_mass_column,
        max_impedance,
        *,
        origin_mass_column=None,
        normalisation="row",
        min_flow=0.0,
        directed=False,
    ):
        """Generate OD flows ready for :class:`~transport_flow_model.Demand`.

        Parameters
        ----------
        production_column : str
            Trips produced per zone ``T_i`` (e.g. employed residents).
        destination_mass_column : str
            Destination mass ``n_j`` (e.g. jobs).
        max_impedance : float
            Maximum network impedance from origin to destination.
        origin_mass_column : str, optional
            Origin mass ``m_i``; defaults to ``production_column``.
        normalisation, min_flow
            See :func:`radiation_flows`.

        Returns
        -------
        pandas.DataFrame
            Columns ``[origin_id, destination_id, value]`` with **zone** ids,
            and ``origin_node_id`` / ``destination_node_id`` for assignment.
        """
        origin_mass_column = origin_mass_column or production_column
        zones, nodes = self._zone_nodes(
            zones,
            zone_id_column,
            zone_to_node_mapping,
            [production_column, origin_mass_column, destination_mass_column],
        )
        impedance = self.impedance_matrix(nodes, directed=directed)
        t = radiation_flows(
            impedance,
            zones[production_column].to_numpy(dtype=float),
            zones[origin_mass_column].to_numpy(dtype=float),
            zones[destination_mass_column].to_numpy(dtype=float),
            max_impedance=max_impedance,
            normalisation=normalisation,
            min_flow=min_flow,
        )
        result = matrix_to_frame(
            t, zones["zone_id"], "value", "origin_id", "destination_id"
        )
        node_of = dict(zip(zones["zone_id"], nodes))
        result["origin_node_id"] = result["origin_id"].map(node_of)
        result["destination_node_id"] = result["destination_id"].map(node_of)
        return result
