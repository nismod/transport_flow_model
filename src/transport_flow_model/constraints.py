"""Constrain or validate a zone-level OD matrix against aggregate OD data.

Synthetic OD matrices (e.g. from the radiation model) are often estimated at
a fine zoning, while observed flows exist only between larger areas, for
example a census travel-to-work table between districts. These functions:

- aggregate a fine OD matrix to groups (:func:`aggregate_od`);
- scale it so the group-to-group totals match observed targets, while
  keeping the fine pattern within each group pair
  (:func:`scale_to_aggregate`);
- measure how well one OD matrix matches another
  (:func:`common_part_of_commuters`, :func:`od_fit`).

All functions work on dense ``Z x Z`` arrays with one group label per zone.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

#: Methods for :func:`scale_to_aggregate`.
SCALING_METHODS = ("block", "ipf")


def _group_index(groups: Sequence, labels: Sequence | None = None):
    groups = np.asarray(groups)
    if labels is None:
        labels = np.array(sorted(pd.unique(groups)))
    else:
        labels = np.asarray(labels)
    position = {label: i for i, label in enumerate(labels)}
    missing = set(pd.unique(groups)) - set(position)
    if missing:
        raise ValueError(f"Zone groups not in labels: {sorted(missing)[:10]}")
    return np.array([position[g] for g in groups]), labels


def _indicator(index: np.ndarray, n_groups: int) -> np.ndarray:
    """Z x G matrix with a 1 where zone z is in group g."""
    a = np.zeros((len(index), n_groups))
    a[np.arange(len(index)), index] = 1.0
    return a


def aggregate_od(
    flows: np.ndarray,
    origin_groups: Sequence,
    destination_groups: Sequence | None = None,
    labels: Sequence | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sum a ``Z x Z`` OD matrix to a ``G x G`` matrix of group totals.

    Parameters
    ----------
    flows : numpy.ndarray
        Zone-level OD matrix.
    origin_groups, destination_groups : sequence
        Group label of each zone, as origin and as destination. The
        destination groups default to the origin groups.
    labels : sequence, optional
        Group order of the result. Defaults to the sorted unique labels.

    Returns
    -------
    (matrix, labels)

    Examples
    --------
    >>> import numpy as np
    >>> flows = np.array([[0, 1, 2], [3, 0, 4], [5, 6, 0]], dtype=float)
    >>> matrix, labels = aggregate_od(flows, ["a", "a", "b"])
    >>> matrix.tolist(), labels.tolist()
    ([[4.0, 6.0], [11.0, 0.0]], ['a', 'b'])
    """
    destination_groups = origin_groups if destination_groups is None else destination_groups
    o_index, labels = _group_index(origin_groups, labels)
    d_index, _ = _group_index(destination_groups, labels)
    g = len(labels)
    matrix = _indicator(o_index, g).T @ np.asarray(flows, dtype=float) @ _indicator(d_index, g)
    return matrix, labels


def scale_to_aggregate(
    flows: np.ndarray,
    groups: Sequence,
    targets: np.ndarray,
    labels: Sequence | None = None,
    *,
    method: str = "ipf",
    production: np.ndarray | None = None,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """Scale a zone OD matrix to match group-to-group targets.

    Parameters
    ----------
    flows : numpy.ndarray
        ``Z x Z`` zone-level OD matrix (the prior pattern).
    groups : sequence
        Group label of each zone (e.g. its district).
    targets : numpy.ndarray
        ``G x G`` target totals between groups, in the order of ``labels``.
    labels : sequence, optional
        Group order of ``targets``. Defaults to the sorted unique labels.
    method : {"ipf", "block"}
        ``"block"`` multiplies every cell of each group pair by
        ``target / modelled``, so group-pair totals match exactly but the
        zone row totals change. ``"ipf"`` alternates that block scaling with
        rescaling each zone row back to ``production``, so that both are met
        (approximately, to ``tolerance``). Before fitting, the targets are
        rescaled within each origin group so their row total equals the
        production of that group, so the two constraints are consistent.
    production : numpy.ndarray, optional
        Trips produced per zone, for ``"ipf"``. Defaults to the row sums of
        ``flows``.

    Notes
    -----
    Group pairs with a positive target but no modelled flow cannot be
    matched, because scaling cannot create flow. They are left at zero, and
    their target is lost. Check with :func:`aggregate_od` afterwards.

    Examples
    --------
    >>> import numpy as np
    >>> flows = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
    >>> targets = np.array([[1.0, 3.0], [0.5, 0.0]])
    >>> scaled = scale_to_aggregate(flows, ["a", "a", "b"], targets, method="block")
    >>> aggregate_od(scaled, ["a", "a", "b"])[0].tolist()
    [[1.0, 3.0], [0.5, 0.0]]
    """
    if method not in SCALING_METHODS:
        raise ValueError(f"method must be one of {SCALING_METHODS}, got {method!r}")
    flows = np.array(flows, dtype=float)
    index, labels = _group_index(groups, labels)
    g = len(labels)
    targets = np.asarray(targets, dtype=float)
    if targets.shape != (g, g):
        raise ValueError(f"targets must be {g} x {g} for labels {list(labels)}")
    membership = _indicator(index, g)

    def block_scale(f, t):
        modelled = membership.T @ f @ membership
        with np.errstate(divide="ignore", invalid="ignore"):
            factor = np.where(modelled > 0, t / modelled, 0.0)
        return f * factor[np.ix_(index, index)]

    if method == "block":
        return block_scale(flows, targets)

    production = flows.sum(axis=1) if production is None else np.asarray(production, dtype=float)
    group_production = membership.T @ production
    row_targets = targets.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        consistent = targets * np.where(row_targets > 0, group_production / row_targets, 0.0)[:, None]

    result = flows
    for _ in range(max_iterations):
        result = block_scale(result, consistent)
        rows = result.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            result = result * np.where(rows > 0, production / rows, 0.0)[:, None]
        modelled = membership.T @ result @ membership
        gap = np.abs(modelled - consistent).sum() / max(consistent.sum(), 1e-12)
        if gap < tolerance:
            break
    return result


def common_part_of_commuters(modelled: np.ndarray, observed: np.ndarray) -> float:
    """Common part of commuters (CPC, Lenormand et al. 2012), in [0, 1].

    ``CPC = 2 * sum(min(T_model, T_obs)) / (sum(T_model) + sum(T_obs))``. A value
    of 1 means the matrices are identical.

    >>> import numpy as np
    >>> common_part_of_commuters(np.array([[0, 2], [1, 0]]), np.array([[0, 1], [1, 0]]))
    0.8
    """
    modelled = np.asarray(modelled, dtype=float)
    observed = np.asarray(observed, dtype=float)
    total = modelled.sum() + observed.sum()
    return float(2.0 * np.minimum(modelled, observed).sum() / total) if total > 0 else float("nan")


def od_fit(modelled: np.ndarray, observed: np.ndarray, labels: Sequence) -> tuple[dict, pd.DataFrame]:
    """Fit statistics and a long comparison table for two ``G x G`` OD matrices.

    Returns
    -------
    (metrics, table)
        metrics: ``cpc``, ``r2`` (of pair flows), ``rmse``, the share of flow
        that stays within its group (``modelled_internal_share`` and
        ``observed_internal_share``), and totals. table: origin, destination,
        modelled, observed, difference, ratio.
    """
    modelled = np.asarray(modelled, dtype=float)
    observed = np.asarray(observed, dtype=float)
    labels = np.asarray(labels)
    o, d = np.meshgrid(np.arange(len(labels)), np.arange(len(labels)), indexing="ij")
    table = pd.DataFrame(
        {
            "origin": labels[o.ravel()],
            "destination": labels[d.ravel()],
            "modelled": modelled.ravel(),
            "observed": observed.ravel(),
        }
    )
    table["difference"] = table["modelled"] - table["observed"]
    with np.errstate(divide="ignore", invalid="ignore"):
        table["ratio"] = table["modelled"] / table["observed"]
    residual = ((table["modelled"] - table["observed"]) ** 2).sum()
    spread = ((table["observed"] - table["observed"].mean()) ** 2).sum()
    metrics = {
        "cpc": common_part_of_commuters(modelled, observed),
        "r2": float(1 - residual / spread) if spread > 0 else float("nan"),
        "rmse": float(np.sqrt(residual / len(table))),
        "modelled_total": float(modelled.sum()),
        "observed_total": float(observed.sum()),
        "modelled_internal_share": float(np.trace(modelled) / modelled.sum()) if modelled.sum() else float("nan"),
        "observed_internal_share": float(np.trace(observed) / observed.sum()) if observed.sum() else float("nan"),
    }
    return metrics, table
