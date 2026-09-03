"""Origin-destination demand for the v0 assignment API.

:class:`Demand` stores demand in COO form — parallel ``origin_id``,
``destination_id`` and ``value`` columns — as a :class:`pyarrow.Table`.
Identifiers refer to network nodes; where demand is defined between zones
that differ from network nodes (e.g. split TNTP centroids), optional
``origin_zone`` / ``destination_zone`` columns carry the zone identifiers
alongside the node-level ids.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

#: Canonical demand table columns; ``origin_zone`` and ``destination_zone``
#: are optional extras for zone<->node bookkeeping.
DEMAND_COLUMNS = ("origin_id", "destination_id", "value")


class Demand:
    """Immutable origin-destination demand in COO form."""

    def __init__(
        self,
        od: pa.Table | pd.DataFrame,
        columns: Mapping[str, str] | None = None,
    ):
        table = _as_table(od)
        if columns:
            table = table.rename_columns(
                [columns.get(name, name) for name in table.column_names]
            )
        missing = [c for c in DEMAND_COLUMNS if c not in table.column_names]
        if missing:
            raise ValueError(f"Missing required demand columns: {missing}")
        if not pa.types.is_floating(table["value"].type):
            table = table.set_column(
                table.column_names.index("value"),
                "value",
                pc.cast(table["value"], pa.float64()),
            )
        self._table = table.combine_chunks()

    @classmethod
    def from_dataframe(
        cls,
        od: pd.DataFrame,
        columns: Mapping[str, str] | None = None,
    ) -> Demand:
        """Build from a pandas dataframe of origin/destination/value rows."""
        return cls(od, columns=columns)

    @classmethod
    def from_tntp(cls, instance: Any) -> Demand:
        """Build from a parsed ``TNTPInstance`` (requires its trip matrix).

        Node-level ids reflect any centroid splitting done by the TNTP
        reader; the original zone ids are kept in ``origin_zone`` and
        ``destination_zone`` columns.
        """
        from transport_flow_model import io

        if not isinstance(instance, io.TNTPInstance):
            raise TypeError(
                "from_tntp expects a TNTPInstance; parse files with io.read_tntp"
            )
        if instance.od is None:
            raise ValueError("TNTPInstance has no trip matrix")
        data = instance.od.to_dataframe(copy=True)
        data["origin_zone"] = instance.zone_ids(data["origin_id"])
        data["destination_zone"] = instance.zone_ids(data["destination_id"])
        return cls(data, columns={"flow": "value"})

    def to_table(self) -> pa.Table:
        """The canonical demand table."""
        return self._table

    def to_dataframe(self) -> pd.DataFrame:
        """The demand table as a pandas DataFrame (a fresh copy)."""
        return self._table.to_pandas()

    @property
    def n_pairs(self) -> int:
        return self._table.num_rows

    @property
    def total(self) -> float:
        """Total demand across all OD pairs."""
        return pc.sum(self._table["value"]).as_py() or 0.0

    def __repr__(self) -> str:
        return f"Demand(n_pairs={self.n_pairs}, total={self.total})"


def _as_table(od: pa.Table | pd.DataFrame) -> pa.Table:
    if isinstance(od, pa.Table):
        return od
    if isinstance(od, pa.RecordBatch):
        return pa.Table.from_batches([od])
    if isinstance(od, pd.DataFrame):
        return pa.Table.from_pandas(od, preserve_index=False)
    raise TypeError("od must be a pyarrow Table or RecordBatch, or a pandas DataFrame")
