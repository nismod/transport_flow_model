"""Immutable network for the v0 assignment API.

:class:`Network` holds fixed topology plus per-link attributes as a
:class:`pyarrow.Table` (immutable by construction), with node factorization
and CSR adjacency arrays built lazily for algorithm backends that need them.
Builders accept pandas/geopandas dataframes, pyarrow tables and TNTP
instances.
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import Any, Mapping, NamedTuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

#: Canonical link table columns; any further columns are carried as
#: link attributes.
LINK_COLUMNS = ("edge_from", "edge_to", "edge_id")


class CSR(NamedTuple):
    """Compressed sparse row adjacency over node indices.

    For node index ``u``, its outgoing links are
    ``links[indptr[u]:indptr[u + 1]]`` (positions into the link table) and
    the corresponding head nodes are ``heads[indptr[u]:indptr[u + 1]]``.
    """

    indptr: np.ndarray
    heads: np.ndarray
    links: np.ndarray


class Network:
    """Immutable network topology with link attributes.

    Construct from any tabular link data with ``edge_from``, ``edge_to`` and
    ``edge_id`` columns (or pass ``columns`` to rename source columns to
    those names). Attribute columns such as ``cost`` and ``capacity`` are
    kept alongside and passed through to assignment backends.
    """

    def __init__(
        self,
        links: pa.Table | pd.DataFrame,
        columns: Mapping[str, str] | None = None,
    ):
        table = _as_table(links)
        if columns:
            table = table.rename_columns(
                [columns.get(name, name) for name in table.column_names]
            )
        missing = [c for c in LINK_COLUMNS if c not in table.column_names]
        if missing:
            raise ValueError(f"Missing required link columns: {missing}")
        n_unique = pc.count_distinct(table["edge_id"]).as_py()
        if n_unique != table.num_rows:
            raise ValueError("edge_id values must be unique")
        self._table = table.combine_chunks()

    @classmethod
    def from_dataframe(
        cls,
        links: pd.DataFrame,
        columns: Mapping[str, str] | None = None,
    ) -> Network:
        """Build from a pandas or geopandas dataframe (geometry is dropped)."""
        return cls(links, columns=columns)

    @classmethod
    def from_tntp(cls, net: str | Path | Any) -> Network:
        """Build from a TNTP network file path or a parsed ``TNTPInstance``."""
        instance = _tntp_instance(net)
        return cls(instance.network.to_dataframe(copy=False))

    def to_table(self) -> pa.Table:
        """The canonical link table (topology and attribute columns)."""
        return self._table

    def to_dataframe(self) -> pd.DataFrame:
        """The link table as a pandas DataFrame (a fresh copy)."""
        return self._table.to_pandas()

    @property
    def n_links(self) -> int:
        return self._table.num_rows

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def link_ids(self) -> pa.ChunkedArray:
        """Original link identifiers, in link table order."""
        return self._table["edge_id"]

    @cached_property
    def node_ids(self) -> pa.Array:
        """Unique node identifiers, in order of first appearance."""
        stacked = pa.chunked_array(
            self._table["edge_from"].chunks + self._table["edge_to"].chunks
        )
        return pc.unique(stacked)

    @cached_property
    def _node_indices(self) -> tuple[np.ndarray, np.ndarray]:
        """(tail, head) node indices per link, as int64 arrays."""
        tails = pc.index_in(self._table["edge_from"], self.node_ids)
        heads = pc.index_in(self._table["edge_to"], self.node_ids)
        return (
            _readonly(tails.to_numpy().astype("int64")),
            _readonly(heads.to_numpy().astype("int64")),
        )

    @property
    def link_tails(self) -> np.ndarray:
        """Tail node index per link (positions into :attr:`node_ids`)."""
        return self._node_indices[0]

    @property
    def link_heads(self) -> np.ndarray:
        """Head node index per link (positions into :attr:`node_ids`)."""
        return self._node_indices[1]

    def node_index(self, ids: Any) -> np.ndarray:
        """Map node identifiers to node indices (-1 where not present)."""
        indices = pc.index_in(pa.array(ids, type=self.node_ids.type), self.node_ids)
        return pc.fill_null(indices, -1).to_numpy().astype("int64")

    def link_index(self, ids: Any) -> np.ndarray:
        """Map link identifiers to link table positions (-1 where not present)."""
        link_ids = self.link_ids.combine_chunks()
        indices = pc.index_in(pa.array(ids, type=link_ids.type), link_ids)
        return pc.fill_null(indices, -1).to_numpy().astype("int64")

    @cached_property
    def csr(self) -> CSR:
        """Forward-star (CSR) adjacency arrays over node indices."""
        tails, heads = self._node_indices
        order = np.argsort(tails, kind="stable")
        indptr = np.zeros(self.n_nodes + 1, dtype="int64")
        np.add.at(indptr, tails + 1, 1)
        np.cumsum(indptr, out=indptr)
        return CSR(
            indptr=_readonly(indptr),
            heads=_readonly(heads[order]),
            links=_readonly(order),
        )

    def attribute(self, name: str) -> pa.ChunkedArray:
        """A link attribute column by name."""
        if name not in self._table.column_names:
            raise KeyError(f"Network has no link attribute {name!r}")
        return self._table[name]

    def __repr__(self) -> str:
        return f"Network(n_nodes={self.n_nodes}, n_links={self.n_links})"


def _as_table(links: pa.Table | pd.DataFrame) -> pa.Table:
    if isinstance(links, pa.Table):
        return links
    if isinstance(links, pa.RecordBatch):
        return pa.Table.from_batches([links])
    if isinstance(links, pd.DataFrame):
        links = links.drop(columns=[c for c in ("geometry",) if c in links.columns])
        return pa.Table.from_pandas(pd.DataFrame(links), preserve_index=False)
    raise TypeError(
        "links must be a pyarrow Table or RecordBatch, or a (geo)pandas DataFrame"
    )


def _tntp_instance(net: Any):
    from transport_flow_model import io

    if isinstance(net, io.TNTPInstance):
        return net
    return io.read_tntp(net)


def _readonly(values: np.ndarray) -> np.ndarray:
    values.setflags(write=False)
    return values
