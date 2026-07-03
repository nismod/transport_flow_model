"""Core wrappers around extension module."""

from __future__ import annotations


import pandas as pd
import pyarrow as pa

import transport_flow_model._core as _core


def version() -> str:
    """Return the extension version."""
    return _core.version()


def allocate(
    network: pa.Table | pa.RecordBatch | pd.DataFrame,
    od: pa.Table | pa.RecordBatch | pd.DataFrame,
    *,
    capacity_constrained: bool = False,
    directed: bool = True,
) -> dict[str, pa.Table]:
    """Allocate OD demand"""
    result = _core.allocate_ffi(
        _to_table(network),
        _to_table(od),
        capacity_constrained,
        directed,
    )
    return {name: _from_ffi_stream(payload) for name, payload in result.items()}


def disrupt(
    network: pa.Table | pa.RecordBatch | pd.DataFrame,
    od_flows: pa.Table | pa.RecordBatch | pd.DataFrame,
    failed_edges: list[int],
    *,
    capacity_constrained: bool = True,
    directed: bool = True,
) -> dict[str, pa.Table]:
    """Reroute failed-edge flows"""
    result = _core.disrupt_ffi(
        _to_table(network),
        _to_table(od_flows),
        failed_edges,
        capacity_constrained,
        directed,
    )
    return {name: _from_ffi_stream(payload) for name, payload in result.items()}


def shortest_paths_from(
    network: pa.Table | pa.RecordBatch | pd.DataFrame,
    origin: int,
    *,
    directed: bool = True,
) -> pa.Table:
    """Return (node_id: u64, cost: f64) table for all nodes reachable from origin."""
    result = _core.shortest_paths_from_ffi(_to_table(network), origin, directed)
    return _from_ffi_stream(result)


def _to_table(data: pa.Table | pa.RecordBatch | pd.DataFrame) -> pa.Table:
    if isinstance(data, pa.Table):
        return data
    if isinstance(data, pa.RecordBatch):
        return pa.Table.from_batches([data])
    if isinstance(data, pd.DataFrame):
        return pa.Table.from_pandas(data, preserve_index=False)
    raise TypeError(
        "data must be a pyarrow Table, pyarrow RecordBatch, or pandas DataFrame"
    )


def _from_ffi_stream(data: object) -> pa.Table:
    reader = pa.RecordBatchReader._import_from_c_capsule(data)
    return reader.read_all()
