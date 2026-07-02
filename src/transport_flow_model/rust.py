"""Experimental Python helpers for the Rust transport-flow extension."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pyarrow as pa

try:
    from transport_flow_model import _rust
except ImportError:  # pragma: no cover - depends on optional native build
    _rust = None


def is_available() -> bool:
    """Return whether the optional Rust extension is importable."""
    return _rust is not None


def version() -> str:
    """Return the Rust extension crate version."""
    rust_module = _require_rust()
    return rust_module.version()


def allocate_arrow(
    network: pa.Table | pa.RecordBatch | pd.DataFrame,
    od: pa.Table | pa.RecordBatch | pd.DataFrame,
    *,
    capacity_constrained: bool = False,
    directed: bool = True,
) -> dict[str, pa.Table]:
    """Allocate OD demand through the Rust extension using Arrow IPC streams."""
    rust_module = _require_rust()
    result = rust_module.allocate_ipc(
        _to_ipc_stream(network),
        _to_ipc_stream(od),
        capacity_constrained,
        directed,
    )
    return {name: _from_ipc_stream(payload) for name, payload in result.items()}


def disrupt_arrow(
    network: pa.Table | pa.RecordBatch | pd.DataFrame,
    od_flows: pa.Table | pa.RecordBatch | pd.DataFrame,
    failed_edges: list[str],
    *,
    capacity_constrained: bool = True,
    directed: bool = True,
) -> dict[str, pa.Table]:
    """Reroute failed-edge flows through the Rust extension using Arrow IPC."""
    rust_module = _require_rust()
    result = rust_module.disrupt_ipc(
        _to_ipc_stream(network),
        _to_ipc_stream(od_flows),
        failed_edges,
        capacity_constrained,
        directed,
    )
    return {name: _from_ipc_stream(payload) for name, payload in result.items()}


def _require_rust() -> Any:
    if _rust is None:
        raise RuntimeError(
            "The optional Rust extension is not installed. "
            "Run `pixi run rust-build` before using transport_flow_model.rust."
        )
    return _rust


def _to_table(data: pa.Table | pa.RecordBatch | pd.DataFrame) -> pa.Table:
    if isinstance(data, pa.Table):
        return data
    if isinstance(data, pa.RecordBatch):
        return pa.Table.from_batches([data])
    if isinstance(data, pd.DataFrame):
        return pa.Table.from_pandas(data, preserve_index=False)
    raise TypeError("data must be a pyarrow Table, pyarrow RecordBatch, or pandas DataFrame")


def _to_ipc_stream(data: pa.Table | pa.RecordBatch | pd.DataFrame) -> bytes:
    table = _to_table(data)
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def _from_ipc_stream(data: bytes) -> pa.Table:
    with pa.ipc.open_stream(data) as reader:
        return reader.read_all()
