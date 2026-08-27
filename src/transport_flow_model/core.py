"""Core wrappers around extension module."""

from __future__ import annotations


import pandas as pd
import pyarrow as pa

import transport_flow_model._core as _core


def version() -> str:
    """Return the extension version."""
    return _core.version()


class PreparedNetwork:
    """A network parsed once, for callers that make many calls against it.

    Each of the module-level functions parses its link table, interns
    identifiers and rebuilds the graph before doing any work, and that is
    most of what a call costs. Loops over origins or over disruption
    scenarios pay it every iteration; preparing the network once and
    calling methods on the result pays it once::

        prepared = core.prepare(links)
        for scenario in scenarios:
            prepared.disrupt(paths, scenario.removed_links)

    The prepared network is a cache of parsed inputs, not a new
    interchange format: it is built from an Arrow table and every method
    returns Arrow tables. Anything reachable through it is also reachable
    through the module-level functions.
    """

    __slots__ = ("_inner",)

    def __init__(self, network: pa.Table | pa.RecordBatch | pd.DataFrame):
        self._inner = _core.PreparedNetwork(_to_table(network))

    @property
    def n_links(self) -> int:
        return self._inner.n_links

    @property
    def n_nodes(self) -> int:
        """Nodes reached by the network's own links."""
        return self._inner.n_nodes

    def allocate(
        self,
        od: pa.Table | pa.RecordBatch | pd.DataFrame,
        *,
        capacity_constrained: bool = False,
        directed: bool = True,
    ) -> dict[str, pa.Table]:
        """Allocate OD demand; see :func:`allocate`."""
        result = self._inner.allocate(
            _to_table(od),
            capacity_constrained,
            directed,
        )
        return {name: _from_ffi_stream(payload) for name, payload in result.items()}

    def skim(
        self,
        od_pairs: pa.Table | pa.RecordBatch | pd.DataFrame,
        *,
        directed: bool = True,
    ) -> pa.Table:
        """Least-cost travel time per OD pair; see :func:`skim`."""
        result = self._inner.skim(_to_table(od_pairs), directed)
        return _from_ffi_stream(result)

    def disrupt(
        self,
        od_flows: pa.Table | pa.RecordBatch | pd.DataFrame,
        failed_edges: list[object],
        *,
        capacity_constrained: bool = True,
        directed: bool = True,
    ) -> dict[str, pa.Table]:
        """Reroute failed-edge flows; see :func:`disrupt`."""
        result = self._inner.disrupt(
            _to_table(od_flows),
            failed_edges,
            capacity_constrained,
            directed,
        )
        return {name: _from_ffi_stream(payload) for name, payload in result.items()}

    def __repr__(self) -> str:
        return f"PreparedNetwork(n_links={self.n_links}, n_nodes={self.n_nodes})"


class PreparedDisruption:
    """A network and a baseline path set, both parsed once, for scenario runs.

    Reading the path table is the bulk of a scenario's cost — it is
    typically far larger than the link table — and none of it depends on
    which links fail. Preparing once turns a scenario into a scan over
    already-parsed rows::

        prepared = core.prepare_disruption(links, base_paths)
        for scenario in scenarios:
            prepared.scenario(scenario.removed_links)
    """

    __slots__ = ("_inner",)

    def __init__(
        self,
        network: pa.Table | pa.RecordBatch | pd.DataFrame,
        od_flows: pa.Table | pa.RecordBatch | pd.DataFrame,
    ):
        self._inner = _core.PreparedDisruption(_to_table(network), _to_table(od_flows))

    def scenario(
        self,
        failed_edges: list[object],
        *,
        capacity_constrained: bool = True,
        directed: bool = True,
    ) -> dict[str, pa.Table]:
        """Reroute the baseline flows using any of ``failed_edges``."""
        result = self._inner.scenario(
            failed_edges,
            capacity_constrained,
            directed,
        )
        return {name: _from_ffi_stream(payload) for name, payload in result.items()}


def prepare(network: pa.Table | pa.RecordBatch | pd.DataFrame) -> PreparedNetwork:
    """Parse a link table once for reuse across many calls."""
    return PreparedNetwork(network)


def prepare_disruption(
    network: pa.Table | pa.RecordBatch | pd.DataFrame,
    od_flows: pa.Table | pa.RecordBatch | pd.DataFrame,
) -> PreparedDisruption:
    """Parse a network and baseline paths once for a run of scenarios."""
    return PreparedDisruption(network, od_flows)


def allocate(
    network: pa.Table | pa.RecordBatch | pd.DataFrame,
    od: pa.Table | pa.RecordBatch | pd.DataFrame,
    *,
    capacity_constrained: bool = False,
    directed: bool = True,
) -> dict[str, pa.Table]:
    """Allocate OD demand"""
    return prepare(network).allocate(
        od,
        capacity_constrained=capacity_constrained,
        directed=directed,
    )


def disrupt(
    network: pa.Table | pa.RecordBatch | pd.DataFrame,
    od_flows: pa.Table | pa.RecordBatch | pd.DataFrame,
    failed_edges: list[object],
    *,
    capacity_constrained: bool = True,
    directed: bool = True,
) -> dict[str, pa.Table]:
    """Reroute failed-edge flows"""
    return prepare_disruption(network, od_flows).scenario(
        failed_edges,
        capacity_constrained=capacity_constrained,
        directed=directed,
    )


def skim(
    network: pa.Table | pa.RecordBatch | pd.DataFrame,
    od_pairs: pa.Table | pa.RecordBatch | pd.DataFrame,
    *,
    directed: bool = True,
) -> pa.Table:
    """Least-cost travel time for each OD pair, in input order.

    ``od_pairs`` needs ``origin_id`` and ``destination_id`` columns; any
    others (such as a demand ``value``) are ignored. Returns a table of
    ``origin_id``, ``destination_id`` and ``cost``, with a null ``cost``
    where the destination cannot be reached.

    One shortest-path tree is built per distinct origin, so asking for many
    pairs at once costs one search per origin rather than one per pair.
    """
    return prepare(network).skim(od_pairs, directed=directed)


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
