# ADR-0002: Arrow tables are the internal interchange

- **Status:** Accepted
- **Date:** 2026-08-27 (amended 2026-08-27: prepared handles)

## Context

The package began as pandas dataframes end to end. Performance goals since then
(macro-scale networks, roughly 10M edges and 1M OD pairs, in minutes) moved the
hot paths — shortest paths, allocation, disruption — into a Rust extension
built with PyO3 and maturin, exposed as `transport_flow_model._core`.

That creates a boundary, and boundaries need a data representation. The one in
use is Apache Arrow: `Network` and `Demand` hold `pyarrow.Table`s,
`AssignmentResult` and `ScenarioResult` return them, and the Rust side speaks
`arrow-rs` natively.

Two properties of the current code make this a decision rather than an
accident, and neither was written down:

1. `transport_flow_model.core` is the **only** module that imports `_core`. No
   other module in `src/`, `tests/` or `scripts/` touches the extension
   directly.
2. Data crosses via the **Arrow C stream interface**, not Arrow IPC. Rust
   returns PyCapsule-wrapped `ArrowArrayStream`s, which `core._from_ffi_stream`
   imports with `pa.RecordBatchReader._import_from_c_capsule(...).read_all()`.
   There is no serialization step and no copy of the buffers.

(Note for readers of older docs: `docs/source/development.rst` described this
as "Arrow IPC streams" through `core.allocate_arrow` / `core.disrupt_arrow`.
Both statements are out of date; see `core.py` for the current wrappers.)

A later measurement added a third property the record has to account for.
Every call across the boundary parses its Arrow inputs, interns identifiers and
rebuilds the graph before doing any work, and that is *most* of what a call
costs: for `core.shortest_paths_from` on chicago-sketch, 85% of the call is
parse and graph construction and 15% is the actual search. Callers that loop —
`convergence.relative_gap` over origins, `disruption.disrupt` over scenarios —
pay it every iteration. A strict reading of "Arrow tables are the interchange"
would forbid the obvious fix, which is to hold the parsed form.

## Decision

**`pyarrow.Table` is the interchange format between modules and across the
Python/Rust boundary. `transport_flow_model.core` is the only module that may
import `_core`.**

- Public data classes wrap Arrow tables (`Network.to_table()`,
  `Demand.to_table()`) and results are Arrow tables.
- Column operations use `pyarrow.compute`, or NumPy on zero-copy views where
  an array is the natural shape (as `convergence.link_costs` does).
- pandas remains supported at the *edges*: `Network.from_dataframe`,
  `Demand.from_dataframe`, `to_dataframe()`, the readers in `io.py`, the
  dataset registry, and the deprecated `model.py`. `core._to_table` also
  accepts a `DataFrame` for convenience. Those are conversion points, not
  processing.
- New extension entry points are added as thin wrappers in `core.py` following
  the existing shape: normalize inputs with `_to_table`, call into `_core`,
  import the returned capsule(s) with `_from_ffi_stream`. A wrapper calls
  either a `_core.<name>_ffi` function or a method on one of the handles
  below.

Parsed state may be cached behind an opaque handle, under three conditions:

- it is **constructed from Arrow** (`core.prepare(links)`),
- **every method returns Arrow**, and
- it is **never the only way to reach a capability** — each method has a
  module-level counterpart taking tables, implemented as `prepare(...)` plus
  one call.

So a handle is a cache, not a second interchange format. Data still crosses as
Arrow; what the handle saves is re-deriving from it. Code that does one call
should keep using the free functions.

## Consequences

For contributors:

- **No pandas in hot paths.** Converting to a `DataFrame` inside a loop, or to
  drive a per-row operation, defeats the reason the boundary exists. If a
  computation is easier to express in pandas, do it at the edge, once.
- **No direct `_core` imports outside `core.py`.** Anything that needs the
  extension goes through a wrapper. This keeps the FFI surface auditable, keeps
  the extension optional to reason about, and means a change to the Rust
  signature has exactly one Python call site to update. `grep -rn "_core"
  src/` should return hits in `core.py` only.
- **A shared handle must not accumulate state between calls.** Demand and path
  tables may name nodes the network does not have; those ids are interned so
  results can be labelled with them. `PreparedNetwork::scoped` rolls each
  call's additions back, at a cost proportional to the number of unknown ids
  rather than the size of the network. `test_prepared_network_does_not_leak_unknown_demand_ids`
  fails without it.
- Adding a Rust function is a two-sided change — `#[pyfunction]` plus
  `arrow_ffi` conversion in `core/src/`, and a wrapper in `core.py` — so it is
  reviewable as one diff.
- The extension is **not optional at runtime**: `import transport_flow_model`
  reaches `core.py` and therefore `_core`, so a checkout without
  `maturin develop` cannot import the package at all.
- Zero-copy is only zero-copy if it stays in Arrow. `.to_pandas()` on a large
  result table copies it.
- Other language wrappers can target the same Arrow schemas without depending
  on Python dataframe internals.

## Alternatives considered

- **Arrow IPC (serialized buffers) across the boundary.** Language-agnostic in
  the same way, and easier to debug, but it serializes and copies on every
  call. The C stream interface gets the same schema-level portability without
  the copy.
- **pandas as the interchange, converting only at the FFI call.** Would keep
  the older code simpler, but it puts a copy and a dtype-inference step on
  every crossing, and pandas dtypes (object columns, `NaN`-as-missing) do not
  map cleanly onto what the Rust side needs.
- **Extension functions imported directly wherever needed.** Fewer layers, but
  the FFI surface would be spread across the codebase and every Rust signature
  change would ripple.
- **Stateless batched entry points instead of a handle** (pass every origin or
  every scenario in one call). This keeps the interchange rule untouched and is
  the right shape wherever the batch is natural, so it is preferred where it
  fits. It does not cover the disruption loop, where each scenario's result is
  consumed before the next is chosen, so a handle is needed as well. The two
  are complementary, not alternatives.
