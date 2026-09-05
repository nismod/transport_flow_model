## Context
The Rust core must be pleasant from Python and add zero copy overhead.

There is no pure-Python fallback and none is planned: the extension is a hard requirement,
and `import transport_flow_model` already reaches `core.py` and therefore `_core`, so a
checkout without `maturin develop` cannot import the package at all. That is recorded in
`docs/adr/0002-arrow-tables-are-the-internal-interchange.md`. Maintaining a second
implementation of every algorithm — and a test suite that runs against both — costs more
than it buys when the wheels cover the platforms we target (ws6-03).

## Task
- PyO3 module `tfm._core`: build CCH from CSR arrays (accept numpy/arrow zero-copy),
  customize(metric), partial_customize(edge_ids, values), skims, assign_flows.
- Long-lived handle objects owning preprocessed structures; expose thread-count control.
- Arrow C data interface (arrow-rs <-> pyarrow) for flow/skim outputs.
- Release the GIL around compute **where it is a local change** — a `py.allow_threads`
  wrapper around a call that already owns its inputs. If a call would need restructuring to
  hand owned data to the Rust side, leave it and note it; the point is not to block this
  issue on making every entry point `Send`.

## Acceptance criteria
- No data copies for inputs/outputs >= 1MB (verify with memory profiling).
- Handles are built once and reused across calls; a scenario or origin loop does not
  re-parse its inputs.
- Where the GIL is released, a second Python thread makes progress during a long call;
  where it is not, the reason is recorded in the code or in ADR-0002.

## Status
Partly done ahead of this issue:
- Arrow C stream interface both ways (`core.py` `_from_ffi_stream`, `arrow_ffi.rs`).
- Long-lived handles: `core.prepare(links) -> PreparedNetwork` and
  `core.prepare_disruption(links, paths) -> PreparedDisruption`, permitted by ADR-0002 as a
  cache of parsed state.
- Batched entry point `core.skim(network, od_pairs)`.

Still open: CCH itself (ws1-02 … ws1-06), thread-count control, GIL release, and the
zero-copy verification. Note that today's readers *do* copy — `read_network_batches` builds
a `Vec<Edge>` — so the >= 1MB criterion is not met and will want a CSR-backed path.

## References
- maturin + abi3 wheels: https://www.maturin.rs ; arrow-rs FFI docs.
- Existing Cargo.toml/PyO3 scaffold in the repo — extend, don't fork.
- `docs/adr/0002-arrow-tables-are-the-internal-interchange.md` — the boundary rules,
  including when an opaque handle is permitted.
