## Context
The Rust core must be pleasant from Python and add zero copy overhead, and the package
must keep working (slower) without the compiled extension.

## Task
- PyO3 module `tfm._core`: build CCH from CSR arrays (accept numpy/arrow zero-copy),
  customize(metric), partial_customize(edge_ids, values), skims, assign_flows.
- Long-lived handle objects owning preprocessed structures; release the GIL around all
  compute; expose thread-count control.
- Arrow C data interface (arrow-rs <-> pyarrow) for flow/skim outputs.
- Pure-Python fallback path (igraph/scipy) selected automatically; single test suite runs
  against both backends.

## Acceptance criteria
- `pip install` from sdist works without Rust only losing speed, not features (document
  which sizes are infeasible in fallback).
- No data copies for inputs/outputs >= 1MB (verify with memory profiling).

## References
- maturin + abi3 wheels: https://www.maturin.rs ; arrow-rs FFI docs.
- Existing Cargo.toml/PyO3 scaffold in the repo — extend, don't fork.
