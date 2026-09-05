## Context
`pip install transport_flow_model` must work without a Rust toolchain, or the Rust core
will gate adoption.

## Task
- maturin-based build; abi3 wheels for Linux (manylinux2014 x86_64 + aarch64), macOS
  (universal2), Windows; sdist with graceful no-rust fallback (ws1-07).
- Release workflow: tag -> build matrix -> test wheels against the validation suite ->
  publish to PyPI (trusted publishing); Zenodo DOI per release (repo already has one).
- Optional conda-forge feedstock (separate follow-up if demand exists).

## Acceptance criteria
- Fresh venv on all three OSes: install + run SiouxFalls assignment from wheel, no
  compiler present.

## References
- https://www.maturin.rs ; PyO3 abi3 docs; cibuildwheel-vs-maturin-action tradeoffs.
