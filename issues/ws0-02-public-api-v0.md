## Context
`flow_allocation.py` / `flow_disruptions.py` are script-shaped. Before the routing and
assignment backends multiply (sequential heuristic, MSA, FW/BFW, STAQ) we need a stable
API so backends are swappable and results comparable.

## Task
Design and implement a small, frozen v0 API, e.g.:
- `Network` (immutable topology + link attributes; CSR arrays internally; builders from
  geopandas GeoDataFrame and TNTP).
- `Demand` (OD in COO form: origin_id, destination_id, value; zone<->node mapping).
- `assign(network, demand, method="sequential"|"msa"|"fw"|"bfw"|"staq", **opts) -> AssignmentResult`
  with `link_flows`, `skims`, `gap_history`, `paths` (optional), provenance metadata.
- `disrupt(network, scenarios, demand, ...) -> DisruptionResults` (scenario = sparse set of
  link attribute deltas; see ws4-01).
- Config: keep JSON configs as a thin layer that maps 1:1 onto the API (pydantic schema).

## Acceptance criteria
- Existing scripts reimplemented as ~20-line drivers over the API; identical outputs on
  West Yorkshire benchmark (bit-for-bit or documented diffs).
- Semver adopted; CHANGELOG started; deprecation policy documented.
- API reference generated in Sphinx.

## Implementation notes
- Results as pyarrow tables from day one (matches existing pyarrow dependency, and makes
  the Rust zero-copy path in ws1-07 natural).
- Every `AssignmentResult` carries: method, iterations, relative gap, wall time, seed,
  package + rust-core versions — needed for the benchmarking harness (ws0-03) and for
  reproducible criticality rankings.
