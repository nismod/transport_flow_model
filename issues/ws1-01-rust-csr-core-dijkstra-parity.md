## Context
igraph gives us plain Dijkstra with no preprocessing; it is the bottleneck against the
~10M-edge / ~1M-OD target. First step of the Rust routing core: a cache-friendly graph
representation and a reference Dijkstra we can test everything else against.

## Task
- Rust crate `tfm-core` (extend existing PyO3 crate): CSR adjacency (`first_out: Vec<u32>`,
  `head: Vec<u32>`, `weight: Vec<f64>` or `u32` fixed-point — decide and document), node id
  remapping table to/from user ids.
- Dijkstra with 4-ary heap; one-to-all and one-to-many variants; path unpacking.
- Property tests: random graphs, compare distances vs petgraph and (via Python tests) igraph.
- Micro-benchmarks (Criterion) on West Yorkshire and Chicago-regional.

## Acceptance criteria
- Exact distance parity with igraph on 1000 random OD pairs per test network.
- Documented decision on weight type (recommend u32 deciseconds fixed-point: faster,
  deterministic, matches RoutingKit practice; keep f64 behind a feature flag).

## References
- RoutingKit (C++ reference implementations of CH/CCH/PHAST): https://github.com/RoutingKit/RoutingKit
- Bast et al. (2016) "Route Planning in Transportation Networks", LNCS 9220 — survey/overview.
