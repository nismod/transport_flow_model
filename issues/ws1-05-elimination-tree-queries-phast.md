## Context
With a customized CCH we get (a) point-to-point queries via elimination-tree search (no
priority queue at all) and (b) PHAST-style one-to-all: an upward search from the source
followed by a linear top-down sweep over the hierarchy. One-to-all sweeps are what make
full OD-matrix skims and network loading fast, and they vectorize/parallelize well
(Delling et al. report GPU suitability).

## Task
- Elimination-tree point-to-point query with path unpacking.
- PHAST one-to-all: level-ordered downward edge array laid out contiguously; process
  levels in reverse with tight loops; multi-source batching (compute k sources per sweep
  using SIMD lanes — "batched PHAST").
- Benchmark against Dijkstra one-to-all on GB-scale network; report per-source amortized
  time single-threaded and multi-threaded.

## Acceptance criteria
- Exact parity with Dijkstra one-to-all.
- >= 20x speedup per source vs Dijkstra on GB-scale (literature suggests much more with
  batching; record what we achieve and why).

## References
- Delling, Goldberg, Nowatzyk, Werneck (2013) "PHAST: Hardware-accelerated shortest path
  trees", JPDC 73(7) (orig. IPDPS 2011).
- CCH Survey arXiv:2502.10519 — elimination tree queries.
- Strasser, Zeitz "Using Incremental Many-to-One Queries to Build a Fast and Tight
  Heuristic for A* in Road Networks", ACM JEA, doi:10.1145/3571282 (Lazy RPHAST) — for the
  incremental many-to-one variant worth having for ad-hoc OD subsets.
