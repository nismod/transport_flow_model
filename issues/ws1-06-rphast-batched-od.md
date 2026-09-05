## Context
Our OD matrices touch a subset of nodes (zone centroids), not all nodes: many-to-many on
|S| sources x |T| targets. RPHAST restricts the downward sweep to the union search space
of the target set — reported orders-of-magnitude faster than alternatives for exactly this
shape. This issue delivers the workhorse `skim_matrix()` and `load_paths()` primitives.

## Task
- RPHAST: extract restricted downward subgraph for target set T (once per T per metric);
  run upward search + restricted sweep per source.
- Many-to-many driver: parallel over sources (rayon), optional tiling of T for cache.
- Path extraction for network loading: unpack shortcut parents into original-edge flow
  increments without materializing per-OD path lists (aggregate directly into per-thread
  flow arrays, then reduce). Provide optional explicit path output for debugging/logit.
- Expose: `skims(sources, targets) -> matrix`, `assign_flows(demand) -> link_flows`.

## Acceptance criteria
- 1M OD pairs on GB-scale network: skims + flow loading within single-digit minutes on a
  16-core workstation (aligned with the DEVELOPMENT.md target); memory bounded and reported.
- Deterministic flows independent of thread count.

## References
- Delling, Goldberg, Werneck (2011) "Faster Batched Shortest Paths in Road Networks"
  (RPHAST), ATMOS/OASIcs.
- Buchhold, Sanders, Wagner (2019) — batched point-to-point adaptation of CCH for
  assignment network loading.
