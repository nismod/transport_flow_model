## Context
CCH splits work into: (1) metric-independent ordering + chordal supergraph, (2) fast
metric customization, (3) queries. Step 1 needs a nested-dissection order from balanced
small separators. Quality of this order dominates customization and query speed.

## Task
- Integrate a partitioner: InertialFlowCutter (best published orders for CCH) via FFI, or
  KaHIP/METIS bindings as fallback; behind a `Partitioner` trait.
- Compute separator-based nested dissection order; build the CCH supergraph (chordal
  completion) and elimination tree.
- Persist order + supergraph (this is the expensive, once-per-topology artifact —
  emphasize in docs that *disruption scenarios do not invalidate it*).

## Acceptance criteria
- Order computed for GB-scale network in < 10 min on a workstation; supergraph size and
  elimination-tree height reported.
- Deterministic given a seed.

## References
- Dibbelt, Strasser, Wagner (2016) "Customizable Contraction Hierarchies", ACM JEA 21.
- Gottesbüren, Hamann, Uhl, Wagner (2019) "Faster and Better Nested Dissection Orders for
  Customizable Contraction Hierarchies" (InertialFlowCutter), Algorithms 12(9).
- Bläsius, Buchhold, Wagner, Zeitz, Zündorf (2025) "Customizable Contraction Hierarchies —
  A Survey", arXiv:2502.10519 — read first; consolidates all engineering advances.
