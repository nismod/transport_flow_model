## Context
Customization installs a metric (congested times; disrupted links as INFINITY) into the
fixed CCH supergraph via triangle relaxations. This is the operation that makes both
per-iteration assignment updates (WS2) and per-scenario disruption updates (WS4) cheap.
Buchhold, Sanders & Wagner engineered this to tenths of seconds for continental networks.

## Task
- Basic customization: process edges bottom-up by lower-triangle relaxation; then
  (optional, measure) perfect customization for minimal shortcut weights.
- Parallelize: level-based parallelism over the elimination tree / separator decomposition
  (rayon); SIMD-friendly triangle loops.
- *Partial* re-customization: given a sparse set of changed original edges, only reprocess
  affected shortcuts (bottom-up from changed edges). This is the key primitive for ws4-01.
- Support infinite weights (u32::MAX with saturating adds) so removed links need no
  topology change.

## Acceptance criteria
- Full customization: < 1 s on GB-scale network (multithreaded); partial re-customization
  for a 100-link disruption: < 50 ms (targets to be confirmed against hardware, but this
  order of magnitude is what the literature reports).
- Query parity with Dijkstra under 20 random metrics including infinities.

## References
- Dibbelt, Strasser, Wagner (2016), ACM JEA 21 — basic + perfect customization.
- Buchhold, Sanders, Wagner (2019) "Real-time Traffic Assignment Using Engineered
  Customizable Contraction Hierarchies", ACM JEA 24(2), doi:10.1145/3362693 — upper-triangle
  enumeration variant, engineering details.
- CCH Survey arXiv:2502.10519 §customization.
