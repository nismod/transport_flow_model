## Context
Classic CH (Geisberger et al.) is the prerequisite technique for PHAST and a useful
baseline before CCH. Metric-dependent preprocessing; very fast point-to-point queries.

## Task
- CH preprocessing: node ordering via lazy-update priority (edge difference + contracted
  neighbours), witness searches with hop/settle limits, shortcut insertion.
- Bidirectional CH query (stall-on-demand optional, measure whether it pays off).
- Shortcut unpacking to original-edge paths (needed for link flow loading later).
- Serialize/deserialize preprocessed hierarchy (bincode / arrow).

## Acceptance criteria
- Exact parity with Dijkstra on all test networks.
- Query time on Chicago-regional and a GB-scale OSM extract reported in benchmark suite;
  expect ~2-4 orders of magnitude over Dijkstra for point-to-point.

## References
- Geisberger, Sanders, Schultes, Vetter (2012) "Exact Routing in Large Road Networks Using
  Contraction Hierarchies", Transportation Science 46(3).
- RoutingKit for engineering details (order of tie-breaking matters).

## Implementation notes
- Keep preprocessing single-metric for now; CCH (ws1-03/04) is the answer for changing
  metrics — don't over-invest in dynamic classic-CH updates.
