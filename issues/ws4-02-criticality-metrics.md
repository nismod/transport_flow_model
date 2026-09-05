## Context
Comparable, well-defined metrics per scenario are the product. Align definitions with the
established vulnerability literature so results are citable.

## Task
Implement per-scenario and aggregated metrics:
- Rerouting cost: change in total (generalized) travel cost for served demand.
- Unmet/isolated demand: OD pairs disconnected or unserved under capacity constraints
  (natural output of STAQ mode, ws2-06).
- Accessibility loss: change in access measures (e.g. jobs reachable within threshold) per
  zone — needs skims per scenario (ws1-06).
- Link criticality ranking: expected consequence of each link's failure; joint metrics for
  multi-link scenarios (marginal + Shapley-style attribution as stretch).
- Aggregations across scenario ensembles: exceedance curves of consequence, per-link
  expected annual consequence.

## Acceptance criteria
- Metrics documented with formulas; unit tests on toy networks with hand-computed values;
  parquet schema versioned.

## References
- Jenelius, Petersen, Mattsson (2006) "Importance and exposure in road network
  vulnerability analysis", Transportation Research A 40(7).
- Taylor (2017) "Vulnerability Analysis for Transportation Networks" (book).
- Mattsson & Jenelius (2015) "Vulnerability and resilience of transport systems",
  Transportation Research A 81 — survey framing.
