## Context
Gate for the disruption workstream: ~10k multi-link hazard scenarios on the national-scale
network in an overnight run or less, with equilibrium-quality reassignment on the subset
of scenarios that matter.

## Task
- Two-stage sweep: (1) cheap screen of all scenarios (partial re-customization +
  sequential/AON consequence estimate), (2) full BFW or STAQ reassignment (warm-started)
  on the top-K consequential scenarios; document the screen's ranking fidelity (compare
  stage-1 vs stage-2 rankings on a sample).
- Cluster-friendly execution: process-level sharding of scenario sets; merge parquet.
- Publish throughput/quality tradeoff table; add representative subset to nightly CI.

## Acceptance criteria
- 10k scenarios, GB-scale network, <= 12h on documented hardware (or written gap analysis
  + follow-ups); stage-1 vs stage-2 Kendall tau reported.

## References
- Warm-start behavior: Buchhold et al. (2019) observe few iterations to re-converge after
  small metric changes — quantify for our disruption deltas.
