## Context
Core novelty of the package: disruption scenarios as *sparse metric deltas* on fixed
topology, evaluated via CCH partial re-customization instead of per-scenario Dijkstra
rebuilds. The resilience literature (typically igraph/networkx recomputation per scenario)
has largely not adopted these speedup structures — this is the gap we exploit.

## Task
- `Scenario` type: list of (link_id, attribute, new_value|scale|INF) + metadata (hazard id,
  return period, probability). Damage -> attribute mapping is ws4-03's job.
- Engine: for each scenario — partial_customize (ws1-04), warm-started reassignment
  (sequential screen, or BFW/STAQ from base flows), metric extraction (ws4-02), restore.
- Batching: scenario-level parallelism with per-thread CCH metric buffers (topology/order
  shared, metrics per thread); deterministic outputs; checkpoint/resume for long sweeps.
- Streaming results to partitioned parquet (scenario_id partition).

## Acceptance criteria
- Correctness: scenario results identical to full rebuild on 50 random scenarios.
- Throughput on West Yorkshire: >= 100 single-link scenarios/minute with sequential-screen
  reassignment (record equivalent numbers for BFW/STAQ modes).

## References
- Dibbelt, Strasser, Wagner (2016) CCH — customization as the metric-update primitive.
- Buchhold, Sanders, Wagner (2019) — engineering of fast (re)customization.
- Note: link removal = infinite weight; nested-dissection order (ws1-03) stays valid, so
  preprocessing is never repeated across scenarios.
