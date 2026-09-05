## Context
Gate for the routing workstream: demonstrate the DEVELOPMENT.md target — national-scale
network (~10M directed edges), ~1M OD pairs, results in minutes — and publish the numbers.

## Task
- Build a GB-scale (or comparable OSM-derived) test network with capacities/speeds;
  synthetic 1M-pair demand via ws3-01 models.
- Run: (a) preprocessing (order + supergraph), (b) customization, (c) full skim + loading;
  record times, memory, thread scaling (1..32 cores).
- Compare against igraph baseline (extrapolated if infeasible) and, externally, against
  cppRouting and AequilibraE path computation on the same instance where formats allow.
- Write up as a docs benchmark page; wire the medium-size subset into nightly CI.

## Acceptance criteria
- End-to-end (customize + 1M-OD skims + loading) in <= 10 min on documented hardware;
  otherwise a written analysis of the gap and follow-up issues.

## References
- Benchmarks in Buchhold et al. (2019) and Schneck & Nökel (2020), TRR 2674(1),
  doi:10.1177/0361198119898455 (42x over Dijkstra on their largest network) — sanity range.
