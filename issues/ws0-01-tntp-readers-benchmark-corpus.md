## Context
The repo currently validates against a single West Yorkshire dataset. To validate equilibrium
assignment (WS2) and benchmark routing (WS1) we need instances with *published reference
solutions*. The de-facto standard is the TNTP format used by the Transportation Networks
for Research repository (SiouxFalls ... Chicago-regional) and the recent 20-city US traffic
assignment benchmark dataset already noted in DEVELOPMENT.md.

## Task
- Implement readers for TNTP `_net.tntp` (links: init_node, term_node, capacity, length,
  free_flow_time, b, power, speed, toll, link_type) and `_trips.tntp` (OD matrix) files,
  returning the package's `Network` / `Demand` objects.
- Handle 1-based node indexing, zone-only origins, and the "first thru node" convention
  (centroid connectors must not be used as through-paths).
- Add a `datasets` module that fetches/caches SiouxFalls, Anaheim, Barcelona,
  Chicago-Sketch and the 20-city corpus (with checksums; keep large files out of git).
- Store the published best-known equilibrium link flows / objective values alongside, for
  use as regression fixtures in WS2.

## Acceptance criteria
- `tfm.io.read_tntp(net, trips)` round-trips SiouxFalls; totals match published demand.
- Fixtures available in CI (small instances vendored, large ones cached).
- Docs page listing datasets, licenses, provenance.

## References
- Transportation Networks for Research: https://github.com/bstabler/TransportationNetworks
  (TNTP format spec in repo README).
- 20-city traffic assignment benchmark dataset (see DEVELOPMENT.md literature list).
- AequilibraE TNTP examples for cross-checking parsing: https://aequilibrae.com

## Implementation notes
- Parse with pandas `read_csv(sep=r"\s+", comment="~")`; strip trailing `;`.
- Keep an internal canonical schema: `from_id:int64, to_id:int64, free_flow_cost:f64,
  capacity:f64, alpha:f64, beta:f64` (+ optional geometry for GIS-sourced networks) so
  GIS (geopandas) and TNTP paths converge on one representation.
