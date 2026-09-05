## Context
Adoption requires an end-to-end narrative: zones -> OD estimation -> assignment ->
disruption -> criticality -> intervention ranking, on the West Yorkshire dataset.

## Task
- 4 executable notebooks (nbsphinx/myst in docs, run in CI at reduced size):
  1. Build a network from OSM/GeoDataFrame; skims with the Rust core.
  2. Estimate OD (gravity/radiation) and assign (BFW vs STAQ); interpret gap/flows.
  3. Disruption sweep + criticality maps (geopandas/folium outputs).
  4. Hazard-driven risk (fragility -> EAD) and intervention comparison.
- Each notebook ends with "what to read": the references from the relevant issues.

## Acceptance criteria
- `pixi run docs` builds all; total CI runtime < 15 min at reduced size.
