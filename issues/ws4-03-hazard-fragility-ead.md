## Context
Connect the scenario engine to climate-hazard analysis: hazard footprints -> exposure ->
fragility -> damage states -> link attribute deltas -> consequences -> expected annual
damages/losses (EAD/EAL). Interfaces should match the existing nismod/OPSIS IRV tooling
(snail for raster-network intersection, snkit for network cleaning, irv workflows).

## Task
- Exposure: intersect link geometries with hazard rasters per return period (delegate to
  snail; accept precomputed exposure tables too).
- Fragility: pluggable curves (damage fraction | failure probability vs intensity);
  Monte Carlo sampling of damage states -> `Scenario` objects (ws4-01), with common random
  numbers across intervention options (for ws5-03 variance reduction).
- Consequence -> risk: integrate consequences over return periods (trapezoidal on
  exceedance probability) for EAD; direct (asset damage, via cost-per-unit tables) +
  indirect (ws4-02 metrics) reported separately.
- Worked example: West Yorkshire + a public flood hazard layer, end-to-end notebook.

## Acceptance criteria
- EAD computation matches a hand-computed 3-return-period example; end-to-end notebook
  runs in CI (reduced size).

## References
- Koks et al. (2019) "A global multi-hazard risk analysis of road and railway
  infrastructure assets", Nature Communications 10 — canonical pipeline shape.
- Oh, Deshmukh, Hastak / HAZUS fragility conventions for transport assets.
- nismod/snail, nismod/snkit, nismod/open-gira — interface targets.
