## Context
DEVELOPMENT.md plans OD estimation via proportional downscaling, radiation, maybe gravity.
Implement gravity and radiation behind one interface with proper balancing, so demand
synthesis for data-poor regions (a core IRV use case) is first-class.

## Task
- `estimate_od(zones, productions, attractions, impedance, model=...)`:
  - Gravity: doubly-constrained entropy-maximizing form (Wilson) with exponential and
    power deterrence; impedance = CCH skims (free-flow or congested — document circularity
    and offer one outer feedback loop).
  - Radiation: parameter-free Simini et al. form; finite-sample correction; uses ranked
    opportunities within radius (needs sorted skims — cheap with ws1-06).
  - Furness/IPF balancing to margins with convergence diagnostics.
- Sparsification controls (min flow threshold) to keep OD size manageable at national scale.

## Acceptance criteria
- Reproduces textbook gravity example exactly; radiation reproduces the commuting example
  from Simini et al. supplementary on public data.
- 1M-pair national OD synthesized end-to-end (zones -> skims -> OD) in the pipeline.

## References
- Wilson (1967) "A statistical theory of spatial distribution models", Transportation
  Research 1(3).
- Simini, Gonzalez, Maritan, Barabasi (2012) "A universal model for mobility and migration
  patterns", Nature 484.
- Ortuzar & Willumsen, "Modelling Transport" ch.5 — Furness/IPF.
