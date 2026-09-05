## Context
Criticality/risk conclusions rest on scenario *differences* of equilibrium flows; we need
systematic evidence our equilibria are right, at gap levels where differences are stable.

## Task
- Validation suite over TNTP corpus: for each instance x method, assert (a) Beckmann
  objective within tolerance of best-known, (b) link-flow RMSE vs reference below
  threshold scaled to gap, (c) invariance of flows to thread count and warm start.
- Cross-tool check on 2 instances: AequilibraE (bfw) and, if practical, TrafficAssignment.jl.
- Scenario-difference stability test: perturb one link's capacity by 20%, solve both to
  1e-3/1e-4/1e-5 gap, show at which gap the flow-difference field stabilizes (informs
  default tolerances for WS4).

## Acceptance criteria
- CI job (nightly) green across corpus; docs page with tables.

## References
- Boyce, Ralevic-Dekic, Bar-Gera (2004) J. Transp. Eng. 130(1) — convergence-for-
  differences argument.
- github.com/bstabler/TransportationNetworks — best-known solutions.
