## Context
The program goal is selecting portfolios of resilience interventions. With gradients of
expected annual loss w.r.t. continuous intervention parameters (capacity added, fragility
shift from raising/protecting an asset), portfolio design becomes differentiable
optimization under a budget — a genuinely novel capability at this scale.

## Task
- Parameterize candidate interventions theta (per-asset protection level in [0,1] mapping
  to fragility-curve shift and/or capacity change; cost model c(theta)).
- Objective: EAL(theta) estimated over the hazard scenario ensemble (ws4-03) with common
  random numbers; gradient via ws5-01 pathwise derivatives through (smoothed) assignment
  per scenario; penalty/projection for budget.
- Optimize on West Yorkshire case study; compare against (a) greedy ranking by ws4-02 link
  criticality, (b) genetic algorithm baseline; report EAL-vs-budget frontiers.
- Discuss and handle discreteness: relax-and-round for binary build/no-build; report
  integrality gap empirically.

## Acceptance criteria
- Case-study notebook: differentiable portfolio matches or beats greedy at equal budget on
  >= 3 budget levels; methodology writeup suitable to seed a paper.

## References
- Faturechi, Miller-Hooks (2015) "Measuring the performance of transportation
  infrastructure systems in disasters", J. Infrastructure Systems 21(1) — decision framing.
- Koks et al. (2019) Nat. Comms 10 — adaptation prioritization context.
- ws5-01 references for the differentiation machinery.
