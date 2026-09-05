## Context
With d(equilibrium link flows)/d(OD) available (ws5-01), OD calibration to counts becomes
bilevel optimization solvable with Adam/L-BFGS: min over OD of count misfit + prior
divergence, s.t. flows = UE(OD). Replaces heuristic matrix adjustment; supersedes the
baseline in ws3-02 behind the same interface.

## Task
- Implement `calibrate_od(..., method="implicit-grad")`: loss = weighted least squares on
  counts (or GEH-like robust loss) + KL(od || od_prior); optimizer with positivity via
  softplus/log parameterization; minibatching over counts if large.
- Experiments on 20-city instances: synthetic ground truth (perturb OD, generate counts at
  15% of links, add noise) — report OD recovery vs Spiess baseline; wall time.
- Uncertainty: Hessian-vector products for local posterior approximation (stretch).

## Acceptance criteria
- Beats Spiess baseline on count fit at equal prior-divergence on >= 3 instances; runs on
  Chicago-Sketch scale.

## References
- Cascetta, Nguyen (1988) Transportation Research B — estimation framework/identifiability.
- Computational-graph OD estimation literature (e.g. Wu, Zhou et al.; Ma & Qian) — prior
  art using explicit unrolling; we contribute the implicit-diff variant on top of a fast
  equilibrium solver.
