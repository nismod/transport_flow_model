## Context
Method of Successive Averages: simplest convergent user-equilibrium algorithm
(all-or-nothing loading + step 1/k averaging). Slow near the optimum but trivial to verify
— establishes the equilibrium plumbing (loop, gap metric, warm starts) end-to-end.

## Task
- `assign(..., method="msa")`: per iteration — customize CCH with current times,
  all-or-nothing load via ws1-06, average flows, compute relative gap (ws0-03).
- Convergence controls: max iters, target gap, wall-clock budget; gap trajectory in result.
- Validate on SiouxFalls/Anaheim vs published equilibrium link flows (loose tolerance;
  MSA converges slowly — assert gap decreases and flows approach reference).

## Acceptance criteria
- Reaches 1e-3 relative gap on SiouxFalls; result object fully populated.

## References
- Sheffi (1985) "Urban Transportation Networks" ch.5 (free PDF from MIT) — MSA + Beckmann.
- Beckmann, McGuire, Winsten (1956) Studies in the Economics of Transportation.
