## Context (optional layer)
Deterministic UE concentrates flow on single paths; for disruption analysis a stochastic
spread (accounting for perception error / unobserved heterogeneity) can be more realistic
and is smoother — which also helps differentiability in WS5.

## Task
- K-alternative route generation per OD (via penalty method or randomized customization
  metrics on CCH); path-size logit choice among generated routes; SUE outer loop (MSA on
  path flows).
- Config: theta (scale), path-size formulation, K, generation method; document biases of
  selective path sets.

## Acceptance criteria
- SUE flows on SiouxFalls reproduce published examples qualitatively; theta -> large
  recovers UE flows within tolerance.

## References
- Ben-Akiva, Bierlaire (1999) discrete choice for route choice; Frejinger, Bierlaire on
  path sampling.
- Prato (2009) "Route choice modeling: past, present and future research directions",
  J. Choice Modelling 2(1) — survey of path generation + correction terms.
