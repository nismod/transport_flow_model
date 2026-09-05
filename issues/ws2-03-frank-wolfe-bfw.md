## Context
Frank-Wolfe with exact line search, then conjugate/bi-conjugate FW (BFW), is the classic
link-based equilibrium workhorse; combined with CCH for both shortest paths and loading it
is what Schneck & Nökel productionized (42x speedup, bi-conjugate FW). Target: practical
1e-4..1e-5 gaps on large networks.

## Task
- FW: AON direction via CCH, exact line search on Beckmann objective (bisection/Newton on
  1D), flow update; BFW per Mitradjieva & Lindberg.
- Warm start from sequential allocator or previous scenario's flows (important for ws4:
  scenario deltas re-converge in few iterations).
- Parallel loading already handled in ws1-06; ensure per-iteration customization uses
  partial re-customization when few link costs changed materially (threshold trick from
  Buchhold et al.).

## Acceptance criteria
- 1e-4 relative gap on Chicago-Sketch and at least one 20-city instance; link flows match
  published reference solutions within tolerance consistent with gap level.
- Wall-time comparison vs MSA and vs AequilibraE on same instance documented.

## References
- Mitradjieva, Lindberg (2013) "The Stiff Is Moving — Conjugate Direction Frank-Wolfe
  Methods with Applications to Traffic Assignment", Transportation Science 47(2).
- Schneck, Nökel (2020) "Accelerating Traffic Assignment with Customizable Contraction
  Hierarchies", TRR 2674(1), doi:10.1177/0361198119898455.
- Buchhold, Sanders, Wagner (2019), ACM JEA 24(2), doi:10.1145/3362693.
- Bläsius, Feilhauer, Jung, Laupichler, Sanders, Zündorf (2025) "Synergistic Traffic
  Assignment", arXiv:2502.04343 — read for further engineering ideas.
