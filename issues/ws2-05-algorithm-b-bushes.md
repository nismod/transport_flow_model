## Context (stretch)
Bush/origin-based methods (Dial's Algorithm B, OBA, TAPAS) reach 1e-8+ gaps far faster
than link-based methods and give route-level detail. Valuable when scenario differences
must be resolved very finely, and a natural fit with per-origin CCH searches. Stretch
after ws2-03 is solid.

## Task
- Implement Algorithm B: per-origin acyclic bushes; shift flow from longest to shortest
  paths via Newton steps using dt/dx; bush improvement (add shortcut-free improving links,
  drop unused).
- Reuse CCH one-to-all trees for bush initialization and improvement scans.
- Compare convergence (gap vs time) against BFW on Chicago-regional.

## Acceptance criteria
- 1e-6 gap on Chicago-Sketch faster than BFW; flows consistent with BFW at matched gap.

## References
- Dial (2006) "A path-based user-equilibrium traffic assignment algorithm that obviates
  path storage and enumeration", Transportation Research B 40(10).
- Bar-Gera (2002) "Origin-Based Algorithm for the Traffic Assignment Problem",
  Transportation Science 36(4); Bar-Gera (2010) TAPAS, Transportation Research C.
- Boyles et al. textbook ch. on bush-based methods.
