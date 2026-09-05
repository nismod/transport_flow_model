## Context
For damage/loss analysis, static equilibrium over-assigns beyond capacity and the current
sequential heuristic has no equilibrium semantics. Quasi-dynamic assignment with strict
capacity constraints and residual point queues (STAQ; Bliemer et al.) is the middle
ground: static demand period, hard capacities, queues with spillback-free vertical
storage, squeezing flows through bottlenecks. This is the principled replacement for the
sequential allocator in risk work — queues and unmet demand are exactly the loss metrics
we report.

## Task
- Implement STAQ-style two-phase loading: (1) squeezing phase propagating reduction
  factors at bottleneck nodes, (2) queuing phase building residual queues consuming the
  period; route choice fixed per iteration via CCH paths; outer loop to equilibrium on
  perceived costs.
- Outputs: link flows, v/c, queue sizes, delays, demand served vs residual — feed ws4-02
  metrics directly.
- Validate qualitatively vs static equilibrium (queues appear where v/c>1 was) and on any
  published STAQ example reproducible from the papers.

## Acceptance criteria
- Runs on West Yorkshire within 5x the BFW time; documented behavior on a constructed
  bottleneck example matches theory (queue = inflow - capacity x duration).

## References
- Bliemer, Raadsen, Smits, Zhou, Bell (2014) "Quasi-dynamic traffic assignment with
  residual point queues incorporating a first order node model", Transportation Research B 68.
- Bliemer & Raadsen work on STAQ / static assignment with queuing (see DEVELOPMENT.md list).
- Raadsen, Bliemer (2019+) on path-based static assignment with queues.
