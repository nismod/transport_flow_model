## Context
`disrupt()` needs a baseline assignment that carries `paths`: it finds the flows whose
baseline path uses a failed link and reroutes those. `PreparedDisruption` is built from
`base.paths`, and `disrupt()` raises `ValueError` when that table is missing.

An equilibrium result has no such table. The `"msa"` method returns `paths=None` and rejects
`include_paths=True`, because an equilibrium is an average of many all-or-nothing solutions:
each OD pair's demand is split over several routes, so there is no single path to record. The
last pass's paths are a different — and much worse — solution than the flows beside them, so
returning those would be misleading.

The consequence is that today only `"sequential"` can produce a disruption baseline, and that
is exactly the method `ws0-04` is meant to characterize as a screening heuristic rather than
something to report criticality from. Criticality results therefore rest on an
order-dependent heuristic with no equilibrium interpretation.

## Task
Give `disrupt()` a baseline it can use from an equilibrium method. Options, not exclusive:

1. **Path flows from a bush-based method.** `ws2-05` (Algorithm B) maintains per-origin bushes
   and can emit a path-flow decomposition directly. This is the principled route and gives
   proportional path flows.
2. **Decompose equilibrium link flows into paths** after the fact, for the OD pairs a scenario
   actually affects. Path-flow decomposition is not unique, so this needs a stated rule
   (e.g. maximum-entropy) and a note that the choice affects rerouting.
3. **Reroute from link flows instead of paths.** Change the disruption engine to take a link
   flow vector plus demand, and re-solve on the disrupted network rather than rerouting the
   affected paths. This is the largest change and interacts with `ws4-01`'s scenario engine,
   but it removes the path requirement altogether and is the only option that works for any
   assignment method.

Whichever is taken, `assign(..., "msa")` followed by `disrupt(..., base=result)` should work,
or fail with a message that names this issue.

## Acceptance criteria
- A disruption run can use an equilibrium baseline end to end on SiouxFalls.
- The rule used to obtain path flows (or to avoid needing them) is documented, including its
  non-uniqueness where that applies.
- `tests/test_disruption*.py` pass unmodified; the sequential baseline keeps working.

## References
- `src/transport_flow_model/disruption.py` — `disrupt()`, `_links_with_base_flow`,
  `PreparedDisruption`.
- `src/transport_flow_model/assignment.py` — `_assign_msa`, which raises on `include_paths`.
- `ws2-05-algorithm-b-bushes.md`, `ws4-01-scenario-engine-recustomization.md`,
  `ws0-04-sequential-allocator-order-dependence.md`.
