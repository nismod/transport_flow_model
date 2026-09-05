## Context
The current capacity-constrained *sequential* allocation routes OD pairs in some order,
consuming capacity as it goes. Results (and therefore criticality rankings) depend on that
order, and the procedure has no equilibrium interpretation. We should quantify this before
building on top of it, and document where the heuristic is/isn't adequate.

## Task
- Experiment: on West Yorkshire + 2-3 TNTP instances, run the sequential allocator with
  N=50 random OD orderings (fixed seeds). Report distribution of: total cost, per-link
  flows (max abs/rel deviation), and rank correlation (Kendall tau) of link criticality
  scores across orderings.
- Compare against MSA/FW equilibrium flows (once ws2-02 lands) on the same instances.
- Write a short methods page: "Sequential allocation: semantics, limitations, when to use".

## Acceptance criteria
- Reproducible notebook/script + docs page with the numbers.
- A recommendation in the docs (expected: sequential = fast screening only; equilibrium or
  STAQ for reported criticality results).

## Implementation notes
- Keep the allocator; it stays useful as a warm start for FW and as a cheap screen in
  large scenario sweeps (ws4). This issue is about honest characterization, not removal.
