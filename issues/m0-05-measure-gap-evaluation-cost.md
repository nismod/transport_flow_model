## Context
`convergence.relative_gap` loops in Python over unique origins calling
`core.shortest_paths_from`, building one shortest-path tree per origin. An iterative method
already builds equivalent trees inside `core.allocate` for its all-or-nothing step, so
evaluating the gap every iteration was expected to roughly double the shortest-path work.

## Task
- Measure it: what fraction of iteration time, on SiouxFalls and one larger instance.
- Document it where an iterative-method author will see it.
- File — but do not implement — the follow-up that returns per-origin minimum-cost totals
  from the Rust AON pass. Fusing those computations changes the Rust/Python interface and
  deserves its own review.
- If the measurement shows the cost is negligible even at scale, that is a valid result:
  record it and close the follow-up.

## Acceptance criteria
- The measurement is repeatable, not a one-off number pasted into a doc.
- The figure is recorded where an iterative-method author will find it.
- The follow-up is filed with the measurement attached, or explicitly closed as not worth
  pursuing.

## Result
Not negligible, and worse than expected: gap evaluation costs about **three times** an
all-or-nothing pass — 75-80% of a would-be iteration — measured by
`scripts/profile_gap_cost.py` (`pixi run profile-gap-cost`):

| instance | links | origins | AON (s) | gap (s) | gap / AON | gap share of iteration |
| --- | --- | --- | --- | --- | --- | --- |
| siouxfalls | 76 | 24 | 0.0017 | 0.0051 | 3.0x | 75% |
| anaheim | 914 | 38 | 0.0057 | 0.0219 | 3.8x | 79% |
| chicago-sketch | 2950 | 386 | 0.1631 | 0.4683 | 2.9x | 74% |

The dominant cost is not the search but graph construction: `shortest_paths_from_ffi` calls
`read_network_ffi` on every call, so the per-origin loop rebuilds the whole graph once per
origin. Recorded in the `convergence.py` module docstring and in `ARCHITECTURE.md`;
follow-up filed as `m0-11`.
