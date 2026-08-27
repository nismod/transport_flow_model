## Context
Evaluating the relative gap costs about three times an all-or-nothing pass — 75-80% of a
would-be equilibrium iteration. Measured with `scripts/profile_gap_cost.py`
(`pixi run profile-gap-cost`), median of 3-5 single-threaded repeats:

| instance | links | origins | AON (s) | gap (s) | trees only (s) | gap / AON | gap share |
| --- | --- | --- | --- | --- | --- | --- | --- |
| siouxfalls | 76 | 24 | 0.0017 | 0.0051 | 0.0012 | 3.0x | 75% |
| anaheim | 914 | 38 | 0.0057 | 0.0219 | 0.0109 | 3.8x | 79% |
| chicago-sketch | 2950 | 386 | 0.1631 | 0.4683 | 0.2838 | 2.9x | 74% |

(No iterative method exists yet, so "AON" is `assign(..., method="sequential")` and the
share is `t_gap / (t_aon + t_gap)`: the two halves of a would-be MSA iteration timed
separately. A real loop reuses its congested costs, so treat the share as approximate.)

The measurement points at a cause the original framing did not anticipate. The trees are
only part of the cost, and the trees themselves are expensive for the wrong reason:
`shortest_paths_from_ffi` calls `arrow_ffi::read_network_ffi` on **every** call
(`core/src/lib.rs`), so the per-origin loop in `relative_gap` re-reads the link table and
rebuilds the graph once per origin. On chicago-sketch, 386 bare `shortest_paths_from` calls
cost 0.28s, more than a whole `core.allocate` pass over the same 386 origins (0.16s), which
builds the graph once and also loads flows and assembles paths.

## Task
Remove the duplicated work. Two options, which are not exclusive:

1. **Prepare once, query many.** A `core` entry point that builds the graph once and answers
   many origins — either a batched `shortest_paths_from_many(network, origins)` or a prepared
   network handle. This removes the per-origin rebuild without changing what `relative_gap`
   computes, and benefits any caller doing many searches.
2. **Fuse into the AON pass.** Have `core.allocate` return per-origin minimum-cost totals
   alongside its existing outputs, so an iterative method gets the shortest-path term of the
   gap for free from the loading step it already performs.

Option 1 is the smaller change and, on this evidence, recovers the larger share; option 2
removes the remaining tree work for iterative methods specifically. Measure after each.

Either way this changes the Rust/Python interface, which is why it is filed separately rather
than folded into the measurement: it needs its own review. See ADR-0002 for the boundary
conventions any new entry point must follow.

## Acceptance criteria
- `scripts/profile_gap_cost.py` shows a materially lower `gap / AON` ratio on
  chicago-sketch, and the numbers in the `convergence.py` module docstring, `ARCHITECTURE.md`
  and `issues/m0-05-measure-gap-evaluation-cost.md` are updated to match.
- `relative_gap` returns identical values; `tests/test_convergence.py` passes unmodified
  (published SiouxFalls flows still give a gap below 1e-10).
- Any new extension entry point is wrapped in `core.py` and imported nowhere else.

## References
- `src/transport_flow_model/convergence.py` — the per-origin loop.
- `core/src/lib.rs` — `shortest_paths_from_ffi` and its `read_network_ffi` call.
- `docs/adr/0002-arrow-tables-are-the-internal-interchange.md`.
