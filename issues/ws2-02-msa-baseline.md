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

## Status

**Done.** `assign(network, demand, "msa", ...)` fills the reserved `"msa"` stub, registered
with `@register_method` per ADR-0001. Options: `max_iterations`, `target_gap`,
`time_limit_s`, `cost_function`, `distance_cost`, `directed`.

Measured on SiouxFalls (this machine, single-threaded):

| | |
| --- | --- |
| passes to reach relative gap 1e-3 | **743** (0.51 s, ~0.7 ms per pass) |
| Beckmann objective vs `BEST_KNOWN` | **+0.167%** |
| gap after the default 50 passes | 1.5e-2 |
| convergence rate `g[k]/g[2k]` | 2.024, 1.983, 2.039, 1.984 (theory 2.0) |
| first gap | 8.821, the same all-or-nothing gap the benchmark harness reports for `"sequential"` |

So it converges at the textbook O(1/k) rate to the published optimum. Reaching 1e-4 would
take roughly 7 500 passes — MSA is the slow baseline other methods are measured against, not
the method of choice, which is the point of having it.

**The gap costs nothing.** Iteration *k* performs an all-or-nothing load `y` at the current
costs `t` anyway, and that load puts every OD pair on its min-cost path — so `dot(t, y)` *is*
the shortest-path travel time term of the relative gap at the current flows `x`, and the gap
is `dot(t, x) / dot(t, y) - 1`. No extra skim and no `relative_gap()` call.
`test_free_gap_matches_independent_relative_gap` checks each reported gap against an
independent `relative_gap()` evaluation of the iterate it belongs to, so the two code paths
cross-check each other.

Notes on behaviour:

- `paths` is always `None`; `include_paths=True` raises. An equilibrium averages many
  all-or-nothing solutions and has no single path per OD pair, so it cannot yet be a
  disruption baseline — filed as `m0-13`.
- `max_iterations` defaults to 50, a screening budget rather than a converged one. It is left
  at 50 deliberately: the benchmark harness and the CI perf job both budget 50, and raising it
  would silently change their baselines. `provenance.relative_gap` and `gap_history` say what
  was actually achieved.
- With unreachable demand the gap is over the assigned demand only, since unassigned demand
  loads no links. `relative_gap()` would instead raise on such a network.

Validation against published *link flows* on Anaheim, and comparison against other tools, are
`ws2-04`.
