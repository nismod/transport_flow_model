## Context
Assignment needs volume-delay functions. TNTP instances use BPR with per-link alpha/beta;
UK strategic models often use other speed-flow curves (e.g. COBA/DfT forms). Make this
pluggable and shared between Python and the Rust core.

## Task
- `CostFunction` abstraction: t(x) given free-flow time, capacity, params. Implement:
  BPR (t0*(1+alpha*(x/c)^beta)), conical (Spiess 1990), and a piecewise-linear
  speed-flow curve loader (for DfT-style curves).
- Each must provide t(x), integral T(x)=∫t (for the Beckmann objective), and dt/dx (for
  Newton steps in bush-based methods and for ws5 gradients).
- Rust implementations mirrored for the hot loop; golden tests Python vs Rust.

## Acceptance criteria
- Beckmann objective computed correctly for SiouxFalls at published equilibrium (matches
  literature value to 1e-6 relative).

## Status

**Partly done.** Landed in `src/transport_flow_model/costs.py`:

- `CostFunction`, a `typing.Protocol` with `travel_time(x)`, `integral(x)` and
  `derivative(x)`, each vectorised over links in network link order (float64
  arrays in, float64 array out; a scalar `x` broadcasts).
- `BPR`, a frozen dataclass implementing it, with `BPR.from_network(network,
  distance_cost=...)` reading the `cost`, `alpha`, `beta`, `capacity` and
  `length` link attributes. Links without all of `alpha`/`beta`/`capacity` are
  fixed-cost; `beta == 0` follows the published convention `t0 * (1 + alpha)`;
  a non-positive capacity is treated as uncongested; `derivative` returns 0 at
  `x == 0` for `beta < 1` rather than an infinity.
- `beckmann_objective(network, flows, cost_function=..., distance_cost=...)`.
- `convergence.link_costs` now delegates to `BPR.from_network(...).travel_time`
  and its outputs are unchanged (pinned to rtol 1e-12 by
  `test_bpr_link_costs_match_published_costs`).

The acceptance criterion is met, and on all three cached instances rather than
SiouxFalls alone. `beckmann_objective` on the published best-known flows
reproduces `datasets.BEST_KNOWN[name].objective` to a measured relative error
of 0 (bit-identical doubles), well inside the required 1e-6:

| instance | objective | published | rel. error |
| --- | --- | --- | --- |
| siouxfalls | 4231335.28710744 | 4231335.28710744 | 0 |
| anaheim | 1286032.171096032 | 1286032.171096032 | 0 |
| chicago-sketch (`distance_cost=0.04`) | 17313018.73874779 | 17313018.73874779 | 0 |

`tests/test_costs.py` also checks `derivative` against a central finite
difference and `integral` against a trapezoid quadrature of `travel_time`
(both rtol 1e-6).

Still open in this issue:

- **Conical** volume-delay functions (Spiess 1990).
- **DfT-style piecewise-linear speed-flow curve loader.**
- **Rust mirrors** of the cost functions for the hot loop, with golden tests
  comparing Python against Rust.
- Selecting a cost function from `assign()` / `RunConfig` — cost functions are
  constructed from link attributes only.

## References
- Spiess (1990) "Conical Volume-Delay Functions", Transportation Science 24(2).
- Bureau of Public Roads (1964) Traffic Assignment Manual — BPR.
- Boyles, Lownes, Unnikrishnan (2023+) "Transportation Network Analysis" free textbook —
  clean reference for all of WS2: https://sboyles.github.io/blubook.html
