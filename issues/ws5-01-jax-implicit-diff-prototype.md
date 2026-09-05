## Context
Static user equilibrium solves Beckmann's convex program; equilibrium flows are an
implicit function of parameters (OD demand, capacities, cost params). Differentiating
*through* the equilibrium via the implicit function theorem (not unrolling solver
iterations) enables gradient-based OD calibration (ws5-02) and intervention optimization
(ws5-03). Recent literature: Li & Nie (2026) "Traffic assignment as a differentiable
program"; end-to-end differentiable traffic simulation with route choice (arXiv:2604.11380).

## Task
- JAX prototype (small/medium networks, dense enough OD): solve UE (FW or projected
  gradient on path/bush flows), register a custom VJP from equilibrium optimality
  conditions (variational inequality / fixed point of the loading map) using
  jaxopt/optimistix-style implicit diff with matrix-free CG for the linear solve.
- Gradient checks: d(link flows)/d(demand), d(total cost)/d(capacity) vs central finite
  differences on SiouxFalls (rel error < 1e-4 target away from degeneracy).
- Investigate + document non-smoothness: shortest-path switching makes flows piecewise
  smooth; test where FD and IFT disagree; note logit-smoothed loading (ws2-07) as the
  regularized alternative.
- Architecture memo: pure-JAX forward vs JAX-wrapping-Rust forward (custom_vjp around
  ws1/ws2 kernels) — feeds ws5-04 decision.

## Acceptance criteria
- Notebook + module with passing gradient checks; memo of smoothness findings.

## References
- Blondel, Berthet, Cuturi, Frostig, Hoyer, Llinares-Lopez, Pedregosa, Vert (2022)
  "Efficient and Modular Implicit Differentiation", NeurIPS (jaxopt).
- Li, Nie (2026) "Traffic assignment as a differentiable program", SSRN.
- "End-to-end differentiable network traffic simulation with dynamic route choice",
  arXiv:2604.11380.
- Julia analogues if ever needed: ImplicitDifferentiation.jl, InferOpt.jl (differentiating
  through shortest-path layers via perturbation).
- TorchOpt (github.com/metaopt/torchopt) if a PyTorch route is preferred by collaborators.
