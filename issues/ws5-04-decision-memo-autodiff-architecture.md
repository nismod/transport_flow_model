## Context
Close the exploratory phase with a decision: where does differentiability live long-term?
Options: (A) pure-JAX assignment (GPU-friendly PHAST-like sweeps in XLA, everything
differentiable, but duplicates the Rust solver), (B) JAX custom_vjp wrapping the Rust
solver (single fast forward path; VJP linear solves implemented in Rust or scipy),
(C) confine autodiff to a research extra, keep core non-differentiable.

## Task
- Benchmark forward pass: Rust CCH assignment vs JAX prototype (CPU + GPU) on
  Chicago-Sketch and one 20-city instance.
- Benchmark gradient: implicit VJP cost relative to forward solve for both architectures.
- Assess maintenance cost (deps, wheels, team skills) and GPU availability in target
  deployment environments; recommend and create follow-up issues.

## Acceptance criteria
- Written memo in docs/adr/ with benchmark tables and a recommendation accepted by
  maintainers.

## References
- JAX GPU shortest-path/scan patterns; Delling et al. PHAST GPU results (JPDC 2013) as
  evidence one-to-all sweeps map to GPUs.
