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

## References
- Spiess (1990) "Conical Volume-Delay Functions", Transportation Science 24(2).
- Bureau of Public Roads (1964) Traffic Assignment Manual — BPR.
- Boyles, Lownes, Unnikrishnan (2023+) "Transportation Network Analysis" free textbook —
  clean reference for all of WS2: https://sboyles.github.io/blubook.html
