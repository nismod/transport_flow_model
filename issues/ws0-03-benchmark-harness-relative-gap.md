## Context
Current benchmarks measure wall time. For equilibrium methods, time is meaningless without
solution quality: the standard is the *relative gap* (excess cost over shortest-path cost).
Boyce, Ralevic-Dekic & Bar-Gera (2004) argue gaps of 1e-4 or better are needed before flow
differences between scenarios are trustworthy — directly relevant since our product is
scenario *differences*.

## Task
- Add relative gap computation: `gap = (sum_a t_a(x_a) x_a - sum_od d_od * c_od_min) / sum_od d_od * c_od_min`.
- Benchmark harness records (instance, method, threads, iterations, gap trajectory, wall
  time, peak RSS) to parquet; plots gap-vs-time.
- CI perf job on small instances with regression thresholds (fail on >20% slowdown or gap
  regression at fixed iteration budget); nightly job for large instances.

## Acceptance criteria
- One command (`pixi run bench`) produces a comparable report across methods/backends.
- CI catches an artificially introduced 2x slowdown in a test of the harness itself.

## References
- Boyce, Ralevic-Dekic, Bar-Gera (2004) "Convergence of Traffic Assignments: How Much is
  Enough?" J. Transportation Engineering 130(1).
- Existing Criterion (Rust) and pyinstrument tasks in DEVELOPMENT.md — integrate, don't duplicate.
