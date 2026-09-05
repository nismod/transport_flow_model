## Context
Credibility docs: show our numbers match established tools, and document methods with
citations so results are usable in publications.

## Task
- Cross-validation report: skims vs cppRouting/igraph; equilibrium vs AequilibraE (and
  TrafficAssignment.jl if easy) on shared TNTP instances; wall-time table with hardware
  disclosed (avoid benchmarketing: same convergence targets, same thread counts).
- Methods pages per component (routing, assignment, OD, disruption, risk) with the
  formulas and references now scattered across issues; statement of limitations
  (static vs dynamic, no spillback in STAQ point queues, path-set biases in logit).

## Acceptance criteria
- Docs pages published; discrepancies vs other tools explained or filed as bugs.
