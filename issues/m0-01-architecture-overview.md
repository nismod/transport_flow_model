## Context
There is no single document describing how the package fits together. A developer joining to
work on assignment has to reconstruct the module map, the run data flow and the table schemas
by reading the source. `DEVELOPMENT.md` covers planned work and the literature, not the code
as it stands.

## Task
Write `ARCHITECTURE.md` covering:
- the module map;
- the run data flow: config → `RunConfig` → `Network`/`Demand` → `assign` →
  `AssignmentResult` → losses/`disrupt` → outputs;
- the Python/Rust boundary;
- the extension points;
- the table schemas, in one place.

Under ~200 lines. Document what exists today; aspirational work stays in `DEVELOPMENT.md`.

## Acceptance criteria
- Fits in a day-1 reading path alongside `README.md` and the source files it points at.
- Every claim is checked against the code, not against the workplan.
- Names where a new assignment method goes and links to the ADR holding the contract.
