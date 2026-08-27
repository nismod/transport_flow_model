## Context
Two decisions shape most of what a new contributor writes, and neither is recorded anywhere:
how assignment methods plug in, and what data format crosses module and language boundaries.
Both currently exist only as convention in the code — the backend contract is inferable only
from the single implemented backend.

## Task
Create `docs/adr/` with a template and an index, then write two records.

**ADR-001, assignment methods are registered backends.** The key content is the *backend
contract*: a backend is called as `backend(network, demand, include_paths=..., **options)`
and returns a plain `dict` with keys `link_flows`, `skims`, `unassigned`, and optionally
`paths`, `gap_history`, `iterations`, `relative_gap`. `assign()` wraps that into an
`AssignmentResult` and attaches `Provenance`; backends never construct `AssignmentResult`
themselves. Also record that reserved method names should be *filled in*, not renamed, since
they are referenced in tests, the changelog and the workplan; and the M0-04 rule that
iterative methods keep link flows `float64`.

**ADR-002, Arrow tables are the internal interchange.** Record that only `core.py` imports
`_core`, and what that means for contributors: no pandas in hot paths, no direct `_core`
imports elsewhere.

## Acceptance criteria
- `docs/adr/` contains a template, an index and the two records.
- A developer can implement a new backend from ADR-001 alone, without reading
  `_assign_sequential`.
- The records state facts about the code with file references, and are checked against the
  code rather than against this issue.
