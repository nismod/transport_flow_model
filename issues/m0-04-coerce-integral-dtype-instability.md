## Context
`assignment.coerce_integral` casts float `flow`/`cost` columns to `int64` when every value
is integral, and `link_flows_table` applies it to every backend's output. For an iterative
method that makes the output dtype depend on the iteration count and on the input data:
iteration 1 (all-or-nothing on an integral OD matrix) produces integral flows and is cast to
`int64`; from iteration 2 the averaged flows are fractional and stay `float64`.

An incoming assignment-method developer hits this immediately, and the symptom (a dtype that
changes with `max_iterations`) is a long way from the cause.

## Task
- Add an explicit opt-out: `link_flows_table(..., coerce=True)`, defaulting to current
  behaviour so existing callers are unaffected.
- Document that iterative methods pass `coerce=False`, and why.
- Add a regression test.
- Record the rule in ADR-001.
- Do **not** delete `coerce_integral`: the legacy-parity behaviour it preserves for the
  sequential method is deliberate and pinned by
  `tests/test_assignment_api.py::test_sequential_matches_legacy_allocate`.

## Acceptance criteria
- `link_flows_table(..., coerce=False)` returns `float64` flows regardless of whether the
  values happen to be integral.
- The three existing call sites keep the coercing default; `test_sequential_matches_legacy_allocate`
  and the disruption tests pass unmodified.
- The public API in `__init__.py` is unchanged.
