# ADR-0001: Assignment methods are registered backends

- **Status:** Accepted
- **Date:** 2026-08-27

## Context

Assignment — deciding which route each unit of demand takes, and hence how much
flow ends up on each link — is the core modelling problem of this package, and
there is no single right method. The workplan calls for several: the existing
sequential heuristic, the method of successive averages (MSA), Frank-Wolfe and
its bi-conjugate variant, and STAQ. They differ in cost, convergence and the
options they accept, but they consume the same inputs and produce the same
kinds of output.

`assignment.py` already implements this as a name-keyed registry: `METHODS`
maps a method name to a callable, `register_method(name)` decorates a callable
into it, and `assign(network, demand, method=..., **options)` looks the name up
and wraps whatever comes back. Callers select a method by string — including
`scripts/benchmark_assignment.py`, whose `default_methods()` benchmarks
everything in `METHODS` and skips what raises `NotImplementedError`.

What did *not* exist was any statement of the contract a backend must satisfy.
It was inferable only from the one implemented example, `_assign_sequential`.
That is the gap this record closes.

## Decision

**A new assignment method is a function registered into `METHODS`. It is never
a new public entry point, and never a subclass.**

### The backend contract

A backend is called by `assign()` as:

```python
backend(network, demand, include_paths=<bool>, **options)
```

- `network` is a `Network`, `demand` is a `Demand` (see ADR-0002 for what they
  hold). Both are immutable; a backend must not attempt to mutate them.
- `include_paths` is always passed by keyword, so every backend must accept it.
- `**options` are the method-specific keyword arguments the caller passed to
  `assign()` (for example `capacity_constrained=True` for `"sequential"`, or a
  `max_iterations` budget for an iterative method). Declare them as explicit
  keyword parameters with defaults — an unknown option should raise `TypeError`
  from the call rather than being silently ignored.
- `seed` is *not* forwarded to the backend. It is recorded in provenance only.
  A stochastic method that needs it should take it as one of its `**options`.

A backend returns a plain `dict`:

| Key | Required | Type | Meaning |
| --- | --- | --- | --- |
| `link_flows` | yes | `pyarrow.Table` | Network link table plus a `flow` column, in network link order |
| `skims` | yes | `pyarrow.Table` | `origin_id`, `destination_id`, `cost` per OD pair |
| `unassigned` | yes | `pyarrow.Table` | `origin_id`, `destination_id`, `value` — demand that could not be assigned |
| `paths` | no | `pyarrow.Table` or `None` | Per-path detail; populate only when `include_paths=True` |
| `gap_history` | no | iterable of `float` | Relative gap after each iteration |
| `iterations` | no | `int` | Iterations run; defaults to `0` |
| `relative_gap` | no | `float` or `None` | Final gap, if the method computes one |

The three required keys are indexed directly, so omitting one raises
`KeyError`. The optional keys are read with `.get()`.

**Backends never construct `AssignmentResult` and never construct
`Provenance`.** `assign()` does both: it times the backend call, wraps the
dict, and attaches `Provenance` (method name, options, iterations, relative
gap, wall time, seed, package and core versions). This is what makes
provenance uniform across methods — a backend that built its own result would
be free to lie about, or forget, any of it.

### The signature is machine-read, so it is load-bearing

`AssignmentConfig.options()` in `config.py` decides what to pass a backend by
reading `inspect.signature(METHODS[method]).parameters` — it emits
`capacity_constrained` for `"sequential"` and not for `"msa"`, because that is
what the two functions declare. Before this it emitted both unconditionally,
so `{"method": "msa"}` in a config file raised `TypeError` and config-driven
MSA could not run at all.

That turns "declare them as explicit keyword parameters with defaults" above
from advice into a requirement other code depends on. A backend that took
`**kwargs` instead would advertise no options, and a config would silently
pass it none. The three parameters `assign()` supplies itself — `network`,
`demand`, `include_paths` — are not options and are excluded by name.

A config only ever *loses* an option it left at its default this way. An
option the config explicitly sets and the method cannot accept raises
instead: silently dropping a requested `cost_function` would assign the run
with BPR while the config asked for another curve, with nothing in the
results to show it.

Cost functions follow the same shape a step lower down: `COST_FUNCTIONS` in
`costs.py` maps a name to a class and `build_cost_function(name, network,
**params)` builds one, so a curve can be named from a config exactly as a
method can. Unlike backends they have no uniform constructor —
`SpeedFlow.from_table` needs a curve table as well as a network — so
`build_cost_function` bridges that rather than the classes pretending to a
signature they do not share.

### Link flows stay `float64` for iterative methods

`link_flows_table(links, network_flows)` applies `coerce_integral` by default,
casting a `flow` column whose values are all integral to `int64`. That
preserves the legacy allocator's output dtypes, which is deliberate and
test-covered for `"sequential"`.

It is wrong for an iterative method. Iteration 1 is all-or-nothing; on integral
demand it produces integral flows, which would be cast to `int64`. From
iteration 2 the averaged flows are fractional and stay `float64`. The output
dtype would then depend on the iteration count and on the input data.

**Iterative backends therefore call `link_flows_table(..., coerce=False)`.**
The core extension always emits `flow` as `float64`, so opting out is stably
`float64`.

### Reserved names are filled in, not renamed

`METHODS` reserves `"fw"`, `"bfw"` and `"staq"`, registered to stubs that raise
`NotImplementedError` naming the workplan issue. Implementing one means
replacing its `METHODS[name] = _not_implemented(...)` line with a
`@register_method(name)` function under the same name.

`"msa"` has been through this and is the worked example: the stub line went,
`_assign_msa` took its place under `@register_method("msa")`, and the only
edit outside `assignment.py` was dropping `"msa"` from the parametrized
`test_planned_methods_not_implemented`. It is also the first backend to use
the optional keys — `gap_history`, `iterations` and `relative_gap` — and the
first to need `link_flows_table(..., coerce=False)`.

Those names are referenced outside `assignment.py` — the parametrized
`test_planned_methods_not_implemented` in `tests/test_assignment_api.py`, the
`## [Unreleased]` section of `CHANGELOG.md`, and the workplan — so renaming one
is a breaking change to the public API dressed up as an internal edit. Add a
new name only for a genuinely new method.

## Consequences

- Adding a method touches one file and adds no public API surface. Nothing in
  `__init__.py` changes; `assign` remains the only entry point.
- Every method gets provenance, timing and benchmark coverage for free, and
  `scripts/benchmark_assignment.py` picks it up with no change.
- The dict return is unvalidated. A backend that returns the wrong shape fails
  with a `KeyError` or a schema error downstream, not with a helpful message.
  We accept that in exchange for the freedom to add optional keys without
  changing every backend; a validating wrapper can be added later without
  breaking existing backends.
- `METHODS` is mutable module state, so a downstream package could register its
  own backend. That is not currently a supported extension point — the registry
  is not part of the public API, and out-of-tree backends get no compatibility
  guarantee.
- `AssignmentResult.from_tables` (for reloading saved tables) does not carry
  `gap_history`; a result round-tripped through storage loses its gap
  trajectory.

## Alternatives considered

- **An abstract base class per method.** More ceremony for no gain: backends
  share no state and no behaviour, only a call signature. A function is the
  smaller thing that works, and keeps `METHODS` trivially introspectable.
- **A separate public function per method** (`assign_msa`, `assign_fw`, …).
  Every new method would then widen the public API and every caller wanting to
  switch method would need a code change rather than a config change. It would
  also break `RunConfig`, which selects a method by name from JSON.
- **Backends construct `AssignmentResult` themselves.** Rejected: provenance
  would be optional in practice, and the timing would have to be duplicated in
  every backend.
