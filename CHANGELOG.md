# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the major version is 0, minor releases may contain breaking changes;
any breaking change is listed here under "Changed" or "Removed", and
deprecations are announced at least one minor release before removal (see
the versioning policy in the documentation).

## [Unreleased]

### Added

- Drafting v0 public API (`transport_flow_model` top level):
  - `Network`: immutable network topology and link attributes backed by a
    pyarrow table, with node factorization and CSR adjacency arrays, and
    builders from (geo)pandas dataframes, pyarrow tables and TNTP instances.
  - `Demand`: origin-destination demand in COO form (`origin_id`,
    `destination_id`, `value`) with a TNTP builder that tracks zone-to-node
    mapping for split centroids.
  - `assign(network, demand, method=..., **options) -> AssignmentResult`
    with a registry of assignment backends. `"sequential"` (the existing
    capacity-constrained heuristic) and `"msa"` (user equilibrium) are
    implemented; `"fw"`, `"bfw"` and `"staq"` are registered and raise
    `NotImplementedError`
  - `AssignmentResult` with pyarrow tables (`link_flows`, `skims`,
    `unassigned`, optional `paths`), `gap_history`, and `Provenance`
    metadata (method, options, iterations, relative gap, wall time, seed,
    package and core versions).
  - `Scenario` / `LinkDelta`: disruption scenarios as sparse sets of link
    attribute deltas with metadata. Starts with link removal;
    other deltas raise `NotImplementedError`
  - `disrupt(network, scenarios, demand=..., base=...) -> DisruptionResults`
    with per-scenario rerouted/isolated/link-flow/loss tables and a
    `summary()` aggregate; unaffected scenarios are skipped by default and
    reported.
  - `RunConfig`: pydantic schema for JSON run configs mapping 1:1 onto the
    API (`load_network()`, `load_demand()`, `load_scenarios()`).

- `pydantic >= 2` is now a dependency.
- CHANGELOG (this file), semantic versioning and deprecation policy
  (documented in the Sphinx docs under "Versioning and deprecation").
- Sphinx API reference for the public API, including `relative_gap` and
  `link_costs`.
- User guide "Equilibrium Assignment": all-or-nothing versus user
  equilibrium, `assign(..., "msa")`, reading `relative_gap` and
  `gap_history`, what `ConvergenceWarning` means, Boyce et al.'s 1e-4
  threshold, and choosing between `BPR`, `Conical` and `SpeedFlow` —
  including why a run must be scored with the curve it was assigned with
  (the same converged siouxfalls flows read 9.963e-04 against BPR and
  6.262e-02 against a conical curve).
- User guide "Config-Driven Runs": the JSON schema, its blocks, method
  options, naming a cost function, and the two scripts. The config schema is
  versioned public interface and previously had no user-facing guide.
- The docs landing page now describes the package and how to install it
  instead of opening on a bare table of contents.
- `scripts/profile_gap_cost.py` (`pixi run profile-gap-cost`): measures what
  evaluating the relative gap costs relative to an all-or-nothing pass.
  Recorded in the `convergence` module docstring, along with why it once
  cost three times as much.
- `assign(..., "msa")`: the method of successive averages, the first
  user-equilibrium method. It averages repeated all-or-nothing loads at
  congested costs with step `1/k`, and takes `max_iterations`,
  `target_gap`, `time_limit_s`, `cost_function`, `distance_cost` and
  `directed`. On SiouxFalls it reaches a relative gap of 1e-3 in 743
  passes (0.5s), with a Beckmann objective 0.17% above the published
  optimum, falling like 1/k as the theory says. Each gap costs nothing:
  the all-or-nothing load an iteration performs anyway puts every OD pair
  on its min-cost path, so it *is* the shortest-path term of the gap.
  `paths` is always `None` — an equilibrium has no single path per OD
  pair, so `include_paths=True` raises and an MSA result cannot yet be a
  disruption baseline (issue m0-13). `max_iterations` defaults to 50,
  which is a screening budget, not a converged one: on SiouxFalls it stops
  around a gap of 1.5e-2, two orders above the default `target_gap`. A run
  that stops short of its target now warns with `ConvergenceWarning`,
  naming the passes performed, the gap reached and which budget ran out,
  instead of returning an unconverged answer silently. The gap MSA reports
  is always a measurement of the flows it hands back — all three exits stop
  before averaging again, where budget exhaustion used to average once more
  and report the previous iterate's gap.
- `ConvergenceWarning`, exported from the top level: an iterative method
  stopped before reaching its target gap. A `UserWarning`, because the
  partly converged result is still usable; silence it with
  `warnings.simplefilter` where that is the intent.
- `costs` module with the `CostFunction` protocol — `travel_time(x)`,
  `integral(x)` and `derivative(x)`, each vectorised over links in network
  link order — and `BPR`, a frozen dataclass implementing it from per-link
  `free_flow`, `capacity`, `alpha`, `beta` and an optional additive
  `distance_cost * length` term. `BPR` and `beckmann_objective(network,
  flows)` (the objective user equilibrium minimizes, validated against
  `datasets.BEST_KNOWN` for siouxfalls, anaheim and chicago-sketch) are
  exported from the top level. `link_costs` now delegates to
  `BPR.from_network(...).travel_time(...)`; its results are unchanged.
  Mirrored Rust implementations with golden tests remain open in ws2-01;
  measured at 0.064–2.1% of an MSA pass, the cost function is not a hot
  loop.
- `Conical`, the Spiess (1990) conical volume-delay function, satisfying
  `CostFunction`. Unlike BPR's power curve it stays finite and increasing
  at every flow, which suits Newton-style bush methods. Its curvature
  parameter `alpha` must exceed 1 and is **not** read from a link
  attribute: every TNTP network already has an `alpha` attribute holding
  BPR's 0.15, which is not a conical alpha. Pass it to `from_network`
  instead; the default of 4.0 is a curve shape, not a calibration. The
  closed-form integral and derivative are checked against quadrature and a
  finite difference in `tests/test_costs.py`.
- `SpeedFlow`, a piecewise-linear speed-against-flow-per-lane cost
  function — the shape a DfT TAG speed-flow table takes — with
  `SpeedFlow.from_table(curves, network)` to assign one curve per link from
  a tidy `curve_id, flow_per_lane, speed` table. Speed is clamped below at
  a positive `min_speed` so travel time and its integral stay finite;
  outside the breakpoints the nearest segment's line is extrapolated.
  **Units are the caller's responsibility and are not checked** — flow per
  lane, speed and length must agree, or the result is silently wrong. The
  test fixture is synthetic and clearly labelled; no TAG data ships yet.
  `Conical` and `SpeedFlow` are exported from the top level alongside `BPR`.
- `COST_FUNCTIONS` and `build_cost_function(name, network, **params)` in
  `costs`, mirroring `assignment.METHODS`, so a curve can be named from a
  config as a method can. `link_costs` and `relative_gap` take a
  `cost_function=`, defaulting to BPR exactly as before, so a run can be
  scored with the curve it was assigned with rather than silently against
  BPR. `beckmann_objective` already took one.
- `AssignmentConfig` takes `cost_function` (a `name` plus its parameters)
  and a free-form `method_options`, and `options(network=None)` is now
  method-aware.
- `core.skim(network, od_pairs) -> pa.Table`: least-cost travel time per OD
  pair, one shortest-path tree per distinct origin over a network parsed
  once, with a null cost where the destination is unreachable.
- `core.prepare_disruption(links, paths) -> PreparedDisruption`: a network
  and a baseline path set parsed once, with a `scenario(failed_edges)`
  method. The path table is usually far larger than the link table and
  nothing in it depends on which links fail, so a scenario run should not
  re-read it every time.
- `core.prepare(links) -> PreparedNetwork`: a network parsed once, with
  `allocate` and `skim` methods that reuse it. Every `core` call
  otherwise re-parses its link table, interns ids and rebuilds the graph
  before doing any work, which callers looping over origins or scenarios
  pay every iteration. The module-level functions are unchanged and are
  now implemented as `prepare(...)` plus one call. ADR-0002 is amended to
  record when a handle is permitted.
- `PreparedNetwork.set_costs(costs)`: replace every link's cost in place, in
  network link order, from a numpy array, a pyarrow array or any sequence.
  An iterative method changes every link's cost every iteration, and the
  only alternative was re-parsing a whole link table with `core.prepare` —
  the per-iteration re-parse the handle exists to remove. The graph is
  rebuilt from the edge list on every call and nothing derived from costs
  is cached, so this is the one permitted mutation of a prepared network;
  it rejects a wrong-length input and any value that is not finite.
- Criterion benchmarks for the Arrow reader path (`read_network_batches`,
  `prepare_network_batches`), so `pixi run extension-bench` catches a
  regression in how fast a network is parsed.
- `CONTRIBUTING.md`: development environment, the canonical Pixi task table
  (moved out of `README.md`, and now including `pixi run bench`) and pull
  request expectations.
- `ARCHITECTURE.md`: module map, run data flow, the Python/Rust boundary,
  extension points and the table schemas in one place.
- Architecture decision records under `docs/adr/`, with a template and an
  index: ADR-0001 records the assignment backend contract (what `assign()`
  passes a backend, what dict it must return, and that reserved method
  names are filled in rather than renamed); ADR-0002 records that
  `pyarrow.Table` is the internal interchange and that
  `transport_flow_model.core` is the only module importing the `_core`
  extension.

### Changed

- **The documentation is restructured along Diátaxis lines**, into
  `tutorials/`, `how-to/`, `reference/` and `explanation/`. Every page has
  moved, so **published URLs under the old paths break** — the docs site has
  no version selector and no redirects. Previously everything lived in
  `guides/` and was, in Diátaxis terms, explanation-with-examples regardless
  of what the page was named.
  - New `tutorials/getting-started`: assign SiouxFalls, measure how far the
    result is from equilibrium, converge it, then remove the busiest link
    and read the loss. All on vendored data, every step executed.
  - How-to pages are named for the goal (`converge-an-assignment`,
    `evaluate-disruptions`, `estimate-od-demand`, `run-from-a-config`)
    rather than the feature.
  - `od-estimation` is split into `how-to/estimate-od-demand` and
    `explanation/radiation-model`. Its twelve skipped doctests are now
    executed — which surfaced that its "Integrating with Flow Allocation"
    example could never have run: the radiation model emits zone ids and
    assignment needs node ids, so the example failed with a dtype error. The
    how-to now maps between them.
  - `equilibrium` and `configuration` are each split across their two modes.
  - No `# doctest: +SKIP` remains anywhere in the docs; doctests rise from
    181 to 192.
- The user guides teach the v0 public API. `data-models`,
  `least-cost-allocation`, `multiple-flows-and-capacity`, `disruptions` and
  `losses` were written against `transport_flow_model.model`, which the API
  reference labels deprecated and the versioning policy formally deprecates
  — so the guides contradicted the reference, and a reader following them in
  order learned the deprecated API end to end. They now use `Network`,
  `Demand`, `assign`, `disrupt` and `Scenario`; every example reproduces the
  same numbers as before. Doctests rise from 120 to 181, all executed.
- `AssignmentConfig.options()` reads the registered backend's own signature
  instead of always emitting `capacity_constrained` and `directed`, and
  takes an optional `network` (needed only to build a `cost_function`). A
  config naming `"msa"` previously raised `TypeError: _assign_msa() got an
  unexpected keyword argument 'capacity_constrained'`, so config-driven MSA
  could not run at all. An option the config *sets* that the method cannot
  accept now raises rather than being dropped; one left at its default is
  dropped silently. ADR-0001 records that this makes a backend's declared
  keyword parameters load-bearing.
- The Rust Arrow readers resolve and downcast each numeric column once per
  record batch instead of looking it up by name for every row. Parsing a
  network is roughly 1.6-1.8x faster, so a `core.shortest_paths_from` call
  on chicago-sketch drops from 668us to 370us.
- `README.md` now describes what the package does since the v0 API landed,
  rather than the earlier sequential capacity-constrained allocator, and
  notes that an editable install still needs the Rust extension built.
- The development docs page no longer duplicates the environment, task and
  lint instructions now in `CONTRIBUTING.md`, and its description of the
  Rust boundary is corrected: the wrappers are `core.allocate` and
  `core.disrupt` (not `allocate_arrow`/`disrupt_arrow`), and data crosses
  through the Arrow C stream interface rather than Arrow IPC.
- `disrupt` prepares the network and baseline paths once for the whole
  scenario run, and indexes those paths by link so a scenario reaches only
  the flows it affects. On chicago-sketch a scenario drops from 72ms to
  0.23ms, so 10 000 scenarios go from about 12 minutes to under a minute.
  A scenario that removes a link carrying no flow used to cost as much as
  one that reroutes real traffic.
- `relative_gap` and `RadiationModel.generate` ask for all their
  shortest-path costs in one `core.skim` call instead of looping over
  origins. Evaluating the relative gap on chicago-sketch drops from 0.47s
  to 0.046s, from three times an all-or-nothing pass to 0.4 times one, so
  an iterative method can afford to check convergence every iteration.
- `assignment.link_flows_table` takes a `coerce` keyword (default `True`,
  the previous behaviour). Iterative assignment methods should pass
  `coerce=False` so link flows stay `float64`: the integral flows of an
  all-or-nothing first iteration were cast to `int64` while later averaged
  iterations stayed `float64`, making the output dtype depend on the
  iteration count and on the input data.
