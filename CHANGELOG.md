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
    capacity-constrained heuristic) is implemented; `"msa"`, `"fw"`,
    `"bfw"` and `"staq"` are registered and raise `NotImplementedError`
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
- Sphinx API reference for the public API.

### Changed

- `assignment.link_flows_table` takes a `coerce` keyword (default `True`,
  the previous behaviour). Iterative assignment methods should pass
  `coerce=False` so link flows stay `float64`: the integral flows of an
  all-or-nothing first iteration were cast to `int64` while later averaged
  iterations stayed `float64`, making the output dtype depend on the
  iteration count and on the input data.
