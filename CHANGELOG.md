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

- Radiation model: `alpha`, the scale parameter of the extended radiation
  model (Yang et al. 2014), in `radiation_probabilities`, `radiation_flows`
  and as `alphas` in `calibrate_radiation`. `None` keeps the original model.
- Disruption scenarios with link cost changes: `LinkDelta(link_id, "cost",
  value=...)` and `LinkDelta(link_id, "cost", scale=...)` (finite,
  non-negative) are now supported alongside removals, with
  `Scenario.from_changes(id, remove=..., cost=..., cost_scale=..., **metadata)`
  and `Scenario.changed_links`. Flows on removed links are rerouted at the
  changed costs. Flows on links whose cost changed keep their path by
  default and are reported in the new `ScenarioResult.repriced` table and in
  `losses` (initial cost is always the baseline path cost);
  `disrupt(..., reroute_changed=True)` reroutes them instead.
  `DisruptionResults.summary()` gains a `repriced_flow` column, and
  `skip_unaffected` also considers changed links. `ScenarioResult.link_flows`
  carries the scenario's link costs. `transport_flow_model.core.disrupt`
  accepts `reroute_edges` (flows released and reallocated while the edges
  stay available).

- Radiation model: `include_self` lets a zone be its own destination, at the
  intrazonal impedance on the diagonal. `calibrate_radiation` grid-searches the
  impedance, origin mass, `max_impedance` and `include_self` by fit to
  aggregate observed flows.
- OD constraints and validation (`transport_flow_model.constraints`):
  `aggregate_od` (zone OD to group totals), `scale_to_aggregate` (match
  group-to-group targets with `"block"` scaling or `"ipf"` that also keeps zone
  production), `common_part_of_commuters` and `od_fit` (fit metrics and a
  comparison table).
- Radiation model: separate origin and destination masses
  (`RadiationModel.generate(..., destination_relevance_column=...)`), a
  `normalisation` option (`"finite"`, `"row"` or `"none"`),
  `RadiationModel.flows` for Demand-ready OD flows with `min_flow` pruning,
  `RadiationModel.impedance_matrix` (igraph Dijkstra, optionally directed),
  and the vectorised matrix functions `radiation_probabilities` and
  `radiation_flows`.

### Changed

- Radiation model: `radiation_probabilities(..., include_self=True,
  normalisation="finite")` now raises `ValueError`, because the finite-size
  correction assumes a zone is not its own destination. Use `"row"` (the
  `radiation_flows` default) or `"none"`.
- Release workflow: builds abi3 wheels with maturin for manylinux and
  musllinux (x86_64, aarch64), macOS (x86_64, arm64) and Windows (x64), plus
  an sdist, and publishes them with PyPI trusted publishing when a GitHub
  release is published. The previous workflow built a single
  `linux_x86_64` wheel that PyPI rejects. The sdist now contains only the
  files needed to build. Release steps are in `DEVELOPMENT.md`.
- Disruption: a scenario with deltas on attributes other than `cost` still
  raises `NotImplementedError`, now before any scenario is evaluated.

- Packaging: the build backend is now maturin, so wheels and source installs
  (including `pip install git+...`) contain the Python package and the compiled
  `_core` extension. Previously the hatchling build shipped only `_version.py`.
  The version now comes from `Cargo.toml` (`0.2.0-rc.2`), and
  `transport_flow_model.__version__` reads the installed package metadata.
- Radiation model: intervening opportunities `s_ij` are now the zones at
  least as close to the origin as the destination (Simini et al. 2012).
  Previously they were all zones within `distance_threshold`. This changes the
  probabilities that `RadiationModel.generate` returns.
- Radiation model: the per-pair Python loop is replaced by vectorised
  computation (about 1 s for 2,500 zones, excluding shortest paths).
  The private helpers `_shortest_paths_from`, `_count_intervening_opportunities`
  and `_calculate_probability` are removed.

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
