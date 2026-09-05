# Architecture

How the code fits together **today**. Planned work lives in
[`DEVELOPMENT.md`](DEVELOPMENT.md); decisions and their reasoning live in
[`docs/adr/`](docs/adr/).

The package models transport network flows for infrastructure risk analysis: a
hazard damages links, traffic reroutes or fails to travel, and the resulting
cost is what the analysis is after.

## Modules

Everything is under `src/transport_flow_model/`.

| Module | Responsibility |
| --- | --- |
| `network.py` | `Network`: immutable topology plus per-link attributes as a `pyarrow.Table`; node factorization and lazily built CSR adjacency; builders from (geo)pandas, pyarrow and TNTP. |
| `demand.py` | `Demand`: OD demand in COO form (one row per pair), with a TNTP builder that tracks zone-to-node mapping for split centroids. |
| `assignment.py` | `assign()`, the `METHODS` registry and `register_method`; the `"sequential"` backend; reserved stubs for `"msa"`, `"fw"`, `"bfw"`, `"staq"`; `AssignmentResult` and `Provenance`. |
| `convergence.py` | `relative_gap()` (convergence measure) and `link_costs()` (the BPR volume-delay function, currently hardcoded here). |
| `disruption.py` | `disrupt()`, `Scenario`, `LinkDelta`, `ScenarioResult`, `DisruptionResults`. |
| `core.py` | The only module that imports the Rust extension `_core`. Wraps `allocate()`, `disrupt()`, `skim()`, `shortest_paths_from()`, `prepare()`, `prepare_disruption()`, `version()`. |
| `datasets.py` | Registry of benchmark datasets with checksums and cached downloads; `BEST_KNOWN` published objective values and equilibrium flows. |
| `io.py` | TNTP readers (`read_tntp`, `read_tntp_flows`, `TNTPInstance`). |
| `config.py` | `RunConfig`, a pydantic schema for JSON run configs, mapping 1:1 onto the API. |
| `radiation.py` | `RadiationModel` for OD estimation. |
| `model.py` | Legacy tabular classes, deprecated. Kept importable while functionality is ported; `compute_losses` still comes from here. |

The Rust core is in `core/src/`: `core.rs` (graph algorithms, allocation,
disruption), `arrow_ffi.rs` (Arrow conversion), `lib.rs` (the PyO3 module).

## Run data flow

```
config.json
   │  RunConfig.from_json
   ▼
RunConfig ──load_network()──▶ Network ─┐
          ──load_demand()───▶ Demand ──┼──▶ assign(network, demand, method, **options)
          ──load_scenarios()─▶ [Scenario]                    │
                                  │                          ▼
                                  │                  AssignmentResult
                                  │                (link_flows, skims,
                                  │                 unassigned, paths?,
                                  │                 gap_history, provenance)
                                  │                          │
                                  └──────────┬───────────────┘
                                             ▼
                            disrupt(network, scenarios, base=...)
                                             │
                                             ▼
                                     DisruptionResults
                                (per-scenario rerouted / isolated /
                                 link_flows / losses, plus summary())
                                             │
                                             ▼
                                    CSV / Parquet outputs
```

`assign()` looks the method name up in `METHODS`, times the backend call, and
wraps the returned dict into an `AssignmentResult` with `Provenance` attached.
`disrupt()` runs a baseline assignment (or takes one), prepares the network and
the baseline paths once, then for each scenario reroutes the baseline flows
whose paths use a removed link.

`relative_gap(network, demand, flows)` is a post-hoc quality measure, not part
of the pipeline: it recomputes congested link costs, skims the least-cost time
for every OD pair, and returns how far total travel time exceeds
shortest-path travel time. Iterative methods will also call it internally to
decide when to stop.

Entry points: `scripts/flow_model/flow_allocation.py` and
`scripts/flow_model/flow_disruptions.py` (config-driven runs),
`scripts/benchmark_assignment.py` (gap, wall time and peak RSS per case).

## The Python/Rust boundary

`core.py` is the whole of it. Inbound, `_to_table` normalizes a
`pyarrow.Table`, `RecordBatch` or `pandas.DataFrame` into a table and hands it
to `_core.<name>_ffi`. Outbound, Rust returns PyCapsule-wrapped Arrow C
streams, which `_from_ffi_stream` imports with
`pa.RecordBatchReader._import_from_c_capsule(...).read_all()` — no
serialization, no buffer copy.

| Python | Rust | Returns |
| --- | --- | --- |
| `core.allocate(network, od, capacity_constrained=, directed=)` | `allocate_ffi` | `od_flows`, `network_flows`, `unassigned_od` |
| `core.disrupt(network, od_flows, failed_edges, capacity_constrained=, directed=)` | `PreparedDisruption` | `rerouted_flows`, `network_flows`, `isolated_od`, `losses` |
| `core.skim(network, od_pairs, directed=)` | `skim_ffi` | one table: `origin_id`, `destination_id`, `cost` (null if unreachable) |
| `core.shortest_paths_from(network, origin, directed=)` | `shortest_paths_from_ffi` | one table: `node_id`, `cost` |
| `core.prepare(network)` | `PreparedNetwork` | a handle with `allocate` and `skim` methods |
| `core.prepare_disruption(network, od_flows)` | `PreparedDisruption` | a handle with a `scenario(failed_edges)` method |
| `core.version()` | `version` | extension version string |

No other module may import `_core`; see
[ADR-0002](docs/adr/0002-arrow-tables-are-the-internal-interchange.md). The
extension is required, not optional: `import transport_flow_model` reaches
`core.py`, so a checkout without `maturin develop` cannot import the package.

## Extension points

- **A new assignment method** is a function registered into `METHODS` with
  `@register_method("name")`. It receives
  `(network, demand, include_paths=..., **options)` and returns a dict; it
  never builds an `AssignmentResult` itself. The reserved names `"msa"`,
  `"fw"`, `"bfw"` and `"staq"` are filled in, not renamed. The full contract
  is
  [ADR-0001](docs/adr/0001-assignment-methods-are-registered-backends.md).
- **A batched query across the boundary** — `core.skim(network, od_pairs)`
  returns least-cost travel time per OD pair; `core.prepare(links)` and
  `core.prepare_disruption(links, paths)` return handles whose methods reuse
  one parse.
- **A new benchmark dataset** is an entry in `datasets.DATASETS` (URL,
  checksum, licence, provenance), optionally with a `BEST_KNOWN` published
  objective for validation.
- **A new disruption effect** is a `LinkDelta` attribute. Only link removal is
  implemented; other deltas raise `NotImplementedError`.
- **Volume-delay functions** are *not* yet an extension point. BPR is
  hardcoded in `convergence.link_costs`, reading the `alpha`, `beta` and
  `capacity` link attributes when all three are present and treating the
  network as fixed-cost otherwise.

## Table schemas

All tables are `pyarrow.Table`. Identifier columns keep the caller's type
(string or integer); `flow`, `cost` and `value` are `float64` unless noted.

**Link table** (`Network.to_table()`) — required `edge_from`, `edge_to`,
`edge_id` (unique); any other column is carried as a link attribute.
Attributes the code reads by name: `cost` (free-flow travel time, required by
assignment), `capacity`, `length`, `alpha`, `beta`.

**Demand table** (`Demand.to_table()`) — `origin_id`, `destination_id`,
`value` (cast to `float64` on construction); optional `origin_zone` /
`destination_zone` carry zone ids where they differ from node ids.

**`AssignmentResult`**

| Table | Columns |
| --- | --- |
| `link_flows` | the link table plus `flow`, in network link order |
| `skims` | `origin_id`, `destination_id`, `cost` (flow-weighted mean over assigned paths) |
| `unassigned` | `origin_id`, `destination_id`, `value` |
| `paths` (optional) | `origin_id`, `destination_id`, `flow`, `edge_path` (list of edge ids), `cost` |

plus `gap_history: tuple[float, ...]` and `provenance` (method, options,
iterations, relative gap, wall time, seed, package and core versions).

**`ScenarioResult`** — `rerouted` (path columns), `isolated`
(`origin_id`, `destination_id`, `value`), `link_flows`, and `losses`
(`origin_id`, `destination_id`, `flow`, `initial_cost`, `disrupted_cost`,
`rerouting_loss`). `DisruptionResults.summary()` aggregates to `scenario_id`,
`rerouted_flow`, `isolated_flow`, `rerouting_loss`.

### Flow dtypes

`link_flows_table` casts an all-integral `flow` column to `int64` by default
(`coerce_integral`), preserving the legacy allocator's output for the
`"sequential"` method. **Iterative methods must pass `coerce=False`** so link
flows stay `float64`: their first, all-or-nothing iteration produces integral
flows on integral demand while later averaged iterations do not, so the dtype
would otherwise depend on the iteration count.

## Cost of evaluating convergence

Evaluating `relative_gap` needs one shortest-path cost per OD pair. It asks
for them all in one `core.skim` call, which parses the network once and builds
one tree per distinct origin, and costs roughly 0.3-0.5x an all-or-nothing
pass — cheap enough to check every iteration. See the module docstring of
`convergence.py` for the measured numbers.

That is a recent change, and the general lesson is in `ADR-0002`: **do not put
a `core` call inside a Python loop without thinking about what it re-parses**.
Every `core` function parses its inputs and rebuilds the graph before doing any
work — on chicago-sketch, 85% of a `shortest_paths_from` call. Two loops used
to pay that per iteration:

| loop | before | after | how |
| --- | --- | --- | --- |
| `relative_gap` over origins | 0.47 s | 0.046 s | one batched `core.skim` |
| `disrupt` per scenario | 72 ms | 2.3 ms | `core.prepare_disruption` once |

Batch the call where a batch is natural; where it is not, parse once with
`core.prepare` or `core.prepare_disruption` and call methods on the handle.

Remaining headroom in the scenario loop: `PreparedDisruption.scenario` scans
every parsed path row to find the affected ones, so a no-op scenario still
costs 2.1 ms of the 2.3 ms on chicago-sketch's 93k paths. An index from link to
the paths using it would make that proportional to the flow actually affected —
filed as `issues/m0-12-index-paths-by-link-for-scenarios.md`.
