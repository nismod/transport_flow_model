# Issue bodies

One markdown file per issue: the file holds the body only, while title, labels and milestone
are listed below and passed by whatever creates the issue:

    gh issue create --title "<title>" --body-file issues/<file> \
      --label "<labels>" --milestone "<milestone>"

Nothing here has been filed on GitHub yet. Bodies cross-reference each other by workplan ID
(`ws2-02`) rather than issue number, since numbers are unknown before creation — the
reserved assignment-method stubs in `assignment.py` cite those same IDs.

## M0 — handover

Work to unblock assignment-method development: orientation docs, the decisions that already
existed implicitly in the code, and the latent problems an incoming developer would hit
first. m0-06 to m0-10 were reserved for the assignment methods themselves and were never
used; those are tracked as `ws2-*` below.

| ID | File | Title | Labels | Milestone | Status |
| --- | --- | --- | --- | --- | --- |
| m0-01 | [m0-01-architecture-overview.md](m0-01-architecture-overview.md) | Write ARCHITECTURE.md | docs | M1 Foundations & measurement | **Done** |
| m0-02 | [m0-02-contributing-and-readme.md](m0-02-contributing-and-readme.md) | Add CONTRIBUTING.md and refocus the README | docs | M1 Foundations & measurement | **Done** |
| m0-03 | [m0-03-architecture-decision-records.md](m0-03-architecture-decision-records.md) | Record architecture decisions as ADRs | docs | M1 Foundations & measurement | **Done** |
| m0-04 | [m0-04-coerce-integral-dtype-instability.md](m0-04-coerce-integral-dtype-instability.md) | Link flow dtype depends on iteration count | python | M1 Foundations & measurement | **Done** |
| m0-05 | [m0-05-measure-gap-evaluation-cost.md](m0-05-measure-gap-evaluation-cost.md) | Measure the cost of evaluating the relative gap | performance, python | M1 Foundations & measurement | **Done** |
| m0-11 | [m0-11-fuse-gap-evaluation-into-aon.md](m0-11-fuse-gap-evaluation-into-aon.md) | Stop rebuilding the graph per origin when evaluating the gap | performance, rust, python | M3 Equilibrium & quasi-dynamic assignment | **Done** |
| m0-12 | [m0-12-index-paths-by-link-for-scenarios.md](m0-12-index-paths-by-link-for-scenarios.md) | Index baseline paths by link so a scenario skips untouched flows | performance, rust | M4 Disruption & criticality at scale | Open |

## WS0-WS6 — the workplan

The full roadmap, in six workstreams across milestones M1-M6. Statuses are as of the last
review: the three WS0 issues marked done, and part of ws1-07, were delivered before this
backlog was written down, so they are recorded here rather than re-filed.

| ID | File | Title | Labels | Milestone | Status |
| --- | --- | --- | --- | --- | --- |
| ws0-01 | [ws0-01-tntp-readers-benchmark-corpus.md](ws0-01-tntp-readers-benchmark-corpus.md) | WS0-01: TNTP readers and reference benchmark corpus (SiouxFalls..20-city) | ws0-foundations, python | M1 Foundations & measurement | **Done** — `io.read_tntp`, `datasets` registry with checksums and `BEST_KNOWN`, datasets guide |
| ws0-02 | [ws0-02-public-api-v0.md](ws0-02-public-api-v0.md) | WS0-02: Freeze public API v0 (Network, Demand, assign, disrupt) | ws0-foundations, python | M1 Foundations & measurement | **Done** — v0 API, pydantic `RunConfig`, scripts are thin drivers, CHANGELOG + Sphinx reference |
| ws0-03 | [ws0-03-benchmark-harness-relative-gap.md](ws0-03-benchmark-harness-relative-gap.md) | WS0-03: Benchmark harness with relative-gap metric and CI perf regression | ws0-foundations, performance | M1 Foundations & measurement | **Done** — `relative_gap`, `scripts/benchmark_assignment.py`, `perf.yml` with `--handicap` self-test |
| ws0-04 | [ws0-04-sequential-allocator-order-dependence.md](ws0-04-sequential-allocator-order-dependence.md) | WS0-04: Characterize order-dependence of the sequential allocator | ws0-foundations, research, docs | M1 Foundations & measurement | Open |
| ws1-01 | [ws1-01-rust-csr-core-dijkstra-parity.md](ws1-01-rust-csr-core-dijkstra-parity.md) | WS1-01: Rust CSR graph core with Dijkstra parity vs igraph | ws1-routing, rust | M2 Routing engine (CCH/PHAST) | Open |
| ws1-02 | [ws1-02-contraction-hierarchies.md](ws1-02-contraction-hierarchies.md) | WS1-02: Contraction Hierarchies (build, bidirectional query, unpacking) | ws1-routing, rust | M2 Routing engine (CCH/PHAST) | Open |
| ws1-03 | [ws1-03-nested-dissection-order.md](ws1-03-nested-dissection-order.md) | WS1-03: Nested-dissection order and CCH supergraph (InertialFlowCutter/METIS) | ws1-routing, rust | M2 Routing engine (CCH/PHAST) | Open |
| ws1-04 | [ws1-04-cch-customization.md](ws1-04-cch-customization.md) | WS1-04: CCH customization (parallel, partial re-customization, infinite weights) | ws1-routing, rust, performance | M2 Routing engine (CCH/PHAST) | Open |
| ws1-05 | [ws1-05-elimination-tree-queries-phast.md](ws1-05-elimination-tree-queries-phast.md) | WS1-05: Elimination-tree queries and PHAST one-to-all sweeps | ws1-routing, rust, performance | M2 Routing engine (CCH/PHAST) | Open |
| ws1-06 | [ws1-06-rphast-batched-od.md](ws1-06-rphast-batched-od.md) | WS1-06: RPHAST batched many-to-many skims and network loading | ws1-routing, rust, performance | M2 Routing engine (CCH/PHAST) | Open |
| ws1-07 | [ws1-07-pyo3-arrow-bindings.md](ws1-07-pyo3-arrow-bindings.md) | WS1-07: PyO3 bindings with zero-copy Arrow I/O | ws1-routing, rust, python | M2 Routing engine (CCH/PHAST) | **Partly** — Arrow C-stream I/O, handles and a batched skim done; no CCH, no GIL release, and the readers still copy |
| ws1-08 | [ws1-08-scale-validation-10m-edges.md](ws1-08-scale-validation-10m-edges.md) | WS1-08: Scale gate — 10M edges / 1M OD in minutes | ws1-routing, performance, epic-gate | M2 Routing engine (CCH/PHAST) | Open |
| ws2-01 | [ws2-01-link-cost-functions.md](ws2-01-link-cost-functions.md) | WS2-01: Pluggable link cost functions (BPR, conical, DfT speed-flow) | ws2-assignment, python, rust | M3 Equilibrium & quasi-dynamic assignment | Open |
| ws2-02 | [ws2-02-msa-baseline.md](ws2-02-msa-baseline.md) | WS2-02: MSA user-equilibrium baseline | ws2-assignment, python | M3 Equilibrium & quasi-dynamic assignment | Open |
| ws2-03 | [ws2-03-frank-wolfe-bfw.md](ws2-03-frank-wolfe-bfw.md) | WS2-03: Frank-Wolfe and bi-conjugate FW on the CCH kernel | ws2-assignment, rust, performance | M3 Equilibrium & quasi-dynamic assignment | Open |
| ws2-04 | [ws2-04-equilibrium-validation-suite.md](ws2-04-equilibrium-validation-suite.md) | WS2-04: Equilibrium validation suite vs published solutions and other tools | ws2-assignment, research, epic-gate | M3 Equilibrium & quasi-dynamic assignment | Open |
| ws2-05 | [ws2-05-algorithm-b-bushes.md](ws2-05-algorithm-b-bushes.md) | WS2-05: Bush-based Algorithm B for high-precision equilibria (stretch) | ws2-assignment, rust, stretch | M3 Equilibrium & quasi-dynamic assignment | Open |
| ws2-06 | [ws2-06-staq-quasi-dynamic.md](ws2-06-staq-quasi-dynamic.md) | WS2-06: STAQ-style quasi-dynamic assignment with residual queues | ws2-assignment, rust, research | M3 Equilibrium & quasi-dynamic assignment | Open |
| ws2-07 | [ws2-07-stochastic-loading-psl.md](ws2-07-stochastic-loading-psl.md) | WS2-07: Path-size-logit stochastic loading (optional layer) | ws2-assignment, python, stretch | M3 Equilibrium & quasi-dynamic assignment | Open |
| ws3-01 | [ws3-01-gravity-radiation-od.md](ws3-01-gravity-radiation-od.md) | WS3-01: Gravity and radiation OD models with Furness/IPF balancing | ws3-od-estimation, python | M3 Equilibrium & quasi-dynamic assignment | Open |
| ws3-02 | [ws3-02-od-calibration-interface.md](ws3-02-od-calibration-interface.md) | WS3-02: OD calibration to link counts — interface + Spiess baseline | ws3-od-estimation, python | M3 Equilibrium & quasi-dynamic assignment | Open |
| ws4-01 | [ws4-01-scenario-engine-recustomization.md](ws4-01-scenario-engine-recustomization.md) | WS4-01: Scenario engine via CCH partial re-customization | ws4-disruption, rust, performance | M4 Disruption & criticality at scale | Open |
| ws4-02 | [ws4-02-criticality-metrics.md](ws4-02-criticality-metrics.md) | WS4-02: Criticality and consequence metrics suite | ws4-disruption, python, docs | M4 Disruption & criticality at scale | Open |
| ws4-03 | [ws4-03-hazard-fragility-ead.md](ws4-03-hazard-fragility-ead.md) | WS4-03: Hazard exposure, fragility sampling, and EAD/EAL aggregation | ws4-disruption, python, research | M4 Disruption & criticality at scale | Open |
| ws4-04 | [ws4-04-scale-10k-scenarios.md](ws4-04-scale-10k-scenarios.md) | WS4-04: Scale gate — 10k hazard scenarios overnight (two-stage sweep) | ws4-disruption, performance, epic-gate | M4 Disruption & criticality at scale | Open |
| ws5-01 | [ws5-01-jax-implicit-diff-prototype.md](ws5-01-jax-implicit-diff-prototype.md) | WS5-01: JAX prototype — implicit differentiation through user equilibrium | ws5-differentiable, research, python | M5 Differentiable layer | Open |
| ws5-02 | [ws5-02-gradient-od-calibration.md](ws5-02-gradient-od-calibration.md) | WS5-02: Gradient-based bilevel OD calibration to counts | ws5-differentiable, research, python | M5 Differentiable layer | Open |
| ws5-03 | [ws5-03-differentiable-intervention-portfolio.md](ws5-03-differentiable-intervention-portfolio.md) | WS5-03: Differentiable resilience-intervention portfolio optimization (case study) | ws5-differentiable, research | M5 Differentiable layer | Open |
| ws5-04 | [ws5-04-decision-memo-autodiff-architecture.md](ws5-04-decision-memo-autodiff-architecture.md) | WS5-04: ADR — autodiff architecture (pure JAX vs JAX-wrapping-Rust) | ws5-differentiable, research, docs | M5 Differentiable layer | Open |
| ws6-01 | [ws6-01-tutorial-notebooks.md](ws6-01-tutorial-notebooks.md) | WS6-01: End-to-end tutorial notebooks (OD -> assignment -> disruption -> risk) | ws6-release, docs | M6 Release & documentation | Open |
| ws6-02 | [ws6-02-cross-validation-methods-docs.md](ws6-02-cross-validation-methods-docs.md) | WS6-02: Cross-validation report and methods documentation | ws6-release, docs, research | M6 Release & documentation | Open |
| ws6-03 | [ws6-03-release-wheels-ci.md](ws6-03-release-wheels-ci.md) | WS6-03: Binary wheels (maturin/abi3) and release pipeline | ws6-release, rust, python | M6 Release & documentation | Open |

## Milestones

- M1 Foundations & measurement
- M2 Routing engine (CCH/PHAST)
- M3 Equilibrium & quasi-dynamic assignment
- M4 Disruption & criticality at scale
- M5 Differentiable layer
- M6 Release & documentation
