# Functionality

- OD estimation
  - proportional downscaling
  - network radiation model
  - ¿ gravity model
- disruption/reassignment
  - unconstrained
  - capacity-constrained
  - speed-flow relationship
  - ¿ diversity of route-choice (logit)
  - ¿ accessibility disruption
  - source to destination set

# Goals

- performance
  - macro scale with granularity
  - should be ~minutes for ~10M-edge / ~1M OD
- usability
  - clear documentation and tutorials
  - refer to papers for methods motivation
  - import and use in current projects

# Benchmarking and profiling

Three layers, from micro to end-to-end:

- `pixi run bench` — assignment benchmark harness
  (`scripts/benchmark_assignment.py`): runs each (instance, method, threads)
  case in a fresh subprocess and records instance, method, threads,
  iterations, gap trajectory, wall time and peak RSS to parquet, plus a
  markdown report and gap-vs-time plots, under
  `benchmark_results/assignment/`. `--suite small` (vendored SiouxFalls) or
  `--suite large` (downloaded TNTP instances); `--baseline`/`--write-baseline`
  for regression checks.
- `pixi run extension-bench` — Criterion micro-benchmarks of the Rust core
  (`core/benches/core.rs`).
- `pixi run profile-flow-scripts` — py-spy flamegraphs of the end-to-end flow
  scripts; pyinstrument is also available in the dev environment for ad-hoc
  profiling.

Wall time for equilibrium methods is only meaningful together with solution
quality, so every benchmark case records the *relative gap*
(`transport_flow_model.relative_gap`): the excess of total travel time over
total shortest-path travel time at congested costs. Boyce, Ralevic-Dekic &
Bar-Gera (2004, doi:10.1061/(ASCE)0733-947X(2004)130:1(49)) recommend gaps
of 1e-4 or better before flow differences between scenarios are trustworthy.

CI (`.github/workflows/perf.yml`) benchmarks the small suite on every PR and
push to main, failing on >20% median wall-time slowdown or on relative-gap
regression at a fixed iteration budget against the baseline cached from the
latest main build; a nightly job runs the large instances and uploads
parquet results and plots as artifacts.

# Software and Literature review (WIP)

`spopt-r` spatial optimization algorithms for R

- http://walker-data.com/spopt-r/index.html
- regionalization, facility location, route optimization, and corridor routing
- Rust backend for graph and routing algorithms

`traffic-flow` Python package for macroscopic transport modelling

- https://github.com/petervanya/traffic-flow
- trip generation, distribution, assignment
- calibration to observed flows

`STAQ` (Static Traffic Assignment with Queuing)

- https://doi.org/10.1080/23249935.2018.1453561
- https://doi.org/10.1016/j.trb.2014.07.001
- capacity-constrained traffic assignment model designed for strategic planning
  - limits link flows to physical road capacities
  - represents congestion by shifting excess traffic into vertical or horizontal residual queues.

`flownet` (R package for network processing, route enumeration, and PSL
traffic assignment)

- https://github.com/SebKrantz/flownet
- https://github.com/SebKrantz/OptimalAfricanRoads
- https://doi.org/10.32614/CRAN.package.flownet
- https://sebkrantz.github.io/Rblog/2026/02/09/introducing-flownet-efficient-transport-modeling-in-r/

`AequilibraE` (Python package for comprehensive transportation modeling and
user-equilibrium traffic assignment)

- https://github.com/AequilibraE/aequilibrae
- https://doi.org/10.21949/1527574

`cppRouting` (R/C++ routing engine featuring contraction hierarchies and
traffic assignment solvers)

- https://github.com/vlarmet/cppRouting
- https://doi.org/10.32614/CRAN.package.cppRouting

`dodgr` (R package for many-to-many pairwise distances on dual-weighted
directed graphs)

- https://github.com/UrbanAnalyst/dodgr
- https://github.com/ATFutures/dodgr
- https://doi.org/10.32866/6945

`Madina` (Python package by MIT City Form Lab for pedestrian/bicycle trip
routing and Urban Network Analysis)

- https://github.com/City-Form-Lab/madina
- https://doi.org/10.1016/j.jtrangeo.2025.

`UXsim` (Lightweight macroscopic and mesoscopic traffic flow simulator in pure
Python)

- https://github.com/toruseo/UXsim
- https://doi.org/10.21105/joss.07617

`vehicle-routing-solver` (Python wrapper for Google OR-Tools capacitated
vehicle routing with time windows)

- https://github.com/KNCn23/vehicle-routing-solver

`ch` (Go-based implementation of contraction hierarchies, bidirectional
Dijkstra, and isochrones)

- https://github.com/LdDl/ch

Path-Sized Logit (PSL)

- Ben-Akiva, M., & Bierlaire, M. (1999). Discrete Choice Methods and Their
  Applications to Short-Term Travel Decisions. In Handbook of Transportation
  Science (pp. 5-34).
- https://doi.org/10.1007/978-1-4615-5203-1_2

Optimal African Roads Network Study

- Krantz, S. (2024). Optimal Investments in Africa's Road Network. World Bank
  Policy Research Working Paper 10893.
- https://doi.org/10.1596/1813-9450-10893

Contraction Hierarchies (CH) Algorithm

- Geisberger, R., Sanders, P., Schultes, D., & Delling, D. (2008). Contraction
  Hierarchies: Faster and Simpler Hierarchical Routing in Road Networks. In WEA
  2008, LNCS 5034 (pp. 319-333).
- https://doi.org/10.1007/978-3-540-68552-4_24

Dial's Algorithm B (Bush-Based Static User Equilibrium)

- Dial, R. B. (2006). A path-based user-equilibrium traffic assignment algorithm
  that obviates path storage and enumeration. Transportation Research Part B:
  Methodological, 40(10), 917-936.
- https://doi.org/10.1016/j.trb.2006.02.008

Bar-Gera's Origin-Based Algorithm (OBA)

- Bar-Gera, H. (2002). Origin-Based Algorithm for the Traffic Assignment
  Problem. Transportation Science, 36(4), 398-417.
- https://doi.org/10.1287/trsc.36.4.398.549

Capacity Constrained Route Planner (CCRP) Evacuation Algorithm

- Lu, Q., George, B., & Shekhar, S. (2005). Capacity Constrained Routing
  algorithms for evacuation planning: A summary of results. In SSTD 2005, LNCS
  3633 (pp. 291-307).
- https://doi.org/10.1007/11535331_17

Quasi-Dynamic Traffic Assignment with Residual Point Queues

- Bliemer, M. C. J., & Raadsen, M. P. H. (2020). Quasi-dynamic traffic
  assignment with residual point queues. Transportation Research Part B:
  Methodological.
- https://doi.org/10.1080/23249935.2020.1720862

FluxNet Neural PDE Framework

- Exact discrete conservation using modular capacity-constrained transport
  heads.
- https://arxiv.org/html/2602.01941

Dataset of traffic networks and assignment for 20 cities

- https://doi.org/10.6084/m9.figshare.24235696
- could use for performance benchmarking and validation/comparison of assignment methods

# Versioning

The package follows semantic versioning; the public API is the set of
top-level `transport_flow_model` exports plus the JSON config schema and
script CLIs. Breaking changes and deprecations are recorded in
`CHANGELOG.md`, and the full policy (including result bit-stability across
patch releases and the deprecation window) is documented in
`docs/source/reference/versioning.rst`.
