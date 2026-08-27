Development
===========

Source code
-----------

The source repository is hosted on GitHub:

.. code-block:: console

   git clone git@github.com:nismod/transport_flow_model.git
   cd transport_flow_model

If SSH access is not configured, use the HTTPS URL instead:

.. code-block:: console

   git clone https://github.com/nismod/transport_flow_model.git
   cd transport_flow_model

Environment, tasks and pull requests
------------------------------------

``CONTRIBUTING.md`` in the repository root is the canonical reference for the
development environment, the full list of Pixi tasks, and what a pull request
is expected to contain. In short:

.. code-block:: console

   pixi install
   pixi run extension-build
   pixi run test

``ARCHITECTURE.md`` describes how the modules fit together, and ``docs/adr/``
records the decisions behind the design — start with ADR-0001 if you are
adding an assignment method.

No static type checker is currently configured for this project. There is no
``mypy``, ``pyright``, or equivalent Pixi task in ``pyproject.toml`` at the
moment. For agents and CI automation, do not assume a type-check command
exists: add a type-checking dependency and a dedicated Pixi task before making
type checks a required validation step.

Benchmark data preparation
--------------------------

The generated West Yorkshire benchmark dataset is created by:

.. code-block:: console

   pixi run prepare-benchmark-data

This task runs ``scripts/benchmark_preparation.py``. It uses OSMnx to download
the West Yorkshire driving road network and residential/commercial land-use
polygons. The script writes generated data under
``benchmark_data/west_yorkshire/``:

- ``osmnx_road_network.gpkg`` with ``nodes`` and ``edges`` layers.
- ``osmnx_landuse_zones.gpkg`` with all downloaded land-use polygons and the
  subset used for OD generation.
- ``processed_data/network/network.csv`` for flow allocation.
- ``processed_data/od/od.csv`` from the radiation-model OD estimate.
- ``processed_data/damages/failure_set.csv`` for disruption benchmarking.

The default OD generation uses the 10 largest residential and 10 largest
commercial polygons. Use ``--max-zones-per-landuse`` to scale the OD matrix up
or ``0`` to use all downloaded zones:

.. code-block:: console

   pixi run prepare-benchmark-data -- --max-zones-per-landuse 25 --overwrite

Use ``--overwrite`` when regenerating the benchmark dataset. ``benchmark_data/``
and OSMnx ``cache/`` output are ignored by Git.

``config.west_yorkshire.json`` refers to this generated data, so
``tests/test_config.py`` does not fully pass until it has been prepared.

Assignment benchmarks
---------------------

``scripts/benchmark_assignment.py`` (``pixi run bench``) is the benchmark
harness for assignment methods. It runs each (instance, method, threads) case
in a fresh subprocess and records the gap trajectory, wall time and peak RSS:

.. code-block:: console

   pixi run bench --suite small
   pixi run bench --suite large --repeats 5

``--suite small`` uses the vendored SiouxFalls instance; ``--suite large``
downloads TNTP instances into the dataset cache. Pass ``--baseline
summary.json`` to fail on a wall-time or relative-gap regression, and
``--write-baseline`` to record one.

Wall time for equilibrium methods is only meaningful together with solution
quality, so every case records the relative gap
(:func:`transport_flow_model.relative_gap`) alongside its timings.

Benchmarking the flow scripts
-----------------------------

The repository also includes a stdlib-only benchmark harness for integration
timing of the two command-line flow scripts:

- ``scripts/flow_model/flow_allocation.py``
- ``scripts/flow_model/flow_disruptions.py``

Run the smoke benchmark with the example config:

.. code-block:: console

   pixi run benchmark-scripts-smoke

Run the default benchmark task with:

.. code-block:: console

   pixi run benchmark-scripts

The benchmark script defaults to ``config.example.json`` and accepts additional
configs with repeated ``--config`` arguments:

.. code-block:: console

   python scripts/benchmark_flow_scripts.py \
     --config config.example.json \
     --config path/to/larger-config.json \
     --repeats 3

Each benchmark run copies the configured input data into a temporary working
directory and writes script outputs to temporary results paths. This keeps the
tracked example data and results directories unchanged while still measuring the
script workflow, including CSV input/output.

Benchmark outputs are written under ``benchmark_results/``:

- ``flow_script_benchmark.csv`` appends timing history.
- ``flow_script_benchmark_latest.json`` stores the latest run.

``benchmark_results/`` is ignored by Git. Use ``--keep-workdirs`` when debugging
script outputs from an individual benchmark run.

Useful benchmark options:

.. code-block:: console

   python scripts/benchmark_flow_scripts.py --help
   python scripts/benchmark_flow_scripts.py --repeats 5
   python scripts/benchmark_flow_scripts.py --max-seconds 30
   python scripts/benchmark_flow_scripts.py --keep-workdirs

To benchmark the generated West Yorkshire data, prepare the data first and pass
the West Yorkshire config:

.. code-block:: console

   pixi run prepare-benchmark-data -- --overwrite
   python scripts/benchmark_flow_scripts.py --config config.west_yorkshire.json

Profiling scripts
-----------------

CPU/time flamegraphs are generated with ``py-spy`` through
``scripts/profile_flow_scripts.py``. The default profile config is
``config.west_yorkshire.json`` and output SVGs are written to
``profile_results/``:

.. code-block:: console

   pixi run profile-flow-scripts

This writes:

- ``profile_results/flow_allocation.svg``
- ``profile_results/flow_disruptions.svg``

The default profile records a bounded 30-second sampling window per script at
10 samples per second. This avoids turning large benchmark profiles into much
longer full-script runs. Increase the duration or sampling rate when more detail
is needed:

.. code-block:: console

   pixi run profile-flow-scripts -- --duration 60 --rate 25

Profile one script at a time with:

.. code-block:: console

   pixi run profile-flow-allocation
   pixi run profile-flow-disruptions

When profiling only disruptions, the wrapper runs allocation first so
``flow_disruptions.py`` has the required ``flow_od_paths`` inputs. If those
outputs already exist and should be reused, pass ``--skip-setup-allocation``:

.. code-block:: console

   pixi run profile-flow-disruptions -- --skip-setup-allocation

``profile_results/`` is ignored by Git.

Rust extension
--------------

The package uses a Rust core, and it is not optional: importing
``transport_flow_model`` reaches ``transport_flow_model.core``, which imports
the compiled extension. Build it when setting up a checkout, and rebuild it
after changing anything under ``core/``:

.. code-block:: console

   pixi run extension-build

This task runs ``maturin develop --manifest-path Cargo.toml`` and installs the
PyO3 module as ``transport_flow_model._core`` in the Pixi environment.

The extension is structured in two layers:

- A Python-independent Rust core for graph, OD, allocation, disruption, unit
  tests, and Criterion benchmarks (``core/src/core.rs``).
- A PyO3 wrapper that exchanges in-memory Arrow data with Python
  (``core/src/lib.rs`` and ``core/src/arrow_ffi.rs``).

Data crosses the boundary through the Arrow C stream interface: Python passes
Arrow tables in, and Rust returns PyCapsule-wrapped record batch streams, with
no serialization step and no copy of the buffers. Only
``transport_flow_model.core`` imports the extension; it exposes
``core.allocate``, ``core.disrupt``, ``core.shortest_paths_from`` and
``core.version``. Other language wrappers can target the same Arrow schemas
without depending on Python data-frame internals. See ADR-0002 in
``docs/adr/``.

Run unit tests with:

.. code-block:: console

   pixi run extension-test

Run microbenchmarks with:

.. code-block:: console

   pixi run extension-bench

The scaffold intentionally avoids extra graph/routing crates for now. Add new
Rust dependencies only when they replace substantial local complexity or are
needed for a specific algorithmic feature.
