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

Development environment
-----------------------

The project uses ``pixi`` for the development environment. From the repository
root, run project tasks through ``pixi run`` so commands use the pinned
dependencies from ``pyproject.toml`` and ``pixi.lock``.

.. code-block:: console

   pixi install

For a direct editable install without Pixi, use:

.. code-block:: console

   pip install -e .

Pixi is the preferred workflow for contributors and automation agents because it
also installs the configured development and documentation dependencies.

Tests
-----

Run the full test suite with:

.. code-block:: console

   pixi run test

This currently runs:

.. code-block:: console

   python -m pytest tests

Linting and formatting
----------------------

The project currently uses Ruff for linting and formatting.

Run lint checks with:

.. code-block:: console

   pixi run lint

Run formatting with:

.. code-block:: console

   pixi run format

These tasks are defined in ``pyproject.toml``. If a new lint or formatting tool
is added later, add it there and prefer a Pixi task over documenting an ad hoc
command.

Type checking
-------------

No static type checker is currently configured for this project. There is no
``mypy``, ``pyright``, or equivalent Pixi task in ``pyproject.toml`` at the
moment.

For agents and CI automation, do not assume a type-check command exists. Add a
type-checking dependency and a dedicated Pixi task before making type checks a
required validation step.

Documentation
-------------

Build the HTML documentation with:

.. code-block:: console

   pixi run docs

Run doctests with:

.. code-block:: console

   pixi run doctest

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
commercial polygons. This keeps preparation and later allocation runs practical
on the current pure-Python shortest-path implementation. Use
``--max-zones-per-landuse`` to scale the OD matrix up or ``0`` to use all
downloaded zones:

.. code-block:: console

   pixi run prepare-benchmark-data -- --max-zones-per-landuse 25 --overwrite

Use ``--overwrite`` when regenerating the benchmark dataset. ``benchmark_data/``
and OSMnx ``cache/`` output are ignored by Git.

Benchmarking scripts
--------------------

The repository includes a stdlib-only benchmark harness for integration timing
of the two command-line flow scripts:

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

Experimental Rust extension
---------------------------

The repository includes an experimental Rust scaffold for future high
performance allocation and disruption algorithms. The current Python
implementation remains the default runtime path. Build the native extension
only when developing or benchmarking the Rust code:

.. code-block:: console

   pixi run rust-build

This task runs ``maturin develop --manifest-path Cargo.toml`` and installs the
PyO3 module as ``transport_flow_model._rust`` in the Pixi environment. The
public helper module is ``transport_flow_model.rust``.

The extension is structured in two layers:

- A Python-independent Rust core for graph, OD, allocation, disruption, unit
  tests, and Criterion benchmarks.
- A PyO3 wrapper that exchanges in-memory Arrow IPC streams with Python.

The Arrow IPC boundary keeps file I/O in the wrapper language. Python callers
can pass ``pyarrow.Table``, ``pyarrow.RecordBatch``, or ``pandas.DataFrame`` to
``transport_flow_model.rust.allocate_arrow`` and
``transport_flow_model.rust.disrupt_arrow``. Other language wrappers can target
the same Arrow stream schemas without depending on Python data-frame internals.

Run Rust unit tests with:

.. code-block:: console

   pixi run rust-test

Run Rust microbenchmarks with:

.. code-block:: console

   pixi run rust-bench

The scaffold intentionally avoids extra graph/routing crates for now. Add new
Rust dependencies only when they replace substantial local complexity or are
needed for a specific algorithmic feature.

Pixi command reference
----------------------

The current Pixi tasks are:

.. list-table::
   :header-rows: 1

   * - Command
     - Purpose
   * - ``pixi run test``
     - Run ``python -m pytest tests``.
   * - ``pixi run lint``
     - Run Ruff lint checks.
   * - ``pixi run format``
     - Format Python code with Ruff.
   * - ``pixi run docs``
     - Build Sphinx HTML documentation.
   * - ``pixi run doctest``
     - Run Sphinx doctests.
   * - ``pixi run prepare-benchmark-data``
     - Generate the ignored West Yorkshire benchmark input dataset.
   * - ``pixi run benchmark-scripts-smoke``
     - Run a quick script-level benchmark using ``config.example.json``.
   * - ``pixi run benchmark-scripts``
     - Run script-level timing benchmarks and write CSV/JSON outputs.
   * - ``pixi run profile-flow-scripts``
     - Write flamegraphs for both allocation and disruption scripts.
   * - ``pixi run profile-flow-allocation``
     - Write a flamegraph for allocation only.
   * - ``pixi run profile-flow-disruptions``
     - Write a flamegraph for disruption only.
   * - ``pixi run rust-build``
     - Build and install the experimental PyO3 Rust extension.
   * - ``pixi run rust-test``
     - Run Rust unit tests for the native algorithm scaffold.
   * - ``pixi run rust-bench``
     - Run Criterion benchmarks for the native algorithm scaffold.
