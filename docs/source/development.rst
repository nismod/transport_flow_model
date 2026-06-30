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
