# Transport Flow Model

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19566285.svg)](https://doi.org/10.5281/zenodo.19566285)

This Python package implements iterative capacity-constrained network flow
allocation. It routes flows through networks sequentially until link capacities
are exhausted, providing a simplified tool for infrastructure risk and
resilience analysis.

It is part of the open-source [National Infrastructure Systems Model (NISMOD)
ecosystem](https://github.com/nismod) developed by the [Oxford Programme for
Sustainable Infrastructure Systems (OPSIS)](https://opsis.eci.ox.ac.uk) at the
University of Oxford.

To run the model:

```bash
pip install -e .
python scripts/flow_model/flow_allocation.py ./config.example.json
python scripts/flow_model/flow_disruptions.py ./config.example.json
```

## Development

We recommend the use of [`pixi`](https://pixi.prefix.dev) to manage a
development environment.

To run the tests (see `[tool.pixi.tasks]` within `pyproject.toml`):

```bash
pixi run test
```

Useful Pixi commands:

| Command                             | Purpose                                                                         |
| ----------------------------------- | ------------------------------------------------------------------------------- |
| `pixi run test`                     | Run the pytest suite.                                                           |
| `pixi run lint`                     | Run Ruff lint checks.                                                           |
| `pixi run format`                   | Format Python code with Ruff.                                                   |
| `pixi run docs`                     | Build the Sphinx HTML documentation.                                            |
| `pixi run doctest`                  | Run Sphinx doctests in the documentation.                                       |
| `pixi run prepare-benchmark-data`   | Download and prepare the generated West Yorkshire benchmark dataset.            |
| `pixi run benchmark-scripts-smoke`  | Run a quick integration benchmark against `config.example.json`.                |
| `pixi run benchmark-scripts`        | Time the allocation and disruption scripts and write benchmark CSV/JSON output. |
| `pixi run profile-flow-scripts`     | Write CPU/time flamegraphs for allocation and disruption.                       |
| `pixi run profile-flow-allocation`  | Write a flamegraph for `flow_allocation.py`.                                    |
| `pixi run profile-flow-disruptions` | Write a flamegraph for `flow_disruptions.py`.                                   |
| `pixi run extension-build`          | Build and install the experimental PyO3 Rust extension in the Pixi environment. |
| `pixi run extension-test`           | Run unit tests for the extension.                                               |
| `pixi run extension-bench`          | Run Criterion benchmarks for the extension.                                     |

To add a new package dependency, make sure to use `--pypi` to include it
in the `pyproject.toml` `dependencies` table:

```bash
pixi add --pypi geopandas
```

To add a new development dependency, make sure to use `--pypi` and `--feature
dev` to include it in the `pyproject.toml` `[dependency-groups] dev` table:

```bash
pixi add --pypi --feature dev pytest
```

## Acknowledgments

This research received funding from the UK FCDO Climate Compatible Growth
Programme. The views expressed here do not necessarily reflect the UK
government's official policies.
