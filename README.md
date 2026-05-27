# Transport Flow Model

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19566285.svg)](https://doi.org/10.5281/zenodo.19566285)

Scripts to run a transport flow model with capacity constraints
The model assumes flows along edges without constrains, till they reach capacity

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
