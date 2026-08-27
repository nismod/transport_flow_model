# Contributing

Thanks for working on `transport_flow_model`. This file covers the development
environment, the task runner, and what a pull request is expected to contain.

For an orientation to the codebase, read [`ARCHITECTURE.md`](ARCHITECTURE.md)
and then [`docs/adr/`](docs/adr/) — in particular
[ADR-0001](docs/adr/0001-assignment-methods-are-registered-backends.md) if you
are adding an assignment method.

## Development environment

We recommend [`pixi`](https://pixi.prefix.dev), which installs the pinned
development and documentation dependencies from `pyproject.toml` and
`pixi.lock`:

```bash
pixi install
pixi run extension-build   # build the Rust extension; required
pixi run test
```

The Rust extension is **not optional**: `import transport_flow_model` reaches
`transport_flow_model.core`, which imports the compiled `_core` module, so a
checkout without it cannot import the package at all. Rebuild it whenever you
change anything under `core/`.

Without pixi, reproduce what CI does (see `.github/workflows/test.yml`) — this
needs a Rust toolchain:

```bash
python -m venv .venv
. .venv/bin/activate
pip install --group dev --group doc -e .   # needs pip >= 25.1 for --group
maturin develop --release
```

### Dependencies

Add a runtime dependency with `--pypi` so it lands in the `pyproject.toml`
`dependencies` table:

```bash
pixi add --pypi geopandas
```

Add a development dependency with `--pypi --feature dev`, so it lands in
`[dependency-groups] dev`:

```bash
pixi add --pypi --feature dev pytest
```

## Tasks

Tasks are defined in `pyproject.toml` under `[tool.pixi.feature.*.tasks]`.
This table is the canonical list.

| Command | Purpose |
| --- | --- |
| `pixi run test` | Run the pytest suite (`python -m pytest tests`). |
| `pixi run lint` | Run Ruff lint checks. |
| `pixi run format` | Format Python code with Ruff. |
| `pixi run docs` | Build the Sphinx HTML documentation. |
| `pixi run doctest` | Run the Sphinx doctests. |
| `pixi run bench` | Assignment benchmark harness: relative gap, wall time and peak RSS per case (`scripts/benchmark_assignment.py`). |
| `pixi run extension-build` | Build and install the PyO3 Rust extension into the environment. |
| `pixi run extension-test` | Run the Rust unit tests (`cargo test`). |
| `pixi run extension-bench` | Run the Criterion micro-benchmarks (`cargo bench`). |
| `pixi run prepare-benchmark-data` | Download and prepare the generated West Yorkshire benchmark dataset. |
| `pixi run benchmark-scripts-smoke` | Quick script-level benchmark against `config.example.json`. |
| `pixi run benchmark-scripts` | Time the allocation and disruption scripts, writing CSV/JSON output. |
| `pixi run profile-flow-scripts` | py-spy flamegraphs for both flow scripts. |
| `pixi run profile-flow-allocation` | Flamegraph for `flow_allocation.py` only. |
| `pixi run profile-flow-disruptions` | Flamegraph for `flow_disruptions.py` only. |

Pass extra arguments after the task name, e.g.
`pixi run bench --suite small --repeats 1`.

If you add a lint, format or benchmark tool, add a task for it here rather than
documenting an ad hoc command.

## Pull requests

Every pull request should:

- **Be one logical change.** A fix, a refactor and a doc rewrite belong in
  three PRs, not one.
- **Pass `pixi run test`.** New behaviour needs a test; a bug fix needs a test
  that fails without it.
- **Pass `pixi run lint`, and be formatted** with `pixi run format`.
- **Add a `CHANGELOG.md` entry** under `## [Unreleased]`, in the appropriate
  `### Added` / `### Changed` / `### Removed` / `### Fixed` section. Anything
  that changes the public API — the top-level `transport_flow_model` exports,
  the JSON config schema, or the script CLIs — must say so; see
  `docs/source/versioning.rst` for the versioning and deprecation policy.
- **Build the docs** with `pixi run docs` and `pixi run doctest` if you touched
  anything under `docs/`, any docstring that appears in the API reference, or
  any behaviour a doctest exercises.

Additionally, **anything touching the assignment loop** — `assignment.py`,
`convergence.py`, or the Rust core — should include a benchmark run:

```bash
pixi run bench --suite small
```

Report the relative gap and wall time in the PR. CI runs the same small suite
on every PR and fails on a >20% median wall-time slowdown or a relative-gap
regression against the baseline from the latest main build; a nightly job runs
the larger instances. Wall time alone is not a result — for equilibrium
methods it only means something alongside the gap it achieved.

### Architecture decisions

If your change establishes a contract other code has to satisfy, or changes a
boundary between layers, add an ADR in the same PR. See
[`docs/adr/README.md`](docs/adr/README.md).
