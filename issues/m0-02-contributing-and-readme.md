## Context
`README.md` carries the pixi task table and the development instructions, and its description
of the package — "routes flows through networks sequentially until link capacities are
exhausted" — describes the package as it was before the v0 API landed. There is no
`CONTRIBUTING.md`, so PR expectations are unwritten.

## Task
- Add `CONTRIBUTING.md`: environment setup, the pixi task table moved out of `README.md`, and
  PR expectations — tests, lint/format clean, a `CHANGELOG.md` entry under `## [Unreleased]`,
  and a benchmark run for anything touching the assignment loop.
- Trim `README.md` and update its description to what the package does now.

## Acceptance criteria
- The task table has one canonical home; the docs do not carry a third, diverging copy.
- The task table matches `pyproject.toml`.
- A developer can get from a fresh clone to a green test run using `README.md` and
  `CONTRIBUTING.md` alone.
