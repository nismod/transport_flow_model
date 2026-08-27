# Transport Flow Model

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19566285.svg)](https://doi.org/10.5281/zenodo.19566285)
[![docs](https://github.com/nismod/transport_flow_model/actions/workflows/docs.yml/badge.svg)](https://nismod.github.io/transport_flow_model)

This Python package models flows on transport networks for infrastructure risk
and resilience analysis. It assigns origin-destination demand to network
routes, evaluates what happens to those flows when links are disrupted, and
quantifies the resulting rerouting cost and loss of access.

Performance-critical routing and allocation run in a Rust core; data is handled
as Apache Arrow tables throughout.

It is part of the open-source [National Infrastructure Systems Model (NISMOD)
ecosystem](https://github.com/nismod) developed by the [Oxford Programme for
Sustainable Infrastructure Systems (OPSIS)](https://opsis.eci.ox.ac.uk) at the
University of Oxford.

## Installation

The Rust extension is required, so an editable install also needs a build step
(and a Rust toolchain):

```bash
pip install -e .
maturin develop --release
```

Contributors should use [`pixi`](https://pixi.prefix.dev) instead — see
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## Usage

```python
from transport_flow_model import Network, Demand, assign, disrupt

network = Network.from_dataframe(links)           # edge_from, edge_to, edge_id, cost, ...
demand = Demand.from_dataframe(od)                # origin_id, destination_id, value

result = assign(network, demand, method="sequential", include_paths=True)
result.link_flows      # per-link flow
result.skims           # per-OD-pair cost

summary = disrupt(network, scenarios, base=result).summary()   # loss per scenario
```

Or run the config-driven scripts:

```bash
python scripts/flow_model/flow_allocation.py ./config.example.json
python scripts/flow_model/flow_disruptions.py ./config.example.json
```

## Documentation

- [User guides and API reference](https://nismod.github.io/transport_flow_model)
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — how the code fits together
- [`docs/adr/`](docs/adr/) — architecture decision records
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — development environment, tasks and PR
  expectations
- [`DEVELOPMENT.md`](DEVELOPMENT.md) — planned functionality, benchmarking
  approach and literature review
- [`CHANGELOG.md`](CHANGELOG.md) — release notes; the public API is versioned

## Acknowledgments

This research received funding from the UK FCDO Climate Compatible Growth
Programme. The views expressed here do not necessarily reflect the UK
government's official policies.
