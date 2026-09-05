## Context
`core.prepare_disruption(links, paths)` parses the network and the baseline path set once
for a whole scenario run, which took a chicago-sketch scenario from 72ms to 2.3ms. What is
left is a linear scan: `PreparedOdPaths::for_failed_edges` walks **every** parsed path row
to find the ones whose `edge_path` touches a failed link, and clones
`current_edge_flows` for each scenario.

On chicago-sketch (93 513 path rows) a scenario that removes a link carrying no flow still
costs 2.1ms of the 2.3ms — the work is proportional to the size of the path set, not to how
much flow the scenario actually affects. Most scenarios in a hazard sweep affect a small
fraction of paths, so the scan dominates.

To reproduce:

```python
import time
from transport_flow_model import Network, Demand, datasets, assign, core

instance = datasets.load_tntp("chicago-sketch")
network, demand = Network.from_tntp(instance), Demand.from_tntp(instance)
base = assign(network, demand, include_paths=True)
prepared = core.prepare_disruption(network.to_table(), base.paths)
one_link = [network.to_table()["edge_id"].to_pylist()[0]]

def best(failed, n=5):
    prepared.scenario(failed)
    return min(
        (lambda s=time.perf_counter(): (prepared.scenario(failed), time.perf_counter() - s)[1])()
        for _ in range(n)
    )

print(best(one_link), best([]))  # per scenario, no-op scenario
```


| instance | links | path rows | per scenario | no-op scenario |
| --- | --- | --- | --- | --- |
| anaheim | 914 | 1 406 | 0.2 ms | 0.1 ms |
| chicago-sketch | 2950 | 93 513 | 2.3 ms | 2.1 ms |

At 10k scenarios that is about 0.4 min on a 2950-link network. WS4-04 targets 10k scenarios
at national scale, where the path set is orders of magnitude larger.

## Task
- Build a link -> paths index once inside `PreparedOdPaths` (for each link, the row indices
  of the paths using it), so a scenario visits only the paths its failed links touch.
- Avoid the per-scenario `current_edge_flows.clone()`: it is read-only input to
  `core::disrupt_with_preprocessed`, so it can be borrowed.
- Re-measure and update the numbers in `ARCHITECTURE.md` ("Remaining headroom in the
  scenario loop").
- Consider whether `disruption._affects_flow` in Python becomes redundant once a no-op
  scenario is nearly free — it exists to skip scenarios that carry no baseline flow.

## Acceptance criteria
- A no-op scenario's cost no longer scales with the number of path rows.
- `tests/test_disruption.py` and `tests/test_disruption_api.py` pass unmodified, and
  `test_prepared_disruption_matches_the_unprepared_call` still holds.
- The index is built once per `prepare_disruption`, not per scenario.

## References
- `core/src/arrow_ffi.rs` — `PreparedOdPaths` and `for_failed_edges`.
- `core/src/core.rs` — `disrupt_with_preprocessed`, which already takes preprocessed inputs.
- `ws4-04-scale-10k-scenarios.md` — the scale gate this feeds.
