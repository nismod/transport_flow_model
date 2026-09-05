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

## Result: done

`PreparedOdPaths` gained a CSR index over internal link ids — `indptr` plus `rows_by_edge`,
not `Vec<Vec<u32>>`, which would be one heap allocation per link. The counting pass is free:
`prepare_od_paths_batches` already walked every path's `edge_path` to accumulate
`current_edge_flows`, so only the fill pass is new. `for_failed_edges` now gathers rows from
the index instead of scanning, which also deletes the `is_affected` test and the
`vec![false; n_links]` it needed. `current_edge_flows` is no longer cloned per scenario;
`disrupt_with_preprocessed` borrows it.

Measured on this machine, min-of-7:

| instance | path rows | per scenario | no-op | 10k scenarios |
| --- | --- | --- | --- | --- |
| anaheim | 1 406 | 2.2 → **0.117** ms | 2.1 → **0.070** ms | 0.4 → **0.02** min |
| chicago-sketch | 93 513 | 2.3 → **0.228** ms | 2.1 → **0.101** ms | 0.4 → **0.04** min |

The gathered rows are sorted, which does double duty: it drops the duplicate a path picks up
when it uses two failed links, and it restores path-table row order. That order is
observable — `demands_from_affected_flows` preserves input order into
`allocate_unconstrained`, which keeps within-origin insertion order — so gathering
link-by-link would have permuted `rerouted_flows`. Removing the sort fails
`test_affected_flows_keep_path_table_row_order` and
`test_path_using_two_failed_links_is_affected_once`.

### Cost

Building the index adds ~5 ms to a 51 ms `prepare_disruption` on chicago-sketch, so the
one-shot `core.disrupt` path — which prepares a handle and runs a single scenario — goes
from 55.4 ms to 61.1 ms, **about 10% slower**. Accepted rather than deferred behind a
`OnceCell`: the one-shot path is the legacy `model.py` wrapper and the tests, while every
scenario sweep, which is what WS4-04 gates on, is 10x faster.

### Still open

A scenario's floor is now O(links), not O(paths): `edges_with_flows_removed_from_affected_paths`
copies the whole `Vec<Edge>` per scenario, which is 0.10 ms of the 0.23 ms on chicago-sketch
and would be ~480 MB of copying per scenario at 10M edges. `disruption._affects_flow` is
also still an O(links) `pc.is_in` per scenario in Python. Neither was in scope here; both
belong with ws4-04.
