#!/usr/bin/env python
"""Measure what evaluating the relative gap costs per assignment iteration.

:func:`transport_flow_model.relative_gap` loops in Python over unique
origins, calling :func:`transport_flow_model.core.shortest_paths_from` once
per origin. An iterative equilibrium method builds equivalent shortest-path
trees inside ``core.allocate`` for its all-or-nothing step, so checking
convergence every iteration repeats that work.

:func:`relative_gap` now asks for every OD pair in one
:func:`transport_flow_model.core.skim` call, which parses the network once
and builds one tree per origin. The ``loop`` column below still times the
per-origin ``shortest_paths_from`` calls it used to make, as a reference
point for how much that cost.

The two halves of an iteration are measured separately:

- ``t_aon``: one all-or-nothing pass, ``assign(..., method="sequential")``,
  which is what an equilibrium iteration costs apart from the averaging.
- ``t_gap``: one ``relative_gap(...)`` call on the resulting flows.

and reported as ``t_gap / (t_aon + t_gap)`` — the share of a hypothetical
iteration spent deciding whether to stop.

That share is a proxy, and for ``"msa"`` it is now superseded: MSA takes the
shortest-path term straight from the all-or-nothing load it performs anyway,
so its gap costs nothing and its true share is zero. The ``msa (ms/pass)``
column times a real MSA run instead. The proxy is still what a method that
*does* call :func:`~transport_flow_model.relative_gap` per iteration would
pay, so it is kept.

Usage::

    python scripts/profile_gap_cost.py
    python scripts/profile_gap_cost.py --instance siouxfalls --repeats 5
    python scripts/profile_gap_cost.py --instance chicago-sketch --json out.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

#: Instances measured by default: one small vendored, one large downloaded.
DEFAULT_INSTANCES = ("siouxfalls", "chicago-sketch")

#: Passes to time an MSA run over. Enough to average out startup, far fewer
#: than convergence needs — see ``issues/ws2-02-msa-baseline.md``.
MSA_PASSES = 20


def main() -> int:
    args = parse_args()
    rows = []
    for name in args.instance or DEFAULT_INSTANCES:
        try:
            rows.append(measure(name, repeats=args.repeats))
        except Exception as error:  # noqa: BLE001 - report and carry on
            print(f"{name}: skipped ({type(error).__name__}: {error})", file=sys.stderr)
    if not rows:
        print("No instances measured", file=sys.stderr)
        return 1
    print(render(rows))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=2))
        print(f"\nWritten to {args.json}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--instance",
        action="append",
        help="Dataset name (repeatable); see transport_flow_model.datasets.",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--json", type=Path, help="Also write measurements here.")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    return args


def measure(name: str, *, repeats: int) -> dict:
    """Time one all-or-nothing pass and one gap evaluation on ``name``."""
    import numpy as np

    from transport_flow_model import Demand, Network, assign, datasets, relative_gap
    from transport_flow_model import core

    instance = datasets.load_tntp(name)
    network = Network.from_tntp(instance)
    demand = Demand.from_tntp(instance)
    best_known = datasets.BEST_KNOWN.get(name)
    distance_cost = best_known.distance_cost if best_known is not None else 0.0

    origins = np.unique(demand.to_table()["origin_id"].to_numpy(zero_copy_only=False))

    aon_times = []
    gap_times = []
    tree_times = []
    msa_times = []
    gap = None
    msa_gap = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = assign(network, demand, method="sequential")
        aon_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        gap = relative_gap(network, demand, result, distance_cost=distance_cost)
        gap_times.append(time.perf_counter() - start)

        # What the same trees cost one origin at a time, as relative_gap
        # used to ask for them.
        links = network.to_table()
        start = time.perf_counter()
        for origin in origins:
            core.shortest_paths_from(links, int(origin), directed=True)
        tree_times.append(time.perf_counter() - start)

        # A real iterative method, whose gap costs nothing extra.
        start = time.perf_counter()
        msa = assign(
            network,
            demand,
            method="msa",
            max_iterations=MSA_PASSES,
            target_gap=0.0,
            distance_cost=distance_cost,
        )
        msa_times.append((time.perf_counter() - start) / msa.provenance.iterations)
        msa_gap = msa.provenance.relative_gap

    t_aon = statistics.median(aon_times)
    t_gap = statistics.median(gap_times)
    t_trees = statistics.median(tree_times)
    t_msa = statistics.median(msa_times)
    return {
        "instance": name,
        "n_links": network.n_links,
        "n_nodes": network.n_nodes,
        "n_od_pairs": demand.n_pairs,
        "n_origins": int(origins.size),
        "repeats": repeats,
        "aon_s": t_aon,
        "gap_s": t_gap,
        "per_origin_loop_s": t_trees,
        "msa_s_per_pass": t_msa,
        "msa_passes": MSA_PASSES,
        "msa_gap": msa_gap,
        "gap_share_of_iteration": t_gap / (t_aon + t_gap),
        "gap_vs_aon": t_gap / t_aon,
        "relative_gap": gap,
    }


def render(rows: list[dict]) -> str:
    header = (
        "| instance | links | origins | AON (s) | gap (s) | loop (s) | "
        "gap / AON | gap share of iteration | msa (ms/pass) |"
    )
    lines = [header, "| --- " * 9 + "|"]
    for row in rows:
        lines.append(
            f"| {row['instance']} | {row['n_links']} | {row['n_origins']} | "
            f"{row['aon_s']:.4f} | {row['gap_s']:.4f} | {row['per_origin_loop_s']:.4f} | "
            f"{row['gap_vs_aon']:.2f}x | {row['gap_share_of_iteration']:.0%} | "
            f"{row['msa_s_per_pass'] * 1e3:.2f} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
