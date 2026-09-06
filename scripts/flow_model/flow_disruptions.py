#!/usr/bin/env python
"""Disruption model for rerouting and flow isolation analysis.

Driver over the transport_flow_model v0 API: reload the baseline
assignment written by flow_allocation.py, evaluate link-removal scenarios
with ``disrupt``, and aggregate losses per scenario.

Run this after flow_allocation.py.
"""

import argparse
import json

import pandas as pd
import pyarrow as pa

from transport_flow_model import AssignmentResult, Network, RunConfig, disrupt

# Loss columns kept for output compatibility: rerouting_cost is the cost of
# carrying each rerouted flow on its new path; the attribute columns are
# only non-zero if the baseline paths carry those attributes.
RerouteLossColumns = ["rerouting_cost", "rerouting_length_m", "rerouting_time_hr"]


def main(config: RunConfig):
    results_path = config.paths.results / "flow_disruptions"
    results_path.mkdir(parents=True, exist_ok=True)
    flow_folder = config.paths.results / "flow_od_paths"

    # Reload the baseline assignment (paths and link flows) from CSV
    paths = pd.read_csv(flow_folder / "od_flows.csv")
    paths["edge_path"] = paths["edge_path"].map(
        lambda s: json.loads(s.replace("'", '"'))
    )
    network = Network(pd.read_csv(flow_folder / "network_edge_total_flows.csv"))
    base = AssignmentResult.from_tables(
        link_flows=network.to_table(),
        paths=pa.Table.from_pandas(paths, preserve_index=False),
    )

    results = disrupt(
        network,
        config.load_scenarios(),
        base=base,
        **config.assignment.options(network),
    )

    losses = pd.concat(
        [frame for result in results for frame in scenario_loss_frames(result)],
        axis=0,
        ignore_index=True,
    )
    sum_columns = ["flow"] + RerouteLossColumns
    losses = losses.groupby("edge_id").agg({c: "sum" for c in sum_columns})
    losses = losses.reset_index()
    losses["total_flow_loss"] = losses["rerouting_cost"] + losses["flow"]
    losses.to_csv(results_path / "flow_disruption_losses.csv", index=False)


def scenario_loss_frames(result) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-OD loss rows for one scenario: rerouted flows cost their new
    path; isolated flows are counted as lost."""
    rerouted = result.rerouted.to_pandas()
    isolated = result.isolated.to_pandas().rename(columns={"value": "flow"})
    if len(rerouted) > 0:
        rerouted["rerouting_cost"] = rerouted["cost"] * rerouted["flow"]
    else:
        for column in RerouteLossColumns:
            rerouted[column] = pd.Series(dtype=float)
    for column in RerouteLossColumns:
        isolated[column] = 0.0
    rerouted["edge_id"] = result.scenario.id
    isolated["edge_id"] = result.scenario.id
    return rerouted, isolated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="flow_disruption",
        description="Allocate origin-destination flows to a network with some node/edge disruptions",
    )
    parser.add_argument("config", help="Path to config.json")
    args = parser.parse_args()
    main(RunConfig.from_json(args.config))
