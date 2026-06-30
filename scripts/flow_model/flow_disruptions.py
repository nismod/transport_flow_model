#!/usr/bin/env python
# coding: utf-8
"""Disruption model for rerouting and flow isolation analysis

Run this after initial flow_allocation.py
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from transport_flow_model.model import Network, ODFlows
from transport_flow_model.config import load_config


def main(config):
    processed_data_path = Path(config["paths"]["data"])
    results_data_path = Path(config["paths"]["results"])

    # Create a folder for the flow disruption outputs. Example name given here
    disruption_results_path = results_data_path / "flow_disruptions"
    disruption_results_path.mkdir(parents=True, exist_ok=True)

    #
    # OD Inputs
    #
    flow_folder = results_data_path / "flow_od_paths"

    # Specify flow OD data path
    od_flows_file = flow_folder / "od_flows.csv"

    # Specify path of network dataframe with the pre-disruption flows
    edge_flows_file = flow_folder / "network_edge_total_flows.csv"

    # Specify the names of the important columns in the pre-disruption OD file
    flow_column = "flow"  # Total tons column
    edge_path_column = "edge_path"
    cost_column = "cost"  # The cost criteria used in the OD assignment
    distance_column = (
        "length_m"  # Include this if there is interest in estimating new distance
    )
    time_column = "time_hr"  # Include this if there is interest in estimating new time
    network_attribute_columns = [distance_column, time_column]
    if network_attribute_columns is not None:
        rerouting_loss_columns = [f"rerouting_{cost_column}"] + [
            f"rerouting_{c}" for c in network_attribute_columns
        ]
    else:
        rerouting_loss_columns = [f"rerouting_{cost_column}"]

    # Load flow and network data
    flow_df = pd.read_csv(od_flows_file)
    network_df = pd.read_csv(edge_flows_file).rename(columns={"id": "edge_id"})

    # Parse edge_path from JSON string representation
    flow_df[edge_path_column] = flow_df[edge_path_column].map(
        lambda s: json.loads(s.replace("'", '"'))
    )

    # Create Network and ODFlows objects
    network = Network(network_df)
    od_flows = ODFlows(flow_df)

    #
    # Damage scenario inputs
    #

    # Get the set of damaged edges This comes from the exposure/vulnerability
    # analysis where we assemble the unique set of failed edges Get the list of
    # edges of the initiating sector to fail
    failure_id_column = "edge_id"
    damages_results_path = processed_data_path / "damages"
    failure_edges = pd.read_csv(damages_results_path / "failure_set.csv")

    #
    # Process disruptions
    #

    # Start the failure simulations by looping over each failure scenario
    ef_list = []
    for row in failure_edges.itertuples():
        fail_edges = getattr(row, failure_id_column)
        # Convert to list if only single edge
        if not isinstance(fail_edges, list):
            fail_edges = [fail_edges]

        # Check if there is pre-disruption flow on failed edges
        if network_df[network_df["edge_id"].isin(fail_edges)][flow_column].sum() > 0:
            # Run disruption analysis using Network.disrupt()
            disruption_result = network.disrupt(
                od_flows,
                failed_edges=fail_edges,
                capacity_constrained=True,
                directed=True,
            )

            rerouted_flows = disruption_result.rerouted_flows.to_dataframe()
            isolated_flows = disruption_result.isolated_od.to_dataframe()

            # Calculate rerouting losses if flows were rerouted
            if len(rerouted_flows) > 0:
                # Store old costs for comparison
                rerouted_flows[f"rerouting_{cost_column}"] = (
                    rerouted_flows[cost_column]
                ) * rerouted_flows[flow_column]
                # Calculate attribute losses if available
                if network_attribute_columns is not None:
                    for attr_l in network_attribute_columns:
                        if attr_l in rerouted_flows.columns:
                            rerouted_flows[f"rerouting_{attr_l}"] = rerouted_flows[
                                attr_l
                            ]
            else:
                for loss_column in rerouting_loss_columns:
                    rerouted_flows[loss_column] = pd.Series(dtype=float)

            # Initialize loss columns for isolated flows
            for loss_column in rerouting_loss_columns:
                isolated_flows[loss_column] = 0.0

            # Tag with failure scenario
            if len(fail_edges) == 1:
                rerouted_flows[failure_id_column] = fail_edges[0]
                isolated_flows[failure_id_column] = fail_edges[0]
            else:
                rerouted_flows[failure_id_column] = str(fail_edges)
                isolated_flows[failure_id_column] = str(fail_edges)

            ef_list.append(rerouted_flows)
            ef_list.append(isolated_flows)

    ef_list = pd.concat(ef_list, axis=0, ignore_index=True)
    sum_columns = [flow_column] + rerouting_loss_columns
    ef_list = (
        ef_list.groupby(failure_id_column)
        .agg(dict([(c, "sum") for c in sum_columns]))
        .reset_index()
    )

    ef_list["total_flow_loss"] = (
        ef_list[f"rerouting_{cost_column}"] + ef_list[flow_column]
    )
    ef_list.to_csv(
        disruption_results_path / "flow_disruption_losses.csv",
        index=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="flow_disruption",
        description="Allocate origin-destination flows to a network with some node/edge disruptions",
    )
    parser.add_argument("config", help="Path to config.json")
    args = parser.parse_args()
    CONFIG = load_config(args.config)
    main(CONFIG)
