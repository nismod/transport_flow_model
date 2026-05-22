#!/usr/bin/env python
# coding: utf-8
"""This code estimates the routes between Origin-Destination pairs over a network graph under capacity constraints"""

import argparse
import os
import pandas as pd
from transport_flow_model.model import Network, OD, ODFlows
from transport_flow_model.flow_utils import (
    load_config,
    od_flow_allocation_capacity_constrained,
)


def main(config):
    processed_data_path = config["paths"]["data"]
    output_data_path = config["paths"]["results"]

    ######################
    #     Model Inputs   #
    ######################
    # Specify flow OD data path
    flow_od_folder = os.path.join(processed_data_path, "od")
    # Specify path of network dataframe
    network_data_folder = os.path.join(processed_data_path, "network")

    # Create a folder for the PD flow outputs. Exmaple name given here
    results_folder = os.path.join(output_data_path, "flow_od_paths")
    os.makedirs(results_folder, exist_ok=True)

    # Read network CSV and normalize to the internal schema.
    network = Network.from_csv(
        os.path.join(network_data_folder, "network.csv"),
        {
            # Specify the network topology and ID columns
            "from_id": "edge_from",
            "to_id": "edge_to",
            "id": "edge_id",
            # Specify the cost criteria for flow assingment, which is a column in the network dataframe
            # The cost criteria column contains values that determine the least cost path assignment
            # For example:
            #     - If the criteia is shortest distance then the cost criteria column would be length of the edges
            #     - If the criteria is shortest time then the cost criteria column would be time along edges
            "flow_capacity": "capacity",
            "gcost_usd_per_ton": "cost",
            # We can also specify other attributes we want to estimate
            # For example, we might want to get the distance and time along least cost path for comparison
            "length_m": "length_m",
            "time_hr": "time_hr",
        },
    )

    # Read OD CSV and normalize to the internal schema.
    od = OD.from_csv(
        os.path.join(flow_od_folder, "od.csv"),
        # Specify OD columns for origin, destination, and flow values
        {
            "origin_id": "origin_id",
            "destination_id": "destination_id",
            "tons": "flow",
        },
    )

    ########################
    #     Model run        #
    ########################

    # Create the network graph
    network_dataframe = network.to_dataframe(copy=False)
    od_dataframe = od.to_dataframe(copy=False)

    network_attribute_columns = ["length_m", "time_hr"]  # optional columns

    network_dataframe["flow"] = 0  # To assign an initial flow to every edge

    flow_routes, unassigned_routes, network_dataframe = (
        od_flow_allocation_capacity_constrained(
            od_dataframe,
            network_dataframe,
            attribute_list=network_attribute_columns,
        )
    )

    ########################
    #     Outputs          #
    ########################

    # Store network dataframe with final flows
    network_dataframe.to_csv(
        os.path.join(results_folder, "network_edge_total_flows.csv"), index=False
    )
    # Store unassigned OD flows
    unassigned_routes = pd.concat(unassigned_routes, axis=0, ignore_index=True)
    unassigned_routes.to_csv(
        os.path.join(results_folder, "unassigned_od_flows.csv"), index=False
    )

    od_flows = ODFlows(flow_routes)

    od_flows_dataframe = od_flows.to_dataframe(copy=False)
    # We might have more flow columns in the OD matrix have we would like to partition
    # Similar to how the total flow might be divided among different routes for the same OD-pair
    # Example we might have columns of different industry specific flows
    flow_sub_columns = list(
        set(od_dataframe.columns) - {"origin_id", "destination_id", "flow"}
    )

    # partition the total assigned flow along each path in proportion to o-d sub flows
    if flow_sub_columns:
        # rename (total) flow to "assigned_flow"
        od_flows_dataframe.rename(columns={"flow": "assigned_flow"}, inplace=True)
        # merge back on original "flow" and any other flows
        od_flows_dataframe = pd.merge(
            od_flows_dataframe,
            od_dataframe,
            how="left",
            on=["origin_id", "destination_id"],
        )
        total_sub_flows = od_flows_dataframe[flow_sub_columns]
        assigned_proportion = od_flows_dataframe.assigned_flow / od_flows_dataframe.flow
        assigned_sub_flows = total_sub_flows.multiply(assigned_proportion, axis="index")
        od_flows_dataframe[flow_sub_columns] = assigned_sub_flows
        od_flows_dataframe.drop("assigned_flow", axis=1, inplace=True)

    od_flows = ODFlows(od_flows_dataframe)

    od_flows.to_csv(os.path.join(results_folder, "od_flows.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="flow_allocation",
        description="Allocate origin-destination flows to a network",
    )
    parser.add_argument("config", help="Path to config.json")
    args = parser.parse_args()
    CONFIG = load_config(args.config)
    main(CONFIG)
