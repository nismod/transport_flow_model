#!/usr/bin/env python
# coding: utf-8
"""This code estimates the routes between Origin-Destination pairs over a network graph under capacity constraints"""
import argparse
import os
import pandas as pd
from transport_flow_model.flow_utils import *


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

    # Read the network file with the topology.
    network_dataframe = pd.read_csv(os.path.join(network_data_folder, "network.csv"))
    # Specify the network topology and ID columns
    network_topology_columns = ["from_id", "to_id"]
    network_id_column = "id"
    network_capacity_column = "flow_capacity"  # Assumption is that a capacity value exists in the network dataframe
    """
    Specify the cost criteria for flow assingment, which is a column in the network dataframe
    The cost criteria column contains values that determine the least cost path assignment
    For example:
        - If the criteia is shortest distance then the cost criteria column would be length of the edges
        - If the criteria is shortest time then the cost criteria column would be time along edges
    """
    network_cost_column = "gcost_usd_per_ton"
    """
    We can also specify other attributes we want to estimate
    For example, we might want to get the distance and time along least cost path for comparison
    """
    network_attribute_columns = ["length_m", "time_hr"]

    # Read the OD file with the OD matrix information. Assuming here that it is a parquet file
    od_dataframe = pd.read_csv(os.path.join(flow_od_folder, "od.csv"))
    # Specify the OD column names for identifying the origin and desitnation nodes and flow values
    origin_id_column = "origin_id"
    destination_id_column = "destination_id"
    flow_column = "tons"  # Or vehicles, whichever unit we have for flows and capacity

    ########################
    #     Model run        #
    ########################

    # Create the network graph
    network_dataframe[flow_column] = 0  # To assign an initial flow to every edge
    # Specify the minimum columns needed from the OD matrix. It might have more columns
    od_df = od_dataframe[[origin_id_column, destination_id_column, flow_column]]
    # Specify the minimum columns needed from the network dataframe. It might have more columns
    # Also the topology columns need to be first to make the graph
    n_df = network_dataframe[
        network_topology_columns
        + [network_id_column, network_capacity_column, flow_column, network_cost_column]
        + network_attribute_columns
    ]
    flow_routes, unassigned_routes, n_df = od_flow_allocation_capacity_constrained(
        od_df,
        n_df,
        flow_column,
        network_cost_column,
        network_id_column,
        attribute_list=network_attribute_columns,
        origin_id_column=origin_id_column,
        destination_id_column=destination_id_column,
        network_capacity_column=network_capacity_column,
    )

    ########################
    #     Outputs          #
    ########################

    # Store network dataframe with final flows
    n_df.to_csv(os.path.join(results_folder, "network_edge_total_flows.csv"))
    # Store unassinged OD flows
    if len(unassigned_routes) > 0:
        unassigned_routes = pd.concat(unassigned_routes, axis=0, ignore_index=True)
        unassigned_routes.to_csv(
            os.path.join(results_folder, "unassigned_od_flows.csv")
        )

    if len(flow_routes) > 0:
        flow_routes = pd.concat(flow_routes, axis=0, ignore_index=True)
        # We might have more flow columns in the OD matrix have we would like to partition
        # Similar to how the total flow might be divided among different routes for the same OD-pair
        # Example we might have columns of different industry specific flows
        flow_sub_columns = [
            c
            for c in od_dataframe.columns.values.tolist()
            if c not in [origin_id_column, destination_id_column, flow_column]
        ]
        if len(flow_sub_columns) > 0:
            od_dataframe.rename(columns={flow_column: "initial_flow"}, inplace=True)
            flow_routes = pd.merge(
                flow_routes,
                od_dataframe,
                how="left",
                on=[origin_id_column, destination_id_column],
            )
            flow_routes[flow_sub_columns] = flow_routes[flow_sub_columns].multiply(
                flow_routes[flow_column] / flow_routes["initial_flow"], axis="index"
            )
            flow_routes.drop("initial_flow", axis=1, inplace=True)

        flow_routes.to_csv(os.path.join(results_folder, "od_flows.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="flow_allocation",
        description="Allocate origin-destination flows to a network",
    )
    parser.add_argument("config")
    args = parser.parse_args()
    CONFIG = load_config(args.config)
    main(CONFIG)
