#!/usr/bin/env python
# coding: utf-8
"""Disruption model for rerouting and flow isolation analysis
"""
import sys
import os

import pandas as pd
import geopandas as gpd
import duckdb
pd.options.mode.chained_assignment = None
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
import numpy as np
from transport_flow_model.flow_utils import *

def main(config):
    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['processed_data']
    results_data_path = config['paths']['results']


    # Create a folder for the flow disruption outputs. Example name given here
    results_folder = os.path.join(results_data_path,"flow_disruptions")
    os.makedirs(results_folder,exist_ok=True)

    ######################
    #     OD Inputs      #
    ######################
    # Specify flow OD data path
    flow_od_folder = os.path.join(output_data_path,"/path_to_OD_data/")
    # Specify path of network dataframe with the pre-disruption flows
    network_flow_folder = os.path.join(processed_data_path,"/path_to_network_data_with_flows/")

    # Specify the names of the important columns in the pre-disruption OD file 
    flow_column = "total_tons" # Total tons column
    flow_value = "total_value_euro" # Total flow value column in some monetory unit 
    flow_value_columns ["list of commodity specific columns with Euro values"] + [flow_value]
    network_id_column = "id" # The ID column of the network edge file
    edge_path_column = "edge_path"
    cost_column = "gcost_usd_tons" # The cost criteria used in the OD assignment
    distance_column = "length_m" # Include this if there is interest in estimating new distance 
    time_column = "time_hr" # Include this if there is interest in estimating new time
    network_attribute_columns = [distance_column,time_column]
    if network_attribute_columns is not None:
        rerouting_columns = [f"rerouting_{cost_column}"] + ["rerouting_{c}" for c in network_attribute_columns]
    else:
        rerouting_columns = [f"rerouting_{cost_column}"]
    # Get all the relevant columns in the OD file
    od_columns = ["origin_id","destination_id",edge_path_column,cost_column,distance_column,time_column]

    od_flows_file = os.path.join(flow_od_folder)
    edge_flows_file = os.path.join(network_flow_folder)

    od_file_size = "large"
    if od_file_size == "large":
        # OPTION 1 - Use DUCKDB to read OD file if it is very large
        flow_df = duckdb.query(f'SELECT * FROM "{od_flows_file}";').df()
    else:
        # OR read OD file as geoparquet directly
        flow_df = pd.read_parquet(od_flows_file)
    
    network_df = pd.read_parquet(edge_flows_file)

    ###################################
    #     Damage scenario inputs      #
    ###################################
    # Get the set of damaged edges
    # This comes from the exposure/vulnerability analysis where we assemble the unique set of failed edges
    # Get the list of edges of the initiating sector to fail
    failure_id_column = "id"
    damages_results_path = os.path.join("/path/to/damage/set")
    failure_edges = pd.read_parquet(os.path.join(damages_results_path,"failure_set.parquet"))
    

    ##########################################################
    #     Process the OD flows to identify edges on paths    #
    ##########################################################
    # Get the paths indexes of every edge in the OD dataframe
    # This step could also be a pre computation script and its output stored in advance
    # It is a slow step if the OD is very big
    edge_path_idx = get_flow_paths_indexes_and_edges_dataframe(flow_df,edge_path_column,id_column=network_id_column)
    
    # Start the failure simiulations by looping over each failure scenario corresponding to an inidviual failed edge
    # This step should be parallelised
    ef_list = []
    for row in failure_edges.itertuples():
        fail_edges = getattr(row,failure_id_column)
        # Convert to list if only single edge
        if isinstance(fail_edges,list) == False:
            fail_edges = [fail_edges]

        if network_df[network_df["id"].isin(fail_edges)][flow_column].sum() > 0: 
            # Rerouting done only if the pre-disruption flow on edge > 0
            rerouted_flows, isolated_flows = flow_disruption_estimation(network_df,fail_edges,
                                                flow_df,edge_path_idx,"id",flow_column,
                                                cost_column,attribute_list=network_attribute_columns)

            rerouting_loss = []
            isolation_loss = []
            if len(rerouted_flows) > 0:
                for rf in rerouted_flows:
                    rf[f"rerouting_{cost_column}"] = (rf[cost_column]  - rf[f"old_{cost_column}"])*rf[flow_column]
                    if attribute_list is not None:
                        for attr_l in attribute_list:
                            rf[f"rerouting_{attr_l}"] = rf[attr_l]  - rf[f"old_{attr_l}"]

                    rerouting_loss.append(rf[[rerouting_loss_columns]].sum(axis=0))
                rerouting_loss = pd.concat(rerouting_loss,axis=0,ignore_index=True)
            else:
                rerouting_loss = pd.DataFrame([0]*len(rerouting_loss_columns),columns=rerouting_loss_columns)

            if len(isolated_flows) > 0:
                for isl in isolated_flows:
                    isolation_loss.append(rf[[flow_value_columns]].sum(axis=0))
            
                isolation_loss = pd.concat(isolation_loss,axis=0,ignore_index=True)
            else:
                isolation_loss = pd.DataFrame([0]*len(flow_value_columns),columns=flow_value_columns)

            
            del rerouted_flows, isolated_flows

            if len(fail_edges) == 1:
                rerouting_loss[failure_id_column] = fail_edges[0]
                isolation_loss[failure_id_column] = fail_edges[0]
            else:
                rerouting_loss[failure_id_column] = str(fail_edges)
                isolation_loss[failure_id_column] = str(fail_edges)

            ef_list.append(rerouting_loss)
            ef_list.append(isolation_loss)

        print (f"* Done with failure scenario {row.Index} out of {len(fail_edges)}")

        ef_list = pd.concat(ef_list,axis=0,ignore_index=True)
        sum_columns = [c for c in ef_list.columns.values.tolist() if c != failure_id_column]
        ef_list = ef_list.groupby(failure_id_column).agg(dict([(c,"sum") for c in sum_columns])).reset_index()
        ef_list["total_flow_loss"] = ef_list[f"rerouting_{cost_column}"] + ef_list[flow_value]
        ef_list.to_csv(os.path.join(results_folder,
                        "flow_disruption_losses.csv"),index=False)

if __name__ == "__main__":
    CONFIG = load_config()
    main(CONFIG)