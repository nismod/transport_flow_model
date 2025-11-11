"""Functions for utlis for transport flow modelling
"""
import sys
import os
import json
import snkit
import numpy as np
import pandas as pd
import igraph as ig
import geopandas as gpd
from collections import defaultdict
from itertools import chain
from tqdm import tqdm
tqdm.pandas()

def load_config():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    config_path = os.path.join(script_dir,'..','..', 'config.json')

    with open(config_path, 'r') as config_fh:
        config = json.load(config_fh)
    return config

def get_flow_paths_indexes_and_edges_dataframe(flow_dataframe,path_criteria,id_column="id"):
    edge_path_index = []
    for v in flow_dataframe.itertuples():
        path = getattr(v,path_criteria)
        edge_path_index += list(zip(path,[v.Index]*len(path)))
    del flow_dataframe
    return pd.DataFrame(edge_path_index,columns=[id_column,"path_index"])

def find_minimal_flows_along_overcapacity_paths(over_capacity_ods,network_dataframe,
                                        over_capacity_edges,edge_id_paths,edge_id_column,flow_column):
    """
    Parameters
    ---------
    over_capacity_ods: pandas.Dataframe
        Pandas DataFrame of OD flow paths with over capacity flows
    network_dataframe: pandas.Dataframe
        Pandas DataFrame of network edges with existing flows and capacities
    over_capacity_edges: list
        List of over capacity edges
    edge_id_paths: Dictionary
        List of OD dataframe indexes for paths which are overcapacity
    edge_id_column: str
        Name of the edge ID column
    flow_column: str
        Name of flow column
    
    Returns
    -------
    over_capacity_ods : pandas.DataFrame
        - With updated values of flows for OD pairs with over capacity
    """
    over_capacity_edges_df = pd.DataFrame([
                                (
                                    path_key,path_idx
                                ) for path_key,path_idx in edge_id_paths.items() if path_key in over_capacity_edges
                            ],columns = [edge_id_column,"path_indexes"]
                                )
    over_capacity_edges_df = pd.merge(over_capacity_edges_df,
                                network_dataframe[[edge_id_column,"residual_capacity","added_flow"]],
                                how="left",
                                on=[edge_id_column])
    over_capacity_edges_df["edge_path_flow"] = over_capacity_edges_df.progress_apply(
                                        lambda x:over_capacity_ods[
                                            over_capacity_ods.path_indexes.isin(x.path_indexes)
                                            ][flow_column].values,
                                        axis=1
                                        )
    over_capacity_edges_df["edge_path_flow_cor"] = over_capacity_edges_df.progress_apply(
                                        lambda x:list(
                                            1.0*x.residual_capacity*x.edge_path_flow/x.added_flow),
                                        axis=1
                                        )
    over_capacity_edges_df["path_flow_tuples"] = over_capacity_edges_df.progress_apply(
                                        lambda x:list(zip(x.path_indexes,x.edge_path_flow_cor)),axis=1)

    min_flows = []
    for r in over_capacity_edges_df.itertuples():
        min_flows += r.path_flow_tuples

    min_flows = pd.DataFrame(min_flows,columns=["path_indexes","min_flows"])
    min_flows = min_flows.sort_values(by=["min_flows"],ascending=True)
    min_flows = min_flows.drop_duplicates(subset=["path_indexes"],keep="first")

    over_capacity_ods = pd.merge(over_capacity_ods,min_flows,how="left",on=["path_indexes"])
    del min_flows, over_capacity_edges_df
    over_capacity_ods["residual_flows"] = over_capacity_ods[flow_column] - over_capacity_ods["min_flows"]

    return over_capacity_ods

def get_path_indexes_for_edges(edge_ids_with_paths,selected_edge_list):
    """
    Parameters
    ---------
    edge_ids_with_paths: Dictionary
        Dictionary with key as Edge ID and values as indexs of OD dataframe
    selected_edge_list: list
        List of edge ID for which we want indexes

    Returns
    -------
    list: Of all indexes in the OD dataframe that contain the edges
    """
    return list(
            set(
                list(
                    chain.from_iterable([
                        path_idx for path_key,path_idx in edge_ids_with_paths.items() if path_key in selected_edge_list
                                        ]
                                        )
                    )
                )
            )

def get_flow_paths_indexes_of_edges(flow_dataframe,path_criteria):
    """
    Parameters
    ---------
    flow_dataframe: pandas.Dataframe
        Pandas DataFrame of OD flow paths and their flows
    path_criteria: str
        Name of column with the list of edges

    Returns
    -------
    edge_path_index : Dictionary
        - Key: Edge ID values
        - Value: Index of all OD-pairs that invovle the edge ID
    """
    edge_path_index = defaultdict(list)
    for v in flow_dataframe.itertuples():
        for k in getattr(v,path_criteria):
            edge_path_index[k].append(v.Index)

    del flow_dataframe
    return edge_path_index

def get_flow_on_edges(save_paths_df,edge_id_column,edge_path_column,
    flow_column):
    """
    Parameters
    ---------
    save_paths_df: pandas.Dataframe
        Pandas DataFrame of OD flow paths and their flows
    edge_id_column: str
        Name of the edge ID column
    edge_path_column: str
        Name of Edge PAth column - General edge_path
    flow_column: str
        Name of flow column
    
    Returns
    -------
    network_df : pandas.DataFrame
        - edge_id_column - Column of Edge IDs
        - flow_column - Column of total flow values along each edge
    """
    edge_flows = defaultdict(float)
    for row in save_paths_df.itertuples():
        for item in getattr(row,edge_path_column):
            edge_flows[item] += getattr(row,flow_column)

    return pd.DataFrame([(k,v) for k,v in edge_flows.items()],columns=[edge_id_column,flow_column])

def update_flow_and_overcapacity(od_dataframe,network_dataframe,flow_column,edge_id_column="id",
                                network_capacity_column="capacity",subtract=False):
    """
    Parameters
    ---------
    od_dataframe: pandas.Dataframe
        Pandas DataFrame of OD flow paths and their flows
    network_dataframe: pandas.Dataframe
        Pandas DataFrame of network edges with existing flows and capacities
    flow_column: str
        Name of flow column that we want to compare to capacity
    edge_id_column: str
        Name of the edge ID column
    network_capacity_column: str
        Name of network capacity column
    substract: Boolean
        False to add True to subtract
    
    Returns
    -------
    network_dataframe : pandas.DataFrame
        - With updates value of flows and over capacity estimations
    """
    edge_flows = get_flow_on_edges(od_dataframe,edge_id_column,"edge_path",flow_column)
    edge_flows.rename(columns={flow_column:"added_flow"},inplace=True)
    network_dataframe = pd.merge(network_dataframe,edge_flows,how="left",on=[edge_id_column]).fillna(0)
    del edge_flows
    if subtract is True:
        network_dataframe[flow_column] = network_dataframe[flow_column] - network_dataframe["added_flow"]
    else:
        network_dataframe[flow_column] += network_dataframe["added_flow"]
    network_dataframe["over_capacity"] = network_dataframe[network_capacity_column] - network_dataframe[flow_column]

    return network_dataframe

def network_od_path_estimations_multiattribute(graph,
    source, target, cost_criteria,path_id_column,attribute_list=None):
    """Estimate the paths, distances, times, and costs for given OD pair

    Parameters
    ---------
    graph
        igraph network structure
    source
        String/Float/Integer name of Origin node ID
    target
        String/Float/Integer name of Destination node ID
    cost_criteria : str
        name of generalised cost criteria to be used: gcost
    path_id_column: str
        name of ID column for getting the edge path
    attribute_list: list
        list of names of columns whose values need to be added up

    Returns
    -------
    path_list: DataFrame
        Dataframe with the path attributes

    """
    paths = graph.get_shortest_paths(source, target, weights=cost_criteria, output="epath")

    paths_list = []
    if attribute_list is None:
        for path in paths:
            path_dict = {'edge_path':[],cost_criteria:0}
            if path:
                for n in path:
                    path_dict['edge_path'].append(graph.es[n][path_id_column])
                    path_dict[cost_criteria] += graph.es[n][cost_criteria]

            paths_list.append(path_dict)
    else:
        for path in paths:
            path_dict = dict([('edge_path',[]),(cost_criteria,0)] + [(a,0) for a in attribute_list])
            if path:
                for n in path:
                    path_dict['edge_path'].append(graph.es[n][path_id_column])
                    path_dict[cost_criteria] += graph.es[n][cost_criteria]
                    for a in attribute_list:
                        path_dict[a] += graph.es[n][a]

            paths_list.append(path_dict)

    return pd.DataFrame(paths_list)

def network_od_paths_assembly_multiattributes(points_dataframe,graph,
                                cost_criteria,path_id_column,
                                origin_id_column,destination_id_column,
                                attribute_list=None,store_edge_path=True):
    """Assemble estimates of OD paths, distances, times, costs and tonnages on networks

    Parameters
    ----------
    points_dataframe : pandas.DataFrame
        OD nodes and their tonnages
    graph
        igraph network structure
    cost_criteria : str
        name of generalised cost criteria to be used: gcost
    path_id_column: str
        name of ID column for getting the edge path
    origin_id_column: str
        name of Origin node ID column
    destination_id_column: str
        name of Destination node ID column
    attribute_list: list
        list of names of columns whose values need to be added up
    store_edge_path: Boolean
        True if the edge path is to be stored. False otherwise

    Returns
    -------
    save_paths_df : pandas.DataFrame
        - origin - String node ID of Origin
        - destination - String node ID of Destination
        - edge_path - List of string of edge ID's for paths with minimum generalised cost flows
        - cost_criteria - Float values of estimated generalised cost for paths with minimum generalised cost flows
        - attribute_list - Float values of estimated values for paths with minimum generalised cost flows

    """
    save_paths_df = []
    points_dataframe = points_dataframe.set_index(origin_id_column)
    origins = list(set(points_dataframe.index.values.tolist()))
    for origin in origins:
        try:
            destinations = list(set(points_dataframe.loc[[origin], destination_id_column].values.tolist()))

            get_path_df = network_od_path_estimations_multiattribute(
                    graph, origin, destinations, cost_criteria,path_id_column,
                    attribute_list=attribute_list)
            get_path_df[origin_id_column] = origin
            get_path_df[destination_id_column] = destinations
            save_paths_df.append(get_path_df)
        except:
            print(f"* no path between {origin}-{destinations}")
    
    save_paths_df = pd.concat(save_paths_df,axis=0,ignore_index=True)
    if store_edge_path is False:
        save_paths_df.drop("edge_path",axis=1,inplace=True)

    points_dataframe = points_dataframe.reset_index()
    save_paths_df = pd.merge(points_dataframe,save_paths_df,how='left', on=[
                             origin_id_column, destination_id_column]).fillna(0)
    save_paths_df = save_paths_df[save_paths_df[origin_id_column] != 0]

    return save_paths_df

def create_igraph_from_dataframe(graph_dataframe, directed=False, simple=False):
    # This might be in Snkit or Snail or Open-GIRA
    graph = ig.Graph.TupleList(
        graph_dataframe.itertuples(index=False),
        edge_attrs=list(graph_dataframe.columns)[2:],
        directed=directed
    )
    if simple:
        graph.simplify()

    es, vs, simple = graph.es, graph.vs, graph.is_simple()
    d = "directed" if directed else "undirected"
    s = "simple" if simple else "multi"
    print(
        "Created {}, {} {}: {} edges, {} nodes.".format(
            s, d, "igraph", len(es), len(vs)))

    return graph

def od_flow_allocation_capacity_constrained(flow_ods,
                                            network_dataframe,
                                            flow_column,
                                            cost_column,
                                            path_id_column,
                                            attribute_list=None,
                                            origin_id_column="origin_id",
                                            destination_id_column="destination_id",
                                            over_capacity_threshold=1.0e-3,
                                            network_capacity_column="capacity",
                                            directed=False,
                                            simple=False,
                                            store_edge_path=True):
    network_dataframe["over_capacity"] = network_dataframe[network_capacity_column] - network_dataframe[flow_column]
    capacity_ods = []
    unassigned_paths = []
    while len(flow_ods.index) > 0:
        graph = create_igraph_from_dataframe(
                    network_dataframe[network_dataframe["over_capacity"] > over_capacity_threshold],
                    directed=directed,simple=simple)
        graph_nodes = [x['name'] for x in graph.vs]
        unassigned_paths.append(flow_ods[~((flow_ods[origin_id_column].isin(graph_nodes)) & (flow_ods[destination_id_column].isin(graph_nodes)))])
        flow_ods = flow_ods[(flow_ods[origin_id_column].isin(graph_nodes)) & (flow_ods[destination_id_column].isin(graph_nodes))]
        if len(flow_ods.index) > 0:
            flow_ods = network_od_paths_assembly_multiattributes(
                                    flow_ods,graph,
                                    cost_column,
                                    path_id_column,
                                    origin_id_column,
                                    destination_id_column,
                                    attribute_list=attribute_list,
                                    store_edge_path=store_edge_path)
            unassigned_paths.append(flow_ods[flow_ods[cost_column] == 0])
            flow_ods = flow_ods[flow_ods[cost_column] > 0]
            if len(flow_ods.index) > 0:
                network_dataframe["residual_capacity"] = network_dataframe["over_capacity"]
                network_dataframe = update_flow_and_overcapacity(flow_ods,
                                        network_dataframe,flow_column,
                                        edge_id_column=path_id_column,
                                        network_capacity_column=network_capacity_column)
                over_capacity_edges = network_dataframe[network_dataframe["over_capacity"] < -1.0*over_capacity_threshold][path_id_column].values.tolist()
                if len(over_capacity_edges) > 0:
                    edge_id_paths = get_flow_paths_indexes_of_edges(flow_ods,"edge_path")
                    edge_paths_overcapacity = get_path_indexes_for_edges(edge_id_paths,over_capacity_edges)
                    if store_edge_path is False:
                        cap_ods = flow_ods[~flow_ods.index.isin(edge_paths_overcapacity)]
                        cap_ods.drop(["edge_path"],axis=1,inplace=True)
                        capacity_ods.append(cap_ods)
                        del cap_ods
                    else:
                        capacity_ods.append(flow_ods[~flow_ods.index.isin(edge_paths_overcapacity)])

                    over_capacity_ods = flow_ods[flow_ods.index.isin(edge_paths_overcapacity)]
                    over_capacity_ods["path_indexes"] = over_capacity_ods.index.values.tolist()
                    over_capacity_ods = find_minimal_flows_along_overcapacity_paths(over_capacity_ods,
                                                                network_dataframe,
                                                                over_capacity_edges,
                                                                edge_id_paths,path_id_column,flow_column)
                    cap_ods = over_capacity_ods.copy() 
                    cap_ods.drop(["path_indexes",flow_column,"residual_flows"],axis=1,inplace=True)
                    cap_ods.rename(columns={"min_flows":flow_column},inplace=True)
                    if store_edge_path is False:
                        cap_ods.drop(["edge_path"],axis=1,inplace=True)
                    
                    capacity_ods.append(cap_ods)
                    del cap_ods

                    over_capacity_ods["residual_ratio"] = over_capacity_ods["residual_flows"]/over_capacity_ods[flow_column]
                    over_capacity_ods.drop(["path_indexes",flow_column,"min_flows"],axis=1,inplace=True)
                    over_capacity_ods.rename(columns={"residual_flows":flow_column},inplace=True)

                    network_dataframe.drop("added_flow",axis=1,inplace=True)
                    network_dataframe = update_flow_and_overcapacity(over_capacity_ods,
                                                        network_dataframe,flow_column,path_id_column,
                                                        network_capacity_column=network_capacity_column,
                                                        subtract=True)
                    network_dataframe.drop("added_flow",axis=1,inplace=True)
                    flow_ods = over_capacity_ods[over_capacity_ods["residual_ratio"] > 0.01] # This is to stop the OD assignment if say 99% of flow is assigned
                    if attribute_list is not None:
                        flow_ods.drop(["edge_path",cost_column,
                                        "residual_ratio"] + attribute_list,axis=1,inplace=True)
                    else:
                        flow_ods.drop(["edge_path",cost_column,
                                        "residual_ratio"],axis=1,inplace=True)
                    del over_capacity_ods
                else:
                    if store_edge_path is False:
                        flow_ods.drop(["edge_path"],axis=1,inplace=True)
                    capacity_ods.append(flow_ods)
                    network_dataframe.drop(["residual_capacity","added_flow"],axis=1,inplace=True)
                    flow_ods = pd.DataFrame()

    return capacity_ods, unassigned_paths, network_dataframe