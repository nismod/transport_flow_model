"""Load maritime source data and map buffered chokepoints to network edges."""

import os

import geopandas as gpd
import pandas as pd
from pyproj import Geod


def chokepoint_buffer_distance(name):
    """Buffer radius in EPSG:3857 map units."""
    if name == "Luzon Strait":
        return 200_000

    if name in {"Makassar Strait", "Ombai Strait"}:
        return 110_000

    if name in {
        "Taiwan Strait", "Bohai Strait", "Bering Strait",
        "Korea Strait", "Mindoro Strait", "Torres Strait",
    }:
        return 170_000

    return 60_000


def import_data():
    """Read GeoJSON ports/chokepoints and maritime network GeoPackages.

    Paths are relative to the maritime working directory. GeoJSON geometries
    are retained as supplied; no polygon reconstruction or transpose is needed.
    """


    ### Import maritime network edges and nodes
    path_file = './raw_data/network/edges_maritime_corrected.gpkg'

    edges_maritime_gdf = gpd.read_file(path_file)

    # Define the geodetic calculator for WGS84 ellipsoid
    geod = Geod(ellps="WGS84")

    # Compute geodesic distance (in km) for each geometry
    edges_maritime_gdf["distance_wgs84"] = edges_maritime_gdf["geometry"].apply(
        lambda geom: geod.geometry_length(geom) / 1000  # Convert meters to km
    )

    edges_maritime_gdf['dist_difference'] = edges_maritime_gdf['distance'] -  edges_maritime_gdf['distance_wgs84']


    # GeoPandas reads the feature properties, geometry, and CRS directly.
    chokepoints_df = gpd.read_file(
        './raw_data/failure_set/Chokepoints.geojson'
    ).to_crs(epsg=4326).reset_index(drop=True)
    chokepoints_df['portname'] = chokepoints_df['portname'].replace({
        'Suez': 'Suez Canal',
        'Panama': 'Panama Canal',
        'Turkish_Straits': 'Bosporus Strait',
        'Turkish Straits': 'Bosporus Strait',
        'Bab el-Mandeb': 'Bab el-Mandeb Strait',
        'Gibraltar': 'Gibraltar Strait',
        'Oresund': 'Oresund Strait',
    })


    return edges_maritime_gdf, chokepoints_df



def chokepoint_edge(chokepoints_df, edges_maritime_gdf, cp_index=None,
                    file_name="Output_Data/chokepoint_edges"):
    """Identify maritime edges intersecting each selected chokepoint buffer.

    Saves the mapping as both CSV and JSON and returns it as a DataFrame.
    ``file_name`` is the output path without a file extension.
    """
    # Reproject to a projected CRS that supports global coverage (We use 3857 only here to compute rough sizes, for distances we operate in WGS84)
    chokepoints_df = chokepoints_df.to_crs(epsg=3857)
    edges_maritime_gdf = edges_maritime_gdf.to_crs(epsg=3857)


    # Ensure cp_index is always a list. By default, process all chokepoints.
    if cp_index is None:
        cp_indices = list(range(len(chokepoints_df)))
    elif isinstance(cp_index, int):
        cp_indices = [cp_index]
    else:
        cp_indices = list(cp_index)

    selected = chokepoints_df.iloc[cp_indices]

    # Build one record per chokepoint/edge intersection. The source data has no
    # dedicated edge-ID column, so the directed from/to node pair is its ID.
    edge_records = []
    for idx, chokepoint in selected.iterrows():
        # Preserve the previous Taiwan/Bohai exceptions without relying on
        # their row positions in the source file.
        buffer_distance = chokepoint_buffer_distance(chokepoint['portname'])
        chokepoint_buffer = chokepoint.geometry.buffer(buffer_distance)

        intersects = edges_maritime_gdf.geometry.intersects(chokepoint_buffer)
        for edge in edges_maritime_gdf.loc[intersects].itertuples():
            edge_records.append({
                "chokepoint_name": chokepoint["portname"],
                "edge_id": f"{edge.from_id}_{edge.to_id}",
                "from_id": edge.from_id,
                "to_id": edge.to_id,
            })

    chokepoint_edges_df = (
        pd.DataFrame(
            edge_records,
            columns=["chokepoint_name", "edge_id", "from_id", "to_id"],
        )
        .drop_duplicates()
        .sort_values(["chokepoint_name", "edge_id"])
        .reset_index(drop=True)
    )

    output_directory = os.path.dirname(file_name)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    chokepoint_edges_df.to_csv(f"{file_name}.csv", index=False)
    chokepoint_edges_df.to_json(f"{file_name}.json", orient="records", indent=2)

    print(
        f"Saved {len(chokepoint_edges_df)} chokepoint-edge mappings to "
        f"{file_name}.csv and {file_name}.json"
    )
    return chokepoint_edges_df






