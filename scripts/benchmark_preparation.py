#!/usr/bin/env python
"""Prepare a larger OSM road-network dataset for benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import osmnx as ox
import pandas as pd

from transport_flow_model.model import Network, OD
from transport_flow_model.radiation import RadiationModel


DEFAULT_PLACE = "West Yorkshire, England, United Kingdom"
DEFAULT_OUTPUT_DIR = Path("benchmark_data") / "west_yorkshire"
NETWORK_GPKG_FILENAME = "osmnx_road_network.gpkg"
LANDUSE_GPKG_FILENAME = "osmnx_landuse_zones.gpkg"
NETWORK_CSV_PATH = Path("processed_data") / "network" / "network.csv"
OD_CSV_PATH = Path("processed_data") / "od" / "od.csv"
ZONES_CSV_PATH = Path("processed_data") / "zones" / "landuse_zones.csv"
ZONE_MAPPING_CSV_PATH = Path("processed_data") / "zones" / "zone_node_mapping.csv"
RADIATION_PROBABILITIES_CSV_PATH = (
    Path("processed_data") / "od" / "radiation_probabilities.csv"
)
DAMAGE_CSV_PATH = Path("processed_data") / "damages" / "failure_set.csv"
DEFAULT_MAX_ZONES_PER_LANDUSE = 10
DEFAULT_DISTANCE_THRESHOLD_M = 50_000.0
DEFAULT_FLOW_CAPACITY = 1.0e12
DEFAULT_SPEED_M_PER_HOUR = 50_000.0
LANDUSE_TAGS = ("residential", "commercial")


class BenchmarkPreparationError(RuntimeError):
    """Raised when benchmark data cannot be prepared safely."""


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    network_gpkg_path = output_dir / NETWORK_GPKG_FILENAME
    landuse_gpkg_path = output_dir / LANDUSE_GPKG_FILENAME
    network_csv_path = output_dir / NETWORK_CSV_PATH
    od_csv_path = output_dir / OD_CSV_PATH
    zones_csv_path = output_dir / ZONES_CSV_PATH
    zone_mapping_csv_path = output_dir / ZONE_MAPPING_CSV_PATH
    radiation_probabilities_csv_path = output_dir / RADIATION_PROBABILITIES_CSV_PATH
    damage_csv_path = output_dir / DAMAGE_CSV_PATH

    ensure_outputs_available(
        [
            network_gpkg_path,
            landuse_gpkg_path,
            network_csv_path,
            od_csv_path,
            zones_csv_path,
            zone_mapping_csv_path,
            radiation_probabilities_csv_path,
            damage_csv_path,
        ],
        overwrite=args.overwrite,
    )

    print(
        f"Fetching OSM road network for {args.place!r} "
        f"with network_type={args.network_type!r}",
        flush=True,
    )
    graph = ox.graph_from_place(
        args.place,
        network_type=args.network_type,
        simplify=True,
    )
    nodes, edges = ox.graph_to_gdfs(graph, nodes=True, edges=True)

    if edges.empty:
        raise BenchmarkPreparationError("OSMnx returned no edges for the requested area")
    if "length" not in edges.columns:
        raise BenchmarkPreparationError("OSMnx edge data is missing the length column")

    print(
        f"Fetching residential and commercial land-use polygons for {args.place!r}",
        flush=True,
    )
    landuse = fetch_landuse_zones(args.place)
    zones = build_landuse_zones(landuse)
    selected_zones = select_radiation_zones(
        zones,
        max_zones_per_landuse=args.max_zones_per_landuse,
    )
    print(
        f"Mapping {len(selected_zones)} selected land-use zones to network nodes",
        flush=True,
    )
    zone_mapping = map_zones_to_network_nodes(graph, selected_zones)

    output_dir.mkdir(parents=True, exist_ok=True)
    network_csv_path.parent.mkdir(parents=True, exist_ok=True)
    od_csv_path.parent.mkdir(parents=True, exist_ok=True)
    zones_csv_path.parent.mkdir(parents=True, exist_ok=True)
    zone_mapping_csv_path.parent.mkdir(parents=True, exist_ok=True)
    radiation_probabilities_csv_path.parent.mkdir(parents=True, exist_ok=True)
    damage_csv_path.parent.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        remove_existing_outputs(
            [
                network_gpkg_path,
                landuse_gpkg_path,
                network_csv_path,
                od_csv_path,
                zones_csv_path,
                zone_mapping_csv_path,
                radiation_probabilities_csv_path,
                damage_csv_path,
            ]
        )

    ox.save_graph_geopackage(graph, filepath=network_gpkg_path, directed=True)
    network_csv = build_network_csv(edges)
    network_csv.to_csv(network_csv_path, index=False)
    print("Writing land-use zones and zone mappings", flush=True)
    save_landuse_geopackage(
        landuse_gpkg_path,
        all_zones=zones,
        selected_zones=selected_zones,
    )
    write_zone_tables(
        selected_zones=selected_zones,
        zone_mapping=zone_mapping,
        zones_csv_path=zones_csv_path,
        zone_mapping_csv_path=zone_mapping_csv_path,
    )

    print("Generating radiation OD matrix", flush=True)
    od_matrix, probabilities = generate_radiation_od(
        network_csv_path=network_csv_path,
        zones=selected_zones,
        zone_mapping=zone_mapping,
        distance_threshold=args.distance_threshold_m,
    )
    probabilities.to_csv(radiation_probabilities_csv_path, index=False)
    od_matrix.to_csv(od_csv_path, index=False)
    failure_set = build_failure_set(
        network_csv_path=network_csv_path,
        od_matrix=od_matrix,
    )
    failure_set.to_csv(damage_csv_path, index=False)

    print("Prepared benchmark network dataset", flush=True)
    print(f"  place: {args.place}", flush=True)
    print(f"  network_type: {args.network_type}", flush=True)
    print(f"  nodes: {len(nodes)}", flush=True)
    print(f"  edges: {len(edges)}", flush=True)
    print(f"  landuse_zones: {len(zones)}", flush=True)
    print(f"  radiation_zones: {len(selected_zones)}", flush=True)
    print(f"  od_pairs: {len(od_matrix)}", flush=True)
    print(f"  failure_edges: {len(failure_set)}", flush=True)
    print(f"  network_geopackage: {network_gpkg_path}", flush=True)
    print(f"  landuse_geopackage: {landuse_gpkg_path}", flush=True)
    print(f"  network_csv: {network_csv_path}", flush=True)
    print(f"  od_csv: {od_csv_path}", flush=True)
    print(f"  damage_csv: {damage_csv_path}", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch and export an OSM road network for benchmark configs."
    )
    parser.add_argument(
        "--place",
        default=DEFAULT_PLACE,
        help=f"OSM place query to fetch. Default: {DEFAULT_PLACE!r}.",
    )
    parser.add_argument(
        "--network-type",
        default="drive",
        help="OSMnx network type to fetch. Default: 'drive'.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing GeoPackage and network CSV outputs.",
    )
    parser.add_argument(
        "--max-zones-per-landuse",
        type=int,
        default=DEFAULT_MAX_ZONES_PER_LANDUSE,
        help=(
            "Maximum largest polygons per land-use class to use in radiation OD "
            "generation. Use 0 to include all downloaded zones. "
            f"Default: {DEFAULT_MAX_ZONES_PER_LANDUSE}."
        ),
    )
    parser.add_argument(
        "--distance-threshold-m",
        type=float,
        default=DEFAULT_DISTANCE_THRESHOLD_M,
        help=(
            "Maximum network distance in metres for the radiation model. "
            f"Default: {DEFAULT_DISTANCE_THRESHOLD_M:g}."
        ),
    )
    args = parser.parse_args()
    if args.max_zones_per_landuse < 0:
        parser.error("--max-zones-per-landuse must be 0 or greater")
    if args.distance_threshold_m <= 0:
        parser.error("--distance-threshold-m must be greater than 0")
    return args


def ensure_outputs_available(paths: list[Path], *, overwrite: bool) -> None:
    if overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        existing_paths = "\n".join(f"  - {path}" for path in existing)
        raise BenchmarkPreparationError(
            "Refusing to overwrite existing benchmark outputs without --overwrite:\n"
            f"{existing_paths}"
        )


def remove_existing_outputs(paths: list[Path]) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def build_network_csv(edges: pd.DataFrame) -> pd.DataFrame:
    edge_rows = edges.reset_index()
    required_index_columns = {"u", "v", "key"}
    missing_index_columns = required_index_columns - set(edge_rows.columns)
    if missing_index_columns:
        raise BenchmarkPreparationError(
            "OSMnx edge data is missing expected edge index columns: "
            f"{sorted(missing_index_columns)}"
        )

    network = pd.DataFrame(
        {
            "from_id": edge_rows["u"],
            "to_id": edge_rows["v"],
            "id": [
                f"{u}_{v}_{key}"
                for u, v, key in zip(
                    edge_rows["u"],
                    edge_rows["v"],
                    edge_rows["key"],
                    strict=False,
                )
            ],
            "length_m": edge_rows["length"],
            "flow_capacity": DEFAULT_FLOW_CAPACITY,
            "gcost_usd_per_ton": edge_rows["length"],
            "time_hr": edge_rows["length"] / DEFAULT_SPEED_M_PER_HOUR,
        }
    )
    return network.sort_values(
        ["from_id", "to_id", "id"],
        kind="mergesort",
    ).reset_index(drop=True)


def fetch_landuse_zones(place: str) -> gpd.GeoDataFrame:
    landuse = ox.features_from_place(
        place,
        tags={"landuse": list(LANDUSE_TAGS)},
    )
    if landuse.empty:
        raise BenchmarkPreparationError(
            "OSMnx returned no residential or commercial land-use features"
        )
    landuse = landuse[landuse["landuse"].isin(LANDUSE_TAGS)].copy()
    landuse = landuse[
        landuse.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    ].copy()
    if landuse.empty:
        raise BenchmarkPreparationError(
            "OSMnx returned no polygonal residential or commercial land-use areas"
        )
    return landuse


def build_landuse_zones(landuse: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    zones = landuse.reset_index().copy()
    element_type_column, osmid_column = osm_identifier_columns(zones)
    if element_type_column is None or osmid_column is None:
        raise BenchmarkPreparationError(
            "OSMnx land-use data is missing expected OSM identifier columns"
        )

    projected = ox.projection.project_gdf(zones)
    zones["area_m2"] = projected.geometry.area.to_numpy()
    zones["zone_id"] = [
        f"{landuse_type}_{element_type}_{osmid}"
        for landuse_type, element_type, osmid in zip(
            zones["landuse"],
            zones[element_type_column],
            zones[osmid_column],
            strict=False,
        )
    ]
    zones = zones[zones["area_m2"] > 0].copy()
    if zones.empty:
        raise BenchmarkPreparationError("All downloaded land-use zones have zero area")

    return zones.sort_values(["landuse", "zone_id"], kind="mergesort").reset_index(
        drop=True
    )


def osm_identifier_columns(
    data: pd.DataFrame,
) -> tuple[str | None, str | None]:
    if {"element_type", "osmid"}.issubset(data.columns):
        return "element_type", "osmid"
    if {"element", "id"}.issubset(data.columns):
        return "element", "id"
    return None, None


def select_radiation_zones(
    zones: gpd.GeoDataFrame,
    *,
    max_zones_per_landuse: int,
) -> gpd.GeoDataFrame:
    if max_zones_per_landuse == 0:
        selected = zones.copy()
    else:
        selected = (
            zones.sort_values(["landuse", "area_m2"], ascending=[True, False])
            .groupby("landuse", group_keys=False)
            .head(max_zones_per_landuse)
            .copy()
        )

    missing_landuse = set(LANDUSE_TAGS) - set(selected["landuse"])
    if missing_landuse:
        raise BenchmarkPreparationError(
            "Cannot generate residential-to-commercial OD without zones for: "
            f"{sorted(missing_landuse)}"
        )

    return selected.sort_values(["landuse", "zone_id"], kind="mergesort").reset_index(
        drop=True
    )


def map_zones_to_network_nodes(graph, zones: gpd.GeoDataFrame) -> pd.DataFrame:
    projected_graph = ox.projection.project_graph(graph)
    projected = zones.to_crs(projected_graph.graph["crs"])
    representative_points = gpd.GeoSeries(
        projected.geometry.representative_point(),
        crs=projected.crs,
    )
    zone_points = gpd.GeoDataFrame(
        {"zone_id": zones["zone_id"].to_numpy()},
        geometry=representative_points,
        crs=projected.crs,
    )
    network_nodes = ox.graph_to_gdfs(
        projected_graph,
        nodes=True,
        edges=False,
    ).loc[:, ["geometry"]]
    network_nodes["node_id"] = network_nodes.index
    joined = gpd.sjoin_nearest(
        zone_points,
        network_nodes,
        how="left",
        distance_col="nearest_node_distance_m",
    )
    joined = (
        joined.sort_values(["zone_id", "nearest_node_distance_m"], kind="mergesort")
        .drop_duplicates("zone_id", keep="first")
        .sort_values("zone_id", kind="mergesort")
    )
    if joined["node_id"].isna().any():
        raise BenchmarkPreparationError("Could not map all land-use zones to network nodes")
    return pd.DataFrame(
        {
            "zone_id": joined["zone_id"].to_numpy(),
            "node_id": joined["node_id"].to_numpy(),
        }
    )


def save_landuse_geopackage(
    path: Path,
    *,
    all_zones: gpd.GeoDataFrame,
    selected_zones: gpd.GeoDataFrame,
) -> None:
    columns = ["zone_id", "landuse", "area_m2", "geometry"]
    all_zones.loc[:, columns].to_file(path, layer="all_landuse_zones", driver="GPKG")
    selected_zones.loc[:, columns].to_file(
        path,
        layer="radiation_zones",
        driver="GPKG",
    )


def write_zone_tables(
    *,
    selected_zones: gpd.GeoDataFrame,
    zone_mapping: pd.DataFrame,
    zones_csv_path: Path,
    zone_mapping_csv_path: Path,
) -> None:
    zone_columns = ["zone_id", "landuse", "area_m2"]
    selected_zones.loc[:, zone_columns].to_csv(zones_csv_path, index=False)
    zone_mapping.to_csv(zone_mapping_csv_path, index=False)


def generate_radiation_od(
    *,
    network_csv_path: Path,
    zones: pd.DataFrame,
    zone_mapping: pd.DataFrame,
    distance_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    network = Network.from_csv(
        network_csv_path,
        {
            "from_id": "edge_from",
            "to_id": "edge_to",
            "id": "edge_id",
            "length_m": "cost",
        },
    )
    radiation = RadiationModel(network)
    probabilities = radiation.generate(
        zones=zones.loc[:, ["zone_id", "area_m2"]],
        zone_id_column="zone_id",
        zone_to_node_mapping=zone_mapping,
        relevance_column="area_m2",
        distance_threshold=distance_threshold,
    )
    if probabilities.empty:
        raise BenchmarkPreparationError(
            "Radiation model generated no OD probabilities; increase "
            "--distance-threshold-m or use more connected zones"
        )

    zone_types = zones.set_index("zone_id")["landuse"]
    origin_area = zones.set_index("zone_id")["area_m2"]
    zone_to_node = zone_mapping.set_index("zone_id")["node_id"]
    probabilities = probabilities.copy()
    probabilities["origin_landuse"] = probabilities["origin"].map(zone_types)
    probabilities["destination_landuse"] = probabilities["destination"].map(zone_types)
    filtered = probabilities[
        (probabilities["origin_landuse"] == "residential")
        & (probabilities["destination_landuse"] == "commercial")
    ].copy()
    if filtered.empty:
        raise BenchmarkPreparationError(
            "Radiation model generated no residential-to-commercial OD pairs"
        )

    filtered["origin_area_m2"] = filtered["origin"].map(origin_area)
    filtered["destination_area_m2"] = filtered["destination"].map(origin_area)
    filtered["origin_node_id"] = filtered["origin"].map(zone_to_node)
    filtered["destination_node_id"] = filtered["destination"].map(zone_to_node)
    filtered["tons"] = filtered["origin_area_m2"] * filtered["probability"]
    od_matrix = (
        filtered.rename(
            columns={
                "origin": "origin_zone_id",
                "destination": "destination_zone_id",
                "origin_node_id": "origin_id",
                "destination_node_id": "destination_id",
            }
        )
        .loc[
            :,
            [
                "origin_id",
                "destination_id",
                "origin_zone_id",
                "destination_zone_id",
                "origin_area_m2",
                "destination_area_m2",
                "probability",
                "tons",
            ],
        ]
        .sort_values(["origin_id", "destination_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    return od_matrix, probabilities


def build_failure_set(
    *,
    network_csv_path: Path,
    od_matrix: pd.DataFrame,
) -> pd.DataFrame:
    network = Network.from_csv(
        network_csv_path,
        {
            "from_id": "edge_from",
            "to_id": "edge_to",
            "id": "edge_id",
            "flow_capacity": "capacity",
            "gcost_usd_per_ton": "cost",
        },
    )
    for row in od_matrix.itertuples(index=False):
        od = OD(
            pd.DataFrame(
                {
                    "origin_id": [row.origin_id],
                    "destination_id": [row.destination_id],
                    "flow": [row.tons],
                }
            )
        )
        allocation = network.allocate(od, capacity_constrained=False, directed=True)
        od_flows = allocation.od_flows.to_dataframe()
        if not od_flows.empty and od_flows.loc[0, "edge_path"]:
            return pd.DataFrame({"edge_id": [od_flows.loc[0, "edge_path"][0]]})

    raise BenchmarkPreparationError(
        "Could not find a directed network path for any generated OD pair"
    )


if __name__ == "__main__":
    raise SystemExit(main())
