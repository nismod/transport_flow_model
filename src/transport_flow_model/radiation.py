"""Radiation model for OD estimation using network distance."""

from collections import defaultdict
from heapq import heappop, heappush
from math import inf

import pandas as pd


class RadiationModel:
    """Radiation model for origin-destination flow estimation.

    The radiation model is a parameter-free stochastic model for predicting
    human mobility and OD flows. This implementation uses network distance
    rather than geographic distance.

    Mathematical formula:
        P_ij = (1 / (1 - m_i/M)) * (m_i * m_j) / ((m_i + s_ij) * (m_i + m_j + s_ij))

    Where:
        - m_i, m_j = relevance (opportunities/population) at origin/destination
        - s_ij = total relevance of zones between i and j (within distance threshold)
        - M = total relevance across all zones
    """

    CAPACITY_EPSILON = 1.0e-9

    def __init__(self, network):
        """Initialize radiation model with a network.

        Parameters
        ----------
        network : Network
            Network object with edge topology and cost attributes
        """
        self.network = network

    def generate(
        self,
        zones,
        zone_id_column,
        zone_to_node_mapping,
        relevance_column,
        distance_threshold,
    ):
        """Generate OD probability matrix using radiation model.

        Parameters
        ----------
        zones : pandas.DataFrame
            DataFrame with zone data containing zone_id_column and relevance_column
        zone_id_column : str
            Column name for zone identifiers in zones DataFrame
        zone_to_node_mapping : pandas.DataFrame
            DataFrame mapping zones to network nodes with columns:
            - zone_id: zone identifier (matches zones[zone_id_column])
            - node_id: network node identifier
        relevance_column : str
            Column name in zones DataFrame for relevance values (population/employment)
        distance_threshold : float
            Maximum network distance to consider for intervening opportunities

        Returns
        -------
        pandas.DataFrame
            DataFrame with columns [origin, destination, probability]
            where origin and destination are zone_ids and probability is float in [0, 1]
        """
        # Input validation
        if zone_id_column not in zones.columns:
            raise ValueError(f"Column '{zone_id_column}' not found in zones DataFrame")
        if relevance_column not in zones.columns:
            raise ValueError(
                f"Column '{relevance_column}' not found in zones DataFrame"
            )

        if "zone_id" not in zone_to_node_mapping.columns:
            raise ValueError("zone_to_node_mapping must have 'zone_id' column")
        if "node_id" not in zone_to_node_mapping.columns:
            raise ValueError("zone_to_node_mapping must have 'node_id' column")

        # Rename zone_id_column to standard name for easier handling
        zones = zones.rename(columns={zone_id_column: "zone_id"}).copy()
        zones = zones[["zone_id", relevance_column]].rename(
            columns={relevance_column: "relevance"}
        )

        # Verify all mapped zones exist in zones DataFrame
        mapped_zones = set(zone_to_node_mapping["zone_id"].values)
        available_zones = set(zones["zone_id"].values)
        missing = mapped_zones - available_zones
        if missing:
            raise ValueError(
                f"Zones in mapping not found in zones DataFrame: {missing}"
            )

        # Verify all mapped nodes exist in network
        network_data = self.network.to_dataframe()
        network_nodes = set(network_data["edge_from"].unique()).union(
            set(network_data["edge_to"].unique())
        )
        mapped_nodes = set(zone_to_node_mapping["node_id"].values)
        missing_nodes = mapped_nodes - network_nodes
        if missing_nodes:
            raise ValueError(f"Nodes in mapping not found in network: {missing_nodes}")

        # Build lookup tables
        zone_to_node = dict(
            zip(zone_to_node_mapping["zone_id"], zone_to_node_mapping["node_id"])
        )
        relevance_map = dict(zip(zones["zone_id"], zones["relevance"]))
        total_relevance = zones["relevance"].sum()

        # List of zones with their data
        zone_list = sorted(zones["zone_id"].values)

        if len(zone_list) < 2:
            raise ValueError("Need at least 2 zones to generate OD matrix")

        results = []

        # For each origin zone
        for origin_zone_id in zone_list:
            origin_node = zone_to_node[origin_zone_id]
            m_i = relevance_map[origin_zone_id]

            # Get shortest paths from origin to all other zones
            distances = self._shortest_paths_from(origin_node)

            # For each destination zone
            for destination_zone_id in zone_list:
                if origin_zone_id == destination_zone_id:
                    continue

                destination_node = zone_to_node[destination_zone_id]
                distance_to_dest = distances.get(destination_node, inf)

                # If destination is unreachable or beyond threshold, skip
                if distance_to_dest > distance_threshold or distance_to_dest == inf:
                    continue

                m_j = relevance_map[destination_zone_id]

                # Count intervening opportunities
                s_ij = self._count_intervening_opportunities(
                    zone_to_node,
                    distances,
                    origin_zone_id,
                    destination_zone_id,
                    distance_threshold,
                    relevance_map,
                )

                # Calculate probability using radiation formula
                probability = self._calculate_probability(
                    m_i, m_j, s_ij, total_relevance
                )

                if probability > self.CAPACITY_EPSILON:
                    results.append(
                        {
                            "origin": origin_zone_id,
                            "destination": destination_zone_id,
                            "probability": probability,
                        }
                    )

        if not results:
            return pd.DataFrame(columns=["origin", "destination", "probability"])

        return pd.DataFrame(results)

    def _shortest_paths_from(self, origin_node):
        """Compute shortest paths from origin_node to all reachable nodes.

        Parameters
        ----------
        origin_node : str
            Origin node identifier

        Returns
        -------
        dict
            Mapping {node_id → distance}
        """
        network_data = self.network.to_dataframe()

        # Build adjacency list
        adjacency = defaultdict(list)
        has_cost = "cost" in network_data.columns

        for row in network_data.itertuples(index=False):
            edge_from = row.edge_from
            edge_to = row.edge_to
            edge_cost = getattr(row, "cost") if has_cost else 1.0

            adjacency[edge_from].append((edge_to, float(edge_cost)))
            adjacency[edge_to].append((edge_from, float(edge_cost)))

        # Dijkstra's algorithm
        distances = {origin_node: 0.0}
        heap = [(0.0, origin_node)]

        while heap:
            current_dist, current_node = heappop(heap)

            if current_dist > distances.get(current_node, inf):
                continue

            for neighbor, edge_cost in adjacency[current_node]:
                new_dist = current_dist + edge_cost

                if new_dist < distances.get(neighbor, inf):
                    distances[neighbor] = new_dist
                    heappush(heap, (new_dist, neighbor))

        return distances

    def _count_intervening_opportunities(
        self,
        zone_to_node,
        distances,
        origin_zone_id,
        destination_zone_id,
        distance_threshold,
        relevance_map,
    ):
        """Count total relevance of zones between origin and destination.

        Intervening opportunities are zones that are:
        - NOT the origin or destination
        - Within distance_threshold from origin

        Parameters
        ----------
        zone_to_node : dict
            Mapping {zone_id → node_id}
        distances : dict
            Shortest distances from origin: {node_id → distance}
        origin_zone_id : str
            Origin zone identifier
        destination_zone_id : str
            Destination zone identifier
        distance_threshold : float
            Maximum distance to consider
        relevance_map : dict
            Mapping {zone_id → relevance}

        Returns
        -------
        float
            Sum of relevance for intervening zones
        """
        s_ij = 0.0

        for zone_id, node_id in zone_to_node.items():
            # Skip origin and destination
            if zone_id == origin_zone_id or zone_id == destination_zone_id:
                continue

            distance_from_origin = distances.get(node_id, inf)

            # Include if within distance threshold
            if distance_from_origin <= distance_threshold:
                s_ij += relevance_map[zone_id]

        return s_ij

    def _calculate_probability(self, m_i, m_j, s_ij, total_relevance):
        """Calculate OD probability using radiation formula.

        Formula: P_ij = (1 / (1 - m_i/M)) * (m_i * m_j) / ((m_i + s_ij) * (m_i + m_j + s_ij))

        Parameters
        ----------
        m_i : float
            Relevance at origin
        m_j : float
            Relevance at destination
        s_ij : float
            Relevance of intervening opportunities
        total_relevance : float
            Total relevance across all zones

        Returns
        -------
        float
            Probability in [0, 1]
        """
        if total_relevance < self.CAPACITY_EPSILON:
            return 0.0

        # Normalization factor
        normalization = 1.0 - (m_i / total_relevance)

        if normalization < self.CAPACITY_EPSILON:
            return 0.0

        normalization = 1.0 / normalization

        # Numerator
        numerator = m_i * m_j

        # Denominator
        denominator = (m_i + s_ij) * (m_i + m_j + s_ij)

        if denominator < self.CAPACITY_EPSILON:
            return 0.0

        probability = normalization * (numerator / denominator)

        # Clamp to [0, 1]
        return max(0.0, min(1.0, probability))
