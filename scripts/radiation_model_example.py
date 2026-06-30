#!/usr/bin/env python
# coding: utf-8
"""Example usage of the RadiationModel for OD estimation."""

import pandas as pd

from transport_flow_model import Network, RadiationModel


def main():
    """Example: Generate travel-to-work flows using radiation model."""

    # 1. Create a simple network
    # This represents a road network with 5 nodes (A, B, C, D, E)
    network_data = pd.DataFrame(
        {
            "edge_from": ["A", "B", "B", "B", "D"],
            "edge_to": ["B", "C", "D", "A", "E"],
            "edge_id": ["AB", "BC", "BD", "BA", "DE"],
            "cost": [5.0, 3.0, 4.0, 5.0, 6.0],  # Costs in km
        }
    )
    network = Network(network_data)
    print("Network edges:", len(network_data))
    print(network_data)
    print()

    # 2. Define residential zones with population counts
    zones = pd.DataFrame(
        {
            "zone_id": [1, 2, 3, 4],
            "population": [1000, 2500, 1500, 800],  # Workers by residence
        }
    )
    print("Zones with residential population:")
    print(zones)
    print()

    # 3. Map zones to network nodes (user-provided mapping)
    # This represents where residential centers are located on the network
    mapping = pd.DataFrame(
        {
            "zone_id": [1, 2, 3, 4],
            "node_id": ["A", "B", "C", "D"],
        }
    )
    print("Zone-to-node mapping:")
    print(mapping)
    print()

    # 4. Generate OD probability matrix using radiation model
    rad = RadiationModel(network=network)

    probabilities = rad.generate(
        zones=zones,
        zone_id_column="zone_id",
        zone_to_node_mapping=mapping,
        relevance_column="population",
        distance_threshold=10.0,  # Max 10 km for work commute
    )

    print("Generated OD probability matrix (parameter-free radiation model):")
    print(probabilities)
    print()

    # 5. Calculate estimated flows from probabilities
    # Assume total outflows from each zone (e.g., workforce size)
    total_outflows = pd.DataFrame(
        {
            "zone_id": [1, 2, 3, 4],
            "outflow": [1000, 2500, 1500, 800],  # Total workers per zone
        }
    )

    # Merge with probabilities to get estimated flows
    flows = probabilities.merge(total_outflows, left_on="origin", right_on="zone_id")
    flows["flow"] = flows["outflow"] * flows["probability"]
    flows = flows[["origin", "destination", "flow", "probability"]]

    print("Estimated travel-to-work flows:")
    print(flows[flows["flow"] > 0.1])  # Only show non-negligible flows
    print()

    # 6. Summary statistics
    print("Summary Statistics:")
    print(f"  Total OD pairs: {len(probabilities)}")
    print(f"  Total flow generated: {flows['flow'].sum():.0f} workers")
    print(f"  Average flow per pair: {flows['flow'].mean():.1f} workers")
    print(f"  Non-zero probabilities: {(probabilities['probability'] > 0).sum()}")


if __name__ == "__main__":
    main()
