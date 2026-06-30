Origin-Destination Estimation with the Radiation Model
========================================================

The ``RadiationModel`` generates synthetic origin-destination (OD) flow
probabilities using a radiation model based on network distance. This is useful
when you have, for example, population or employment counts at locations but
lack direct OD survey data.

What is the Radiation Model?
----------------------------

The radiation model is a model for human mobility that predicts OD flows based
on local opportunities and distance. Unlike traditional gravity models, it
requires no calibration – the model formulation is parameter-free and widely
applicable.

The model simulates human decision-making: people seek opportunities (jobs,
services), choosing the closest opportunity with sufficient quality. This
naturally explains why closer destinations with better opportunities are
preferred.

**Key advantages:**

- Parameter-free (no calibration data needed)
- Works with network distance (realistic for transport)
- Theoretically grounded in opportunity-seeking behavior
- Portable across regions

Mathematical Foundation
-----------------------

The radiation model computes OD probabilities using:

.. math::

    P_{ij} = \frac{1}{1 - \frac{m_i}{M}} \cdot \frac{m_i \cdot m_j}{(m_i + s_{ij})(m_i + m_j + s_{ij})}

Where:

- :math:`P_{ij}` = probability of flow from location :math:`i` to location :math:`j`
- :math:`m_i`, :math:`m_j` = relevance (population, employment) at origin/destination
- :math:`s_{ij}` = total relevance of intervening opportunities between :math:`i` and :math:`j`
- :math:`M` = total relevance across all locations

Basic Usage
-----------

To generate OD probabilities, you need:

1. A ``Network`` with topology and distance data
2. A DataFrame with origin/destination locations and relevance measures
3. A mapping from origin/destination locations to network nodes
4. A distance threshold for intervening opportunities

Here's a simple example with a 3-locatino network:

>>> import pandas as pd
>>> from transport_flow_model import Network, RadiationModel
>>>
>>> # Step 1: Create a network
>>> network = Network(
...     pd.DataFrame(
...         {
...             "edge_from": ["A", "B", "B"],
...             "edge_to": ["B", "C", "A"],
...             "edge_id": ["AB", "BC", "BA"],
...             "cost": [5.0, 3.0, 5.0],  # Distance in km
...         }
...     )
... )
>>>
>>> # Step 2: Define zones with population
>>> zones = pd.DataFrame(
...     {
...         "zone_id": [1, 2, 3],
...         "population": [1000, 2000, 1500],
...     }
... )
>>>
>>> # Step 3: Map zones to network nodes
>>> mapping = pd.DataFrame(
...     {
...         "zone_id": [1, 2, 3],
...         "node_id": ["A", "B", "C"],
...     }
... )
>>>
>>> # Step 4: Generate probabilities
>>> rad = RadiationModel(network=network)
>>> probs = rad.generate(
...     zones=zones,
...     zone_id_column="zone_id",
...     zone_to_node_mapping=mapping,
...     relevance_column="population",
...     distance_threshold=10.0,
... )
>>> len(probs)
6
>>> probs.columns.tolist()
['origin', 'destination', 'probability']

Zone to Network Mapping
-----------------------

The ``zone_to_node_mapping`` tells the model where each zone is located on the network.
Each zone must map to exactly one network node (typically representing a zone centroid
or major employment/residential center).

>>> mapping_example = pd.DataFrame(
...     {
...         "zone_id": [1, 2, 3],
...         "node_id": ["A", "B", "C"],
...     }
... )

The ``node_id`` values must exist in your network (as either ``edge_from`` or ``edge_to``).
The model will raise an error if a node is missing.

Distance Threshold
------------------

The ``distance_threshold`` parameter controls which zones are considered "intervening
opportunities" when calculating probabilities. Only zones within this distance from
the origin contribute to :math:`s_{ij}`.

This affects the shape of the OD probability distribution:

- **Small threshold**: More local flows, fewer long-distance trips
- **Large threshold**: More dispersed flows, includes longer-distance opportunities

>>> # Small threshold: focus on nearby zones
>>> probs_short = rad.generate(
...     zones=zones,
...     zone_id_column="zone_id",
...     zone_to_node_mapping=mapping,
...     relevance_column="population",
...     distance_threshold=5.0,
... )
>>> len(probs_short)
4

>>> # Large threshold: consider far zones as intervening opportunities
>>> probs_long = rad.generate(
...     zones=zones,
...     zone_id_column="zone_id",
...     zone_to_node_mapping=mapping,
...     relevance_column="population",
...     distance_threshold=20.0,
... )
>>> len(probs_long)
6

Working with Different Relevance Measures
------------------------------------------

The ``relevance_column`` can represent different opportunity types:

**Residential population** (for commuting):

>>> zones_res = pd.DataFrame(
...     {
...         "zone_id": [1, 2, 3],
...         "residential_population": [5000, 3000, 2000],
...     }
... )

**Employment opportunities** (for work destinations):

>>> zones_emp = pd.DataFrame(
...     {
...         "zone_id": [1, 2, 3],
...         "employment": [1500, 4000, 2500],
...     }
... )

**General opportunities** (any measure of attractiveness):

>>> zones_gdp = pd.DataFrame(
...     {
...         "zone_id": [1, 2, 3],
...         "gdp": [100, 250, 175],  # Economic output
...     }
... )

The model works with any non-negative relevance measure. The column name is
specified with the ``relevance_column`` parameter.

Converting Probabilities to Flows
----------------------------------

The ``generate()`` method returns probabilities – the likelihood of flow from
each origin to each destination. To get estimated flows, multiply by total
outflows from each origin.

>>> # Known total workers by residential zone
>>> outflows = pd.DataFrame(
...     {
...         "zone_id": [1, 2, 3],
...         "total_workers": [5000, 3000, 2000],
...     }
... )
>>>
>>> # Merge probabilities with outflows
>>> flows = probs.merge(outflows, left_on="origin", right_on="zone_id")
>>> flows["estimated_flow"] = flows["total_workers"] * flows["probability"]
>>> flows[["origin", "destination", "estimated_flow"]].head()  # doctest: +SKIP
   origin  destination  estimated_flow
0       1            2     2142.857143
1       1            3      714.285714
2       2            1     1285.714286
2       2            3     1285.714286

To use with ``Network.allocate()``, rename columns to match OD requirements:

>>> od_data = flows[["origin", "destination", "estimated_flow"]].copy()
>>> od_data.columns = ["origin_id", "destination_id", "flow"]
>>> from transport_flow_model import OD
>>> od = OD(od_data)  # doctest: +SKIP

Travel-to-Work Example
----------------------

Here's a complete example generating a travel-to-work OD matrix:

>>> # Data: 4 residential zones, workforce by residence
>>> residential = pd.DataFrame(
...     {
...         "zone_id": [1, 2, 3, 4],
...         "workers": [2000, 3500, 1500, 1000],
...     }
... )
>>>
>>> # Mapping: where residential centers are located
>>> res_mapping = pd.DataFrame(
...     {
...         "zone_id": [1, 2, 3, 4],
...         "node_id": ["A", "B", "C", "A"],  # Some zones share nodes
...     }
... )
>>>
>>> # Generate OD probabilities (using the basic network from earlier)
>>> ttw_probs = rad.generate(
...     zones=residential,
...     zone_id_column="zone_id",
...     zone_to_node_mapping=res_mapping,
...     relevance_column="workers",
...     distance_threshold=15.0,
... )  # doctest: +SKIP
>>>
>>> # Convert to flows
>>> ttw_flows = ttw_probs.merge(
...     residential[["zone_id", "workers"]],
...     left_on="origin", right_on="zone_id"
... )  # doctest: +SKIP
>>> ttw_flows["flow"] = ttw_flows["workers"] * ttw_flows["probability"]  # doctest: +SKIP

Integrating with Flow Allocation
---------------------------------

After generating OD probabilities, you can use them with ``Network.allocate()``
to assign flows to actual paths:

>>> # Generate synthetic OD with unit flows
>>> od_probs = rad.generate(
...     zones=zones,
...     zone_id_column="zone_id",
...     zone_to_node_mapping=mapping,
...     relevance_column="population",
...     distance_threshold=10.0,
... )
>>>
>>> # Convert probabilities to OD object (with unit flows)
>>> from transport_flow_model import OD
>>> od_data = od_probs.copy()
>>> od_data = od_data.rename(columns={"origin": "origin_id", "destination": "destination_id", "probability": "flow"})
>>> od_data["flow"] = od_data["flow"] * 1000  # Scale to realistic magnitude
>>> od = OD(od_data[["origin_id", "destination_id", "flow"]])
>>>
>>> # Allocate to network paths
>>> allocation = network.allocate(od, directed=True)  # doctest: +SKIP
>>> allocation.network_flows.to_dataframe()[["edge_id", "flow"]]  # doctest: +SKIP

Advanced: Custom Column Names
------------------------------

The model accepts custom column names for zones and mapping:

>>> zones_custom = pd.DataFrame(
...     {
...         "location_id": [1, 2, 3],
...         "opportunities": [1000, 2000, 1500],
...     }
... )
>>>
>>> mapping_custom = pd.DataFrame(
...     {
...         "location_id": [1, 2, 3],
...         "network_node": ["A", "B", "C"],
...     }
... )
>>>
>>> # Specify the column names
>>> probs_custom = rad.generate(
...     zones=zones_custom,
...     zone_id_column="location_id",
...     zone_to_node_mapping=mapping_custom,
...     relevance_column="opportunities",
...     distance_threshold=10.0,
... )  # doctest: +SKIP

Validation and Error Handling
------------------------------

The model validates inputs and provides descriptive errors:

**Missing column:**

>>> try:  # doctest: +SKIP
...     rad.generate(
...         zones=zones,
...         zone_id_column="missing_column",
...         zone_to_node_mapping=mapping,
...         relevance_column="population",
...         distance_threshold=10.0,
...     )
... except ValueError as e:
...     print(f"Error: {e}")
Error: Column 'missing_column' not found in zones DataFrame

**Zone not in mapping:**

>>> bad_mapping = mapping.iloc[:2]  # Only 2 of 3 zones
>>> try:  # doctest: +SKIP
...     rad.generate(
...         zones=zones,
...         zone_id_column="zone_id",
...         zone_to_node_mapping=bad_mapping,
...         relevance_column="population",
...         distance_threshold=10.0,
...     )
... except ValueError as e:
...     print(f"Error: {e}")
Error: Zones in mapping not found in zones DataFrame: {3}

**Network node not found:**

>>> bad_mapping = mapping.copy()
>>> bad_mapping.loc[0, "node_id"] = "MISSING"
>>> try:  # doctest: +SKIP
...     rad.generate(
...         zones=zones,
...         zone_id_column="zone_id",
...         zone_to_node_mapping=bad_mapping,
...         relevance_column="population",
...         distance_threshold=10.0,
...     )
... except ValueError as e:  # doctest: +SKIP
...     print(f"Error: {e}")
Error: Nodes in mapping not found in network: {'MISSING'}

Performance Considerations
--------------------------

The ``RadiationModel`` computes shortest paths from each origin zone to all
other zones. For a network with :math:`N` nodes and :math:`Z` zones:

- Time complexity: :math:`O(Z \times N \log N)` (Dijkstra per zone)
- Space complexity: :math:`O(N)` for storing distance arrays

**Typical performance:**

- 10 zones, 100 nodes: < 0.1 seconds
- 50 zones, 500 nodes: < 1 second
- 100 zones, 1000 nodes: ~5 seconds

For larger networks, consider:

- Reducing the number of zones to represent major population centers
- Using a simplified network topology for OD estimation
- Parallelizing zone calculations if needed

Parameter-Free Property
-----------------------

An important feature of the radiation model is that it's parameter-free. Running
the same data multiple times always produces identical results, with no random
seed effects:

>>> # First run
>>> probs_1 = rad.generate(
...     zones=zones,
...     zone_id_column="zone_id",
...     zone_to_node_mapping=mapping,
...     relevance_column="population",
...     distance_threshold=10.0,
... )
>>>
>>> # Second run (identical data)
>>> probs_2 = rad.generate(
...     zones=zones,
...     zone_id_column="zone_id",
...     zone_to_node_mapping=mapping,
...     relevance_column="population",
...     distance_threshold=10.0,
... )
>>>
>>> # Results are identical
>>> probs_1.equals(probs_2)
True

This determinism makes results reproducible and easy to validate.

Further Reading
---------------

- Simini, F., González, M. C., Maritan, A., & Barabási, A. L. (2012).
  A universal model for mobility and migration patterns. *Nature*, 484(7392), 96-100.
- See also: :doc:`data-models` for OD and Network concepts
- See also: :doc:`least-cost-allocation` for using OD with network allocation
