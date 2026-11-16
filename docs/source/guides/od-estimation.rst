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
- :math:`s_{ij}` = intervening opportunities: the total destination relevance
  of every other location :math:`k` at least as close to :math:`i` as :math:`j` is
  (:math:`d_{ik} \le d_{ij}`, excluding :math:`i` and :math:`j`)
- :math:`M` = total relevance across all locations

Origin and destination relevance can differ (see
`Separate Origin and Destination Masses`_ below); :math:`m_j` and :math:`s_{ij}`
then use the destination relevance.

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

The ``distance_threshold`` parameter is the maximum network distance from
origin to destination. Destinations further away get zero probability.
Intervening opportunities do not depend on the threshold: they are the zones
closer to the origin than the destination.

- **Small threshold**: only local destinations
- **Large threshold**: longer trips are possible as well

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

To use with ``assign()``, rename columns to match ``Demand`` requirements:

>>> od_data = flows[["origin", "destination", "estimated_flow"]].copy()
>>> od_data.columns = ["origin_id", "destination_id", "value"]
>>> from transport_flow_model import Demand
>>> demand = Demand(od_data)  # doctest: +SKIP

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

After generating OD probabilities, you can use them with ``assign()``
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
>>> # Convert probabilities to Demand (with unit flows)
>>> from transport_flow_model import Demand, assign
>>> od_data = od_probs.copy()
>>> od_data = od_data.rename(columns={"origin": "origin_id", "destination": "destination_id", "probability": "value"})
>>> od_data["value"] = od_data["value"] * 1000  # Scale to realistic magnitude
>>> demand = Demand(od_data[["origin_id", "destination_id", "value"]])
>>>
>>> # Assign to network paths (ids in demand must be network node ids)
>>> result = assign(network, demand, "sequential", directed=True)  # doctest: +SKIP
>>> result.link_flows.to_pandas()[["edge_id", "flow"]]  # doctest: +SKIP

Separate Origin and Destination Masses
--------------------------------------

For journeys to work, trips are produced by workers at home and attracted by
jobs. Give the destination mass with ``destination_relevance_column``: it is
used for :math:`m_j` and for the intervening opportunities :math:`s_{ij}`.
Because :math:`m_i` and :math:`s_{ij}` are added together, the two masses
should be in comparable units (e.g. scale jobs to the same total as workers).

>>> commute_zones = pd.DataFrame(
...     {
...         "zone_id": [1, 2, 3],
...         "workers": [1000, 200, 0],
...         "jobs": [100, 300, 800],
...     }
... )
>>> commute_probs = rad.generate(
...     zones=commute_zones,
...     zone_id_column="zone_id",
...     zone_to_node_mapping=mapping,
...     relevance_column="workers",
...     destination_relevance_column="jobs",
...     distance_threshold=10.0,
...     normalisation="row",
... )
>>> commute_probs.groupby("origin")["probability"].sum().round(6).to_dict()
{1: 1.0, 2: 1.0}

Zone 3 has no workers, so it produces no trips. ``normalisation="row"``
rescales each origin's probabilities to sum to one over the destinations it can
reach. The default, ``"finite"``, applies Simini's finite-size correction
:math:`1 / (1 - m_i / M)` instead.

Flows
-----

``RadiationModel.flows`` multiplies the probabilities by trip production and
returns a table ready for ``Demand``. Its ``origin_id`` and ``destination_id``
columns hold zone ids, and the matching network nodes are in
``origin_node_id`` and ``destination_node_id``. Flows below ``min_flow`` are
dropped, and their total is spread back over each origin's other destinations.

>>> commute = rad.flows(
...     commute_zones,
...     "zone_id",
...     mapping,
...     production_column="workers",
...     destination_mass_column="jobs",
...     max_impedance=10.0,
... )
>>> commute.groupby("origin_id")["value"].sum().round(6).to_dict()
{1: 1000.0, 2: 200.0}
>>> demand = Demand(
...     commute[["origin_node_id", "destination_node_id", "value"]].rename(
...         columns={"origin_node_id": "origin_id", "destination_node_id": "destination_id"}
...     )
... )

Precomputed Impedance
---------------------

``radiation_probabilities`` and ``radiation_flows`` work on any dense
zone-by-zone impedance matrix, such as travel time or generalised cost. Use
``RadiationModel.impedance_matrix`` to compute one from a network with
Dijkstra. Set ``directed=True`` to respect link direction.

>>> import numpy as np
>>> from transport_flow_model.radiation import radiation_flows
>>> impedance = rad.impedance_matrix(["A", "B", "C"])
>>> flows = radiation_flows(impedance, [1000, 200, 0], destination_mass=[100, 300, 800])
>>> flows.sum(axis=1).round(6).tolist()
[1000.0, 200.0, 0.0]

Constraining to Observed Aggregate Flows
----------------------------------------

Observed commuting flows are often available only between larger areas, such
as a census travel-to-work table between districts. ``aggregate_od`` sums a
zone OD matrix to those areas for validation. ``scale_to_aggregate`` scales
it to match the observed totals, keeping the radiation model's pattern within
each pair of areas.

>>> from transport_flow_model import aggregate_od, od_fit, scale_to_aggregate
>>> districts = ["north", "north", "south"]
>>> observed = np.array([[600.0, 400.0], [100.0, 0.0]])
>>> modelled, labels = aggregate_od(flows, districts)
>>> metrics, table = od_fit(modelled, observed, labels)
>>> sorted(metrics)[:3]
['cpc', 'modelled_internal_share', 'modelled_total']
>>> constrained = scale_to_aggregate(flows, districts, observed, method="ipf")
>>> np.round(constrained.sum(axis=1), 6).tolist()  # zone production kept
[1000.0, 200.0, 0.0]

With ``method="ipf"``, the observed totals are first rescaled within each
origin area to match the modelled production there. This allows, for
example, a census from a different year. The matrix then alternates between
matching the area-pair totals and the zone production. ``method="block"``
matches the area-pair totals exactly, but changes zone production. Area
pairs with observed flow but no modelled flow cannot be matched, because
scaling cannot create flow.

``common_part_of_commuters`` (CPC) is a common measure of fit between two OD
matrices: 1 means identical, and 0 means no overlap.

Trips Within a Zone
-------------------

By default a zone is never its own destination. With ``include_self=True``,
the origin also competes as a destination, at the intrazonal impedance on the
diagonal of the impedance matrix (for example, a mean travel time within the
zone). The model then decides how many trips stay within the zone, rather
than the user fixing a share. The origin's own destination mass is never
counted as an intervening opportunity. ``include_self=True`` needs
``normalisation="row"`` (the default for ``radiation_flows``) or ``"none"``:
the ``"finite"`` correction assumes a zone is not its own destination, so
that combination raises ``ValueError``.

>>> d_self = impedance.copy()
>>> np.fill_diagonal(d_self, 1.0)
>>> with_self = radiation_flows(
...     d_self, [1000, 200, 0], destination_mass=[100, 300, 800], include_self=True
... )
>>> bool(with_self[0, 0] > 0)
True

Calibrating to Observed Aggregate Flows
---------------------------------------

The radiation model has no free parameter, and it depends only on the order
in which destinations are ranked by impedance. What can still be chosen is
which impedance to rank by, the ``max_impedance`` cut-off, the origin mass,
and whether trips may stay within a zone. ``calibrate_radiation`` runs every
combination and ranks them by fit to observed aggregate flows:

>>> from transport_flow_model import calibrate_radiation
>>> results = calibrate_radiation(
...     {"cost": d_self},
...     np.array([1000.0, 200.0, 0.0]),
...     np.array([100.0, 300.0, 800.0]),
...     districts,
...     observed,
...     max_impedances=[5.0, 10.0],
...     include_self=[False, True],
... )
>>> list(results.columns[:4])
['impedance', 'origin_mass', 'max_impedance', 'include_self']

For a continuous parameter to fit, pass ``alphas``: values of the scale
parameter :math:`\alpha` of the extended radiation model (Yang et al. 2014).
Smaller values weaken the effect of intervening opportunities, giving longer
trips. The paper finds values near zero for zones smaller than about 10 km.
``None`` is the original model.

>>> results = calibrate_radiation(
...     {"cost": d_self},
...     np.array([1000.0, 200.0, 0.0]),
...     np.array([100.0, 300.0, 800.0]),
...     districts,
...     observed,
...     alphas=[None, 0.1, 0.5, 1.0],
... )
>>> len(results)
4

Use the best row's settings, then (optionally) ``scale_to_aggregate`` to
match the observed totals exactly.

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

There are two steps:

- **Impedance**: Dijkstra from each distinct zone node (igraph, in batches of
  ``chunk_size`` sources), :math:`O(Z \times N \log N)`.
- **Radiation**: one sort per origin over the dense matrix,
  :math:`O(Z^2 \log Z)` time and :math:`O(Z^2)` memory.

**Indicative timings:**

- 2,500 zones, radiation step only: about 1 second
- 2,500 zones on a 90,000-node, 180,000-link network, impedance: about
  40 seconds

The dense matrices need :math:`8 Z^2` bytes each (50 MB at 2,500 zones).
That is fine up to roughly 10,000 zones.

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
