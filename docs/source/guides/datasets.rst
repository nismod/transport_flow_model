Benchmark datasets
==================

``tfm.datasets`` provides published benchmark problem instances with known
reference solutions, for validating and benchmarking routing and assignment
methods. Small instances are vendored with the package; larger ones are
downloaded on first use, verified against checksums, and cached in
``$TFM_CACHE_DIR`` (default ``~/.cache/transport-flow-model``).

>>> from transport_flow_model import datasets
>>> datasets.available()
['anaheim', 'barcelona', 'chicago-sketch', 'siouxfalls', 'usa-20-cities']

TNTP instances
--------------

The `Transportation Networks for Research
<https://github.com/bstabler/TransportationNetworks>`_ repository publishes
classic traffic assignment instances in TNTP format, each with best-known
user-equilibrium link flows:

.. list-table::
   :header-rows: 1

   * - Name
     - Zones
     - Nodes
     - Links
     - License / provenance
   * - ``siouxfalls``
     - 24
     - 24
     - 76
     - Open research data, `TransportationNetworks/SiouxFalls
       <https://github.com/bstabler/TransportationNetworks/tree/master/SiouxFalls>`_
       (vendored with the package)
   * - ``anaheim``
     - 38
     - 416
     - 914
     - Open research data, `TransportationNetworks/Anaheim
       <https://github.com/bstabler/TransportationNetworks/tree/master/Anaheim>`_
   * - ``barcelona``
     - 110
     - 1020
     - 2522
     - Open research data, `TransportationNetworks/Barcelona
       <https://github.com/bstabler/TransportationNetworks/tree/master/Barcelona>`_
   * - ``chicago-sketch``
     - 387
     - 933
     - 2950
     - Open research data, `TransportationNetworks/Chicago-Sketch
       <https://github.com/bstabler/TransportationNetworks/tree/master/Chicago-Sketch>`_

Data are donated to the repository and provided as-is for research use; cite
the repository (Transportation Networks for Research Core Team) when
publishing results.

Load an instance as ``Network`` and ``OD`` objects with :func:`load_tntp`:

>>> instance = datasets.load_tntp("siouxfalls")
>>> instance.n_zones, instance.n_nodes, instance.n_links
(24, 24, 76)
>>> network = instance.network.to_dataframe()
>>> list(network.columns)
['edge_from', 'edge_to', 'edge_id', 'capacity', 'cost', 'alpha', 'beta', 'length', 'speed', 'toll', 'link_type']
>>> float(instance.od.to_dataframe()["flow"].sum())
360600.0

``cost`` is the link free-flow time; ``alpha`` and ``beta`` are the BPR
volume-delay parameters, so the congested link cost at flow ``x`` is
``cost * (1 + alpha * (x / capacity)**beta)``.

Reading TNTP files directly
---------------------------

:func:`tfm.io.read_tntp` reads any pair of ``_net.tntp`` / ``_trips.tntp``
files (for example from a local clone of the TransportationNetworks
repository). It handles the TNTP conventions:

- 1-based node ids, with zones (demand centroids) numbered first;
- the ``<FIRST THRU NODE>`` convention: nodes below it are centroids that may
  start or end a trip but must not be routed *through*. Such centroids are
  split into an origin-only node (original id) and a destination-only node
  (original id plus ``centroid_offset``), and OD destinations are remapped to
  match. Map result node ids back to zone ids with
  :meth:`TNTPInstance.zone_ids`.

>>> import transport_flow_model as tfm
>>> paths = datasets.fetch("siouxfalls")
>>> instance = tfm.io.read_tntp(paths["net"], paths["trips"])
>>> instance.first_thru_node  # 1: every node may be routed through
1

Reference solutions
-------------------

Each TNTP instance ships published best-known equilibrium link flows for use
as regression fixtures:

>>> flows = datasets.best_known_flows("siouxfalls")
>>> list(flows.columns)
['edge_from', 'edge_to', 'flow', 'cost']

``datasets.BEST_KNOWN`` records the Beckmann user-equilibrium objective
evaluated on those flows (the published link costs are reproduced by the BPR
function of each ``_net.tntp`` to ~1e-14; ``chicago-sketch`` uses generalized
cost with an additional 0.04/mile distance term, recorded as
``distance_cost``):

>>> datasets.BEST_KNOWN["siouxfalls"].objective
4231335.28710744

20-city US traffic assignment benchmark
---------------------------------------

``usa-20-cities`` fetches the *unified and validated traffic dataset for 20
U.S. cities* (San Francisco, Seattle, Chicago, New York, ...): GMNS-style
``_link.csv`` / ``_node.csv`` / ``_od.csv`` inputs per city plus reference
assignment results from TransCAD, AequilibraE and UXsim. It is a single
276 MB zip archive, extracted into the cache on first fetch::

    paths = datasets.fetch("usa-20-cities")
    paths["dir"]  # extracted root, one directory per city

License: CC BY 4.0. Provenance: `figshare
<https://doi.org/10.6084/m9.figshare.24235696>`_. Citation: Xu, X., Zheng,
Z., Hu, Z. et al. A unified dataset for the city-scale traffic assignment
model in 20 U.S. cities. *Sci Data* 11, 325 (2024).
`doi:10.1038/s41597-024-03149-8 <https://doi.org/10.1038/s41597-024-03149-8>`_.
