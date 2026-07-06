API reference
=============

The public v0 API is the set of names exported from
``transport_flow_model``: build a :class:`~transport_flow_model.Network`
and a :class:`~transport_flow_model.Demand`, run
:func:`~transport_flow_model.assign`, and evaluate disruption
:class:`~transport_flow_model.Scenario` sets with
:func:`~transport_flow_model.disrupt`. Results are
:class:`pyarrow.Table` values with provenance metadata.

Network
-------

.. automodule:: transport_flow_model.network
   :members:
   :undoc-members:

Demand
------

.. automodule:: transport_flow_model.demand
   :members:
   :undoc-members:

Assignment
----------

.. automodule:: transport_flow_model.assignment
   :members:
   :undoc-members:

Disruption
----------

.. automodule:: transport_flow_model.disruption
   :members:
   :undoc-members:

Configuration
-------------

.. automodule:: transport_flow_model.config
   :members:
   :undoc-members:

Origin-destination estimation
-----------------------------

.. automodule:: transport_flow_model.radiation
   :members:

Input/output
------------

.. automodule:: transport_flow_model.io
   :members:

Benchmark datasets
------------------

.. automodule:: transport_flow_model.datasets
   :members:

Legacy model classes (deprecated)
---------------------------------

The tabular classes in :mod:`transport_flow_model.model` predate the v0
API. They remain importable while functionality is ported, but new code
should use the API above; see :doc:`versioning`.

.. automodule:: transport_flow_model.model
   :members: Network, OD, ODFlows, NetworkFlows, compute_losses
   :no-index:
