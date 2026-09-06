Configuration schema
====================

A JSON run config validated by :class:`~transport_flow_model.RunConfig`. The
schema is part of the versioned public interface (see :doc:`versioning`), so
a field will not change meaning without a changelog entry.

To use it, see :doc:`../how-to/run-from-a-config`.

Top-level blocks
----------------

======================= ========== ==================================================
Block                   Required   Contents
======================= ========== ==================================================
``paths``               yes        Input and output directories
``network``             no         Link table path and column mapping
``demand``              no         OD table path and column mapping
``scenarios``           no         Failure set path and id column
``assignment``          no         Method, its options and the cost function
======================= ========== ==================================================

Unknown top-level keys are ignored, so a ``comment`` key is permitted.

``paths``
---------

=================== ========== ================================================
Field               Required   Meaning
=================== ========== ================================================
``data``            yes        Root for the input tables below
``results``         yes        Where results are written
``incoming_data``   no         Unprocessed source data
``figures``         no         Figure output
=================== ========== ================================================

Relative paths resolve against the current working directory.

``network`` and ``demand``
--------------------------

Each takes a ``path`` (relative to ``paths.data``, unless absolute) and a
``columns`` mapping from the file's own column names to the API's. ``.parquet``
files are read with :func:`pandas.read_parquet`, anything else with
:func:`pandas.read_csv`, in both cases reading only the mapped columns.

Defaults match this repository's processed-data conventions:

>>> from transport_flow_model.config import NetworkConfig, DemandConfig
>>> NetworkConfig().path
PosixPath('network/network.csv')
>>> sorted(NetworkConfig().columns.items())[:3]
[('flow_capacity', 'capacity'), ('from_id', 'edge_from'), ('gcost_usd_per_ton', 'cost')]
>>> DemandConfig().path, sorted(DemandConfig().columns.items())
(PosixPath('od/od.csv'), [('destination_id', 'destination_id'), ('origin_id', 'origin_id'), ('tons', 'value')])

``scenarios``
-------------

=================== ============================ =============================
Field               Default                      Meaning
=================== ============================ =============================
``path``            ``damages/failure_set.csv``  Failure set table
``id_column``       ``edge_id``                  Column naming the link to remove
=================== ============================ =============================

One single-link removal scenario is built per row.

``assignment``
--------------

======================== ================= =====================================
Field                    Default           Meaning
======================== ================= =====================================
``method``               ``"sequential"``  Name from ``assignment.METHODS``
``capacity_constrained`` ``true``          Sequential only
``directed``             ``true``          Treat links as one-way
``cost_function``        ``null``          ``{"name": ..., **params}``
``method_options``       ``{}``            Passed to the method unfiltered
======================== ================= =====================================

``cost_function`` names a curve from
:data:`transport_flow_model.costs.COST_FUNCTIONS` — ``"bpr"``, ``"conical"``
or ``"speed_flow"`` — and the remaining keys are its parameters.

How options are resolved
------------------------

``options()`` returns the keyword arguments for the chosen method. It reads
that method's own signature, so ``capacity_constrained`` reaches
``"sequential"``, which declares it, and not ``"msa"``, which does not:

>>> from transport_flow_model.config import AssignmentConfig
>>> AssignmentConfig(method="sequential").options()
{'capacity_constrained': True, 'directed': True}
>>> AssignmentConfig(method="msa").options()
{'directed': True}

Only a field left at its **default** is dropped that way. A field the config
*sets* is an instruction, so a method that cannot accept it raises:

>>> AssignmentConfig.model_validate(
...     {"method": "msa", "capacity_constrained": True}
... ).options()
Traceback (most recent call last):
    ...
ValueError: Assignment method 'msa' does not accept 'capacity_constrained', which this config sets; its options are cost_function, directed, distance_cost, max_iterations, target_gap, time_limit_s

The same applies to a cost function a method cannot use — ``"sequential"``
assigns at fixed costs and takes no curve:

>>> import pandas as pd
>>> from transport_flow_model import Network
>>> network = Network.from_dataframe(
...     pd.DataFrame(
...         {
...             "edge_from": ["A"],
...             "edge_to": ["B"],
...             "edge_id": ["AB"],
...             "cost": [1.0],
...             "capacity": [100.0],
...         }
...     )
... )
>>> AssignmentConfig.model_validate(
...     {"method": "sequential", "cost_function": {"name": "conical"}}
... ).options(network)
Traceback (most recent call last):
    ...
ValueError: Assignment method 'sequential' does not accept 'cost_function', which this config sets; its options are capacity_constrained, directed

``method_options`` is passed through unfiltered, so an unknown key there
raises from the method itself rather than being dropped here.

Loading
-------

>>> import json, tempfile, pathlib
>>> from transport_flow_model import RunConfig
>>> with tempfile.TemporaryDirectory() as tmpdir:
...     path = pathlib.Path(tmpdir) / "run.json"
...     _ = path.write_text(
...         json.dumps({"paths": {"data": "./data", "results": "./results"}})
...     )
...     config = RunConfig.from_json(path)
>>> config.assignment.method
'sequential'
