Config-Driven Runs
==================

A JSON config names the input tables, their column mappings and the
assignment method, so a run is described by a file rather than a script.
:class:`~transport_flow_model.RunConfig` validates it and returns the same
API objects you would have built by hand.

The config schema is part of the versioned public interface — see
:doc:`../versioning` — so a field will not change meaning without a
changelog entry.

Loading a config
----------------

>>> import json, tempfile, pathlib
>>> from transport_flow_model import RunConfig
>>> config_text = {
...     "paths": {"data": "./data", "results": "./results"},
...     "assignment": {"method": "sequential", "capacity_constrained": True},
... }
>>> with tempfile.TemporaryDirectory() as tmpdir:
...     path = pathlib.Path(tmpdir) / "run.json"
...     _ = path.write_text(json.dumps(config_text))
...     config = RunConfig.from_json(path)
>>> config.assignment.method
'sequential'

Blocks
------

``paths``
    ``data`` and ``results`` are required; ``incoming_data`` and ``figures``
    are optional. Relative paths resolve against the current working
    directory.

``network`` and ``demand``
    A ``path`` (relative to ``paths.data``) and a ``columns`` mapping from
    the file's own column names to the API's. Both have defaults matching
    this repository's processed-data conventions, so a config that uses
    those conventions can omit them entirely.

>>> config.network.path
PosixPath('network/network.csv')
>>> config.network.columns["from_id"], config.network.columns["gcost_usd_per_ton"]
('edge_from', 'cost')

``scenarios``
    A ``path`` to a failure set and the ``id_column`` naming the link to
    remove, one single-link scenario per row.

``assignment``
    The method and its options. Covered below.

Then load the objects:

.. code-block:: python

    network = config.load_network()
    demand = config.load_demand()
    scenarios = config.load_scenarios()

    result = assign(network, demand, config.assignment.method,
                    **config.assignment.options(network))

Choosing a method and its options
---------------------------------

``options()`` returns the keyword arguments for the chosen method, and it
reads that method's own signature to decide what it may pass. So
``capacity_constrained`` reaches ``"sequential"``, which accepts it, and is
dropped for ``"msa"``, which does not:

>>> from transport_flow_model.config import AssignmentConfig
>>> AssignmentConfig(method="sequential").options()
{'capacity_constrained': True, 'directed': True}
>>> AssignmentConfig(method="msa").options()
{'directed': True}

Only a field left at its **default** is dropped that way. A field the config
*sets* is an instruction, so a method that cannot accept it raises rather
than quietly ignoring it:

>>> asked = AssignmentConfig.model_validate(
...     {"method": "msa", "capacity_constrained": True}
... )
>>> asked.options()
Traceback (most recent call last):
    ...
ValueError: Assignment method 'msa' does not accept 'capacity_constrained', which this config sets; its options are cost_function, directed, distance_cost, max_iterations, target_gap, time_limit_s

Method-specific options go in ``method_options``, which is passed through
untouched — an unknown one raises from the method itself:

>>> equilibrium = AssignmentConfig.model_validate(
...     {"method": "msa", "method_options": {"max_iterations": 200, "target_gap": 1e-3}}
... )
>>> equilibrium.options()
{'directed': True, 'max_iterations': 200, 'target_gap': 0.001}

Choosing a cost function
------------------------

``cost_function`` names one of the registered volume-delay functions
(:doc:`equilibrium`) plus its parameters. It is built against the network,
which is why ``options()`` takes one when this field is set:

.. code-block:: json

    {
      "paths": {"data": "./data", "results": "./results"},
      "assignment": {
        "method": "msa",
        "cost_function": {"name": "conical", "alpha": 4.0},
        "method_options": {"max_iterations": 500, "target_gap": 0.0001}
      }
    }

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
>>> conical_config = AssignmentConfig.model_validate(
...     {"method": "msa", "cost_function": {"name": "conical", "alpha": 4.0}}
... )
>>> type(conical_config.options(network)["cost_function"]).__name__
'Conical'

Naming a cost function a method cannot use raises for the same reason as
above — ``"sequential"`` assigns at fixed costs and takes no curve:

>>> AssignmentConfig.model_validate(
...     {"method": "sequential", "cost_function": {"name": "conical"}}
... ).options(network)
Traceback (most recent call last):
    ...
ValueError: Assignment method 'sequential' does not accept 'cost_function', which this config sets; its options are capacity_constrained, directed

Running the scripts
-------------------

Two scripts take a config path and run end to end:

.. code-block:: bash

    python scripts/flow_model/flow_allocation.py ./config.json
    python scripts/flow_model/flow_disruptions.py ./config.json

``flow_allocation.py`` assigns demand and writes the flow tables to
``paths.results``. ``flow_disruptions.py`` assigns a baseline, evaluates the
failure set, and writes per-scenario losses.
