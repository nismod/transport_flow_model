Run a Model from a Config File
==============================

Describe a run in JSON instead of code, so it can be version-controlled,
diffed and handed to someone else. For the full schema, see
:doc:`../reference/configuration`.

Load a config and run it
------------------------

:meth:`~transport_flow_model.RunConfig.from_json` validates the file and
returns the API objects.

.. code-block:: python

    from transport_flow_model import RunConfig, assign

    config = RunConfig.from_json("./config.json")
    network = config.load_network()
    demand = config.load_demand()

    result = assign(
        network,
        config.load_demand(),
        config.assignment.method,
        **config.assignment.options(network),
    )

``options()`` returns the keyword arguments for the chosen method, so the
same two lines run whichever method the file names.

Use the command-line scripts
----------------------------

Two scripts do the above end to end and write their output to
``paths.results``:

.. code-block:: bash

    python scripts/flow_model/flow_allocation.py ./config.json
    python scripts/flow_model/flow_disruptions.py ./config.json

``flow_allocation.py`` assigns demand and writes the flow tables.
``flow_disruptions.py`` assigns a baseline, evaluates the failure set named
in the ``scenarios`` block, and writes per-scenario losses.

Point at your own column names
------------------------------

Input files rarely use the API's column names. The ``network`` and
``demand`` blocks carry a ``columns`` mapping from the file's names to the
API's, so no preprocessing step is needed:

.. code-block:: json

    {
      "paths": {"data": "./data", "results": "./results"},
      "network": {
        "path": "links.csv",
        "columns": {
          "from_node": "edge_from",
          "to_node": "edge_to",
          "link_id": "edge_id",
          "free_flow_time": "cost",
          "lanes_capacity": "capacity"
        }
      }
    }

Both blocks default to this repository's processed-data conventions, so a
config following those can omit them entirely.

Run an equilibrium assignment
-----------------------------

Method-specific options go in ``method_options``:

.. code-block:: json

    {
      "paths": {"data": "./data", "results": "./results"},
      "assignment": {
        "method": "msa",
        "method_options": {"max_iterations": 2000, "target_gap": 0.001}
      }
    }

>>> from transport_flow_model.config import AssignmentConfig
>>> equilibrium = AssignmentConfig.model_validate(
...     {"method": "msa", "method_options": {"max_iterations": 2000, "target_gap": 1e-3}}
... )
>>> equilibrium.options()
{'directed': True, 'max_iterations': 2000, 'target_gap': 0.001}

Note that ``capacity_constrained`` is absent: MSA does not accept it, and
``options()`` reads the method's own signature to decide what to pass. See
:doc:`../reference/configuration` for what happens when a config sets an
option its method cannot take.

Assign with a different cost curve
----------------------------------

Name it in ``cost_function`` with its parameters. It is built against the
network, which is why ``options()`` needs one here:

.. code-block:: json

    {
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
