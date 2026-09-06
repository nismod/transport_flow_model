Transport Flow Model
====================

``transport-flow-model`` models flows on transport networks for
infrastructure risk and resilience analysis. It assigns origin-destination
demand to network routes, evaluates what happens to those flows when links
are disrupted, and quantifies the resulting rerouting cost and loss of
access.

Performance-critical routing and allocation run in a Rust core; data is
handled as Apache Arrow tables throughout.

Installation
------------

The Rust extension is required, so an editable install also needs a build
step (and a Rust toolchain):

.. code-block:: bash

   pip install -e .
   maturin develop --release

Contributors should use `pixi <https://pixi.prefix.dev>`_ instead — see
:doc:`development`.

At a glance
-----------

.. code-block:: python

   from transport_flow_model import Network, Demand, assign, disrupt

   network = Network.from_dataframe(links)   # edge_from, edge_to, edge_id, cost, ...
   demand = Demand.from_dataframe(od)        # origin_id, destination_id, value

   result = assign(network, demand, method="sequential", include_paths=True)
   result.link_flows          # per-link flow
   result.skims               # per-OD-pair cost

   summary = disrupt(network, scenarios, base=result).summary()

Start with :doc:`guides/data-models` for the objects involved, then
:doc:`guides/least-cost-allocation` and :doc:`guides/equilibrium` for
assignment, and :doc:`guides/disruptions` for scenario analysis. Runs can be
driven from a JSON file instead — see :doc:`guides/configuration`.

.. toctree::
   :maxdepth: 2
   :caption: User guides

   guides/data-models
   guides/datasets
   guides/od-estimation
   guides/least-cost-allocation
   guides/equilibrium
   guides/multiple-flows-and-capacity
   guides/disruptions
   guides/losses
   guides/configuration

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api
   versioning

.. toctree::
   :maxdepth: 2
   :caption: Development

   development


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
