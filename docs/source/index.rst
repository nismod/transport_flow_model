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

Where to start
--------------

This documentation is organised along `Diátaxis <https://diataxis.fr>`_
lines, so what you need depends on what you are doing:

- **New here?** :doc:`tutorials/getting-started` takes you from a fresh
  install to an assigned, converged and disrupted network in one sitting.
- **Have a job to do?** The :ref:`how-to guides <how-to-guides>` each solve
  one problem — converge an assignment, evaluate disruptions, estimate a
  demand matrix.
- **Need a fact?** :doc:`reference/api` documents every public name, and
  :doc:`reference/configuration` the JSON schema.
- **Want to understand it?** The :ref:`explanation <explanation>` pages
  cover why equilibrium matters, what the relative gap measures, and how
  criticality rankings can mislead.

Design decisions and internals live in the repository rather than here:
``ARCHITECTURE.md`` for how the code fits together, ``docs/adr/`` for the
decision record, and ``CONTRIBUTING.md`` for development workflow.

.. toctree::
   :maxdepth: 2
   :caption: Tutorial

   tutorials/getting-started

.. _how-to-guides:

.. toctree::
   :maxdepth: 2
   :caption: How-to guides

   how-to/assign-demand
   how-to/converge-an-assignment
   how-to/constrain-by-capacity
   how-to/evaluate-disruptions
   how-to/estimate-od-demand
   how-to/run-from-a-config

.. toctree::
   :maxdepth: 2
   :caption: Reference

   reference/api
   reference/configuration
   reference/datasets
   reference/versioning

.. _explanation:

.. toctree::
   :maxdepth: 2
   :caption: Explanation

   explanation/data-models
   explanation/equilibrium
   explanation/criticality-and-losses
   explanation/radiation-model

.. toctree::
   :maxdepth: 2
   :caption: Development

   development


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
