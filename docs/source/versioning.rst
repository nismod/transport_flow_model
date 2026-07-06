Versioning and deprecation
==========================

Semantic versioning
-------------------

``transport-flow-model`` follows `Semantic Versioning 2.0.0
<https://semver.org/spec/v2.0.0.html>`_. The **public API** is the set of
names exported at the top of the ``transport_flow_model`` package (its
``__all__``), together with the documented behaviour of the command-line
scripts and the JSON config schema (:class:`~transport_flow_model.config.RunConfig`).
Anything else — underscore-prefixed names, module internals, the Rust
extension module ``transport_flow_model._core`` — may change without
notice.

While the major version is 0:

- **Minor** releases (``0.x.0``) may contain breaking changes to the
  public API. Every breaking change is listed in the `changelog
  <https://github.com/nismod/transport-flow-model/blob/main/CHANGELOG.md>`_.
- **Patch** releases (``0.x.y``) contain only fixes and non-breaking
  additions.

From ``1.0.0`` onwards, breaking changes will only appear in major
releases.

Result stability
----------------

Numerical results are part of the interface for benchmarking and
criticality rankings: a patch release must reproduce assignment results
bit-for-bit for the same inputs, method and options. Changes that alter
results (new tie-breaking, reordered iteration, algorithm fixes) are
minor releases and are called out in the changelog. Every
:class:`~transport_flow_model.assignment.AssignmentResult` carries
:class:`~transport_flow_model.assignment.Provenance` (method, options,
iterations, relative gap, wall time, seed, package and core versions) so
results can be traced to the code that produced them.

Deprecation policy
------------------

- A feature is deprecated in a minor release before it is removed:
  removal happens at the earliest **one minor release after** the
  deprecation is announced in the changelog.
- Deprecated Python callables emit :class:`DeprecationWarning` where
  practical, and their docstrings say what to use instead and the
  release in which they will be removed.
- Currently deprecated:

  - :func:`transport_flow_model.config.load_config` — use
    :meth:`~transport_flow_model.config.RunConfig.from_json`; removal in
    0.4.0.
  - :mod:`transport_flow_model.model` as a public interface — use
    :class:`~transport_flow_model.Network`,
    :class:`~transport_flow_model.Demand`,
    :func:`~transport_flow_model.assign` and
    :func:`~transport_flow_model.disrupt`. Removal will be announced in
    the changelog at least one minor release in advance.
