Why Equilibrium, and What the Gap Measures
==========================================

Assigning demand to least-cost paths puts every traveller on the route that
is cheapest *at free-flow cost*. Nothing in that pass notices that the
traffic it assigns makes those routes slower. On a congested network the
result is not a description of what would happen — it is the first guess of
an iterative method that has not been run.

**User equilibrium** is the state Wardrop described: no traveller can lower
their own cost by switching route. Reaching it means solving for the flow
pattern and the costs together, since each determines the other.

To *do* this, see :doc:`../how-to/converge-an-assignment`. This page is about
what the numbers mean.

The relative gap
----------------

The gap measures how much higher total travel time is than it would be if
everyone took the cheapest route at the current costs. It is zero at
equilibrium, and on siouxfalls a single all-or-nothing pass scores **8.821**
— total travel time nearly ten times the shortest-path total. Sequential
allocation is a starting point and a capacity screening tool, not an
equilibrium.

Boyce, Ralevic-Dekic & Bar-Gera (2004) argue that gaps of **1e-4 or better**
are needed before flow differences between scenarios can be trusted. That
matters directly for criticality work: a ranking is built from differences
between runs, so noise from an unconverged solution propagates straight into
it. See :doc:`criticality-and-losses`.

Why MSA is the baseline, not the recommendation
-----------------------------------------------

The method of successive averages mixes each new all-or-nothing load into
the running solution with weight ``1/k``. The averaging is what makes it
converge — and it converges at ``O(1/k)``, so each further decimal place of
gap costs roughly ten times the passes. On siouxfalls:

===================== ====================
passes                relative gap
===================== ====================
1 (all-or-nothing)    8.821e+00
50 (the default)      1.573e-02
743                   9.963e-04
===================== ====================

That is why MSA is simple, reliable and slow, and why it is the baseline
other equilibrium methods are measured *against* rather than the method of
choice. It is also why the default budget of 50 passes warns rather than
staying quiet: it lands two orders above the default target, and nothing
else in the result would tell you.

The gap is free — for MSA
-------------------------

Each MSA iteration performs an all-or-nothing load anyway, and that load
puts every OD pair on its cheapest path — so it already contains the
shortest-path term the gap needs. Checking convergence every iteration costs
nothing extra, which is why MSA reports a gap history at no charge.

:func:`~transport_flow_model.relative_gap` exists for the other case:
scoring a solution you did not produce, or one from a method that reports no
gap of its own. That costs roughly half an all-or-nothing pass.

Volume-delay functions
----------------------

A volume-delay function turns link flow into link travel time. Three ship
with the package, all satisfying the
:class:`~transport_flow_model.costs.CostFunction` protocol:

- :class:`~transport_flow_model.BPR` — the Bureau of Public Roads power
  curve, the TNTP convention and the default;
- :class:`~transport_flow_model.Conical` — Spiess (1990), which stays finite
  and increasing at every flow where BPR's curve goes vertical at capacity,
  which suits Newton-style methods;
- :class:`~transport_flow_model.SpeedFlow` — piecewise-linear speed against
  flow per lane, the shape a DfT TAG speed-flow table takes.

The choice is a modelling choice, not a performance one: one ``travel_time``
evaluation is between 0.06% and 2% of an MSA pass, so the curve is not a hot
loop.

**A gap is only meaningful against the curve that produced the flows.** The
same converged siouxfalls flows score ``9.963e-04`` measured with BPR — the
curve they were assigned with — and ``6.262e-02`` measured with a conical
curve. Neither number is wrong; they are gaps for two different models, and
only one of them describes that run. This is why
:func:`~transport_flow_model.relative_gap` and
:func:`~transport_flow_model.beckmann_objective` take a ``cost_function``,
and why leaving it at the default while assigning with something else
silently answers a different question.

Checking a solution
-------------------

User equilibrium minimizes the Beckmann objective, which gives an
independent check: a converged run's objective should sit just above the
published best-known value. The 743-pass run above lands **0.167%** over
``datasets.BEST_KNOWN["siouxfalls"]``, which is what a gap of 1e-3 buys.

Why an equilibrium has no paths
-------------------------------

``paths`` is always ``None`` for an equilibrium result. An equilibrium is an
average of many all-or-nothing solutions and has no single path per OD pair,
so ``include_paths=True`` raises rather than returning the last pass's
paths, which would describe a different and much worse solution than the
flows beside them.

The practical consequence is that an MSA result cannot yet serve as a
:func:`~transport_flow_model.disrupt` baseline, since rerouting needs to
know which paths used a failed link.
