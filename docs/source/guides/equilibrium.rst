Equilibrium Assignment
======================

:doc:`least-cost-allocation` puts every traveller on the route that is
cheapest *at free-flow cost*. Nothing in that pass notices that the traffic
it assigns makes those routes slower. On a congested network the result is
not a description of what would happen — it is the first guess of an
iterative method that has not been run.

**User equilibrium** is the state Wardrop described: no traveller can lower
their own cost by switching route. Reaching it means solving for the flow
pattern and the costs together, since each determines the other.

How far off is one pass?
------------------------

The *relative gap* measures it: how much higher total travel time is than it
would be if everyone took the cheapest route at the current costs. Zero at
equilibrium.

>>> from transport_flow_model import Network, Demand, assign, datasets, relative_gap
>>> instance = datasets.load_tntp("siouxfalls")
>>> network = Network.from_tntp(instance)
>>> demand = Demand.from_tntp(instance)
>>> aon = assign(network, demand, "sequential")
>>> f"{relative_gap(network, demand, aon):.3e}"
'8.821e+00'

A gap of 8.8 means total travel time is nearly ten times the shortest-path
total. Sequential allocation is a starting point and a capacity screening
tool, not an equilibrium.

Method of successive averages
-----------------------------

``method="msa"`` averages repeated all-or-nothing loads at congested costs,
mixing each new load into the running solution with weight ``1/k``. The
averaging is what makes it converge.

It is the baseline equilibrium method: simple, reliable, and slow near the
optimum, which is exactly why other methods are measured against it.

>>> import warnings
>>> with warnings.catch_warnings(record=True) as caught:
...     warnings.simplefilter("always")
...     screening = assign(network, demand, "msa", max_iterations=50)
>>> screening.provenance.iterations
50
>>> f"{screening.provenance.relative_gap:.3e}"
'1.573e-02'

Two and a half orders of magnitude better than one pass — and still nowhere
near converged. ``max_iterations`` defaults to 50, which is a screening
budget, not a converged one, so the run says so rather than handing back an
unconverged answer in silence:

>>> print(str(caught[0].message))
msa stopped after 50 passes at relative gap 1.573e-02, above the target 1.000e-04; max_iterations reached. Raise max_iterations or relax target_gap; provenance.relative_gap and gap_history record what was reached.

That is a :class:`~transport_flow_model.ConvergenceWarning`. The result is
still returned and still usable — a partly converged flow pattern is the
right answer to a screening run — but treating it as an equilibrium would
be a mistake, and nothing else in the result would tell you.

``gap_history`` records the whole trajectory, which is what you want when
deciding on a budget:

>>> [f"{gap:.3e}" for gap in screening.gap_history[:3]]
['8.821e+00', '1.474e+00', '9.609e-01']

Converging
----------

Ask for a gap instead of a pass count and MSA runs until it gets there.
Convergence is ``O(1/k)``, so each further decimal place costs roughly ten
times the passes:

>>> result = assign(network, demand, "msa", target_gap=1e-3, max_iterations=2000)
>>> result.provenance.iterations
743
>>> f"{result.provenance.relative_gap:.3e}"
'9.963e-04'

Boyce, Ralevic-Dekic & Bar-Gera (2004) argue that gaps of **1e-4 or better**
are needed before flow differences between scenarios can be trusted — which
matters directly if you are ranking links by criticality, since the ranking
is built from differences between runs.

The reported gap always describes the flows returned beside it, whichever
way the run stopped: reaching the target, running out of time, or exhausting
the iteration budget.

Checking against a published solution
-------------------------------------

User equilibrium minimizes the Beckmann objective, so the objective of a
converged run should sit just above the published best-known value:

>>> from transport_flow_model import beckmann_objective
>>> objective = beckmann_objective(network, result)
>>> published = datasets.BEST_KNOWN["siouxfalls"].objective
>>> f"{100 * (objective / published - 1):.3f}%"
'0.167%'

Choosing a cost function
------------------------

A volume-delay function turns link flow into link travel time. Three ship
with the package, all satisfying the
:class:`~transport_flow_model.costs.CostFunction` protocol:

- :class:`~transport_flow_model.BPR` — the Bureau of Public Roads power
  curve, the TNTP convention and the default;
- :class:`~transport_flow_model.Conical` — Spiess (1990), which stays finite
  and increasing at every flow where BPR's curve goes vertical at capacity;
- :class:`~transport_flow_model.SpeedFlow` — piecewise-linear speed against
  flow per lane, the shape a DfT TAG speed-flow table takes.

Pass one to ``assign`` with ``cost_function=``, or name it from a config
file (see :doc:`configuration`).

>>> from transport_flow_model import Conical
>>> conical = Conical.from_network(network, alpha=4.0)
>>> with warnings.catch_warnings():
...     warnings.simplefilter("ignore")
...     conical_result = assign(
...         network, demand, "msa", cost_function=conical, max_iterations=200
...     )
>>> f"{conical_result.provenance.relative_gap:.3e}"
'3.156e-03'

**Score a run with the curve it was assigned with.** ``relative_gap`` takes
a ``cost_function`` too, and defaults to BPR — so evaluating a run against
the wrong curve silently answers a different question. The BPR-converged
flows above are three orders from equilibrium when measured with a conical
curve instead:

>>> f"{relative_gap(network, demand, result, cost_function=conical):.3e}"
'6.262e-02'

Neither number is wrong; they are gaps for two different models. The one
that describes *this* run is the one whose curve produced it.

Cost, and what it buys
----------------------

The gap is free for MSA. Each iteration performs an all-or-nothing load
anyway, and that load puts every OD pair on its cheapest path — so it
already contains the shortest-path term the gap needs. Checking convergence
every iteration costs nothing extra.

:func:`~transport_flow_model.relative_gap` is for the other case: scoring a
solution you did not produce, or one produced by a method that does not
report a gap of its own. It costs roughly half an all-or-nothing pass.

``paths`` is always ``None`` for an equilibrium result. An equilibrium is an
average of many all-or-nothing solutions and has no single path per OD pair,
so ``include_paths=True`` raises rather than returning the last pass's paths,
which would describe a different and much worse solution than the flows
beside them. This means an MSA result cannot yet be a
:func:`~transport_flow_model.disrupt` baseline.
