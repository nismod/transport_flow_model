Converge an Assignment
======================

Run assignment to user equilibrium, decide when to stop, and choose the
volume-delay curve. For why any of this is necessary, see
:doc:`../explanation/equilibrium`.

Run MSA to a target gap
-----------------------

Ask for a gap and give a budget large enough to reach it.

>>> import warnings
>>> from transport_flow_model import Network, Demand, assign, datasets, relative_gap
>>> instance = datasets.load_tntp("siouxfalls")
>>> network = Network.from_tntp(instance)
>>> demand = Demand.from_tntp(instance)
>>> result = assign(network, demand, "msa", target_gap=1e-3, max_iterations=2000)
>>> result.provenance.iterations
743
>>> f"{result.provenance.relative_gap:.3e}"
'9.963e-04'

The reported gap always describes the flows returned beside it, whichever
way the run stopped — target reached, time limit, or iteration budget.

Handle an unconverged run
-------------------------

If the budget runs out first, the run warns rather than returning an
unconverged answer in silence. ``max_iterations`` defaults to 50, which is a
screening budget:

>>> with warnings.catch_warnings(record=True) as caught:
...     warnings.simplefilter("always")
...     screening = assign(network, demand, "msa", max_iterations=50)
>>> print(str(caught[0].message))
msa stopped after 50 passes at relative gap 1.573e-02, above the target 1.000e-04; max_iterations reached. Raise max_iterations or relax target_gap; provenance.relative_gap and gap_history record what was reached.

Either raise ``max_iterations``, relax ``target_gap``, or — if a screening
run is what you wanted — silence the warning deliberately:

>>> with warnings.catch_warnings():
...     warnings.simplefilter("ignore", category=UserWarning)
...     screening = assign(network, demand, "msa", max_iterations=50)
>>> f"{screening.provenance.relative_gap:.3e}"
'1.573e-02'

To fail loudly instead, promote it to an error with
``warnings.simplefilter("error", ConvergenceWarning)``.

Choose a budget from the gap history
------------------------------------

``gap_history`` records the trajectory, which is how you pick a budget
without guessing:

>>> [f"{gap:.3e}" for gap in screening.gap_history[:3]]
['8.821e+00', '1.474e+00', '9.609e-01']

Cap the wall clock instead
--------------------------

``time_limit_s`` stops the run on elapsed time, checked after each gap
evaluation. Useful for a scenario sweep where a fixed pass count would make
some runs far slower than others.

>>> with warnings.catch_warnings():
...     warnings.simplefilter("ignore", category=UserWarning)
...     timed = assign(network, demand, "msa", time_limit_s=0.05, max_iterations=5000)
>>> timed.provenance.iterations > 1
True

Use a different volume-delay curve
----------------------------------

Pass any :class:`~transport_flow_model.costs.CostFunction` as
``cost_function``, or name one from a config file (see
:doc:`run-from-a-config`).

>>> from transport_flow_model import Conical
>>> conical = Conical.from_network(network, alpha=4.0)
>>> with warnings.catch_warnings():
...     warnings.simplefilter("ignore", category=UserWarning)
...     conical_result = assign(
...         network, demand, "msa", cost_function=conical, max_iterations=200
...     )
>>> f"{conical_result.provenance.relative_gap:.3e}"
'3.156e-03'

**Score the run with the same curve you assigned it with.**
:func:`~transport_flow_model.relative_gap` defaults to BPR, so passing the
curve is not optional if you assigned with something else:

>>> f"{relative_gap(network, demand, conical_result, cost_function=conical):.3e}"
'3.156e-03'

Omitting it silently measures a different model — the same flows score an
order of magnitude worse against the default BPR curve, and that number
describes a run nobody performed.

>>> f"{relative_gap(network, demand, conical_result):.3e}"
'7.773e-02'
