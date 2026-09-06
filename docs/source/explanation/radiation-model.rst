The Radiation Model
===================

:class:`~transport_flow_model.RadiationModel` generates synthetic
origin-destination flow probabilities from location counts and network
distance. It exists for the common case where you have population or
employment at locations but no OD survey data at all.

To use it, see :doc:`../how-to/estimate-od-demand`.

What it models
--------------

The radiation model predicts flows from local opportunities and distance. It
treats travel as opportunity-seeking: a traveller takes the nearest
destination whose opportunities are good enough, so a destination competes
not only on its own attractiveness but against everything that lies between
it and the origin.

That intervening-opportunity term is what distinguishes it from a gravity
model. A gravity model decays flow with distance directly, and needs
parameters fitted to observed data to say how fast. The radiation model
derives the decay from the *distribution of opportunities in between*, which
is why it needs no calibration.

Mathematical foundation
-----------------------

.. math::

    P_{ij} = \frac{1}{1 - \frac{m_i}{M}} \cdot
             \frac{m_i \cdot m_j}{(m_i + s_{ij})(m_i + m_j + s_{ij})}

- :math:`P_{ij}` — probability of flow from location :math:`i` to :math:`j`
- :math:`m_i`, :math:`m_j` — relevance (population, employment) at origin
  and destination
- :math:`s_{ij}` — total relevance of intervening opportunities between them
- :math:`M` — total relevance across all locations

Distance enters only through :math:`s_{ij}`: what counts as "in between" is
decided by network distance and the ``distance_threshold`` you set. So the
threshold is the one modelling choice the method leaves you, and it is a
statement about trip length, not a fitted parameter.

Being parameter-free
--------------------

There is no calibration step and no random component. The same inputs always
produce the same probabilities, which makes results reproducible and makes
the model portable to regions with no survey data to fit against — the
property that motivates using it here, since infrastructure risk analysis
often targets exactly those regions.

The cost is that it cannot be tuned to match a known matrix. Where OD survey
data exists, a calibrated model will fit it better; the radiation model is
what you reach for when there is nothing to calibrate against.

What it does not know
---------------------

The model proposes demand between any two zones it is given. It has no
notion of whether the network can carry that trip, so an estimated matrix
can contain pairs with no path at all — they surface as ``unassigned``
demand at assignment time rather than as an error here.

It also has no notion of trip purpose, time of day, or mode. The output is a
single undifferentiated matrix of relative propensities, which you scale
into trips yourself.

Further reading
---------------

- Simini, F., González, M. C., Maritan, A., & Barabási, A. L. (2012). A
  universal model for mobility and migration patterns. *Nature*, 484(7392),
  96–100.
- :doc:`data-models` — the ``Network`` and ``Demand`` objects involved
- :doc:`../how-to/estimate-od-demand` — generating and assigning a matrix
