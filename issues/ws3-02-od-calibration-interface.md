## Context
OD matrices should be adjustable to observed link counts (traffic counters). Classic
methods (e.g. Spiess' gradient approach) are a special case of the differentiable bilevel
formulation planned in ws5-02. This issue designs the interface and ships a simple
baseline so WS5 slots in later.

## Task
- Define `calibrate_od(od0, counts, network, method=...)`: counts = (link_id, observed
  flow, weight); returns adjusted OD + fit report (GEH statistic, R^2, per-link residuals).
- Baseline implementation: Spiess (1990) gradient method with assignment proportions from
  the current equilibrium (paths/proportions available from ws1-06/ws2-03).
- Regularization toward the prior OD (relative-entropy penalty) to avoid overfitting
  sparse counts; document identifiability limits.

## Acceptance criteria
- On SiouxFalls with synthetic counts from a known OD perturbation, recovers the
  perturbation direction; GEH < 5 on 85% of counted links (standard practice threshold).

## References
- Spiess (1990) "A gradient approach for the O-D matrix adjustment problem", CRT report.
- Cascetta & Nguyen (1988) unified framework for OD estimation, Transportation Research B.
- UK DfT TAG unit M3.1 (GEH acceptance criteria context).
