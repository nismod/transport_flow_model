# Issue bodies

One markdown file per issue, in the same format as the WS0-WS6 workplan
bundle: the file holds the body only, while title, labels and milestone are
listed below and passed by whatever creates the issue (`gh issue create
--body-file issues/<file>`).

The **M0** series is the handover that unblocks assignment-method work: docs,
recorded decisions, and the two latent problems an incoming developer would hit
first. M0-01 to M0-05 are done; the files are kept as the record of what was
asked for and what was delivered.

| ID | File | Title | Labels | Milestone | Status |
| --- | --- | --- | --- | --- | --- |
| m0-01 | [m0-01-architecture-overview.md](m0-01-architecture-overview.md) | Write ARCHITECTURE.md | docs | M1 Foundations & measurement | Done |
| m0-02 | [m0-02-contributing-and-readme.md](m0-02-contributing-and-readme.md) | Add CONTRIBUTING.md and refocus the README | docs | M1 Foundations & measurement | Done |
| m0-03 | [m0-03-architecture-decision-records.md](m0-03-architecture-decision-records.md) | Record architecture decisions as ADRs | docs | M1 Foundations & measurement | Done |
| m0-04 | [m0-04-coerce-integral-dtype-instability.md](m0-04-coerce-integral-dtype-instability.md) | Link flow dtype depends on iteration count | python | M1 Foundations & measurement | Done |
| m0-05 | [m0-05-measure-gap-evaluation-cost.md](m0-05-measure-gap-evaluation-cost.md) | Measure the cost of evaluating the relative gap | performance, python | M1 Foundations & measurement | Done |
| m0-11 | [m0-11-fuse-gap-evaluation-into-aon.md](m0-11-fuse-gap-evaluation-into-aon.md) | Stop rebuilding the graph per origin when evaluating the gap | performance, rust, python | M3 Equilibrium & quasi-dynamic assignment | Open |

m0-06 to m0-10 are reserved for the assignment methods themselves and are
tracked by the WS2 workplan issues (`ws2-01` … `ws2-07`), not here.
