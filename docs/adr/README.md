# Architecture decision records

Short records of decisions that shape the codebase, kept next to the code so
they can be reviewed in the same pull request as the change they describe.

An ADR is worth writing when a decision constrains code that other people will
write — an interface contract, a data representation, a boundary between
layers. It is not a design document or a plan: aspirational work belongs in
[`DEVELOPMENT.md`](../../DEVELOPMENT.md), and how the code fits together today
belongs in [`ARCHITECTURE.md`](../../ARCHITECTURE.md).

## Records

| ADR | Title | Status |
| --- | --- | --- |
| [0001](0001-assignment-methods-are-registered-backends.md) | Assignment methods are registered backends | Accepted |
| [0002](0002-arrow-tables-are-the-internal-interchange.md) | Arrow tables are the internal interchange | Accepted |

## Adding one

1. Copy [`0000-template.md`](0000-template.md) to `NNNN-short-title.md`, taking
   the next free number.
2. Fill it in and add a row to the table above.
3. Open it as part of the pull request that makes the change.

A record describes the decision as it stood when it was accepted. Do not edit
an accepted record to reflect later changes: supersede it with a new one, and
mark the old one `Superseded by ADR-NNNN`.

These files are Markdown and live outside `docs/source/`, so they are not part
of the Sphinx build.
