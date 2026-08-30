# ADR-0004 — A "dead" result is contextual; retries are allowed with a cited reason
Status: Accepted (2026-08-30)

## Context
Ideas measured flat under one stack can work under another; single-seed results inside the noise band prove
nothing; implementations can be wrong.
## Decision
Cards record `dead_under: {model, loss, features, seeds, date}` rather than a bare `dead`. The selector may reopen a
card when it cites one of: (a) the champion stack changed in a way the card's mechanism interacts with; (b) the
evidence was weak (1 seed or |delta| < 0.002); (c) the earlier score sits far below what the literature reports,
suggesting a bug. The organizers' own null results (more fields, bigger k) were measured on FM + logloss, 3 seeds:
solid for that stack only.
## Consequences
Each retry can cost one of the three convergence "lives", so the reason must be in the journal.
