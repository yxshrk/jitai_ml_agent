# ADR-0006 — What counts as an iteration
Status: Proposed (2026-08-30)

## Context
`docs` caps a run at 50 iterations and defines convergence on the validation score, but never defines "iteration".
A pre-screening loop on a holdout carved from train (e.g. the last two train days) would let the agent test cheaply
before spending a validation evaluation — but treating those as "free" is a rules risk.
## Decision (provisional)
Every train-and-evaluate cycle is an iteration and is journaled. The inner/outer split is parked until the
organizers answer whether only validation evaluations count. The harness logs both counts so either reading can be
reported.
## Consequences
Conservative and defensible; if the organizers allow pre-screening, the design switch is a flag.
