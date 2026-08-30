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

## Update 2026-08-30 — both readings supported; Yash's reading is "one turn of the loop"
Evidence in `docs` cuts both ways: Figure 1 and "reflect + revise ... loops back into the next iteration" describe an
iteration as one turn of the loop (a *generation* here), while "100 iterations of the official baseline take about
28 min on a single CPU core" prices an iteration as one training run (a *node*). The harness therefore takes
`--iteration-unit node|generation` (default `node`, the conservative reading) and journals both counts. In practice
the difference rarely binds: with k = 3, counting nodes allows ~16 generations, and the convergence rule normally
stops a run within 5–8 generations. The question goes to the organizers; the flag is flipped if they say "turn".
