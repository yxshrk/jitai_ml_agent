# ADR-0009 — Explore k branches per generation in parallel, then merge winners
Status: Proposed (2026-08-30) — Yash's idea, adapted to the scoring rules

## Context
Yash proposed exploring several changes at once as parallel branches (game-tree style), then consolidating:
compare, combine what composes, drop what doesn't. The literature does this (ML-Master: 3 parallel MCTS workers;
R&D-Agent: parallel exploration with result fusion; MLE-STAR: parallel solutions merged by an ensembling step;
AIRA: crossover operator). Our compute makes it nearly free: a full FM run is ~15 s and the machine has 10 cores.
Two rules constrain it: the 50-iteration cap (ADR-0006 counts every train-and-evaluate cycle as an iteration),
and the convergence rule, which is defined on validation improvement per iteration.

## Decision
1. The unit of the loop becomes a **generation**: the Selector proposes k candidates (k = 3 by default), forced
   to differ in method family / `target_component` (portfolio diversity); k Implementer calls produce k scripts;
   the harness smoke-tests and runs them **in parallel** (subprocess pool); the referee scores all k.
2. **Counting:** every branch is a node and counts toward the 50-node cap (conservative). The convergence rule is
   applied per generation: a generation "improves" if its best node beats the best-so-far by more than ε;
   three consecutive non-improving generations = converged. Both counts (nodes, generations) are journaled so
   either reading of "iteration" can be reported; the organizers will be asked which they mean.
3. **Consolidation = a merge operator.** After each generation, a Consolidator role reads all k results (deltas,
   learning curves, what each changed) and may propose, for the next generation, a **merge** node that combines the
   winners whose mechanisms are orthogonal (e.g. a loss change + a feature change), alongside fresh candidates.
   A merge is a normal node: it is scored like any other and must beat the champion by ≥ ε to be accepted.
4. **Guard rails:** breadth multiplies token cost by ~k (k proposer/implementer calls per generation) — Feasibility
   is scored in coarse tiers, so k stays small; breadth also multiplies validation-overfitting risk (AIRA), so the
   anti-overfit protocol of ADR-0008 (ε acceptance, multi-seed confirmation, independent final selection) is
   mandatory, not optional; parallel runs pin BLAS to one thread each so k runs do not contend.

## Consequences
~3× more hypotheses tested per generation for ~1× wall-clock; the ε/N rule is easier to survive (three shots per
generation instead of one); combination of orthogonal wins becomes an explicit, journaled step. Costs: k× tokens,
and a rules question to settle with the organizers. If they count nodes, k = 3 gives ~16 generations — still more
than the convergence rule normally allows.
