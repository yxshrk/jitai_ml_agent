# ADR-0012 — One statistic for acceptance and convergence; the Critic judges scope from the diff
Status: Accepted (2026-08-30), from a code review of live_04 relayed by Yash

## Context
Rules changed at different times had drifted apart, and live_04 showed the cost:
- Acceptance (ADR-0010/0011) moves the champion on the seed-mean (≥ 0.0005, t ≥ 2.5), but convergence tracked the
  best *single-seed* primary of *any* node against ε = 0.002. Generation 1 accepted four nodes and was logged "no
  improvement"; generation 3 rejected node_012 on seeds (t = 2.43) yet its lucky single seed (0.6036) reset the
  streak and became `best`. The counter and the rule contradicted each other in the same function.
- Champion selection took the best single-seed node and required it to be accepted, so an accepted node with a
  lower single seed could never become champion when the top one failed confirmation.
- The champion's confirmation seeds were discarded on every champion change and recomputed the next generation
  (`001_seed1` was written twice, 20:33 and 20:39); `designate_final` kept a third cache.
- `pstdev` on n = 3 underestimates σ by ~18 %.
- The Critic saw the whole candidate script but never the parent or the diff, and the Explorer's prompt hard-coded
  "a within-user pairwise loss is the champion" (true in live_02, false in live_04). Result: the Implementer's first
  two versions of the field-aware wildcard (node_001) were correct edits of the logloss parent; the Critic, told the
  champion was BPR, sent them back twice until BPR was added. node_001's +0.0012 was therefore mostly the BPR gain
  measured as +0.0011 beside it (node_002); the "merge field-aware + BPR" (node_006) and the lr-halving node (node_008)
  produced predictions byte-identical to node_001 (md5 4ff8aa3e…) and were scored as real experiments.
- The prompts still told the roles "a node must beat the champion by ≥ 0.002" (the pre-ADR-0010 rule).

## Decision
1. **Convergence tracks the champion's seed-mean** (`referee.Convergence`, now used by the loop): early-stopping
   semantics — `best` is the reference set at the last rise of more than ε; a generation whose champion mean does
   not exceed `best + ε` adds one to the streak; N = 3 such generations = converged. Small accepted gains accumulate
   toward ε instead of each being "no improvement"; a rejected node cannot move `best`. The literal single-seed best
   is still recorded (`best_single_seed`) for the judges.
2. **Champion = the accepted node with the largest seed-mean gain** (`pick_champion`), not the best single seed.
3. **One seed cache** (`state.seed_cache`, keyed `node:seed`, never cleared) serves confirmation, prefetch and the
   final designation. Sample SD (`statistics.stdev`) with the 0.0002 floor; two-sample test kept — seeds are not
   paired across scripts (a BPR script permutes a different array and draws a different-shaped V).
4. **No-op detector:** the referee hashes `predictions.csv`; a node identical to its parent is journaled as NO-OP,
   rejected without seeds, and shown as such to the Diagnostician and Consolidator.
5. **The Critic receives the unified diff, the parent's docstring and the parent's actual stack** (the accepted
   method chain from node 0) and must judge scope from the diff; the Implementer is told the parent's stack; the
   Explorer is told to be orthogonal to "the champion stack" from the context, never a hard-coded method.
6. **Rule text is generated from `config.py`** (`rules_text()`) so what the roles read cannot drift from the code.
7. Wall clock counts running time across resumes; the Diagnostician's view of the last generation is no longer cut
   at 6,000 characters.

## Consequences
Runs stop when the *confirmed* champion stalls, not when a lucky seed happens or fails to happen; live_04 under
this rule would have converged at generation 3 (champion mean +0.0012 over the baseline mean 0.60144) — and then
resumed on the corrected reading when generation 4's ensemble rose to +0.0030 cumulative. Attribution in the
journal and in the cards is trustworthy again: a wildcard's card describes what its diff actually did. Unit tests
cover `Convergence`, `pick_champion`, `confirm_stats`, the seed-cache migration and `summarize`.
