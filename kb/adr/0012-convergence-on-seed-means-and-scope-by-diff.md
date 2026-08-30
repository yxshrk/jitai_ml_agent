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
1. **Convergence counts generations without a seed-confirmed champion change** (`referee.Convergence`, used by the
   loop): a generation whose accepted best node replaces the champion resets the streak; any other generation adds
   one; N = 3 = converged. *Revised the same day by Yash* from a first version that let the reference move only on a
   > ε rise of the champion's seed-mean: replayed on live_04, every faithful reading of ε stops the run at
   generation 3, one generation before node_015 (+0.0017 confirmed, t 7.65, the best model of any run) existed. The
   seed test (mean gain ≥ 0.0005 at ≥ 2.5 standard errors, i.e. several times the seed SD) is a stricter noise
   filter than a fixed 0.002 on one seed, which is what ε exists for. ε keeps two jobs: the single-seed screen
   recorded per node (`single_seed_accept`), and `referee.OfficialRule` — the organizers' rule read literally on
   single-seed bests, tracked every generation and reported in `summary.official_rule` (where it would have
   stopped) so the judges see both. Guard against false positives (Yash, after the risk was quantified): the streak
   resets only when the champion's fresh-seed mean has risen by **≥ 0.001 since the last reset** (`RESET_MIN_GAIN`,
   cumulative like early stopping's `min_delta` against the best seen: ε/2 on a statistic with about a third of the
   single-seed noise, a 4σ event under the null). One false champion (observed gain ≈ +0.0005–0.0007 under the null)
   cannot buy three generations; a staircase of small real gains adds up and counts; smaller confirmed gains still
   move the champion. `--convergence official` switches the stopping rule to the literal one, and every summary
   reports `official_rule_submission` — the node the literal rule would have submitted — next to ours. The caps
   (50 nodes, 6 h, dollars) bound the extra generations this allows.
8. **The acceptance statistic is a pooled-variance z-test on fresh seeds.** The 3-vs-3 t-test at t ≥ 2.5 had
   2–4 degrees of freedom and passed 3–6 % of null candidates (the 0.0005 floor is only ≈ 2 SE and did not bind;
   live_04's node_017, +0.0005 at t 2.63, is the shape of a false positive — both merges built on it went
   negative). Seed-to-seed noise is a property of the data + model family (every node measured: σ ≈ 0.0002–0.0005),
   so σ is **pooled over every fresh-seed run of the run** (Bessel, blended with the prior 0.0003 at 4 df) and the
   test is z = Δ̄ / (σ·√(1/n_c + 1/n_ch)) ≥ 3.0 with Δ̄ ≥ 0.0005. **Seed 0 is excluded from both means**: it is the
   selected screen (a maximum of k draws) and biases the candidate by ≈ 0.25 SE; the decision uses three fresh seeds
   (1–3), with two more (4–5) when 2.0 ≤ z < 3.0. Per-test false-positive rate ≈ 0.13 %, ≈ 0.4 % per generation.
   Replayed with σ = 0.0003: node_015 z 6.8, node_001 5.0, node_002 4.7, node_003 4.3, node_004 3.8, live_02's
   ensemble 3.7 — all still accepted; node_017 (z 2.0) and node_012 (z 2.3) go to the adaptive seeds instead.
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
this rule runs on through generation 4's ensemble (streak reset by a confirmed change) and stops after three flat
generations, while the literal rule is reported to have converged at generation 3. Token accounting (ADR-0013) now
reports cached and uncached input separately, and the run-journal block goes only to the roles that plan. Attribution in the
journal and in the cards is trustworthy again: a wildcard's card describes what its diff actually did. Unit tests
cover `Convergence`, `pick_champion`, `confirm_stats`, the seed-cache migration and `summarize`.

## Amendment (2026-08-31, after live_07): who may be designated for submission

live_07's designation picked node_019 — fresh-seed mean 0.6049 against the champion's 0.6044, a gap of more than one
standard error — although the run had rejected it as champion (+0.0005 at z 1.94, just under the borderline band that
would have triggered two more seeds). The rule above worked as written and produced a submission the run itself had
rejected: a story the judges would rightly ask about.

Decision: designation has two modes, chosen before a run (`--designation`, `config.DESIGNATION_DEFAULT = 'strict'`):
- **strict** (default): only accepted nodes may be designated — the champion lineage, re-ranked by fresh-seed mean
  among themselves; the best unaccepted candidate is reported as `best_unaccepted` and journaled, never submitted.
  Reasons: re-testing the best-looking of several rejected nodes at a lower bar after seeing their means re-enters the
  winner's curse this ADR exists to stop (≈ 5–7 % false designation for a top-3 at z ≥ 2); the stake is one seed SD
  on validation; "the run never submits a node it rejected" needs no follow-up question.
- **adaptive**: an unaccepted node that leads on fresh-seed mean receives `MAX_CONFIRM_SEEDS` fresh seeds and is
  eligible only if its gain over the champion's fresh-seed mean is ≥ `MIN_EFFECT` at z ≥ `Z_BORDER` — the search's
  own borderline test, run once, with the outcome journaled either way. Defensible when the leader is a strict
  superset of the champion (live_07's node_019: the champion's ordering with the session model substituted for two
  cohorts) and the user accepts the small selection bias for a small expected gain.
The one-SE tie-break toward an accepted node applies in both modes. `tests/test_designation.py` covers both.
