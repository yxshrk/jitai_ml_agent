# Approach — what we did and why, with the evidence

Companion to `kb/ARCHITECTURE.md` (what is built). This document explains the reasoning: each principle, the
published result it rests on (papers in `kb/literature/`, summarised in `agent-design-notes.md`), and what our own
runs measured. Decisions are recorded individually in `kb/adr/`.

## The problem shapes the agent

The benchmark scores a *converged* run under a strict rule — three consecutive iterations without a > 0.002 gain on
validation ends the run — and the organizers' own baseline has a seed noise of 0.0008. So the scarce resource is not
compute (a training run is 15 s) but **iterations that are not wasted on noise**, and the scored deliverable is as
much the agent's *log of reasoning* (Innovation 20 %, Autonomy 20 %, robustness inside Technical Execution 35 %)
as the model's number. Every design choice below follows from that reading.

## 1. Knowledge before search

**What we did.** Before any autonomous run: a frozen spec (`kb/spec`), a measured data-facts layer from a dedicated
EDA pass (`kb/data/facts.md`), and a library of method cards (`kb/methods`) — one per technique, each with
preconditions checkable against the facts, an honest expected gain, an implementation recipe against the baseline
script, and what it composes with.

**Why.** The Meta/UCL study of research agents (AIRA, arXiv 2507.02554) found that with weak proposal operators,
better search (MCTS, evolutionary) gains nothing, while better operators lift MLE-bench-lite medals from 39.6 % to
47.7 %: *proposal quality, not search cleverness, is the bottleneck.* MLE-STAR (arXiv 2506.15692) attributes much
of its +18-point gain over AIDE to retrieving task-appropriate methods before writing code; R&D-Agent (arXiv
2505.14738) loses 9 % relative when its memory context is removed.

**Measured here.** The Selector's first three picks in both live runs were the organizers' own top leads (a
ranking-aligned loss, recency weighting, the duration-unknown flag), and the ranking-aligned loss was the accepted
gain in both runs (+0.0022 / +0.0016 single seed; +0.0017 / +0.0016 over three seeds).

## 2. A graph of whole scripts, orchestrated by code

**What we did.** The loop is deterministic Python. A node is a whole runnable script plus its score; each
generation branches k = 3 nodes from the champion; LLM roles (diagnose, select, implement, critique, fix,
consolidate) act once each inside a node's creation and never hand work to each other.

**Why.** Every strong published system has this shape — AIDE's solution tree with draft / debug / improve
(arXiv 2502.13138), AI-Scientist-v2's staged best-first tree (arXiv 2504.08066), ML-Master's MCTS
(arXiv 2506.16499), R&D-Agent's exploration graph — and none is a pipeline of stage-specialised agents handing off
to one another. R&D-Agent's ablation is the direct measurement: replacing structured exploration with chain-shaped
exploration drops the medal rate from 35.1 % to 25.3 %, the largest loss of any component.

**Measured here.** Zero manual interventions in both runs; a generation that errors counts as non-improving and the
run continues; the journal is produced as a by-product of the loop rather than assembled afterwards.

## 3. One component per node, three branches per generation, merges of orthogonal wins

**What we did.** Each candidate names exactly one `target_component` (loss, features, data weighting, …); the
three branches of a generation must differ in component; a Consolidator may reserve a slot to merge two winners
whose components differ, retest a parked idea on a changed stack, or explore away from a flat champion.

**Why.** MLE-STAR's central result is that refining one pipeline component at a time — chosen by ablation — beats
whole-script rewriting (any-medal 43.9 % vs AIDE's 25.8 %). Parallel branches with fusion are in ML-Master (three
workers), R&D-Agent (parallel exploration and result fusion) and AIRA (a crossover operator). Diversity across the
branches is the cheap way to survive the ε/N rule: three shots per generation instead of one.

**Measured here.** After the "edit, don't rewrite" instruction and a 200-line diff guard, node diffs fell from
433 / 452 / 275 lines (live_01) to 34 / 13 / 4 / 15 / 7 / 4 (live_02) for the same kinds of change, and the cost of a
generation from $1.02 to $0.71.

## 4. Code judges; the model proposes

**What we did.** A referee validates every prediction file, scores it with the organizers' untouched `evaluate.py`,
computes deltas, runs confirmation seeds, applies the acceptance rule, picks the champion, and applies the official
convergence rule. No LLM ever sees a number and decides whether it is good.

**Why.** Every system above uses an external fitness function; AIDE calls it the stateless objective `h(s)`. An LLM
asked "is +0.0018 an improvement?" will sometimes say yes, and over fifty iterations that drifts the run. Our
referee is also the mechanism that makes the journal's numbers trustworthy by construction.

## 5. Statistical discipline against the winner's curse

**What we did.** Any candidate with a positive single-seed delta is re-run with two more seeds (the champion's are
cached); it is accepted only if the seed-mean gain is ≥ 0.001 and ≥ 2.5 standard errors. The final submission is
chosen among the top-3 by seed mean, never by a single seed. The official convergence rule is applied literally on
single-seed scores, so the run stops when the organizers say it should.

**Why.** AIRA's most important finding is a systematic gap between validation and test: selecting by test instead of
validation would add 9–13 points absolute, robust re-ranking of the top-k recovers ~10 %, and longer search makes the
gap worse. The organizers' ε = 0.002 is calibrated for one seed; picking the best of three single-seed branches is a
maximum of noisy draws and is biased upward.

**Measured here.** The BPR node scored +0.0022 on seed 0 and +0.0017 over three seeds. In live_02 generation 2, all
three branches looked positive on one seed (+0.0005, +0.0006, +0.0005) and were +0.0000, +0.0001, +0.0002 over three
— three false champions avoided in one generation. Node-level seed spread was 0.0004–0.0008.

## 6. Leakage prevented by construction

**What we did.** Agent scripts run in a workspace holding train (all columns), validation (features + label) and
the two side tables — no test rows, no whole-month statistic file. Test features live in a private directory read
once, by the submission step. A static scan rejects scripts naming the raw data, the test file or `../`, comments
included; a Critic checks every script for outcome columns used as features and for non-causal history features.

**Why.** MLE-STAR's ablation shows what leakage does: without its checker, validation accuracy rose from 0.8188 to
0.8677 while test collapsed from 0.8033 to 0.7343. The benchmark's "hidden" test labels are in the public download,
so the firewall has to be structural.

**Measured here.** The firewall caught a raw-directory reference in a first draft (live_01 node_002) and sent it
back; the Critic has vetoed nothing else — leakage attempts have not appeared, but the check costs one call.

## 7. Bounded recovery

**What we did.** Implementer ≤ 2 revise rounds; Fixer once per stage (smoke, full); a node failing twice is
abandoned; a crashed generation is journaled and counted as non-improving; the loop never dies.

**Why.** AI-Scientist-v2 caps debugging at depth 3; ML-Master marks a node terminal after 3 failed improvements;
AIRA bounds its debug loop. The judges score how failures are handled, not how many occur.

**Measured here.** In the offline end-to-end test a deliberately broken script failed its smoke test, was repaired,
re-run and scored. Live runs so far: no runtime failures; one static-firewall bounce; one oversized-diff bounce.

## 8. Context and cost discipline

**What we did.** A single stable prompt prefix (spec, contract, facts, cards ≈ 12 K tokens) byte-identical for
every role, served from the provider's cache; the dynamic part is one journal line per node plus the parent script;
smoke tests (`SMOKE_EPOCHS=1`) before every full run.

**Why.** AIDE's summarisation operator and AIRA's scoped memory (siblings for improve, ancestors for debug) both
exist to stop the context from growing with the run; R&D-Agent debugs on data subsets first. Feasibility is scored
on tokens and wall-clock.

**Measured here.** live_01: 236 K of 370 K input tokens served from cache; a generation costs $0.71–1.02 and 4–6
minutes, of which training is about one minute.

## 9. Memory across runs

**What we did.** After every run, `distill.py` folds each node back into its card: a measured line (stack, seeds,
verdict, diff size), the evidence reference, and a status of `alive` or `dead_under {stack, delta}`.

**Why.** R&D-Agent's memory ablation (−9 %); and the observation that "dead" is contextual — a feature flat on a
pointwise FM may matter under a pairwise loss (ADR-0004).

**Measured here.** Recency weighting on the BPR stack was measured neutral in live_01 and again in live_02 because
runs did not yet share memory; after distillation the card carries both measurements and the retest is closed.

## 10. Calibration as an artifact

Every candidate carries the Selector's expected delta and its basis; the journal records the realised delta; the
Selector's prompt now contains the measured calibration (predicted 0.006 / 0.004 / 0.003 → realised +0.0022 /
+0.0005 / −0.0003). The expected-vs-realised curve over a run is evidence, for the Innovation criterion, that the
agent reasons about effect sizes rather than guessing.

## What this is not

Not a pipeline of stage agents; not an LLM judging its own results; not a 50-iteration grind — AIRA shows that
searching longer widens the validation–test gap, and the convergence rule is a feature we obey rather than a limit we
work around.

## Evidence so far

| run | nodes | accepted | champion Δ (single / 3-seed) | false positives caught | cost | wall-clock |
|---|---|---|---|---|---|---|
| live_01 | 7 | BPR loss | +0.0022 / +0.0017 | (rule not yet in place) | $2.70 | 16 min |
| live_02 | 16 (converged after 5 generations) | BPR loss; designated = 5-seed ensemble of it (3-seed mean 0.6039, +0.0025 vs baseline mean) | +0.0016 / +0.0016 (t = 8.2) | 9 single-seed "wins" of +0.0002…+0.0006 rejected on seeds | $4.25 | 30 min |

Details: `runs/<run_id>/journal.md`, `LOG.md`.
