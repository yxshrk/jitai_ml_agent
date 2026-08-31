# Campaign post-mortem (mechanical aggregation over 108 completed runs)

Generated from every journal via tools/postmortem.py; narrative below cites only
aggregated numbers. In-flight runs at writing (novel_l1, qb_a, qb_b, omega_1,
omega_1k, novel_r1, f1) get addenda on completion.

## Headline numbers
- 108 completed runs; 88 accepted nodes, 199 measured rejections, 96 execution
  failures. Failure causes: LLM reply parse/truncation 65 (dominant), timeouts 12,
  other exec 10, dict-history bug 5 (fixed), Anthropic thinking-starvation 3
  (diagnosed), budget 1.
- THE dominant reliability tax was LLM output parsing (65 lost iterations —
  roughly one per run). Every measured fix (output budgets, selector budget,
  medium effort) reduced but never eliminated it.

## The winning arc, independently replicated
- run_bigclock_07: package sweep (0.60424, cleared eps) -> ensemble-design close
  ACCEPTED (+0.0013) -> 0.60558. DESIGNATED.
- run_bigclock_r4 (recovered from an aborted-kill; finished unwatched): package
  attempt rejected, then a 5-seed ensemble close ACCEPTED (+0.0034 over parent,
  cleared eps) -> 0.60524, #2 all-time — the same arc, independently. Despite
  losing 3 iterations to the (since-fixed) selector-truncation bug.
- All other ensemble closes on strong singles were REJECTED by measurement
  (r2 0.60379, max_r4 0.60295, c1 0.60229, probe_1 0.60447-gated): close payoff
  is member-structure-dependent (see blend theory notes), succeeding ~2/8 times.

## The sub-floor ceiling cluster (the campaign's sharpest statistical finding)
Ten+ runs independently produced candidates ABOVE their champions by +0.0001..
+0.0009 — under our acceptance floor — topping out at 0.60573 (night_e/f),
0.60481-0.60485 (l2, novel_r1), 0.60447-0.6045 (probe_1, f3, c2, chains).
Interpretation: a reachable ~0.6045-0.6057 zone of real-but-individually-
sub-evidentiary effects exists. Validation-tuned selection (used by competitor
entries) banks these on the leaderboard; our paired-confirm discipline refuses
them. We predict this difference largely evaporates on hidden test.

## Replicated mechanisms (measured >= 2 independent runs)
- gauge-fixed user-centered BCE: +0.0013..+0.0026 on three different parents (3x).
- package-sweep openers with basin priors: 0.6034-0.6047 openers, tight cluster (5x).
- seed-ensemble payoff on suitable members: +0.0013..+0.0034 (2x accepted).
- decayed-positive user sampling: +0.0028 opener (qb_b; single so far, in flight).
- 1K regime inversion (logloss >> BPR, no recency): discovery + in-run replication
  + fresh-seed confirmation + independent evaluator (max_1k_c, 0.6524).

## Operational lessons (each cost us measurable score before diagnosis)
1. Prompt-visible budgets: agents budgeted for the 600s default until the true
   timeout was injected (depth jumped ~10x after).
2. Machine contention starves sweeps silently (final_deck: 15-probe sweep on a
   contended laptop -> weak opener). Dose: 1 run/laptop, 2/coral, 3/ruby.
3. Disk-full on helpers caused 4+ silent run deaths before diagnosis; watchdog
   caught each within minutes (robustness rubric evidence).
4. Deep per-call reasoning (high/xhigh effort, Sonnet default thinking) starves
   whole-file code replies across BOTH providers (measured 5 runs + 1 provider).
5. Search depth dose-response: returns flatten past ~50-80 well-ranked probes;
   width without full-fidelity finals actively harms (see EXPERIMENTS_RECHECK).

## What we would do differently (report section)
Metric-aligned losses before capacity; moderate depth from day 1; regime-check
before transferring verdicts between datasets; paired-seed confirmation from the
start; contention/disk monitoring on the fleet from day 1.

## Addendum: omega post-mortem (the "everything run" result)
omega_1 (0.60398, 4 iters): matrix opener cleared eps; then gauge-BCE measured
+0.0002 (vs +0.0026 on novel_r1's pointwise-heavy parent — MECHANISM OVERLAP: a
BPR+strong-reg package has little user-baseline waste left to remove); a gated
rescue-checked ensemble close measured +0.0003 and was honestly refused.
omega_2 (0.60335): weak sweep draw (adaptive-stop suspected), gauge-BCE +0.0013
on the weak base. CONCLUSION: improvements do not add — they overlap; total run
quality is set by (a) opener draw and (b) member-structure luck at the close, not
by the count of known-good mechanisms applied. The omegas are the cleanest
demonstration that the ~0.6045 single-model wall is mechanism-saturated.

## Addendum: full-campaign mechanical post-mortem (31 Aug morning, 123 runs, tools/postmortem.py)
- Sub-floor true positives: rejected nodes with real gains (+0.0004-0.0008) recur — incl.
  a deterministic 0.60573 recency node (night_e/f, ABOVE the champion's 0.60558) that is
  correctly NON-designatable ("converged, not peak") and statistically noise (+0.00015,
  single seed). Lesson for future harness: two-stage floor (sub-floor -> reseed-confirm)
  instead of hard rejection; est. +0.001-0.002 cumulative signal recoverable.
- Failure taxonomy: llm_parse 65 / timeout 15 / exec 10 / provider 3 / other 6 — all
  recovered autonomously; parse truncation is the dominant robustness cost (token
  headroom > reasoning effort, matching the effort-grid measurement).
- Accept/reject ratio 108/240; every top result across both benchmarks ends in an
  ensemble close; selector streakiness (ignoring best-card evidence) is the residual
  inefficiency — addressed in the final steered run (disclosed launch directive).

## v2_r8 case study (31 Aug): diagnosis-gated menus + telemetry blindness
Chain: n1 node script omitted epoch history -> n2/n3 selector rationales explicitly
state curve telemetry was unusable -> fallback diagnoses (overfit, metric-mismatch)
-> selector policy (pick cards treating the diagnosis) never surfaced
context-stratified-pairs (treats ranking-mismatch|data-shift; the campaign's best
new card, absent from ALL consideration sets) -> three sub-eps iterations -> converged
0.60244. Two systemic lessons: (1) enforce the node telemetry contract mechanically
(reject/flag history-less nodes) so diagnosis is never blind; (2) surface top
measured-win cards to the selector regardless of diagnosis match (a "wildcard slot"),
since diagnosis errors otherwise hide the strongest evidence. Both are future-work;
the live steered runs (v2_s7/r9) bypass the issue via launch directives.

## The stale-knowledge class (31 Aug late morning — biggest systemic finding)
Audit of every selector-visible input found TWO stale layers that shaped the whole
campaign: (1) MENU.md CURRENT DIRECTIVE still recommended the watch-time objective
family (measured dead Fri night) to every selector since Friday — and --draft-tiers
"steering" never reached the selector at all (proposer-only), so directive-based
steering (v2_s7, v2_1k draft slots) silently failed; (2) 17 method cards carried
obsolete statuses — champion components labeled running-elsewhere, and the
championship-winning closes (ensemble-design-sweep, package-dial-sweep,
stage-matrix-sweep) labeled UNTRIED. Selectors systematically under-valued proven
cards and re-litigated dead ones. Notable: bigclock_07 won DESPITE the stale
directive (its selector overrode dead advice). Fixes shipped: evidence-ranked MENU
directive + measured-win-any-diagnosis rule + all statuses set to ledger verdicts;
menu_m3/m4 are the first runs with fully-correct knowledge. Lesson for the report:
knowledge-base freshness is as load-bearing as the acceptance statistics — a
research system needs mechanical status regeneration from the ledger, not manual
curation.

## Full-campaign error post-mortem + reliability fixes (31 Aug evening, 128 runs)
Failure census: llm_parse/truncation 65 (dominant; mitigated by token headroom),
timeout 17, exec errors 10 (fixer-recovered), telemetry-emission bugs 5, provider 3,
budget 1, smoke-timeout kills >=2 (120s cap killed heavy-architecture first attempts,
incl. ruby_x1's first composite try), no-op ensemble collapse 2 (member training real
but aggregation collapsed to anchor; caught by the byte-identical guard — integrity
worked, close lost). Fixes shipped for subsequent launches: smoke timeout 360s;
proposer ENSEMBLE_CONTRACT (distinct member seeds + member-distinctness assertion +
per-member logging); no-op collapse now fixer-eligible (one repair shot instead of a
burned node). Report lesson: the two highest-cost reliability defects were both
*silent success-shaped failures* (identical-output ensembles, missing telemetry) —
guards that make degeneracy loud are worth more than retries.

## Decision bench + decision replay (1 Sep early hours)
Built two evaluation tools that separate agent JUDGMENT from EXECUTION without training:
(1) decision bench — selector picks on frozen decision-states with known-good sets;
found its worst picks traced to stale card evidence (freq-adaptive-reg 6x-flat card
still read neutral), not bad reasoning; after evidence corrections + two general
principles (no re-sweep after clearing win; insufficient-telemetry is an honest
diagnosis) bench = 8/8. (2) decision replay — selector chooses over an evolving state,
each pick executed instantly from cached tuned artifacts: 3 decisions reached 0.60599
(above the designated 0.605575) when the close is executed per its card (design search,
never below best member). Conclusion (tempered per external review): the replay is an optimistic cached-
artifact counterfactual — evidence that judgment is strong, not proof it suffices;
the bench cannot yet distinguish good judgment from an always-pick-the-close constant
policy (fix: held-out states, conflicting-optima scenarios, constant-policy baselines).
Close-node code-generation fidelity remains the leading suspect for the live gap (guards shipped: ensemble contract, no-op fixer,
design-search language). Bench constitution: fixes may only be evidence corrections or
general principles, never state-specific answers.
