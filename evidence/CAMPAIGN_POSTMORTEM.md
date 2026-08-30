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
