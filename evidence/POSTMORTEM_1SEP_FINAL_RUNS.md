# Post-mortem: the 1 Sep final-night runs (f7 → f8 → f9, ff1, c9)

Consolidated evaluation of every run after the 31 Aug ledger closed. All times
SGT, 1 Sep 2026. Companion documents: `SESSION_31AUG_LEDGER.md` (harness state
these runs started from, plus the evidence-only exhibits), `logs/RUNS.md`
(scoreboard), per-run `logs/run_*/` (journals, prompts, self-critiques).

## Scoreboard

| run | tier | machine | iters | best | stop | wall | what it proved |
|---|---|---|---|---|---|---|---|
| f7 | measured-memory | ruby | 4 | 0.604474 | converged | 44 min | judgment fixes hold; grey-zone confirm tax is real |
| ff1 | clean, fast-forward | laptop | 3 | 0.601838 (baseline) | converged | 34 min | shakeout: found 3 harness holes for ~$8 |
| f8 | measured-memory | ruby | 4 | 0.604714 | converged | 39 min | first stacked-accept lineage; killed by a 503 counted as a strike |
| f9 | measured-memory | ruby | 4 | **0.605102** | converged | 36 min | best autonomous single model ever; endgame banking gap |
| c9 | clean (no digest) | coral | 4 | 0.603324 | converged | 30 min | clean arm recovered from 2 failed builds via gate+fixer, then converged on build-failure strikes |

Designations are UNCHANGED: Pure = bigclock_07 **0.605575**, 1K = omega_1k
0.66892. No run tonight beat the designation; the measured single-run ceiling
(0.6055–0.6060) survived three attempts with progressively better harnesses.

## Run-by-run

### f7 (launched 23:53, the 31 Aug config)
- n1 package-dial-sweep ACCEPTED 0.60447 (5-member internal ensemble).
- n2 context-stratified-pairs peak 0.60500 but +0.0005 gain failed the 0.0009
  floor + 2 reseeded confirms (grey-zone tax: ~3 nodes spent on a rejection).
- n3 heterogeneous close: cold cross-family members probed weak (0.6016 vs
  anchor 0.6045); executor correctly FELL BACK to incumbent. n4 hetero-objective
  blend 0.6041 < champion. Converged on 3 strikes.
- Self-critique (accurate): bundled opener prevented attribution; n2's 0.6050
  deserved paired-seed replication, not abandonment.
- Verdict: every decision defensible; no old failure modes recurred; lost to
  variance + confirm arithmetic, not judgment.

### ff1 (clean + fast-forward, 1-epoch everything; diagnostic only)
- Purpose: audit decisions/code without training time. 12 planned iters, $8.
- Found: (1) proposer can label a node `debug` and BYPASS the smoke gate
  (0.5884 script trained full-length) → gate now covers debug; (2) smoke-gate
  margin 0.005 mislabels ordinary screen variance as breakage → retuned to
  0.010 from the fidelity sweep's measured populations; (3) mechanism-screen
  implementations are systematically weak (0.5901, 0.5884 vs baseline 0.6018).
- Verdict: the shakeout paid for itself several times over.

### f8 (all judgment fixes + snippets + gate; sol@high proposer, sol fixer)
- n1 package-dial-sweep 0.60411; n2 context-stratified-pairs ACCEPTED 0.60471 —
  first live success of a snippet-backed card, and the first run to stack two
  accepted mechanisms.
- n3 hetero close rejected (0.60468 < champion, fresh members weak). n4: the
  provider returned HTTP 503 on the SELECTOR call; the harness counted the
  outage as convergence strike 3 and ended the run.
- Post-hoc: the state f8 died in extends to 0.605765 when blended with other
  runs' polished artifacts (evidence-only), and its champion is genuinely
  blend-complementary (+0.0003 over the best pair without it).
- Fixes triggered: transient provider errors now retry once and never count as
  a convergence strike (tested with an exact replay).
- Counterfactual (replayed state through the live selector, 3 reps): iteration
  4 would have been social-mtl-heads via 17–20 minimal edit hunks — a new-family
  strengthener per the retry-close doctrine, i.e. a sane move, not a blunder.

### f9 (adds edits-mode improves, 503 fix, rewrite gate)
- n1 package-dial-sweep ACCEPTED **0.605102 as a SINGLE MODEL** — best opener
  ever (48 probes, found an out-of-basin winner at dropout 0.34; prior winners'
  band was 0.16–0.28). Partly better generated search code, partly luck the
  code bought.
- n2 context-stratified-pairs via edits (production first for edits-mode; 94%
  of parent preserved): 0.605118, +0.00002 → rejected. Correct: the stronger
  base had already banked that headroom.
- n3 farm-close: fresh members measured 0.5975–0.6035 vs anchor 0.6051 → blend
  lost, fell back, rejected. n4 hetero-objective 0.6046 → strike 3, converged.
- THE ENDGAME GAP: at n3/n4 the selector explicitly rejected seed-ensemble as
  "only +0.0003..+0.0013" because it cannot reset the epsilon streak — but on a
  probable-final iteration the deliverable is best-so-far, not the streak.
- Manually executed its declined moves (evidence-only): 5-seed ensemble
  0.605303 (+0.0002, NOT the digest's promised +0.0004..+0.001 — diminishing
  returns on an already-tuned single); all-own-artifacts blend 0.60546. So even
  perfect banking leaves f9 short of 0.605575: the loss was ~0.0003 of ceiling,
  not a blunder.
- Fixes triggered: BANK-THE-LAST-GAIN principle (reliable = measured; fresh-
  member closes inherit observed fidelity weakness) — bench fixture went
  bad-pick 3/3 → acceptable 3/3. Farm contract now prefers the run's OWN
  trained nodes as members (a rejected sibling within ~0.002 of the champion is
  a finished, measured member; fresh code only for families never built).

### c9 (clean/no-digest, coral; converged 0.603324, 4 iters, 30 min)
- n1 mechanism-screen crashed (unseen-id indexing); n2/n3 attempts gate-caught
  at ~0.590; the THIRD build passed and was ACCEPTED at 0.603324 — the
  gate+fixer loop converting implementation failure into a working screen.
- Converged immediately after: the two failed builds were strikes 1-2 and the
  accepted screen's +0.0015 was sub-epsilon (strike 3). One working iteration
  total; final 0.603324 vs the clean arm's prior best 0.6041 (pure_clean2).
- Fixes triggered: methodology-only screen skeleton snippet (probe loop,
  successive-halving budget, unseen-id invariant); doctrine now distinguishes
  implementation-dead (pivot after two failed builds) from evidence-dead —
  both landed AFTER c9's launch (its brain cached the older library), so c9 is
  the motivating case, not a test of the fix. Open question its self-critique
  raises: should failed BUILDS count convergence strikes at all?

## Cross-run findings (the report's spine)

1. **Judgment is no longer the bottleneck.** Across f7–f9 every selection was
   defensible and most were optimal-in-hindsight; the two genuine judgment gaps
   found tonight (post-rejected-close strengthening, endgame banking) were
   converted into general principles and verified on real-state benches
   (close_rejected 2/2 bad → 0/3; bank_last_gain 3/3 bad → 0/3).
2. **Implementation fidelity is the bottleneck, and it is quantified.** Fresh
   implementations of the same method spread 0.592–0.600 (9-rep sweep); fresh
   farm members land 0.001–0.004 below their family's potential; no coder
   config (sol medium/high, terra) eliminates the variance. The protections
   that work: smoke gate at the measured margin (truncates the broken tail),
   reference snippets (paper-faithful, citation-tier), edits-mode (accepted
   artifacts evolve byte-identically instead of being re-typed — 3/3 live
   bench, first production use in f9 n2).
3. **The ceiling is real and measured.** Three runs with successively better
   harnesses landed 0.6045 → 0.6047 → 0.6051, and exhaustive post-hoc banking
   of f9's own artifacts reaches 0.60546. Only cross-run artifact pooling
   (0.605765) clears the designation — which is exactly what a single
   fresh-memory run cannot do, and why bigclock_07 (a full campaign's endgame)
   remains the designation.
4. **Ensemble arithmetic:** blends need members that are individually
   competitive (within ~0.001–0.002) AND decorrelated. Fresh members fail (1);
   own-lineage siblings can fail (2) (f9's two strong nodes were 94% identical:
   +0.0002). The strong blends in evidence all pool artifacts that each took a
   full run to polish.
5. **Infrastructure is part of the experiment.** A 503 ended f8; watchers,
   retry-with-no-strike, and the spend ledger are as load-bearing as the ML.

## Harness deltas shipped tonight (all committed on `clean-agent`, 158 tests)

smoke sanity gate (draft/improve/debug; margin 0.010 from measured populations)
· citation-tier reference snippets, injected per-selection (7 cards incl.
screen skeleton) · edits-mode improve proposals (exact-match hunks, one repair
round) + rewrite gate backstop · transient-error retry, no convergence strike ·
guidance: new-family-after-failed-close, embedded-component check, endgame
margin arithmetic, bank-the-last-gain, implementation-dead pivot, own-lineage
farm members · 3 literature-only clean cards · real-state clean bench track
(8/11 good, 0 bad) · failure-replay test suite · fast-forward mode · per-role
effort override · fixer promoted to sol.
