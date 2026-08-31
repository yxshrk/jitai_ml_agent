# Final submission recipe (FINAL, Sun morning; 1K artifact policy updated Mon per external review)

## 1K ARTIFACT POLICY (Mon 31 Aug, per decision memo):
- Preferred (C): if run_v2_1k designates an in-run ensemble close, submit THAT exact
  artifact (member checkpoints + aggregation as recorded in its journal).
- Fallback (A): faithful reproduction of run_omega_1k node_005's converged checkpoint
  (0.66892). NOTE: our first replay scored 0.6772 — reconcile the procedure difference
  (final-fit epochs/early-stop state) or disclose reconstruction limitations; never
  label a differing rerun as the 0.66892 checkpoint.
- Rejected (B): the post-run 3-seed ensemble (val 0.6802) is DEVELOPMENT EVIDENCE ONLY
  (better-than-converged artifact violates checkpoint-at-convergence absent an explicit
  organizer ruling). evidence/test_submission_1k_omega.csv is therefore NOT the
  submission unless C fails AND A proves unreconstructable (then disclose fully).
- Receipt fields per artifact: run/node id, stop reason, config+preprocessing hashes,
  member seeds/epochs, aggregation + tie policy, exact validation metrics, CSV row
  count + sha256, selected in-run vs reconstructed. Probe disclosure: evidence/PROBE_MANIFEST.md.


## DESIGNATED CHAMPIONS (agent-designated, compliant with "checkpoint at convergence")
- **Pure: 0.60558 — run_bigclock_07 (coral) node_006, FULLY UNSEEDED.** Arc:
  baseline 0.6018 -> two-stage random dial search over the DCN package (accepted
  0.60424, clears epsilon) -> ensemble-design sweep (trained 7 members, agent
  validation-selected 3: seeds 42-44, per-user rank average) -> 0.605575.
  Config (agent-found, random-search dials): dropout 0.18, wd 9e-5, lr 0.001,
  StepLR gamma 0.57 / step 2, recency half-life 7d, DCN-lite + 0.5 BPR hybrid.
  6 iterations, 17 min wall, 115,315 tokens, zero interventions. Beats the seeded
  champion (0.60513) with no seed-script caveat. Test CSV: BUILT + VALIDATED (tools/predict_test_bc07.py;
  members reproduced at 0.6040-0.6044, ensemble 0.60561 ~= run value; 170,588 rows).
  [superseded: seeded run_desig_seeded_03 0.60513 -> now disclosed evidence]
- **1K: 0.6524 — run_max_1k_c (coral) node_002, unseeded.** The agent's 48-cell
  cross-stage factorial discovered a regime INVERSION vs Pure: DCN-lite with PURE
  LOGLOSS (no BPR: its diagnostics measured bpr-hybrid 0.593 vs logloss 0.646),
  NO recency weighting, dropout 0.13, k24 — closed as a validation-selected
  2-member ensemble (seeds 42, 1051). VERIFIED three ways: independent from-scratch
  evaluator reproduces 0.652403 exactly; in-run re-implementation scored 0.65221;
  fresh seeds 7/99 (never seen by the run) scored 0.64804/0.64735. Test CSV:
  REBUILD PENDING from this recipe.
  [superseded: run_desig_1k_01 0.63874 -> disclosed evidence]
- **27K (bonus, out-of-protocol scaling demo): 0.67263** — 5-seed ensemble on
  RTX 4090 (seeds 42-46, ruby); singles 0.6609-0.6633.
- NOTE: farm-greedy 0.60577/0.60602 are val-selected -> EVIDENCE ONLY. The honest
  consecutive-seed ensemble reproduces at 0.60513 in two independent agent runs.

## [STALE — SUPERSEDED by the designated-champions section above and the compliance
## ruling below. Team-built pool ensembles are WRITEUP EVIDENCE ONLY. The Pure
## submission is bc07 node_006: 3 members, seeds 42-44, per-user rank average,
## via tools/predict_test_bc07.py — ALREADY BUILT. Do NOT build a 5-seed Pure CSV.]
## Pure (primary) — historical Saturday decision
Best validation: **0.60577** — rank-average ensemble of 5 frozen-stack seeds
{46, 74, 93, 91, 60}, greedy-selected on validation from the 60-seed farm
(coral, farm_results.jsonl; optimizer log ens_opt.log).
- Baseline comparison: +0.0042 (vs 0.6016). Best single seed 0.6053; seed-42 0.6050.
- DECISION (made Sat evening, three-signal triangulation): KEEP the greedy 5-seed
  ensemble. Signals: full-val 0.6058 (best); late-val (Apr 26-28) inconclusive —
  collapses to single-seed selection on 3 days of data (selection overfit, discounted);
  random-exposure TEST-WINDOW probe (log_random Apr 29-May 8, evaluation-only, legal —
  not part of hidden test): all candidates within ~0.002, ensemble at the top, stable
  transfer for every candidate. tools/RANDOM_PROBE.md has the table. The ensemble is
  at/near the top of every signal and carries the variance-reduction rationale.
- Test-time procedure: train chosen seeds on TRAIN ONLY (organizer ruling), predict
  test with each, per-user rank-average, submit via evidence/submission.py +
  official submit.py --check. ONE test touch.

## COMPLIANCE RULING (Sat night, brief re-read): the scored submission must be
## "the validation-best checkpoint AT CONVERGENCE" of a run — an AGENT-designated
## node. Team-built pool ensembles are therefore WRITEUP EVIDENCE ONLY. Pure
## submission = the designated run's champion (expected: agent's in-run 5-seed
## ensemble node ~0.6058). Same for 1K (agent run w/ ensemble card); 27K agent
## run if time permits, else labeled out-of-protocol scaling demo.

## [evidence only] extended 159-seed pool greedy = 0.60602 (11 members:
s199, seedfarm_74, seedfarm_46, s196, s109, s191, s147, seedfarm_89, s105, s126,
seedfarm_100 — seedext seeds are 102-200 of the same frozen config). DESIGNATION
PENDING Sunday-morning random-probe transfer check (11-member made ~10 more
val-guided picks than the 5-member; pick by probe + parsimony, not val digit).

## 1K (bonus) — FROZEN Sat evening
Recipe: 5-seed per-user rank-average of the 1K-tuned config
(lr 0.00168, dropout 0.21, wd 3.7e-5, k 24, recency half-life 7, 6 epochs;
zoo/frozen_stack_1k.py), seeds {42,43,44,45,46}.
Validated: singles 0.6073-0.6216 (wide seed variance is why ensembling pays);
**ensemble 0.6323 valid primary**. Arc: default transfer 0.6134 -> tuned single
0.6214 -> ensemble 0.6323. No official 1K baseline exists; we report absolutes.
Test-time: train 5 seeds on train window only, predict test, rank-average, submit.
