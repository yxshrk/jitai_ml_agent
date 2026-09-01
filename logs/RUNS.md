# Autonomous runs — overnight 28 Aug

All runs: 0 interventions, official convergence rule, validation only.
Champion candidate: run_real_01 node_002 (0.6042). Manual 3-seed best: zoo/best.py 0.6039±0.0010.

| run | directive | stop | iters | best valid primary | delta | wall | LLM cost |
|---|---|---|---|---|---|---|---|
| 01 | default tiers | converged | 5 | **0.6042** | **+0.0026** | 23.5m | $0.90 |
| 02 | + overfit insight | converged | 5 | 0.6040 | +0.0024 | 14.2m | $0.57 |
| 03 | watch-time (ignored: policy bug, fixed) | converged | 3 | 0.6033 | +0.0015 | 11.6m | $0.66 |
| 04 | watch-time themes (FM base) | converged | 3 | 0.6026 | +0.0008 | 15.6m | $0.63 |
| 05 | compound DCN+themes | converged | 3 | 0.6033 | +0.0015 | 12.8m | $0.68 |

## Sunday 30 Aug — fan-out doctrine evolution (see git history for harness changes between runs)
| family | runs | best | note |
|---|---|---|---|
| unseeded atomic (05-11) | 7 | 0.60424 (06) | ensemble + duration-heads discoveries |
| synergy cards (12-15) | 4 | 0.60404 | first package proposals |
| effort grid (16-22,27,28) | 7 | 0.60319 | high/xhigh: measured FAILURE (truncation) |
| dial-sweep wave (23-26) | 4 | 0.60468 (25) | first in-node search + ensemble |
| chains (c1,c2,l1,25a,25b) | 5 | 0.60468 | ceiling confirmed; philosophy retired |
| bigclock wave (01-07) | 7 | **0.60558 (07) = DESIGNATED** | dial sweep cleared epsilon -> ensemble-design close |
| deep/anthropic/multidraft | 6 | 0.60462 (g1) | sonnet thinking-starvation measured |
| max wave + experiments | 12 | in flight | honest-clock depth, 1K deep runs, clean A/B |
All journals in logs/run_*/; cross-machine copies rsynced back on completion.

## Late harvest (31 Aug)
| run | machine | stop | iters | best | note |
|---|---|---|---|---|---|
| qb_b | coral | converged | 4 | 0.60466 | decayed-positive sampling; below champion, no re-designation |
| novel_r1 | ruby | max_hours | 5 | 0.60447 | pair-kernel line on ruby; below champion, no re-designation |
| final_f1 | ruby | max_hours | 4 | 0.60403 | dial-jitter member-bank close (GPU); below champion, no re-designation |
| clean_c1 | coral | KILLED n=2 | - | - | discarded: mid-run METHODS.md dedupe sync = intervention taint; relaunched as clean_c2 with fixed library |
| combo_r1 | ruby | converged | 4 | 0.60496 | SEEDED composition test: pair-kernel+frozen-stack = +0.0003 over frozen alone — mechanisms barely stack; disclosed experiment, not designation-eligible per policy |
| omega_1k | ruby | converged | 8 | 0.66892 | 1K BREAKTHROUGH: causal session features (+0.0192); independent eval exact-match; fresh-seed replication in flight; re-designation pending |
| clean_c2 | coral | converged | 3 | 0.60335 | clean cards-only run, safe path, no threat |
| final_s1 | coral | converged | 3 | 0.60184 | 3 straight rejections incl. session-time-features +0.0002 on Pure (weak-transfer confirmed); gated variant untried |
| final_s3 | coral | converged | 4 | 0.60396 | full-library lottery ticket; NOTE node_003 gated-session composite measured 0.6044 but grey-zone rejected — reconcile in morning |
| omega1k audits | ruby | complete | - | 0.6766/0.6762/0.6652 | replication + shuffle: breakthrough confirmed |
| clean_r3 | ruby | converged | 3 | 0.60381 | GPU clean run; session +0.0001 and gauge-bce flat from dial-swept base — mechanisms overlap confirmed |
| omega CSV | ruby | built+checked | - | ens-val 0.68020 | evidence/test_submission_1k_omega.csv (4.13M rows, checker OK); old max_1k_c CSV preserved pending decision |
| final_s2 | coral | converged | 4 | 0.60499 | best of final wave: gauge-bce -> hetero-ensemble close (+0.0010, first measured win for that card); gated-session +0.0006 sub-eps |
| final_s4 | coral | converged | 3 | 0.60521 | context-stratified-pairs FIRST WIN +0.0015 in one node; converged before any close — close-on-top untried |
| 1k_push | ruby | converged | 7 | 0.65977 | 2nd independent 1K run: session features +0.022 from different base; below omega designation |
| novel_l1 | laptop | max_hours | 5 | 0.60524 | CLEAN researcher run: invented-card pair-kernel +0.0014; its 0.6055 ensemble FAILED fresh-seed confirm (discipline held); official best node_004 |
| final_r6 | ruby | converged | 3 | 0.60258 | weak dial-sweep draw; gap-conditioned-recency +0.0003 sub-eps (first measurement) |
| final_s5 | coral | converged | 4 | 0.60399 | selector took stage-matrix path, not context-pairs; its close (0.60449) rejected sub-eps. CAMPAIGN COMPLETE |
| last_r7/last_1k/last_s6 | ruby/coral | KILLED early | - | - | discarded: library synced mid-run (intervention taint) + pre-floor-v2 harness; relaunched as v2_* |
| v2_r8 | ruby | converged | 3 | 0.60244 | floor-v2 accepted +0.0006 foothold but 2 rejects converged it; lesson: sub-eps accepts still strike (official rule) |
| v2_s7 | coral | converged | 3 | 0.60362 | steered run IGNORED directive — root cause: draft-tiers reach proposer only; MENU CURRENT DIRECTIVE (selector channel) was stale w/ dead watch-time themes since Fri. MENU fixed for successor runs |
| v2_r9 | ruby | KILLED early | - | - | stale-knowledge lottery ticket cut to free GPU for v2_1k session node; corrected-knowledge GPU run queued for when v2_1k ends |
| menu_m4 | coral | converged | 3 | 0.60335 | corrected-knowledge run GENUINELY chose seq-deepfm card (unsteered) — measured 0.6033 vs external 0.6047: one-node port shortfall or transfer gap |
| v2_1k | ruby | converged | 3 | 0.62110 (baseline) | steered 1K run failed: session node crashed on impl, no in-run ensemble -> 1K artifact = faithful A-form per policy |
| pod experiment | runpod | terminated | - | - | 2x 2xRTX4090 secure pods: CPU quota oversubscribed (~1 effective core/trainer despite caps) — unusable for CPU-bound loop; ~$1.8 spent; lesson: GPU pods != CPU capacity |
| menu_m3/m5 | coral | DIED (disk full) | - | - | coral disk refilled to 100%; coral retired as run host |
| ruby_x1 | ruby | converged | 6 | 0.60443 | GENUINE composite adoption (+0.0026 one node); close attempted but node bug produced parent-identical predictions — no-op guard correctly rejected |
| ruby_x2 | ruby | converged | 4 | 0.60401 | 3rd genuine composite adoption (0.60454, grey-confirm miss); close executed but weak |
| ruby_y1 | ruby | converged | 3 | 0.60362 | phase guidance WORKED (opened with composite) but weak impl (+0.0018 sub-eps) -> fast convergence |
| ruby_x3 | ruby | converged | 3 | 0.60425 | 4th genuine composite adoption; converged before close |
| composite-close probe | ruby | evidence | - | 0.60477 | 3-seed rank-avg of x1s agent-authored composite: close pays only +0.0003 on this family (correlated members); chain ceiling ~0.0048 short of hopes; gap to teammate reference = implementation depth |
| ruby_y2 | ruby | converged | 3 | 0.60361 | quiet path (gauge accept, no composite pick) |
| recipe search | laptop | evidence | - | 0.605938 | direct equal-weight cross-family blends over agent-authored members: pairkern+comp42+comp43 best; robust cluster 0.6058-0.6059; evidence-only (team-mixed), agent path = in-run hetero close |
| cpu_c2 | cpupod | converged | 4 | 0.60399 | quiet path |
| decision replay | laptop | evidence | 3 decisions | 0.60599 | OPTIMISTIC CACHED-ARTIFACT COUNTERFACTUAL (external review wording): selector choices replayed over cached tuned artifacts reach 0.60599 under favorable conditions (global ranks, canned curves, no reseed gate); indicative of judgment quality, NOT an autonomous result |
| ruby_y3 | ruby | converged | 3 | 0.60400 | pre-bench library run; grey-accepted freq (+0.0008 with z-pass — floor-v2 working as intended) then converged |
| cpu_c1 | cpupod | converged | 5 | 0.60403 | 2nd genuine composite adoption held as peak; close never landed |
| ruby_w1 | ruby | converged | 4 | 0.60412 | FIRST live farm-close execution: accepted, no collapse (contracts held); gain small (+0.0002) from weak member base |
| pure_clean | cpupod | converged | 3 | 0.60284 | old-doctrine clean (literature-only knowledge): reg-package win node_001 (+0.0010, 4 coupled dials via probe sweep), then converged; self-critique flags early convergence + bundled-factor attribution — provenance-tier exhibit |
| showcase | ruby | converged | 3 | 0.60426 | final full-stack showcase run: regularized package accepted then converged; 111.9k tokens, 109 min |
| ruby_z1 | ruby | converged | 3 | 0.60356 | weak opener never recovered; converged low; 110.9k tokens, 183 min |
| ruby_z2 | ruby | converged | 4 | 0.60514 | strong iter-1 package (0.60514) but the ensemble close never improved on it; 155.7k tokens, 202 min |
| pure_clean2 | ruby | converged | 3 | 0.60415 | BEST clean run ever: literature-only + research doctrine; provenance tier now 0.6041; 53.9k tokens, 111 min |
| cpu_c3 | cpupod | max_hours | 4 | 0.60437 | composite adopted early then wall-clock cap; pod cores made nodes expensive; 142.2k tokens, 361 min |
| farm_f1 | ruby | converged | 3 | 0.60464 | first farm-capable run: two clean accepts then endgame mis-instrument (same-family close, ceiling < eps) -> converged early; root-caused to stale CONVERGENCE_PRESSURE steer; fixed + benched (endgame_eps_math 3/3) |
| farm_f2 | ruby | converged | 3 | 0.60364 | eps-arithmetic fired but chose bare-reach atom over wide-margin close; root-caused -> margin principle + endgame_margin_not_reach fixture (6/6) |
| farm_f3 | ruby | converged | 3 | 0.60314 | JUDGMENT FIX PROVEN LIVE: iter-3 chose farm-close with verbatim eps+margin reasoning; plan died on hallucinated script_source (we never gave it valid paths) -> code-member contract fix |
| farm_f4r | ruby | KILLED n=3 | - | - | resumed f4@3 memory-free with prompt logging; picked composite (card evidence 0.606116 human-assisted label + blind curve) -> killed per protocol; root causes fixed (evidence relabel, history normalizer) |
| farm_f4r2 | ruby | converged | 3 | 0.60450 | FIRST LIVE AUTONOMOUS FARM-CLOSE: resumed f4@3, chose farm-close (flat-signal diag on real curve), plan repaired once, 4 probes -> 2 full (dcn 0.6045, composite 0.6025) -> honest singleton fallback -> +0.0016 accepted -> converged; 5 min segment; resumed lineage, not designation-eligible |
| farm_f5 | ruby | KILLED n=1 | - | - | clean from-scratch memory-free candidate: opened with farm-close (bench had graded this neutral; my rubric was lenient) -> 3 of 4 cold-written members crashed, 4th scored 0.38, executor fell back to incumbent -> killed per protocol; fix: farm card precondition (closing move, never an opener) + fixture marks it bad |
| farm_f6 | ruby | KILLED n=5 | - | 0.60424 | clean memory-free candidate: opener package-dial-sweep (right call, defective implementation 0.6014 rejected) -> composite 0.60424 ACCEPTED -> gauge-bce 0.60419 rejected -> hetero close chosen correctly from 0.6042 base but all 4 plan members were cold rewrites (one crashed, best 0.6023) -> incumbent fallback; iter 5 re-chose the close -> killed; fix: anchor member = champion script via script_source |
| farm_f6r | ruby | budget_exhausted | 5 | 0.60424 | RESUMED f6@4 with ANCHORED farm plan: member 1 = champion script (src), 3 derived code members; full blend 0.60497 beat best member 0.60443 (real cross-family win) but grey-zone confirm reruns reseeded members (anchor 0.6037/0.6034) -> incumbent fallback twice -> z-test rejected (+0.0007 not seed-repeatable); iter 5 refused by $150 ledger cap; resumed lineage, not designation-eligible |
| farm_f7 | ruby | converged | 4 | 0.60447 | judgment fixes held; grey-zone confirm tax cost the promising ctx-pairs branch (153611 tokens, 44 min) |
| ff1 | laptop | converged | 3 | 0.60184 | clean fast-forward shakeout: found debug-gate bypass, margin miscal, screen fidelity weakness for ~$8 (75100 tokens, 34 min) |
| f8 | ruby | converged | 4 | 0.60471 | first stacked-accept lineage (snippet-backed ctx-pairs); killed by 503-as-strike; state extends to 0.60576 with cross-run artifacts (132067 tokens, 39 min) |
| f9 | ruby | converged | 4 | 0.60510 | best autonomous single model ever (48-probe opener); endgame banking gap found+fixed; own-artifact ceiling 0.60546 (177285 tokens, 36 min) |
| c9 | coral | converged | 4 | 0.60332 | clean/no-digest arm: 2 failed screen builds (gate-caught) then accepted screen; converged on build-failure strikes; snippet+pivot fixes landed post-launch (87296 tokens, 30 min) |
| f10b | ruby | KILLED n=3 | - | 0.60358 | weak opener 0.6036; killed for time (menu/cards inconsistent at launch) |
| f11b | ruby | KILLED n=2 | - | 0.60306 | lean+FAST: 24 coarse + 512 short 'refine' probes + 2 finals = worst opener; root cause of FAST rewrite |
| f12 | ruby | KILLED n=0 | - | - | relaunched as f14/f15 per plan |
| c11 | coral | converged | 3 | 0.60184 | clean lean: three failed builds of the same card (offset-encoding IndexError x2, gate x1) via forced debug routing; both bugs fixed (TASK_BRIEF invariant; smoke failures return to selection) |
| c11b | coral | KILLED n=1 | - | - | relaunched with fixes then retired: clean arm dropped for the final tickets |
| f13 | laptop | KILLED n=1 | - | - | parallel ticket, killed for the fixed-card relaunch |
| f14 | coral | KILLED n=2 | - | 0.60305 | bigclock profile: opener ensembled 5 weak members in-node (card text); root cause of the card fix |
| f15 | ruby | KILLED n=2 | - | 0.60326 | same as f14 |
| f16 | ruby | KILLED n=2 | - | 0.60372 | fixed card, plain bigclock profile; iteration 2 died on a proposer error ('int' not iterable, unresolved) |
| f17 | coral | KILLED n=1 | - | 0.60343 | fixed card, 554 s opener (bigclock pace) but heavy-reg basin; replaced by f18 |
| f18 | coral | converged | 5 | 0.60400 | bigclock-plus profile: first draw below baseline, second draft gauge-fixed-bce 0.6040 (eps-clearing), ctx/reg riders flat, final ensemble close BLOCKED by the rewrite gate (bug, fixed) (166741 tokens, 38 min) |
