# Method cards

Compiled from `research/kuairand-and-watchtime.md`,
`research/models-losses-hparams.md`, `MENU.md`, and the read-only
`zoo/EXPERIMENTS.md`. Measured statuses also incorporate the run-04/05 journal
numbers behind MENU's CURRENT DIRECTIVE. A `measured-dead` card must not be
selected again in the measured form; a materially different mechanism may get
its own card.

Frozen-stack validation references: Pure = 0.6047; 1K = 0.6134 (literal frozen
default, seed 42).





## Search-scope policy (escape your own priors)
Measured fact from this campaign: winning dials are often UNINTUITIVE — weight decay
1e-3 (100x the canonical default), validation-best checkpoint at 0.5 epochs, lr 0.00168.
Hand-picked grids of round numbers systematically miss such optima. Therefore stage-1
of any dial search must be RANDOM SEARCH over WIDE log-uniform ranges (Bergstra &
Bengio, JMLR 2012), not a grid of values you consider reasonable: sample >= 20 configs
from lr 1e-4..1e-2 (log), weight decay 1e-6..3e-3 (log), dropout 0.05..0.5,
recency half-life 2..21 days; checkpoint every half epoch from 0.5 onward and let the
data pick the stopping point. Then refine (stage 2) locally around the best sample.
Never narrow a range because a value "seems too extreme" — extremes have won here.


## Advanced search mechanics (use inside fan-out nodes)
- SUCCESSIVE HALVING (Hyperband/ASHA, Li et al. 2018) is REQUIRED for any sweep of
  more than 16 candidates: evaluate all at a short budget (1-2 epochs), kill the
  bottom half, double the survivors' budget, repeat until <= 4 survive — then train
  EVERY survivor at FULL fidelity and pick the winner from those full-length scores
  only. Never commit a config on the basis of short-budget scores (measured failure:
  a 208-probe sweep that skipped the full-fidelity final round finished at 0.6024,
  losing to a 59-probe sweep that ranked at full length, 0.6043).
- SNAPSHOT ENSEMBLING (Huang et al., ICLR 2017) — CONSIDER it in ensemble closes (untried here; card status governs): with a cyclic or restarted LR,
  save several checkpoints from ONE training and use them as extra ensemble members
  free of retraining cost. The ensemble-design sweep may mix seed-members and
  snapshot-members and let validation pick the blend.
- CAPACITY DIALS BELONG IN THE SEARCH SPACE: include embedding dim k {8..48} and MLP
  width {64..256} in stage-1 random ranges alongside regularization — capacity x
  regularization is the canonical interaction (deep CTR literature) and must be
  searched jointly, not fixed by habit.
- DATA-SIDE DIALS: recency SHAPE (exponential vs step vs linear decay), and
  segment-weighted training (by tab or duration bucket) are legitimate searchable
  families largely unexplored here; probe them in combo sweeps.
- ARC REMINDER: after a dial-swept package is accepted, the combo-sweep card (add-on
  mechanisms probed ON the tuned champion, incl. pairs) is the highest-expected-value
  next play, BEFORE the ensemble close. The record run skipped it — do not.

## Depth policy (FAST — measured 1 Sep: depth buys variance, not score)
Measured openers: 14 probes -> 0.6042 (bigclock_07), 48 probes -> 0.6051 (f9),
48 probes -> 0.6036 (f10b). Beyond ~20 well-ranked probes the sweep's outcome is
dominated by the random draw, not by depth, while wall-clock is a scored resource.
Therefore:
- Stage-1: 16-24 SHORT probes (2-3 epochs), half from the measured basins, half
  wide; STOP EARLY when the best-found has not improved over the last 8 probes.
- One refinement pass (4-6 probes) around the winner; the top 2 candidates get
  full-length final trainings; commit only on full-length scores.
- Target: an opener node in ~10-15 minutes. Spend the saved clock on the NEXT
  iterations (a re-swept rider mechanism, then the close), not on more probes.

## Depth policy (overrides brevity instincts)
- PROBE PARALLELISM: probes are small models — run them CONCURRENTLY, not one at a
  time. On CUDA, train 4-6 probe variants in parallel (separate processes via
  multiprocessing spawn, or interleaved in one process); on CPU, use one process per
  probe across cores (each with a bounded thread count). A sequential sweep on an
  idle device wastes most of the budget; measure per-probe wall time in progress.log.

Searches must be EXHAUSTIVE, not token gestures. Superseded by the MODERATE policy above where they conflict; historical minimums: stage-1 >= 16 probes, stage-2 >= 10; probes at FULL training length
on the full data whenever the device is fast (GPU) or the timeout is in hours — short
subsampled probes are a last resort and mis-rank configs near the optimum. Stop when the adaptive rule above fires or the grid is covered, whichever is first. Reserve
time only for the final training + ensemble close. Depth is free under the rules;
shallow searches are the known cause of the remaining score gap.

## Clock policy (read this)
The wall-clock ceiling is 6 HOURS per run and feasibility is graded in coarse tiers —
spending clock on deeper search is the cheapest resource trade available. When the node
timeout is measured in hours: probe at full training length (short probes mis-rank
configs near the optimum), afford 16-24 matrix cells, and refine twice. Reserve time
for the final full training + the ensemble close.


## Measured search basins (cross-run probe evidence — sample HERE first)
Aggregated from ~600 logged probes across prior runs on PURE (full tables in each
run's progress.log). The productive basin for the DCN package:
- dropout 0.16-0.28 (winners cluster 0.17-0.23)
- weight_decay 5e-5..5e-4 (log-uniform)
- lr 0.0003..0.0014 with StepLR gamma 0.45-0.68, step 1-2 epochs
- recency half_life 4-15 days (7 +/- a few is safe)
- probe scores in-basin: 0.6026-0.6044 single-model; out-of-basin USUALLY drops fast
  — but not always: the best single model ever measured (run_f9 n1, 0.605102) won at
  dropout 0.34, OUTSIDE this band (lr 0.0012, gamma 0.57, hl 8.4), found by a 48-probe
  wide search. The basin is where winners cluster, not a boundary.
Champion configs measured: (0.18, 9e-5, lr 0.001, gamma 0.57/2, hl 7.0) -> 0.6042
single / 0.6056 ensembled.
Guidance: draw ~50% of stage-1 samples from this basin, ~50% wide exploration
(basins can move with architecture changes); refine locally as usual. On 1K the regime INVERTS vs Pure
(measured, run_max_1k_c 48-cell factorial + fresh-seed confirmed): PURE LOGLOSS
beats the BPR hybrid by ~0.05 (bpr-hybrid cells mean 0.593 vs logloss 0.646);
NO recency weighting wins (half_life None); best known config = dcn-lite,
logloss, dropout 0.13, k 24, lr 0.00168, StepLR gamma 0.95 -> single 0.648-0.650,
2-member rank ensemble 0.6524. gauge-fixed-bce is UNTRIED on 1K and is the top
candidate rider given the pointwise loss dominates there. 1K has large remaining
headroom — deeper search pays there (unlike Pure).

## Measured campaign digest (this benchmark, cross-run memory — full seed tables in zoo/EXPERIMENTS*.md)

Interaction facts, each replicated across >= 3 seeds unless noted:
- dcn-lite ALONE from baseline: ~0.6014 (WORSE). dcn-lite + regularization-schedule +
  bpr-hybrid + recency-weighting as one package: 0.6047 +/- 0.0003. The package is the
  unit that works; the atoms measure at or below baseline.
- NEW (measured in-run): the package PLUS a click auxiliary BCE head at weight 0.1
  scored 0.60477 single-model (exp_anthropic node_003) — the click aux, dead in
  isolation, is a live rider ON the full package. Untested with an ensemble close.
- bpr-hybrid ALONE: ~0.6003 (worse). Inside the package: contributes ~+0.001.
- recency-weighting (7d) on the regularized package: +0.001 mean (+0.004 at best seed);
  on a weak parent its grey-zone confirms fail — needs >= 3 seeds to detect.
- regularization-schedule alone on baseline FM: +0.0015 (works standalone).
- seed-ensemble gain is BASE-DEPENDENT: +0.0014 off a 0.6042 single (bigclock n6 ->
  0.6056); +0.0004..+0.001 off mid-strength champions; only +0.0002 off a heavily
  tuned 0.6051 single (f9 exhibit, 1 Sep). Ensemble of baseline: only ~0.6028.
- duration-regime-heads on baseline: +0.0014 in the unseeded wave, but NEVER confirmed above eps since (card verdict: measured-flat on Pure).
- Measured DEAD on this benchmark (do not re-try in measured form): extra feature
  fields (all variants), larger embeddings (k>16 on Pure), deeper crosses, SWA/EMA
  single-model, listwise softmax UNDER OUR SHORT-PEAK REGIME (see listwise-regime
  card for the untried regime). ARCHITECTURE-CONDITIONAL, NOT DEAD: sequence/history
  models and watch-time auxiliaries measured dead on FM/DCN but ALIVE inside the
  seq-deepfm-composite package (see card) — the dead-list is conditional on the base
  architecture.
- Records: best ENSEMBLE = 0.605575 (bigclock_07: package + half-epoch checkpoint +
  5-seed close); best SINGLE model = 0.605102 (f9 n1: 48-probe swept package, no
  ensemble). A from-scratch run must bundle the package in ONE node to keep iteration
  budget for the close — but note the close's gain shrinks the better the single.

## Combination & interaction guidance (literature-grounded)

Published methods ship as PACKAGES, not atoms — evaluate them the way their papers do:
- **Capacity needs regularization.** DCNv2 and DeepFM report their architectures WITH
  dropout, L2/weight decay, and tuned LR schedules; the bare architecture on a small
  dataset commonly measures at or below a well-regularized baseline (DCNv2 paper;
  FuxiCTR/BARS reproducibility studies). If proposing dcn-lite or finalmlp, propose it
  TOGETHER with regularization-schedule in one node.
- **Objective changes ride on a stable trainer.** BPR-style pairwise objectives are
  reported with their own regularization and tuned steps (Rendle et al., BPR); hybrid
  pointwise+pairwise losses are evaluated as the blend, not the parts. Pair bpr-hybrid
  with regularization-schedule, not with the raw baseline trainer.
- **Temporal weighting shows on top of a regularized ranker** (temporal-dynamics /
  recency literature, e.g. Koren's temporal CF onward): its mean effect is small, so
  on an underfit or overfit parent it drowns in noise. Prefer recency-weighting as a
  rider on an already-accepted regularized stack, and confirm with >= 3 seeds.
- **Ensembling is last.** seed-ensemble amplifies whatever champion exists; apply it to
  the best single stack, not the baseline (ensemble of a weak model = slightly less
  weak model).
- Practical rule: one NOVEL mechanism per node stays the norm, but a node may bundle a
  novel mechanism with the PROVEN regularization-schedule package (and/or an accepted
  ancestor's components) when the literature setup does so — cite the pairing. 


### bpr-hybrid: Within-user BPR + pointwise hybrid
- mechanism: Form positive/negative pairs only inside each user's impressions and optimize logistic score differences. Mix 0.5 BPR with 0.5 BCE so ranking alignment does not discard pointwise stabilization.
- treats: metric-mismatch | flat-signal
- reference_primary: 0.6048
- preconditions: Training batches expose complete user groups; retain rows from one-class users for the BCE term. The known-best parent already contains this and should not receive another mix sweep.
- citation: Rendle et al., BPR; RankTower (arXiv:2407.12385); `zoo/EXPERIMENTS.md` E5
- expected_gain / cost: Already delivered primary 0.6048 at seed 42; changing the 0.5 mix is expected <=0.001 / low once grouping exists.
- status_pure: measured-win as champion component (0.5/0.5 hybrid inside the 0.6047 package; pure BCE 0.6038 and pure BPR 0.6036 alone are measured dead)
- status_1k: untried

### dcn-lite: DCNv2-lite interaction head
- mechanism: Add one or two explicit cross layers and a small MLP over the five field embeddings. This supplies bounded higher-order interactions without the parameter cost of xDeepFM/AutoInt.
- kind: opportunity
- treats: underfit | flat-signal
- reference_primary: 0.6039
- preconditions: Use k=16, hidden around 128, GAUC early stopping, and the known-best hybrid loss; more depth is within noise and overfits.
- citation: DCNv2; FuxiCTR/BARS; `zoo/EXPERIMENTS.md` E1-E3
- expected_gain / cost: Accepted stack primary 0.6039 +/- 0.0010, +0.0023 over official baseline / low.
- status_pure: measured-win as champion architecture (core of every >=0.6042 package)
- status_1k: untried

### regularization-schedule: Compound dropout, row-L2, weight decay, and LR decay
- mechanism: Apply MLP dropout around 0.3, accessed-row embedding L2, AdamW decay for dense weights, and decay LR on plateau/epoch. The compound package aims to keep validation ranking alive past epoch 2-3.
- treats: overfit
- reference_primary: none
- preconditions: Learning curves peak early then fall. Do not repeat the single-dose dropout 0.15 or AdamW 1e-4 variants; make a coherent aggressive package and select on GAUC.
- citation: `research/models-losses-hparams.md` section 4; MENU CURRENT DIRECTIVE
- expected_gain / cost: Plausible +0.002-0.008 if it changes peak epoch; single-dose forms were flat around 0.604-0.605 / low.
- status_pure: measured-win as package component (joint dose in champion; single-dose variants measured flat)
- status_1k: untried

### recency-weighting: Exponential training recency weighting
- mechanism: Weight training rows by age relative to the validation boundary so recent behavior dominates stale impressions. Keep evaluation and sampling unchanged.
- treats: data-shift
- reference_primary: none
- preconditions: Temporal split and trustworthy dates are available; normalize weights and retain enough effective sample size.
- citation: standard temporal covariate-shift weighting; KuaiRand date-split research notes
- expected_gain / cost: Expected small but robust shift correction, around +0.001-0.004 / low.
- status_pure: measured-win as package component (7d half-life in champion; single-dose sub-eps, e.g. night_e +0.0007)
- status_1k: measured-dead (half-life 3 scored 0.6120 and half-life 14 scored 0.6125; both hurt versus seed-42 frozen half-life 7 at 0.6134)

### package-dial-sweep: Literature package with internal dial search
- mechanism: Implement the full capacity+regularization package (dcn-lite + bpr-hybrid + regularization-schedule + recency-weighting) as ONE node whose script runs a TWO-STAGE internal search, then trains the final model. Stage 1 (coarse): 8-12 short probes over wide ranges — dropout {0.15..0.4}, weight decay {3e-5..3e-3 log-spaced}, LR step-decay variants, recency half-life {3.5, 7, 14}. Stage 2 (refine): 6-10 probes on a DENSER grid centered on stage-1's winner, at longer probe length (4-6 epochs, full rows). Choose values yourself; do not just copy example numbers. Then ONE full-length training with the winning dials, checkpointing EVERY HALF-EPOCH and keeping the validation-best snapshot. When the run clock allows (timeout permitting), close by rank-averaging 5 consecutive seeds of the final config inside the same node or as the follow-up node. Use the wall-clock: probe time is cheap relative to the 6h ceiling; log every probe's config and score in metrics.json history.
- kind: opportunity
- reference_primary: none
- treats: overfit | underfit
- preconditions: Use the npz fast path; keep probes short; the final training must be full-length. The sweep is internal to the node — one iteration, one artifact.
- citation: standard hyperparameter search practice (random/grid search, Bergstra & Bengio JMLR 2012); package composition per DCNv2/BPR training setups.
- expected_gain / cost: package at tuned dials measured 0.6047 +/- 0.0003; untuned dials measured 0.595-0.602 — the sweep is what closes that gap / medium-high runtime (one node).
- status_pure: measured-win (bigclock_07 n3 0.60424; novel_l1 n2 0.60387; the standard strong opener)
- status_1k: untried



### stage-matrix-sweep: Cross-stage combination search
- mechanism: ONE node probes COMBINATIONS ACROSS PIPELINE STAGES rather than dials of one recipe: pick 2-3 options per stage — architecture {FM, dcn-lite}, loss {logloss, bpr-hybrid}, weighting {uniform, recency-7d}, regularization {mild, strong} — and probe the cross-product (or a fractional subset of ~12-16 cells) with short trainings; identify which stage choices matter and which combination wins; refine the winner (denser dials or longer probes); train it full-length with half-epoch checkpointing. Log the full matrix of probe scores in metrics.json history — the matrix itself is evidence about interaction structure.
- kind: opportunity
- reference_primary: none
- treats: underfit | overfit | flat-signal
- preconditions: Fractional designs are fine when the full product exceeds the time budget; keep probes comparable (same epochs/rows). Final training full-length.
- citation: factorial experiment design (Fisher; standard DOE practice); ablation-study methodology.
- expected_gain / cost: subsumes package-dial-sweep with broader coverage; measured best known combination scores 0.6047 +/- 0.0003 at tuned dials / very high runtime (one node; use a long timeout).
- status_pure: measured-win (qb_b n1 0.60466; final_f1 n1 0.60403; high variance across draws)
- status_1k: untried

### combo-sweep: Add-on combination search on the tuned package
- mechanism: Given an accepted tuned package champion, ONE node probes which add-ons belong on it: short trainings (3-5 epochs) of {champion alone, +ordinal-watch-ratio aux, +duration-regime-heads, +CWM-censored aux, +recency variant}, and promising PAIRS of the individually-best add-ons; select on validation; train the winner full-length with half-epoch checkpointing. Log every probe combo + score in metrics.json history. This answers "which mechanisms compound" inside one iteration instead of spending a strike per add-on.
- kind: opportunity
- reference_primary: none
- treats: overfit | flat-signal
- preconditions: Do NOT play this immediately after an eps-clearing package/dial accept — a second same-family sweep re-searches conquered ground; COMPOUND with a different-family opportunity or a close instead (strategy layer). Parent must be an accepted tuned package (not the raw baseline). Budget probes to the timeout; final training full-length.
- citation: ablation-study methodology (standard practice); ESMM multi-task aux framing (Ma et al., SIGIR 2018); duration-bias line (D2Q KDD 2022, CWM KDD 2024).
- expected_gain / cost: individually measured add-ons range +0.0008..+0.0015 on suitable parents; compounding unknown — that is what this node measures / high runtime (one node).
- status_pure: measured-mixed (novel_r1 n1 +0.0005 sub-eps; 1k_push n6 catastrophic on 1K; use narrowly)
- status_1k: untried

### ensemble-design-sweep: Ensemble configuration search at close
- kind: opportunity
- reference_primary: 0.605575 (run_bigclock_07 n6 close, +0.0013)
- GATE (measured + reviewed): seed-ensembling pays only when members are competent
  AND usefully diverse. Before accepting any ensemble: (1) drop any member >0.0010
  primary below the median member or with anomalous tie/quantization rates;
  (2) compute rescue-vs-harm on validation pairs (consensus corrects anchor-wrong
  pairs = rescue; overturns anchor-right pairs = harm), weighted per GAUC user
  weighting; require rescue/harm > 1.2 and positive net rescue; (3) prefer a
  best-anchored soft combination (anchor weight ~0.6 to the best member, soft
  pairwise votes with per-member margin temperature, exact ties = 0.5 votes) over
  raw hard rank averaging — 5-item slates make hard ranks brittle (measured: 4 of
  5 hard-rank closes on strong singles SUBTRACTED). Never select members by the
  candidate ensembles' own validation scores.
- mechanism: The closing ensemble node probes its own design instead of assuming it: member count {3,5,7}, combination rule {per-user rank average, probability average}, optionally member diversity (consecutive seeds vs seeds+dial-jitter). Short-probe the options where affordable, pick on validation, produce the final ensemble. Log all probed designs.
- treats: overfit
- preconditions: Apply to the best accepted single-model champion. This is the canonical last node of a run.
- citation: Deep Ensembles (Lakshminarayanan et al., NeurIPS 2017); rank aggregation practice.
- expected_gain / cost: +0.0004..+0.0015 depending on parent seed variance / medium-high runtime.
- status_pure: measured-win (bigclock_07 n6 close +0.0013 -> 0.605575 CHAMPION; final_s2 hetero variant +0.0010)
- status_1k: untried


### gauge-fixed-bce: User-centered BCE (within-user gauge fixing)  [MEASURED WIN]
- mechanism: Batch complete user slates; replace the pointwise BCE logits with user-centered logits (logit minus that user's batch-mean logit, plus one learned global bias). Gradient of the pointwise term then sums to zero within each user, so it can only learn relative deviations — the only thing GAUC/nDCG measure. Keep the BPR term unchanged.
- reference_primary: none
- treats: metric-mismatch
- citation: gauge-fixing rationale (per-user constant shifts leave the metric invariant); pairwise-consistency literature.
- expected_gain / cost: MEASURED +0.0026 over a 0.6018-family parent (run_novel_r1
  node_003, accepted 0.60447); very cheap / low.
- status_pure: measured-win (novel_r1)
- status_1k: untried


### temporal-pair-kernel: Temporally-local pair sampling
- mechanism: Keep 1 negative per positive, but draw 70% of negatives with probability proportional to exp(-|day_pos - day_neg|/2) (fallback uniform when no opposite label within 3 days); 30% uniform. Pair weight = sqrt(w_pos * w_neg). Redraw each epoch. Changes WHICH comparisons constrain the model, not their scalar weights — orthogonal to recency row-weighting.
- reference_primary: none
- treats: data-shift
- citation: temporal-drift rationale; distinct from measured recency-weighting (row weights).
- expected_gain / cost: +0.0008-0.0014 measured (run_novel_l1 n4: 0.60387->0.60524) / low.
- status_pure: measured-win (run_novel_l1)
- status_1k: untried




### decayed-positive-sampling: Recency-decayed positive-count user sampling
- mechanism: Sample WHICH user receives each BPR update proportionally to (sum over their positives of 2^(-days_ago/3))^0.5, then draw one observed pos + one neg from that user. Aligns training attention with positive-count-weighted GAUC and recency; changes gradient ALLOCATION, complementary to row recency weighting. Fixed dials (h=3d, alpha=0.5); no grid.
- reference_primary: none
- treats: metric-mismatch | data-shift
- citation: idea salvaged from a public entry whose own scores are leakage-invalid (nigelyeap; the SAMPLER is train-only and clean); external review-ranked.
- expected_gain / cost: 0..+0.0005 / low.
- status_pure: measured-win (run_qb_b n1 package: 0.60466)
- status_1k: untried


### seed-ensemble: Seed ensemble of the champion configuration
- mechanism: Cancel variance across random initializations by training the champion configuration at several consecutive seeds and per-user rank-averaging their validation predictions.
- treats: flat-signal
- reference_primary: none (ensemble level tracks its parent; no fixed reference)
- preconditions: A champion config must already exist and all promising single-model moves must be exhausted. This is a CLOSING move, not an opening move.
- citation: Deep ensembles literature (Lakshminarayanan et al. 2017)
- expected_gain / cost: BASE-DEPENDENT (measured 1 Sep): +0.0014 off a 0.6042 single (bigclock n6 -> 0.6056) but only +0.0002 off a 0.6051 heavily-tuned single (f9 5-seed exhibit -> 0.6053) — the gain shrinks as the single approaches the 0.6055-0.6060 ceiling because tuning already banked the variance profit. Five member fits cost ~5x one champion run / high training, low implementation.
- status_pure: measured-alive
- status_1k: measured-alive (0.6323)

Proposer invocation pattern: generate a node that execs `zoo/ensemble_node.py` and passes the champion's exact winning CLI configuration as one quoted `--member-args` string, for example `uv run python zoo/ensemble_node.py --data-dir <d> --out-dir <o> --seed 42 --member-script zoo/polish_stack.py --member-args "--lr 0.0007 --dropout 0.2 --k 16" --n-members N --member-epochs 8`.

- choosing N: N is YOUR decision, not a default. More members cancel more seed variance but cost linearly more wall-clock (a scored resource); returns diminish beyond ~5. Reason from your observed seed spread and remaining time budget, and state the choice in your hypothesis.
- member diversity: At run time, reason about whether seed diversity is enough or whether to write a custom node instead of calling `zoo/ensemble_node.py`. A custom node may train members with deliberately varied configurations around the champion (for example, different dropout, learning rate, or half-life choices) and rank-average their outputs. Configuration diversity can cancel correlated errors that seed diversity alone cannot, but one bad or outlier member can drag down the whole committee, so keep variations modest. This is an agent decision, not a prescribed range.




## Lower-confidence cards (selectable; screen at ONE seed before investing)
Skepticism on record: top-tail-rider may mis-model our slates (validation has ~5 impressions/user, so nDCG@5 is full-slate ordering, not top-of-many); full-slate-gauc-loss contradicts the measured "additional negatives hurt" evidence.




## Measured-dead archive (NOT selectable — verdicts preserved for the record)

#### [dead] listwise-softmax: Per-user listwise softmax
- mechanism: Apply a temperature-scaled softmax across every impression for one user and cross-entropy toward that user's positive mass. Hybridize with BCE only as specified by the measured implementation.
- treats: metric-mismatch | flat-signal
- reference_primary: 0.5991
- preconditions: Full user histories must remain intact inside batches and one-class groups must be skipped for the listwise term.
- citation: ListNet; PSL (arXiv:2411.00163); sampled-softmax work (arXiv:2201.02327); run_real_04 node_003
- expected_gain / cost: Expected parity with BPR, but measured primary 0.5991 versus FM baseline 0.6018 / low-medium.
- status_pure: measured-dead (0.5991 primary, run 04)
- status_1k: untried



#### [dead] item-aggregates: Train-window item/author target aggregates
- mechanism: Add Bayesian-smoothed video/author long_view rates computed only on training dates. Item-varying rates can affect within-user order, unlike user-only rates.
- treats: flat-signal | data-shift
- reference_primary: 0.6038
- preconditions: Strict train-window computation with no full-period statistic files or validation leakage.
- citation: standard target encoding; `zoo/EXPERIMENTS.md` E4
- expected_gain / cost: Measured 0.6038 versus 0.6047 without aggregates / low.
- status_pure: measured-dead (0.6038 primary; Pure-only popularity-prior blends also failed)
- status_1k: untried

#### [dead] content-features: Video content categorical features
- mechanism: Add video_type, upload_type, frequent music ID, and first-tag categorical embeddings to provide cold-item semantics beyond video ID.
- treats: flat-signal | data-shift
- reference_primary: 0.6039
- preconditions: Features must be impression-time legal and available for validation; avoid full-period counters.
- citation: KuaiRand feature schema; `zoo/EXPERIMENTS.md` E6
- expected_gain / cost: Measured 0.6039 versus 0.6048 without content / medium.
- status_pure: measured-dead (0.6039 primary)
- status_1k: untried

#### [dead] lightgbm-lambdarank: LightGBM LambdaRank and NN blend
- mechanism: Train user-grouped LambdaRank on legal categorical and train-only aggregate features, then rank-average it with the neural model. Lambda gradients target nDCG ordering directly.
- treats: metric-mismatch | flat-signal
- reference_primary: 0.5974
- preconditions: Requires LightGBM plus careful grouped sorting and leakage-safe engineered features; unavailable under the generated-script numpy/torch-only contract.
- citation: LightGBM LambdaRank documentation; `zoo/EXPERIMENTS.md` E8
- expected_gain / cost: Alone measured 0.5974; all tested NN blends were worse (best 0.6034) / medium.
- status_pure: measured-dead (0.5974 primary alone; 0.6034 best blend)
- status_1k: untried


### heterogeneous-ensemble-design: Validation-selected cross-mechanism ensemble
- mechanism: Train members under DIFFERENT mechanisms (e.g. temporal-pair-kernel, gauge-fixed-bce, decayed-positive, frozen regularized stack) rather than jittered copies of one recipe; validation-select the member subset and aggregation (rank vs probability average, optional per-member weights) before scoring. Diversity across mechanisms is the untested axis — jittered same-recipe closes are measured at +0.0013.
- kind: opportunity
- treats: variance | plateau
- reference_primary: 0.605938 (direct evidence probe: EQUAL-WEIGHT rank blend of one pair-kernel member 0.60515 + two composite seeds 0.6044-0.6047 = 0.605938; champion-ens + one composite seed = 0.605886; many cross-family combos cluster 0.6058-0.6059 — decorrelated families are where closes pay)
- preconditions: At least 2 mechanism families measured above 0.6040 in this run's lineage AS EXISTING ARTIFACTS (measured 1 Sep: in-run fresh-member versions scored only 0.6041/0.6046 — f7 n4, f9 n4; the 0.6059 reference uses polished cross-run artifacts); select on validation only.
- citation: run_bigclock_07 n6 (jitter close +0.0013); evidence/blend_audit.md (caveat)
- expected_gain / cost: +0.0005-0.0020 primary / medium (several member trainings).
- status_pure: measured-win (run_final_s2 n4 +0.0010; cross-family blend probe 0.605828 evidence) — the endgame play: build members from BOTH the DCN package and seq-deepfm families, then rank-aggregate across families
- status_1k: untried

### context-stratified-pairs: Same-context BPR negative stratification
- kind: opportunity
- mechanism: Draw a fraction of BPR negatives from the SAME (user, date, hour) or (user, date, tab) context as the positive (~30% same-context on Pure — sparser sessions than 1K; fall back same-day, then uniform). Pairs the model must separate are the ones the metric actually scores: contemporaneous impressions in one slate. Distinct from temporal-pair-kernel (soft day-distance kernel); this is hard context stratification.
- treats: ranking-mismatch | data-shift
- reference_primary: none on Pure
- preconditions: Contexts with no opposite-label row need a fallback tier; keep total negatives per positive unchanged.
- citation: convergent recommendation of two independent reviews (gpt-5.6-sol consult; external playbook §4.2), both from this campaign's own journals.
- expected_gain / cost: +0.0005-0.0015 est / low.
- status_pure: measured-win (run_final_s4 n3: 0.60521, +0.0015 from 0.6038 gauge base — largest single-mechanism gain of the final wave; NO ensemble close attempted on this base yet — highest-priority follow-up)
- status_1k: untried

### diverse-family-farm-close: One-node multi-family member farm + cross-family blend
- kind: opportunity
- note: STRATEGY BUNDLE (composes package-dial-sweep + temporal-pair-kernel + seq-deepfm-composite + recency-weighting + rank aggregation). Cards are normally atomic methods; bundles are the pragmatic vehicle for strategies until the v3 strategy layer.
- mechanism: ONE node, in three phases: (PHASE 1, ~30 min) train ONE probe-fidelity member (1-2 epochs) per candidate family, then evaluate ALL blend combinations of the probe members on validation — blending is nearly free (rank-average of saved score vectors) — to map which families COMPLEMENT; (PHASE 2) full-train only the 2-3 complementary families; (PHASE 3) blend the full members, RE-VERIFY the winning combination at full fidelity (probe-level correlations are an assumption, not a guarantee), emit. Original single-phase form: ONE node that reproduces the campaign's measured cross-family evidence internally: train ONE member from EACH measured-win family, each per its own card's recipe — (a) the regularized DCN package (package-dial-sweep dials, ~0.6042), (b) temporal-pair-kernel on that package (~0.6045-52), (c) seq-deepfm-composite (~0.6044), (d) a recency-weighted FM/DCN variant (~0.6045-50). VALIDATE each member's primary individually (progress-log it; ADMIT only members >=0.6040), then per-user or global RANK-AVERAGE the admitted members. Cross-family decorrelation is the entire point: same-family seed ensembles measured +0.0003; cross-family equal-weight blends of exactly these families measured 0.6058-0.6065 (team evidence probe, 31 Aug).
- treats: variance | plateau
- reference_primary: 0.605863 (selection-free ALL-family equal blend of one member per clean run; best combos 0.6060-0.6065)
- verdict_pure: external-win
- evidence_primary: 0.605863
- preconditions: MEMBER SOURCE CALIBRATION (measured 1 Sep): the 0.6058-0.6065 reference evidence blends POLISHED artifacts that each took a full run to produce. Members freshly written inside one node measured 0.5975-0.6046 (f7 n3, f8 n3, f9 n3 all fell back to the incumbent) — first-draft fidelity tax 0.001-0.004. PREFER script_source members from THIS run's own trained nodes (a rejected sibling within ~0.002 of the champion is a finished, measured member); expect the reference-class gain ONLY when >=2 strong distinct artifacts already exist in the lineage. This is a CLOSING move, never an opener: its members must be derived from an already-established champion script (a strong single model measured in THIS run); members written cold from the baseline at iteration 1 are untested code and measured as broken (gauc below 0.5) or weak, and the executor then falls back to the incumbent, spending the iteration for nothing. Budget the node like a sweep (it is 4 trainings + blend): use most of the timeout; log every member's config+primary; obey the ensemble contract (distinct seeds, member-distinctness assertion, never emit parent-identical predictions). A member that fails to train is dropped, not blended.
- citation: team evidence probes 31 Aug (logs/RUNS.md recipe-search line); component recipes: package-dial-sweep, temporal-pair-kernel, seq-deepfm-composite, recency-weighting cards.
- expected_gain / cost: +0.0035-0.0045 over baseline IN ONE NODE (eps-clearing) if >=3 members admit; degrades gracefully to the best single member / high runtime (one node, plan 60-90 min).
- status_pure: untried as a single node (every component + the blend measured separately)
- status_1k: untried
