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
- SNAPSHOT ENSEMBLING (Huang et al., ICLR 2017) — USE IT in every ensemble close: with a cyclic or restarted LR,
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

## Depth policy (MODERATE — calibrated by measured dose-response)
Measured across ~30 runs: opener quality flattens past ~50-80 WELL-RANKED probes
(59 probes -> 0.60426; 133 -> 0.60436; 208 without full-fidelity finals -> 0.60238
FAIL; 600+ -> timeout/uncommitted). Therefore:
- Stage-1: 40-80 FULL-FIDELITY probes, half from the measured basins, half wide;
  STOP EARLY when the best-found has not improved over the last 15 probes.
- One refinement pass (~10-15 probes) around the winner; top 3-4 candidates get
  full-length final trainings; commit only on full-length scores.
- RESERVE the remainder of the node budget for the final training; and reserve a
  later iteration for the gated ensemble close. Never let a sweep consume the
  whole node budget (measured failure mode).

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
- probe scores in-basin: 0.6026-0.6044 single-model; out-of-basin drops fast.
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
- seed-ensemble (5 consecutive seeds, per-user rank average) on any champion: +0.0004..+0.001.
  On the full package champion: 0.6051. Ensemble of baseline: only ~0.6028.
- duration-regime-heads on baseline: +0.0014 (agent-discovered, replicated).
- Measured DEAD on this benchmark (do not re-try in measured form): extra feature
  fields (all variants), sequence/history models, watch-time regression losses,
  larger embeddings (k>16 on Pure), deeper crosses, SWA/EMA, listwise softmax.
- The full winning stack (package + half-epoch checkpoint + 5-seed ensemble) = 0.6051;
  a from-scratch run must bundle the package in ONE node to have iteration budget left
  for checkpointing + ensemble.

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
- status_pure: running-elsewhere (frozen known-best stack; pure BCE 0.6038 and pure BPR 0.6036 are measured dead)
- status_1k: untried

### dndcg-lambda: Delta-nDCG lambda weighting for top-5 groups
- mechanism: Weight within-user positive/negative pair gradients by the absolute nDCG@5 change caused by swapping the pair. Build batches as complete groups sized like the evaluator's top-5 lists.
- treats: metric-mismatch | flat-signal
- reference_primary: none
- preconditions: A correct grouped BPR/listwise implementation and stable position/discount computation are available; binary groups must contain both labels.
- citation: LambdaRank/LambdaLoss literature; `research/models-losses-hparams.md` section 2
- expected_gain / cost: Likely +0.000-0.003 primary because five-item binary lists leave little extra discrimination / low-medium.
- status_pure: running-elsewhere (C3 campaign in CURRENT DIRECTIVE dead-list ledger)
- status_1k: untried


### dcn-lite: DCNv2-lite interaction head
- mechanism: Add one or two explicit cross layers and a small MLP over the five field embeddings. This supplies bounded higher-order interactions without the parameter cost of xDeepFM/AutoInt.
- treats: underfit | flat-signal
- reference_primary: 0.6039
- preconditions: Use k=16, hidden around 128, GAUC early stopping, and the known-best hybrid loss; more depth is within noise and overfits.
- citation: DCNv2; FuxiCTR/BARS; `zoo/EXPERIMENTS.md` E1-E3
- expected_gain / cost: Accepted stack primary 0.6039 +/- 0.0010, +0.0023 over official baseline / low.
- status_pure: running-elsewhere (core known-best architecture)
- status_1k: untried

### finalmlp: FinalMLP two-stream fusion
- mechanism: Feed embeddings through two gated MLP streams and combine them with a bilinear fusion head. The streams can capture complementary feature interactions while keeping the network compact.
- treats: underfit | flat-signal
- reference_primary: none
- preconditions: Only try from a regularized k=16 parent after objective changes; compare multiple seeds because expected architecture deltas match seed noise.
- citation: FinalMLP, AAAI 2023 (arXiv:2304.00902); FuxiCTR/BARS
- expected_gain / cost: Estimated +0.001-0.005 over DCN-lite with high uncertainty / medium.
- status_pure: running-elsewhere (C4 formal closure campaign)
- status_1k: untried

### mtl-shared-bottom: Shared-bottom multi-task heads
- mechanism: Share embeddings/trunk across long_view and a few dense auxiliary outcomes, with small task heads and total auxiliary weight 0.1. Auxiliaries regularize sparse ID representations through extra training-only gradients.
- treats: underfit | overfit | flat-signal
- reference_primary: 0.6039
- preconditions: click/effective_view/like are targets only, never validation inputs; use 2-4 correlated tasks and guard against seesaw effects.
- citation: shared-bottom/ESMM literature; HoME (arXiv:2408.05430); `zoo/EXPERIMENTS.md` E3
- expected_gain / cost: Aux 0.1 tied but edged no-aux, 0.6039 versus 0.6038 three-seed mean / low.
- status_pure: running-elsewhere (kept in known-best stack)
- status_1k: untried



### regularization-schedule: Compound dropout, row-L2, weight decay, and LR decay
- mechanism: Apply MLP dropout around 0.3, accessed-row embedding L2, AdamW decay for dense weights, and decay LR on plateau/epoch. The compound package aims to keep validation ranking alive past epoch 2-3.
- treats: overfit
- reference_primary: none
- preconditions: Learning curves peak early then fall. Do not repeat the single-dose dropout 0.15 or AdamW 1e-4 variants; make a coherent aggressive package and select on GAUC.
- citation: `research/models-losses-hparams.md` section 4; MENU CURRENT DIRECTIVE
- expected_gain / cost: Plausible +0.002-0.008 if it changes peak epoch; single-dose forms were flat around 0.604-0.605 / low.
- status_pure: running-elsewhere (C2 joint grid; single-dose variants measured dead)
- status_1k: untried

### swa-ema: SWA or EMA checkpoint averaging
- mechanism: Maintain an exponential or stochastic average of weights across late checkpoints, then score the averaged model. Averaging reduces optimizer/seed variance without adding inference models.
- treats: overfit | data-shift
- reference_primary: none
- preconditions: Training must produce several useful near-peak checkpoints; do not average far past a sharply collapsing validation curve.
- citation: SWA (Izmailov et al.); EMA training practice; `research/models-losses-hparams.md` seed-variance guidance
- expected_gain / cost: Estimated +0.000-0.003 primary, mainly variance reduction / low.
- status_pure: running-elsewhere (C2 schedule campaign)
- status_1k: untried

### embedding-dim-down: Reduce embedding dimension to k=8
- mechanism: Halve ID embedding capacity from k=16 to k=8, reducing memorization in sparse user/item rows and forcing more shared signal. Keep the loss and head unchanged for a clean capacity test.
- treats: overfit
- reference_primary: none
- preconditions: Curves peak early and k=32 is already worse; use identical seeds and early stopping because capacity effects are small.
- citation: `research/models-losses-hparams.md` section 4; MENU item 6 and E2
- expected_gain / cost: k=32 scored 0.6039 versus k=16 0.6047; k=8 may recover <=0.003 through regularization / low.
- status_pure: running-elsewhere (C2 dimension sweep)
- status_1k: untried

### duration-regime-heads: Short/long duration regime heads
- mechanism: Route examples through separate prediction heads for videos below versus above the 18-second long_view threshold regime, while sharing embeddings and trunk. This lets ranking functions differ where label censoring changes.
- treats: metric-mismatch | data-shift | underfit
- reference_primary: none
- preconditions: Both regimes have enough examples and the route uses impression-known duration only; regularize heads toward their shared parent.
- citation: D2Q duration debiasing, KDD 2022 (arXiv:2206.06003); KuaiRand research notes
- expected_gain / cost: Estimated +0.002-0.006 if duration bias is material / low-medium.
- status_pure: running-elsewhere (C3 campaign)
- status_1k: untried

### user-metadata-crosses: Coarse user-metadata by item/context crosses
- mechanism: Cross stable coarse user attributes with item/author/context IDs so features vary across candidates within a user. Back off rare crosses to avoid exploding sparse capacity.
- treats: flat-signal | data-shift
- reference_primary: none
- preconditions: Legal user metadata must exist at inference and each cross must include item-side variation; user-constant features alone cannot change GAUC/nDCG.
- citation: standard sparse recommender feature crossing; GAUC invariance analysis in `research/models-losses-hparams.md`
- expected_gain / cost: Estimated +0.001-0.005 with high sparsity risk / medium.
- status_pure: running-elsewhere (C3 campaign)
- status_1k: untried

### session-time-features: Session and fine time context
- mechanism: Derive causal session position/gap features from impression timestamps and add hour/day context crosses. These represent transient intent and distribution changes that static IDs miss.
- treats: data-shift | flat-signal
- reference_primary: none
- preconditions: Compute using current/past impressions only, preserve row order, and never use future session events; hour/day basics may already be in the parent.
- citation: KuaiRand schema/research notes; MENU temporal-context item 9
- expected_gain / cost: Basic hour/day is bundled in the winning stack; incremental session features estimated +0.001-0.005 / medium.
- status_pure: running-elsewhere (C3 campaign)
- status_1k: untried

### recency-weighting: Exponential training recency weighting
- mechanism: Weight training rows by age relative to the validation boundary so recent behavior dominates stale impressions. Keep evaluation and sampling unchanged.
- treats: data-shift
- reference_primary: none
- preconditions: Temporal split and trustworthy dates are available; normalize weights and retain enough effective sample size.
- citation: standard temporal covariate-shift weighting; KuaiRand date-split research notes
- expected_gain / cost: Expected small but robust shift correction, around +0.001-0.004 / low.
- status_pure: running-elsewhere (C1 historical campaign)
- status_1k: measured-dead (half-life 3 scored 0.6120 and half-life 14 scored 0.6125; both hurt versus seed-42 frozen half-life 7 at 0.6134)

### covisit-svd-init: Item co-visitation SVD initialization
- mechanism: Build a train-only user-item incidence matrix, factor its item co-visitation Gram matrix, and initialize item embeddings from the top singular vectors. Fine-tune normally afterward.
- treats: underfit | flat-signal
- reference_primary: none
- preconditions: Co-visitation uses only training impressions and item embedding dimensions align; useful mainly when random embeddings learn too slowly.
- citation: classical item-item collaborative filtering and spectral initialization
- expected_gain / cost: Expected +0.000-0.004; likely only accelerates early learning / medium.
- status_pure: running-elsewhere (C1 historical campaign)
- status_1k: untried

### seed-architecture-ensemble: Seed and diverse-architecture rank ensemble
- mechanism: Train the winning configuration across 3-5 seeds and optionally a genuinely distinct architecture, then average scores or per-user ranks. Diversity reduces variance and uncorrelated ranking errors.
- treats: data-shift | flat-signal
- reference_primary: 0.6047
- preconditions: Final-stage only; component models must be individually competitive and predictions aligned row-for-row.
- citation: `research/models-losses-hparams.md` section 4; `zoo/EXPERIMENTS.md` E7
- expected_gain / cost: Five-seed rank average scored 0.6047 versus seed mean 0.6039, a variance reducer rather than level gain / high training, low implementation.
- status_pure: running-elsewhere (C2 architecture ensemble; seed ensemble measured)
- status_1k: untried (three frozen-default seeds measured individually; no 1K prediction ensemble measured)


### package-dial-sweep: Literature package with internal dial search
- mechanism: Implement the full capacity+regularization package (dcn-lite + bpr-hybrid + regularization-schedule + recency-weighting) as ONE node whose script runs a TWO-STAGE internal search, then trains the final model. Stage 1 (coarse): 8-12 short probes over wide ranges — dropout {0.15..0.4}, weight decay {3e-5..3e-3 log-spaced}, LR step-decay variants, recency half-life {3.5, 7, 14}. Stage 2 (refine): 6-10 probes on a DENSER grid centered on stage-1's winner, at longer probe length (4-6 epochs, full rows). Choose values yourself; do not just copy example numbers. Then ONE full-length training with the winning dials, checkpointing EVERY HALF-EPOCH and keeping the validation-best snapshot. When the run clock allows (timeout permitting), close by rank-averaging 5 consecutive seeds of the final config inside the same node or as the follow-up node. Use the wall-clock: probe time is cheap relative to the 6h ceiling; log every probe's config and score in metrics.json history.
- treats: overfit | underfit
- preconditions: Use the npz fast path; keep probes short; the final training must be full-length. The sweep is internal to the node — one iteration, one artifact.
- citation: standard hyperparameter search practice (random/grid search, Bergstra & Bengio JMLR 2012); package composition per DCNv2/BPR training setups.
- expected_gain / cost: package at tuned dials measured 0.6047 +/- 0.0003; untuned dials measured 0.595-0.602 — the sweep is what closes that gap / medium-high runtime (one node).
- status_pure: untried
- status_1k: untried



### stage-matrix-sweep: Cross-stage combination search
- mechanism: ONE node probes COMBINATIONS ACROSS PIPELINE STAGES rather than dials of one recipe: pick 2-3 options per stage — architecture {FM, dcn-lite}, loss {logloss, bpr-hybrid}, weighting {uniform, recency-7d}, regularization {mild, strong} — and probe the cross-product (or a fractional subset of ~12-16 cells) with short trainings; identify which stage choices matter and which combination wins; refine the winner (denser dials or longer probes); train it full-length with half-epoch checkpointing. Log the full matrix of probe scores in metrics.json history — the matrix itself is evidence about interaction structure.
- treats: underfit | overfit | flat-signal
- preconditions: Fractional designs are fine when the full product exceeds the time budget; keep probes comparable (same epochs/rows). Final training full-length.
- citation: factorial experiment design (Fisher; standard DOE practice); ablation-study methodology.
- expected_gain / cost: subsumes package-dial-sweep with broader coverage; measured best known combination scores 0.6047 +/- 0.0003 at tuned dials / very high runtime (one node; use a long timeout).
- status_pure: untried
- status_1k: untried

### combo-sweep: Add-on combination search on the tuned package
- mechanism: Given an accepted tuned package champion, ONE node probes which add-ons belong on it: short trainings (3-5 epochs) of {champion alone, +ordinal-watch-ratio aux, +duration-regime-heads, +CWM-censored aux, +recency variant}, and promising PAIRS of the individually-best add-ons; select on validation; train the winner full-length with half-epoch checkpointing. Log every probe combo + score in metrics.json history. This answers "which mechanisms compound" inside one iteration instead of spending a strike per add-on.
- treats: overfit | flat-signal
- preconditions: Parent must be an accepted tuned package (not the raw baseline). Budget probes to the timeout; final training full-length.
- citation: ablation-study methodology (standard practice); ESMM multi-task aux framing (Ma et al., SIGIR 2018); duration-bias line (D2Q KDD 2022, CWM KDD 2024).
- expected_gain / cost: individually measured add-ons range +0.0008..+0.0015 on suitable parents; compounding unknown — that is what this node measures / high runtime (one node).
- status_pure: untried
- status_1k: untried

### ensemble-design-sweep: Ensemble configuration search at close
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
- status_pure: untried
- status_1k: untried


### swa-then-ensemble: Per-member weight averaging before seed ensembling
- mechanism: For each ensemble member, average the weights of the last N half-epoch checkpoints (SWA; or a constant-LR tail) to flatten the member, THEN per-user rank-average across seed members. Distinct from the measured-dead plain EMA/SWA single-model attempt: here averaging is per-member inside an ensemble.
- treats: overfit
- citation: Izmailov et al. 2018 (SWA); Wortsman et al., Model Soups, ICML 2022.
- expected_gain / cost: +0.0003..+0.001 on AUC-scale metrics in literature; unknown here / low.
- status_pure: untried
- status_1k: untried

### freq-adaptive-reg: Frequency-adaptive embedding regularization / SSE
- mechanism: Scale embedding weight decay inversely with ID frequency (rare user/author rows get stronger decay), or apply Stochastic Shared Embeddings (randomly swap embedding indices with small probability) on top of the package regularization.
- treats: overfit
- citation: Wu et al., Stochastic Shared Embeddings, NeurIPS 2019; adaptive sparse-embedding regularization literature.
- expected_gain / cost: consistent small gains over uniform reg in papers; unknown here / low-med.
- status_pure: untried
- status_1k: untried

### group-hard-pair-bpr: User-grouped hard-pair mining for the BPR term
- mechanism: Batch by user and weight the hardest (max-violation) positive/negative pairs within each user group in the BPR loss, instead of uniform sampled pairs — aligns training pressure with GAUC exactly.
- treats: metric-mismatch | flat-signal
- citation: PDAOM (differentiable Group-AUC optimization); hierarchical group-wise ranking literature.
- expected_gain / cost: GAUC gains reported over plain pairwise; likely small here / med.
- status_pure: untried
- status_1k: untried

### adversarial-recency: Adversarial-validation importance weights instead of fixed half-life
- mechanism: Train a small classifier to distinguish early-train vs late-train days (proxy for the test shift) using only train rows; use its per-row probability as importance weights in place of the hand-tuned exponential decay. Learns the shift shape instead of assuming it.
- treats: data-shift
- citation: adversarial validation practice (e.g. Lightweight Boosting with Adversarial Validation, 2023).
- expected_gain / cost: replaces a hand dial with a learned one; unknown, plausibly +-0.0005 / low.
- status_pure: untried
- status_1k: untried

### gbdt-diversity-member: LightGBM/CatBoost member for ensemble diversity
- mechanism: Train a GBDT on train-only encodings of the 5 IDs (frequency counts, train-window rates per user/author/tab-duration) and add it as ONE member of the rank-average ensemble. Alone it will likely be weaker than the neural champion; the play is decorrelated errors. All encodings computed on the train window only (no leakage; no external data).
- treats: flat-signal
- citation: RecSys Challenge 2024 winning ensembles (NN + LightGBM + CatBoost); MLWave ensembling guide. NOTE: gbdt-lambdarank alone was measured weak here — this card is ONLY the diversity-member use.
- expected_gain / cost: high-variance; frequently the biggest lever in challenge ensembles / med (needs lightgbm dependency — use sklearn GradientBoosting or pure-torch trees if lightgbm unavailable; stdlib+numpy+torch constraint applies).
- status_pure: untried
- status_1k: untried


### gauge-fixed-bce: User-centered BCE (within-user gauge fixing)  [MEASURED WIN]
- mechanism: Batch complete user slates; replace the pointwise BCE logits with user-centered logits (logit minus that user's batch-mean logit, plus one learned global bias). Gradient of the pointwise term then sums to zero within each user, so it can only learn relative deviations — the only thing GAUC/nDCG measure. Keep the BPR term unchanged.
- treats: metric-mismatch
- citation: gauge-fixing rationale (per-user constant shifts leave the metric invariant); pairwise-consistency literature.
- expected_gain / cost: MEASURED +0.0026 over a 0.6018-family parent (run_novel_r1
  node_003, accepted 0.60447); very cheap / low.
- status_pure: measured-win (novel_r1)
- status_1k: untried

### signed-sketch-residual: Signed co-consumption sketch rank blend
- mechanism: Compute recency-weighted per-user long-view residuals r_ui = sqrt(w)(y - user_mean); give each user a fixed 64-dim Rademacher hash vector; video sketch z_i = normalize(sum_u r_ui * h_u); user taste p_u = normalize(sum_j r_uj * z_j); graph score = p_u . z_i with self-contribution removed. Blend WITHIN-USER RANKS: final = rank(champion) + alpha * rank(graph), alpha in {0.05,0.1,0.2} chosen on a train-only rolling holdout. Numpy index_add over a [7600,64] array — minutes of compute.
- treats: flat-signal
- citation: signed-feedback CF + lightweight graph propagation (LightGCN lineage), compressed to sketches; NOT covered by the measured-dead co-visitation SVD INIT (this is a separate scorer blended at rank level, not an initialization).
- expected_gain / cost: +0.0003..+0.0012 if errors decorrelate; possibly flat / low-med.
- status_pure: untried
- status_1k: untried

### temporal-pair-kernel: Temporally-local pair sampling
- mechanism: Keep 1 negative per positive, but draw 70% of negatives with probability proportional to exp(-|day_pos - day_neg|/2) (fallback uniform when no opposite label within 3 days); 30% uniform. Pair weight = sqrt(w_pos * w_neg). Redraw each epoch. Changes WHICH comparisons constrain the model, not their scalar weights — orthogonal to recency row-weighting.
- treats: data-shift
- citation: temporal-drift rationale; distinct from measured recency-weighting (row weights).
- expected_gain / cost: +0.0002..+0.0008 speculative / low.
- status_pure: untried
- status_1k: untried



### broad-to-recent-curriculum: Phase-scheduled recency
- mechanism: Same rows and model; phase the recency weighting — first half-epoch uniform, then one epoch at 7d half-life, final half-epoch at 3.5d with LR x0.3, optimizer not reset; keep half-epoch validation-best checkpointing so the broad checkpoint can still win.
- treats: data-shift
- citation: curriculum/fine-tune-on-recent practice for temporal splits.
- expected_gain / cost: +0.0001..+0.0007; the cheapest experiment in this family / low.
- status_pure: untried
- status_1k: untried


### listwise-regime: Listwise objective in its own training regime
- mechanism: Per-user listwise (softmax cross-entropy over each user's slate) objective, trained in the regime where a public competitor measured it working: HIGHER capacity (k~32), LOWER lr (~3e-4), LONG training (up to ~100 epochs) with validation patience ~10 instead of our rapid-decay early-peak schedule. The hypothesis: listwise loss changes the overfitting dynamics, so the "listwise is dead" verdict measured under our short-peak regime does not transfer. Probe as a PACKAGE (loss + capacity + schedule together).
- treats: metric-mismatch | underfit
- citation: public Track 2 solution github.com/vrisdng/tiktok-techjam (reports 0.6019 repro -> 0.6034 listwise -> 0.60417 tuned 5-seed, claimed 6-sigma); public solutions are explicitly in-scope per the brief's Resource policy. Our own listwise-softmax card is measured-dead UNDER OUR REGIME ONLY.
- expected_gain / cost: their measured +0.0016-0.0023 over baseline-repro in their regime; unknown whether it stacks with our package or gauge-fixed-bce / med (long trainings).
- status_pure: untried (in this regime)
- status_1k: untried


### social-mtl-heads: Multi-signal auxiliary heads (like/follow/comment/forward)
- mechanism: Add SEVERAL small auxiliary BCE heads at once on the shared representation — like, follow, comment, forward (and optionally click) — each at low weight (~0.05-0.1), targets from the training columns only (never inputs). Distinct from our measured single-aux cards: the hypothesis is that the BUNDLE of sparse social signals regularizes the shared embedding jointly where any one signal is too sparse to matter.
- treats: overfit | flat-signal
- citation: ESMM-style multi-task (Ma et al., SIGIR 2018; explicitly endorsed by the brief's appendix A.3); observed working in a public Track 2 solution (github.com/9irija/TikTok_TechJam: DeepFM + 4 social heads, +0.0030 primary, 3-seed verified) — public solutions are in-scope per the resource policy.
- expected_gain / cost: their +0.0030 total includes architecture change; the aux-bundle increment here is unknown — probe on our package / low-med. NOTE: our data export may lack like/follow/comment/forward columns — if unavailable in the npz, use available aux targets (click, play_time_ms-derived) as a reduced bundle and note the limitation.
- status_pure: untried
- status_1k: untried


### hetero-objective-ensemble: Rank-blend of DIFFERENT-objective models
- mechanism: Train 2-3 members that differ by OBJECTIVE, not just seed — e.g. (a) the champion package with its BPR-hybrid loss, (b) a gauge-fixed-bce variant, (c) a lambda-weighted pairwise variant — then per-user rank-average them (fixed simple weights like equal or 0.5/0.25/0.25; do NOT sweep weights on validation). Rationale: same-recipe seed members share errors (measured: our homogeneous closes averaged +0.0005 and failed 4/5 times); different objectives decorrelate errors so the blend genuinely cancels mistakes.
- treats: flat-signal | overfit
- citation: ensemble-diversity theory (Krogh & Vedelsby); observed working in a public Track 2 solution (github.com/Rpkw789/autorec-lab: BPR + multi-task + LambdaLoss rank ensemble, single 0.6042 -> ensemble 0.6060 valid); public solutions in-scope per the resource policy. Apply OUR ensemble gate (rescue/harm) before accepting.
- expected_gain / cost: their +0.0018 over best single; ours unknown / med (2-3 full trainings).
- status_pure: untried
- status_1k: untried

### lambda-weighted-pairs: nDCG-weighted pairwise loss (LambdaLoss-style)
- mechanism: Keep the BPR-style pairwise structure but weight each within-user pair by the |nDCG@5 change| that swapping the pair would cause at current ranks (LambdaLoss/LambdaRank weighting) — pairs that can move the top-5 get large gradients, bottom pairs get little. Metric-aligned variant of pairwise training; distinct from the measured-dead LightGBM-LambdaRank (different model class) and from listwise-softmax.
- treats: metric-mismatch
- citation: LambdaLoss framework (Wang et al., CIKM 2018); component of the public autorec-lab ensemble above.
- expected_gain / cost: unknown alone; their blend gives it 37.5% weight, suggesting standalone competence / low-med.
- status_pure: untried
- status_1k: untried


### decayed-positive-sampling: Recency-decayed positive-count user sampling
- mechanism: Sample WHICH user receives each BPR update proportionally to (sum over their positives of 2^(-days_ago/3))^0.5, then draw one observed pos + one neg from that user. Aligns training attention with positive-count-weighted GAUC and recency; changes gradient ALLOCATION, complementary to row recency weighting. Fixed dials (h=3d, alpha=0.5); no grid.
- treats: metric-mismatch | data-shift
- citation: idea salvaged from a public entry whose own scores are leakage-invalid (nigelyeap; the SAMPLER is train-only and clean); external review-ranked.
- expected_gain / cost: 0..+0.0005 / low.
- status_pure: untried
- status_1k: untried

### small-batch-diversity: Batch-size ensemble-diversity experiment
- mechanism: Freeze the champion config; train 3 fixed seeds at each of {current, half, quarter} batch size; compare member quality, pairwise correlation, and midrank-ensemble payoff. Target: similar member quality with disagreement concentrated on correctable pairs (NOT maximum disagreement). One controlled experiment, not a search.
- treats: flat-signal
- citation: public entry (OrangeCat) ablation suggests smaller batches -> more optimizer steps, better sparse-ID learning, more useful seed diversity; our ensemble payoff is config-dependent (measured).
- expected_gain / cost: 0..+0.0006 via ensemble / med.
- status_pure: untried
- status_1k: untried

### relative-watch-component: Relative-advantage watch percentile diversity member
- mechanism: Train-only target = 0.5*midrank_percentile(play_time_ms within video) + 0.5*midrank_percentile(play_time_ms within user x duration_bucket); train a compact model on the 5 IDs to predict it; use ONLY as a 5-10% rank-blend diversity member, never standalone. Distinct from measured-dead raw watch-time regression (relative target, blend-only role).
- treats: flat-signal
- citation: public entry (wecoai) uses this as a blend component; relative-percentile framing removes trivial duration scale.
- expected_gain / cost: probably flat, upside +0.0005 / low-med.
- status_pure: untried
- status_1k: untried

### seed-ensemble: Seed ensemble of the champion configuration
- mechanism: Cancel variance across random initializations by training the champion configuration at several consecutive seeds and per-user rank-averaging their validation predictions.
- treats: flat-signal
- reference_primary: none (ensemble level tracks its parent; no fixed reference)
- preconditions: A champion config must already exist and all promising single-model moves must be exhausted. This is a CLOSING move, not an opening move.
- citation: Deep ensembles literature (Lakshminarayanan et al. 2017)
- expected_gain / cost: Measured Pure primary 0.6058; five full member fits cost roughly five times one champion run / high training, low implementation.
- status_pure: measured-alive
- status_1k: measured-alive (0.6323)

Proposer invocation pattern: generate a node that execs `zoo/ensemble_node.py` and passes the champion's exact winning CLI configuration as one quoted `--member-args` string, for example `uv run python zoo/ensemble_node.py --data-dir <d> --out-dir <o> --seed 42 --member-script zoo/polish_stack.py --member-args "--lr 0.0007 --dropout 0.2 --k 16" --n-members N --member-epochs 8`.

- choosing N: N is YOUR decision, not a default. More members cancel more seed variance but cost linearly more wall-clock (a scored resource); returns diminish beyond ~5. Reason from your observed seed spread and remaining time budget, and state the choice in your hypothesis.
- member diversity: At run time, reason about whether seed diversity is enough or whether to write a custom node instead of calling `zoo/ensemble_node.py`. A custom node may train members with deliberately varied configurations around the champion (for example, different dropout, learning rate, or half-life choices) and rank-average their outputs. Configuration diversity can cancel correlated errors that seed diversity alone cannot, but one bad or outlier member can drag down the whole committee, so keep variations modest. This is an agent decision, not a prescribed range.




## Lower-confidence cards (selectable; screen at ONE seed before investing)
Skepticism on record: top-tail-rider may mis-model our slates (validation has ~5 impressions/user, so nDCG@5 is full-slate ordering, not top-of-many); full-slate-gauc-loss contradicts the measured "additional negatives hurt" evidence.

### top-tail-rider: Smooth top-negative tail (CVaR-style) loss rider
- mechanism: After a half-epoch warm-up, per user take the top-M (M<=8) scoring negatives, form a smooth softmax-tail score t_u (tau=0.25), and add softplus(0.25 + t_u - s_pos) averaged over that user's positives, ramping weight 0->0.10 taken from BPR. Targets exactly what nDCG@5 punishes: observed negatives entering the top of the slate.
- treats: metric-mismatch
- citation: partial-AUC / top-K hard-negative theory (smooth pool variant); distinct from the measured-dead from-scratch hard-negative and listwise-softmax attempts (warm-up + smooth tail + small rider weight).
- expected_gain / cost: +0.0002..+0.0009, mostly via nDCG / med.
- status_pure: untried
- status_1k: untried

### full-slate-gauc-loss: Positive-weighted all-pairs within-user loss
- mechanism: Replace sampled BPR with ALL observed pos-neg pairs per user slate, weight sqrt(w_p*w_n), normalize per user, aggregate users weighted by positive count (exactly GAUC's weighting). ~42 impressions/user makes full enumeration cheap. Keep the pointwise term at current weight.
- treats: metric-mismatch
- citation: AUC-consistent pairwise surrogates (Gao & Zhou). CAVEAT: our measured evidence that adding sampled negatives HURT argues against a big gain — treat as a probe, screen at 1 seed first.
- expected_gain / cost: 0..+0.0008 / low.
- status_pure: untried
- status_1k: untried

### ordinal-watch-ratio-fm: Ordinal watch-ratio auxiliary on FM
- mechanism: Divide play time by min(duration, 18s), bucket the ratio, and train cumulative threshold heads beside long_view BCE. It exposes graded watch depth while preserving the binary scoring head.
- treats: flat-signal | metric-mismatch
- reference_primary: 0.6033
- preconditions: Training outcomes only; cap/clean ratios and preserve ordinal threshold consistency. This card covers the measured FM auxiliary, not a future DCN-native ordinal main objective.
- citation: TPM, KDD 2023 (arXiv:2306.03392); run_real_04 node_002 and run_real_05 node_003
- expected_gain / cost: Best measured FM-aux primary 0.6033, below epsilon over 0.6018 baseline; run-04 form scored 0.6026 / low.
- status_pure: PARENT-CONDITIONAL — dead on the bare FM (original test) but MEASURED WINS as riders on regularized packages (+0.0015 ordinal in run 12/real_05; +0.0018 CWM in deep_l1/run 23). Use on package parents only.
- status_1k: untried

### cwm-censored-fm: CWM-style censored auxiliary on FM
- mechanism: Regress observed truncated watch time from shared FM representations; completed plays are right-censored lower bounds and penalize underprediction only. Combine with the long_view BCE head.
- treats: flat-signal | metric-mismatch | data-shift
- reference_primary: 0.6022
- preconditions: Correct censoring at duration and training-only play_time; this measured card is only the cheap FM auxiliary, not the full counterfactual CWM likelihood.
- citation: Zhao et al., CWM, KDD 2024 (arXiv:2406.07932); run_real_04 node_001
- expected_gain / cost: Published watch-time GAUC is ~0.713-0.715, but the hackathon FM auxiliary measured 0.6022 and failed confirmation / medium.
- status_pure: PARENT-CONDITIONAL — dead on the bare FM (original test) but MEASURED WINS as riders on regularized packages (+0.0015 ordinal in run 12/real_05; +0.0018 CWM in deep_l1/run 23). Use on package parents only.
- status_1k: untried

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

