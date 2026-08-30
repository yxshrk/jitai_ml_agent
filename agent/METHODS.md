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

## Depth policy (overrides brevity instincts)
- PROBE PARALLELISM: probes are small models — run them CONCURRENTLY, not one at a
  time. On CUDA, train 4-6 probe variants in parallel (separate processes via
  multiprocessing spawn, or interleaved in one process); on CPU, use one process per
  probe across cores (each with a bounded thread count). A sequential sweep on an
  idle device wastes most of the budget; measure per-probe wall time in progress.log.

Searches must be EXHAUSTIVE, not token gestures. Hard minimums when a search card is
played with a generous timeout: stage-1 coarse pass >= 16 probe trainings; stage-2
refine >= 10 probes on a denser grid around the winner; probes at FULL training length
on the full data whenever the device is fast (GPU) or the timeout is in hours — short
subsampled probes are a last resort and mis-rank configs near the optimum. Never stop
a search early because it "seems long enough"; stop when the grid is covered. Reserve
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
(basins can move with architecture changes); refine locally as usual. On 1K the
known-good point is (dropout 0.21, wd 4e-5, lr 0.00168, k 24, hl 7) -> 0.621
single; ensembling gains are LARGE on 1K (0.632-0.639 measured).

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
  ancestor's components) when the literature setup does so — cite the pairing. A bundle
  whose joint expected gain clears epsilon = 0.002 also resets the convergence streak.


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

### listwise-softmax: Per-user listwise softmax
- mechanism: Apply a temperature-scaled softmax across every impression for one user and cross-entropy toward that user's positive mass. Hybridize with BCE only as specified by the measured implementation.
- treats: metric-mismatch | flat-signal
- reference_primary: 0.5991
- preconditions: Full user histories must remain intact inside batches and one-class groups must be skipped for the listwise term.
- citation: ListNet; PSL (arXiv:2411.00163); sampled-softmax work (arXiv:2201.02327); run_real_04 node_003
- expected_gain / cost: Expected parity with BPR, but measured primary 0.5991 versus FM baseline 0.6018 / low-medium.
- status_pure: measured-dead (0.5991 primary, run 04)
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

### ordinal-watch-ratio-fm: Ordinal watch-ratio auxiliary on FM
- mechanism: Divide play time by min(duration, 18s), bucket the ratio, and train cumulative threshold heads beside long_view BCE. It exposes graded watch depth while preserving the binary scoring head.
- treats: flat-signal | metric-mismatch
- reference_primary: 0.6033
- preconditions: Training outcomes only; cap/clean ratios and preserve ordinal threshold consistency. This card covers the measured FM auxiliary, not a future DCN-native ordinal main objective.
- citation: TPM, KDD 2023 (arXiv:2306.03392); run_real_04 node_002 and run_real_05 node_003
- expected_gain / cost: Best measured FM-aux primary 0.6033, below epsilon over 0.6018 baseline; run-04 form scored 0.6026 / low.
- status_pure: measured-dead (0.6033 best primary as FM auxiliary)
- status_1k: untried

### cwm-censored-fm: CWM-style censored auxiliary on FM
- mechanism: Regress observed truncated watch time from shared FM representations; completed plays are right-censored lower bounds and penalize underprediction only. Combine with the long_view BCE head.
- treats: flat-signal | metric-mismatch | data-shift
- reference_primary: 0.6022
- preconditions: Correct censoring at duration and training-only play_time; this measured card is only the cheap FM auxiliary, not the full counterfactual CWM likelihood.
- citation: Zhao et al., CWM, KDD 2024 (arXiv:2406.07932); run_real_04 node_001
- expected_gain / cost: Published watch-time GAUC is ~0.713-0.715, but the hackathon FM auxiliary measured 0.6022 and failed confirmation / medium.
- status_pure: measured-dead (0.6022 primary as FM auxiliary)
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
- mechanism: The closing ensemble node probes its own design instead of assuming it: member count {3,5,7}, combination rule {per-user rank average, probability average}, optionally member diversity (consecutive seeds vs seeds+dial-jitter). Short-probe the options where affordable, pick on validation, produce the final ensemble. Log all probed designs.
- treats: overfit
- preconditions: Apply to the best accepted single-model champion. This is the canonical last node of a run.
- citation: Deep Ensembles (Lakshminarayanan et al., NeurIPS 2017); rank aggregation practice.
- expected_gain / cost: +0.0004..+0.0015 depending on parent seed variance / medium-high runtime.
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

### item-aggregates: Train-window item/author target aggregates
- mechanism: Add Bayesian-smoothed video/author long_view rates computed only on training dates. Item-varying rates can affect within-user order, unlike user-only rates.
- treats: flat-signal | data-shift
- reference_primary: 0.6038
- preconditions: Strict train-window computation with no full-period statistic files or validation leakage.
- citation: standard target encoding; `zoo/EXPERIMENTS.md` E4
- expected_gain / cost: Measured 0.6038 versus 0.6047 without aggregates / low.
- status_pure: measured-dead (0.6038 primary; Pure-only popularity-prior blends also failed)
- status_1k: untried

### content-features: Video content categorical features
- mechanism: Add video_type, upload_type, frequent music ID, and first-tag categorical embeddings to provide cold-item semantics beyond video ID.
- treats: flat-signal | data-shift
- reference_primary: 0.6039
- preconditions: Features must be impression-time legal and available for validation; avoid full-period counters.
- citation: KuaiRand feature schema; `zoo/EXPERIMENTS.md` E6
- expected_gain / cost: Measured 0.6039 versus 0.6048 without content / medium.
- status_pure: measured-dead (0.6039 primary)
- status_1k: untried

### lightgbm-lambdarank: LightGBM LambdaRank and NN blend
- mechanism: Train user-grouped LambdaRank on legal categorical and train-only aggregate features, then rank-average it with the neural model. Lambda gradients target nDCG ordering directly.
- treats: metric-mismatch | flat-signal
- reference_primary: 0.5974
- preconditions: Requires LightGBM plus careful grouped sorting and leakage-safe engineered features; unavailable under the generated-script numpy/torch-only contract.
- citation: LightGBM LambdaRank documentation; `zoo/EXPERIMENTS.md` E8
- expected_gain / cost: Alone measured 0.5974; all tested NN blends were worse (best 0.6034) / medium.
- status_pure: measured-dead (0.5974 primary alone; 0.6034 best blend)
- status_1k: untried
