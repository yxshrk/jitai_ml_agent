# Method cards

Compiled from `research/kuairand-and-watchtime.md`,
`research/models-losses-hparams.md`, `MENU.md`, and the read-only
`zoo/EXPERIMENTS.md`. Measured statuses also incorporate the run-04/05 journal
numbers behind MENU's CURRENT DIRECTIVE. A `measured-dead` card must not be
selected again in the measured form; a materially different mechanism may get
its own card.

Frozen-stack validation references: Pure = 0.6047; 1K = 0.6134 (literal frozen
default, seed 42).

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
