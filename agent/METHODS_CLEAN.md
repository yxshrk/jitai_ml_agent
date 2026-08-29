# Method cards

Literature-derived method cards for clean-knowledge runs. These cards describe
mechanisms, applicability, implementation preconditions, citations, and
literature-reported expectations only.

### bpr-hybrid: Within-user BPR + pointwise hybrid
- mechanism: Form positive/negative pairs only inside each user's impressions and optimize logistic score differences. Mix a pairwise ranking objective with BCE so ranking alignment does not discard pointwise stabilization.
- treats: metric-mismatch | flat-signal
- preconditions: Training batches expose complete user groups; retain rows from one-class users for the BCE term.
- citation: Rendle et al., BPR; RankTower (arXiv:2407.12385)
- expected_gain / cost: Pairwise-loss papers commonly report GAUC improvements in the +0.005-0.015 range when the training objective is mismatched with ranking evaluation / low once grouping exists.
- status_pure: untried
- status_1k: untried

### dndcg-lambda: Delta-nDCG lambda weighting for top-5 groups
- mechanism: Weight within-user positive/negative pair gradients by the absolute nDCG@5 change caused by swapping the pair. Build batches as complete groups sized like the evaluator's top-5 lists.
- treats: metric-mismatch | flat-signal
- preconditions: A correct grouped pairwise or listwise implementation and stable position/discount computation are available; binary groups must contain both labels.
- citation: LambdaRank and LambdaLoss literature
- expected_gain / cost: Lambda-weighted objectives are reported to improve top-ranked ordering most when the training and evaluation objectives differ / low-medium.
- status_pure: untried
- status_1k: untried

### listwise-softmax: Per-user listwise softmax
- mechanism: Apply a temperature-scaled softmax across every impression for one user and cross-entropy toward that user's positive mass. A pointwise term may be retained for stabilization.
- treats: metric-mismatch | flat-signal
- preconditions: Full user histories must remain intact inside batches and one-class groups must be skipped for the listwise term.
- citation: ListNet; PSL (arXiv:2411.00163); sampled-softmax work (arXiv:2201.02327)
- expected_gain / cost: Listwise objectives are reported to improve ranking alignment when complete query or user groups fit in training batches / low-medium.
- status_pure: untried
- status_1k: untried

### dcn-lite: DCNv2-lite interaction head
- mechanism: Add one or two explicit cross layers and a small MLP over the field embeddings. This supplies bounded higher-order interactions without the parameter cost of larger interaction networks.
- treats: underfit | flat-signal
- preconditions: Use a compact embedding dimension, a modest hidden layer, ranking-aware early stopping, and limited depth to control overfitting.
- citation: DCNv2; FuxiCTR and BARS benchmark literature
- expected_gain / cost: DCNv2 reports efficient gains from explicit bounded feature crosses on sparse recommendation and prediction tasks / low.
- status_pure: untried
- status_1k: untried

### finalmlp: FinalMLP two-stream fusion
- mechanism: Feed embeddings through two gated MLP streams and combine them with a bilinear fusion head. The streams can capture complementary feature interactions while keeping the network compact.
- treats: underfit | flat-signal
- preconditions: Start from a regularized compact parent and compare multiple seeds because architecture deltas may be variance-sensitive.
- citation: FinalMLP, AAAI 2023 (arXiv:2304.00902); FuxiCTR and BARS benchmark literature
- expected_gain / cost: FinalMLP reports stronger feature interaction modeling than conventional single-stream MLP recommenders across several benchmarks / medium.
- status_pure: untried
- status_1k: untried

### mtl-shared-bottom: Shared-bottom multi-task heads
- mechanism: Share embeddings and a trunk across long_view and a few dense auxiliary outcomes, with small task heads and a low total auxiliary weight. Auxiliaries regularize sparse representations through training-only gradients.
- treats: underfit | overfit | flat-signal
- preconditions: Auxiliary outcomes are targets only, never validation inputs; use a small set of correlated tasks and guard against seesaw effects.
- citation: shared-bottom and ESMM literature; HoME (arXiv:2408.05430)
- expected_gain / cost: Multi-task recommendation papers report gains when auxiliary labels are correlated and negative transfer is controlled / low.
- status_pure: untried
- status_1k: untried

### ordinal-watch-ratio-fm: Ordinal watch-ratio auxiliary on FM
- mechanism: Normalize play time by a capped duration, bucket the ratio, and train cumulative threshold heads beside long_view BCE. This exposes graded watch depth while preserving the binary scoring head.
- treats: flat-signal | metric-mismatch
- preconditions: Use training outcomes only, clean and cap ratios, and preserve ordinal threshold consistency.
- citation: TPM, KDD 2023 (arXiv:2306.03392)
- expected_gain / cost: Ordinal watch-time modeling is reported to capture graded engagement better than a binary target alone when duration effects are controlled / low.
- status_pure: untried
- status_1k: untried

### cwm-censored-fm: CWM-style censored auxiliary on FM
- mechanism: Regress observed truncated watch time from shared representations; completed plays are right-censored lower bounds and penalize underprediction only. Combine this with the long_view classification head.
- treats: flat-signal | metric-mismatch | data-shift
- preconditions: Apply censoring at content duration and use play time only as a training target.
- citation: Zhao et al., CWM, KDD 2024 (arXiv:2406.07932)
- expected_gain / cost: Censored watch-time modeling is reported to reduce bias introduced by completed-play truncation / medium.
- status_pure: untried
- status_1k: untried

### regularization-schedule: Compound dropout, row-L2, weight decay, and LR decay
- mechanism: Apply MLP dropout, accessed-row embedding L2, dense-weight decay, and learning-rate decay. The compound package aims to keep validation ranking improving after an early peak.
- treats: overfit
- preconditions: Learning curves peak early and then fall; tune the package coherently and select on ranking quality.
- citation: standard neural-network regularization and learning-rate scheduling literature
- expected_gain / cost: Compound regularization is broadly reported to improve generalization when sparse embeddings and dense interaction layers overfit / low.
- status_pure: untried
- status_1k: untried

### swa-ema: SWA or EMA checkpoint averaging
- mechanism: Maintain an exponential or stochastic average of weights across late checkpoints, then score the averaged model. Averaging reduces optimizer and seed variance without adding inference models.
- treats: overfit | data-shift
- preconditions: Training must produce several useful near-peak checkpoints; do not average far past a sharply collapsing validation curve.
- citation: SWA (Izmailov et al.); exponential moving-average training literature
- expected_gain / cost: Weight averaging is reported to improve generalization and reduce optimization variance at little inference cost / low.
- status_pure: untried
- status_1k: untried

### embedding-dim-down: Reduce embedding dimension
- mechanism: Reduce ID embedding capacity, limiting memorization in sparse user and item rows and forcing more shared signal. Keep the loss and head unchanged for a clean capacity test.
- treats: overfit
- preconditions: Learning curves indicate overfitting; use identical seeds and early stopping because capacity effects can be small.
- citation: embedding-capacity and regularization literature for sparse recommenders
- expected_gain / cost: Recommender studies report that smaller embeddings can improve generalization when identifiers are sparse and high-capacity embeddings memorize / low.
- status_pure: untried
- status_1k: untried

### duration-regime-heads: Short/long duration regime heads
- mechanism: Route examples through separate prediction heads for short and long videos while sharing embeddings and trunk. This lets ranking functions differ where label censoring changes.
- treats: metric-mismatch | data-shift | underfit
- preconditions: Both regimes have enough examples, routing uses impression-known duration only, and heads are regularized toward their shared parent.
- citation: D2Q duration debiasing, KDD 2022 (arXiv:2206.06003)
- expected_gain / cost: Duration-debiasing literature reports gains when engagement labels systematically vary with content length / low-medium.
- status_pure: untried
- status_1k: untried

### user-metadata-crosses: Coarse user-metadata by item/context crosses
- mechanism: Cross stable coarse user attributes with item, author, or context IDs so features vary across candidates within a user. Back off rare crosses to avoid exploding sparse capacity.
- treats: flat-signal | data-shift
- preconditions: Metadata must be legal at inference and each cross must include candidate-side variation; user-constant features alone cannot change within-user ranking.
- citation: sparse recommender feature-crossing literature
- expected_gain / cost: Feature-crossing studies report gains when crosses expose interactions that factorized low-order models miss, with sparsity as the main risk / medium.
- status_pure: untried
- status_1k: untried

### session-time-features: Session and fine time context
- mechanism: Derive causal session position and gap features from impression timestamps and add time-context crosses. These represent transient intent and distribution changes that static IDs miss.
- treats: data-shift | flat-signal
- preconditions: Compute features from current and past impressions only, preserve row order, and never use future session events.
- citation: session-aware recommendation and temporal-context literature
- expected_gain / cost: Session-aware recommenders report improved ranking when short-term intent differs from long-term user preference / medium.
- status_pure: untried
- status_1k: untried

### recency-weighting: Exponential training recency weighting
- mechanism: Weight training rows by age relative to the validation boundary so recent behavior dominates stale impressions. Keep evaluation and sampling unchanged.
- treats: data-shift
- preconditions: A temporal split and trustworthy dates are available; normalize weights and retain enough effective sample size.
- citation: temporal covariate-shift and importance-weighting literature
- expected_gain / cost: Recency weighting is reported to help under temporal drift when recent samples better match the evaluation distribution / low.
- status_pure: untried
- status_1k: untried

### covisit-svd-init: Item co-visitation SVD initialization
- mechanism: Build a train-only user-item incidence matrix, factor its item co-visitation matrix, and initialize item embeddings from leading singular vectors. Fine-tune normally afterward.
- treats: underfit | flat-signal
- preconditions: Co-visitation uses training impressions only and the factor dimension matches the item embedding dimension.
- citation: classical item-item collaborative filtering and spectral-initialization literature
- expected_gain / cost: Spectral initialization is reported mainly to accelerate learning and provide a useful collaborative prior when random embeddings learn slowly / medium.
- status_pure: untried
- status_1k: untried

### seed-architecture-ensemble: Seed and diverse-architecture rank ensemble
- mechanism: Train a competitive configuration across several seeds and optionally add a genuinely distinct architecture, then average scores or per-user ranks. Diversity reduces variance and uncorrelated ranking errors.
- treats: data-shift | flat-signal
- preconditions: Final-stage only; component models must be individually competitive and predictions aligned row-for-row.
- citation: deep-ensemble and rank-aggregation literature
- expected_gain / cost: Ensemble literature reports more reliable gains when members are both accurate and diverse / high training, low implementation.
- status_pure: untried
- status_1k: untried

### seed-ensemble: Seed ensemble of the champion configuration
- mechanism: Cancel variance across random initializations by training the champion configuration at several seeds and per-user rank-averaging their validation predictions.
- treats: flat-signal
- preconditions: A champion configuration must already exist and promising single-model moves must be exhausted. This is a standard closing move, not an opening move.
- citation: Deep Ensembles (Lakshminarayanan et al. 2017)
- expected_gain / cost: Deep-ensemble literature reports improved predictive stability and generalization from averaging independently initialized models / high training, low implementation.
- status_pure: untried
- status_1k: untried

- member diversity: At run time, reason about whether seed diversity is enough or whether to write a custom node instead of calling `zoo/ensemble_node.py`. A custom node may train members with deliberately varied configurations around the champion (for example, different dropout, learning rate, or half-life choices) and rank-average their outputs. Configuration diversity can cancel correlated errors that seed diversity alone cannot, but one bad or outlier member can drag down the whole committee, so keep variations modest. This is an agent decision, not a prescribed range.

### item-aggregates: Train-window item/author target aggregates
- mechanism: Add Bayesian-smoothed video and author long_view rates computed only on training dates. Candidate-varying rates can affect within-user order, unlike user-only rates.
- treats: flat-signal | data-shift
- preconditions: Compute statistics strictly within the training window with no validation or future leakage.
- citation: target-encoding and empirical-Bayes smoothing literature
- expected_gain / cost: Smoothed target encoding is reported to help categorical models when repeated entities carry stable outcome signal and leakage is controlled / low.
- status_pure: untried
- status_1k: untried

### content-features: Video content categorical features
- mechanism: Add legal content categorical embeddings to provide cold-item semantics beyond video ID.
- treats: flat-signal | data-shift
- preconditions: Features must be available at impression time and for validation; avoid future or full-period counters.
- citation: content-aware recommendation literature
- expected_gain / cost: Content-aware recommenders report improved cold-item generalization when metadata complements collaborative identifiers / medium.
- status_pure: untried
- status_1k: untried

### lightgbm-lambdarank: LightGBM LambdaRank and NN blend
- mechanism: Train user-grouped LambdaRank on legal categorical and train-only aggregate features, then rank-average it with a neural model. Lambda gradients target nDCG ordering directly.
- treats: metric-mismatch | flat-signal
- preconditions: Requires grouped sorting, leakage-safe engineered features, and an execution environment that provides LightGBM.
- citation: LightGBM LambdaRank documentation and learning-to-rank literature
- expected_gain / cost: LambdaRank literature reports strong top-ranked ordering on grouped tabular problems, while blending can help when model errors are complementary / medium.
- status_pure: untried
- status_1k: untried
