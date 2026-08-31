# Method cards

Literature-derived method cards for clean-knowledge runs. These cards describe
mechanisms, applicability, implementation preconditions, citations, and
literature-reported expectations only.

## Research doctrine (general methodology, never campaign results)

- SCREEN BEFORE YOU BET. Early iterations should buy information cheaply: one
  fan-out node can probe many mechanisms or dial settings with short trainings
  (2-3 epochs, optional subsample) and train only the winner at full fidelity.
  Random search over wide log-uniform ranges beats hand-picked grids (Bergstra
  & Bengio, JMLR 2012); successive halving allocates probe budget efficiently
  (Hyperband/ASHA, Li et al. 2018). A broad screen dominates a single narrow
  bet whenever several plausible mechanisms are untried.
- BUDGET ARITHMETIC. A short probe costs a small fraction of a node's
  wall-clock budget while a full-fidelity training costs many probes; spend
  most of a search node on probes and reserve the remainder for the final
  full-length training. Finishing a search node early is wasted information,
  not efficiency.
- IMPLEMENTATION-DEAD IS NOT EVIDENCE-DEAD, AND BOTH FORCE A PIVOT. A method
  whose implementations keep FAILING (crashes, gate rejections) in this run has
  taught you nothing about the method but plenty about your ability to build it
  now: after two failed builds of the same card, pivot to a mechanically simpler
  card that tests a related hypothesis instead of attempting a third build. A
  method that built correctly and measured flat is evidence-dead: record it and
  do not retry variants.
- KEEP YOUR OWN LEDGER. In-run measurements on THIS dataset outrank any
  literature prior. When the journal shows a mechanism produced an accepted
  gain, exploit it: tune, regularize, or compose within that direction before
  opening unrelated bets. When the journal shows a mechanism family rejected
  twice, treat that family as dead for this run and do not retry variants of
  the same mechanism.
- COMPOUND CLEARED WINS. After an accepted change the new champion is the
  parent; stack the next change on top of it rather than restarting from the
  baseline.
- DOSAGE IS EXHAUSTED BY A SWEEP. If the accepted champion came from a dial
  search that already sampled regularization dials (weight decay, dropout,
  embedding size, LR schedule/checkpoint), re-applying a regularization or
  schedule card is not a new hypothesis — those dials were just searched. The
  next change must introduce a NEW MECHANISM (a different objective, negative
  sampler, feature source, or architecture), which shifts the optimum and
  earns its own re-tuning; then close over what worked.
- HONEST TELEMETRY. When the learning curve is missing or unusable, the
  correct diagnosis is insufficient-telemetry; do not guess a pathology.
  Prefer a low-risk, diagnosis-independent broad move (screening, leakage-safe
  features) over a treatment chosen on a guessed diagnosis.
- CLOSE WITH DIVERSITY. Ensembles gain most from members that are individually
  competitive AND decorrelated: independent initializations plus modest
  configuration variation reduce correlated errors (Deep Ensembles,
  Lakshminarayanan et al. 2017). Reserve the final iterations to close with an
  ensemble of the champion family before the convergence rule ends the run.
- ENDGAME MARGIN ARITHMETIC. Near the end of a run (streak building, or few
  iterations left), a candidate is only worth an iteration if its expected gain
  clears the acceptance threshold WITH margin; a small treatment whose typical
  effect sits at or below epsilon is dominated by an ensemble close — PROVIDED
  the close's members already exist as trained artifacts or reuse the champion
  verbatim. A close that requires building new members from scratch inherits
  first-draft implementation risk and loses its evidential edge. Do the
  comparison explicitly before spending a late iteration on a small treatment.
- CONVERGENCE PRESSURE. The run ends after consecutive sub-epsilon iterations;
  as the streak grows, shift from exploring new mechanisms toward finishing
  moves with reliable literature-reported payoff (ensembling, checkpoint
  averaging) so the run ends with its strongest artifact.

### bpr-hybrid: Within-user BPR + pointwise hybrid
- mechanism: Form positive/negative pairs only inside each user's impressions and optimize logistic score differences. Mix a pairwise ranking objective with BCE so ranking alignment does not discard pointwise stabilization.
- treats: metric-mismatch | flat-signal
- preconditions: Training batches expose complete user groups; retain rows from one-class users for the BCE term.
- citation: Rendle et al., BPR; RankTower (arXiv:2407.12385)
- expected_gain / cost: Pairwise-loss papers commonly report GAUC improvements in the +0.005-0.015 range when the training objective is mismatched with ranking evaluation / low once grouping exists.
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

### regularization-schedule: Compound dropout, row-L2, weight decay, and LR decay
- kind: opportunity
- mechanism: Apply MLP dropout, accessed-row embedding L2, dense-weight decay, and learning-rate decay. The compound package aims to keep validation ranking improving after an early peak.
- treats: overfit
- preconditions: Learning curves peak early and then fall; tune the package coherently and select on ranking quality.
- citation: standard neural-network regularization and learning-rate scheduling literature
- expected_gain / cost: Compound regularization is broadly reported to improve generalization when sparse embeddings and dense interaction layers overfit / low.
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

### seed-architecture-ensemble: Seed and diverse-architecture rank ensemble
- kind: opportunity
- mechanism: Train a competitive configuration across several seeds and optionally add a genuinely distinct architecture, then average scores or per-user ranks. Diversity reduces variance and uncorrelated ranking errors.
- treats: data-shift | flat-signal
- preconditions: Final-stage only; component models must be individually competitive and predictions aligned row-for-row.
- citation: deep-ensemble and rank-aggregation literature
- expected_gain / cost: Ensemble literature reports more reliable gains when members are both accurate and diverse / high training, low implementation.
- status_pure: untried
- status_1k: untried

### seed-ensemble: Seed ensemble of the champion configuration
- kind: opportunity
- mechanism: Cancel variance across random initializations by training the champion configuration at several seeds and per-user rank-averaging their validation predictions.
- treats: flat-signal
- preconditions: A champion configuration must already exist and promising single-model moves must be exhausted. This is a standard closing move, not an opening move.
- citation: Deep Ensembles (Lakshminarayanan et al. 2017)
- expected_gain / cost: Deep-ensemble literature reports improved predictive stability and generalization from averaging independently initialized models / high training, low implementation.
- status_pure: untried
- status_1k: untried

- member diversity: At run time, reason about whether seed diversity is enough or whether to write a custom node instead of calling `zoo/ensemble_node.py`. A custom node may train members with deliberately varied configurations around the champion (for example, different dropout, learning rate, or half-life choices) and rank-average their outputs. Configuration diversity can cancel correlated errors that seed diversity alone cannot, but one bad or outlier member can drag down the whole committee, so keep variations modest. This is an agent decision, not a prescribed range.

### hyperparam-random-search: Wide random-search dial sweep with short probes
- kind: opportunity
- treats: overfit | underfit | flat-signal
- mechanism: One fan-out node samples many configurations (learning rate, weight decay, dropout, embedding dimension, epoch/checkpoint choice) from wide log-uniform ranges, scores each with a short probe training, prunes losers early, then trains the best configuration at full fidelity. Random search covers a high-dimensional dial space far better than a hand-picked grid of round numbers.
- preconditions: Probes must share the exact data pipeline and evaluator with the final training; budget probes so the node finishes inside the timeout; log every probe config and score for auditability.
- citation: Bergstra & Bengio, JMLR 2012 (random search); Li et al. 2018 (Hyperband/ASHA successive halving)
- expected_gain / cost: Random-search literature reports it reliably outperforms grid search at equal budget because only a few dials matter and wide sampling finds unintuitive optima / low per probe, one node total.
- status_pure: untried
- status_1k: untried

### heterogeneous-ensemble-design: Cross-family ensemble close (probe, map complementarity, full-train, verify)
- kind: opportunity
- mechanism: Combine models from DIFFERENT mechanism families (for example a regularized cross-network, a pairwise-ranking model, a sequence-aware model, a recency-weighted factorization model) by per-user rank averaging. Executed by the harness in three phases from a typed plan: (1) train a short probe of every candidate family; (2) rank-average every probe subset to map which families complement each other (blending saved score vectors costs nothing); (3) fully retrain only the complementary families from scratch, blend, and re-verify the winning combination at full fidelity before emitting. The incumbent is kept if no blend beats it.
- treats: flat-signal | plateau
- preconditions: A strong single model already exists and single-model moves have stopped clearing the convergence bar; at least two genuinely different mechanism families are available to field. This is a closing move: play it when the remaining headroom for any one model is smaller than the gain diversity can supply, not as an opener. Members must be single fits (no internal search), with distinct seeds and families.
- citation: Dietterich, Ensemble Methods in Machine Learning (2000): diversity of errors, not member count, drives ensemble gain; Netflix Prize blending reports (Koren 2009; Töscher et al. 2009): heterogeneous model blends outperform homogeneous ones; Breiman, Bagging Predictors (1996) for variance cancellation.
- expected_gain / cost: Ensemble literature consistently reports that decorrelated members gain more than additional seeds of one model; gain is bounded by the strength of the best member / high training (several fits), low implementation (harness-executed).
- status_pure: untried
- status_1k: untried

### context-stratified-pairs: Same-context pairwise negative stratification
- mechanism: Draw a fraction of pairwise (BPR) negatives from the SAME context as the positive (e.g. same user-and-time-bucket or same surface), with tiered fallback to coarser contexts when no opposite-label row exists; keep total negatives per positive unchanged. Rationale: ranking metrics score contemporaneous impressions in one slate, so the pairs the model must separate are same-context pairs (in-batch/slate negative sampling practice).
- treats: metric-mismatch | data-shift
- preconditions: A pairwise loss with within-user grouping already present; contexts without opposite-label rows need a fallback tier.
- citation: BPR (Rendle et al. 2009); in-batch/slate negative sampling practice in production rankers (e.g. sampled softmax two-tower literature, Yi et al. 2019).
- expected_gain / cost: Small single-mechanism refinements of negative sampling typically report low-single-digit basis-point ranking gains / low.
- status_pure: untried
- status_1k: untried

