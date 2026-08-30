---
id: regularization-adversarial-personalized-ranking
family: regularization
target_component: regularization
source: He et al., "Adversarial Personalized Ranking for Recommendation," SIGIR 2018
  (https://hexiangnan.github.io/papers/sigir18-adversarial-ranking.pdf) ([hexiangnan.github.io](https://hexiangnan.github.io/papers/sigir18-adversarial-ranking.pdf?utm_source=openai))
applies_when:
  - same-user logistic BPR is competitive (live_05:node_002 gained +0.00168 fresh-seed mean)
  - validation primary peaks sharply and then declines (node_002: 0.6036 at epoch 8 to 0.6006 at epoch 12)
  - ordinary L2 and delayed learning-rate decay added only +0.00033 and +0.00034 fresh-seed mean on BPR
expected_delta: [0.000, 0.0012]
expected_delta_basis: APR directly regularizes the proven BPR objective against worst-case embedding perturbations,
  but adds no information; recent BPR regularization gains stayed below acceptance scale, so cap the prior at 0.0012
cost: ~30 changed lines; two gradient evaluations per batch, runtime ~2x node_002; numpy only
composes_with: [loss-bpr-pairwise-within-user, model-field-aware-fm-embeddings, model-dcn-cross-head]
conflicts_with: [loss-ranksvm-margin-pairs]
status: untried
evidence: []
---
## Claim
Add an adversarial BPR term evaluated after perturbing FM embeddings in the batch-gradient direction, encouraging
same-user rankings that remain correct throughout a small worst-case parameter neighborhood.

## Mechanism (why it moves within-user ranking)
APR robustifies pairwise preference differences rather than pointwise calibration. Because every training example
remains a same-user positive-negative pair, user-constant terms still cancel; perturbations instead test whether
the row-varying video, author, tab, and duration interactions support a stable ordering.

## How to implement on node_000
1. First apply `loss-bpr-pairwise-within-user`; split `step` into gradient computation and Adam application.
2. Compute clean BPR gradients `gV, gW` for each pair minibatch without updating parameters.
3. Set `delta = epsilon * gV / (sqrt(sum(gV*gV)) + 1e-12)`, initially testing `epsilon=0.05`.
4. Temporarily add `delta` to `V`, recompute BPR gradients, then restore the unperturbed `V`.
5. Apply Adam to `clean_grad + lambda_adv * adversarial_grad`, initially `lambda_adv=0.5`.
6. Keep validation-primary early stopping, checkpoints, predictions, and the score-extra path unchanged.

## Risks / failure modes
- Two gradient passes double training time, though node_002 is only 18 seconds and remains far below the limit.
- Excessive perturbation can destroy useful sparse-ID memorization; test one conservative epsilon first.
- Do not perturb only user biases: additive user terms cannot alter either metric.

## Measured
_Verdict:_ no measurement yet

