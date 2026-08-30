---
id: model-din-history-attention
family: model
target_component: model
source: Zhou et al. 2018 "Deep Interest Network for CTR prediction" (KDD) §4 — target attention over the user's behaviour sequence; allowed by rules.md (any library), ADR-0014; torch 2.8 CPU
applies_when:
  - users have a usable but short history: median 35 train rows, p90 103 (facts §2) — attention over ≤ 50 items is cheap
  - 96.6 % of valid rows are authors the user never saw in train (facts §10.3), so history must be matched by ATTRIBUTES (author, tag, duration bucket, tab), not by id — the DIN query/key must be attribute embeddings
  - the champion is a BPR FM whose user × item term is a single dot product; attention adds a candidate-conditioned summary of what this user long-viewed before
expected_delta: [0.0, 0.004]
expected_delta_basis: history-user-aggregates (a crude, unconditioned version of the same information) measured +0.0008–0.0010 on the FM and 0 on BPR; attention conditions the history on the candidate, which is what those aggregates lack; the ceiling is bounded by how much taste the 35-row histories carry
cost: ~110 lines (torch module with FM logit + attention branch, BPR training loop, history padding); runtime 3–6 min on CPU at 4 threads; library: torch 2.8 CPU (installed)
composes_with: [loss-bpr-pairwise-within-user, features-exposure-session, ensembling-seed-average, ensembling-heterogeneous-rank-average]
conflicts_with: [model-lightgbm-lambdarank]
status: dead_under [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average x1 (best Δ -0.0001)]
evidence: [live_07:node_017]
---
## Claim
Scoring a candidate against a candidate-weighted summary of the user's earlier long-views (attention keyed by
author, tag and duration bucket) lets taste reach the 97 % of candidates whose author the user never saw, which
neither the FM's user × video term nor per-user rate aggregates can do.

## Mechanism (why it moves within-user ranking)
For each candidate the attention weights differ, so the pooled history vector differs per row of the same user —
the summary is row-varying and can reorder the list. DIN's insight (paper §4.2): a user's interest is
multi-faceted; the relevant facet is selected by the candidate. Trained with BPR pairs the user-constant part still
cancels, so the branch is forced to learn candidate-specific ordering.

## How to implement on node_000
1. Keep node_000's encoding; add per-video attributes from video_features_basic: author_id, tag (first tag id),
   dur_bucket, plus tab per row. Build for every user the list of their train rows with long_view = 1, sorted by
   time_ms, truncated to the last 50 (pad to 50 with a mask). For TRAIN rows, mask out history items with
   time_ms ≥ the row's own (strictly earlier only); for valid/extra rows use the whole train history.
2. torch module (device cpu; `torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', 1)))`;
   `torch.manual_seed(seed)`): the FM (embedding tables W, V as `nn.Embedding`, same 16-d logit as node_000) plus an
   attention branch: query q = concat(E_author, E_tag, E_dur, E_tab)(candidate); keys k_j = same for history items;
   a_j = MLP([q, k_j, q ⊙ k_j, q − k_j]) → 36 → 1, masked softmax over j; pooled h = Σ a_j k_j;
   score = FM logit + MLP([q, h, q ⊙ h]) → 32 → 1.
3. Train with within-user BPR pairs exactly as the champion (pair sampler unchanged), Adam lr 1e-3, batch 8192,
   ≤ 40 epochs, early stopping on validation primary, `SMOKE_EPOCHS` caps epochs; history per epoch as in node_000.
4. Predict valid and the extra file with the best checkpoint; write per contract.

## Risks / failure modes
- Leakage: a train row's own outcome, or any later row, in its history — the time mask is mandatory (Critic checks).
- Empty histories (no positives yet) must attend to nothing → pooled zero vector; ~15 % of train rows early in time.
- Runtime: build the padded history tensor once, index it per batch; never loop over users inside the epoch.
- Overfitting: the branch adds capacity on 1.1 M rows — dropout 0.2 on q and h, L2 1e-6 on embeddings, patience 4.
- Determinism on CPU is exact given the seed and fixed threads; MPS/CUDA are forbidden by the contract.

## Measured
_Verdict:_ never accepted in 1 measurements on 1 stack(s); official FM + loss-bpr-pairwise-within-user + ensembling-seed-average x1 (best Δ -0.0001)
- live_07:node_017 on [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average]: primary 0.6040, single-seed Δ -0.0001 — rejected; 137 changed lines
