---
id: loss-warp-within-user-rank-weighting
family: ranking-loss
target_component: loss
source: Weston, Bengio, and Usunier, "WSABIE: Scaling Up to Large Vocabulary Image Annotation," IJCAI 2011,
  https://www.ijcai.org/Proceedings/11/Papers/460.pdf; reference implementation in LightFM,
  https://github.com/lyst/lightfm/blob/master/lightfm/lightfm.py ([ijcai.org](https://www.ijcai.org/Proceedings/11/Papers/460.pdf))
applies_when:
  - same-user positive and negative training rows are available (facts §7: most train users are discriminative)
  - BPR is competitive, but its nDCG@5 improvement is smaller than its GAUC improvement (journal nodes 003 and 009)
  - ordinary hard-negative BPR, LambdaRank, RankSVM, and ApproxNDCG have failed, but none estimated a positive's
    rank from the number of negative probes and weighted its update by that estimated rank
expected_delta: [0.0, 0.0000]
expected_delta_basis: bounded (ADR-0018) at +0.0000 by the oracle for 'pair-sampling' — facts §11.3: same-tab negatives at 30 / 70 / 100 % 0.6030 / 0.6024 / 0.5880 vs 0.6031; matched / hard / cohort pair cards measured ≤ +0.0001; was: WARP explicitly emphasizes positives estimated to rank poorly and is a standard top-of-list
  alternative to BPR, but it adds no information and several related loss refinements already failed
cost: ~35 changed lines on the BPR sampler; 2–4x pair-scoring runtime with ten probes; numpy only
composes_with: [ensembling-seed-average, model-neural-factorization-machine, features-exposure-session]
conflicts_with: [loss-bpr-pairwise-within-user, loss-bpr-hard-negatives, loss-ranksvm-margin-pairs,
  loss-lambdarank-pairs]
status: untried
evidence: [ceiling:oracle]
---
## Claim
Use WARP's repeated violation search and harmonic rank weighting on same-user positive-negative impression rows,
rather than giving every uniformly sampled BPR pair equal importance.

## Mechanism (why it moves within-user ranking)
For a positive row, repeatedly sample negatives from the same user until one violates
`score_pos >= score_neg + margin`. The number of probes estimates how many negatives outrank the positive; weighting
the hinge update by the harmonic loss of that estimated rank concentrates capacity near the top of the user's list.
Unlike the rejected max-of-m hard-negative probe, both the stopping time and estimated rank control the gradient.

## How to implement on node_000
1. Apply the existing same-user BPR grouping, retaining each positive's negative-pool start and count.
2. In each epoch, score up to ten independently sampled negatives per positive and select the first margin violator.
3. Estimate `rank = max(1, floor((neg_count - 1) / probes))` and precompute harmonic weights `H[rank]`.
4. Update only violating pairs with hinge gradient multiplied by `H[rank]`, normalized by batch weight sum.
5. Use margin 1.0, preserve validation-primary early stopping, and keep inference unchanged.
6. Seed every probe from the member's existing `numpy.Generator`; `SMOKE_EPOCHS` still caps epochs.

## Risks / failure modes
- Training users have longer histories than evaluation lists, so estimated ranks may overweight errors irrelevant to top five.
- If most pairs violate early, WARP degenerates into a noisy weighted hinge loss; log probes per accepted pair.
- Ten probes can correlate ensemble members unless each member retains its independently seeded sampler.
- Harmonic weights can create large gradients; normalize them and clip the effective pair weight at five.

## Measured
_Verdict:_ no measurement yet
- ceiling:oracle on [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average]: BOUNDED <= +0.0000 for the signal family 'pair-sampling' — facts §11.3: same-tab negatives at 30 / 70 / 100 % 0.6030 / 0.6024 / 0.5880 vs 0.6031; matched / hard / cohort pair cards measured ≤ +0.0001 (facts §11, kb/data/screens/CEILING.md)
