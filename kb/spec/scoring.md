# Scoring (frozen; `docs` §2.6 + `baseline_scores.json`)

## Judging weights
| criterion | weight | what is measured |
|---|---|---|
| Technical Execution | 35 % | hidden-test delta over baseline **and** robustness |
| Innovation & Problem Insight | 20 % | what the agent chose to try and why; use of published methods |
| Impact & Relevance | 20 % | autonomy — number of manual interventions |
| Feasibility & Practicality | 15 % | LLM tokens + agent wall-clock, 3 coarse tiers; **only scored if hidden-test primary beats the baseline** |
| Presentation | 10 % | finals only |

## Primary metric
`delta(m) = score_agent(m) − score_baseline(m)` on hidden test for m ∈ {GAUC, nDCG@5}; score = mean of the deltas.
Because primary is itself the mean of the two metrics: **score = test_primary − 0.5946**. One number.

## Reference numbers
| model | valid GAUC | valid nDCG@5 | valid primary | test GAUC | test nDCG@5 | test primary |
|---|---|---|---|---|---|---|
| random | 0.4993 | 0.4675 | 0.4834 | 0.4996 | 0.4511 | 0.4753 |
| item popularity | 0.6387 | 0.5227 | 0.5807 | 0.6308 | 0.5121 | 0.5715 |
| **FM official** | 0.6674 | 0.5357 | **0.6016** | 0.6610 | 0.5282 | **0.5946** |
| oracle (labels as scores) | 1.0000 | 0.6968 | 0.8484 | 1.0000 | 0.7289 | 0.8645 |

FM std over 5 seeds: **0.0008** on every metric. Our reproduction (2026-08-30, seed 0): valid 0.6015 / test 0.5953.

## Derived facts
- Test cohorts: **27.1 %** all-negative users (nDCG ≡ 0 for any model), **9.2 %** all-positive (nDCG ≡ 1),
  **63.7 %** discriminative (the only users GAUC sees). Hence oracle nDCG = 1 − 0.271 = 0.729.
- Valid: oracle nDCG 0.6968 ⇒ **30.3 %** all-negative users — valid is composed differently from test.
- Headroom above FM on test: GAUC 0.339, nDCG@5 0.201, primary **0.270**. FM has captured 30.7 % of the range.
- FM's valid → test gap: **−0.0070** primary. Expect our gains to shrink on test as well.
- Noise floor: a valid delta under **0.002** is noise (≈ 2.5 σ). Confirm any win on ≥ 3 seeds.
- Sharper diagnostic: nDCG@5 restricted to discriminative users, `(nDCG − all_pos_share) / disc_share`, is
  1.57× more sensitive than the raw number on test (FM: 0.685 against a ceiling of 1.0).

## Realistic expectations (priors, not measurements)
Item popularity → FM is worth +0.023; all of personalisation. A strong outcome is roughly +0.01 to +0.03 on test
primary. Any validation primary far above the low 0.61s should be suspected of leakage before being celebrated.
