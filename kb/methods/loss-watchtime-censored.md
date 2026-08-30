---
id: loss-watchtime-censored
family: watch-time
target_component: loss
source: kb/literature/watchtime/2406.07932_cwm.pdf (counterfactual watch time, censored regression); organizer README "unexplored" #4; facts §3
applies_when:
  - the label is a deterministic function of watch time and length: long_view = play_time_ms >= min(duration_ms, 18 s) (facts §3)
  - 17.2 % of train rows are completed plays, i.e. censored observations of watch time (facts §3)
  - play_time_ms is available on train rows as a target (task.md: outcome columns are legal targets)
expected_delta: [0.0, 0.0003]
expected_delta_basis: measured (ADR-0018): best seed-mean gain +0.0004 over 2 measurement(s), so the promise is capped at the record; was: predicting the mechanism that generates the label rather than the label; CWM reports gains on
  KuaiRand-Pure on its own label, but our FM already sees duration_ms via dur_bucket, so treat 0.008 as the ceiling
cost: ~90 lines (second head sharing V, censored loss, transform of play time); runtime ~1.3x; numpy only
composes_with: [features-duration-unknown-flag, data-weighting-recency, model-dcn-cross-head]
conflicts_with: []
status: dead_under [official FM + loss-bpr-pairwise-within-user x1 (best Δ +0.0003); official FM + field-aware FM embeddings x1 (best Δ -0.0021)]
evidence: [live_02:node_013, live_04:node_016]
---
## Claim
Add a second head that regresses (log) watch time with a one-sided loss for completed plays — a completed play only
says "would have watched at least this long" — and use it as an auxiliary signal for the long_view head.

## Mechanism (why it moves within-user ranking)
Two rows of one user with the same label carry different information: a 3 s watch and a 17 s watch of an 18 s video
are both long_view = 0, but the second is nearly positive. Regressing watch time recovers that ordering
information; censoring keeps completed plays from teaching "the user stops at the video's end" (CWM §3).

## How to implement on node_000
1. Target t = log1p(play_time_ms / 1000); censored flag c = (play_time_ms >= duration_ms).
2. Second head: s2 = b2 + w2[x].sum() + the same interaction term (V is shared; only the bias vector is new).
3. Loss2 per row: uncensored → (s2 − t)^2; censored → max(0, t − s2)^2 (penalise only under-prediction).
4. Total gradient on V = logloss gradient + lambda x loss2 gradient (lambda 0.1–0.3); w2/b2 get only loss2.
5. Keep the long_view logit s1 as the prediction; optionally score = s1 + alpha x (s2 − log1p(min(dur, 18 s)/1000)).

## Risks / failure modes
- Watch time is heavy-tailed: always transform (log1p); clip t at, say, log1p(600).
- lambda too high pulls V toward regression and hurts the ranking head — sweep 0.05 / 0.1 / 0.3.
- duration_ms = 0 rows have no meaningful threshold — exclude them from loss2.

## Measured
_Verdict:_ never accepted in 2 measurements on 2 stack(s); official FM + loss-bpr-pairwise-within-user x1 (best Δ +0.0003); official FM + field-aware FM embeddings x1 (best Δ -0.0021)
- live_02:node_013 on [official FM + loss-bpr-pairwise-within-user]: primary 0.6035, single-seed Δ +0.0004, seed-mean Δ +0.0003 (t 1.85) — rejected; 50 changed lines
- live_04:node_016 on [official FM + field-aware FM embeddings]: primary 0.6010, single-seed Δ -0.0021 — rejected; 51 changed lines
