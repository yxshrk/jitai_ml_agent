# What drives watching on short-video feeds — literature, and what of it KuaiRand-Pure can see (2026-08-31, research session)

Yash's own drivers: catchy music; continuation of a series; story-times from favourite creators; travel videos of
places known or wanted. Each is measured below on train/valid (champion = live_07 node_009, valid 0.6041; a
difference in P(long_view) is the marginal effect, `add` is the additive gain on the champion at the best weight).

## 1. Literature (drivers, and how platforms model them)
- **The hook.** Retention curves cliff in the first seconds: industry benchmarks put 50–60 % of drop-offs in the first
  3 s and "good" 3-second retention at > 70 %; completion of < 15 s videos ≈ 90 %, falling past 30 s. In our label
  terms: a short video must be *completed*, a long one needs 18 s — the hook decides the long one, the payoff the
  short one. Item-side, so its ceiling here is the video oracle (+0.0003).
- **Duration bias / watch-time modelling (Kuaishou).** D2Q (KDD '22) models watch time as quantile regression within
  duration bins; TPM (tree-based progressive regression) and CWM (counterfactual watch time) deconfound duration and
  exposure context. All target the *item's* expected watch time for a user — the FM's dur_bucket × user × video
  already carries the within-user part; our calibration check (CEILING §1) shows no duration bias left to fix.
- **Familiarity bias (YouTube, 2026).** Watch time allocates heavily to *familiar* content — watch frequency, creator
  and genre affinity, recency — enough that YouTube post-ranking debiases against it (LAFB: −0.63 pp familiar share
  at neutral total watch time). Creator affinity is the strongest single driver of repeat watching; recency of the
  last consumption is the strongest predictor of repeat consumption (RepeatNet).
- **Immersion / continuous flow (Kuaishou, TOIS '25; CIKM '23).** Within a session, engagement is a state that
  rises with consecutive watches and decays with fatigue; ImmersRec predicts it from in-session history. This is
  our session-position/density effect (facts §10.5) — real, but only 2 % of the scored pair mass is within a session.
- **Series and creator pages.** Continuation of a series is a within-session, same-creator phenomenon; on Kuaishou
  it lives in the profile/series tabs (KuaiRand tabs 5/6), not the main feed.

## 2. Yash's four drivers, measured in KuaiRand-Pure
| driver | observable proxy | marginal effect | coverage of valid rows | `add` on champion | verdict |
|---|---|---|---|---|---|
| catchy music | user long-viewed this `music_id` before (train) | P = 0.400 vs 0.313 | 1.2 % (heard at all: 3.5 %) | 0 (champion already ranks them at pct 0.68) | **music_id is 1:1 with videos in Pure** (median 1 video per music, p99 = 2; 3 % of rows on a music with ≥ 5 videos) — "catchy music" is unobservable here |
| favourite creators | user long-viewed this `author_id` in the last 7 days (the analog of valid-labels-as-history for test) | P = 0.413 vs 0.313 (0.424 over 13 days) | **0.2 %** (author seen at all in 7 d: 1.0 %; 13 d: 3.4 %) | 0 | real and strong, but 96.6 % of feed impressions are creators the user has never seen — Pure is a discovery feed |
| series continuation | previous impression same author (label known, train only) | gap < 60 s: 0.932 vs 0.006; tab 5/6: **1.000 vs 0.000**; tab 1: 0.516 vs 0.251; gap > 1 h: 0.52 vs 0.37 | tab 5/6 ≈ 3 % of rows; label unobservable at scoring | label-free run-length proxies ≤ +0.0005 (peer's forward.py) | deterministic where it happens — and it happens outside the scored feed, with a label we cannot see |
| topic (travel, drama…) | user long-viewed this `tag` before / candidate shares the tag of the last long-view | 0.361 vs 0.263 / 0.359 vs 0.310 | 51 % / 12 % | **0** (the marginal effect is user-activity confounding; within user the FM has it) | tags are coarse categories; other-half user × tag oracle +0.0002 |
| the hook | video's train P(play ≥ 3 s / 7 s / 18 s), completion, P(long | past 3 s); user patience × hook tercile; |log dur − log patience| | standalone 0.54–0.57 | 100 % | 0 each | item-side, bounded by the video oracle |

## 3. What this says
1. Every one of these drivers is **real in the data where it can be observed** — the effects are 1.3× (topic, music,
   creator) to 150× (series within a minute) — and every one is **either invisible at scoring time (needs the
   previous label) or covers 0.2–3 % of impressions**, because Pure's standard log is a 13-day discovery feed:
   the user has never seen the creator, the music is unique to the video, the series lives in another tab.
2. That is why the champion plateaus and why the literature's biggest lever (familiarity) is closed here. The
   platforms see months of history; **KuaiRand-1K has 11.7 M interactions for 1,000 users (~11,700 per user) and
   27K 322 M for 27,285 (~11,800 per user)** against Pure's ~44 — there, creator affinity, music reuse, topic
   history and series runs are dense, observable features and the familiarity literature applies directly. If the
   bonus datasets are attempted, the first cards should be exactly these four: creator-affinity (recency-weighted),
   music-affinity, tag/topic history, and same-author run continuation.
3. For Pure itself the only remaining sliver is the 0.2–1 % of test rows whose creator the user long-viewed in the
   valid week (legal history at test time); a train+valid refit gives the FM those rows for free (≤ +0.001 total).

Sources: D2Q https://arxiv.org/abs/2206.06003 · CWM https://arxiv.org/html/2406.07932v1 · TPM https://arxiv.org/pdf/2306.03392 ·
LAFB https://arxiv.org/html/2602.07987 · RepeatNet https://arxiv.org/pdf/1812.02646 · ImmersRec https://dl.acm.org/doi/10.1145/3748303 ·
immersion CIKM'23 https://doi.org/10.1145/3583780.3615099 · KuaiRand https://arxiv.org/pdf/2208.08696 (Table 1) ·
hook benchmarks https://www.opus.pro/blog/tiktok-length-format-retention-data, https://aibrify.com/blog/youtube-shorts-retention-curve-playbook
