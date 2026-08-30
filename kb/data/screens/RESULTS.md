# Feature screens on valid against live_07 node_003 (FM+BPR, primary 0.6031), 2026-08-31
- within_user.py: node_002 session features — session_pos varies within-user 0.369 / GAUC 0.508; recent_10m 0.245 / 0.507; previous_gap 0.438 / 0.508. Valid rows per user span median 4,562 min.
- screen.py: target statistics (OOF) — video rate GAUC 0.639, tab×video 0.649, author 0.637, music 0.637, tag 0.563; +0.035–0.039 over a tab×dur prior (0.6054).
- gbdt_probe.py (see transcript): additive on champion ≤ +0.0004 each; LightGBM binary 0.5931, lambdarank 0.5983; blend with FM 0.6044.
- forward.py: exposure context in the split (gap next/prev, next/prev same author/video/tag, ±10 min counts, first/last) ≤ +0.0005 on champion; 71% of next exposures > 1 h.
- stack.py: lambdarank on FM score + 62 features, 5-fold CV over users: 0.6014 OOF, blend 0.6038; features without FM 0.5999.
- Recency: node_003 on train ≤ 2022-04-14 (78% of rows) 0.5999 vs full 0.6031 → +0.003 from the last 7 days; motivates the train+valid refit at designation.

## Ceiling study, 2026-08-31 (research session) — see `CEILING.md`
- pairs.py: 69 % of the champion's GAUC error mass is tab-1 × tab-1 pairs on different days (error 0.379); 2 % within 10 min; cross-tab pairs solved (0.186).
- oracles.py: valid-week video LOO oracle +0.0003, leaky statistics file +0.0000, other-half user-day +0.0000, random log 0, train taste ≤ +0.0001; only user×author / user×music other-half rates carry signal (+0.002 at 3 % coverage).
- knn.py: item–item kNN over train history (positives / negatives / watch fraction / co-exposure) adds 0 on the champion.
- ablations: FM without user field 0.5932 (personalisation +0.008); train + half of valid scored on the other half 0.5821 vs 0.5826; from 04-12 0.6004, from 04-15 0.5949; same-tab BPR 30 % / 70 % / 100 % = 0.6030 / 0.6024 / 0.5880; 20 seeds 0.6044; + node_010 lineage 0.6047.
- BEHAVIOUR.md: Yash's four drivers (music, creators, series, topic) measured — all real where observable (1.3×–150×), all invisible or 0.2–3 % coverage in Pure; music_id is 1:1 with videos; 1K/27K have ~11.7 K interactions per user where they become dense.
- blend009.py (review session, after live_07): lambdarank GBDT on OOF target stats + node_002 session features + tab/dur (alone 0.5996) z-blended with node_003 (single-seed BPR) 0.6052 at w 0.5 (+0.0021); with node_009 (5-seed rank average, 0.6041) 0.6042 / 0.6043 / 0.6046 / 0.6047 at w 0.25 / 0.5 / 0.75 / 1.0 (+0.0006 best); within-user rank blend with 009 ≤ 0.6045. Seed averaging already removes most of what the heterogeneous member removes: ensembling-campaign headroom above the champion ≈ +0.0005.
