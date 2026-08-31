# Validation research results

All decisions below use only the fixed train and validation splits. No test metric was used to choose a model.

| Candidate | Validation primary | Decision |
| --- | ---: | --- |
| Organizer FM, seed 0 | 0.60150 | Reference run |
| Pairwise BPR FM, seed 0 | 0.60139 | Reject |
| Context FM: hour + is_rand, seed 0 | 0.60150 | Reject |
| Context FM: weekday, seed 0 | 0.60168 | Insufficient alone |
| Context FM: hour + weekday, seed 0 | 0.60143 | Reject |
| Context FM: hour + weekday + is_rand, seed 0 | 0.60246 | Keep |
| Context FM: hour + weekday + is_rand, seed 1 | 0.60253 | Keep |
| Context FM: hour + weekday + is_rand, seed 2 | 0.60209 | Keep |
| Context FM + video age, seed 0 | 0.60122 | Reject |
| Context FM + click auxiliary head, seed 0 | 0.60189 | Reject |
| Mean of three kept Context FM members | 0.60236 | Candidate ensemble |
| Mean-logit ensemble of three kept Context FM members | **0.60289** | Earlier leader |
| Context FM, 32-dimensional embeddings, seed 3 | 0.60221 | Reject: extra capacity overfit |
| Context FM wide-cross bundle, seed 3 | 0.59973 | Reject: exact crosses overfit |
| Regularized Context FM, seed 0 | 0.59751 | Reject: dropout/decay over-regularized |
| Causal Sequence DeepFM, seed 0 | 0.53871 | Reject: unstable optimization |
| Context FM rank ensemble, seeds 0–2 | 0.60276 | Reject: below mean-logit aggregation |
| Context FM + BPR refinement, seed 3 | 0.60290 | Near-tie; reserve for diversity ensemble test |
| 70% Context-FM mean + 30% Context-BPR, seed 3 | **0.60334** | Best observed; validation-selected BPR weight |
| Same fixed 30% BPR blend, seed 4 | 0.60293 | Confirmation; two-seed mean 0.60314 |
| Corrected Sequence DeepFM, seeds 5–8 | 0.604735 / 0.604594 / 0.604555 / 0.604551 | Four-seed mean **0.604609** |
| Four-member Sequence DeepFM ensemble | 0.604483 logits / 0.604559 per-user ranks | Reject: neither improves on member mean |
| Rolling earlier-validation author metadata, seed 5 | 0.603923 | Reject: causal metadata update is weaker than frozen train history |
| Recency-weighted author pooling, seeds 5-8 | 0.604776 / 0.604661 / 0.604452 / 0.604577 | Reject: four-seed mean 0.604616, indistinguishable from leader |
| BCE then within-user pairwise DeepFM, seed 5 | 0.604773 | Reject: near-tie, not a robust gain |
| Multi-task DeepFM, auxiliary weight 0.1, seed 5 | 0.604308 | Reject: auxiliary feedback transfer hurt |
| Positive-only causal author history, seed 5 | 0.604659 | Reject: lower than all-exposure history |
| Candidate-conditioned author-history attention, seed 5 | 0.604519 | Reject: lower than mean-pool control |
| Causal positive-history tag/music matching, seed 5 | 0.604100 | Reject: lower than mean-pool control |
| Feedback-conditioned causal author history, seed 5 | 0.604442 | Reject: lower than mean-pool control |
| Censor-aware watch-time auxiliary, seeds 5-11 | 0.605084 / 0.604608 / 0.604466 / 0.604567 / 0.604954 / 0.604714 / 0.604862 | Keep: richer training-only supervision |
| Selected four-member watch-time mean-logit ensemble (11, 5, 6, 7) | **0.605521** | Previous leader; validate fixed recipe on an earlier chronological holdout |
| Watch-time + CrossNet, seed 5 | 0.605110 | Near tie; not used in selected ensemble |
| Watch-time + click-cascade auxiliary, seed 5 | 0.605193 | Small single-seed gain, but a 5--50% blend with the selected watch-time ensemble scored 0.605410--0.605500; reject as non-complementary |
| Click-to-long-view probability-product inference, seed 5 | 0.604316 | Reject: directly ranking by the learned click cascade was weaker than the direct long-view head |
| Causal session metadata + watch-time auxiliary, seeds 5--11 | 0.605754 / 0.605179 / 0.605601 / 0.605380 / 0.605366 / 0.605115 / 0.605614 | Keep: prior-impression gap and within-session position are outcome-free but useful context |
| Selected two-member session-aware watch-time mean-logit ensemble (5, 7) | **0.606116** | Current validation leader; selected from a fixed seven-seed sweep, so confirm on an earlier chronological holdout before submission |
| API-directed local search: 24 probes + four fresh-seed confirmations | 0.605603 baseline / 0.605432 low-dropout / 0.605709 lower-LR + stronger-decay | Do not promote: the paired optimizer change gained only 0.000106 over its fresh baseline mean and remains below the 0.606116 leader |

The current leader adds censor-aware, training-only watch-time supervision and causal, label-free session context to the Sequence DeepFM. It averages the independently trained seed-5 and seed-7 members. The next research iteration should validate this fixed recipe on an earlier chronological holdout and retain the validation-only model-selection rule.

The API-directed search used five bounded Structured Output decisions (about $0.0057 estimated), selected only from a locally fixed configuration menu, and never received code, raw data, or test results. It exhausted its planned 24 probes and 12 confirmations in 2,433 seconds, below the one-hour cap.
