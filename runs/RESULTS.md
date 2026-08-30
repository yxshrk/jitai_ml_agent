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
| Mean-logit ensemble of three kept Context FM members | **0.60289** | Current leader |
| Context FM, 32-dimensional embeddings, seed 3 | 0.60221 | Reject: extra capacity overfit |
| Context FM wide-cross bundle, seed 3 | 0.59973 | Reject: exact crosses overfit |
| Regularized Context FM, seed 0 | 0.59751 | Reject: dropout/decay over-regularized |
| Causal Sequence DeepFM, seed 0 | 0.53871 | Reject: unstable optimization |
| Context FM rank ensemble, seeds 0–2 | 0.60276 | Reject: below mean-logit aggregation |
| Context FM + BPR refinement, seed 3 | 0.60290 | Near-tie; reserve for diversity ensemble test |

The contextual fields only help as a group. The next research iteration should prioritize train-history features with strict time ordering and out-of-fold safeguards; retain the validation-only model-selection rule and record each outer experiment's hypothesis, diff, metrics, and any recovery event.
