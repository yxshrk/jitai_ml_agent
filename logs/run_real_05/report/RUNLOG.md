# Run log (per iteration)

| n | action | hypothesis | change | primary | accepted | error / recovery |
|---|---|---|---|---|---|---|
| 0 | reproduce_baseline | reproduce official FM baseline and calibrate seed noise | baseline seeds [42]: primaries [0.6018], mean 0.6018, sigma 0.0008 | 0.6018 | yes | - |
| 1 | draft | Replacing the baseline FM objective with the known-best one-cross-layer DCN-lite plus MLP128 trained using an equal BCE/BPR hybrid will improve validation primary by approximately 0.003-0.008. | Replacing the baseline FM objective with the known-best one-cross-layer DCN-lite plus MLP128 trained using an equal BCE/BPR hybrid will improve validation primary by approximately 0.003-0.008. | 0.6014 | no | - |
| 2 | draft | Replacing FM logloss with an equal-weight hybrid of impression-level BCE and within-user BPR will improve validation primary by approximately 0.003-0.008. | Replacing FM logloss with an equal-weight hybrid of impression-level BCE and within-user BPR will improve validation primary by approximately 0.003-0.008. | 0.6003 | no | - |
| 3 | draft | Adding a 0.3-weight cumulative ordinal watch-ratio auxiliary loss to the baseline FM while retaining BCE training and validation-GAUC early stopping will improve validation primary by approximately 0.002-0.006. | Adding a 0.3-weight cumulative ordinal watch-ratio auxiliary loss to the baseline FM while retaining BCE training and validation-GAUC early stopping will improve validation primary by approximately 0.002-0.006. (delta +0.0015) | 0.6033 | yes | - |
