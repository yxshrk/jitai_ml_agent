# Final long-shot campaign — validation only

## Protocol and reproduced control

- Only `data/real_ws/train.npz` and `val.npz` (and their matching CSV video IDs)
  are used. Every metric below was produced by `data/official/evaluate.py`; no
  test split was read. Exploration uses seed 42. A seed-42 primary at least
  0.603600 triggers seeds 43 and 44.
- Long-shot final training uses the frozen five-field strong-L0 architecture and
  objective for a fixed five half-epochs. The fixed horizon was chosen before
  long-shot evaluation from the reproduced seed-42 control's best checkpoint,
  avoiding repeated official-validation reads in train-only-tuned branches.
- The existing DIMS control was rerun before any long-shot cell. Independent
  rescoring of its saved predictions reproduced:

| seed | GAUC | nDCG@5 | primary | delta vs 0.6016 | runtime |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.672499 | 0.538352 | 0.605425 | +0.003825 | 35.7s |
| 43 | 0.671157 | 0.537405 | 0.604281 | +0.002681 | 69.2s |
| 44 | 0.671834 | 0.537289 | 0.604561 | +0.002961 | 70.1s |
| **mean ± population std** | **0.671830 ± 0.000548** | **0.537682 ± 0.000476** | **0.604756 ± 0.000487** | **+0.003156** | — |

All deltas versus control below use the reproduced mean, **0.604756**. “Confirmed
win” follows the requested absolute rule (three-seed mean at least 0.603600).
Because this campaign is explicitly benchmarked against strong-L0, the text also
states when such a result is still a kill relative to the actual control.

## 1. Frequency-dependent ID masking and early freeze

Training-only user/video IDs are independently replaced by two added UNK rows
with `p = p0 * (1 + count)^(-alpha)`. The age variant multiplies this probability
linearly from 1× on the latest day to 2× on the oldest day. Validation IDs are
never masked. The freeze variant stops all embedding updates after half-epoch 3
and runs two more half-epochs on cross/MLP/head parameters with accelerated decay.

| cell | exact config | seed | GAUC | nDCG@5 | primary | Δ control | Δ 0.6016 | low-history primary | runtime | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| L1.1 | p0=.1, alpha=.5, age boost=0, no freeze, 5 halves | 42 | 0.671163 | 0.537520 | 0.604342 | -0.000414 | +0.002742 | 0.597742 | 16.6s | confirmation triggered |
|  | same | 43 | 0.670156 | 0.536726 | 0.603441 | -0.001315 | +0.001841 | 0.596408 | 32.0s |  |
|  | same | 44 | 0.670429 | 0.537261 | 0.603845 | -0.000911 | +0.002245 | 0.596191 | 33.1s |  |
|  | **mean ± std** | 42/43/44 | **0.670583 ± 0.000425** | **0.537169 ± 0.000331** | **0.603876 ± 0.000368** | **-0.000880** | **+0.002276** | **0.596780 ± 0.000685** | — | **confirmed win (absolute); kill vs control** |
| L1.2 | p0=.2, alpha=1, age boost=1 (up to 2×), no freeze, 5 halves | 42 | 0.671316 | 0.537485 | 0.604400 | -0.000355 | +0.002800 | 0.597620 | 16.1s | confirmation triggered |
|  | same | 43 | 0.670174 | 0.536677 | 0.603426 | -0.001330 | +0.001826 | 0.596353 | 32.1s |  |
|  | same | 44 | 0.670548 | 0.537100 | 0.603824 | -0.000932 | +0.002224 | 0.596124 | 32.8s |  |
|  | **mean ± std** | 42/43/44 | **0.670679 ± 0.000476** | **0.537087 ± 0.000330** | **0.603883 ± 0.000400** | **-0.000873** | **+0.002283** | **0.596699 ± 0.000658** | — | **confirmed win (absolute); kill vs control** |
| L1.3 | L1.2 + freeze embeddings after 3 halves, dense-only for 2 halves | 42 | 0.671014 | 0.537129 | 0.604072 | -0.000684 | +0.002472 | 0.595978 | 15.6s | confirmation triggered |
|  | same | 43 | 0.669260 | 0.536178 | 0.602719 | -0.002037 | +0.001119 | 0.594775 | 29.5s |  |
|  | same | 44 | 0.668123 | 0.536590 | 0.602356 | -0.002400 | +0.000756 | 0.593612 | 30.4s |  |
|  | **mean ± std** | 42/43/44 | **0.669466 ± 0.001190** | **0.536632 ± 0.000389** | **0.603049 ± 0.000738** | **-0.001707** | **+0.001449** | **0.594788 ± 0.000966** | — | **kill** |

**Branch verdict: kill.** Rare-ID masking is a mild regularizer but removes
identity signal the strong dropout/weight-decay package already controls. The
best altered mean is 0.000873 below strong-L0. Early freezing is clearly worse,
and all variants lower the low-history segment, directly contradicting the
round-2 concentration hypothesis for this idea.

## 2. Rolling-day robust reweighting

Selection is wholly inside the train window. A seed-42 pilot trained through
April 18; BCE losses on April 16–19 identified April 19 as the worst of the last
four days. Each capped config then trained through April 19 and was scored by the
official evaluator on the chronological pseudo-validation of April 20–21. This
used three config cells, after which the selected setting was fixed for official
validation. Seven-day recency weighting remains active and the extra day weights
are normalized jointly.

| cell | train-only config | pseudo seed | pseudo GAUC | pseudo nDCG@5 | pseudo primary | runtime | selection verdict |
|---|---|---:|---:|---:|---:|---:|---|
| L2.1 | worst day 20220419, cap 1.5×, last_days=4, worst_k=1 | 42 | 0.656133 | 0.464087 | **0.560110** | 12.8s | selected |
| L2.2 | same, cap 2× | 42 | 0.655551 | 0.464049 | 0.559800 | 12.9s | kill |
| L2.3 | same, cap 3× | 42 | 0.655226 | 0.464317 | 0.559772 | 12.9s | kill |

The finalized L2.1 official-validation reads were:

| exact config | seed | GAUC | nDCG@5 | primary | Δ control | Δ 0.6016 | low-history primary | runtime | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| worst 20220419, cap 1.5×, 5 halves | 42 | 0.672078 | 0.538231 | 0.605155 | +0.000399 | +0.003555 | 0.596390 | 16.5s | promising; confirmation triggered |
| same | 43 | 0.669724 | 0.536540 | 0.603132 | -0.001624 | +0.001532 | 0.594574 | 22.7s |  |
| same | 44 | 0.671095 | 0.536787 | 0.603941 | -0.000815 | +0.002341 | 0.593773 | 22.8s |  |
| **mean ± std** | 42/43/44 | **0.670966 ± 0.000966** | **0.537186 ± 0.000746** | **0.604076 ± 0.000831** | **-0.000680** | **+0.002476** | **0.594912 ± 0.001095** | — | **confirmed win (absolute); kill vs control** |

**Branch verdict: kill.** The mild cap gave an encouraging seed-42 bump, but it
did not replicate and the confirmed mean is below strong-L0. Larger caps were
already worse on the train-only fold, suggesting that aggressively following a
single hard recent day adds variance rather than robust temporal adaptation.
Low-history users regress more strongly than the aggregate.

## 3. Asymmetric label smoothing at the 18-second threshold

Play time is used only to create training targets. Near means within ±20% of
`min(duration_ms, 18000)`; inference uses no play-time field. Each pair below is
`(positive epsilon, negative epsilon)`, so the asymmetry deliberately softens
positives more. Short videos have a separate completion-boundary schedule.

| cell | exact config: long-near / short-near / far | seed | GAUC | nDCG@5 | primary | Δ control | Δ 0.6016 | low-history primary | runtime | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| L3.1 | (.10,.03) / (.05,.015) / (.01,.003), width=.20, 5 halves | 42 | 0.672539 | 0.538148 | 0.605343 | +0.000588 | +0.003743 | 0.596868 | 16.4s | promising; confirmation triggered |
|  | same | 43 | 0.669830 | 0.536925 | 0.603378 | -0.001378 | +0.001778 | 0.594315 | 29.2s |  |
|  | same | 44 | 0.670915 | 0.536762 | 0.603838 | -0.000918 | +0.002238 | 0.594029 | 29.4s |  |
|  | **mean ± std** | 42/43/44 | **0.671095 ± 0.001113** | **0.537278 ± 0.000619** | **0.604186 ± 0.000839** | **-0.000570** | **+0.002586** | **0.595071 ± 0.001276** | — | **confirmed win (absolute); kill vs control** |
| L3.2 | (.20,.06) / (.10,.03) / (.02,.006), width=.20, 5 halves | 42 | 0.672339 | 0.538042 | 0.605191 | +0.000435 | +0.003591 | 0.596851 | 16.3s | promising; confirmation triggered |
|  | same | 43 | 0.669854 | 0.536916 | 0.603385 | -0.001371 | +0.001785 | 0.594673 | 29.5s |  |
|  | same | 44 | 0.670784 | 0.536623 | 0.603703 | -0.001052 | +0.002103 | 0.594609 | 29.2s |  |
|  | **mean ± std** | 42/43/44 | **0.670992 ± 0.001025** | **0.537194 ± 0.000612** | **0.604093 ± 0.000787** | **-0.000663** | **+0.002493** | **0.595377 ± 0.001042** | — | **confirmed win (absolute); kill vs control** |
| L3.3 | (.10,.03) / (.05,.015) / (0,0), width=.20, 5 halves | 42 | 0.672508 | 0.538256 | 0.605382 | +0.000626 | +0.003782 | 0.596917 | 16.4s | promising; confirmation triggered |
|  | same | 43 | 0.669634 | 0.536820 | 0.603227 | -0.001529 | +0.001627 | 0.594143 | 29.0s |  |
|  | same | 44 | 0.670926 | 0.536767 | 0.603847 | -0.000909 | +0.002247 | 0.594566 | 29.3s |  |
|  | **mean ± std** | 42/43/44 | **0.671022 ± 0.001175** | **0.537281 ± 0.000690** | **0.604152 ± 0.000906** | **-0.000604** | **+0.002552** | **0.595209 ± 0.001220** | — | **confirmed win (absolute); kill vs control** |

**Branch verdict: kill.** Boundary-only smoothing is the best seed-42 cell and
the mild schedule has the best confirmed mean, but none survives comparison to
strong-L0. Stronger smoothing is slightly worse, consistent with the binary
threshold carrying real signal rather than mainly annotation uncertainty. The
apparent seed-42 gains are selection noise, and low-history users do not benefit.

## 4. Frozen empirical-Bayes user adapter

A strong-L0 model is trained once and frozen. Its candidate representation is
the sum of frozen item and author embeddings. Each user's prototype is the mean
representation of train positives minus the mean representation of train
negatives. The score adjustment is exactly
`n/(n+tau) * dot(prototype, candidate_repr)`. Tau is chosen chronologically
inside train: fit/history through April 19 and pseudo-validation on April 20–21.

| cell | train-only config | pseudo seed | pseudo GAUC | pseudo nDCG@5 | pseudo primary | shared base runtime | selection verdict |
|---|---|---:|---:|---:|---:|---:|---|
| L4.1 | tau=20, adapter scale=1 | 42 | 0.656503 | 0.464272 | **0.560388** | 14.4s | selected |
| L4.2 | tau=50, adapter scale=1 | 42 | 0.656451 | 0.464265 | 0.560358 | 14.4s | kill |

The finalized L4.1 official-validation reads were:

| exact config | seed | GAUC | nDCG@5 | primary | Δ control | Δ 0.6016 | low-history primary | runtime | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| positive-mean − negative-mean prototype, tau=20, scale=1, 5 halves | 42 | 0.672346 | 0.538161 | 0.605254 | +0.000498 | +0.003654 | 0.596249 | 17.4s | promising; confirmation triggered |
| same | 43 | 0.669858 | 0.536858 | 0.603358 | -0.001398 | +0.001758 | 0.594152 | 24.3s |  |
| same | 44 | 0.670911 | 0.536829 | 0.603870 | -0.000886 | +0.002270 | 0.593936 | 24.0s |  |
| **mean ± std** | 42/43/44 | **0.671039 ± 0.001020** | **0.537283 ± 0.000621** | **0.604161 ± 0.000801** | **-0.000595** | **+0.002561** | **0.594779 ± 0.001043** | — | **confirmed win (absolute); kill vs control** |

**Branch verdict: kill.** Tau selection is stable enough to prefer weaker
shrinkage, and seed 42 is directionally positive, but the adapter fails to
replicate and lowers the three-seed mean. Frozen ID embeddings already encode
much of this preference geometry; the post-hoc dot product mostly perturbs a
better nonlinear ranking. Sparse users show the largest regression.

## 5. Bonus: four-bin discrete-time survival hazard

This bonus was attempted because all four required branches closed quickly.
Four conditional stop hazards span equal fractions of
`min(duration_ms, 18000)`. A stop before the boundary is an observed event and
post-event bins are masked; reaching the boundary is right-censoring after
surviving all four bins. Training uses `0.5 * hazard NLL + 0.5 * within-user
BPR`, and inference ranks by log survival probability. Play time is training
only.

| cell | exact config | seed | GAUC | nDCG@5 | primary | Δ control | Δ 0.6016 | low-history primary | runtime | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| LB.1 | 4 bins, censor-correct hazard NLL/BPR=.5/.5, 5 halves | 42 | 0.667488 | 0.535863 | 0.601676 | -0.003080 | +0.000076 | 0.595599 | 17.3s | **kill; stop early** |

The high-risk head loses substantial GAUC and misses the confirmation trigger,
so no further survival config or seeds were run. Compressing watch behavior into
four hazards makes optimization harder without adding inference-time information;
the direct binary/BPR objective is much better aligned with the official ranker.

## Final conclusions

The low-history segment is the bottom third of train users sorted by
`(history_count, user_id)` to break ties exactly: 8,737 of 26,210 train users,
covering 19,164 validation rows and 5,928 validation users. The reproduced
control scores **0.596882 ± 0.000801** primary on this segment across three
seeds. All candidate segment values were produced by the official scorer on the
corresponding saved predictions.

| idea | best measured config | primary | Δ strong control | low-history primary | final conclusion |
|---|---|---:|---:|---:|---|
| ID masking + freeze | L1.2, 3-seed mean | 0.603883 ± 0.000400 | -0.000873 | 0.596699 ± 0.000658 | kill; freeze is worse still |
| rolling-day reweighting | L2.1, 3-seed mean | 0.604076 ± 0.000831 | -0.000680 | 0.594912 ± 0.001095 | kill; seed-42 bump does not replicate |
| asymmetric smoothing | L3.1, 3-seed mean | **0.604186 ± 0.000839** | **-0.000570** | 0.595071 ± 0.001276 | best long-shot, but kill vs control |
| empirical-Bayes adapter | L4.1, 3-seed mean | 0.604161 ± 0.000801 | -0.000595 | 0.594779 ± 0.001043 | kill; redundant/noisy post-hoc geometry |
| survival bonus | LB.1, seed 42 | 0.601676 | -0.003080 | 0.595599 | clear kill |

**None of the four long-shots (or bonus) works against the reproduced strong-L0
control.** Several three-seed means exceed the deliberately weaker absolute gate
of 0.603600 and therefore satisfy the task's literal “confirmed win” definition,
but calling them replacements would be misleading: every mean is lower than
0.604756. Their seed-42 improvements consistently reverse at seed 43, exactly
the failure mode the confirmation rule is meant to expose.

The best absolute-confirmed long-shot is mild asymmetric smoothing (L3.1), so
`zoo/ls_best.py` cleanly implements that exact measured config as required. It is
an archival campaign winner, **not** the deployment recommendation; retain the
existing strong-L0+recency control. The low-history hypothesis is rejected for
all candidates: each candidate's segment mean is below the control segment, with
the largest losses from reweighting and the empirical-Bayes adapter.

Every completed training/evaluation stayed below eight minutes (the longest
reported final run was 33.1s; train-only tuning sub-runs were at most 14.4s).
The only failed invocation was an import-path launcher error before training or
scoring; it was fixed and rerun, and no metric was fabricated or logged for it.
