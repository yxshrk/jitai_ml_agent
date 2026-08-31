# JITAI — README, 2:45 video, and skeptical-judge Q&A

**Draft pack · based on the latest team-supplied status · Pure: 0.605575 · 1K: 0.66892**

## Editor notes — do not paste this section into the public README

The drafts use the current designated artifacts, not an assumed success from the live final wave. If an eligible in-run ensemble supersedes the 1K checkpoint, update its score, artifact form, run/node reference, exporter, and matching video/Q&A wording together; otherwise keep the faithful single-checkpoint policy. The 0.6802 post-run ensemble and the 27K demonstration remain explicitly outside the submitted-result claims.

**One command needs your current path:** set `JITAI_1K_PREDICT_SCRIPT` to the exporter named in the latest 1K artifact manifest. In the earlier uploaded ZIP, `tools/predict_test_1k_winner.py` still contains the former two-member recipe with seeds `(42, 1051)`, so that snapshot does not establish the correct exporter for 0.66892. If that file has since been updated, use its current path after checking the manifest. The run CLI, Pure predictor, validation-audit command, submission-check command, and positional `evidence.render` invocation below were checked against the ZIP; no training or test prediction was executed to prepare these drafts.

The new-search command uses `--knowledge full` deliberately: **no executable/checkpoint seed is not the same as the CLI’s literature-only `--knowledge clean` mode**. It runs the current checked-in policy, not the historical champion policy. Ensure the linked evidence files and data/artifact restoration instructions are present in the public checkout. Current outcome and compliance statements come from your latest update, not a fresh audit of overnight code.

---

## Deliverable 1 — README first screen

Copy the Markdown inside this block into the repository README. Links are relative to the repository root.

````markdown
# JITAI — autonomous ML research you can audit

**TikTok TechJam 2026 · Track 2**

JITAI diagnoses a recommendation model, selects a treatment, edits executable model code, retrains, evaluates, and keeps or rejects the result without a human changing the run. A solution-tree harness owns scoring, resource accounting, promotion, and termination; a cited method-card library carries measured discoveries into later runs.

## Results: what we are submitting

Task: rank each user's **logged impressions**, not the entire catalog. Target: `long_view`. Primary: `mean(GAUC, nDCG@5)` under the organizer evaluator.

| Dataset / scope | Validation primary | Artifact and status |
|---|---:|---|
| **KuaiRand-Pure — required** | **0.605575** | `run_bigclock_07`, `node_006`: **in-run three-member per-user rank ensemble**, seeds 42–44; best checkpoint at epsilon convergence |
| **KuaiRand-1K — bonus** | **0.66892** | **Converged single checkpoint**, with a faithful reproduction policy; prior JITAI designation: 0.6524 |
| KuaiRand-27K — demonstration only | 0.67263 | **Out-of-protocol scaling demonstration; not an eligible submitted benchmark result** |

Pure improves on the published **0.6016 validation baseline by +0.003975 (approximately +0.0040)**. Its designated run used **6 top-level iterations, approximately 16 minutes, approximately 115k LLM tokens, and 0 behavior-changing human interventions**, including a logged failure and autonomous debug recovery.

The **0.6802 post-run 1K ensemble is development evidence, not the submitted artifact**. Results on different dataset variants are not directly comparable, and no row above is a measured hidden-test score.

**Start with [the submission recipe](SUBMISSION_RECIPE.md), [the run index](logs/RUNS.md), and [Pure's original journal](logs/run_bigclock_07/journal.jsonl).**

## What the agent learned

These are local improvements from separate campaign experiments, not additive contributions to the designated Pure result.

| Discovery | Measured local primary gain | Evidence to inspect |
|---|---:|---|
| Temporal-pair kernel | +0.0014 | Agent selected an untried card; measured gain was twice its prediction |
| Context-stratified pairs | +0.0015 | Context-matched ranking comparisons |
| Heterogeneous ensemble close | +0.0010 | Complementary predictions, assembled inside a run |
| Causal session features on 1K | +0.019 / +0.022 | Two independent runs; independent rescoring, fresh seeds, and shuffle audit |

Approximately **130 runs and 250 measured ledger cells** are disclosed. Tested negative directions include watch-time losses, listwise objectives, attention, extra feature crosses, hard-negative BPR, and unsuccessful post-hoc blends; these are scoped experimental findings, not claims that an entire method family can never work.

## Reproduce and inspect

Prepare the approved train/validation exports and restore the designated-run artifacts as described in [SUBMISSION_RECIPE.md](SUBMISSION_RECIPE.md). Configure your own provider credentials for a new agent run; inspecting saved evidence does not require an LLM call.

```bash
uv sync
uv run pytest -q

# NEW unseeded search using the current policy, not a replay of historical prompts.
# Full knowledge means cited methods + measured campaign knowledge; no seed script.
uv run python -m harness.cli run --data-dir data/real_ws \
  --dataset pure --baseline-script zoo/baseline_ws.py --knowledge full \
  --confirm-runs 2 --max-iters 50 --max-hours 6 \
  --run-dir logs/run_reproduction_new

# Independently evaluate saved Pure validation predictions; no retraining.
uv run python tools/validate_bc07_robustness.py

# Render the existing journal and trajectory; this is reporting, not re-evaluation.
uv run python -m evidence.render logs/run_bigclock_07

# After artifact lock: rebuild Pure predictions using the frozen three-member recipe.
# Requires the label-free test feature export specified in SUBMISSION_RECIPE.md.
CUDA_VISIBLE_DEVICES="" uv run python tools/predict_test_bc07.py
uv run python evidence/submission.py --check \
  evidence/test_submission_pure.csv data/test_features/test.npz

# 1K: set this to the CURRENT exporter named in the designated artifact's manifest.
# It must reproduce the 0.66892 single checkpoint, not the development ensemble.
uv run python "${JITAI_1K_PREDICT_SCRIPT:?Set the manifest-matched 1K exporter path}"
uv run python evidence/submission.py --check \
  evidence/test_submission_1k.csv data/test_features_1k/test.npz
```

A fresh LLM search is not guaranteed to reproduce the same trajectory or score. The historical journal, saved predictions, and frozen submission recipe identify the result being claimed.

## Evidence pack

| Entry point | What it establishes |
|---|---|
| [Submission recipe](SUBMISSION_RECIPE.md) | Exact designated artifacts, reproduction/export policy, and associated manifests |
| [Run index](logs/RUNS.md) · [machine-readable inventory](evidence/run_inventory.csv) | Complete campaign navigation, including failed, steered, interrupted, and ineligible runs |
| [Pure journal](logs/run_bigclock_07/journal.jsonl) · [trajectory](logs/run_bigclock_07/report/trajectory.png) | Original decisions, measured outcomes, failure/recovery, and termination |
| [Method cards](agent/METHODS.md) · [experiment ledgers](zoo/) | Citations, hypotheses, measured annotations, and negative results |
| [Evidence directory](evidence/) | 1K audit receipts, Pure robustness, probe manifest, resource/cost records, and submission checks |
| [Harness](harness/) · [contracts](CONTRACTS.md) | Evaluation boundary, repeatability gate, convergence implementation, and accounting |

## Compliance and statistical limits

> **Autonomy boundary:** Pure began without a supplied candidate solution or checkpoint; human-authored methods and measured cross-run knowledge were available. Zero interventions refers to behavior changes inside designated runs, not to human development of the system. Launch steering and between-run changes are disclosed separately.
>
> **Data boundary:** Final models train on training rows only; validation supports evaluation and selection, while hidden-test outcomes are structurally excluded from the research and prediction path. The unresolved evaluation-only random-exposure probe is disclosed in its manifest, excluded from submitted designation, and absent from agent-facing prompts and cards; we do not claim written approval that was never received.
>
> **Stopping and promotion:** The official rule remains three consecutive iterations improving by at most **0.002**, with a 50-iteration / six-hour limit. Accepted sub-epsilon gains still count toward that streak. After a 123-run post-mortem found **0/10** grey confirmations passing, the internal floor changed **prospectively from 0.0007 to 0.0005**, retaining two reseeds and `z >= 2`; this checks seed repeatability, not hidden-test significance. Historical over-budget development runs are flagged and excluded from scored-artifact eligibility.
>
> **Uncertainty:** Adaptive validation reuse creates selection optimism. Pure's model-based hidden-test forecast is approximately **0.5977 ± 0.002**, or approximately **+0.0031** over the published 0.5946 test baseline, after a winner's-curse adjustment; it is **not measured test performance or a calibrated confidence interval**. The actual hidden-test evaluation is decisive.
````

---

## Deliverable 2 — 2:45 video script

**Spoken script: 432 words.** Speak only the voiceover text; shot directions, overlays, headings, and `[PAUSE]` are not spoken. The nine beats total 2:45; take each marked pause for roughly half a second and let the final screen hold briefly.

Keep “Recorded run replay” visible on historical footage. Label the temporal/context-pair clips **campaign experiments**, not steps in the designated Pure run; label fresh-seed and shuffle numbers **audit results**, not submitted artifacts. If the memorization panels are shown, describe them as stress tests rather than a proof of no memorization. Do not put the model-based hidden-test forecast in a column headed “test result.”

### Beat 1 · 0:00–0:17 · The question

**On screen:** Final results table, then terminal launch. Overlay: “JITAI · autonomous recommender research”; “Within-user long_view · primary = mean(GAUC, nDCG@5)”.

**Voiceover:**

> When a machine learning agent reports a better score, can you trust how it got there? JITAI lets you inspect the answer. It diagnoses a recommender, chooses an experiment, edits the model, retrains, evaluates, and records why it keeps or rejects the result.

**[PAUSE]**

### Beat 2 · 0:17–0:37 · Who decides what

**On screen:** Terminal launch, followed by journal diagnosis and verdict. Overlay: “Agent: diagnose / choose / edit · Harness: evaluate / account / stop”.

**Voiceover:**

> The task is ordering each user's logged videos by long view, not searching the entire catalog. The language model chooses and implements treatments; the harness owns evaluation, budgets, promotion, and stopping. Every decision has a recorded trail. Humans built the harness and method library. Once the designated run launched, no human changed its behavior.

### Beat 3 · 0:37–0:58 · A measured discovery

**On screen:** METHODS.md: temporal-pair card, untried status followed by measured-win annotation; trajectory chart for the relevant discovery run, not the Pure champion. Overlay the exact local deltas +0.0014 and +0.0015.

**Voiceover:**

> Across the campaign, the agent selected an untried temporal pairing card: compare a positive impression with a negative nearby in time. It delivered twice its predicted gain. Context matched pairs produced another measured improvement. These are gains in separate experiments, not numbers we add together or attribute to the champion's own trajectory.

**[PAUSE]**

### Beat 4 · 0:58–1:14 · Failure is part of the record

**On screen:** Designated-run journal: failure, autonomous diagnosis/debug recovery, then verdict. Label the footage “Recorded run replay”. Briefly show a rejected method in the ledger.

**Voiceover:**

> The journals also preserve failure. An experiment in the designated run failed, and the agent handled debugging and recovery without a human edit. Watch time losses, listwise objectives, attention, hard negative mining, and unsuccessful blends remain documented rather than disappearing from the story.

### Beat 5 · 1:14–1:42 · The larger gain had to survive an audit

**On screen:** 1K trajectory and memorization-evidence panels. Headline: “Designated checkpoint: 0.66892 validation”. Audit overlay: “Two independent runs: +0.019 / +0.022”; “Exact independent re-evaluation”; “Fresh seeds: 0.6766 / 0.6762”; “Within-hour shuffle: 0.6652”. Label fresh seeds and shuffle as audits, not submitted scores.

**Voiceover:**

> On the one K dataset, causal session context produced much larger gains in two independent runs. Future outcomes never become inference features. We challenged the result three ways: an independent evaluator matched exactly; fresh random seeds scored higher; and retraining after a within hour shuffle retained most of the improvement. These checks strengthen the evidence; they do not prove every possible confound absent. The submitted number remains the converged checkpoint, not a stronger ensemble assembled afterwards.

**[PAUSE]**

### Beat 6 · 1:42–2:00 · We audited the agent’s decisions too

**On screen:** RUNS.md scoreboard with the post-mortem numbers as text overlays, then journal verdict. Overlay: “123-run review: 0 / 10 grey confirms passed”; “Prospective floor: 0.0007 → 0.0005”; “Official ε = 0.002 unchanged”.

**Voiceover:**

> We audited the decision policy too. A campaign review found that the grey zone floor had rejected repeatable small gains. We lowered that floor prospectively, retaining two reseeds and the repeatability check. The official convergence threshold did not change: accepted gains below epsilon still count toward stopping.

### Beat 7 · 2:00–2:17 · The exact Pure artifact

**On screen:** Pure trajectory ends at node_006; show the three-member artifact row. Overlay: “0.605575 validation · +0.003975 vs 0.6016”; “6 iterations · 16 min · ≈115k LLM tokens · 0 interventions”.

**Voiceover:**

> Our designated Pure run reached zero point six zero five six, about four thousandths above baseline, using a three member rank ensemble selected inside the run. It converged in six iterations, sixteen minutes, and roughly one hundred fifteen thousand language model tokens.

**[PAUSE]**

### Beat 8 · 2:17–2:35 · Show the whole campaign

**On screen:** RUNS.md scoreboard, full evidence index, and card-to-later-run linkage. Overlay: “≈130 disclosed runs · ≈250 measured cells”; “Development exceptions labelled”; “Hidden-test performance unmeasured”.

**Voiceover:**

> About one hundred thirty runs and two hundred fifty measured cells are disclosed, including development exceptions. Discoveries become cited method cards that later runs can apply. Validation was reused, so we acknowledge selection optimism; our hidden test forecast is an estimate, not a measured result or a confidence guarantee.

### Beat 9 · 2:35–2:45 · Close on what is reproducible

**On screen:** Reproduction commands, then final results table. Keep the 27K row visibly greyed/labeled “Out-of-protocol demonstration”. End on “Inspect the journal. Reproduce the artifact.”

**Voiceover:**

> Inspect the journal, reproduce the submitted artifact, and judge the complete research process. JITAI's result is a better recommender, with the evidence needed to question it.

**[PAUSE]**

---

## Deliverable 3 — skeptical-judge Q&A

These answers assume that the final wave has not replaced either designation. Keep the relevant evidence open, but answer the question first rather than leading with a tour of the repository.

### 1. Why not submit the 0.6802 1K ensemble—and are you counting the 27K demonstration as a compliant result?

The submitted artifacts are Pure’s in-run three-member ensemble at 0.605575 and 1K’s converged single checkpoint at 0.66892; 0.6802 is post-run development evidence and 27K’s 0.67263 is an out-of-protocol demonstration.

### 2. With roughly 130 runs and searches inside iterations, aren’t you evading the cap—and is “six iterations” misleading?

The organizer permits multiple disclosed runs, but six refers only to the designated Pure run’s top-level iterations; subordinate trials and full campaign logs are disclosed separately, without presenting that run as six model fits.

### 3. How can you claim zero intervention when humans wrote the method cards and steered some runs?

Zero means no behavior-changing human action inside a designated run, not no human development: we disclose the human-built harness, measured method library, launch steering where used, and between-run changes.

### 4. Is this autonomous research, or just executing a human-authored recipe library?

We claim autonomous selection, implementation, testing, and knowledge reuse—not invention of every method—and the untried temporal-pair card’s measured gain doubled the agent’s prediction before becoming evidence for later runs.

### 5. Did lowering the acceptance floor manufacture progress or postpone official convergence?

The prospective 0.0007→0.0005 promotion-floor change followed a 123-run review with zero of ten grey confirmations passing; two reseeds and z≥2 remain, and accepted sub-epsilon gains still advance the unchanged stopping streak.

### 6. Do two reseeds and a large z-statistic actually establish that a small improvement is real?

The gate measures repeatability under training randomness on the same validation set, not generalization uncertainty or multiple-search significance, so we do not present its z-statistic as proof of hidden-test improvement.

### 7. After selecting the maximum from so many runs, why should we believe 0.605575—or your predicted test score?

We report 0.605575 as selected validation performance and approximately 0.5977±0.002 as an assumption-dependent, winner’s-curse-adjusted test forecast—not a measured result or calibrated confidence interval—with an estimated +0.0031 over the published test baseline.

### 8. Why is the Pure gain only about 0.004 after such a large campaign?

Within-user ordering, rather than global calibration, is the objective, and our tested larger models, richer losses, and feature expansions plateaued while pair sampling and ensemble design improved; that is an empirical finding, not a claimed ceiling.

### 9. Could the large 1K session-feature gain be leakage, memorization, or an artifact of file order?

Training is train-only and session features exclude future outcomes; exact independent rescoring, fresh-seed results of 0.6766/0.6762, and 0.6652 after within-hour shuffling support the mechanism without proving every possible confound absent.

### 10. Could rank-average ties and the evaluator’s tie ordering explain the Pure result?

Although 10.8% of Pure slates had ensemble ties, 200 random tie resolutions moved primary by only about ±0.00009, so arbitrary tie order does not explain the approximately 0.004 gain.

### 11. You ran a random-exposure probe without receiving the promised organizer ruling—was that hidden feedback?

We disclose the probe and missing written ruling, exclude it from submitted designation, and verify no prompt or card references, without claiming that this retrospectively erases prior human exposure.

### 12. Some development runs exceeded six hours—why should we accept the compliance claim?

Those non-designated development overruns are flagged and excluded from scored-artifact eligibility, while the submitted runs are within budget; we disclose the between-iteration backstop defect rather than retrospectively rewriting those runs.

---

## Last consistency pass — editor checklist

Before publishing, make the final results table, `SUBMISSION_RECIPE.md`, manifests, Devpost fields, and video end card agree on **score, split, run/node, artifact form, and resource scope**. Keep the Pure “three members” wording, label the 27K row out of protocol, retain the probe/overrun disclosures, and keep the prospective gate change separate from the unchanged 0.002 stopping rule. Show only actual recorded recovery footage, and do not relabel a reconstruction or fault-injection test as the original incident.

**Source basis:** latest team update for all current measurements and decisions; the uploaded `mle-agent-endgame.zip` for inspected command syntax; the adopted *JITAI Final-Wave Decision Memo*, §§3.3–3.5, for the evidence-first README and video structure. The current 1K exporter and newly written manifests were not present in that earlier code snapshot.
