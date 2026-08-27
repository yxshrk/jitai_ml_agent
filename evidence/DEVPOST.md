# {Project name} — an autonomous ML research agent for KuaiRand long-view prediction

*TikTok TechJam 2026, Track 2. Draft skeleton — fill every `{...}` from the final run's `report/`.*

## The problem

Predict `long_view` (did the user watch ≥ min(duration, 18s)?) on KuaiRand-Pure, scored by
within-user GAUC + nDCG@5 (primary = mean). The official FM baseline sits at **0.6016
validation primary**; published SOTA-adjacent work (CWM, KDD'24) reaches GAUC ~0.71.
The real challenge is the meta-task: build an agent that *autonomously* runs the
research loop — hypothesize, implement, train, evaluate, decide — under a 50-iteration /
6-hour budget, with every decision auditable.

## Our agent architecture

- **Solution tree, not a chat transcript.** Every iteration is a node: one whole runnable
  script + metrics + a one-line journal entry. Each LLM call sees only the task brief,
  the improvement MENU, the one-line-per-node journal, and the parent node's full code —
  context stays flat at ~{X}K tokens no matter how long the run gets, and the static
  prefix is byte-identical for prompt caching.
- **Whole-file nodes.** The proposer always regenerates the complete script — no diff
  application, no patch-failure mode; every node is independently runnable and the
  harness computes diffs for the log itself.
- **Harness-owned search policy** (the LLM never decides control flow): 3 initial drafts
  from distinct strategy families; on failure, debug the same node to max depth 2;
  otherwise greedily improve the current best node; forced branch to a different MENU
  tier after 5 stagnant iterations. Timeouts, validity checks, best-node argmax, and
  stopping (epsilon=0.002 over 3 iterations, cap 50 / 6h) all live in the harness.
- **Noise-aware sigma acceptance rule.** We calibrate seed noise from 3 baseline seeds,
  then accept a change only if delta ≥ 2*sigma (floor 0.002 = the official epsilon); deltas in
  the 0–2*sigma band get one reseed confirmation run; everything else is reverted. No
  seed-noise mirages in the final score, and convergence counts accepted deltas only.
- **Role routing.** A frontier model writes drafts/improvements (and the judged
  hypothesis text); a cheap model handles the mechanical roles — metric parsing,
  bug classification, journal summarization, first-attempt debug fixes (escalating on
  failure). Token split: {frontier tokens} / {cheap tokens}.
- **Structural leakage guard.** The agent workspace mounts train/ and val/ only; the
  test window exists solely in the harness's private dir and is touched exactly once,
  by the final submission step.

## What the agent discovered

{Narrative of the run: which hypotheses it tried, in what order, which were accepted
and why — e.g. "iteration {n}: {hypothesis} → {+delta}". Pull from RUNLOG.md and
trajectory.png. Call out at least one non-obvious accepted change and one
instructive rejection/recovery.}

**Final result: validation primary {best_primary} ({+delta} over the 0.6016 baseline);
test primary {test_primary}.** Trajectory chart and full per-iteration log attached.

## Resources

- Iterations: {n_iterations} ({n_accepted} accepted / {n_rejected} rejected / {n_errors} errored)
- Human interventions: {n_interventions}
- Tokens: {tokens_in} in / {tokens_out} out (per-role split in results.md)
- Wall-clock: {wall_clock} — GPU: {gpu_desc}

## Limitations

- Greedy tree search is tuned for the 6h regime; over longer horizons it would need
  broader exploration (multi-node beams, ensembling passes).
- Sigma is calibrated once at run start from baseline seeds; heavier models may have
  different noise profiles, so borderline acceptances carry some risk.
- The MENU encodes human priors from the literature; the agent searches within it and
  can only leave it via the forced-branch valve.
- {Anything the run itself exposed.}
