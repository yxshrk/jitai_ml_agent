# jit.ai — TechJam 2026 Track 2 video script, Version 6 (2:45)

Target 2:45 at ~175 wpm (about 470 spoken words). Same scene order. Bracketed directions are not spoken.

### Opening (0:00 to 0:14) [#s-title]

Hi, we're team jit.ai, and this is our submission for TikTok TechJam 2026, Track 2. We built an autonomous research agent that trains models to rank videos by how likely a user is to long-watch them. In one clean run, with zero mid-run interventions, it took the official baseline from 0.6016 to 0.6056.

### The loop (0:14 to 0:23) [#s-loop, one key per word]

Every iteration has four steps: diagnose, treat, retrain, measure. The agent reads the last model's validation curve, names the failure mode, and picks a treatment aimed at it.

### The system and the rule (0:23 to 0:37) [#s-arch, three keys]

Here is the full system. The agent proposes; the harness executes, scores and journals. And the competition's rule ends the run: three iterations in a row without beating the best by 0.002, and it stops.

### The run we submitted (0:37 to 1:17) [#s-log, one key per iteration]

Now the run we submitted. Six iterations, seventeen minutes, on a CPU. It reproduces the baseline at 0.6018. Iteration one: overfit diagnosis, a stage-matrix sweep. 0.6020. Dead end. Iteration two: recency weighting. 0.6017. Dead end, two strikes. Iteration three: a package-dial sweep, tuning the regularised DCN-lite package with its own search inside one iteration. 0.6042. Accepted, strikes reset. Iteration four: the proposal crashes. Logged as void, the loop carries on. Iteration five: a watch-ratio auxiliary loss. 0.6030. Dead end. Iteration six: an ensemble-design sweep, seven seed members trained, three selected. 0.6056. Accepted, but under the threshold, so the run stops itself.

### No person (1:17 to 1:22) [#s-noperson]

No human intervention at all. No one chose an experiment, edited a script, or touched anything.

### The harness (1:22 to 1:55) [#s-harness, point at cards]

The agent's judgment matters, but the harness matters just as much. Six things. One, a structural leakage guard: the hidden test window is never mounted in the agent's workspace. Two, a calibrated acceptance gate: three baseline seeds set the noise floor, and borderline gains must repeat. Three, method cards: forty-two methods from cited literature, each carrying its measured status across roughly 146 runs. Four, decision benches: we freeze the exact situations where a live run went wrong and replay the agent against them after every change. Five, typed ensemble plans: the agent writes the plan, deterministic code runs it, and if nothing beats the incumbent, the incumbent stays. Six, it fails safely: a one-epoch smoke test throws out broken scripts before any full training, and improvements edit the best script instead of rewriting it.

### Honest choices (1:55 to 2:12) [#s-principles]

Our first principle is the clean run: no seed scripts, no recipe handed over; every turn you saw was the agent's own. And we chose honesty over score. Blending across seeded runs by hand, I reached 0.6065; a blend of the agent's own artifacts reaches 0.6058; a post-run ensemble on the bonus benchmark hit 0.6802. None was the agent's own single run, so none is submitted. All are disclosed.

### What we learned on the last night (2:12 to 2:30) [#s-model, "The ceiling"]

On the final night, three hardened harness generations tried to beat our own champion, and none did. The best single model ever, 0.6051, and everything that run built recombined by hand, 0.6055, both landed under it. The champion sits at the measured ceiling of one run, and the gap to human-level research is implementation fidelity, not judgment.

### Close (2:30 to 2:40) [#s-receipts]

Six of fifty iterations, seventeen minutes, 115 thousand tokens, no GPU, zero mid-run interventions. 0.6016 to 0.6056. We're team jit.ai. Thank you.
