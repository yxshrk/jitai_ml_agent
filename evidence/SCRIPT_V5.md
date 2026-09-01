# jit.ai — TechJam 2026 Track 2 video script, Version 5 (record this)

Target 3:05 at ~180 wpm (about 560 spoken words). Scenes unchanged in order;
scene 6 (harness) and scene 9 (now "The ceiling") have new text. Every number
has a file behind it (evidence/POSTMORTEM_1SEP_FINAL_RUNS.md, RESULTS_AND_RESOURCES.md).
Bracketed directions are not spoken.

### Opening (0:00 to 0:18) [#s-title]
Hi, we're team jit.ai, and this is our submission for TikTok TechJam 2026, Track 2.
We built an autonomous research agent that trains models to rank videos by how likely a user is to long-watch them. In one clean run, with zero mid-run interventions, it took the official baseline from 0.6016 to 0.6056.
Our team is Rohan Kulshrestha, Aditya Ghosh, Yash Raj Khandelwal and Avinash Parthiban Elangovan. Let's get into it.

### The loop (0:18 to 0:30) [#s-loop, one key per word]
Every iteration has four steps: diagnose, treat, retrain, measure. The agent reads the last model's validation curve and names the failure mode, and picks a treatment aimed at it; each measurement feeds the next diagnosis.

### The system and the rule (0:30 to 0:50) [#s-arch, three keys]
Here is the full system. The agent proposes; the harness executes, scores and journals.
The competition's convergence rule ends the run: an iteration only counts as progress if it beats the best so far by more than 0.002. Three in a row that fail, and the run stops.

### The run we submitted (0:50 to 1:40) [#s-log, one key per iteration]
Now the run we submitted. Six iterations, seventeen minutes, on a CPU. It reproduces the baseline at 0.6018.
Iteration one: overfit diagnosis, a stage-matrix sweep. 0.6020. Dead end.
Iteration two: recency weighting alone. 0.6017. Dead end. Two strikes.
Iteration three: a package-dial sweep, tuning the regularised DCN-lite package with its own two-stage search inside one iteration. 0.6042. Accepted, strikes reset.
Iteration four: the proposal crashes. Logged as void, the loop carries on.
Iteration five: a watch-ratio auxiliary loss. 0.6030. Dead end.
Iteration six: an ensemble-design sweep. Seven seed members trained, three validation-selected, per-user rank average. 0.6056. Accepted, but under 0.002, so third strike. The rule fires and the run stops itself.

### No person (1:40 to 1:45) [#s-noperson]
No person chose an experiment, edited a script, or restarted anything.

### The harness (1:45 to 2:27) [#s-harness, point at cards]
The agent's judgment matters, but the harness matters just as much. Six things.
One, a structural leakage guard: the hidden test window is never mounted in the agent's workspace.
Two, a calibrated acceptance gate: three baseline seeds set the noise floor, and borderline gains must repeat.
Three, method cards. Forty-two, from cited literature, each carrying its measured status from about 250 experiment cells across roughly 146 runs.
Four, decision benches on real states. We freeze the exact situations where a live run went wrong and replay the agent against them after every change. Fixes may only be evidence corrections or general principles, never canned answers.
Five, typed ensemble plans: the agent writes the plan, deterministic code runs it, and if nothing beats the incumbent, the incumbent stays. No LLM in the measurement path.
Six, it fails safely. Every new script first runs a one-epoch smoke test, and a gate throws out anything that scores like broken code before we spend a full training on it. When the agent improves its best script, it edits that script instead of rewriting it, so working code stays working. And if the API drops a call, we retry it instead of counting it as a failed experiment.

### Honest choices (2:27 to 2:50) [#s-principles]
Our first principle is the clean run: no seed scripts, no recipe handed over; every turn you saw was the agent's own. And we chose honesty over score more than once. By blending across seeded runs by hand I reached 0.6065, which shows what an agent could do if it kept artifacts across runs and ignored the convergence rule; a blend of the agent's own artifacts reaches 0.6058; a post-run ensemble on the bonus benchmark hit 0.6802. None was the agent's own single run, so none is submitted. All are disclosed. Even the agent's own gains have to repeat on a fresh seed before they count.

### What we learned on the last night (2:50 to 3:10) [#s-model, "The ceiling"]
On the final night we turned the agent on itself: three hardened harness generations tried to beat our own champion, and none did. The best single model ever, 0.6051, and everything that run built re-combined by hand, 0.6055, both landed under it. So the champion sits at the measured ceiling of one run, and we know exactly what separates it from human-level research: implementation fidelity and keeping what it builds, not judgment. Every failure that taught us that is journaled, benched, and fixed.

### Close (3:10 to 3:22) [#s-receipts]
That run: six of a possible fifty iterations, seventeen minutes, 115 thousand tokens, no GPU, zero mid-run interventions. From 0.6016 to 0.6056. We're team jit.ai. Thank you.

### Where the innovation is said (no extra beat needed; judges' "Innovation & Insight" criterion)
Scene 6 already names the novel pieces in order: benches on real states with a fix constitution (04), typed ensemble plans with a deterministic can't-lose executor (05), the measured-margin smoke gate + constrained patches + outage retry (06). Scene 9's table is the insight: the ceiling, the tiers, and the named gap.

### Priority cuts (V5 runs ~3:20 at 180 wpm; cut in this order to reach ~3:05)
1. Scene 8: drop "Even the agent's own gains ... before they count." ~4 s
2. Scene 9: drop "Every failure that taught us that is journaled, benched, and fixed." ~4 s
3. Scene 6, item six: drop the outage clause. ~4 s
