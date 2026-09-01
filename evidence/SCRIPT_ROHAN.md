# TikTok Tech Jam 2026 — Track 2 Submission Script

## Version 1 — Original dictation (verbatim)

Hi guys. We are Team JIT AI. This is our submission for TikTok Tech Jam 2026 Track 2. Basically, this is an agent that teaches, that trains models to rank videos based on how likely they are to be watched, long watched by a user. Ours managed to go from value X to value Y with absolutely zero human interventions during the run. Our team is made of myself, Rohan Kulshreshth, Aditya Ghosh, Yashraj Khandelwal, and Avinash Parthiban Elangovan. Anyway, let's get into it.

So, as you can see here, it takes a couple iterations to get to its final place, and I'm going to walk you through the architecture and what one real run even looks like. So first of all, what is the steps it takes for each iteration? First of all, it diagnoses, which means it looks at how the last model was trained and what exactly was the failure mode, what caused it to not perform as it should have. Then it does treatment, which means it decides what to do differently in this upcoming run to avoid the previously encountered failure mode. Then it actually executes the training, which is the retraining, and then it measures the output again to repeat the step from the diagnosis again.

So let me show you the whole architecture diagram. You can see it here. It starts on the train block when you would attempt the baseline. It would then be evaluated, and keep in mind there's a rule for this competition where if you get below 0.0002 of an increase, that automatically means you only have three attempts at it. And if you don't go above 0.0002 in three attempts, it converges and your run is over. If it does, it keeps going. It will keep a memory of that run to talk about what it encountered, what it hypothesized, what it ended up implementing and how it worked. So this is useful for it to look back on in future iterations. Then it goes ahead with diagnosis. It does the treatment and rewrites the training script. It does the training as well, and then the cycle repeats.

So let's walk through a real run. Here we see it starts at baseline. This is exactly when the start point is, just training the base model. The first thing our model tried is a stage matrix, da da da da da da da da da. And finally, it reaches the final value of Y. No human interventions at all, and clever handling of errors, of issues in training, etc. Every single thing was handled by the agent by itself.

So the agent is surely important, and how it makes the right decisions is important. But what's even more so important is the way, is the harness the agent exists inside. So here are some of the six key things we factored in to make this harness as robust as it is.

First of all, there's the structural leakage guard. This is one of the most important things. We need to make sure that the hidden test window is never mounted into the agent's workspace. It should never be able to peek into the values it will be tested on, because this makes the whole thing useless. If it already knows the answers, there is no test.

Second of all, an acceptance gate. We want to make sure that we don't account, or we try to mitigate random noise being taken as positive improvement.

Third of all, all of the methods and the tricks it tries to sidestep every issue encountered in diagnosis comes from real academic literature. It's not just the model saying, I don't like this, maybe let me just try tune this number in this way or that way. It's a very targeted, specific way of correctly targeting and addressing the failure modes. So there's more than 42 cards, all from cited literature, and maybe one or two that I've personally thought of or tried to workshop and put it together, and all of them have been tested and tried out countless times. You can see our website for the full breakdown of all the cards and how well they work.

Then of course there's a decision bench, which is effectively like a test bench for just the agent without actually having to be plugged into the whole training system, where it can just be tested on the way it makes decisions at every turn. You want to know how it performs in every situation it might find itself in a real run without wasting the time, effort, or money in a real run.

Then of course there's ensemble plans. It needs to know how to assemble itself in the right combination to blend complementing models together. There is no LLM. It's totally deterministic and mathematical.

And finally, there's a telemetry-gated diagnosis, because you only want the side curves as reasons to do things different if those are heavily recorded and there's enough information to be gleaned from them for it not to just end up being random noise and potentially harm your own agent.

So here, like I was saying, here are a bunch of the methods that we were talking about. I'm not going to go through all of them now, but if you go to our website, you'll see all of them here.

So here's the key ethos and way of thinking that we really kept in mind when making this whole thing. We only wanted clean runs. What that means is the run that we submit should never be as a result of having run many, many, many, many runs with many, many iterations, knowing the right recipe, and then just handing that to our agent. That's effectively saying, this is the way you get out of a maze. Let me hand you the map. Let me watch you get out of the maze. There's nothing impressive about that. It needs to take each turn, figure out where it went wrong, try it again, try it again until it gets out by itself. The point of an auto research agent isn't that it can follow rules; it's that it can figure out the right path out by itself.

Second of all, method cards, like I was saying, the 42 cards we were talking about earlier, it's all very cited, all based on real things. It's not a free-flowing agent which just vibes up random things and goes as it pleases. It's all based in real academia to avoid hallucination, etc.

Second of all, the floor, which is to avoid noise. So it's very carefully calibrated against our noise floor and making sure that borderline gains are repeatable and not random.

Then of course, failure data. Every single issue that we've ever, ever encountered throughout this whole thing, which is probably over a few thousand runs, a few thousand experiments, every single thing is recorded and journaled. Not everything is made available to the agent, but we've taken liberties on deciding which ones to expose to the agent to make its decision-making better. And that being said, you don't want to overload it.

And of course, the agent's moves at every single point are pretty specific and limited. It's not given complete control. So the harness is the one that executes everything. The agent is the one that can call tools and dictate where it goes next.

We ended up going with a medium effort on 5.6, Seoul, because high was not necessarily any better. In fact, sometimes it overthought things and produced worse results in the end game, specifically in the blending situation. You'll find more on that on the website. Low was didn't think enough. So medium was really the right sweet spot.

So we've done more than 139 runs isn't even all. It's just with our latest mechanism. Here are the numbers that. In conclusion, this is the stat that our agent has. It takes 17 minutes. It takes roughly six iterations on average, and this is our delta. This is our final answer, and this is this. Thank you.

---

## Version 2 — Tightened presentation script (Rohan's draft, placeholders to fill)

### Opening

Hi, we’re Team JIT AI, and this is our submission for TikTok TechJam 2026, Track 2.

We built an autonomous research agent that trains ranking models to predict how likely a user is to long-watch a video. In one clean run, it improved our score from **[X]** to **[Y]**, with zero human intervention.

Our team is Rohan Kulshreshth, Aditya Ghosh, Yashraj Khandelwal, and Avinash Parthiban Elangovan.

Let’s get into it.

### The autonomous loop

I’ll first explain the architecture, and then show you what happened during a real run.

Every iteration has four steps: **diagnose, treat, retrain, and measure**.

First, the agent diagnoses the previous model. It examines how the model was trained, how it behaved, and which failure mode most likely limited its performance.

Second, it selects a treatment: a targeted change intended to address that specific failure mode.

Third, the harness rewrites and executes the training script.

Finally, it measures the new model. That evidence becomes the input to the next diagnosis, and the cycle repeats.

### Architecture and stopping rule

Here is the complete architecture.

The process begins by training and evaluating a baseline model. Under the competition’s convergence rule, an improvement below **0.0002** counts as a non-improving attempt. If the system fails to exceed that threshold for three attempts, the run converges and stops. Otherwise, the loop continues.

After every attempt, the system writes a structured memory: what it observed, what it hypothesized, what it changed, and whether that change worked. The agent can use this evidence in later iterations instead of repeatedly rediscovering the same lessons.

It then diagnoses the current model, chooses a treatment, rewrites the training script, retrains, evaluates, and repeats.

### A real clean run

Now let’s walk through a real run.

It begins with the baseline: **[BASELINE SCORE]**.

In the first iteration, the agent identifies **[FAILURE MODE]** and selects **[METHOD / “STAGE MATRIX” — CONFIRM NAME]**. That changes the score from **[VALUE]** to **[VALUE]**.

Next, it observes **[DIAGNOSIS]**, applies **[TREATMENT]**, and reaches **[VALUE]**.

**[CONTINUE THE RUN CHRONOLOGICALLY: one sentence per iteration, including any error and how the system recovered.]**

Finally, after **[N]** iterations, it reaches **[Y]**.

No person chose the next experiment, edited the script, or recovered the run when something failed. The agent and its harness handled the entire process autonomously.

### Why the harness matters

The agent’s reasoning matters, but the harness around it matters just as much. We built six safeguards and capabilities into it.

**First: a structural leakage guard.** The hidden test window is never mounted inside the agent’s workspace. The agent cannot inspect the data on which it will ultimately be judged. If it could see the answers, the evaluation would be meaningless.

**Second: an acceptance gate.** A higher score is not automatically treated as a real improvement. The gate helps prevent random variation from being mistaken for progress.

**Third: literature-grounded method cards.** The agent does not invent arbitrary fixes or blindly tune parameters. It chooses from more than 42 targeted methods, primarily grounded in cited academic literature and mapped to specific failure modes. We document the complete set, the sources, and our test results on the website.

**Fourth: a decision bench.** This tests the agent’s decisions independently of the expensive training loop. We can place it in representative scenarios and evaluate what it would do next, without spending the time and compute required for a full run.

**Fifth: deterministic ensemble planning.** When complementary models need to be combined, the blend is selected mathematically and deterministically. No LLM chooses the ensemble weights.

**Sixth: telemetry-gated diagnosis.** The agent may use secondary training curves only when they are recorded with enough coverage and reliability to support a diagnosis. Otherwise, noisy telemetry could send the next experiment in the wrong direction.

These are only some of the available methods. The website contains the complete breakdown.

### Our design principles

Our first principle is that the submitted result must come from a **clean run**.

We do not run the system repeatedly, learn the winning sequence ourselves, and then hand that recipe back to the agent. That would be like giving someone a map through a maze and claiming they solved it.

In a clean run, the agent must choose each turn itself. It observes the result, identifies what went wrong, and decides what to try next. An autonomous research agent should not merely replay a route; it should discover one.

Our second principle is **grounded action**. The method-card library constrains the agent to concrete, testable interventions supported primarily by published research. This reduces hallucinated or arbitrary experimentation.

Third is **noise awareness**. The system is calibrated against an empirical noise floor so that marginal gains must be repeatable before we trust them.

Fourth is **learning from failure**. Across thousands of experiments, we have journaled the errors and failure modes we encountered. We selectively expose the most useful lessons to the agent: enough to improve its decisions, but not so much that irrelevant history overwhelms it.

Finally, the agent operates with bounded authority. It can call specific tools and choose the next experiment, but the harness controls execution, validates outputs, and enforces the rules.

### Model selection and evaluation

For the reasoning model, we selected **GPT-5.6-Sol at medium reasoning effort**.

Higher effort was not consistently better. In some late-stage decisions—particularly model blending—it overthought the evidence and produced worse choices. Low effort did not reason deeply enough. Medium effort gave us the best balance. The full comparison is available on our website.

We have completed more than **139 runs** using the latest version of the system alone.

**[SHOW EVALUATION RESULTS HERE: success rate, mean/median delta, variance, convergence rate, error-recovery rate, or decision-bench score—whichever you actually measured.]**

### Conclusion

To conclude, our agent takes about **17 minutes** per run, converges in roughly **six iterations on average**, and achieves a final improvement of **[DELTA]**, from **[X]** to **[Y]**, with zero human intervention during the run.

We are Team JIT AI. Thank you.

---

## Fill before recording (Rohan's checklist)

- Confirm the official event styling: “TechJam” or “Tech Jam.”
- Replace **X**, **Y**, **DELTA**, and every bracketed score.
- Confirm whether “stage matrix” is the correct method name.
- Add the actual iteration-by-iteration story; this is currently the biggest missing section.
- Verify the precise wording of the competition’s **0.0002 / three-attempt convergence rule**.
- Confirm the official model name and capitalization for **GPT-5.6-Sol**.
- Replace vague references such as “this stat,” “this number,” or “here” with the exact metric visible on screen.
- If “all 42 methods have been tried countless times” cannot be supported by logged results, use the narrower wording in Version 2.
- Decide whether “thousands of experiments” means full autonomous runs, individual training jobs, or smaller tests, and label it accurately.

## Editing note

Version 1 is intentionally untouched. Make future edits in Version 2 or duplicate it into a new version so the original dictation remains recoverable.

---

## Version 3 — Filled and verified

Target: 3:00. The spoken text is 603 words by strict count (numbers included). At 180 words per minute that is 3:21; at 160 wpm it is 3:46. Apply the priority cut list under the shot list (about 70 words) to land at 3:00 at 180 wpm; measure the first take and cut further from the same list if needed. Every number below has a
file behind it; see the corrections log after this section. Bracketed stage directions are
not spoken. Present at http://localhost:8642/present.html (scene ids in the shot list).

### Opening (0:00 to 0:20)

Hi, we're team jit.ai, and this is our submission for TikTok TechJam 2026, Track 2.

We built an autonomous research agent that trains models to rank videos by how likely a user is to long-watch them. In one clean run, with zero mid-run interventions, it took the official baseline from 0.6016 to 0.6056.

Our team is Rohan Kulshrestha, Aditya Ghosh, Yash Raj Khandelwal and Avinash Parthiban Elangovan. Let's get into it.

### The loop (0:20 to 0:33)

Every iteration has four steps: diagnose, treat, retrain, measure.

The agent reads the last model's validation curve and names the failure mode, then picks a treatment aimed at it. The harness rewrites the script, retrains and measures, and that measurement feeds the next diagnosis.

### Architecture and stopping rule (0:33 to 0:55)

Here is the full system. The agent proposes; the harness executes, scores and journals.

The competition's convergence rule ends the run: an iteration only counts as progress if it beats the best so far by more than 0.002. Three in a row that fail, and the run stops. Hard cap fifty iterations, six hours.

Every iteration is journaled: what it saw, hypothesised, changed, and whether it worked, so later iterations don't relearn it.

### A real clean run (0:55 to 1:47)

Now the run we submitted. Six iterations, seventeen minutes, on a CPU.

It reproduces the baseline at 0.6018.

Iteration one: the diagnosis is overfit, validation peaks at epoch eight then declines. It tries a stage-matrix sweep across architecture, loss and regularisation. 0.6020. Dead end.

Iteration two: recency weighting alone. 0.6017. Dead end. That's two strikes.

Iteration three: a package-dial sweep, tuning the regularised DCN-lite package jointly with its own two-stage search inside one iteration. 0.6042. Accepted, strikes reset.

Iteration four: the proposal crashes. The harness logs it as void and the loop carries on.

Iteration five: an ordinal watch-ratio auxiliary loss. 0.6030. Dead end.

Iteration six: an ensemble-design sweep. Seven seed members trained, three validation-selected, per-user rank average. 0.6056. Accepted, but under 0.002, so third strike. The rule fires and the run stops itself.

No person chose an experiment, edited a script, or restarted anything.

### Why the harness matters (1:47 to 2:17)

The agent's judgment matters, but the harness matters just as much. Six things.

One, a structural leakage guard. The hidden test window is never mounted in the agent's workspace.

Two, an acceptance gate. Three baseline seeds calibrate the noise floor; borderline gains must repeat.

Three, method cards. Exactly 42, from cited literature and our own measured evidence, each carrying its measured status from about 250 cells across 139 completed runs. One card was invented by a run itself.

Four, a decision bench. We test the agent's choices on frozen scenarios without paying for a run.

Five, deterministic ensemble planning. The agent writes a typed plan: families, seeds, blend rule. A deterministic executor probes, enumerates every blend, re-verifies, and keeps the incumbent if nothing beats it. No LLM in the measurement path.

Six, telemetry-gated diagnosis. Secondary curves are only trusted when recorded well enough to support a conclusion.

### Principles and model (2:17 to 2:42)

Our first principle is the clean run. We never learn the recipe ourselves and hand it to the agent; that's giving someone the map and watching them walk out of the maze. Every turn you just saw was the agent's own.

The model is gpt-5.6-sol at medium effort. High and extra-high made proposals truncate and fail in earlier runs. Low passed routine decisions on our bench but failed the endgame epsilon arithmetic. Medium was the measured sweet spot.

### Close (2:42 to 3:00)

That run: six of a possible fifty iterations, seventeen minutes wall-clock, 115 thousand LLM tokens, no GPU, zero mid-run interventions. From 0.6016 to 0.6056, plus 0.0040 on validation primary.

We're team jit.ai. Thank you.

### Optional 15-second beat (only if the recording comes in under 2:45)

Insert after "Four, a decision bench" or replace the model paragraph:

> We test the agent's judgment the way you test code. When a live run made a wrong endgame call, we froze that exact decision state as a bench fixture, fixed the guidance with a general principle rather than a scenario-specific answer, verified the bench passed, and resumed the run at the point of failure.

Source: logs/RUNS.md rows farm_f1, farm_f2, farm_f3; evidence/DEVPOST.md "Testing the agent's judgment like code".

### Timed shot list (present.html scenes, about 3:00)

| # | Time | Scene (present.html) | On screen | Narration | Sec |
|---|---|---|---|---|---|
| 1 | 0:00 | `#s-title` | Title card, team name and four names, headline 0.605575 | Opening, through "Let's get into it." | 20 |
| 2 | 0:20 | `#s-loop` | Four stage cards cycling: diagnose, treat, retrain, measure | The loop | 13 |
| 3 | 0:33 | `#s-arch` | Architecture SVG: agent row, harness row, hidden-test exclusion; point at the convergence check | Architecture and stopping rule | 22 |
| 4 | 0:55 | `#s-log` | Log replay, one keypress per iteration: baseline, then nodes 1 to 6 (green accepted, red dead ends, amber VOID) | Run walkthrough, one keypress per "Iteration N" | 52 |
| 5 | 1:47 | `#s-harness` | Six harness deep-dive cards; hover each as it is named | Six harness items | 30 |
| 6 | 2:17 | `#s-methods` then `#s-decisions` | 42-card library scrolling, then design-decisions scene for the clean-run principle and model choice | Principles and model | 25 |
| 7 | 2:42 | `#s-receipts` | Receipts grid: 6 iterations, 17.0 min, 115,315 tokens, 0 GPU, 0 interventions, 0.6016 to 0.605575, +0.0040 | Close | 18 |

Total 180 seconds with the priority cuts below applied. Cut in this order until the take lands at 3:00:

1. Scene 3: drop the journaling sentence ("Every iteration is journaled ... don't relearn it."). About 7 s. The log replay in scene 4 shows the journal anyway.
2. Scene 6: drop the maze sentence ("that's giving someone the map ..."), keep "We never learn the recipe ourselves and hand it to the agent." About 5 s.
3. Scene 5: drop item six (telemetry-gated diagnosis) and say "Five things" instead of "Six things". About 6 s.
4. Scene 4: cut "across architecture, loss and regularisation" from iteration one and "with its own two-stage search inside one iteration" from iteration three. About 5 s.
5. Scene 5: shorten item five to "Five, deterministic ensemble planning. The agent writes a typed plan; a deterministic executor runs it and keeps the incumbent if nothing beats it. No LLM in the measurement path." About 5 s.

---

## Changes and corrections log

Each line: what V2 said, what V3 says, and the source file that settles it.

1. Convergence rule: V2 said "an improvement below 0.0002". V3 says the run ends after three consecutive iterations that fail to improve the best-so-far by more than 0.002, with a hard cap of 50 iterations and 6 h wall. Sources: ../RULES.md line 25; harness/loop.py lines 89 to 91 and 725 to 729 (`epsilon = 0.002`, `n_converge = 3`, "accepted, rejected, or errored all count", improvement measured vs best-so-far).
2. Acceptance vs convergence: V2 conflated the acceptance gate with the convergence rule. V3 states they are separate: acceptance is a calibrated gate (three baseline seeds set sigma; grey-zone floor 0.0005 with a repeat-seed check). Sources: evidence/DEVPOST.md "Methodology rigor"; harness/loop.py line 486; CLAUDE.md "floor-v2 grey accept".
3. X, Y, delta: X = 0.6016 (official validation baseline), Y = 0.605575 spoken as 0.6056, delta +0.0040 (RESULTS file gives +0.00398) validation primary. Source: evidence/RESULTS_AND_RESOURCES.md results table.
4. Run statistics: V2's "about 17 minutes per run" and "roughly six iterations on average" described as averages. V3 attributes 17.0 min (1,019 s), 6 of 50 iterations, 115,315 tokens, 0 GPU, 0 mid-run interventions to this run, run_bigclock_07, and never says "on average". Source: evidence/RESULTS_AND_RESOURCES.md resource table; site/rundata.js meta.
5. Team name: "Team JIT AI" replaced by "team jit.ai". Source: site/present.html line 85, site/index.html line 13.
6. Team names: V2 had "Rohan Kulshreshth" and "Yashraj Khandelwal". V3 uses the site's spelling: Rohan Kulshrestha, Yash Raj Khandelwal. Rohan confirmed "Kulshrestha"; "Yash Raj" still to confirm (see open questions). Source: site/present.html line 88.
7. Autonomy claim: "zero human intervention" replaced by "zero mid-run interventions" (official definition: only behaviour-changing actions during a run count). Source: ../webinar-transcript-28aug.md lines 53 to 54; evidence/DEVPOST.md "Autonomy & feasibility"; site/CLAUDE.md hard rules.
8. Card count and provenance: "more than 42 targeted methods, primarily grounded in cited academic literature" replaced by "exactly 42 cards, compiled from cited literature and this campaign's own measured evidence; one card invented by a run itself". Sources: `grep -c "^### " agent/METHODS.md` = 42; site/methods.js; site/present.html scene caption "one card was invented by a run itself"; CLAUDE.md (temporal-pair-kernel invented by run_novel_l1, carded after the fact); agent/METHODS.md line 430.
9. "Tried countless times" (V1) and V2's checklist item: replaced by "each card carries its measured status, from about 250 measured cells across 139 completed runs". Source: evidence/DEVPOST.md line 23; README.md line 60.
10. "Thousands of experiments": replaced by 139 completed disclosed runs and about 250 measured experiment cells. Source: evidence/DEVPOST.md line 23; evidence/RESULTS_AND_RESOURCES.md disclosure section.
11. "More than 139 runs": V2 said more than 139 runs with the latest version alone. V3 says 139 completed runs (the whole campaign snapshot, 31 Aug). Source: evidence/RESULTS_AND_RESOURCES.md; README.md line 60.
12. Model and effort: V2's "high overthought the evidence in blending" replaced by the measured version: gpt-5.6-sol at medium; high and xhigh caused proposals to truncate or fail in earlier runs; low passed the routine scenarios but failed endgame_eps_math; gpt-5.6-terra passed all six scenarios at both efforts (not spoken, but available); temperature is rejected by these models (HTTP 400). Sources: logs/RUNS.md line 19 (effort grid runs 16-22, 27, 28: truncation failure); logs/bench_sweep_sol_temp.out (sol low: endgame_eps_math BAD; temp=0.2: HTTP 400); logs/bench_sweep_run.out (sol medium: 4 good, 1 ok, 0 bad); logs/bench_model_sweep.json (terra 6/6 at low and medium); CLAUDE.md conventions.
13. Model name capitalisation: "GPT-5.6-Sol" replaced by "gpt-5.6-sol" as written in the repo. Source: evidence/DEVPOST.md; agent/models.toml.
14. Ensemble planning: V2's "the blend is selected mathematically; no LLM chooses the weights" replaced by the typed-plan description (agent writes plan: families, seeds, blend rule; deterministic executor runs probes, enumerates blends, full-trains complementary families, re-verifies, keeps incumbent on tie or loss). Sources: agent/METHODS.md card diverse-family-farm-close (line 731); logs/bench_farm_close.json (13/13 checks including tie-retains-incumbent, determinism, blend-math); evidence/DEVPOST.md "Testing the agent's judgment like code". Honesty note kept out of the spoken script but recorded here: the designated run's close was the classic ensemble-design sweep (site/rundata.js node_006, chosen_method_id "ensemble-design-sweep"); the farm-close capability has one live execution (logs/RUNS.md ruby_w1, accepted, +0.0002) and the 13/13 assembler bench, but its live autonomous execution is still being tested (farm_f3 plan failed on script_source).
15. Run walkthrough (new): every value verified in site/rundata.js. node_000 0.601838 (three seeds, mean 0.6018, sigma 0.0001); node_001 stage-matrix-sweep 0.601962 rejected (grey-zone confirm failed, mean delta +0.0006); node_002 recency-weighting 0.601681 rejected; node_003 package-dial-sweep 0.604237 accepted (delta +0.0024; two-stage coarse then refine probes visible in the node); node_004 error true, "proposal unparseable/failed", VOID; node_005 ordinal watch-ratio auxiliary 0.602957 rejected; node_006 ensemble-design-sweep 0.605575 accepted (delta +0.0013; 7 members seeds 42 to 48, selected 3 seeds 42 to 44, per_user_rank_average). Stop reason "converged" (logs/run_bigclock_07/summary.json). Strike arithmetic: node_004, node_005 and node_006 (accepted but +0.0013 < 0.002) are three consecutive non-improving iterations under the official rule (harness/loop.py line 727 to 729, 1329 to 1335).
16. "Stage matrix" name: confirmed as the card `stage-matrix-sweep` ("Cross-stage combination search"). Source: agent/METHODS.md line 317; site/rundata.js node_001 chosen_method_id.
17. Event styling: "TechJam" (one word). Source: site/CLAUDE.md, evidence/VIDEO_SCRIPT.md, CLAUDE.md, ../RULES.md all use TechJam.
18. Evaluation results placeholder (V2 "SHOW EVALUATION RESULTS HERE"): removed from the spoken script; the close uses the resource table instead. Bench numbers available if wanted: decision bench 8/8 full-knowledge, 10/10 clean-mode (CLAUDE.md EVAL TOOLING), farm-close assembler 13/13 (logs/bench_farm_close.json). The full-knowledge bench cannot distinguish the agent from an always-pick-close constant policy (CLAUDE.md), so do not present it as a headline.
19. Unsupported softeners dropped: "these are only some of the available methods" (the 42 are the complete library), "many, many runs knowing the right recipe" (kept as the maze metaphor without a count).

## Open questions for Rohan

1. Surname spelling: RESOLVED. Rohan confirmed "Kulshrestha" (matches the site). V1 and V2 keep the old spelling as dictated; V3 and the site are correct.
2. First name: site says "Yash Raj Khandelwal"; you said "Yashraj". V3 follows the site. Confirm.
3. gpt-5.6-sol at medium was never run on the endgame_eps_math scenario in the saved sweeps (logs/bench_sweep_run.out ran only 5 scenarios for sol medium; endgame_eps_math was added to the fixture set afterwards). The script says medium was the "measured sweet spot", which is supported by the truncation failures at high and the low-effort endgame failure, but if you want to say "medium passed the endgame scenario" someone should run tools/bench_model_sweep.py for sol medium first.
4. Farm-close: V3 describes the capability in the harness section (item five) because it is built and benched, but the run you narrate closed with the classic ensemble-design sweep. If you would rather not describe a capability the designated run did not use, replace item five with V2's shorter wording: "the ensemble is chosen deterministically from validation; no LLM picks the members or the weights", which is exactly what node_006 did.
5. The optional 15-second judgment beat is not in the 3:00 timing. Include only if the first take lands under 2:45.
6. The 1K bonus result (0.66892, triple-audited) is not mentioned at all in your structure. It is the second-strongest number we have; a one-line mention in the close ("on the bonus 1K dataset, 0.669, triple-audited") costs about five seconds if you want it. Source: evidence/RESULTS_AND_RESOURCES.md.

## OPTIONAL ADDENDUM (1 Sep, post-final-night — one extra beat if you want it)
After the "clean runs" ethos beat, you can add (~20s):
"And on the very last night, we turned the agent on itself: five more hardened
runs trying to beat our own designated champion. None did — the best single
model ever, and even exhaustively re-combining that run's own artifacts by
hand, both landed just under it. Which is exactly the finding: our champion
sits at the measured ceiling of what one run can do, we can prove where that
ceiling is, and we know precisely which two engineering gaps — implementation
fidelity and artifact persistence — separate it from human-level research.
Every failure that taught us that is journaled, benched, and fixed in the repo."

## Version 4 — 1 Sep morning (records against the updated present.html)

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
Three, method cards. Forty-two, from cited literature, each carrying its measured status from about 250 experiment cells across roughly 146 runs. One card was invented by a run itself.
Four, decision benches on real states. We freeze the exact situations where a live run went wrong and replay the agent against them after every change. Fixes may only be evidence corrections or general principles, never canned answers.
Five, typed ensemble plans: the agent writes the plan, deterministic code runs it, and if nothing beats the incumbent, the incumbent stays. No LLM in the measurement path.
Six, it fails safely: a one-epoch smoke test with a sanity gate calibrated on measured code rejects defective scripts before any full training; improvements are constrained patches, so working code is never re-typed; and a provider outage is retried, never counted as a failed experiment.

### What we learned on the last night (2:27 to 2:52) [#s-model, "The ceiling"]
On the final night we turned the agent on itself: three hardened harness generations tried to beat our own champion. None did. The best single model ever, 0.6051, and even re-combining that run's artifacts by hand, 0.6055, both landed under it. A clean run with only literature, no measured memory, found the same opener from principles alone.
So the champion sits at the measured ceiling of what one run can do, and we know precisely what separates it from human-assisted research: implementation fidelity and keeping what it builds, not judgment. Every failure that taught us that is journaled, benched, and fixed in the repo.

### Close (2:52 to 3:05) [#s-receipts]
That run: six of a possible fifty iterations, seventeen minutes wall-clock, 115 thousand LLM tokens, no GPU, zero mid-run interventions. From 0.6016 to 0.6056, plus 0.0040 on validation primary.
We're team jit.ai. Thank you.

### Priority cuts if a take runs long (in order)
1. Scene 6, item six: drop the outage clause ("and a provider outage ... failed experiment"). ~4 s.
2. Scene 6, item three: drop "One card was invented by a run itself." ~3 s.
3. Scene 9: drop the clean-run sentence ("A clean run with only literature ... principles alone."). ~6 s.
4. Scene 4: cut "with its own two-stage search inside one iteration". ~3 s.
5. Scene 3: drop the acceptance-gate sentence. ~4 s.

## Version 5 — 1 Sep, adds the honest-choices beat (RECORD THIS ONE)

Same as V4 through the harness beat (0:00 to 2:27). Then:

### Honest choices (2:27 to 2:50) [#s-principles]
Our first principle is the clean run: no seed scripts, no recipe handed over; every turn you saw was the agent's own. And we chose honesty over score more than once. By blending across seeded runs by hand I reached 0.6065, which shows what an agent could do if it kept artifacts across runs and ignored the convergence rule; a blend of the agent's own artifacts reaches 0.6058; a post-run ensemble on the bonus benchmark hit 0.6802. None was the agent's own single run, so none is submitted. All are disclosed. Even the agent's own gains have to repeat on a fresh seed before they count.

### What we learned on the last night (2:50 to 3:10) [#s-model, "The ceiling"]
On the final night we turned the agent on itself: three hardened harness generations tried to beat our own champion, and none did. The best single model ever, 0.6051, and everything that run built re-combined by hand, 0.6055, both landed under it. So the champion sits at the measured ceiling of one run, and we know exactly what separates it from human-level research: implementation fidelity and keeping what it builds, not judgment. Every failure that taught us that is journaled, benched, and fixed.

### Close (3:10 to 3:22) [#s-receipts]
That run: six of a possible fifty iterations, seventeen minutes, 115 thousand tokens, no GPU, zero mid-run interventions. From 0.6016 to 0.6056. We're team jit.ai. Thank you.

### Where the innovation is said (no extra beat needed; judges' "Innovation & Insight" criterion)
Scene 6 already names the novel pieces in order: benches on real states with a fix constitution (04), typed ensemble plans with a deterministic can't-lose executor (05), the measured-margin smoke gate + constrained patches + outage retry (06). Scene 7's caption says one card was invented by a run itself. Scene 9's table is the insight: the ceiling, the tiers, and the named gap.

### Priority cuts (V5 runs ~3:20 at 180 wpm; cut in this order to reach ~3:05)
1. Scene 8: drop "Even the agent's own gains ... before they count." ~4 s
2. Scene 9: drop "Every failure that taught us that is journaled, benched, and fixed." ~4 s
3. Scene 6, item six: drop the outage clause. ~4 s
4. Scene 6, item three: drop "One card was invented by a run itself." ~3 s
