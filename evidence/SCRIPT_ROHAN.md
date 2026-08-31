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
