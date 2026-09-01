## Inspiration

Track 2 asks for an agent that does ML research on its own: reproduce a baseline, iterate, submit one converged winner. The easiest version is a loop that calls an LLM and keeps whatever scores higher. We did a little more than that: an agent whose every claim we could audit, whose gains beat measured noise, whose run could be replayed decision by decision, and whose failures were recorded as carefully as its wins. The line that ended up on the site is the design brief: **half of this system is an agent, and the other half exists to check its work.**

## What it does

Given the KuaiRand-Pure starter kit, the agent reproduces the published FM baseline (0.6016, ours 0.601838), then iterates with no person steering:

1. **Diagnose.** Read the last training curve and name the bottleneck (overfit, undertraining, plateau, seed noise).
2. **Treat.** Pick one method from a 42-card library of published ranking techniques. Each card carries the mechanism, when it applies, and what it has measured on this dataset so far. The pick, the rejected alternatives and the reasoning are journaled.
3. **Retrain.** Write a complete training script (or, when improving an accepted parent, an exact search/replace patch) against a frozen input/output contract, on the same time-based train/validation split every iteration.
4. **Measure.** Score with the official evaluator (per-user GAUC and nDCG@5, averaged into one primary). Keep a change only if its gain clears a noise-calibrated gate; stop by the competition's rule (three iterations without +0.002).

The designated run, `bigclock_07`, went **0.6016 → 0.605575** in six iterations, 17 minutes on a CPU, 115,315 tokens, **zero mid-run human interventions**. Inside one iteration it ran its own two-stage hyperparameter search and found non-obvious dials (dropout 0.18, weight decay 9e-5, LR decay ×0.57); in its last it designed its own seed ensemble (seven trained, three validation-selected, per-user rank average). On the bonus KuaiRand-1K benchmark a later run discovered causal session features (gap since last impression, session position, hour/weekday crosses) worth +0.019 and reached **0.66892**, a number we triple-audited before believing (tie-aware re-evaluation matched exactly; fresh seeds scored 0.674/0.677; within-hour shuffle invariance held).

## How we built it

Five days, 354 commits, ~146 completed runs, and many many more incompleted.

**Day 1 (28 Aug): contracts before code.** We wrote the node-script contract, the journal schema and the acceptance policy first, then reproduced the official evaluator on real data with tests. Only then did an LLM touch anything.

**Days 2-3: the search space is human, the search is not.** We measured the levers by hand (DCN-lite, seven-day recency weighting, strong joint regularization with rapid LR decay) and distilled them into method cards with their evidence. A five-field model with strong regularization beat every richer variant ("less is more": L0 0.604660 ± 0.000309 vs L5 0.602991).

**Day 4 (31 Aug, 121 commits): testing judgment like code.** Every time a live run made a measurably wrong decision, we froze that exact state (verbatim journal lines, the real curve) as a benchmark fixture and fixed it with a general principle, never a scenario-specific answer. The clean-knowledge bench went 4/10 → 10/10 on methodology alone. We also added a typed endgame: the agent emits an ensemble plan as data and a deterministic harness executes it.

**Day 5 (1 Sep): the stress test.** With the designation frozen, we spent the final night trying to beat it with five progressively hardened harness generations: a smoke sanity gate calibrated on measured broken-vs-sane populations, constrained-patch improves (accepted code evolves byte-identically instead of being re-typed), retry-without-penalty for provider outages, reference snippets injected only for the selected card. None beat 0.605575. The best single model ever (0.605102) and exhaustive post-hoc banking of that run's own artefacts (0.60546) both landed under it.

**The harness (fixed code, no LLM, never trusts the agent):** screens scripts for test-set access, smoke-runs them for 360 s behind a sanity gate, trains under a timeout, evaluates officially, gates on three-seed sigma with grey-zone reseeds, journals hypothesis/diff/metrics/recovery, and owns convergence. The hidden test set is physically absent from the workspace. 160 tests and the decision benches gate every prompt or knowledge change.

**Stack:** Python 3.11 with `uv`, PyTorch, NumPy, Optuna, the OpenAI Responses API over direct HTTP with per-role token metering (`gpt-5.6-sol` as selector/proposer/reflector, a smaller model as fixer), and a static "flight recorder" site (HTML/CSS/JS, no framework) that replays the designated run from its journal file.

## Challenges we ran into

- **Noise is bigger than most "wins."** Seed sigma on Pure is ~0.0004; epsilon is 0.002. We calibrated sigma from three baseline seeds every run and made the gate, not the LLM, own acceptance. Grey-zone gains get reseeded and must repeat.
- **Decision quality stopped being the bottleneck; implementation fidelity became it.** Once the selector was at bench-zero on real states, fresh re-implementations of a known-good recipe still spread 0.592-0.600. Reference snippets and exact-hunk patches closed most of that gap.
- **Dishonest endgames are tempting.** Runs like to "close" with a same-family ensemble whose ceiling cannot clear epsilon. We wrote the epsilon arithmetic into the guidance (bank the reliable measured gain on a probable-final iteration; a new family after a failed close) and verified it on replayed states.
- **Infrastructure bites at 3 a.m.** A 503 ended a run mid-decision (now retried, never a strike); a debug route rebuilt a dead card three times (now returns to selection); a `pkill` pattern matched its own launch command. Each became a test.
- **Saying no to a higher number.** We had a 0.6065 cross-run blend, a 0.6061 hand-built model and a 0.6802 post-run 1K result. All are human-assisted, all are disclosed as evidence, none was submitted.

## Accomplishments we're proud of

A clean, unseeded, converged run at 0.605575 that sits at the measured single-run ceiling (0.6055-0.6060), established by exhausting our own alternatives rather than by assertion. A machine-verifiable intervention count of zero. Every number on the site and in the report has a run directory behind it. And a site where a judge can step through the run one iteration per keypress and read the agent's own reasoning at each one.

## What we learned

Autonomous research needs accurate memory and verified execution more than clever prompting. Mechanisms must be re-swept, and never grafted onto fixed dials. Negative results are invaluable. The agent is an executor, not a director: a human designed the method space, the levers, the constraints and the acceptance policy; within that boundary the agent's autonomy is real, and the boundary is what makes its results trustworthy.

## What's next

A harness-owned sweep executor (the agent plans searches as data and never re-implements them), search beyond greedy improve-best, and the same loop on the larger KuaiRand variants. Also, the ability to share memory between runs, rather than just iterations, and ideally, no convergence limit or wall clock, so experiments can be run more freely, and a blend approach can be used.

## Resources used

Designated Pure run: 6 iterations, 17.0 min wall-clock, 115,315 tokens, no GPU. Designated 1K run: 8 iterations, 344.8 min, 320,048 tokens, ~5.7 RTX-4090 GPU-hours. Campaign: ~146 runs, ~10.6M tokens, ~143 run-hours, all disclosed in `logs/RUNS.md`, `evidence/RESULTS_AND_RESOURCES.md` and `evidence/POSTMORTEM_1SEP_FINAL_RUNS.md`.
