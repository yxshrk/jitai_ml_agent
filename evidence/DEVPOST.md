# MLE Agent: autonomous research for KuaiRand long-view prediction

*TikTok TechJam 2026, Track 2*

## What we built

We built an autonomous ML research pipeline that turns each iteration into an auditable solution-tree node: a hypothesis, runnable script, diff, learning curve, official metrics, and any recovery event. The loop is **diagnose, select from cited method cards, implement, sigma-gated acceptance, official convergence**. A selector diagnoses the parent curve, chooses one eligible card from `agent/METHODS.md`, cites it, and records a rejected alternative. The proposer implements the smallest coherent whole-file change; the harness owns execution, rollback, and stopping.

Seeds are mechanical: seed 42 explores, promising changes are confirmed at seeds 42/43/44, and closes are agent-designed. The designated run trained 7 ensemble members and validation-selected 3 (seeds 42-44, per-user rank average). The card library carries what earlier runs measured, so failures are not repeated; a reflector runs during stagnation and stores a final self-critique without automatically applying it. The designated runs recorded **zero mid-run human interventions** (the official definition: only behavior-changing actions during a run count).

## How it addresses the problem statement

The system maps directly to Track 2's three tasks. First, every run reproduces the official FM baseline; the designated run reproduced 0.601838 against the published 0.6016. Second, it iterates across architecture, objectives, features, regularization, data weighting, optimization, and ensembling using cited methods on train and validation only. Third, it designates one converged winner for the single final test submission and improves validation primary over baseline.

KuaiRand-Pure runs end to end under the official convergence rule; interventions are counted; smoke tests, timeouts, output validation, one fixer attempt, and route-around logic handle failures. Each journal contains the required hypothesis, parent diff, GAUC, nDCG@5, primary, and recovery record. Reports include iterations, wall-clock time, tokens, and GPU-hours.

## Results

The progression is concrete: official Pure baseline **0.6016**, compact single model **0.6047 ± 0.0003** over three seeds, then a fully unseeded agent run reaching **0.605575** (+0.0040 validation primary over baseline). That run executed its own two-stage random hyperparameter search inside one iteration, finding non-obvious dials (dropout 0.18, weight decay 9e-5, LR decay x0.57), then designed its own ensemble: 7 seed variants trained, 3 validation-selected, combined by per-user rank average. Earlier seeded runs (0.60513) and a val-greedy seed pool (0.60602) are disclosed as development evidence only.

On bonus KuaiRand-1K the agent line kept climbing: a 48-cell factorial run discovered a regime inversion vs Pure (pure logloss, no recency) and designated 0.6524; a later run then discovered causal session features (gap since last impression, session position, hour and weekday crosses) worth **+0.019**, reaching **0.66892**. That number was triple-audited before we believed it: independent tie-aware re-evaluation matched exactly, fresh-seed replications scored 0.674/0.677, and within-hour row-shuffle invariance held at 0.66517. On KuaiRand-27K the same recipe reaches **0.67263** on GPU as an out-of-protocol scaling demo.

The auditable ledger contains **~250 measured cells across 139 completed disclosed runs** (snapshot 31 Aug; logs/RUNS.md plus per-run journals). Three levers survived: **DCN-lite**, **seven-day recency weighting**, and **strong joint regularization plus rapid learning-rate decay**. Sequence modeling was refuted: its affinity prerequisite scored 0.6035, only +0.0019 over baseline, so DIN-lite was correctly gated off. Watch-time objectives also stayed below epsilon (ordinal watch ratio 0.6033, CWM-style censored auxiliary 0.6022).

The field curve showed "less is more." At seed 42 the official five-field L0 model led at 0.604335; kitchen-sink L5 fell to 0.601740. With strong regularization, confirmed L0 reached **0.604660 ± 0.000309**, while strong L5 reached only 0.602991 and did not qualify for confirmation. User affinities, sparse user crosses, and co-visitation initialization likewise failed to beat the controlled stack. The useful signal was modest temporal distribution-shift correction, not more identities or capacity.

## Methodology rigor

We use a fixed seed-42 explore, seeds 42/43/44 confirm protocol. A win must clear the **0.002 epsilon floor**. Three baseline seeds calibrate sigma; changes at or above two sigma are accepted, grey-zone changes receive reseeds with a repeatability check, and regressions are reverted. Convergence is three completed iterations without improvement above 0.002, with a 50-iteration or six-hour backstop.

Leakage prevention is structural: hidden test data is physically absent from the train/validation workspace and available only to the private final-submission step. Final models are retrained on **train only**, following organizer guidance. Transfer was triangulated with full validation, a discounted late-validation slice, and an evaluation-only random-exposure window; only relative ordering is meaningful there because the exposure policy differs from hidden test.

## Testing the agent's judgment like code

The endgame is a typed capability: the agent emits a cross-family ensemble plan as data, and a deterministic harness executes it (cheap probes of diverse families, a blend map over probe predictions, full training for the complementary ones, a re-verified final blend). When a live run made a measurably wrong endgame call, closing with a same-family ensemble whose ceiling could not clear epsilon, we froze that exact decision state as a benchmark fixture, root-caused it to stale guidance, and fixed it with a general epsilon-arithmetic principle rather than a scenario-specific answer. The fixture now passes 3/3 and the corrected run was relaunched. The same decision benches gate model choice: our top tier passes all judgment scenarios at low reasoning effort, the workhorse needs medium for the hardest one. No prompt or knowledge fix ships without a passing bench, and fixes may only be evidence corrections or general principles.

## Autonomy & feasibility

The intervention count is machine-verifiable: every journal record has an `intervention` boolean and each run report aggregates it. The designated runs record **0 mid-run interventions**; runs tainted by mid-run knowledge edits were discarded and disclosed rather than argued. Recorded converged Pure runs took roughly 10-25 minutes on CPU. The designated Pure run (bigclock_07, six iterations) used **115,315 tokens and 17.0 minutes wall-clock** with no GPU; the designated 1K run (omega_1k, eight iterations) used **320,048 tokens and 344.8 minutes** with ~5.7 RTX-4090 GPU-hours (wall-clock is the scored measure). Campaign aggregate: **~10.6M tokens, ~143 run-hours** across ~146 completed runs (139 at the 31 Aug snapshot plus the 1 Sep hardening wave, whose runs and post-mortem are disclosed in evidence/POSTMORTEM_1SEP_FINAL_RUNS.md).

`gpt-5.6-sol` serves as selector, proposer, and reflector; `gpt-5.4-mini` is the fixer. The harness owns seeding, timeouts, acceptance, best-node selection, and convergence.

## Dev tools, APIs, libraries, and datasets used

- **Development/runtime:** Python 3.11, `uv`, Git, and the project harness.
- **ML and analysis:** PyTorch, NumPy, Optuna, and Matplotlib.
- **API:** OpenAI Responses API with direct HTTP integration and per-role token metering.
- **Data and evaluation:** KuaiRand-Pure required benchmark, KuaiRand-1K bonus benchmark, and the organizer starter kit's official evaluator and submission schema. No external training data or pretrained weights were used.

## The knowledge loop

The system is not one run but a research loop across runs. An unseeded run invented a temporal pair-sampling kernel (an untried speculative card that realized +0.0014, double its predicted gain); that measurement was distilled back into the method library; later clean runs draw on it as cited evidence. The same loop worked on 1K: the causal session-feature discovery was audited, carded, and then measured on Pure by a subsequent run at +0.0002, an honest negative (Pure's sessions are too sparse, exactly as a consulted reviewer predicted). Negative results are first-class: watch-time objectives, listwise losses, attention, feature crosses, hard-negative BPR, and post-hoc run blending are all measured dead and disclosed.

## The final-night stress test (1 Sep)

After freezing the designation we spent the last night trying to beat it with progressively hardened harnesses — and treating every failure as data. Five more runs (three memory-tier, two literature-only) plus a one-epoch "fast-forward" shakeout produced: a smoke sanity gate calibrated on measured broken-vs-sane populations; constrained-patch "improve" proposals (accepted artifacts evolve byte-identically instead of being re-typed); retry-without-penalty for provider outages after a 503 ended a run mid-decision; and six general decision principles verified on real-state benches (for example: on a probable-final iteration, bank the reliable measured gain instead of chasing a streak reset). None of it beat the designation — the best single model ever (0.605102) and exhaustive post-hoc banking of that run's own artifacts (0.60546) both landed under it — which is itself the report's cleanest finding: the designated champion sits at the measured single-run ceiling (0.6055-0.6060), and the campaign's remaining gap to human-assisted evidence (0.6061) is implementation fidelity and artifact persistence, not decision quality. Full dissection: evidence/POSTMORTEM_1SEP_FINAL_RUNS.md.

## Limitations & what's next

The plateau around 0.6045-0.6055 appears structural at this scale: added fields, capacity, sequences, watch-time auxiliaries, and blends mostly overfit or land inside noise. Next we would test larger KuaiRand variants, recalibrate sigma for different model families, and broaden search beyond greedy improve-best. Two structural changes matter more than any of those: sharing memory *and artifacts* between runs rather than only between iterations (the card library already carries measured evidence across runs; trained members and probe predictions do not, and the 0.6065 cross-run blend we declined shows what that unlocks), and a harness-owned sweep executor so the agent plans a search as data and never re-implements it. Given no convergence limit or wall-clock cap, the natural endgame is a cross-family blend the run keeps open until it decides to close; under the competition's rule the run spends its last three iterations proving it should stop instead.

The division of labor is plain: the agent is an executor, not a director. A human designed the method-card space, search levers, safety constraints, and acceptance policy. The agent diagnosed runs, selected among those cited choices, wrote and repaired code, executed experiments, rejected unsupported gains, converged, and critiqued itself. Its autonomy is real within that deliberately human-authored research boundary.

## Final designations (frozen 31 Aug)

- Pure: `run_bigclock_07`, 0.605575 (clean, unseeded, official convergence; no final-wave run exceeded it; novel_l1 confirmed at 0.605496, statistically equal, discussed in the README).
- 1K: `run_omega_1k`, 0.66892 (faithful A-form replay CSV; recorded run value claimed, replay caveat disclosed in SUBMISSION_RECIPE.md).
