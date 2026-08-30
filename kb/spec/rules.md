# Rules of the run (frozen; from `docs` §2.2–2.6)

## The agent must
1. **Reproduce the official baseline** (FM; valid primary 0.6016) as its first step. ✅ Done 2026-08-30 — see `LOG.md`.
2. **Iterate autonomously** on train + valid only. It never reads test labels.
3. **Designate one final submission**, scored once on the hidden test split.

## Budget and stopping
- Hard cap: **50 iterations** per benchmark run; **6 h wall-clock** backstop.
- Convergence (official; normally binds first): stop when validation primary has not improved by more than
  **ε = 0.002** over the last **N = 3** consecutive iterations. The scored artifact is the validation-best
  checkpoint at that moment. An early win is never lost; what convergence costs is the unused iterations.
- "Iteration" is not defined precisely in the doc. Reading adopted until the organizers rule otherwise:
  **every train-and-evaluate cycle is an iteration and is journaled** (ADR-0006).

## Allowed / forbidden
| allowed | forbidden |
|---|---|
| any open-source library (PyTorch, LightGBM, RecBole, …) | **external training data** — any dataset other than KuaiRand |
| any paper, public solution, or public code | pretrained weights that were trained on these benchmarks' test labels |
| pretrained weights in general | reading hidden-test labels during development |
| changes to any pipeline stage | |
| KuaiRand-1k / 27k as optional bonus | |

The "hidden" test labels are physically inside the public download. Enforcement is ours: ADR-0005.

## Autonomy (Impact & Relevance, 20 %)
Scored by the **number of manual interventions** needed to reach the converged result; fully autonomous is best.
Anything a human does during a run is an intervention and must be journaled as `intervention: true`.

## Robustness (part of Technical Execution, 35 %)
Scored on how failures are handled, not how many occur: recover, retry, or route around code errors, timeouts,
and unexpected inputs; long runs must not crash, stall, or diverge before the budget is spent.

## Run-log requirements (deliverable 3) — every iteration records
hypothesis · the code diff · resulting valid GAUC / nDCG@5 · error and recovery events.
Run level: number of manual interventions, total LLM tokens (input + output), agent wall-clock, iterations used,
GPU-hours if any.

## Deliverables (docs §2.5)
1. Written project description on Devpost (tools, APIs, libraries, datasets).
2. Public repo: commented code; README with overview, setup, reproduction steps, limitations, contributions.
3. Run & iteration logs as above.
4. Final submission CSV for KuaiRand-Pure (+ bonus CSVs if attempted), a results table (valid-best GAUC / nDCG@5
   and absolute delta over baseline), and resource usage. A ~3-min video is recommended, not required.
