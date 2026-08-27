# mle-agent — TechJam 2026 Track 2

Autonomous ML research agent for KuaiRand-Pure (GAUC / nDCG@5 vs FM baseline 0.5946).
See CONTRACTS.md first. Layout:
- harness/   the loop: apply diff -> train (timeout) -> evaluate -> log -> converge?
- zoo/       hand-verified reference models (torch FM, DeepFM, BPR, multi-task)
- agent/     LLM proposer / reflector / fixer + routing + metering
- evidence/  log renderer: trajectory chart, results table, resource totals
- data/      synthetic/ fixtures now; real KuaiRand-Pure + starter kit when provided
- tests/     pytest; every module gated
