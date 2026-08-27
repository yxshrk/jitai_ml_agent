# Live smoke test record

## Run
- Command: `uv run python -m harness.cli run --data-dir data/synthetic --max-iters 2 --max-tokens 150000`
- Provider/models: openai — proposer/reflector `gpt-5.6-sol`, fixer `gpt-5.4-mini` (agent/models.toml)
- Run id: `run_20260828-011439-bffeac` (journal gitignored under logs/)
- Wall clock: 64.9 s; stop reason: `max_iters` (2 iterations completed as intended)

## Baseline & calibration
- Baseline node_000 = zoo/fm_torch.py; sigma from 3 seeds (42/43/44) = 0.00247
- Baseline validation metrics: gauc 0.6223, ndcg5 0.7044, primary 0.6633

## Iterations
| n | action | hypothesis (abridged) | primary | accepted | tokens in/out |
|---|--------|----------------------|---------|----------|----------------|
| 1 | draft (Tier 1) | hybrid pointwise logloss + within-user BPR pairwise loss | 0.6559 | no (below champion) | 2955 / 3533 |
| 2 | draft (Tier 2) | FM embedding dim 16 -> 32, expect +0.002-0.005 | 0.6623 | no (delta negative) | 3012 / 1757 |

Both generated scripts ran cleanly first try (no fixer calls). Rejections are
correct behavior on the small synthetic split.

## Tokens & cost
- Tokens: 5,967 in / 5,290 out (11,257 total) — all `openai/gpt-5.6-sol/proposer`
- Estimated cost at conservative rates ($15/$60 per 1M): **$0.407**, recorded
  retroactively in the persistent ledger `logs/spend.json` together with two
  pre-smoke model sanity calls (~$0.001). Ledger total after smoke: ~$0.41 of the
  $25 BUDGET_USD hard cap.
