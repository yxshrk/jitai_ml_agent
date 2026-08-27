# Zoo results — REAL KuaiRand-Pure, VALIDATION split only

All runs: `uv run python zoo/<script>.py --data-dir real --out-dir <o> --seed 42`
(default hyperparameters: k=16, lr=1e-3, batch 8192, max 30 epochs, patience 3,
early stopping on validation GAUC). Scored with the vendored official evaluator
(data/official/evaluate.py). CPU (Apple Silicon; MPS unnecessary — epochs run
in ~1-3s). Test split untouched; baseline to beat: FM official valid primary
**0.6016** (seed std 0.0008, epsilon 0.002).

| script       | seed | gauc   | ndcg5  | primary | runtime | delta vs 0.6016 |
|--------------|------|--------|--------|---------|---------|-----------------|
| fm_bpr.py    | 42   | 0.6677 | 0.5358 | 0.6018  | 7.6s    | +0.0002 (within noise) |
| fm_feats.py  | 42   | 0.6683 | 0.5365 | 0.6024  | 11.7s   | +0.0008 (within noise) |
| dcn_lite.py  | 42   | 0.6708 | 0.5375 | 0.6041  | 13.3s   | **+0.0025 (accepted, >= epsilon)** |
| mtl.py       | 42   | 0.6698 | 0.5370 | 0.6034  | 14.5s   | +0.0018 (below epsilon 0.002) |

Honest reading:

- **dcn_lite** (MENU #4, hybrid BPR+logloss + GAUC early stopping) is the only
  run clearing the acceptance floor (+0.0025 >= epsilon 0.002).
- **mtl** (+0.0018) and **fm_feats** (+0.0008) improve but sit below epsilon;
  **fm_bpr** (+0.0002) is a wash — on this dataset the hybrid loss + GAUC-based
  model selection alone roughly reproduces the tuned official FM rather than
  beating it. Models peak at epochs 1-3 and overfit quickly.
- Single-seed numbers; per MENU, deltas under 0.002 are within seed noise.
  A quick valid-only probe (fm_bpr, lr 5e-4, patience 4) reached 0.6021 —
  still noise-level; not adopted.
- No test-split numbers are reported or were computed for any zoo model.

## 2026-08-28 sweep (full log: zoo/EXPERIMENTS.md)

| script       | config                              | seeds    | primary (mean +- std) | delta vs 0.6016 |
|--------------|-------------------------------------|----------|-----------------------|-----------------|
| best.py      | dcn_feats, hidden=128, aux 0.1      | 42,43,44 | **0.6039 +- 0.0010**  | **+0.0023 ACCEPTED** |
| dcn_feats.py | hidden=128, no aux                  | 42,43,44 | 0.6038 +- 0.0011      | +0.0022 accepted (tie) |
| —            | 5-seed rank ensemble of best.py     | 42-46    | 0.6047 (single number)| variance reducer |

Dead branches (seed 42): item aggregates 0.6038, content features 0.6039,
k=32 0.6039, bpr weight != 0.5 all <= 0.6045, LightGBM lambdarank 0.5974
(blends all hurt). New best artifact: `zoo/best.py`.
