# role: proposer | model: gpt-5.6-sol

## SYSTEM
You are an autonomous ML research agent improving a short-video recommender.

Task: predict long_view (binary) per impression; ranking quality is scored with
"primary" = mean of within-user GAUC and per-user nDCG@5, computed by the official
evaluator `harness.evaluate_provisional.evaluate(user_ids, labels, scores)`.
Higher is better. Improvements below 0.002 on validation are noise.

Hard rules for every script you emit (CONTRACTS.md section 3):
- Emit ONE WHOLE runnable Python script. Never a diff, never a fragment.
- CLI: `python <script> --data-dir <d> --out-dir <o> [--seed 42]` via argparse.
  Default seed 42. Deterministic given the seed.
- FAST PATH (use it when present): `<data-dir>/train.npz` and `<data-dir>/val.npz` hold
  pre-encoded arrays — X (int32, 5 offset-encoded fields: user,video,author,tab,dur_bucket),
  y (long_view float32), user, click, play_time_ms, duration_ms, hourmin, date, field_dims.
  Loading them takes ~1s vs ~90s of CSV parsing; training-time budget is scored, so prefer npz.
  Score with the official evaluator: `from data.official.evaluate import evaluate` ->
  dict keys 'GAUC', 'nDCG@5', 'primary' (write metrics.json keys gauc/ndcg5/primary).
  A known-good exemplar of the full pattern is the baseline parent script itself.
- Otherwise read ONLY `<data-dir>/train.csv` and `<data-dir>/val.csv`. The test split does
  not exist in your workspace; any attempt to reference a test file fails the run.
- Train on train.csv, score val.csv, then:
  * write `<out-dir>/predictions.csv` with header row_id,user_id,video_id,score
    (one row per validation row, in file order);
  * compute metrics with `from harness.evaluate_provisional import evaluate` on
    the validation labels and write `<out-dir>/metrics.json` as
    {"gauc": ..., "ndcg5": ..., "primary": ...}.
- Print nothing to stdout/stderr (no progress bars, no logging). Long-running
  fan-out nodes SHOULD append one line per completed probe (config + score) to
  `<out-dir>/progress.log` so the search is observable while it runs; besides
  that, only write the two output files.
- Stay within the runtime timeout (default 600s); prefer small/fast models.
- Read environment variable `SMOKE_EPOCHS` as an integer when present and cap
  every training phase's epoch count to that value. `SMOKE_EPOCHS=1` is the
  harness sanity pass and must still write predictions.csv and metrics.json.
- Columns available: user_id,video_id,tab,hourmin,date,duration_ms,long_view,
  click,like,play_time_ms. long_view/click/like/play_time_ms are OUTCOMES of the
  impression: usable as auxiliary training TARGETS only, never as input features,
  and never read from val.csv except long_view for the metrics computation.
- Allowed libraries: numpy, torch, stdlib only.
- DEVICE: select the best available torch device at startup —
  cuda if torch.cuda.is_available() else cpu — and run all training on it.
  On a GPU, probes are 5-10x faster: use the saved time for deeper search
  (more cells, longer probes), not for finishing early. GPU-hours are
  report-only in this challenge; wall-clock is what is scored.

Propose ONE falsifiable change per iteration relative to the parent script,
stated as a hypothesis. Default to an atomic change; a change may instead be a
literature-grounded PACKAGE (e.g. an architecture together with the
regularization its source paper trains it with) when the method cards'
combination guidance says the components only work together — cite that
pairing. Any proposal MAY fan out internally: the node's script may train a
small set of candidate variants (dial settings, component combinations — e.g.
6-12 short probe trainings on the fast path), select the best on validation,
then train the final model with the winning configuration. Record every probe's
config and score in metrics.json history so the search is auditable. The whole
fan-out is ONE node: budget probes so total runtime stays inside the timeout,
keep probes short (2-3 epochs, optional subsample), and make the final training
full-length. Do not re-try ideas the journal shows were rejected.

Respond with a single JSON object and nothing else. Every proposal is a
discriminated envelope. A farm-close proposal uses the separate contract below;
every ordinary method uses this form and MUST NOT include a farm-close plan:
{"execution_kind":"script",
 "hypothesis": "<one falsifiable sentence with expected effect size>",
 "expected_delta": <honest numeric expectation for validation-primary delta, e.g. 0.0015>,
 "expected_delta_basis": "<one sentence citing a specific card expectation or journal line>",
 "action": "<draft|debug|improve>",
 "parent": "<parent node id you were given>",
 "code": "<the WHOLE script as a JSON string>"}


## Improvement menu
# Improvement menu — the agent's ranked search space

Ground truth from the starter kit: baseline FM (k=16, logloss, Adam) over exactly 5
fields: user_id, video_id, author_id, tab, dur_bucket (10 train-quantile duration
buckets). Beating primary 0.5946 (test) / 0.6016 (valid) is the goal. Published
KuaiRand-Pure reference: CWM (KDD'24) reaches GAUC ~0.713-0.715 → target band 0.70+.
Research basis: ../research/*.md. Expected gains are estimates, not promises.

## BINDING CONSTRAINT (measured, run_real_01 + manual campaign E1-E8)
Every architecture/feature variant plateaus at valid primary ~0.604-0.605, because
ALL models overfit by epoch 2-3 (val GAUC peaks then falls). Single-lever changes
from that plateau land within noise and get rejected. Priorities that follow:
A. **Fight the overfit FIRST**: dropout on embeddings/MLP, weight decay / per-row
   embedding L2, lr decay schedules, label smoothing, smaller lr + more epochs —
   anything that lets training survive past epoch 3. UNEXPLORED and the most
   plausible route past 0.605.
B. **Unexplored objectives**: per-user listwise softmax loss; ordinal watch-ratio
   auxiliary (play_time/duration buckets, TPM-lite); CWM-style censored watch-time
   loss (one-sided regression on play_time truncated at duration).
C. **Compound hypotheses are allowed and encouraged** when single levers plateau:
   one coherent THEME per iteration (e.g. "regularization package: dropout 0.1 +
   weight decay 1e-5 + lr decay"), not one micro-knob.
D. Do NOT re-test dead branches: item-side aggregates, video content features,
   k=32, LightGBM blends, pure BPR or pure logloss (all measured worse, see below).

## CURRENT BEST (28 Aug pm, measured)
**L0 strong-regularization stack: official 5 fields, DCN-lite, heavy regularization
(zoo/ablate_fields.py --regularized): valid primary 0.6047 ± 0.0003 (3-seed, +0.0031
CONFIRMED — tightest std of any winner).** The field-ablation curve shows MORE fields
lose: kitchen-sink L5 = 0.6030. The binding constraint was regularization strength,
not missing information. Schedules: rapid step LR decay is the only schedule family
that beats the plateau (sweep campaign). Context finding: compact agent context beats
full-history 3x-token context (A/B measured). FinalMLP failed 3-seed confirm — closed.

## CURRENT DIRECTIVE (updated Mon 31 Aug — supersedes the watch-time directive, whose
## themes are all MEASURED DEAD: CWM/censored, ordinal watch-ratio, listwise. Do NOT re-test.)
Evidence-ranked priorities (all numbers from run journals):
1. diverse-family-farm-close — ONE node: train one member per measured-win family and rank-average across families (the measured 0.6058-0.6065 evidence, agent-reachable in a single eps-clearing iteration; see card)
2. seq-deepfm-composite — teammate-verified full package at 0.6055-0.6061 validation
   (see card; partial ports fail — implement completely). HIGHEST PRIORITY.
3. context-stratified-pairs on a gauge-fixed-bce or dial-swept base — measured +0.0015
   (run_final_s4 n3: 0.60521), NO ensemble close ever attempted on that base.
4. seq-deepfm-author-history — causal pooled author-history DeepFM, measured externally
   0.6047 across 4 seeds (see card). Untried under this harness.
5. heterogeneous-ensemble-design CROSS-FAMILY close (blend probe: 0.6058 from mixing a composite member with a DCN-package ensemble — decorrelation is where closes pay) or ensemble-design-sweep close on ANY champion >=0.6045 — closes measure +0.0010-0.0013
   (bigclock_07, final_s2). The winning shape in every ledger.
RULE: the library is two-tier — TREATMENT cards match your diagnosis; OPPORTUNITY cards
(architectures, feature families, sweeps, closes) are diagnosis-independent and ranked
by measured evidence in your menu. Weigh opportunities every iteration.

## Tier 1 — do first
1. **Within-user pairwise loss (BPR)** on the same FM features. Build (pos, neg) pairs
   inside each user's training impressions; optimize sigmoid(s_pos - s_neg). Directly
   optimizes what GAUC measures. Hybrid 0.5*BPR + 0.5*logloss is the safe variant.
   Expect +0.005-0.015 GAUC. ~60 lines.
   MEASURED (valid): mix sweep {0,0.3,0.5,0.7,1.0} at seed 42 — 0.5/0.5 hybrid best (0.6048); pure BPR or pure logloss lose ~0.001 (EXPERIMENTS.md E5).

2. **Early stopping + model selection on validation GAUC** (baseline stops on epochs/
   logloss). Free correctness fix. Expect +0.002-0.005.
3. **Finer duration handling**: the label IS duration-defined (long_view = watched
   >= min(duration, 18s)). Add: 50 buckets instead of 10, plus a direct
   duration<=18s indicator field, plus dur_bucket x tab cross. Expect +0.003-0.01.

   MEASURED (valid, 3-seed): in the winning stack (with #4/#9): 0.6039 +- 0.0010; features are part of zoo/best.py.

## Tier 2 — model capacity
4. **DCNv2-lite / FinalMLP head** on the same embeddings (1-2 cross layers, small MLP).
   Architectures beyond that (xDeepFM/AutoInt) overfit at 1.4M rows - skip.
   Expect +0.003-0.01 over tuned FM.
   MEASURED (valid, 3-seed): DCN-lite + #3/#9 features + aux 0.1 = 0.6039 +- 0.0010, delta +0.0023 ACCEPTED — current best (zoo/best.py). hidden 128 > 64 ~ 256; cross layers 1-3 all within noise.

5. **Multi-task shared-bottom**: auxiliary heads for click, like, effective_view at
   loss weight 0.1-0.3 (labels from the log's other signal columns - as TARGETS only,
   never inputs). Expect +0.003-0.008. PLE/MMoE only if this works.
   MEASURED (valid, 3-seed): aux 0.1 on the DCN stack = 0.6039 +- 0.0010 vs 0.6038 +- 0.0011 without — tiny/tied but kept; aux 0.2/0.3 no better (seed 42).

6. **Embedding dim sweep 16->32 (+ per-row L2)**. Cheap. Expect +0.00-0.005.

   MEASURED (valid, seed 42): k=32 = 0.6039 vs k=16 = 0.6047 — 32 is worse; keep 16.

## Tier 2.5 — data-level (MEASURED ALIVE)
6b. **Recency weighting of training samples** (exp decay, 7-day half-life from
    20220421): MEASURED (valid, 3-seed) 0.6043 +- 0.0012, delta +0.0027 CONFIRMED —
    zoo/hist_best.py. Orthogonal to model/loss levers; include in any new stack.
    Corrects train->val distribution shift (val positive rate declines over its week).

## Tier 3 — features
7. **Train-window item/author aggregate rates**: video's and author's long_view rate
   computed ONLY over train dates, smoothed (Bayesian prior), as a scalar feature.
   Item-side only - user-side rates cannot move a within-user metric. Leakage rule:
   aggregates must use train window only. Expect +0.003-0.01.
   MEASURED (valid, seed 42): bucketed smoothed rates (prior 20) HURT: 0.6038 vs 0.6047 without — dead (E4).

8. **Video-side content features** from video_features_basic (video_type, upload_type,
   music_id, tags first-tag). Expect +0.002-0.008.
   MEASURED (valid, seed 42): video_type/upload_type/music_id-top200/first-tag HURT: 0.6039 vs 0.6048 — dead (E6).

9. **Temporal context**: hour-of-day bucket from hourmin, day-of-week from date.
   Expect +0.001-0.005.

   MEASURED (valid, 3-seed): included in the winning stack (with #3); not ablated separately.

## Tier 4 — advanced / stretch
10. **Ordinal watch-ratio auxiliary (TPM-lite)**: predict play_time/duration ordinal
    buckets as an auxiliary task. Expect +0.002-0.008.
11. **LightGBM lambdarank** on target-encoded aggregates over all 12 signals
    (train-window only) as a parallel model; rank-average blend with the NN.
    Expect +0.002-0.006 from the blend.
    MEASURED (valid, seed 42): LGBM alone 0.5974 (below baseline); every rank blend with the NN hurts — dead (E8).

12. **Seed ensemble**: average predictions over 3-5 seeds of the best config.
    Free +0.002-0.005, do at the very end.
    MEASURED (valid): 5-seed rank-average of best.py = 0.6047 vs seed-mean 0.6039 — variance reducer, ~best-single-seed level (E7).

13. **CWM-style censored watch-time loss** (KDD'24) - the published SOTA idea.
    High risk/reward; only if iterations remain.

## Known traps (the agent must respect)
- video_features_statistic counters are FULL-PERIOD aggregates -> temporal leakage.
  Do not use as-is; only train-window recomputations are legal.
- Other feedback signals (click, like, play_time...) are OUTCOMES of the impression:
  usable as auxiliary TARGETS, never as input features.
- GAUC is within-user: user-constant features cannot help GAUC (they can still help
  nDCG ordering? No - also within-user). Spend features on item-side variation.
- Improvements < 0.002 on validation are within noise (baseline seed std 0.0008;
  official epsilon = 0.002). Acceptance rule: keep a change only if val primary
Acceptance: >=0.002 accepts outright; smaller positive deltas enter a 2-reseed z-tested grey confirm (floor 0.0005). Regressions revert.

## STRATEGY LAYER (the philosophy; cards are atomic methods, bundles marked as such)
Open with the strongest unapplied opportunity -> probe cheaply before committing full
budget -> keep small confirmed gains (grey confirm) -> after an eps-clearing accept COMPOUND (next-strongest opportunity from a DIFFERENT family, or begin the close) rather than re-sweeping the same family -> farm DIVERSE-family members,
not same-family seeds -> close by rank-aggregating ACROSS families -> stop when the
rule says stop. (v3 design: promote this to a first-class layer the selector reasons
over, with cards strictly atomic.)


## USER
## Prior runs (do not repeat failed openings)
(none recorded)

## Journal (one line per prior node)
node_000 [baseline] draft "baseline FM" primary=0.6018 ACCEPTED (sigma=0.0001)

Mode: DRAFT. Write a fresh script derived from the parent (the current baseline family) implementing the SELECTED card (or, when a launch directive names a tier, that directive). Keep everything else identical to the parent.

## Selected method (implement THIS)
### package-dial-sweep: Literature package with internal dial search
- mechanism: Implement the full capacity+regularization package (dcn-lite + bpr-hybrid + regularization-schedule + recency-weighting) as ONE node whose script runs a TWO-STAGE internal search, then trains the final model. Stage 1 (coarse): 8-12 short probes over wide ranges — dropout {0.15..0.4}, weight decay {3e-5..3e-3 log-spaced}, LR step-decay variants, recency half-life {3.5, 7, 14}. Stage 2 (refine): 6-10 probes on a DENSER grid centered on stage-1's winner, at longer probe length (4-6 epochs, full rows). Choose values yourself; do not just copy example numbers. Then ONE full-length training with the winning dials, checkpointing EVERY HALF-EPOCH and keeping the validation-best snapshot. When the run clock allows (timeout permitting), close by rank-averaging 5 consecutive seeds of the final config inside the same node or as the follow-up node. Use the wall-clock: probe time is cheap relative to the 6h ceiling; log every probe's config and score in metrics.json history.
- kind: opportunity
- reference_primary: none
- treats: overfit | underfit
- preconditions: Use the npz fast path; keep probes short; the final training must be full-length. The sweep is internal to the node — one iteration, one artifact.
- citation: standard hyperparameter search practice (random/grid search, Bergstra & Bengio JMLR 2012); package composition per DCNv2/BPR training setups.
- expected_gain / cost: package at tuned dials measured 0.6047 +/- 0.0003; untuned dials measured 0.595-0.602 — the sweep is what closes that gap / medium-high runtime (one node).
- status_pure: measured-win (bigclock_07 n3 0.60424; novel_l1 n2 0.60387; the standard strong opener)
- status_1k: untried
selector diagnosis: overfit
selector why: Validation primary peaks at epoch 8 (0.601838) and declines to 0.600686 by epoch 10 while training loss continues trending downward, indicating overfitting. From a fresh baseline with 15 iterations left, this measured-win package is the strongest robust opener: it couples added interaction capacity with the required regularization, LR decay, recency weighting, hybrid ranking loss, half-epoch checkpointing, and wide two-stage dial search. Its measured absolute level near 0.6047 implies about +0.0029 over the current 0.6018 best, clearing the 0.002 convergence epsilon more reliably than an atomic treatment.

## Convergence pressure
streak_state = {'no_improve_streak': 0, 'n_converge': 3, 'iters_left': 15}
The run ends after N consecutive iterations whose best-so-far improvement is <= epsilon = 0.002. Select experiments by expected scientific value given the remaining budget: at every iteration, including the first, prefer the eligible move with the largest evidence-supported expected gain for its cost; an early iteration spent on a small-ceiling treatment is a convergence strike bought at full price. Literature-grounded packages (components whose sources evaluate them together) are one experiment; keep unproven novel ideas atomic. Plan the run so its final iterations produce the strongest possible finished artifact rather than leaving the run un-finalized. Do the epsilon arithmetic before choosing: if the streak means the run ends unless THIS iteration improves best-so-far by at least epsilon, then a move whose own evidence caps its gain below epsilon cannot extend the run no matter how proven it is; on such an iteration prefer the eligible move with the largest evidence-supported expected gain at or above epsilon, and among qualifying moves prefer the one whose evidence clears epsilon with the widest margin: a move whose evidence only just reaches the bar fails it about half the time, so bare arithmetic reach is not parity with a wide-margin alternative (combining decorrelated mechanism families generally out-gains both re-seeding one family and any single atomic mechanism). Read margins against the CURRENT best, not a card's original baseline: an unspent package whose measured absolute score sits near the current best offers almost no headroom, while a close whose evidence exceeds every single-model score in the ledger offers the most. A proven small-gain close is the right pick only when no eligible move has evidence reaching epsilon. Do not change what counts as an iteration in response to the streak.

## Runtime budget (overrides the 600s default above)
THIS run's per-node timeout is 7200 seconds (~120 minutes). A full-length training on the npz fast path costs roughly 40-90s on CPU, far less on GPU. Plan to SPEND ~60-70% of this budget on search probes when playing a search card — e.g. at 2+ hours that is 40+ full-length probes plus refinement, not 8. Reserve the remainder for the final training(s). Finishing a search node in a small fraction of the budget is a defect, not efficiency: unspent budget is free score variance left unexplored.

Directive: draft from Tier 1 of the menu

When implementing ANY ensemble/member card: each member MUST be trained with a distinct seed; after scoring, ASSERT member score vectors are not identical (numpy allclose check between members and against the parent predictions) and print per-member validation primaries to progress output. An ensemble whose final predictions equal the parent's is a no-op and will be rejected by the harness, except when the farm-close executor explicitly selects and records the incumbent fallback.

## Parent node "node_000" (full code)
```python
"""Official-parity FM baseline over the workspace .npz fast path.

Mirrors the starter kit's FM (5 fields: user,video,author,tab,dur_bucket; k=16,
Adam lr 1e-3, logloss, early stop on valid GAUC). Reads <data-dir>/{train,val}.npz.
Obeys CONTRACTS.md section 3."""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.official.evaluate import evaluate  # official oracle


class FM(torch.nn.Module):
    def __init__(self, total_dim, k=16):
        super().__init__()
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)

    def forward(self, x):
        e = self.emb(x)                                  # (B, F, k)
        s = e.sum(1)
        pair = 0.5 * (s * s - (e * e).sum(1)).sum(1)
        return self.bias + self.lin(x).sum((1, 2)) + pair


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=12)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    tr = np.load(os.path.join(a.data_dir, "train.npz"))
    va = np.load(os.path.join(a.data_dir, "val.npz"))
    total_dim = int(tr["field_dims"].sum())
    Xt = torch.from_numpy(tr["X"].astype(np.int64)); yt = torch.from_numpy(tr["y"])
    Xv = torch.from_numpy(va["X"].astype(np.int64))
    model = FM(total_dim)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = torch.nn.BCEWithLogitsLoss()
    n = len(yt); bs = 8192
    best, best_scores, patience = -1.0, None, 0
    history = []
    for epoch in range(a.epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = bce(model(Xt[idx]), yt[idx])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            scores = np.concatenate([model(Xv[i:i + 65536]).numpy()
                                     for i in range(0, len(Xv), 65536)])
        m = evaluate(va["user"], va["y"].astype(int), scores)
        primary = m["primary"]
        history.append({"epoch": epoch + 1, "train_loss": round(float(loss.item()), 5),
                        "val_gauc": round(m.get("GAUC", 0.0), 6), "val_primary": round(primary, 6)})
        if primary > best + 1e-6:
            best, best_scores, patience = primary, scores, 0
        else:
            patience += 1
            if patience >= 2:
                break
    os.makedirs(a.out_dir, exist_ok=True)
    m = evaluate(va["user"], va["y"].astype(int), best_scores)
    with open(os.path.join(a.out_dir, "metrics.json"), "w") as fh:
        json.dump({"gauc": m["GAUC"] if "GAUC" in m else m["gauc"],
                   "ndcg5": m.get("nDCG@5", m.get("ndcg5")),
                   "primary": m["primary"], "history": history}, fh)
    with open(os.path.join(a.out_dir, "predictions.csv"), "w") as fh:
        fh.write("row_id,user_id,video_id,score\n")
        for i, s in enumerate(best_scores):
            fh.write(f"{i},{va['user'][i]},0,{s:.6g}\n")


if __name__ == "__main__":
    main()

```

## Parent learning curve (per epoch)
epoch 1: train_loss 0.57954, val_gauc 0.649846, val_primary 0.588885
epoch 2: train_loss 0.52013, val_gauc 0.661034, val_primary 0.596914
epoch 3: train_loss 0.50487, val_gauc 0.664894, val_primary 0.599873
epoch 4: train_loss 0.50204, val_gauc 0.666306, val_primary 0.600912
epoch 5: train_loss 0.49487, val_gauc 0.667447, val_primary 0.601449
epoch 6: train_loss 0.48704, val_gauc 0.667443, val_primary 0.601641
epoch 7: train_loss 0.4904, val_gauc 0.667123, val_primary 0.601379
epoch 8: train_loss 0.46744, val_gauc 0.667596, val_primary 0.601838
epoch 9: train_loss 0.49261, val_gauc 0.66651, val_primary 0.601065
epoch 10: train_loss 0.46233, val_gauc 0.666294, val_primary 0.600686
DIAGNOSE before proposing: val peaks early then falls = overfit (attack with regularization/schedules); val still rising at stop = underfit (train longer); flat = the idea itself adds no signal (change direction, not dosage). State your diagnosis in the hypothesis.

Respond with the JSON object only.

## REPLY
{"execution_kind":"script","hypothesis":"Because validation peaks before training loss stops improving, overfitting is the diagnosis; a validation-tuned DCN-lite package combining 0.5 BCE/0.5 within-user BPR, embedding/MLP dropout, AdamW, rapid step decay, and recency weighting will improve validation primary by at least 0.0025 over the 0.6018 FM parent.","expected_delta":0.0029,"expected_delta_basis":"The selected package-dial-sweep card reports tuned validation primary near 0.6047 versus the parent’s 0.6018, with untuned configurations substantially worse, making internal dial selection essential.","action":"draft","parent":"node_000","code":"import argparse\nimport csv\nimport datetime\nimport json\nimport math\nimport os\nimport sys\n\nimport numpy as np\nimport torch\n\nsys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\n\n\nclass DCNLite(torch.nn.Module):\n    def __init__(self, total_dim, n_fields=5, k=16, hidden=128, dropout=0.25):\n        super().__init__()\n        self.n_fields = n_fields\n        self.k = k\n        d = n_fields * k\n        self.emb = torch.nn.Embedding(total_dim, k)\n        self.lin = torch.nn.Embedding(total_dim, 1)\n        self.bias = torch.nn.Parameter(torch.zeros(1))\n        self.emb_drop = torch.nn.Dropout(dropout)\n        self.cross_w = torch.nn.Parameter(torch.empty(d))\n        self.cross_b = torch.nn.Parameter(torch.zeros(d))\n        self.cross_out = torch.nn.Linear(d, 1, bias=False)\n        self.mlp = torch.nn.Sequential(\n            torch.nn.Linear(d, hidden),\n            torch.nn.ReLU(),\n            torch.nn.Dropout(dropout),\n            torch.nn.Linear(hidden, hidden // 2),\n            torch.nn.ReLU(),\n            torch.nn.Dropout(dropout),\n            torch.nn.Linear(hidden // 2, 1),\n        )\n        torch.nn.init.normal_(self.emb.weight, std=0.01)\n        torch.nn.init.zeros_(self.lin.weight)\n        torch.nn.init.normal_(self.cross_w, std=0.01)\n        torch.nn.init.xavier_uniform_(self.cross_out.weight)\n        for layer in self.mlp:\n            if isinstance(layer, torch.nn.Linear):\n                torch.nn.init.xavier_uniform_(layer.weight)\n                torch.nn.init.zeros_(layer.bias)\n\n    def forward(self, x):\n        raw = self.emb(x)\n        fm_sum = raw.sum(1)\n        fm = 0.5 * (fm_sum.square() - raw.square().sum(1)).sum(1)\n        z0 = self.emb_drop(raw).reshape(x.shape[0], -1)\n        cross = z0 * (z0 @ self.cross_w).unsqueeze(1) + self.cross_b + z0\n        linear = self.lin(x).sum((1, 2))\n        return self.bias + linear + fm + self.cross_out(cross).squeeze(1) + self.mlp(z0).squeeze(1)\n\n\ndef seed_all(seed):\n    np.random.seed(seed)\n    torch.manual_seed(seed)\n    if torch.cuda.is_available():\n        torch.cuda.manual_seed_all(seed)\n    torch.backends.cudnn.deterministic = True\n    torch.backends.cudnn.benchmark = False\n\n\ndef date_ordinals(values):\n    a = np.asarray(values)\n    out = np.empty(len(a), dtype=np.float32)\n    cache = {}\n    for i, value in enumerate(a):\n        try:\n            key = int(value)\n        except Exception:\n            key = 0\n        if key not in cache:\n            text = str(key)\n            try:\n                if len(text) == 8:\n                    cache[key] = datetime.date(int(text[:4]), int(text[4:6]), int(text[6:8])).toordinal()\n                else:\n                    cache[key] = key\n            except Exception:\n                cache[key] = key\n        out[i] = cache[key]\n    return out\n\n\ndef encode_column(train_values, val_values):\n    mapping = {}\n    train_encoded = np.empty(len(train_values), dtype=np.int64)\n    for i, value in enumerate(train_values):\n        key = str(value)\n        if key not in mapping:\n            mapping[key] = len(mapping)\n        train_encoded[i] = mapping[key]\n    oov = len(mapping)\n    val_encoded = np.asarray([mapping.get(str(v), oov) for v in val_values], dtype=np.int64)\n    return train_encoded, val_encoded, oov + 1\n\n\ndef load_csv_fallback(data_dir):\n    def read_split(path, training):\n        columns = {k: [] for k in ['user_id', 'video_id', 'tab', 'duration_ms', 'date', 'long_view']}\n        with open(path, 'r', newline='') as fh:\n            reader = csv.DictReader(fh)\n            for row in reader:\n                columns['user_id'].append(row['user_id'])\n                columns['video_id'].append(row['video_id'])\n                columns['tab'].append(row['tab'])\n                columns['duration_ms'].append(float(row['duration_ms']))\n                columns['date'].append(row['date'])\n                columns['long_view'].append(float(row['long_view']))\n        return columns\n\n    tr = read_split(os.path.join(data_dir, 'train.csv'), True)\n    va = read_split(os.path.join(data_dir, 'val.csv'), False)\n    tu, vu, du = encode_column(tr['user_id'], va['user_id'])\n    tv, vv, dv = encode_column(tr['video_id'], va['video_id'])\n    tt, vt, dt = encode_column(tr['tab'], va['tab'])\n    author_train = tv.copy()\n    author_val = vv.copy()\n    da = dv\n    train_duration = np.asarray(tr['duration_ms'], dtype=np.float64)\n    val_duration = np.asarray(va['duration_ms'], dtype=np.float64)\n    quantiles = np.unique(np.quantile(train_duration, np.linspace(0.1, 0.9, 9)))\n    td = np.searchsorted(quantiles, train_duration, side='right').astype(np.int64)\n    vd = np.searchsorted(quantiles, val_duration, side='right').astype(np.int64)\n    dd = len(quantiles) + 1\n    dims = np.asarray([du, dv, da, dt, dd], dtype=np.int64)\n    offsets = np.concatenate(([0], np.cumsum(dims)[:-1]))\n    Xt = np.stack([tu, tv, author_train, tt, td], axis=1) + offsets\n    Xv = np.stack([vu, vv, author_val, vt, vd], axis=1) + offsets\n    train = {\n        'X': Xt.astype(np.int64),\n        'y': np.asarray(tr['long_view'], dtype=np.float32),\n        'user': np.asarray(tr['user_id']),\n        'date': np.asarray(tr['date']),\n        'field_dims': dims,\n    }\n    val = {\n        'X': Xv.astype(np.int64),\n        'y': np.asarray(va['long_view'], dtype=np.float32),\n        'user': np.asarray(va['user_id']),\n        'video': np.asarray(va['video_id']),\n    }\n    return train, val, False\n\n\ndef load_data(data_dir):\n    train_npz = os.path.join(data_dir, 'train.npz')\n    val_npz = os.path.join(data_dir, 'val.npz')\n    if os.path.exists(train_npz) and os.path.exists(val_npz):\n        tr_file = np.load(train_npz)\n        va_file = np.load(val_npz)\n        train = {k: tr_file[k] for k in tr_file.files}\n        val = {k: va_file[k] for k in va_file.files}\n        val['video'] = np.zeros(len(val['y']), dtype=np.int64)\n        return train, val, True\n    return load_csv_fallback(data_dir)\n\n\ndef build_pair_pools(users, labels):\n    users = np.asarray(users)\n    labels = np.asarray(labels)\n    order = np.argsort(users, kind='mergesort')\n    sorted_users = users[order]\n    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])\n    positives = []\n    negatives = []\n    pos_offsets = []\n    neg_offsets = []\n    pos_counts = []\n    neg_counts = []\n    for left, right in zip(boundaries[:-1], boundaries[1:]):\n        idx = order[left:right]\n        pos = idx[labels[idx] > 0.5]\n        neg = idx[labels[idx] <= 0.5]\n        if len(pos) and len(neg):\n            pos_offsets.append(len(positives))\n            neg_offsets.append(len(negatives))\n            pos_counts.append(len(pos))\n            neg_counts.append(len(neg))\n            positives.extend(pos.tolist())\n            negatives.extend(neg.tolist())\n    return {\n        'pos': np.asarray(positives, dtype=np.int64),\n        'neg': np.asarray(negatives, dtype=np.int64),\n        'po': np.asarray(pos_offsets, dtype=np.int64),\n        'no': np.asarray(neg_offsets, dtype=np.int64),\n        'pc': np.asarray(pos_counts, dtype=np.int64),\n        'nc': np.asarray(neg_counts, dtype=np.int64),\n    }\n\n\ndef sample_pairs(pool, count, rng):\n    group = rng.integers(0, len(pool['po']), size=count)\n    pslot = pool['po'][group] + np.floor(rng.random(count) * pool['pc'][group]).astype(np.int64)\n    nslot = pool['no'][group] + np.floor(rng.random(count) * pool['nc'][group]).astype(np.int64)\n    return pool['pos'][pslot], pool['neg'][nslot]\n\n\ndef metric_values(evaluator, users, labels, scores):\n    result = evaluator(users, labels.astype(int), scores)\n    return {\n        'gauc': float(result['GAUC'] if 'GAUC' in result else result['gauc']),\n        'ndcg5': float(result.get('nDCG@5', result.get('ndcg5'))),\n        'primary': float(result['primary']),\n    }\n\n\ndef predict(model, Xv, device, batch_size=65536):\n    model.eval()\n    chunks = []\n    with torch.no_grad():\n        for start in range(0, len(Xv), batch_size):\n            xb = torch.as_tensor(Xv[start:start + batch_size], dtype=torch.long, device=device)\n            chunks.append(model(xb).detach().cpu().numpy())\n    return np.concatenate(chunks)\n\n\ndef set_learning_rate(optimizer, base_lr, completed_epochs, step_epochs, gamma):\n    exponent = int(math.floor((completed_epochs + 1e-9) / step_epochs))\n    lr = base_lr * (gamma ** exponent)\n    for group in optimizer.param_groups:\n        group['lr'] = lr\n    return lr\n\n\ndef train_candidate(config, train, val, pair_pool, recency_age, evaluator, device,\n                    seed, epochs, checkpoint_half_epochs):\n    seed_all(seed)\n    rng = np.random.default_rng(seed)\n    X = np.asarray(train['X'], dtype=np.int64)\n    y_np = np.asarray(train['y'], dtype=np.float32)\n    Xv = np.asarray(val['X'], dtype=np.int64)\n    total_dim = int(np.asarray(train['field_dims']).sum())\n    model = DCNLite(total_dim, n_fields=X.shape[1], k=16, hidden=128,\n                    dropout=float(config['dropout'])).to(device)\n    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['lr']),\n                                  weight_decay=float(config['weight_decay']))\n    n = len(y_np)\n    batch_size = 8192 if device.type == 'cuda' else 4096\n    pair_batch = max(512, batch_size // 2)\n    y = torch.as_tensor(y_np, dtype=torch.float32, device=device)\n    recency_np = np.exp2(-recency_age / float(config['half_life'])).astype(np.float32)\n    weights = torch.as_tensor(recency_np, dtype=torch.float32, device=device)\n    best_primary = -1.0\n    best_scores = None\n    best_metrics = None\n    curve = []\n    global_batches = int(math.ceil(n / batch_size))\n    checkpoints = {global_batches - 1}\n    if checkpoint_half_epochs:\n        checkpoints.add(max(0, global_batches // 2 - 1))\n    for epoch in range(epochs):\n        model.train()\n        permutation = torch.randperm(n, device=device)\n        pair_count = global_batches * pair_batch\n        pos_idx, neg_idx = sample_pairs(pair_pool, pair_count, rng)\n        last_loss = 0.0\n        for batch_number, start in enumerate(range(0, n, batch_size)):\n            completed = epoch + batch_number / max(1, global_batches)\n            current_lr = set_learning_rate(optimizer, float(config['lr']), completed,\n                                           float(config['step_epochs']), float(config['gamma']))\n            idx = permutation[start:start + batch_size]\n            xb = torch.as_tensor(X[idx.detach().cpu().numpy()], dtype=torch.long, device=device)\n            logits = model(xb)\n            raw_bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, y[idx], reduction='none')\n            bce = (raw_bce * weights[idx]).sum() / weights[idx].sum().clamp_min(1e-8)\n            p0 = batch_number * pair_batch\n            p1 = p0 + pair_batch\n            pi_np = pos_idx[p0:p1]\n            ni_np = neg_idx[p0:p1]\n            pi = torch.as_tensor(pi_np, dtype=torch.long, device=device)\n            ni = torch.as_tensor(ni_np, dtype=torch.long, device=device)\n            pair_x = np.concatenate([X[pi_np], X[ni_np]], axis=0)\n            pair_logits = model(torch.as_tensor(pair_x, dtype=torch.long, device=device))\n            pscore = pair_logits[:len(pi_np)]\n            nscore = pair_logits[len(pi_np):]\n            pair_weight = 0.5 * (weights[pi] + weights[ni])\n            raw_pair = torch.nn.functional.softplus(-(pscore - nscore))\n            bpr = (raw_pair * pair_weight).sum() / pair_weight.sum().clamp_min(1e-8)\n            loss = 0.5 * bce + 0.5 * bpr\n            optimizer.zero_grad(set_to_none=True)\n            loss.backward()\n            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)\n            optimizer.step()\n            last_loss = float(loss.detach().cpu())\n            if batch_number in checkpoints:\n                scores = predict(model, Xv, device)\n                metrics = metric_values(evaluator, np.asarray(val['user']), np.asarray(val['y']), scores)\n                point = {\n                    'epoch': round(epoch + (batch_number + 1) / global_batches, 3),\n                    'train_loss': round(last_loss, 6),\n                    'lr': float(current_lr),\n                    'primary': metrics['primary'],\n                }\n                curve.append(point)\n                if metrics['primary'] > best_primary:\n                    best_primary = metrics['primary']\n                    best_scores = scores.copy()\n                    best_metrics = metrics\n                model.train()\n    del optimizer\n    model.to('cpu')\n    del model\n    if device.type == 'cuda':\n        torch.cuda.empty_cache()\n    return best_metrics, best_scores, curve\n\n\ndef coarse_configs(seed):\n    rng = np.random.default_rng(seed + 1771)\n    configs = []\n    half_lives = [3.5, 7.0, 14.0]\n    step_choices = [0.75, 1.0, 1.5, 2.0, 2.75]\n    gamma_choices = [0.35, 0.45, 0.55, 0.68, 0.78]\n    for i in range(12):\n        configs.append({\n            'dropout': float(rng.uniform(0.15, 0.40)),\n            'weight_decay': float(10 ** rng.uniform(math.log10(3e-5), math.log10(3e-3))),\n            'lr': float(10 ** rng.uniform(math.log10(3.5e-4), math.log10(1.6e-3))),\n            'step_epochs': float(step_choices[i % len(step_choices)]),\n            'gamma': float(gamma_choices[(i * 2) % len(gamma_choices)]),\n            'half_life': float(half_lives[(i * 2) % len(half_lives)]),\n        })\n    return configs\n\n\ndef refined_configs(winner):\n    patterns = [\n        (0.00, 1.00, 1.00, 1.00, 0.00, 1.00),\n        (-0.025, 0.55, 0.82, 0.82, -0.06, 0.75),\n        (-0.012, 0.75, 1.18, 1.18, 0.04, 1.00),\n        (0.012, 1.35, 0.90, 0.92, -0.03, 1.25),\n        (0.025, 1.80, 1.10, 1.08, 0.06, 1.50),\n        (-0.035, 1.20, 1.00, 1.32, 0.02, 0.75),\n        (0.035, 0.70, 1.28, 0.72, -0.08, 1.25),\n        (0.000, 1.55, 0.74, 1.45, 0.08, 1.00),\n    ]\n    result = []\n    for dd, wm, lm, sm, dg, hm in patterns:\n        result.append({\n            'dropout': float(np.clip(winner['dropout'] + dd, 0.12, 0.45)),\n            'weight_decay': float(np.clip(winner['weight_decay'] * wm, 2e-5, 5e-3)),\n            'lr': float(np.clip(winner['lr'] * lm, 2.5e-4, 2e-3)),\n            'step_epochs': float(np.clip(winner['step_epochs'] * sm, 0.6, 3.5)),\n            'gamma': float(np.clip(winner['gamma'] + dg, 0.25, 0.85)),\n            'half_life': float(np.clip(winner['half_life'] * hm, 2.5, 18.0)),\n        })\n    return result\n\n\ndef append_progress(path, record):\n    with open(path, 'a') as fh:\n        fh.write(json.dumps(record, sort_keys=True) + '\\n')\n\n\ndef main():\n    parser = argparse.ArgumentParser()\n    parser.add_argument('--data-dir', required=True)\n    parser.add_argument('--out-dir', required=True)\n    parser.add_argument('--seed', type=int, default=42)\n    parser.add_argument('--epochs', type=int, default=16)\n    args = parser.parse_args()\n    os.makedirs(args.out_dir, exist_ok=True)\n    progress_path = os.path.join(args.out_dir, 'progress.log')\n    if os.path.exists(progress_path):\n        os.remove(progress_path)\n    seed_all(args.seed)\n    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n    train, val, fast_path = load_data(args.data_dir)\n    if fast_path:\n        from data.official.evaluate import evaluate as evaluator\n    else:\n        from harness.evaluate_provisional import evaluate as evaluator\n    ages = date_ordinals(train['date'])\n    recency_age = ages.max() - ages\n    pair_pool = build_pair_pools(train['user'], train['y'])\n    if len(pair_pool['po']) == 0:\n        raise RuntimeError('No users with both positive and negative labels for BPR training')\n    smoke = os.environ.get('SMOKE_EPOCHS')\n    smoke_cap = int(smoke) if smoke is not None else None\n    coarse_epochs = min(3, smoke_cap) if smoke_cap is not None else 3\n    refine_epochs = min(5, smoke_cap) if smoke_cap is not None else 5\n    final_epochs = min(args.epochs, smoke_cap) if smoke_cap is not None else args.epochs\n    repetitions = 1 if smoke_cap is not None else 6\n    coarse = coarse_configs(args.seed)\n    if smoke_cap is not None:\n        coarse = coarse[:2]\n    history = []\n    coarse_summary = []\n    probe_number = 0\n    for config_id, config in enumerate(coarse):\n        scores = []\n        for rep in range(repetitions):\n            probe_seed = args.seed + 1000 + config_id * 31 + rep\n            metrics, _, curve = train_candidate(config, train, val, pair_pool, recency_age,\n                                                  evaluator, device, probe_seed,\n                                                  coarse_epochs, False)\n            probe_number += 1\n            record = {\n                'stage': 'coarse', 'probe': probe_number, 'config_id': config_id,\n                'replicate': rep, 'seed': probe_seed, 'config': config,\n                'primary': metrics['primary'], 'gauc': metrics['gauc'], 'curve': curve,\n            }\n            history.append(record)\n            append_progress(progress_path, record)\n            scores.append(metrics['primary'])\n        coarse_summary.append({'config': config, 'mean': float(np.mean(scores)),\n                               'std': float(np.std(scores)), 'scores': scores})\n    coarse_summary.sort(key=lambda x: x['mean'], reverse=True)\n    stage1_winner = coarse_summary[0]['config']\n    refined = refined_configs(stage1_winner)\n    if smoke_cap is not None:\n        refined = refined[:2]\n    refine_summary = []\n    for config_id, config in enumerate(refined):\n        scores = []\n        for rep in range(repetitions):\n            probe_seed = args.seed + 100000 + config_id * 37 + rep\n            metrics, _, curve = train_candidate(config, train, val, pair_pool, recency_age,\n                                                  evaluator, device, probe_seed,\n                                                  refine_epochs, True)\n            probe_number += 1\n            record = {\n                'stage': 'refine', 'probe': probe_number, 'config_id': config_id,\n                'replicate': rep, 'seed': probe_seed, 'config': config,\n                'primary': metrics['primary'], 'gauc': metrics['gauc'], 'curve': curve,\n            }\n            history.append(record)\n            append_progress(progress_path, record)\n            scores.append(metrics['primary'])\n        refine_summary.append({'config': config, 'mean': float(np.mean(scores)),\n                               'std': float(np.std(scores)), 'scores': scores})\n    refine_summary.sort(key=lambda x: x['mean'], reverse=True)\n    winning_config = refine_summary[0]['config']\n    final_seed = args.seed + 900001\n    final_metrics, final_scores, final_curve = train_candidate(\n        winning_config, train, val, pair_pool, recency_age, evaluator, device,\n        final_seed, final_epochs, True)\n    final_record = {\n        'stage': 'final', 'seed': final_seed, 'epochs': final_epochs,\n        'config': winning_config, 'primary': final_metrics['primary'],\n        'gauc': final_metrics['gauc'], 'curve': final_curve,\n    }\n    history.append(final_record)\n    append_progress(progress_path, final_record)\n    output_metrics = {\n        'gauc': final_metrics['gauc'],\n        'ndcg5': final_metrics['ndcg5'],\n        'primary': final_metrics['primary'],\n        'winning_config': winning_config,\n        'coarse_summary': coarse_summary,\n        'refine_summary': refine_summary,\n        'history': history,\n    }\n    with open(os.path.join(args.out_dir, 'metrics.json'), 'w') as fh:\n        json.dump(output_metrics, fh)\n    users = np.asarray(val['user'])\n    videos = np.asarray(val['video'])\n    with open(os.path.join(args.out_dir, 'predictions.csv'), 'w') as fh:\n        fh.write('row_id,user_id,video_id,score\\n')\n        for i, score in enumerate(final_scores):\n            fh.write(f'{i},{users[i]},{videos[i]},{score:.8g}\\n')\n\n\nif __name__ == '__main__':\n    main()\n"}
