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
node_001 [<-node_000] draft "Because validation peaks before training loss stops improving, overfitting is the diagnosis; a validation-tuned DCN-lite package combining 0.5 BCE/0.5 within-user BPR, embedding/MLP dropout, AdamW, rapid step decay, and recency weighting will improve validation primary by at least 0.0025 over the 0.6018 FM parent." primary=0.6014 REJECTED

Mode: DRAFT. Write a fresh script derived from the parent (the current baseline family) implementing the SELECTED card (or, when a launch directive names a tier, that directive). Keep everything else identical to the parent.

## Selected method (implement THIS)
### seq-deepfm-composite: Sequence DeepFM + watch-time aux + session context (full package)
- mechanism: The complete measured composite on top of seq-deepfm-author-history: (a) base Sequence DeepFM — 5 fields + hour/weekday/is_rand context + mean-pooled causal 12-author history; (b) censor-aware watch-time AUXILIARY head (play_time treated as censored at duration; training-only supervision) — NOTE: watch-time aux is measured DEAD on FM/DCN architectures but measured ALIVE (+0.0005) on this DeepFM base — the dead-list is architecture-conditional; (c) causal session metadata fields (gap-since-previous-impression buckets, within-session position); (d) close: mean-LOGIT average of 2-3 independently seeded members.
- kind: opportunity
- treats: flat-signal | data-shift | variance
- reference_primary: 0.6043-0.6046 as implemented by autonomous runs (the in-run expectation to plan against). HUMAN-ASSISTED dev evidence, NOT reproduced by any autonomous run and NOT an in-run headroom figure: teammate singles 0.6051-0.6058 across 7 seeds; 2-member mean-logit 0.606116 (independently re-verified from saved score files) — teammate research branch codex/project-2-kuairand-agent runs/RESULTS.md
- preconditions: Implement the FULL package — a partial port measured only 0.6033 (run_menu_m4): the context fields, causal history discipline, and aux head must all be present. 2-member selection from a 7-seed sweep carries selection optimism ~0.0003-5; prefer predeclared consecutive seeds.
- citation: teammate research (Aditya), independently verified; DeepFM (Guo et al. 2017); censored watch-time modeling (CWM, KDD'24)
- expected_gain / cost: package 0.6043-0.6045 as agent-implemented (probe: 3-seed rank-avg 0.60477 — close adds only +0.0003, members correlated); teammate reference 0.6055-0.6061 requires FULL fidelity (context fields + aux head + logit averaging) / high.
- status_pure: measured-win as agent-implemented (0.6043-0.6046); the teammate's 0.6055-0.6061 tier is human-assisted evidence not yet reproduced under this harness
- status_1k: untried
selector diagnosis: overfit
selector why: Validation peaks at epoch 8 and declines while training loss continues downward, indicating overfit. The rejected DCN-package opening should not be repeated. This distinct, literature-grounded full package is eligible despite the excluded overfit/underfit treatment families because it treats flat-signal, data-shift, and variance. Autonomous evidence supports roughly 0.6043-0.6046, an honest +0.0025 to +0.0028 over the current-run best 0.6018, so it has a realistic chance to clear the 0.002 convergence threshold. Implement the full causal-history, context, censored auxiliary, regularized DeepFM, and predeclared 2-3 seed logit-average recipe; do not rely on the higher human-assisted 0.6055-0.6061 tier.

## Convergence pressure
streak_state = {'no_improve_streak': 1, 'n_converge': 3, 'iters_left': 14}
The run ends after N consecutive iterations whose best-so-far improvement is <= epsilon = 0.002. Select experiments by expected scientific value given the remaining budget: at every iteration, including the first, prefer the eligible move with the largest evidence-supported expected gain for its cost; an early iteration spent on a small-ceiling treatment is a convergence strike bought at full price. Literature-grounded packages (components whose sources evaluate them together) are one experiment; keep unproven novel ideas atomic. Plan the run so its final iterations produce the strongest possible finished artifact rather than leaving the run un-finalized. Do the epsilon arithmetic before choosing: if the streak means the run ends unless THIS iteration improves best-so-far by at least epsilon, then a move whose own evidence caps its gain below epsilon cannot extend the run no matter how proven it is; on such an iteration prefer the eligible move with the largest evidence-supported expected gain at or above epsilon, and among qualifying moves prefer the one whose evidence clears epsilon with the widest margin: a move whose evidence only just reaches the bar fails it about half the time, so bare arithmetic reach is not parity with a wide-margin alternative (combining decorrelated mechanism families generally out-gains both re-seeding one family and any single atomic mechanism). Read margins against the CURRENT best, not a card's original baseline: an unspent package whose measured absolute score sits near the current best offers almost no headroom, while a close whose evidence exceeds every single-model score in the ledger offers the most. A proven small-gain close is the right pick only when no eligible move has evidence reaching epsilon. Do not change what counts as an iteration in response to the streak.

## Runtime budget (overrides the 600s default above)
THIS run's per-node timeout is 7200 seconds (~120 minutes). A full-length training on the npz fast path costs roughly 40-90s on CPU, far less on GPU. Plan to SPEND ~60-70% of this budget on search probes when playing a search card — e.g. at 2+ hours that is 40+ full-length probes plus refinement, not 8. Reserve the remainder for the final training(s). Finishing a search node in a small fraction of the budget is a defect, not efficiency: unspent budget is free score variance left unexplored.

Directive: draft from Tier 2 of the menu

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
{"execution_kind":"script","hypothesis":"Validation peaking and then declining diagnoses overfitting; replacing the flat FM with the complete regularized Sequence DeepFM composite—causal 12-author history, temporal and causal session context, censor-aware watch-time auxiliary supervision, and a predeclared three-seed mean-logit close—will improve validation primary by approximately 0.0026 over the 0.6018 parent.","expected_delta":0.0026,"expected_delta_basis":"The selected seq-deepfm-composite card reports autonomous full-package validation primary of 0.6043-0.6046 versus the 0.6018 parent, while warning that partial ports fail.","action":"draft","parent":"node_000","code":"\"\"\"Full Sequence DeepFM composite with causal history, session context, censored\nwatch-time auxiliary supervision, and a predeclared three-seed logit ensemble.\"\"\"\nimport argparse\nimport csv\nimport datetime\nimport json\nimport os\nimport sys\n\nimport numpy as np\nimport torch\n\nsys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\n\n\ndef seed_everything(seed):\n    np.random.seed(seed)\n    torch.manual_seed(seed)\n    if torch.cuda.is_available():\n        torch.cuda.manual_seed_all(seed)\n    if hasattr(torch.backends, \"cudnn\"):\n        torch.backends.cudnn.deterministic = True\n        torch.backends.cudnn.benchmark = False\n\n\ndef date_parts(values):\n    out = np.zeros(len(values), dtype=np.int64)\n    days = np.zeros(len(values), dtype=np.int64)\n    cache = {}\n    for i, value in enumerate(values):\n        iv = int(value)\n        if iv not in cache:\n            text = str(iv)\n            try:\n                dt = datetime.datetime.strptime(text, \"%Y%m%d\").date()\n                cache[iv] = (dt.weekday(), dt.toordinal())\n            except ValueError:\n                cache[iv] = (iv % 7, iv)\n        out[i], days[i] = cache[iv]\n    return out, days\n\n\ndef hour_and_minute(values):\n    values = np.asarray(values, dtype=np.int64)\n    if len(values) == 0:\n        return values.copy(), values.copy()\n    if int(np.max(values)) > 1439:\n        hour = np.clip(values // 100, 0, 23)\n        minute = hour * 60 + np.clip(values % 100, 0, 59)\n    elif int(np.max(values)) > 23:\n        minute = np.clip(values, 0, 1439)\n        hour = minute // 60\n    else:\n        hour = np.clip(values, 0, 23)\n        minute = hour * 60\n    return hour.astype(np.int64), minute.astype(np.int64)\n\n\ndef encode_offsets(train_raw, val_raw):\n    train_cols = []\n    val_cols = []\n    dims = []\n    offset = 0\n    for tr_col, va_col in zip(train_raw, val_raw):\n        mapping = {}\n        tr_enc = np.empty(len(tr_col), dtype=np.int64)\n        for i, value in enumerate(tr_col):\n            if value not in mapping:\n                mapping[value] = len(mapping)\n            tr_enc[i] = mapping[value]\n        unknown = len(mapping)\n        va_enc = np.empty(len(va_col), dtype=np.int64)\n        for i, value in enumerate(va_col):\n            va_enc[i] = mapping.get(value, unknown)\n        dim = unknown + 1\n        train_cols.append(tr_enc + offset)\n        val_cols.append(va_enc + offset)\n        dims.append(dim)\n        offset += dim\n    return np.stack(train_cols, axis=1), np.stack(val_cols, axis=1), np.asarray(dims, dtype=np.int64)\n\n\ndef load_csv_data(data_dir):\n    def read_file(path, training):\n        user = []\n        video = []\n        tab = []\n        hourmin = []\n        date = []\n        duration = []\n        labels = []\n        play = []\n        with open(path, \"r\", newline=\"\") as fh:\n            reader = csv.DictReader(fh)\n            for row in reader:\n                user.append(row[\"user_id\"])\n                video.append(row[\"video_id\"])\n                tab.append(row[\"tab\"])\n                hourmin.append(int(float(row[\"hourmin\"])))\n                date.append(int(float(row[\"date\"])))\n                duration.append(float(row[\"duration_ms\"]))\n                labels.append(float(row[\"long_view\"]))\n                if training:\n                    play.append(float(row[\"play_time_ms\"]))\n        return {\"user_raw\": np.asarray(user), \"video_raw\": np.asarray(video),\n                \"tab_raw\": np.asarray(tab), \"hourmin\": np.asarray(hourmin, dtype=np.int64),\n                \"date\": np.asarray(date, dtype=np.int64),\n                \"duration_ms\": np.asarray(duration, dtype=np.float32),\n                \"y\": np.asarray(labels, dtype=np.float32),\n                \"play_time_ms\": np.asarray(play, dtype=np.float32) if training else None}\n\n    tr = read_file(os.path.join(data_dir, \"train.csv\"), True)\n    va = read_file(os.path.join(data_dir, \"val.csv\"), False)\n    quantiles = np.quantile(tr[\"duration_ms\"], np.linspace(0.0, 1.0, 11)[1:-1])\n    tr_bucket = np.searchsorted(quantiles, tr[\"duration_ms\"], side=\"right\").astype(str)\n    va_bucket = np.searchsorted(quantiles, va[\"duration_ms\"], side=\"right\").astype(str)\n    # The public CSV schema has no author_id; video_id is the deterministic fallback.\n    tr_fields = [tr[\"user_raw\"], tr[\"video_raw\"], tr[\"video_raw\"], tr[\"tab_raw\"], tr_bucket]\n    va_fields = [va[\"user_raw\"], va[\"video_raw\"], va[\"video_raw\"], va[\"tab_raw\"], va_bucket]\n    Xt, Xv, dims = encode_offsets(tr_fields, va_fields)\n    tr[\"X\"] = Xt.astype(np.int32)\n    va[\"X\"] = Xv.astype(np.int32)\n    tr[\"field_dims\"] = dims\n    va[\"field_dims\"] = dims\n    tr[\"user\"] = tr[\"user_raw\"]\n    va[\"user\"] = va[\"user_raw\"]\n    return tr, va, False\n\n\ndef load_data(data_dir):\n    train_npz = os.path.join(data_dir, \"train.npz\")\n    val_npz = os.path.join(data_dir, \"val.npz\")\n    if os.path.exists(train_npz) and os.path.exists(val_npz):\n        trn = np.load(train_npz)\n        van = np.load(val_npz)\n        tr = {key: trn[key] for key in trn.files}\n        va = {key: van[key] for key in van.files}\n        tr[\"video_raw\"] = np.zeros(len(tr[\"y\"]), dtype=np.int64)\n        va[\"video_raw\"] = np.zeros(len(va[\"y\"]), dtype=np.int64)\n        return tr, va, True\n    return load_csv_data(data_dir)\n\n\ndef evaluator(fast_path):\n    if fast_path:\n        from data.official.evaluate import evaluate\n    else:\n        from harness.evaluate_provisional import evaluate\n    return evaluate\n\n\ndef metric_dict(evaluate_fn, users, labels, scores):\n    m = evaluate_fn(users, labels.astype(int), scores)\n    return {\"gauc\": float(m.get(\"GAUC\", m.get(\"gauc\"))),\n            \"ndcg5\": float(m.get(\"nDCG@5\", m.get(\"ndcg5\"))),\n            \"primary\": float(m[\"primary\"])}\n\n\ndef local_tab(X, field_dims):\n    return X[:, 3].astype(np.int64) - int(np.sum(field_dims[:3]))\n\n\ndef build_causal_features(tr, va):\n    ntr = len(tr[\"y\"])\n    nva = len(va[\"y\"])\n    histories_tr = np.full((ntr, 12), -1, dtype=np.int32)\n    histories_va = np.full((nva, 12), -1, dtype=np.int32)\n    gap_tr = np.full(ntr, 8, dtype=np.int64)\n    gap_va = np.full(nva, 8, dtype=np.int64)\n    pos_tr = np.zeros(ntr, dtype=np.int64)\n    pos_va = np.zeros(nva, dtype=np.int64)\n\n    weekday_tr, day_tr = date_parts(tr[\"date\"])\n    weekday_va, day_va = date_parts(va[\"date\"])\n    hour_tr, minute_tr = hour_and_minute(tr[\"hourmin\"])\n    hour_va, minute_va = hour_and_minute(va[\"hourmin\"])\n    time_tr = day_tr * 1440 + minute_tr\n    time_va = day_va * 1440 + minute_va\n    tab_tr = local_tab(tr[\"X\"], tr[\"field_dims\"])\n    tab_va = local_tab(va[\"X\"], tr[\"field_dims\"])\n    rand_tr = (tab_tr != 0).astype(np.int64)\n    rand_va = (tab_va != 0).astype(np.int64)\n\n    users = np.concatenate([np.asarray(tr[\"user\"]), np.asarray(va[\"user\"])])\n    times = np.concatenate([time_tr, time_va])\n    split = np.concatenate([np.zeros(ntr, dtype=np.int8), np.ones(nva, dtype=np.int8)])\n    row = np.concatenate([np.arange(ntr), np.arange(nva)])\n    authors = np.concatenate([tr[\"X\"][:, 2], va[\"X\"][:, 2]]).astype(np.int32)\n    original = np.arange(ntr + nva)\n    order = np.lexsort((original, split, times, users.astype(str)))\n    state = {}\n    gap_edges = np.asarray([0, 1, 2, 5, 10, 30, 60, 180], dtype=np.int64)\n    for combined_index in order:\n        user = users[combined_index]\n        current_time = int(times[combined_index])\n        history, previous_time, previous_pos = state.get(user, ([], None, -1))\n        if previous_time is None:\n            gap_bucket = 8\n            session_pos = 0\n        else:\n            gap_minutes = max(0, current_time - previous_time)\n            gap_bucket = int(np.searchsorted(gap_edges, gap_minutes, side=\"right\") - 1)\n            gap_bucket = max(0, min(7, gap_bucket))\n            session_pos = 0 if gap_minutes > 30 else min(15, previous_pos + 1)\n        hist_values = history[-12:]\n        if split[combined_index] == 0:\n            r = int(row[combined_index])\n            if hist_values:\n                histories_tr[r, :len(hist_values)] = hist_values\n            gap_tr[r] = gap_bucket\n            pos_tr[r] = session_pos\n        else:\n            r = int(row[combined_index])\n            if hist_values:\n                histories_va[r, :len(hist_values)] = hist_values\n            gap_va[r] = gap_bucket\n            pos_va[r] = session_pos\n        history = (history + [int(authors[combined_index])])[-12:]\n        state[user] = (history, current_time, session_pos)\n\n    base_dim = int(np.sum(tr[\"field_dims\"]))\n    offsets = [base_dim, base_dim + 24, base_dim + 31, base_dim + 33, base_dim + 42]\n    context_tr = np.stack([hour_tr + offsets[0], weekday_tr + offsets[1],\n                           rand_tr + offsets[2], gap_tr + offsets[3], pos_tr + offsets[4]], axis=1)\n    context_va = np.stack([hour_va + offsets[0], weekday_va + offsets[1],\n                           rand_va + offsets[2], gap_va + offsets[3], pos_va + offsets[4]], axis=1)\n    total_dim = base_dim + 58\n    return (context_tr.astype(np.int32), context_va.astype(np.int32), histories_tr,\n            histories_va, total_dim)\n\n\nclass ParentFM(torch.nn.Module):\n    def __init__(self, total_dim, k=16):\n        super().__init__()\n        self.emb = torch.nn.Embedding(total_dim, k)\n        self.lin = torch.nn.Embedding(total_dim, 1)\n        self.bias = torch.nn.Parameter(torch.zeros(1))\n        torch.nn.init.normal_(self.emb.weight, std=0.01)\n        torch.nn.init.zeros_(self.lin.weight)\n\n    def forward(self, x):\n        e = self.emb(x)\n        s = e.sum(1)\n        pair = 0.5 * (s * s - (e * e).sum(1)).sum(1)\n        return self.bias + self.lin(x).sum((1, 2)) + pair\n\n\nclass SequenceDeepFM(torch.nn.Module):\n    def __init__(self, total_dim, k=16, dropout=0.20):\n        super().__init__()\n        self.emb = torch.nn.Embedding(total_dim, k)\n        self.lin = torch.nn.Embedding(total_dim, 1)\n        self.bias = torch.nn.Parameter(torch.zeros(1))\n        torch.nn.init.normal_(self.emb.weight, std=0.01)\n        torch.nn.init.zeros_(self.lin.weight)\n        self.mlp = torch.nn.Sequential(\n            torch.nn.Linear(11 * k, 128),\n            torch.nn.ReLU(),\n            torch.nn.Dropout(dropout),\n            torch.nn.Linear(128, 64),\n            torch.nn.ReLU(),\n            torch.nn.Dropout(dropout),\n        )\n        self.main_head = torch.nn.Linear(64, 1)\n        self.watch_head = torch.nn.Linear(64, 1)\n\n    def forward(self, x, context, history):\n        ids = torch.cat([x, context], dim=1)\n        current_e = self.emb(ids)\n        mask = (history >= 0).float().unsqueeze(-1)\n        safe_history = history.clamp_min(0)\n        hist_e = (self.emb(safe_history) * mask).sum(1) / mask.sum(1).clamp_min(1.0)\n        fields = torch.cat([current_e, hist_e.unsqueeze(1)], dim=1)\n        summed = fields.sum(1)\n        pair = 0.5 * (summed.square() - fields.square().sum(1)).sum(1)\n        linear = self.lin(ids).sum((1, 2))\n        hist_linear = (self.lin(safe_history) * mask).sum(1) / mask.sum(1).clamp_min(1.0)\n        deep = self.mlp(fields.flatten(1))\n        logit = self.bias + linear + hist_linear.squeeze(1) + pair + self.main_head(deep).squeeze(1)\n        watch = self.watch_head(deep).squeeze(1)\n        return logit, watch\n\n\ndef predict_parent(model, X, device):\n    model.eval()\n    output = []\n    with torch.no_grad():\n        for start in range(0, len(X), 65536):\n            xb = torch.from_numpy(X[start:start + 65536].astype(np.int64)).to(device)\n            output.append(model(xb).detach().cpu().numpy())\n    return np.concatenate(output)\n\n\ndef train_parent_reference(tr, va, device, epochs):\n    seed_everything(42)\n    total_dim = int(np.sum(tr[\"field_dims\"]))\n    model = ParentFM(total_dim).to(device)\n    opt = torch.optim.Adam(model.parameters(), lr=1e-3)\n    criterion = torch.nn.BCEWithLogitsLoss()\n    n = len(tr[\"y\"])\n    rng = np.random.RandomState(42)\n    X = tr[\"X\"].astype(np.int64)\n    y = tr[\"y\"].astype(np.float32)\n    for _ in range(epochs):\n        model.train()\n        perm = rng.permutation(n)\n        for start in range(0, n, 8192):\n            ids = perm[start:start + 8192]\n            xb = torch.from_numpy(X[ids]).to(device)\n            yb = torch.from_numpy(y[ids]).to(device)\n            opt.zero_grad(set_to_none=True)\n            loss = criterion(model(xb), yb)\n            loss.backward()\n            opt.step()\n    return predict_parent(model, va[\"X\"], device)\n\n\ndef predict_composite(model, X, context, history, device):\n    model.eval()\n    output = []\n    with torch.no_grad():\n        for start in range(0, len(X), 32768):\n            end = start + 32768\n            xb = torch.from_numpy(X[start:end].astype(np.int64)).to(device)\n            cb = torch.from_numpy(context[start:end].astype(np.int64)).to(device)\n            hb = torch.from_numpy(history[start:end].astype(np.int64)).to(device)\n            logits, _ = model(xb, cb, hb)\n            output.append(logits.detach().cpu().numpy())\n    return np.concatenate(output)\n\n\ndef train_member(seed, tr, va, context_tr, context_va, history_tr, history_va,\n                 total_dim, device, epochs, evaluate_fn):\n    seed_everything(seed)\n    model = SequenceDeepFM(total_dim).to(device)\n    opt = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=2e-5)\n    milestones = sorted(set([max(1, epochs // 3), max(2, (2 * epochs) // 3)]))\n    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=0.35)\n    bce = torch.nn.BCEWithLogitsLoss()\n    X = tr[\"X\"].astype(np.int64)\n    y = tr[\"y\"].astype(np.float32)\n    play = np.maximum(tr[\"play_time_ms\"].astype(np.float32), 0.0)\n    duration = np.maximum(tr[\"duration_ms\"].astype(np.float32), 1.0)\n    watch_target = (np.log1p(np.minimum(play, duration)) / 10.0).astype(np.float32)\n    censored = ((play >= duration) & (duration > 1.0)).astype(np.float32)\n    n = len(y)\n    rng = np.random.RandomState(seed)\n    best_primary = -1.0\n    best_scores = None\n    best_state = None\n    patience = 0\n    history_log = []\n    for epoch in range(epochs):\n        model.train()\n        perm = rng.permutation(n)\n        running = 0.0\n        batches = 0\n        for start in range(0, n, 4096):\n            ids = perm[start:start + 4096]\n            xb = torch.from_numpy(X[ids]).to(device)\n            cb = torch.from_numpy(context_tr[ids].astype(np.int64)).to(device)\n            hb = torch.from_numpy(history_tr[ids].astype(np.int64)).to(device)\n            yb = torch.from_numpy(y[ids]).to(device)\n            tb = torch.from_numpy(watch_target[ids]).to(device)\n            zb = torch.from_numpy(censored[ids]).to(device)\n            opt.zero_grad(set_to_none=True)\n            logits, watch_pred = model(xb, cb, hb)\n            main_loss = bce(logits, yb)\n            uncensored_loss = torch.nn.functional.smooth_l1_loss(watch_pred, tb, reduction=\"none\")\n            censored_loss = torch.relu(tb - watch_pred).square()\n            watch_loss = ((1.0 - zb) * uncensored_loss + zb * censored_loss).mean()\n            loss = main_loss + 0.05 * watch_loss\n            loss.backward()\n            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)\n            opt.step()\n            running += float(loss.detach().cpu())\n            batches += 1\n        scheduler.step()\n        scores = predict_composite(model, va[\"X\"], context_va, history_va, device)\n        metrics = metric_dict(evaluate_fn, va[\"user\"], va[\"y\"], scores)\n        history_log.append({\"epoch\": epoch + 1,\n                            \"train_loss\": round(running / max(1, batches), 6),\n                            \"lr\": float(opt.param_groups[0][\"lr\"]),\n                            \"val_gauc\": round(metrics[\"gauc\"], 6),\n                            \"val_primary\": round(metrics[\"primary\"], 6)})\n        if metrics[\"primary\"] > best_primary + 1e-6:\n            best_primary = metrics[\"primary\"]\n            best_scores = scores.copy()\n            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}\n            patience = 0\n        else:\n            patience += 1\n            if patience >= 3:\n                break\n    if best_state is not None:\n        model.load_state_dict(best_state)\n    return best_scores, best_primary, history_log\n\n\ndef main():\n    parser = argparse.ArgumentParser()\n    parser.add_argument(\"--data-dir\", required=True)\n    parser.add_argument(\"--out-dir\", required=True)\n    parser.add_argument(\"--seed\", type=int, default=42)\n    parser.add_argument(\"--epochs\", type=int, default=14)\n    args = parser.parse_args()\n    os.makedirs(args.out_dir, exist_ok=True)\n    seed_everything(args.seed)\n    device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n    tr, va, fast_path = load_data(args.data_dir)\n    evaluate_fn = evaluator(fast_path)\n    smoke = os.environ.get(\"SMOKE_EPOCHS\")\n    epochs = args.epochs if smoke is None else min(args.epochs, max(1, int(smoke)))\n    context_tr, context_va, history_tr, history_va, total_dim = build_causal_features(tr, va)\n\n    parent_epochs = min(8, epochs)\n    parent_scores = train_parent_reference(tr, va, device, parent_epochs)\n    member_scores = []\n    member_history = []\n    member_primaries = []\n    progress_path = os.path.join(args.out_dir, \"progress.log\")\n    member_seeds = [args.seed, args.seed + 1, args.seed + 2]\n    for member_seed in member_seeds:\n        scores, primary, history_log = train_member(\n            member_seed, tr, va, context_tr, context_va, history_tr, history_va,\n            total_dim, device, epochs, evaluate_fn)\n        if np.allclose(scores, parent_scores, rtol=1e-7, atol=1e-8):\n            raise AssertionError(\"Composite member predictions equal parent predictions\")\n        for previous in member_scores:\n            if np.allclose(scores, previous, rtol=1e-7, atol=1e-8):\n                raise AssertionError(\"Distinct-seed ensemble members produced identical scores\")\n        member_scores.append(scores)\n        member_primaries.append(float(primary))\n        member_history.append({\"seed\": int(member_seed), \"best_primary\": float(primary),\n                               \"epochs\": history_log})\n        with open(progress_path, \"a\") as fh:\n            fh.write(json.dumps({\"seed\": int(member_seed), \"val_primary\": float(primary),\n                                 \"model\": \"seq_deepfm_composite\"}) + \"\\n\")\n\n    final_scores = np.mean(np.stack(member_scores, axis=0), axis=0)\n    if np.allclose(final_scores, parent_scores, rtol=1e-7, atol=1e-8):\n        raise AssertionError(\"Ensemble predictions equal parent predictions\")\n    final_metrics = metric_dict(evaluate_fn, va[\"user\"], va[\"y\"], final_scores)\n    metrics_output = {\"gauc\": final_metrics[\"gauc\"], \"ndcg5\": final_metrics[\"ndcg5\"],\n                      \"primary\": final_metrics[\"primary\"], \"history\": member_history,\n                      \"member_primaries\": member_primaries,\n                      \"ensemble\": {\"method\": \"mean_logit\", \"seeds\": member_seeds}}\n    with open(os.path.join(args.out_dir, \"metrics.json\"), \"w\") as fh:\n        json.dump(metrics_output, fh)\n    with open(os.path.join(args.out_dir, \"predictions.csv\"), \"w\") as fh:\n        fh.write(\"row_id,user_id,video_id,score\\n\")\n        videos = va.get(\"video_raw\", np.zeros(len(final_scores), dtype=np.int64))\n        for i, score in enumerate(final_scores):\n            fh.write(f\"{i},{va['user'][i]},{videos[i]},{float(score):.8g}\\n\")\n\n\nif __name__ == \"__main__\":\n    main()\n"}
