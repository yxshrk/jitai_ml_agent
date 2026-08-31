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
### diverse-family-farm-close: One-node multi-family member farm + cross-family blend
- kind: opportunity
- note: STRATEGY BUNDLE (composes package-dial-sweep + temporal-pair-kernel + seq-deepfm-composite + recency-weighting + rank aggregation). Cards are normally atomic methods; bundles are the pragmatic vehicle for strategies until the v3 strategy layer.
- mechanism: ONE node, in three phases: (PHASE 1, ~30 min) train ONE probe-fidelity member (1-2 epochs) per candidate family, then evaluate ALL blend combinations of the probe members on validation — blending is nearly free (rank-average of saved score vectors) — to map which families COMPLEMENT; (PHASE 2) full-train only the 2-3 complementary families; (PHASE 3) blend the full members, RE-VERIFY the winning combination at full fidelity (probe-level correlations are an assumption, not a guarantee), emit. Original single-phase form: ONE node that reproduces the campaign's measured cross-family evidence internally: train ONE member from EACH measured-win family, each per its own card's recipe — (a) the regularized DCN package (package-dial-sweep dials, ~0.6042), (b) temporal-pair-kernel on that package (~0.6045-52), (c) seq-deepfm-composite (~0.6044), (d) a recency-weighted FM/DCN variant (~0.6045-50). VALIDATE each member's primary individually (progress-log it; ADMIT only members >=0.6040), then per-user or global RANK-AVERAGE the admitted members. Cross-family decorrelation is the entire point: same-family seed ensembles measured +0.0003; cross-family equal-weight blends of exactly these families measured 0.6058-0.6065 (team evidence probe, 31 Aug).
- treats: variance | plateau
- reference_primary: 0.605863 (selection-free ALL-family equal blend of one member per clean run; best combos 0.6060-0.6065)
- verdict_pure: external-win
- evidence_primary: 0.605863
- preconditions: Budget the node like a sweep (it is 4 trainings + blend): use most of the timeout; log every member's config+primary; obey the ensemble contract (distinct seeds, member-distinctness assertion, never emit parent-identical predictions). A member that fails to train is dropped, not blended.
- citation: team evidence probes 31 Aug (logs/RUNS.md recipe-search line); component recipes: package-dial-sweep, temporal-pair-kernel, seq-deepfm-composite, recency-weighting cards.
- expected_gain / cost: +0.0035-0.0045 over baseline IN ONE NODE (eps-clearing) if >=3 members admit; degrades gracefully to the best single member / high runtime (one node, plan 60-90 min).
- status_pure: untried as a single node (every component + the blend measured separately)
- status_1k: untried
selector diagnosis: overfit
selector why: Validation peaks at 0.601838 in epoch 8 and then declines while training loss continues downward, indicating overfit. As the first iteration with 15 remaining, the highest-value opener is the one-node multi-family farm: it includes a strongly regularized, checkpoint-selected DCN package, admits only competitive members, and exploits cross-family error diversity. Its evidence_primary of 0.605863 implies about +0.0041 over the current 0.6018 best, clearing the 0.002 convergence threshold with more margin than a single package or atomic treatment. Full-fidelity re-verification and snapshot members should be used as required.

## Typed farm-close plan (HARNESS-EXECUTED; overrides whole-script output)
The selected method is a cross-family ensemble strategy (farm-close or
heterogeneous ensemble design). Do NOT write an orchestration script or include
top-level `code`. Every member must carry its own `code` field: write each
member's single-fit script yourself (you cannot see the filesystem, so never
reference a script path you have not been shown in this conversation). Return the ordinary
hypothesis/expected-delta/action/parent fields in a farm-close envelope. The
harness accepts the legacy `farm_close_plan` alias, but prefer `ensemble_plan`:
{"execution_kind":"farm_close",
 "hypothesis":"...", "expected_delta":<honest numeric expectation>,
 "expected_delta_basis":"...", "action":"<draft|improve>", "parent":"node_NNN",
 "timeout_s":7200,
 "ensemble_plan":{
   "probe_epochs":2,
   "full_member_limit":3, "min_probe_blend_gain":0.0,
   "members":[
     {"family":"<distinct-family-id>",
      "code":"<a COMPLETE single-fit training script as a JSON string, per the
node contract: reads --data-dir/--out-dir/--seed, honors SMOKE_EPOCHS, ONE
training trajectory, no internal search or ensembling>",
      "config":{}, "seed":42}
   ],
   "blend":{"weights":"equal","aggregations":[
     {"method":"rank_average","scope":"per_user"}]}}}

Schema rules enforced before execution: exactly 4-6 members; every family and seed
is distinct; optional member_id values must also be distinct; each member has
exactly one of `script_source` or `code` (a whole generated member script), plus a
`config` object of CLI dials and an integer seed; probe_epochs is 1 or 2;
full_member_limit is 2 or 3; min_probe_blend_gain defaults to 0 and is a strict
promotion threshold. Blends use equal weights and one or two declared
rank-average aggregation rules with `per_user` or `global` scope; the harness
counts and exhaustively evaluates their complete finite subset enumeration, never
truncating it based on observed results. Unknown fields are invalid. Use genuinely
different model/objective families. The harness will train probes and selected
full members concurrently, use the best probe singleton as the promotion anchor,
freeze the winning full recipe, re-evaluate it from saved full vectors, and emit
the final node artifacts. A full singleton or incumbent is a valid recorded
fallback. Budget this sweep for 60-120 minutes without exceeding the supplied
per-node timeout.

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
{"execution_kind":"farm_close","hypothesis":"The validation peak followed by decline diagnoses overfitting; rank-averaging complementary regularized DCN, context-conditioned pair-kernel, causal sequence DeepFM, and recency-weighted hybrid FM members will improve validation primary by about 0.004 through stronger regularization and cross-family error decorrelation.","expected_delta":0.004,"expected_delta_basis":"The selected diverse-family-farm-close card reports cross-family blends at 0.6058-0.6065 versus the parent primary 0.6018, while same-family ensembles add only about 0.0003.","action":"draft","parent":"node_000","timeout_s":7200,"ensemble_plan":{"probe_epochs":2,"full_member_limit":3,"min_probe_blend_gain":0.0,"members":[{"family":"regularized-dcn-package","code":"import argparse, csv, json, os, sys\nimport numpy as np\nimport torch\n\nsys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\n\n\ndef load_data(data_dir):\n    tp = os.path.join(data_dir, 'train.npz')\n    vp = os.path.join(data_dir, 'val.npz')\n    if os.path.exists(tp) and os.path.exists(vp):\n        from data.official.evaluate import evaluate\n        tr, va = np.load(tp), np.load(vp)\n        dims = tr['field_dims'].astype(np.int64)\n        video = va['X'][:, 1].astype(np.int64) - int(dims[0])\n        return tr['X'].astype(np.int64), tr['y'].astype(np.float32), va['X'].astype(np.int64), va['y'].astype(np.int64), va['user'], video, dims, evaluate\n    from harness.evaluate_provisional import evaluate\n    def rows(path, train):\n        with open(path, newline='') as f:\n            out = []\n            for r in csv.DictReader(f):\n                out.append((r['user_id'], r['video_id'], r['video_id'], r['tab'], float(r['duration_ms']), int(float(r['long_view']))))\n        return out\n    rt, rv = rows(os.path.join(data_dir, 'train.csv'), True), rows(os.path.join(data_dir, 'val.csv'), False)\n    cuts = np.quantile(np.asarray([r[4] for r in rt]), np.linspace(0.1, 0.9, 9))\n    maps = []\n    for j in range(4):\n        vals = sorted(set(r[j] for r in rt))\n        maps.append({v: i + 1 for i, v in enumerate(vals)})\n    dims = np.asarray([len(m) + 2 for m in maps] + [11], dtype=np.int64)\n    off = np.concatenate(([0], np.cumsum(dims[:-1])))\n    def enc(rr):\n        x = np.empty((len(rr), 5), np.int64)\n        for i, r in enumerate(rr):\n            for j in range(4): x[i, j] = off[j] + maps[j].get(r[j], 0)\n            x[i, 4] = off[4] + np.searchsorted(cuts, r[4], side='right')\n        return x\n    return enc(rt), np.asarray([r[5] for r in rt], np.float32), enc(rv), np.asarray([r[5] for r in rv], np.int64), np.asarray([r[0] for r in rv]), np.asarray([r[1] for r in rv]), dims, evaluate\n\n\nclass DCN(torch.nn.Module):\n    def __init__(self, total, fields=5, k=16):\n        super().__init__()\n        self.emb = torch.nn.Embedding(total, k)\n        self.lin = torch.nn.Embedding(total, 1)\n        d = fields * k\n        self.cw = torch.nn.Parameter(torch.empty(2, d))\n        self.cb = torch.nn.Parameter(torch.zeros(2, d))\n        self.deep = torch.nn.Sequential(torch.nn.Linear(d, 128), torch.nn.ReLU(), torch.nn.Dropout(0.22), torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Dropout(0.12), torch.nn.Linear(64, 1))\n        self.cross_out = torch.nn.Linear(d, 1)\n        self.bias = torch.nn.Parameter(torch.zeros(1))\n        torch.nn.init.normal_(self.emb.weight, std=0.01)\n        torch.nn.init.zeros_(self.lin.weight)\n        torch.nn.init.normal_(self.cw, std=0.01)\n    def forward(self, x):\n        e = self.emb(x)\n        s = e.sum(1)\n        fm = 0.5 * (s.square() - e.square().sum(1)).sum(1)\n        x0 = e.flatten(1)\n        z = x0\n        for i in range(2): z = x0 * (z * self.cw[i]).sum(1, keepdim=True) + self.cb[i] + z\n        return self.bias + self.lin(x).sum((1, 2)) + fm + self.cross_out(z).squeeze(1) + self.deep(x0).squeeze(1)\n\n\ndef main():\n    ap = argparse.ArgumentParser()\n    ap.add_argument('--data-dir', required=True)\n    ap.add_argument('--out-dir', required=True)\n    ap.add_argument('--seed', type=int, default=42)\n    ap.add_argument('--epochs', type=int, default=14)\n    a = ap.parse_args()\n    smoke = os.environ.get('SMOKE_EPOCHS')\n    epochs = min(a.epochs, int(smoke)) if smoke else a.epochs\n    np.random.seed(a.seed); torch.manual_seed(a.seed)\n    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)\n    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False\n    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n    xt, yt, xv, yv, users, videos, dims, evaluate = load_data(a.data_dir)\n    xt = torch.from_numpy(xt); yt = torch.from_numpy(yt); xv = torch.from_numpy(xv)\n    model = DCN(int(dims.sum())).to(dev)\n    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=3e-5)\n    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.55)\n    lossfn = torch.nn.BCEWithLogitsLoss(label_smoothing=0.02)\n    gen = torch.Generator().manual_seed(a.seed)\n    best = -1.; best_scores = None; wait = 0; hist = []; bs = 8192\n    for ep in range(epochs):\n        model.train(); perm = torch.randperm(len(yt), generator=gen); last = 0.\n        for p in range(0, len(yt), bs):\n            ii = perm[p:p+bs]; xb = xt[ii].to(dev); yb = yt[ii].to(dev)\n            opt.zero_grad(set_to_none=True); loss = lossfn(model(xb), yb); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); last = float(loss.detach().cpu())\n        sched.step(); model.eval(); parts = []\n        with torch.no_grad():\n            for p in range(0, len(xv), 65536): parts.append(model(xv[p:p+65536].to(dev)).cpu().numpy())\n        scores = np.concatenate(parts); m = evaluate(users, yv, scores); primary = float(m['primary'])\n        hist.append({'epoch': ep+1, 'train_loss': round(last, 6), 'val_primary': round(primary, 6)})\n        if primary > best + 1e-6: best, best_scores, wait = primary, scores.copy(), 0\n        else:\n            wait += 1\n            if wait >= 3: break\n    os.makedirs(a.out_dir, exist_ok=True)\n    m = evaluate(users, yv, best_scores)\n    with open(os.path.join(a.out_dir, 'metrics.json'), 'w') as f: json.dump({'gauc': m.get('GAUC', m.get('gauc')), 'ndcg5': m.get('nDCG@5', m.get('ndcg5')), 'primary': m['primary'], 'history': hist}, f)\n    with open(os.path.join(a.out_dir, 'predictions.csv'), 'w') as f:\n        f.write('row_id,user_id,video_id,score\\n')\n        for i, s in enumerate(best_scores): f.write(f'{i},{users[i]},{videos[i]},{s:.9g}\\n')\n\n\nif __name__ == '__main__': main()\n","config":{"epochs":14},"seed":42},{"family":"context-stratified-pair-kernel","code":"import argparse, json, os, sys\nimport numpy as np\nimport torch\n\nsys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\nfrom data.official.evaluate import evaluate\n\n\nclass PairKernel(torch.nn.Module):\n    def __init__(self, total, k=16):\n        super().__init__()\n        self.emb = torch.nn.Embedding(total, k)\n        self.lin = torch.nn.Embedding(total, 1)\n        self.hour = torch.nn.Embedding(24, 8)\n        self.day = torch.nn.Embedding(8, 4)\n        self.kernel = torch.nn.Sequential(torch.nn.Linear(12 + 2*k, 64), torch.nn.ReLU(), torch.nn.Dropout(0.18), torch.nn.Linear(64, 10))\n        self.bias = torch.nn.Parameter(torch.zeros(1))\n        torch.nn.init.normal_(self.emb.weight, std=0.01); torch.nn.init.zeros_(self.lin.weight)\n    def forward(self, x, hour, day):\n        e = self.emb(x)\n        ctx = torch.cat((self.hour(hour), self.day(day), e[:, 3], e[:, 4]), 1)\n        w = 0.5 + torch.sigmoid(self.kernel(ctx))\n        vals = []; q = 0\n        for i in range(5):\n            for j in range(i+1, 5): vals.append((e[:, i] * e[:, j]).sum(1)); q += 1\n        pair = (torch.stack(vals, 1) * w).sum(1)\n        return self.bias + self.lin(x).sum((1, 2)) + pair\n\n\ndef temporal(npz):\n    hm = np.asarray(npz['hourmin']).astype(np.int64)\n    hour = np.where(hm >= 100, hm // 100, hm // 60) % 24\n    raw = np.asarray(npz['date']).astype(np.int64)\n    uniq = {v: i % 7 for i, v in enumerate(sorted(np.unique(raw).tolist()))}\n    day = np.asarray([uniq[int(v)] for v in raw], np.int64)\n    return hour.astype(np.int64), day\n\n\ndef main():\n    ap = argparse.ArgumentParser(); ap.add_argument('--data-dir', required=True); ap.add_argument('--out-dir', required=True); ap.add_argument('--seed', type=int, default=137); ap.add_argument('--epochs', type=int, default=15); a = ap.parse_args()\n    smoke = os.environ.get('SMOKE_EPOCHS'); epochs = min(a.epochs, int(smoke)) if smoke else a.epochs\n    np.random.seed(a.seed); torch.manual_seed(a.seed)\n    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)\n    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False\n    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n    tr = np.load(os.path.join(a.data_dir, 'train.npz')); va = np.load(os.path.join(a.data_dir, 'val.npz'))\n    xt = torch.from_numpy(tr['X'].astype(np.int64)); yt = torch.from_numpy(tr['y'].astype(np.float32)); xv = torch.from_numpy(va['X'].astype(np.int64))\n    ht, dt = temporal(tr); hv, dv = temporal(va); ht = torch.from_numpy(ht); dt = torch.from_numpy(dt); hv = torch.from_numpy(hv); dv = torch.from_numpy(dv)\n    users = va['user']; yv = va['y'].astype(np.int64); dims = tr['field_dims'].astype(np.int64); videos = va['X'][:, 1].astype(np.int64) - int(dims[0])\n    model = PairKernel(int(dims.sum())).to(dev); opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=4e-5); sched = torch.optim.lr_scheduler.StepLR(opt, 2, gamma=0.6); bce = torch.nn.BCEWithLogitsLoss(label_smoothing=0.015)\n    gen = torch.Generator().manual_seed(a.seed); best = -1.; best_scores = None; wait = 0; hist = []; bs = 8192\n    for ep in range(epochs):\n        model.train(); perm = torch.randperm(len(yt), generator=gen); last = 0.\n        for p in range(0, len(yt), bs):\n            ii = perm[p:p+bs]; opt.zero_grad(set_to_none=True)\n            loss = bce(model(xt[ii].to(dev), ht[ii].to(dev), dt[ii].to(dev)), yt[ii].to(dev)); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); last = float(loss.detach().cpu())\n        sched.step(); model.eval(); parts = []\n        with torch.no_grad():\n            for p in range(0, len(xv), 65536): parts.append(model(xv[p:p+65536].to(dev), hv[p:p+65536].to(dev), dv[p:p+65536].to(dev)).cpu().numpy())\n        scores = np.concatenate(parts); m = evaluate(users, yv, scores); primary = float(m['primary']); hist.append({'epoch': ep+1, 'train_loss': round(last, 6), 'val_primary': round(primary, 6)})\n        if primary > best + 1e-6: best, best_scores, wait = primary, scores.copy(), 0\n        else:\n            wait += 1\n            if wait >= 3: break\n    os.makedirs(a.out_dir, exist_ok=True); m = evaluate(users, yv, best_scores)\n    with open(os.path.join(a.out_dir, 'metrics.json'), 'w') as f: json.dump({'gauc': m.get('GAUC', m.get('gauc')), 'ndcg5': m.get('nDCG@5', m.get('ndcg5')), 'primary': m['primary'], 'history': hist}, f)\n    with open(os.path.join(a.out_dir, 'predictions.csv'), 'w') as f:\n        f.write('row_id,user_id,video_id,score\\n')\n        for i, s in enumerate(best_scores): f.write(f'{i},{users[i]},{videos[i]},{s:.9g}\\n')\n\n\nif __name__ == '__main__': main()\n","config":{"epochs":15},"seed":137},{"family":"causal-sequence-deepfm-composite","code":"import argparse, json, os, sys\nfrom collections import defaultdict, deque\nimport numpy as np\nimport torch\n\nsys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\nfrom data.official.evaluate import evaluate\n\n\ndef histories(xtr, xval, length, pad):\n    state = defaultdict(lambda: deque(maxlen=length))\n    ht = np.full((len(xtr), length), pad, np.int64)\n    for i, (u, v) in enumerate(zip(xtr[:, 0], xtr[:, 1])):\n        h = list(state[int(u)])\n        if h: ht[i, -len(h):] = h\n        state[int(u)].append(int(v))\n    hv = np.full((len(xval), length), pad, np.int64)\n    for i, u in enumerate(xval[:, 0]):\n        h = list(state[int(u)])\n        if h: hv[i, -len(h):] = h\n    return ht, hv\n\n\nclass SeqDeepFM(torch.nn.Module):\n    def __init__(self, total, pad, k=16):\n        super().__init__(); self.pad = pad\n        self.emb = torch.nn.Embedding(total + 1, k, padding_idx=pad); self.lin = torch.nn.Embedding(total, 1)\n        self.att = torch.nn.Linear(2*k, 1)\n        self.deep = torch.nn.Sequential(torch.nn.Linear(6*k, 128), torch.nn.ReLU(), torch.nn.Dropout(0.24), torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Dropout(0.14), torch.nn.Linear(64, 1))\n        self.gate = torch.nn.Sequential(torch.nn.Linear(2*k, k), torch.nn.Sigmoid())\n        self.bias = torch.nn.Parameter(torch.zeros(1)); torch.nn.init.normal_(self.emb.weight, std=0.01); torch.nn.init.zeros_(self.lin.weight)\n    def forward(self, x, hist):\n        e = self.emb(x); he = self.emb(hist); q = e[:, 1].unsqueeze(1).expand_as(he)\n        logits = self.att(torch.cat((q, he), 2)).squeeze(2).masked_fill(hist.eq(self.pad), -1e4)\n        alpha = torch.softmax(logits, 1); valid = hist.ne(self.pad).any(1, keepdim=True).float(); pooled = (alpha.unsqueeze(2) * he).sum(1) * valid\n        gated = pooled * self.gate(torch.cat((e[:, 1], pooled), 1))\n        s = e.sum(1); fm = 0.5 * (s.square() - e.square().sum(1)).sum(1)\n        return self.bias + self.lin(x).sum((1, 2)) + fm + self.deep(torch.cat((e.flatten(1), gated), 1)).squeeze(1)\n\n\ndef main():\n    ap = argparse.ArgumentParser(); ap.add_argument('--data-dir', required=True); ap.add_argument('--out-dir', required=True); ap.add_argument('--seed', type=int, default=271); ap.add_argument('--epochs', type=int, default=13); a = ap.parse_args()\n    smoke = os.environ.get('SMOKE_EPOCHS'); epochs = min(a.epochs, int(smoke)) if smoke else a.epochs\n    np.random.seed(a.seed); torch.manual_seed(a.seed)\n    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)\n    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False\n    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n    tr = np.load(os.path.join(a.data_dir, 'train.npz')); va = np.load(os.path.join(a.data_dir, 'val.npz'))\n    xtr = tr['X'].astype(np.int64); xval = va['X'].astype(np.int64); dims = tr['field_dims'].astype(np.int64); total = int(dims.sum()); pad = total\n    htr, hval = histories(xtr, xval, 5, pad)\n    xt = torch.from_numpy(xtr); yt = torch.from_numpy(tr['y'].astype(np.float32)); xv = torch.from_numpy(xval); ht = torch.from_numpy(htr); hv = torch.from_numpy(hval)\n    users = va['user']; yv = va['y'].astype(np.int64); videos = xval[:, 1] - int(dims[0])\n    model = SeqDeepFM(total, pad).to(dev); opt = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=4e-5); sched = torch.optim.lr_scheduler.StepLR(opt, 2, gamma=0.58); bce = torch.nn.BCEWithLogitsLoss(label_smoothing=0.02)\n    gen = torch.Generator().manual_seed(a.seed); best = -1.; best_scores = None; wait = 0; histlog = []; bs = 6144\n    for ep in range(epochs):\n        model.train(); perm = torch.randperm(len(yt), generator=gen); last = 0.\n        for p in range(0, len(yt), bs):\n            ii = perm[p:p+bs]; opt.zero_grad(set_to_none=True); loss = bce(model(xt[ii].to(dev), ht[ii].to(dev)), yt[ii].to(dev)); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); last = float(loss.detach().cpu())\n        sched.step(); model.eval(); parts = []\n        with torch.no_grad():\n            for p in range(0, len(xv), 32768): parts.append(model(xv[p:p+32768].to(dev), hv[p:p+32768].to(dev)).cpu().numpy())\n        scores = np.concatenate(parts); m = evaluate(users, yv, scores); primary = float(m['primary']); histlog.append({'epoch': ep+1, 'train_loss': round(last, 6), 'val_primary': round(primary, 6)})\n        if primary > best + 1e-6: best, best_scores, wait = primary, scores.copy(), 0\n        else:\n            wait += 1\n            if wait >= 3: break\n    os.makedirs(a.out_dir, exist_ok=True); m = evaluate(users, yv, best_scores)\n    with open(os.path.join(a.out_dir, 'metrics.json'), 'w') as f: json.dump({'gauc': m.get('GAUC', m.get('gauc')), 'ndcg5': m.get('nDCG@5', m.get('ndcg5')), 'primary': m['primary'], 'history': histlog}, f)\n    with open(os.path.join(a.out_dir, 'predictions.csv'), 'w') as f:\n        f.write('row_id,user_id,video_id,score\\n')\n        for i, s in enumerate(best_scores): f.write(f'{i},{users[i]},{videos[i]},{s:.9g}\\n')\n\n\nif __name__ == '__main__': main()\n","config":{"epochs":13},"seed":271},{"family":"recency-weighted-hybrid-fm","code":"import argparse, json, math, os, sys\nfrom collections import defaultdict\nimport numpy as np\nimport torch\n\nsys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\nfrom data.official.evaluate import evaluate\n\n\nclass FM(torch.nn.Module):\n    def __init__(self, total, k=16):\n        super().__init__(); self.emb = torch.nn.Embedding(total, k); self.lin = torch.nn.Embedding(total, 1); self.bias = torch.nn.Parameter(torch.zeros(1)); self.drop = torch.nn.Dropout(0.12)\n        torch.nn.init.normal_(self.emb.weight, std=0.01); torch.nn.init.zeros_(self.lin.weight)\n    def forward(self, x):\n        e = self.drop(self.emb(x)); s = e.sum(1); pair = 0.5 * (s.square() - e.square().sum(1)).sum(1)\n        return self.bias + self.lin(x).sum((1, 2)) + pair\n\n\ndef recency(date):\n    d = np.asarray(date).astype(np.int64); uniq = sorted(np.unique(d).tolist()); rank = {v: i for i, v in enumerate(uniq)}; age = np.asarray([len(uniq)-1-rank[int(v)] for v in d], np.float32)\n    w = np.exp(-math.log(2.0) * age / 7.0); return (w / w.mean()).astype(np.float32)\n\n\ndef make_pairs(users, labels, seed):\n    groups = defaultdict(lambda: [[], []])\n    for i, (u, y) in enumerate(zip(users, labels.astype(np.int64))): groups[int(u)][int(y)].append(i)\n    rng = np.random.RandomState(seed); pos = []; neg = []\n    for p, n in groups.values():\n        if p and n:\n            pp = np.asarray(p, np.int64); nn = np.asarray(n, np.int64); pos.extend(pp.tolist()); neg.extend(nn[rng.randint(0, len(nn), len(pp))].tolist())\n    return np.asarray(pos, np.int64), np.asarray(neg, np.int64)\n\n\ndef main():\n    ap = argparse.ArgumentParser(); ap.add_argument('--data-dir', required=True); ap.add_argument('--out-dir', required=True); ap.add_argument('--seed', type=int, default=911); ap.add_argument('--epochs', type=int, default=15); a = ap.parse_args()\n    smoke = os.environ.get('SMOKE_EPOCHS'); epochs = min(a.epochs, int(smoke)) if smoke else a.epochs\n    np.random.seed(a.seed); torch.manual_seed(a.seed)\n    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)\n    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False\n    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n    tr = np.load(os.path.join(a.data_dir, 'train.npz')); va = np.load(os.path.join(a.data_dir, 'val.npz'))\n    xnp = tr['X'].astype(np.int64); ynp = tr['y'].astype(np.float32); xt = torch.from_numpy(xnp); yt = torch.from_numpy(ynp); xv = torch.from_numpy(va['X'].astype(np.int64)); wt = torch.from_numpy(recency(tr['date']))\n    pi, ni = make_pairs(tr['user'], ynp, a.seed); pi = torch.from_numpy(pi); ni = torch.from_numpy(ni)\n    dims = tr['field_dims'].astype(np.int64); users = va['user']; yv = va['y'].astype(np.int64); videos = va['X'][:, 1].astype(np.int64) - int(dims[0])\n    model = FM(int(dims.sum())).to(dev); opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=5e-5); sched = torch.optim.lr_scheduler.StepLR(opt, 2, gamma=0.6)\n    gen = torch.Generator().manual_seed(a.seed); best = -1.; best_scores = None; wait = 0; hist = []; bs = 8192\n    for ep in range(epochs):\n        model.train(); perm = torch.randperm(len(yt), generator=gen); pperm = torch.randperm(len(pi), generator=gen) if len(pi) else None; last = 0.\n        for b, st in enumerate(range(0, len(yt), bs)):\n            ii = perm[st:st+bs]; xb = xt[ii].to(dev); yb = yt[ii].to(dev); wb = wt[ii].to(dev); logits = model(xb)\n            bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, yb, reduction='none'); loss_bce = (bce * wb).sum() / wb.sum().clamp_min(1.)\n            if len(pi):\n                q = (b * bs) % len(pi); jj = pperm[q:min(q+bs, len(pi))]\n                if len(jj) == 0: jj = pperm[:min(bs, len(pi))]\n                sp = model(xt[pi[jj]].to(dev)); sn = model(xt[ni[jj]].to(dev)); loss_pair = torch.nn.functional.softplus(-(sp-sn)).mean(); loss = 0.5 * loss_bce + 0.5 * loss_pair\n            else: loss = loss_bce\n            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); last = float(loss.detach().cpu())\n        sched.step(); model.eval(); parts = []\n        with torch.no_grad():\n            for p in range(0, len(xv), 65536): parts.append(model(xv[p:p+65536].to(dev)).cpu().numpy())\n        scores = np.concatenate(parts); m = evaluate(users, yv, scores); primary = float(m['primary']); hist.append({'epoch': ep+1, 'train_loss': round(last, 6), 'val_primary': round(primary, 6)})\n        if primary > best + 1e-6: best, best_scores, wait = primary, scores.copy(), 0\n        else:\n            wait += 1\n            if wait >= 3: break\n    os.makedirs(a.out_dir, exist_ok=True); m = evaluate(users, yv, best_scores)\n    with open(os.path.join(a.out_dir, 'metrics.json'), 'w') as f: json.dump({'gauc': m.get('GAUC', m.get('gauc')), 'ndcg5': m.get('nDCG@5', m.get('ndcg5')), 'primary': m['primary'], 'history': hist}, f)\n    with open(os.path.join(a.out_dir, 'predictions.csv'), 'w') as f:\n        f.write('row_id,user_id,video_id,score\\n')\n        for i, s in enumerate(best_scores): f.write(f'{i},{users[i]},{videos[i]},{s:.9g}\\n')\n\n\nif __name__ == '__main__': main()\n","config":{"epochs":15},"seed":911}],"blend":{"weights":"equal","aggregations":[{"method":"rank_average","scope":"per_user"},{"method":"rank_average","scope":"global"}]}}}
