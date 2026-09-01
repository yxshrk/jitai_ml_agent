# role: proposer | model: gpt-5.6-sol

## SYSTEM
You are an autonomous ML research agent improving a short-video recommender.

Task: predict long_view (binary) per impression; ranking quality is scored with
"primary" = mean of within-user GAUC and per-user nDCG@5, computed by the official
evaluator `harness.evaluate_provisional.evaluate(user_ids, labels, scores)`.
Higher is better. Improvements below 0.002 on validation are noise.

Hard rules for every script you emit (CONTRACTS.md section 3):
- DRAFT and DEBUG proposals emit ONE WHOLE runnable Python script (never a
  fragment). IMPROVE proposals instead emit "edits" blocks per the envelope
  contract below — the harness applies them to the parent script.
- CLI: `python <script> --data-dir <d> --out-dir <o> [--seed 42]` via argparse.
  Default seed 42. Deterministic given the seed.
- FAST PATH (use it when present): `<data-dir>/train.npz` and `<data-dir>/val.npz` hold
  pre-encoded arrays — X (int32, 5 offset-encoded fields: user,video,author,tab,dur_bucket),
  y (long_view float32), user, click, play_time_ms, duration_ms, hourmin, date, field_dims.
  Loading them takes ~1s vs ~90s of CSV parsing; training-time budget is scored, so prefer npz.
  OFFSET ENCODING INVARIANT: X holds one global index space — each field's ids are
  already shifted by the sum of the previous fields' cardinalities, so the correct
  embedding table has exactly field_dims.sum() rows and X is indexed into it
  directly. Per-field tables sized by field_dims[i] MUST subtract the field's
  offset first, or validation ids overflow (IndexError: index out of range).
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
- Stay within the runtime timeout (the NODE_TIMEOUT_S environment variable,
  set by the harness); prefer small/fast models.
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
For an IMPROVE proposal, do NOT re-emit the whole script. Replace "code" with
targeted edit blocks applied verbatim to the parent script:
 "edits": [{"search": "<exact contiguous snippet copied character-for-character
from the parent script; must occur exactly once>",
            "replace": "<the replacement text>"}, ...]
Edits are applied in order; untouched code stays byte-identical, so the parent's
debugged trainer, data pipeline, and evaluation scaffolding survive unchanged.
Copy search text EXACTLY (whitespace included) from the parent script you were
given. Use several small blocks rather than one giant block.


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
1. RECALIBRATED (1 Sep): the 0.6058-0.6065 close evidence uses POLISHED artifacts;
   in-node fresh members measured 0.5975-0.6046 and every fresh-member close (f7-f9)
   fell back to its incumbent. Closes pay only over EXISTING strong artifacts (own
   trained nodes via script_source, or the champion verbatim via seed ensemble).
   The proven opener is the 48-probe swept package (best single: 0.605102, run_f9).
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

12. **Seed ensemble**: average predictions over 3-5 seeds of the best config, at the
    very end. MEASURED BASE-DEPENDENT (1 Sep): +0.0014 off a 0.6042 single; only
    +0.0002 off a heavily tuned 0.6051 single — the gain shrinks near the
    0.6055-0.6060 ceiling. (E7: 5-seed of best.py 0.6047 vs seed-mean 0.6039.)

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
node_001 [<-node_000] draft "The epoch-8 validation peak followed by decline diagnoses overfitting, and a two-stage dial search over the complete DCN-lite, 0.5-BPR hybrid, dropout/AdamW step-decay, and recency-weighting package will raise validation primary by about 0.0029 versus node_000." primary=0.6039 ACCEPTED
node_002 [<-node_001] draft "The validation curve peaks near 0.6039 and then declines, diagnosing overfit; keeping the accepted regularized DCN-lite package unchanged while drawing 30% of BPR negatives from the positive's same user-day-hour or user-day-tab context will improve validation primary by about 0.0012 by making the pairwise objective better match contemporaneous within-user ranking." primary=0.6042 REJECTED

Mode: DRAFT. Write a fresh script derived from the parent (the current baseline family) implementing the SELECTED card (or, when a launch directive names a tier, that directive). Keep everything else identical to the parent.

## Selected method (implement THIS)
### gauge-fixed-bce: User-centered BCE (within-user gauge fixing)  [MEASURED WIN]
- mechanism: Batch complete user slates; replace the pointwise BCE logits with user-centered logits (logit minus that user's batch-mean logit, plus one learned global bias). Gradient of the pointwise term then sums to zero within each user, so it can only learn relative deviations — the only thing GAUC/nDCG measure. Keep the BPR term unchanged.
- reference_primary: none
- treats: metric-mismatch
- citation: gauge-fixing rationale (per-user constant shifts leave the metric invariant); pairwise-consistency literature.
- expected_gain / cost: MEASURED +0.0026 over a 0.6018-family parent (run_novel_r1
  node_003, accepted 0.60447); very cheap / low.
- status_pure: measured-win (novel_r1)
- status_1k: untried
selector diagnosis: overfit
selector why: Validation peaks at 0.60390 and then declines to roughly 0.60246, indicating overfit. However, overfit-family methods are portfolio-excluded, and the accepted package already contains strong regularization. Gauge-fixed BCE is the strongest eligible unapplied method: it preserves the existing BPR term while removing metric-irrelevant within-user logit shifts, has a measured +0.0026 gain on Pure, and therefore has credible epsilon-clearing upside from the current 0.6039 best.

## Convergence pressure
streak_state = {'no_improve_streak': 1, 'n_converge': 3, 'iters_left': 13}
The run ends after N consecutive iterations whose best-so-far improvement is <= epsilon = 0.002. Select experiments by expected scientific value given the remaining budget: at every iteration, including the first, prefer the eligible move with the largest evidence-supported expected gain for its cost; an early iteration spent on a small-ceiling treatment is a convergence strike bought at full price. Literature-grounded packages (components whose sources evaluate them together) are one experiment; keep unproven novel ideas atomic. Plan the run so its final iterations produce the strongest possible finished artifact rather than leaving the run un-finalized. Do the epsilon arithmetic before choosing: if the streak means the run ends unless THIS iteration improves best-so-far by at least epsilon, then a move whose own evidence caps its gain below epsilon cannot extend the run no matter how proven it is; on such an iteration prefer the eligible move with the largest evidence-supported expected gain at or above epsilon, and among qualifying moves prefer the one whose evidence clears epsilon with the widest margin: a move whose evidence only just reaches the bar fails it about half the time, so bare arithmetic reach is not parity with a wide-margin alternative (combining decorrelated mechanism families generally out-gains both re-seeding one family and any single atomic mechanism). Implementation-dead is not evidence-dead: two failed BUILDS of a card mean pivot to a mechanically simpler card, not a third build attempt. Read margins against the CURRENT best, not a card's original baseline: an unspent package whose measured absolute score sits near the current best offers almost no headroom, while a close whose evidence exceeds every single-model score in the ledger offers the most. A proven small-gain close is the right pick only when no eligible move has evidence reaching epsilon. BANK THE LAST GAIN: when the streak is one short of ending the run and NO eligible move's evidence reaches epsilon, the run is in its final iteration either way — the deliverable is best-so-far, not the streak. Choose the move with the most RELIABLE positive expected gain, where reliable means MEASURED on this benchmark (replicated numbers, measured-win statuses) — an estimated or literature range is not bankable. Two tests: (1) prefer a measured small gain over a larger estimate; (2) if THIS run's journal shows fresh-code members just measured far below the champion, every close that requires writing new members inherits that measured weakness — only a close that reuses the champion verbatim (seed ensemble: retrain the SAME accepted script at new seeds, measured +0.0002..+0.0014 — larger the less tuned the single is) keeps its evidence intact. A banked +0.0005 beats a failed +0.002 attempt. If a close was just REJECTED for a gain that did not repeat, its members were too close to the incumbent: re-rolling the same blend with new seeds is not a new experiment; the bottleneck a failed confirm reveals is MEMBER DIVERSITY, so the next node must add a NEW MECHANISM FAMILY the ledger has not yet given the blend (a measured package from another family), then close again. A dosage or regularization treatment on the existing champion does not qualify even if untried: it cannot decorrelate the next blend because it adds no new family. Strengthening means a NEW mechanism or family member: a component the champion stack ALREADY CONTAINS (check its accepted lineage) is not a strengthener, and re-applying it is a no-op, not a treatment. Do not change what counts as an iteration in response to the streak.

## Runtime budget (overrides the 600s default above)
THIS run's per-node timeout is 7200 seconds (~120 minutes). A full-length training on the npz fast path costs roughly 40-90s on CPU, far less on GPU. Plan to SPEND ~60-70% of this budget on search probes when playing a search card — e.g. at 2+ hours that is 40+ full-length probes plus refinement, not 8. Reserve the remainder for the final training(s). Finishing a search node in a small fraction of the budget is a defect, not efficiency: unspent budget is free score variance left unexplored.

Directive: draft from Tier 3 of the menu

When implementing ANY ensemble/member card: each member MUST be trained with a distinct seed; after scoring, ASSERT member score vectors are not identical (numpy allclose check between members and against the parent predictions) and print per-member validation primaries to progress output. An ensemble whose final predictions equal the parent's is a no-op and will be rejected by the harness, except when the farm-close executor explicitly selects and records the incumbent fallback.

## Parent node "node_001" (full code)
```python
"""Two-stage dial search for a regularized DCN-lite/BPR/recency package.

Uses only the official five offset-encoded fields from the npz fast path. Coarse
probes locate a regularization basin, longer full-row refinement selects the
configuration, and one full-length run checkpoints validation every half epoch.
"""
import argparse
import datetime
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.official.evaluate import evaluate


class DCNLite(torch.nn.Module):
    def __init__(self, total_dim, fields=5, k=16, hidden=96, cross_layers=1,
                 dropout=0.25):
        super().__init__()
        self.fields = fields
        self.k = k
        self.dropout = float(dropout)
        width = fields * k
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.cross_w = torch.nn.ParameterList([
            torch.nn.Parameter(torch.empty(width)) for _ in range(cross_layers)
        ])
        self.cross_b = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros(width)) for _ in range(cross_layers)
        ])
        self.deep = torch.nn.Sequential(
            torch.nn.Linear(width, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(hidden, hidden // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
        )
        self.out = torch.nn.Linear(width + hidden // 2, 1)
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)
        for w in self.cross_w:
            torch.nn.init.normal_(w, std=0.01)
        torch.nn.init.normal_(self.out.weight, std=0.01)
        torch.nn.init.zeros_(self.out.bias)

    def forward(self, x):
        raw = self.emb(x)
        e = F.dropout(raw, p=self.dropout, training=self.training)
        summed = e.sum(1)
        fm = 0.5 * (summed.square() - e.square().sum(1)).sum(1)
        linear = self.lin(x).sum((1, 2))
        x0 = e.reshape(e.shape[0], -1)
        xl = x0
        for w, b in zip(self.cross_w, self.cross_b):
            xl = x0 * (xl * w).sum(1, keepdim=True) + b + xl
        deep = self.deep(x0)
        nonlinear = self.out(torch.cat((xl, deep), dim=1)).squeeze(1)
        return self.bias + linear + fm + nonlinear


def seed_everything(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def date_ages(values):
    vals = np.asarray(values)
    unique = np.unique(vals)
    ordinal = {}
    try:
        for value in unique:
            text = str(int(value)).zfill(8)
            ordinal[value] = datetime.date(int(text[:4]), int(text[4:6]),
                                           int(text[6:8])).toordinal()
    except (TypeError, ValueError):
        ordinal = {value: rank for rank, value in enumerate(sorted(unique.tolist()))}
    newest = max(ordinal.values())
    return np.asarray([newest - ordinal[value] for value in vals], dtype=np.float32)


def build_pair_tables(users, labels):
    users = np.asarray(users)
    labels = np.asarray(labels) >= 0.5
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    cuts = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
    bounds = np.concatenate(([0], cuts, [len(order)]))
    positives = []
    negative_chunks = []
    negative_starts = []
    negative_counts = []
    cursor = 0
    for left, right in zip(bounds[:-1], bounds[1:]):
        idx = order[left:right]
        pos = idx[labels[idx]]
        neg = idx[~labels[idx]]
        if len(pos) and len(neg):
            positives.append(pos.astype(np.int64, copy=False))
            negative_chunks.append(neg.astype(np.int64, copy=False))
            negative_starts.append(np.full(len(pos), cursor, dtype=np.int64))
            negative_counts.append(np.full(len(pos), len(neg), dtype=np.int64))
            cursor += len(neg)
    if not positives:
        return (np.empty(0, dtype=np.int64),) * 4
    return (np.concatenate(positives), np.concatenate(negative_chunks),
            np.concatenate(negative_starts), np.concatenate(negative_counts))


def metric_values(metric):
    return {
        "gauc": float(metric.get("GAUC", metric.get("gauc", 0.0))),
        "ndcg5": float(metric.get("nDCG@5", metric.get("ndcg5", 0.0))),
        "primary": float(metric["primary"]),
    }


def make_coarse_configs(seed):
    rng = np.random.default_rng(seed + 1701)
    count = 12
    dropouts = np.linspace(0.17, 0.39, count)[rng.permutation(count)]
    decays = np.geomspace(4.0e-5, 2.4e-3, count)[rng.permutation(count)]
    lrs = np.geomspace(4.8e-4, 1.35e-3, count)[rng.permutation(count)]
    gammas = np.linspace(0.36, 0.76, count)[rng.permutation(count)]
    half_lives = np.asarray([3.5, 7.0, 14.0] * 4)[rng.permutation(count)]
    steps = np.asarray([1, 2, 3, 2] * 3)[rng.permutation(count)]
    hidden = np.asarray([64, 96, 128, 96] * 3)[rng.permutation(count)]
    crosses = np.asarray([1, 1, 2, 1, 2, 1] * 2)[rng.permutation(count)]
    configs = []
    for i in range(count):
        configs.append({
            "dropout": float(dropouts[i]),
            "weight_decay": float(decays[i]),
            "lr": float(lrs[i]),
            "decay_gamma": float(gammas[i]),
            "decay_step": int(steps[i]),
            "half_life": float(half_lives[i]),
            "hidden": int(hidden[i]),
            "cross_layers": int(crosses[i]),
            "bpr_mix": 0.5,
        })
    return configs


def make_refine_configs(base, seed):
    rng = np.random.default_rng(seed + 2903)
    configs = [dict(base)]
    hidden_choices = np.asarray([64, 80, 96, 112, 128])
    for _ in range(5):
        cfg = dict(base)
        cfg["dropout"] = float(np.clip(
            base["dropout"] + rng.normal(0.0, 0.035), 0.13, 0.43))
        cfg["weight_decay"] = float(np.clip(
            base["weight_decay"] * math.exp(rng.normal(0.0, 0.42)),
            2.5e-5, 3.2e-3))
        cfg["lr"] = float(np.clip(
            base["lr"] * math.exp(rng.normal(0.0, 0.18)), 3.5e-4, 1.7e-3))
        cfg["decay_gamma"] = float(np.clip(
            base["decay_gamma"] + rng.normal(0.0, 0.075), 0.28, 0.84))
        cfg["decay_step"] = int(np.clip(
            base["decay_step"] + rng.choice([-1, 0, 1]), 1, 4))
        cfg["half_life"] = float(np.clip(
            base["half_life"] * math.exp(rng.normal(0.0, 0.24)), 2.8, 18.0))
        nearest = int(np.argmin(np.abs(hidden_choices - base["hidden"])))
        shift = int(rng.choice([-1, 0, 1]))
        cfg["hidden"] = int(hidden_choices[np.clip(nearest + shift, 0,
                                                   len(hidden_choices) - 1)])
        cfg["cross_layers"] = int(np.clip(
            base["cross_layers"] + rng.choice([-1, 0, 1]), 1, 2))
        configs.append(cfg)
    return configs


def append_progress(path, record):
    with open(path, "a") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=14)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    progress_path = os.path.join(args.out_dir, "progress.log")
    if os.path.exists(progress_path):
        os.remove(progress_path)

    train_path = os.path.join(args.data_dir, "train.npz")
    val_path = os.path.join(args.data_dir, "val.npz")
    tr = np.load(train_path)
    va = np.load(val_path)

    field_dims = tr["field_dims"].astype(np.int64)
    total_dim = int(field_dims.sum())
    xt_np = tr["X"].astype(np.int64)
    yt_np = tr["y"].astype(np.float32)
    xv_np = va["X"].astype(np.int64)
    val_users = va["user"]
    val_labels = va["y"].astype(int)
    ages_np = date_ages(tr["date"])

    pair_pos_np, pair_neg_np, pair_start_np, pair_count_np = build_pair_tables(
        tr["user"], yt_np)

    Xt = torch.from_numpy(xt_np).to(device)
    yt = torch.from_numpy(yt_np).to(device)
    Xv = torch.from_numpy(xv_np).to(device)
    ages = torch.from_numpy(ages_np).to(device)
    pair_pos = torch.from_numpy(pair_pos_np).to(device)
    pair_neg = torch.from_numpy(pair_neg_np).to(device)
    pair_start = torch.from_numpy(pair_start_np).to(device)
    pair_count = torch.from_numpy(pair_count_np).to(device)

    smoke = os.environ.get("SMOKE_EPOCHS")
    smoke_cap = int(smoke) if smoke is not None else None
    coarse_epochs = min(3, smoke_cap) if smoke_cap is not None else 3
    refine_epochs = min(6, smoke_cap) if smoke_cap is not None else 6
    final_epochs = min(args.epochs, smoke_cap) if smoke_cap is not None else args.epochs
    coarse_epochs = max(1, coarse_epochs)
    refine_epochs = max(1, refine_epochs)
    final_epochs = max(1, final_epochs)

    n = len(yt_np)
    batch_size = 8192
    history = []

    def predict(model):
        model.eval()
        chunks = []
        with torch.no_grad():
            for left in range(0, len(Xv), 65536):
                chunks.append(model(Xv[left:left + 65536]).detach().cpu().numpy())
        return np.concatenate(chunks)

    def train_candidate(config, epochs, row_fraction, run_seed, stage,
                        probe_index, half_epoch_checks=False, keep_snapshot=False):
        seed_everything(run_seed)
        model = DCNLite(
            total_dim=total_dim,
            fields=xt_np.shape[1],
            k=16,
            hidden=int(config["hidden"]),
            cross_layers=int(config["cross_layers"]),
            dropout=float(config["dropout"]),
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(config["lr"]),
            weight_decay=float(config["weight_decay"]))
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(config["decay_step"]),
            gamma=float(config["decay_gamma"]))

        recency = torch.pow(torch.tensor(2.0, device=device),
                            -ages / float(config["half_life"]))
        recency = recency / recency.mean().clamp_min(1e-8)
        take = max(batch_size, min(n, int(round(n * row_fraction))))
        total_batches = int(math.ceil(take / batch_size))
        best_primary = -1.0
        best_scores = None
        best_metric = None
        best_event = 0.0
        best_state = None
        events = []

        for epoch in range(epochs):
            model.train()
            permutation = torch.randperm(n, device=device)[:take]
            midpoint = max(1, int(math.ceil(total_batches / 2.0)))
            checkpoints = {total_batches}
            if half_epoch_checks:
                checkpoints.add(midpoint)
            loss_sum = 0.0
            seen_batches = 0
            for batch_number, left in enumerate(range(0, take, batch_size), start=1):
                idx = permutation[left:left + batch_size]
                optimizer.zero_grad(set_to_none=True)
                logits = model(Xt[idx])
                point_loss = F.binary_cross_entropy_with_logits(
                    logits, yt[idx], reduction="none")
                bce_loss = (point_loss * recency[idx]).sum() / recency[idx].sum().clamp_min(1e-8)

                if len(pair_pos):
                    selected = torch.randint(len(pair_pos), (len(idx),), device=device)
                    pos_idx = pair_pos[selected]
                    counts = pair_count[selected]
                    offsets = torch.floor(torch.rand(len(idx), device=device) *
                                          counts.to(torch.float32)).to(torch.long)
                    neg_idx = pair_neg[pair_start[selected] + offsets]
                    pair_logits = model(torch.cat((Xt[pos_idx], Xt[neg_idx]), dim=0))
                    pos_logits = pair_logits[:len(idx)]
                    neg_logits = pair_logits[len(idx):]
                    pair_weights = 0.5 * (recency[pos_idx] + recency[neg_idx])
                    pair_loss = (F.softplus(-(pos_logits - neg_logits)) * pair_weights).sum()
                    pair_loss = pair_loss / pair_weights.sum().clamp_min(1e-8)
                    mix = float(config["bpr_mix"])
                    loss = (1.0 - mix) * bce_loss + mix * pair_loss
                else:
                    loss = bce_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                loss_sum += float(loss.detach().item())
                seen_batches += 1

                if batch_number in checkpoints:
                    scores = predict(model)
                    metric = metric_values(evaluate(val_users, val_labels, scores))
                    event_epoch = epoch + batch_number / total_batches
                    event = {
                        "checkpoint_epoch": round(float(event_epoch), 3),
                        "train_loss": round(loss_sum / max(1, seen_batches), 6),
                        "lr": float(optimizer.param_groups[0]["lr"]),
                        "gauc": metric["gauc"],
                        "ndcg5": metric["ndcg5"],
                        "primary": metric["primary"],
                    }
                    events.append(event)
                    if metric["primary"] > best_primary + 1e-9:
                        best_primary = metric["primary"]
                        best_scores = scores.copy()
                        best_metric = metric
                        best_event = event_epoch
                        if keep_snapshot:
                            best_state = {key: value.detach().cpu().clone()
                                          for key, value in model.state_dict().items()}
                    if batch_number != total_batches:
                        model.train()
            scheduler.step()

        record = {
            "stage": stage,
            "probe": int(probe_index),
            "seed": int(run_seed),
            "epochs": int(epochs),
            "row_fraction": float(row_fraction),
            "config": dict(config),
            "best_epoch": round(float(best_event), 3),
            "gauc": best_metric["gauc"],
            "ndcg5": best_metric["ndcg5"],
            "primary": best_metric["primary"],
            "checkpoints": events,
        }
        history.append(record)
        append_progress(progress_path, {
            "stage": stage,
            "probe": int(probe_index),
            "config": dict(config),
            "primary": best_metric["primary"],
        })
        del model, optimizer, scheduler, recency, best_state
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return best_primary, best_scores, best_metric, record

    coarse_configs = make_coarse_configs(args.seed)
    coarse_results = []
    for index, config in enumerate(coarse_configs):
        result = train_candidate(
            config=config,
            epochs=coarse_epochs,
            row_fraction=0.62,
            run_seed=args.seed + 101,
            stage="coarse",
            probe_index=index,
        )
        coarse_results.append((result[0], config))
    coarse_results.sort(key=lambda item: item[0], reverse=True)
    coarse_winner = dict(coarse_results[0][1])

    refine_configs = make_refine_configs(coarse_winner, args.seed)
    refine_results = []
    refine_seed = args.seed + 202
    for index, config in enumerate(refine_configs):
        result = train_candidate(
            config=config,
            epochs=refine_epochs,
            row_fraction=1.0,
            run_seed=refine_seed,
            stage="refine",
            probe_index=index,
        )
        refine_results.append((result[0], dict(config), result[3]))
    refine_results.sort(key=lambda item: item[0], reverse=True)
    winning_config = dict(refine_results[0][1])
    winning_refine_primary = float(refine_results[0][0])

    final_primary, final_scores, final_metric, final_record = train_candidate(
        config=winning_config,
        epochs=final_epochs,
        row_fraction=1.0,
        run_seed=refine_seed,
        stage="final",
        probe_index=0,
        half_epoch_checks=True,
        keep_snapshot=True,
    )

    metrics_payload = {
        "gauc": final_metric["gauc"],
        "ndcg5": final_metric["ndcg5"],
        "primary": final_metric["primary"],
        "winning_config": winning_config,
        "coarse_winner_primary": float(coarse_results[0][0]),
        "winning_refine_primary": winning_refine_primary,
        "final_best_epoch": final_record["best_epoch"],
        "history": history,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics_payload, fh)

    video_offset = int(field_dims[0])
    video_ids = xv_np[:, 1] - video_offset
    with open(os.path.join(args.out_dir, "predictions.csv"), "w") as fh:
        fh.write("row_id,user_id,video_id,score\n")
        for row_id, score in enumerate(final_scores):
            fh.write(f"{row_id},{val_users[row_id]},{video_ids[row_id]},{score:.8g}\n")


if __name__ == "__main__":
    main()

```

## Parent learning curve (per epoch)
epoch None: train_loss None, val_gauc 0.6684930014598695, val_primary 0.6020869002193918
epoch None: train_loss None, val_gauc 0.667487786911569, val_primary 0.6013700484051061
epoch None: train_loss None, val_gauc 0.6684546752315085, val_primary 0.6019407328935534
epoch None: train_loss None, val_gauc 0.6687214284821292, val_primary 0.6025079717384765
epoch None: train_loss None, val_gauc 0.6686283749921312, val_primary 0.602243355406644
epoch None: train_loss None, val_gauc 0.66740364408704, val_primary 0.601370817344846
epoch None: train_loss None, val_gauc 0.669056704516078, val_primary 0.6027727615091769
epoch None: train_loss None, val_gauc 0.6674796206499031, val_primary 0.6015560945276901
epoch None: train_loss None, val_gauc 0.6683867595760864, val_primary 0.6021435047386204
epoch None: train_loss None, val_gauc 0.6688703316891883, val_primary 0.6026107370738683
epoch None: train_loss None, val_gauc 0.6683997532579067, val_primary 0.6018889908069998
epoch None: train_loss None, val_gauc 0.668064553379541, val_primary 0.6018822015201382
epoch None: train_loss None, val_gauc 0.6703064027595645, val_primary 0.6038147643476448
epoch None: train_loss None, val_gauc 0.6706009409312783, val_primary 0.6038957259105826
epoch None: train_loss None, val_gauc 0.6695336223159876, val_primary 0.603297156591925
epoch None: train_loss None, val_gauc 0.6680813827387387, val_primary 0.6024592408256493
epoch None: train_loss None, val_gauc 0.6684748193381596, val_primary 0.6028007625250461
epoch None: train_loss None, val_gauc 0.6686923425740519, val_primary 0.6028874178564803
epoch None: train_loss None, val_gauc 0.6706009409312783, val_primary 0.6038957259105826
DIAGNOSE before proposing: val peaks early then falls = overfit (attack with regularization/schedules); val still rising at stop = underfit (train longer); flat = the idea itself adds no signal (change direction, not dosage). State your diagnosis in the hypothesis.

Respond with the JSON object only.

## REPLY
{"execution_kind":"script","hypothesis":"Validation peaks near 0.6039 and then declines, diagnosing overfit, but replacing pointwise BCE with user-centered gauge-fixed BCE on complete user slates while leaving the accepted BPR and regularization package unchanged will remove metric-irrelevant user-level shifts and improve validation primary by about 0.0022.","expected_delta":0.0022,"expected_delta_basis":"The selected gauge-fixed-bce card measured +0.0026 over the 0.6018-family parent in run_novel_r1 node_003, with negligible additional computational cost.","action":"draft","parent":"node_001","code":"\"\"\"Gauge-fixed BCE over complete user slates in the accepted DCN-lite package.\n\nUses only the official five offset-encoded fields from the npz fast path. Coarse\nprobes locate a regularization basin, longer full-row refinement selects the\nconfiguration, and one full-length run checkpoints validation every half epoch.\nThe pointwise term centers logits within each complete user slate and restores\none learned global bias; the pairwise BPR term is unchanged.\n\"\"\"\nimport argparse\nimport datetime\nimport json\nimport math\nimport os\nimport sys\n\nimport numpy as np\nimport torch\nimport torch.nn.functional as F\n\nsys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\nfrom data.official.evaluate import evaluate\n\n\nclass DCNLite(torch.nn.Module):\n    def __init__(self, total_dim, fields=5, k=16, hidden=96, cross_layers=1,\n                 dropout=0.25):\n        super().__init__()\n        self.fields = fields\n        self.k = k\n        self.dropout = float(dropout)\n        width = fields * k\n        self.emb = torch.nn.Embedding(total_dim, k)\n        self.lin = torch.nn.Embedding(total_dim, 1)\n        self.bias = torch.nn.Parameter(torch.zeros(1))\n        self.cross_w = torch.nn.ParameterList([\n            torch.nn.Parameter(torch.empty(width)) for _ in range(cross_layers)\n        ])\n        self.cross_b = torch.nn.ParameterList([\n            torch.nn.Parameter(torch.zeros(width)) for _ in range(cross_layers)\n        ])\n        self.deep = torch.nn.Sequential(\n            torch.nn.Linear(width, hidden),\n            torch.nn.ReLU(),\n            torch.nn.Dropout(self.dropout),\n            torch.nn.Linear(hidden, hidden // 2),\n            torch.nn.ReLU(),\n            torch.nn.Dropout(self.dropout),\n        )\n        self.out = torch.nn.Linear(width + hidden // 2, 1)\n        torch.nn.init.normal_(self.emb.weight, std=0.01)\n        torch.nn.init.zeros_(self.lin.weight)\n        for w in self.cross_w:\n            torch.nn.init.normal_(w, std=0.01)\n        torch.nn.init.normal_(self.out.weight, std=0.01)\n        torch.nn.init.zeros_(self.out.bias)\n\n    def forward(self, x):\n        raw = self.emb(x)\n        e = F.dropout(raw, p=self.dropout, training=self.training)\n        summed = e.sum(1)\n        fm = 0.5 * (summed.square() - e.square().sum(1)).sum(1)\n        linear = self.lin(x).sum((1, 2))\n        x0 = e.reshape(e.shape[0], -1)\n        xl = x0\n        for w, b in zip(self.cross_w, self.cross_b):\n            xl = x0 * (xl * w).sum(1, keepdim=True) + b + xl\n        deep = self.deep(x0)\n        nonlinear = self.out(torch.cat((xl, deep), dim=1)).squeeze(1)\n        return self.bias + linear + fm + nonlinear\n\n\ndef seed_everything(seed):\n    os.environ.setdefault(\"CUBLAS_WORKSPACE_CONFIG\", \":4096:8\")\n    np.random.seed(seed)\n    torch.manual_seed(seed)\n    if torch.cuda.is_available():\n        torch.cuda.manual_seed_all(seed)\n    if hasattr(torch.backends, \"cudnn\"):\n        torch.backends.cudnn.deterministic = True\n        torch.backends.cudnn.benchmark = False\n\n\ndef date_ages(values):\n    vals = np.asarray(values)\n    unique = np.unique(vals)\n    ordinal = {}\n    try:\n        for value in unique:\n            text = str(int(value)).zfill(8)\n            ordinal[value] = datetime.date(int(text[:4]), int(text[4:6]),\n                                           int(text[6:8])).toordinal()\n    except (TypeError, ValueError):\n        ordinal = {value: rank for rank, value in enumerate(sorted(unique.tolist()))}\n    newest = max(ordinal.values())\n    return np.asarray([newest - ordinal[value] for value in vals], dtype=np.float32)\n\n\ndef build_pair_tables(users, labels):\n    users = np.asarray(users)\n    labels = np.asarray(labels) >= 0.5\n    order = np.argsort(users, kind=\"stable\")\n    sorted_users = users[order]\n    cuts = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1\n    bounds = np.concatenate(([0], cuts, [len(order)]))\n    positives = []\n    negative_chunks = []\n    negative_starts = []\n    negative_counts = []\n    cursor = 0\n    for left, right in zip(bounds[:-1], bounds[1:]):\n        idx = order[left:right]\n        pos = idx[labels[idx]]\n        neg = idx[~labels[idx]]\n        if len(pos) and len(neg):\n            positives.append(pos.astype(np.int64, copy=False))\n            negative_chunks.append(neg.astype(np.int64, copy=False))\n            negative_starts.append(np.full(len(pos), cursor, dtype=np.int64))\n            negative_counts.append(np.full(len(pos), len(neg), dtype=np.int64))\n            cursor += len(neg)\n    if not positives:\n        return (np.empty(0, dtype=np.int64),) * 4\n    return (np.concatenate(positives), np.concatenate(negative_chunks),\n            np.concatenate(negative_starts), np.concatenate(negative_counts))\n\n\ndef build_user_slates(users):\n    users = np.asarray(users)\n    order = np.argsort(users, kind=\"stable\").astype(np.int64, copy=False)\n    sorted_users = users[order]\n    cuts = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1\n    bounds = np.concatenate(([0], cuts, [len(order)])).astype(np.int64)\n    return order, bounds\n\n\ndef gauge_center_logits(logits, users, global_bias):\n    _, inverse = torch.unique(users, sorted=False, return_inverse=True)\n    group_count = int(inverse.max().item()) + 1\n    sums = torch.zeros(group_count, dtype=logits.dtype, device=logits.device)\n    counts = torch.zeros(group_count, dtype=logits.dtype, device=logits.device)\n    sums.scatter_add_(0, inverse, logits)\n    counts.scatter_add_(0, inverse, torch.ones_like(logits))\n    means = sums / counts.clamp_min(1.0)\n    return logits - means[inverse] + global_bias.reshape(())\n\n\ndef metric_values(metric):\n    return {\n        \"gauc\": float(metric.get(\"GAUC\", metric.get(\"gauc\", 0.0))),\n        \"ndcg5\": float(metric.get(\"nDCG@5\", metric.get(\"ndcg5\", 0.0))),\n        \"primary\": float(metric[\"primary\"]),\n    }\n\n\ndef make_coarse_configs(seed):\n    rng = np.random.default_rng(seed + 1701)\n    count = 12\n    dropouts = np.linspace(0.17, 0.39, count)[rng.permutation(count)]\n    decays = np.geomspace(4.0e-5, 2.4e-3, count)[rng.permutation(count)]\n    lrs = np.geomspace(4.8e-4, 1.35e-3, count)[rng.permutation(count)]\n    gammas = np.linspace(0.36, 0.76, count)[rng.permutation(count)]\n    half_lives = np.asarray([3.5, 7.0, 14.0] * 4)[rng.permutation(count)]\n    steps = np.asarray([1, 2, 3, 2] * 3)[rng.permutation(count)]\n    hidden = np.asarray([64, 96, 128, 96] * 3)[rng.permutation(count)]\n    crosses = np.asarray([1, 1, 2, 1, 2, 1] * 2)[rng.permutation(count)]\n    configs = []\n    for i in range(count):\n        configs.append({\n            \"dropout\": float(dropouts[i]),\n            \"weight_decay\": float(decays[i]),\n            \"lr\": float(lrs[i]),\n            \"decay_gamma\": float(gammas[i]),\n            \"decay_step\": int(steps[i]),\n            \"half_life\": float(half_lives[i]),\n            \"hidden\": int(hidden[i]),\n            \"cross_layers\": int(crosses[i]),\n            \"bpr_mix\": 0.5,\n        })\n    return configs\n\n\ndef make_refine_configs(base, seed):\n    rng = np.random.default_rng(seed + 2903)\n    configs = [dict(base)]\n    hidden_choices = np.asarray([64, 80, 96, 112, 128])\n    for _ in range(5):\n        cfg = dict(base)\n        cfg[\"dropout\"] = float(np.clip(\n            base[\"dropout\"] + rng.normal(0.0, 0.035), 0.13, 0.43))\n        cfg[\"weight_decay\"] = float(np.clip(\n            base[\"weight_decay\"] * math.exp(rng.normal(0.0, 0.42)),\n            2.5e-5, 3.2e-3))\n        cfg[\"lr\"] = float(np.clip(\n            base[\"lr\"] * math.exp(rng.normal(0.0, 0.18)), 3.5e-4, 1.7e-3))\n        cfg[\"decay_gamma\"] = float(np.clip(\n            base[\"decay_gamma\"] + rng.normal(0.0, 0.075), 0.28, 0.84))\n        cfg[\"decay_step\"] = int(np.clip(\n            base[\"decay_step\"] + rng.choice([-1, 0, 1]), 1, 4))\n        cfg[\"half_life\"] = float(np.clip(\n            base[\"half_life\"] * math.exp(rng.normal(0.0, 0.24)), 2.8, 18.0))\n        nearest = int(np.argmin(np.abs(hidden_choices - base[\"hidden\"])))\n        shift = int(rng.choice([-1, 0, 1]))\n        cfg[\"hidden\"] = int(hidden_choices[np.clip(nearest + shift, 0,\n                                                   len(hidden_choices) - 1)])\n        cfg[\"cross_layers\"] = int(np.clip(\n            base[\"cross_layers\"] + rng.choice([-1, 0, 1]), 1, 2))\n        configs.append(cfg)\n    return configs\n\n\ndef append_progress(path, record):\n    with open(path, \"a\") as fh:\n        fh.write(json.dumps(record, sort_keys=True) + \"\\n\")\n\n\ndef main():\n    ap = argparse.ArgumentParser()\n    ap.add_argument(\"--data-dir\", required=True)\n    ap.add_argument(\"--out-dir\", required=True)\n    ap.add_argument(\"--seed\", type=int, default=42)\n    ap.add_argument(\"--epochs\", type=int, default=14)\n    args = ap.parse_args()\n\n    seed_everything(args.seed)\n    device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n    os.makedirs(args.out_dir, exist_ok=True)\n    progress_path = os.path.join(args.out_dir, \"progress.log\")\n    if os.path.exists(progress_path):\n        os.remove(progress_path)\n\n    train_path = os.path.join(args.data_dir, \"train.npz\")\n    val_path = os.path.join(args.data_dir, \"val.npz\")\n    tr = np.load(train_path)\n    va = np.load(val_path)\n\n    field_dims = tr[\"field_dims\"].astype(np.int64)\n    total_dim = int(field_dims.sum())\n    xt_np = tr[\"X\"].astype(np.int64)\n    yt_np = tr[\"y\"].astype(np.float32)\n    train_users_np = np.asarray(tr[\"user\"])\n    xv_np = va[\"X\"].astype(np.int64)\n    val_users = va[\"user\"]\n    val_labels = va[\"y\"].astype(int)\n    ages_np = date_ages(tr[\"date\"])\n\n    pair_pos_np, pair_neg_np, pair_start_np, pair_count_np = build_pair_tables(\n        train_users_np, yt_np)\n    slate_order_np, slate_bounds_np = build_user_slates(train_users_np)\n\n    Xt = torch.from_numpy(xt_np).to(device)\n    yt = torch.from_numpy(yt_np).to(device)\n    train_users = torch.from_numpy(train_users_np.astype(np.int64)).to(device)\n    Xv = torch.from_numpy(xv_np).to(device)\n    ages = torch.from_numpy(ages_np).to(device)\n    pair_pos = torch.from_numpy(pair_pos_np).to(device)\n    pair_neg = torch.from_numpy(pair_neg_np).to(device)\n    pair_start = torch.from_numpy(pair_start_np).to(device)\n    pair_count = torch.from_numpy(pair_count_np).to(device)\n\n    smoke = os.environ.get(\"SMOKE_EPOCHS\")\n    smoke_cap = int(smoke) if smoke is not None else None\n    coarse_epochs = min(3, smoke_cap) if smoke_cap is not None else 3\n    refine_epochs = min(6, smoke_cap) if smoke_cap is not None else 6\n    final_epochs = min(args.epochs, smoke_cap) if smoke_cap is not None else args.epochs\n    coarse_epochs = max(1, coarse_epochs)\n    refine_epochs = max(1, refine_epochs)\n    final_epochs = max(1, final_epochs)\n\n    n = len(yt_np)\n    batch_size = 8192\n    history = []\n\n    def make_complete_slate_batches(row_fraction):\n        target = max(batch_size, min(n, int(round(n * row_fraction))))\n        group_order = np.random.permutation(len(slate_bounds_np) - 1)\n        chosen = []\n        chosen_rows = 0\n        for group_id in group_order:\n            chosen.append(int(group_id))\n            chosen_rows += int(slate_bounds_np[group_id + 1] -\n                               slate_bounds_np[group_id])\n            if chosen_rows >= target:\n                break\n\n        batches = []\n        current = []\n        current_size = 0\n        for group_id in chosen:\n            left = int(slate_bounds_np[group_id])\n            right = int(slate_bounds_np[group_id + 1])\n            rows = slate_order_np[left:right]\n            group_size = right - left\n            if current and current_size + group_size > batch_size:\n                batches.append(np.concatenate(current).astype(np.int64, copy=False))\n                current = []\n                current_size = 0\n            current.append(rows)\n            current_size += group_size\n        if current:\n            batches.append(np.concatenate(current).astype(np.int64, copy=False))\n        return batches\n\n    def predict(model):\n        model.eval()\n        chunks = []\n        with torch.no_grad():\n            for left in range(0, len(Xv), 65536):\n                chunks.append(model(Xv[left:left + 65536]).detach().cpu().numpy())\n        return np.concatenate(chunks)\n\n    def train_candidate(config, epochs, row_fraction, run_seed, stage,\n                        probe_index, half_epoch_checks=False, keep_snapshot=False):\n        seed_everything(run_seed)\n        model = DCNLite(\n            total_dim=total_dim,\n            fields=xt_np.shape[1],\n            k=16,\n            hidden=int(config[\"hidden\"]),\n            cross_layers=int(config[\"cross_layers\"]),\n            dropout=float(config[\"dropout\"]),\n        ).to(device)\n        optimizer = torch.optim.AdamW(\n            model.parameters(), lr=float(config[\"lr\"]),\n            weight_decay=float(config[\"weight_decay\"]))\n        scheduler = torch.optim.lr_scheduler.StepLR(\n            optimizer, step_size=int(config[\"decay_step\"]),\n            gamma=float(config[\"decay_gamma\"]))\n\n        recency = torch.pow(torch.tensor(2.0, device=device),\n                            -ages / float(config[\"half_life\"]))\n        recency = recency / recency.mean().clamp_min(1e-8)\n        best_primary = -1.0\n        best_scores = None\n        best_metric = None\n        best_event = 0.0\n        best_state = None\n        events = []\n\n        for epoch in range(epochs):\n            model.train()\n            epoch_batches = make_complete_slate_batches(row_fraction)\n            total_batches = len(epoch_batches)\n            midpoint = max(1, int(math.ceil(total_batches / 2.0)))\n            checkpoints = {total_batches}\n            if half_epoch_checks:\n                checkpoints.add(midpoint)\n            loss_sum = 0.0\n            seen_batches = 0\n            for batch_number, idx_np in enumerate(epoch_batches, start=1):\n                idx = torch.from_numpy(idx_np).to(device)\n                optimizer.zero_grad(set_to_none=True)\n                logits = model(Xt[idx])\n                centered_logits = gauge_center_logits(\n                    logits, train_users[idx], model.bias)\n                point_loss = F.binary_cross_entropy_with_logits(\n                    centered_logits, yt[idx], reduction=\"none\")\n                bce_loss = ((point_loss * recency[idx]).sum() /\n                            recency[idx].sum().clamp_min(1e-8))\n\n                if len(pair_pos):\n                    selected = torch.randint(len(pair_pos), (len(idx),), device=device)\n                    pos_idx = pair_pos[selected]\n                    counts = pair_count[selected]\n                    offsets = torch.floor(torch.rand(len(idx), device=device) *\n                                          counts.to(torch.float32)).to(torch.long)\n                    neg_idx = pair_neg[pair_start[selected] + offsets]\n                    pair_logits = model(torch.cat((Xt[pos_idx], Xt[neg_idx]), dim=0))\n                    pos_logits = pair_logits[:len(idx)]\n                    neg_logits = pair_logits[len(idx):]\n                    pair_weights = 0.5 * (recency[pos_idx] + recency[neg_idx])\n                    pair_loss = (F.softplus(-(pos_logits - neg_logits)) * pair_weights).sum()\n                    pair_loss = pair_loss / pair_weights.sum().clamp_min(1e-8)\n                    mix = float(config[\"bpr_mix\"])\n                    loss = (1.0 - mix) * bce_loss + mix * pair_loss\n                else:\n                    loss = bce_loss\n\n                loss.backward()\n                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)\n                optimizer.step()\n                loss_sum += float(loss.detach().item())\n                seen_batches += 1\n\n                if batch_number in checkpoints:\n                    scores = predict(model)\n                    metric = metric_values(evaluate(val_users, val_labels, scores))\n                    event_epoch = epoch + batch_number / total_batches\n                    event = {\n                        \"checkpoint_epoch\": round(float(event_epoch), 3),\n                        \"train_loss\": round(loss_sum / max(1, seen_batches), 6),\n                        \"lr\": float(optimizer.param_groups[0][\"lr\"]),\n                        \"gauc\": metric[\"gauc\"],\n                        \"ndcg5\": metric[\"ndcg5\"],\n                        \"primary\": metric[\"primary\"],\n                    }\n                    events.append(event)\n                    if metric[\"primary\"] > best_primary + 1e-9:\n                        best_primary = metric[\"primary\"]\n                        best_scores = scores.copy()\n                        best_metric = metric\n                        best_event = event_epoch\n                        if keep_snapshot:\n                            best_state = {key: value.detach().cpu().clone()\n                                          for key, value in model.state_dict().items()}\n                    if batch_number != total_batches:\n                        model.train()\n            scheduler.step()\n\n        record = {\n            \"stage\": stage,\n            \"probe\": int(probe_index),\n            \"seed\": int(run_seed),\n            \"epochs\": int(epochs),\n            \"row_fraction\": float(row_fraction),\n            \"gauge_fixed_bce\": True,\n            \"complete_user_slates\": True,\n            \"config\": dict(config),\n            \"best_epoch\": round(float(best_event), 3),\n            \"gauc\": best_metric[\"gauc\"],\n            \"ndcg5\": best_metric[\"ndcg5\"],\n            \"primary\": best_metric[\"primary\"],\n            \"checkpoints\": events,\n        }\n        history.append(record)\n        append_progress(progress_path, {\n            \"stage\": stage,\n            \"probe\": int(probe_index),\n            \"config\": dict(config),\n            \"primary\": best_metric[\"primary\"],\n        })\n        del model, optimizer, scheduler, recency, best_state\n        if device.type == \"cuda\":\n            torch.cuda.empty_cache()\n        return best_primary, best_scores, best_metric, record\n\n    coarse_configs = make_coarse_configs(args.seed)\n    coarse_results = []\n    for index, config in enumerate(coarse_configs):\n        result = train_candidate(\n            config=config,\n            epochs=coarse_epochs,\n            row_fraction=0.62,\n            run_seed=args.seed + 101,\n            stage=\"coarse\",\n            probe_index=index,\n        )\n        coarse_results.append((result[0], config))\n    coarse_results.sort(key=lambda item: item[0], reverse=True)\n    coarse_winner = dict(coarse_results[0][1])\n\n    refine_configs = make_refine_configs(coarse_winner, args.seed)\n    refine_results = []\n    refine_seed = args.seed + 202\n    for index, config in enumerate(refine_configs):\n        result = train_candidate(\n            config=config,\n            epochs=refine_epochs,\n            row_fraction=1.0,\n            run_seed=refine_seed,\n            stage=\"refine\",\n            probe_index=index,\n        )\n        refine_results.append((result[0], dict(config), result[3]))\n    refine_results.sort(key=lambda item: item[0], reverse=True)\n    winning_config = dict(refine_results[0][1])\n    winning_refine_primary = float(refine_results[0][0])\n\n    final_primary, final_scores, final_metric, final_record = train_candidate(\n        config=winning_config,\n        epochs=final_epochs,\n        row_fraction=1.0,\n        run_seed=refine_seed,\n        stage=\"final\",\n        probe_index=0,\n        half_epoch_checks=True,\n        keep_snapshot=True,\n    )\n\n    metrics_payload = {\n        \"gauc\": final_metric[\"gauc\"],\n        \"ndcg5\": final_metric[\"ndcg5\"],\n        \"primary\": final_metric[\"primary\"],\n        \"method\": \"gauge-fixed-bce\",\n        \"gauge_fixed_bce\": True,\n        \"complete_user_slates\": True,\n        \"winning_config\": winning_config,\n        \"coarse_winner_primary\": float(coarse_results[0][0]),\n        \"winning_refine_primary\": winning_refine_primary,\n        \"final_best_epoch\": final_record[\"best_epoch\"],\n        \"history\": history,\n    }\n    with open(os.path.join(args.out_dir, \"metrics.json\"), \"w\") as fh:\n        json.dump(metrics_payload, fh)\n\n    video_offset = int(field_dims[0])\n    video_ids = xv_np[:, 1] - video_offset\n    with open(os.path.join(args.out_dir, \"predictions.csv\"), \"w\") as fh:\n        fh.write(\"row_id,user_id,video_id,score\\n\")\n        for row_id, score in enumerate(final_scores):\n            fh.write(f\"{row_id},{val_users[row_id]},{video_ids[row_id]},{score:.8g}\\n\")\n\n\nif __name__ == \"__main__\":\n    main()\n"}
