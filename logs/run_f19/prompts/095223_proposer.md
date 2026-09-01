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
node_003 [<-node_001] draft "Validation peaks near 0.6039 and then declines, diagnosing overfit, but replacing pointwise BCE with user-centered gauge-fixed BCE on complete user slates while leaving the accepted BPR and regularization package unchanged will remove metric-irrelevant user-level shifts and improve validation primary by about 0.0022." primary=0.6035 REJECTED

Mode: IMPROVE. Apply one change to the parent script (the current best node) — atomic by default, or a cited package / internal fan-out per the task brief. Prefer the highest-expected-gain untried menu item; use the journal to avoid rejected ideas. Emit "edits" blocks (see the envelope contract), NOT a whole script: the smallest set of search/replace edits that tests the hypothesis. The parent is an accepted, debugged artifact; discarding it is a defect.

## Selected method (implement THIS)
### ensemble-design-sweep: Ensemble configuration search at close
- kind: opportunity
- reference_primary: 0.605575 (run_bigclock_07 n6 close, +0.0013)
- GATE (measured + reviewed): seed-ensembling pays only when members are competent
  AND usefully diverse. Before accepting any ensemble: (1) drop any member >0.0010
  primary below the median member or with anomalous tie/quantization rates;
  (2) compute rescue-vs-harm on validation pairs (consensus corrects anchor-wrong
  pairs = rescue; overturns anchor-right pairs = harm), weighted per GAUC user
  weighting; require rescue/harm > 1.2 and positive net rescue; (3) prefer a
  best-anchored soft combination (anchor weight ~0.6 to the best member, soft
  pairwise votes with per-member margin temperature, exact ties = 0.5 votes) over
  raw hard rank averaging — 5-item slates make hard ranks brittle (measured: 4 of
  5 hard-rank closes on strong singles SUBTRACTED). Never select members by the
  candidate ensembles' own validation scores.
- mechanism: The closing ensemble node probes its own design instead of assuming it: member count {3,5,7}, combination rule {per-user rank average, probability average}, optionally member diversity (consecutive seeds vs seeds+dial-jitter). Short-probe the options where affordable, pick on validation, produce the final ensemble. Log all probed designs.
- treats: overfit
- preconditions: Apply to the best accepted single-model champion. This is the canonical last node of a run.
- citation: Deep Ensembles (Lakshminarayanan et al., NeurIPS 2017); rank aggregation practice.
- expected_gain / cost: +0.0004..+0.0015 depending on parent seed variance / medium-high runtime.
- status_pure: measured-win (bigclock_07 n6 close +0.0013 -> 0.605575 CHAMPION; final_s2 hetero variant +0.0010)
- status_1k: untried
selector diagnosis: overfit
selector why: Validation peaks at 0.60390 and then declines, indicating overfit. The accepted champion already contains the regularized package, while two subsequent atomic riders failed and the no-improvement streak is 2 of 3. No eligible unused method has reliable evidence of the >=0.002 gain now required, so the best finalization move is the measured closing sweep on the accepted champion: retrain its exact script across seeds, filter weak members, require positive rescue/harm evidence, and validation-select the aggregation. Its honest expected gain is about +0.0004 to +0.0015, likely below the convergence threshold but more bankable than another speculative atom.

## Convergence pressure
streak_state = {'no_improve_streak': 2, 'n_converge': 3, 'iters_left': 12}
The run ends after N consecutive iterations whose best-so-far improvement is <= epsilon = 0.002. Select experiments by expected scientific value given the remaining budget: at every iteration, including the first, prefer the eligible move with the largest evidence-supported expected gain for its cost; an early iteration spent on a small-ceiling treatment is a convergence strike bought at full price. Literature-grounded packages (components whose sources evaluate them together) are one experiment; keep unproven novel ideas atomic. Plan the run so its final iterations produce the strongest possible finished artifact rather than leaving the run un-finalized. Do the epsilon arithmetic before choosing: if the streak means the run ends unless THIS iteration improves best-so-far by at least epsilon, then a move whose own evidence caps its gain below epsilon cannot extend the run no matter how proven it is; on such an iteration prefer the eligible move with the largest evidence-supported expected gain at or above epsilon, and among qualifying moves prefer the one whose evidence clears epsilon with the widest margin: a move whose evidence only just reaches the bar fails it about half the time, so bare arithmetic reach is not parity with a wide-margin alternative (combining decorrelated mechanism families generally out-gains both re-seeding one family and any single atomic mechanism). Implementation-dead is not evidence-dead: two failed BUILDS of a card mean pivot to a mechanically simpler card, not a third build attempt. Read margins against the CURRENT best, not a card's original baseline: an unspent package whose measured absolute score sits near the current best offers almost no headroom, while a close whose evidence exceeds every single-model score in the ledger offers the most. A proven small-gain close is the right pick only when no eligible move has evidence reaching epsilon. BANK THE LAST GAIN: when the streak is one short of ending the run and NO eligible move's evidence reaches epsilon, the run is in its final iteration either way — the deliverable is best-so-far, not the streak. Choose the move with the most RELIABLE positive expected gain, where reliable means MEASURED on this benchmark (replicated numbers, measured-win statuses) — an estimated or literature range is not bankable. Two tests: (1) prefer a measured small gain over a larger estimate; (2) if THIS run's journal shows fresh-code members just measured far below the champion, every close that requires writing new members inherits that measured weakness — only a close that reuses the champion verbatim (seed ensemble: retrain the SAME accepted script at new seeds, measured +0.0002..+0.0014 — larger the less tuned the single is) keeps its evidence intact. A banked +0.0005 beats a failed +0.002 attempt. If a close was just REJECTED for a gain that did not repeat, its members were too close to the incumbent: re-rolling the same blend with new seeds is not a new experiment; the bottleneck a failed confirm reveals is MEMBER DIVERSITY, so the next node must add a NEW MECHANISM FAMILY the ledger has not yet given the blend (a measured package from another family), then close again. A dosage or regularization treatment on the existing champion does not qualify even if untried: it cannot decorrelate the next blend because it adds no new family. Strengthening means a NEW mechanism or family member: a component the champion stack ALREADY CONTAINS (check its accepted lineage) is not a strengthener, and re-applying it is a no-op, not a treatment. Do not change what counts as an iteration in response to the streak.

## Runtime budget (overrides the 600s default above)
THIS run's per-node timeout is 7200 seconds (~120 minutes). A full-length training on the npz fast path costs roughly 40-90s on CPU, far less on GPU. Plan to SPEND ~60-70% of this budget on search probes when playing a search card — e.g. at 2+ hours that is 40+ full-length probes plus refinement, not 8. Reserve the remainder for the final training(s). Finishing a search node in a small fraction of the budget is a defect, not efficiency: unspent budget is free score variance left unexplored.

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
{"execution_kind":"script","hypothesis":"Because validation peaks early and then falls, the accepted DCN-lite package is variance-limited by overfit; a gated ensemble-design sweep over 48 distinct-seed exact and dial-jittered retrainings will improve validation primary by approximately 0.0010 through decorrelated error correction.","expected_delta":0.001,"expected_delta_basis":"The selected ensemble-design-sweep card reports +0.0013 in run_bigclock_07 and +0.0010 for the heterogeneous final_s2 close, consistent with Deep Ensembles (Lakshminarayanan et al., NeurIPS 2017).","action":"improve","parent":"node_001","edits":[{"search":"def append_progress(path, record):\n    with open(path, \"a\") as fh:\n        fh.write(json.dumps(record, sort_keys=True) + \"\\n\")\n\n\ndef main():","replace":"def append_progress(path, record):\n    with open(path, \"a\") as fh:\n        fh.write(json.dumps(record, sort_keys=True) + \"\\n\")\n\n\ndef stable_sigmoid(values):\n    values = np.asarray(values, dtype=np.float64)\n    return 1.0 / (1.0 + np.exp(-np.clip(values, -35.0, 35.0)))\n\n\ndef grouped_indices(users):\n    order = np.argsort(np.asarray(users), kind=\"stable\")\n    sorted_users = np.asarray(users)[order]\n    cuts = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1\n    bounds = np.concatenate(([0], cuts, [len(order)]))\n    return [order[left:right] for left, right in zip(bounds[:-1], bounds[1:])]\n\n\ndef average_ranks(values):\n    values = np.asarray(values)\n    order = np.argsort(values, kind=\"stable\")\n    ranks = np.empty(len(values), dtype=np.float64)\n    left = 0\n    while left < len(order):\n        right = left + 1\n        while right < len(order) and values[order[right]] == values[order[left]]:\n            right += 1\n        ranks[order[left:right]] = 0.5 * (left + right - 1)\n        left = right\n    if len(values) > 1:\n        ranks /= float(len(values) - 1)\n    return ranks\n\n\ndef rank_average(member_scores, groups):\n    result = np.zeros_like(np.asarray(member_scores[0]), dtype=np.float64)\n    for idx in groups:\n        local = np.zeros(len(idx), dtype=np.float64)\n        for scores in member_scores:\n            local += average_ranks(np.asarray(scores)[idx])\n        result[idx] = local / float(len(member_scores))\n    return result\n\n\ndef margin_temperature(scores, groups):\n    margins = []\n    for idx in groups:\n        local = np.asarray(scores)[idx]\n        if len(local) < 2:\n            continue\n        rows, cols = np.triu_indices(len(local), 1)\n        diff = np.abs(local[rows] - local[cols])\n        diff = diff[diff > 0.0]\n        if len(diff):\n            margins.append(diff)\n    if not margins:\n        return 1.0\n    return max(1.0e-5, float(np.median(np.concatenate(margins))))\n\n\ndef combine_member_scores(member_scores, groups, rule, anchor_index):\n    if rule == \"probability_average\":\n        return np.mean(np.stack([stable_sigmoid(scores) for scores in member_scores]),\n                       axis=0)\n    if rule == \"per_user_rank_average\":\n        return rank_average(member_scores, groups)\n    if rule.startswith(\"anchored_soft_\"):\n        anchor_weight = float(rule.rsplit(\"_\", 1)[1]) / 100.0\n        count = len(member_scores)\n        weights = np.full(count, (1.0 - anchor_weight) / max(1, count - 1),\n                          dtype=np.float64)\n        weights[anchor_index] = anchor_weight\n        temperatures = [margin_temperature(scores, groups)\n                        for scores in member_scores]\n        result = np.zeros_like(np.asarray(member_scores[0]), dtype=np.float64)\n        for idx in groups:\n            if len(idx) == 1:\n                result[idx] = 0.5\n                continue\n            utility = np.zeros(len(idx), dtype=np.float64)\n            for weight, temperature, scores in zip(weights, temperatures,\n                                                    member_scores):\n                local = np.asarray(scores)[idx]\n                margins = (local[:, None] - local[None, :]) / temperature\n                votes = stable_sigmoid(margins)\n                votes[margins == 0.0] = 0.5\n                utility += weight * ((votes.sum(1) - 0.5) /\n                                     float(len(idx) - 1))\n            result[idx] = utility\n        return result\n    raise ValueError(\"unknown ensemble rule\")\n\n\ndef rescue_harm(anchor_scores, candidate_scores, labels, groups):\n    labels = np.asarray(labels)\n    rescue = 0.0\n    harm = 0.0\n    for idx in groups:\n        if len(idx) < 2:\n            continue\n        rows, cols = np.triu_indices(len(idx), 1)\n        left = idx[rows]\n        right = idx[cols]\n        truth = np.sign(labels[left] - labels[right])\n        usable = truth != 0\n        if not np.any(usable):\n            continue\n        left = left[usable]\n        right = right[usable]\n        truth = truth[usable]\n        anchor_margin = (np.asarray(anchor_scores)[left] -\n                         np.asarray(anchor_scores)[right]) * truth\n        candidate_margin = (np.asarray(candidate_scores)[left] -\n                            np.asarray(candidate_scores)[right]) * truth\n        anchor_correct = (anchor_margin > 0.0).astype(np.float64)\n        anchor_correct[anchor_margin == 0.0] = 0.5\n        candidate_correct = (candidate_margin > 0.0).astype(np.float64)\n        candidate_correct[candidate_margin == 0.0] = 0.5\n        pair_weight = float(len(idx)) / float(len(left))\n        rescue += pair_weight * np.maximum(candidate_correct - anchor_correct,\n                                           0.0).sum()\n        harm += pair_weight * np.maximum(anchor_correct - candidate_correct,\n                                         0.0).sum()\n    ratio = rescue / harm if harm > 0.0 else (1.0e9 if rescue > 0.0 else 0.0)\n    return float(rescue), float(harm), float(ratio)\n\n\ndef tie_statistics(scores, groups):\n    exact = 0\n    quantized = 0\n    total = 0\n    for idx in groups:\n        local = np.asarray(scores)[idx]\n        if len(local) < 2:\n            continue\n        rows, cols = np.triu_indices(len(local), 1)\n        exact += int(np.sum(local[rows] == local[cols]))\n        rounded = np.round(local, 6)\n        quantized += int(np.sum(rounded[rows] == rounded[cols]))\n        total += len(rows)\n    denom = max(1, total)\n    return float(exact / denom), float(quantized / denom)\n\n\ndef jitter_config(base, seed):\n    rng = np.random.default_rng(seed + 47011)\n    config = dict(base)\n    config[\"dropout\"] = float(np.clip(\n        base[\"dropout\"] + rng.normal(0.0, 0.018), 0.13, 0.43))\n    config[\"weight_decay\"] = float(np.clip(\n        base[\"weight_decay\"] * math.exp(rng.normal(0.0, 0.18)),\n        2.5e-5, 3.2e-3))\n    config[\"lr\"] = float(np.clip(\n        base[\"lr\"] * math.exp(rng.normal(0.0, 0.075)), 3.5e-4, 1.7e-3))\n    config[\"decay_gamma\"] = float(np.clip(\n        base[\"decay_gamma\"] + rng.normal(0.0, 0.035), 0.28, 0.84))\n    config[\"half_life\"] = float(np.clip(\n        base[\"half_life\"] * math.exp(rng.normal(0.0, 0.08)), 2.8, 18.0))\n    return config\n\n\ndef main():"},{"search":"    metrics_payload = {\n        \"gauc\": final_metric[\"gauc\"],\n        \"ndcg5\": final_metric[\"ndcg5\"],\n        \"primary\": final_metric[\"primary\"],\n        \"winning_config\": winning_config,\n        \"coarse_winner_primary\": float(coarse_results[0][0]),\n        \"winning_refine_primary\": winning_refine_primary,\n        \"final_best_epoch\": final_record[\"best_epoch\"],\n        \"history\": history,\n    }\n    with open(os.path.join(args.out_dir, \"metrics.json\"), \"w\") as fh:\n        json.dump(metrics_payload, fh)\n","replace":"    groups = grouped_indices(val_users)\n    incumbent_entry = {\n        \"kind\": \"incumbent\",\n        \"index\": -1,\n        \"seed\": int(refine_seed),\n        \"scores\": final_scores.copy(),\n        \"metric\": dict(final_metric),\n        \"config\": dict(winning_config),\n    }\n    member_entries = []\n    members_per_family = 24\n    for family_index, family in enumerate((\"exact\", \"dial_jitter\")):\n        for member_index in range(members_per_family):\n            run_seed = (args.seed + 10000 + member_index if family == \"exact\"\n                        else args.seed + 20000 + member_index)\n            member_config = (dict(winning_config) if family == \"exact\"\n                             else jitter_config(winning_config, run_seed))\n            member_primary, member_scores, member_metric, member_record = train_candidate(\n                config=member_config,\n                epochs=final_epochs,\n                row_fraction=1.0,\n                run_seed=run_seed,\n                stage=\"ensemble_member_\" + family,\n                probe_index=member_index,\n                half_epoch_checks=True,\n            )\n            assert not np.allclose(member_scores, final_scores)\n            for previous in member_entries:\n                assert not np.allclose(member_scores, previous[\"scores\"])\n            entry = {\n                \"kind\": family,\n                \"index\": int(member_index),\n                \"seed\": int(run_seed),\n                \"scores\": member_scores.copy(),\n                \"metric\": dict(member_metric),\n                \"config\": dict(member_config),\n            }\n            member_entries.append(entry)\n            append_progress(progress_path, {\n                \"stage\": \"ensemble_member_result\",\n                \"family\": family,\n                \"member\": int(member_index),\n                \"seed\": int(run_seed),\n                \"primary\": float(member_primary),\n            })\n\n    all_entries = [incumbent_entry] + member_entries\n    for entry in all_entries:\n        exact_ties, quantized_ties = tie_statistics(entry[\"scores\"], groups)\n        entry[\"exact_tie_rate\"] = exact_ties\n        entry[\"quantized_tie_rate\"] = quantized_ties\n    member_primaries = np.asarray([entry[\"metric\"][\"primary\"]\n                                   for entry in all_entries])\n    exact_rates = np.asarray([entry[\"exact_tie_rate\"] for entry in all_entries])\n    quantized_rates = np.asarray([entry[\"quantized_tie_rate\"]\n                                  for entry in all_entries])\n    median_primary = float(np.median(member_primaries))\n    median_exact = float(np.median(exact_rates))\n    median_quantized = float(np.median(quantized_rates))\n    exact_mad = float(np.median(np.abs(exact_rates - median_exact)))\n    quantized_mad = float(np.median(np.abs(quantized_rates - median_quantized)))\n    exact_limit = median_exact + max(0.002, 6.0 * exact_mad)\n    quantized_limit = median_quantized + max(0.02, 6.0 * quantized_mad)\n    for entry in all_entries:\n        entry[\"eligible\"] = bool(\n            entry[\"metric\"][\"primary\"] >= median_primary - 0.0010 and\n            entry[\"exact_tie_rate\"] <= exact_limit and\n            entry[\"quantized_tie_rate\"] <= quantized_limit)\n\n    quality_record = {\n        \"stage\": \"ensemble_quality_gate\",\n        \"median_primary\": median_primary,\n        \"minimum_primary\": median_primary - 0.0010,\n        \"exact_tie_limit\": exact_limit,\n        \"quantized_tie_limit\": quantized_limit,\n        \"members\": [{\n            \"kind\": entry[\"kind\"],\n            \"index\": entry[\"index\"],\n            \"seed\": entry[\"seed\"],\n            \"primary\": entry[\"metric\"][\"primary\"],\n            \"exact_tie_rate\": entry[\"exact_tie_rate\"],\n            \"quantized_tie_rate\": entry[\"quantized_tie_rate\"],\n            \"eligible\": entry[\"eligible\"],\n        } for entry in all_entries],\n    }\n    history.append(quality_record)\n    append_progress(progress_path, quality_record)\n\n    incumbent_prefix = [incumbent_entry] if incumbent_entry[\"eligible\"] else []\n    exact_source = [entry for entry in member_entries\n                    if entry[\"kind\"] == \"exact\" and entry[\"eligible\"]]\n    jitter_source = [entry for entry in member_entries\n                     if entry[\"kind\"] == \"dial_jitter\" and entry[\"eligible\"]]\n    mixed_source = []\n    for index in range(max(len(exact_source), len(jitter_source))):\n        if index < len(exact_source):\n            mixed_source.append(exact_source[index])\n        if index < len(jitter_source):\n            mixed_source.append(jitter_source[index])\n\n    rules = (\"per_user_rank_average\", \"probability_average\",\n             \"anchored_soft_55\", \"anchored_soft_60\", \"anchored_soft_65\")\n    design_candidates = []\n    design_summaries = []\n    for diversity, source in ((\"consecutive_seeds\", exact_source),\n                              (\"seed_dial_jitter\", mixed_source)):\n        for member_count in (3, 5, 7):\n            needed = member_count - len(incumbent_prefix)\n            if needed > len(source):\n                continue\n            by_rule = {rule: [] for rule in rules}\n            for replicate in range(3):\n                start = (replicate * max(1, needed)) % len(source)\n                selected_members = list(incumbent_prefix)\n                selected_members.extend(\n                    source[(start + offset) % len(source)]\n                    for offset in range(needed))\n                assert len({entry[\"seed\"] for entry in selected_members}) == member_count\n                anchor_index = int(np.argmax([\n                    entry[\"metric\"][\"primary\"] for entry in selected_members]))\n                anchor = selected_members[anchor_index]\n                vectors = [entry[\"scores\"] for entry in selected_members]\n                for rule in rules:\n                    candidate_scores = combine_member_scores(\n                        vectors, groups, rule, anchor_index)\n                    assert not np.allclose(candidate_scores, final_scores)\n                    metric = metric_values(evaluate(\n                        val_users, val_labels, candidate_scores))\n                    rescue, harm, ratio = rescue_harm(\n                        anchor[\"scores\"], candidate_scores, val_labels, groups)\n                    gate_pass = bool(rescue > harm and rescue > 0.0 and ratio > 1.2)\n                    record = {\n                        \"stage\": \"ensemble_design_probe\",\n                        \"diversity\": diversity,\n                        \"member_count\": int(member_count),\n                        \"rule\": rule,\n                        \"replicate\": int(replicate),\n                        \"member_seeds\": [int(entry[\"seed\"])\n                                         for entry in selected_members],\n                        \"anchor_seed\": int(anchor[\"seed\"]),\n                        \"anchor_primary\": float(anchor[\"metric\"][\"primary\"]),\n                        \"rescue\": rescue,\n                        \"harm\": harm,\n                        \"rescue_harm_ratio\": ratio,\n                        \"gate_pass\": gate_pass,\n                        \"gauc\": metric[\"gauc\"],\n                        \"ndcg5\": metric[\"ndcg5\"],\n                        \"primary\": metric[\"primary\"],\n                    }\n                    history.append(record)\n                    append_progress(progress_path, record)\n                    candidate = {\n                        \"record\": record,\n                        \"scores\": candidate_scores,\n                        \"metric\": metric,\n                        \"anchor_primary\": float(anchor[\"metric\"][\"primary\"]),\n                    }\n                    by_rule[rule].append(candidate)\n                    design_candidates.append(candidate)\n            for rule in rules:\n                candidates = by_rule[rule]\n                valid = [candidate for candidate in candidates\n                         if candidate[\"record\"][\"gate_pass\"]]\n                summary = {\n                    \"diversity\": diversity,\n                    \"member_count\": int(member_count),\n                    \"rule\": rule,\n                    \"mean_primary\": float(np.mean([\n                        candidate[\"metric\"][\"primary\"] for candidate in candidates])),\n                    \"mean_rescue_harm_ratio\": float(np.mean([\n                        min(candidate[\"record\"][\"rescue_harm_ratio\"], 1.0e6)\n                        for candidate in candidates])),\n                    \"valid_count\": int(len(valid)),\n                    \"candidates\": candidates,\n                }\n                design_summaries.append(summary)\n\n    eligible_designs = [summary for summary in design_summaries\n                        if summary[\"valid_count\"] > 0]\n    if not eligible_designs:\n        raise RuntimeError(\"no ensemble passed the rescue/harm gate\")\n    winning_design = max(\n        eligible_designs,\n        key=lambda summary: (summary[\"mean_primary\"],\n                             summary[\"mean_rescue_harm_ratio\"]))\n    winning_candidates = [candidate for candidate in winning_design[\"candidates\"]\n                          if candidate[\"record\"][\"gate_pass\"]]\n    selected_candidate = max(\n        winning_candidates,\n        key=lambda candidate: (candidate[\"anchor_primary\"],\n                               -candidate[\"record\"][\"replicate\"]))\n    final_scores = selected_candidate[\"scores\"]\n    final_metric = selected_candidate[\"metric\"]\n    assert not np.allclose(final_scores, incumbent_entry[\"scores\"])\n\n    metrics_payload = {\n        \"gauc\": final_metric[\"gauc\"],\n        \"ndcg5\": final_metric[\"ndcg5\"],\n        \"primary\": final_metric[\"primary\"],\n        \"winning_config\": winning_config,\n        \"coarse_winner_primary\": float(coarse_results[0][0]),\n        \"winning_refine_primary\": winning_refine_primary,\n        \"final_best_epoch\": final_record[\"best_epoch\"],\n        \"incumbent_primary\": incumbent_entry[\"metric\"][\"primary\"],\n        \"ensemble_design\": {\n            \"diversity\": winning_design[\"diversity\"],\n            \"member_count\": winning_design[\"member_count\"],\n            \"rule\": winning_design[\"rule\"],\n            \"mean_probe_primary\": winning_design[\"mean_primary\"],\n            \"valid_probe_count\": winning_design[\"valid_count\"],\n            \"selected_replicate\": selected_candidate[\"record\"][\"replicate\"],\n            \"selected_member_seeds\": selected_candidate[\"record\"][\"member_seeds\"],\n            \"anchor_seed\": selected_candidate[\"record\"][\"anchor_seed\"],\n            \"rescue\": selected_candidate[\"record\"][\"rescue\"],\n            \"harm\": selected_candidate[\"record\"][\"harm\"],\n            \"rescue_harm_ratio\": selected_candidate[\"record\"][\"rescue_harm_ratio\"],\n            \"selection_policy\": \"design by mean validation primary; cohort by anchor quality among gate-passing predetermined cohorts\",\n        },\n        \"history\": history,\n    }\n    with open(os.path.join(args.out_dir, \"metrics.json\"), \"w\") as fh:\n        json.dump(metrics_payload, fh)\n"}]}
