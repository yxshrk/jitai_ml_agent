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
node_001 [<-node_000] draft "(proposal failed)" no-metric FAILED
node_002 [<-node_001] debug "Replacing ordinary pointwise logits with complete-slate user-centered BCE logits while leaving the hybrid BPR term and regularization unchanged will improve validation primary by approximately 0.0026 through better alignment with within-user ranking metrics." primary=0.6029 ACCEPTED

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
selector diagnosis: underfit
selector why: Validation continues rising through epoch 10 without a decline, so the prescribed diagnosis is underfit, although gains are flattening. With two convergence strikes, the next method must plausibly improve the current 0.6029 best by at least 0.002. This untried one-node cross-family package has evidence around 0.605863, giving materially more headroom than any eligible atomic treatment or same-family package, while its probe gate drops weak members and its full-fidelity blend produces a finalized artifact.

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
streak_state = {'no_improve_streak': 2, 'n_converge': 3, 'iters_left': 13}
The run ends after N consecutive iterations whose best-so-far improvement is <= epsilon = 0.002. Select experiments by expected scientific value given the remaining budget: at every iteration, including the first, prefer the eligible move with the largest evidence-supported expected gain for its cost; an early iteration spent on a small-ceiling treatment is a convergence strike bought at full price. Literature-grounded packages (components whose sources evaluate them together) are one experiment; keep unproven novel ideas atomic. Plan the run so its final iterations produce the strongest possible finished artifact rather than leaving the run un-finalized. Do the epsilon arithmetic before choosing: if the streak means the run ends unless THIS iteration improves best-so-far by at least epsilon, then a move whose own evidence caps its gain below epsilon cannot extend the run no matter how proven it is; on such an iteration prefer the eligible move with the largest evidence-supported expected gain at or above epsilon, and among qualifying moves prefer the one whose evidence clears epsilon with the widest margin: a move whose evidence only just reaches the bar fails it about half the time, so bare arithmetic reach is not parity with a wide-margin alternative (combining decorrelated mechanism families generally out-gains both re-seeding one family and any single atomic mechanism). Read margins against the CURRENT best, not a card's original baseline: an unspent package whose measured absolute score sits near the current best offers almost no headroom, while a close whose evidence exceeds every single-model score in the ledger offers the most. A proven small-gain close is the right pick only when no eligible move has evidence reaching epsilon. Do not change what counts as an iteration in response to the streak.

## Runtime budget (overrides the 600s default above)
THIS run's per-node timeout is 7200 seconds (~120 minutes). A full-length training on the npz fast path costs roughly 40-90s on CPU, far less on GPU. Plan to SPEND ~60-70% of this budget on search probes when playing a search card — e.g. at 2+ hours that is 40+ full-length probes plus refinement, not 8. Reserve the remainder for the final training(s). Finishing a search node in a small fraction of the budget is a defect, not efficiency: unspent budget is free score variance left unexplored.

Directive: draft from Tier 2 of the menu

When implementing ANY ensemble/member card: each member MUST be trained with a distinct seed; after scoring, ASSERT member score vectors are not identical (numpy allclose check between members and against the parent predictions) and print per-member validation primaries to progress output. An ensemble whose final predictions equal the parent's is a no-op and will be rejected by the harness, except when the farm-close executor explicitly selects and records the incumbent fallback.

## Parent node "node_002" (full code)
```python
import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def metric_values(result):
    def get(*names):
        for name in names:
            if name in result:
                return float(result[name])
        raise KeyError(names[0])
    return {
        "gauc": get("GAUC", "gauc"),
        "ndcg5": get("nDCG@5", "ndcg5", "NDCG@5"),
        "primary": get("primary", "PRIMARY"),
    }


def parse_scalar(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def read_csv_rows(path, training):
    feature_rows = []
    labels = []
    users = []
    videos = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        has_author = reader.fieldnames is not None and "author_id" in reader.fieldnames
        for row in reader:
            user = row["user_id"]
            video = row["video_id"]
            author = row["author_id"] if has_author else "__unknown_author__"
            tab = row.get("tab", "0")
            duration = float(row.get("duration_ms", "0") or 0.0)
            feature_rows.append((user, video, author, tab, duration))
            users.append(parse_scalar(user))
            videos.append(parse_scalar(video))
            if training:
                labels.append(float(row["long_view"]))
            else:
                labels.append(float(row["long_view"]))
    return feature_rows, np.asarray(labels, dtype=np.float32), np.asarray(users), np.asarray(videos)


def encode_csv(train_rows, val_rows):
    duration_train = np.asarray([r[4] for r in train_rows], dtype=np.float64)
    if duration_train.size:
        quantiles = np.quantile(duration_train, np.linspace(0.1, 0.9, 9))
        quantiles = np.maximum.accumulate(quantiles)
    else:
        quantiles = np.zeros(9, dtype=np.float64)

    train_columns = [
        [r[0] for r in train_rows],
        [r[1] for r in train_rows],
        [r[2] for r in train_rows],
        [r[3] for r in train_rows],
    ]
    val_columns = [
        [r[0] for r in val_rows],
        [r[1] for r in val_rows],
        [r[2] for r in val_rows],
        [r[3] for r in val_rows],
    ]
    encoded_train = []
    encoded_val = []
    field_dims = []
    offset = 0
    for train_col, val_col in zip(train_columns, val_columns):
        mapping = {}
        for value in train_col:
            if value not in mapping:
                mapping[value] = len(mapping) + 1
        dim = len(mapping) + 1
        encoded_train.append(np.asarray([mapping.get(v, 0) + offset for v in train_col], dtype=np.int64))
        encoded_val.append(np.asarray([mapping.get(v, 0) + offset for v in val_col], dtype=np.int64))
        field_dims.append(dim)
        offset += dim

    train_bucket = np.searchsorted(quantiles, duration_train, side="right").astype(np.int64)
    val_duration = np.asarray([r[4] for r in val_rows], dtype=np.float64)
    val_bucket = np.searchsorted(quantiles, val_duration, side="right").astype(np.int64)
    encoded_train.append(train_bucket + offset)
    encoded_val.append(val_bucket + offset)
    field_dims.append(10)

    return (
        np.column_stack(encoded_train).astype(np.int64),
        np.column_stack(encoded_val).astype(np.int64),
        np.asarray(field_dims, dtype=np.int64),
    )


def load_data(data_dir):
    train_npz = data_dir / "train.npz"
    val_npz = data_dir / "val.npz"
    if train_npz.exists() and val_npz.exists():
        with np.load(train_npz, allow_pickle=False) as tr:
            x_train = np.asarray(tr["X"], dtype=np.int64)
            y_train = np.asarray(tr["y"], dtype=np.float32).reshape(-1)
            train_users = np.asarray(tr["user"]).reshape(-1)
            field_dims = np.asarray(tr["field_dims"], dtype=np.int64).reshape(-1)
        with np.load(val_npz, allow_pickle=False) as va:
            x_val = np.asarray(va["X"], dtype=np.int64)
            y_val = np.asarray(va["y"], dtype=np.float32).reshape(-1)
            val_users = np.asarray(va["user"]).reshape(-1)
        video_offset = int(field_dims[0]) if field_dims.size > 1 else 0
        val_videos = x_val[:, 1].astype(np.int64) - video_offset
        total_dim = max(int(field_dims.sum()), int(max(x_train.max(initial=0), x_val.max(initial=0))) + 1)
        return x_train, y_train, train_users, x_val, y_val, val_users, val_videos, total_dim, True

    train_rows, y_train, train_users, _ = read_csv_rows(data_dir / "train.csv", True)
    val_rows, y_val, val_users, val_videos = read_csv_rows(data_dir / "val.csv", False)
    x_train, x_val, field_dims = encode_csv(train_rows, val_rows)
    return x_train, y_train, train_users, x_val, y_val, val_users, val_videos, int(field_dims.sum()), False


class FactorizationMachine(nn.Module):
    def __init__(self, total_dim, embedding_dim=16, dropout=0.20, initial_bias=0.0):
        super().__init__()
        self.linear = nn.Embedding(total_dim, 1)
        self.embedding = nn.Embedding(total_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.global_bias = nn.Parameter(torch.tensor(float(initial_bias), dtype=torch.float32))
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

    def raw_score(self, x):
        linear = self.linear(x).sum(dim=1).squeeze(-1)
        emb = self.dropout(self.embedding(x))
        summed = emb.sum(dim=1)
        interaction = 0.5 * (summed.square() - emb.square().sum(dim=1)).sum(dim=1)
        return linear + interaction


def build_user_slates(users):
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    if len(order) == 0:
        return order, np.asarray([0], dtype=np.int64)
    starts = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1]])
    boundaries = np.r_[starts, len(order)].astype(np.int64)
    return order.astype(np.int64), boundaries


def make_slate_batches(sorted_rows, boundaries, rng, target_rows=65536):
    n_users = len(boundaries) - 1
    user_order = np.arange(n_users, dtype=np.int64)
    rng.shuffle(user_order)
    batches = []
    pieces = []
    lengths = []
    count = 0
    for user_index in user_order:
        lo = int(boundaries[user_index])
        hi = int(boundaries[user_index + 1])
        length = hi - lo
        if pieces and count + length > target_rows:
            batches.append((np.concatenate(pieces), np.asarray(lengths, dtype=np.int64)))
            pieces = []
            lengths = []
            count = 0
        pieces.append(sorted_rows[lo:hi])
        lengths.append(length)
        count += length
    if pieces:
        batches.append((np.concatenate(pieces), np.asarray(lengths, dtype=np.int64)))
    return batches


def centered_logits(raw, group_ids, group_count, global_bias):
    sums = torch.zeros(group_count, dtype=raw.dtype, device=raw.device)
    sums.index_add_(0, group_ids, raw)
    counts = torch.bincount(group_ids, minlength=group_count).to(raw.dtype)
    means = sums / counts.clamp_min(1.0)
    return raw - means[group_ids] + global_bias


def pair_indices(y_batch, lengths, rng):
    positive = []
    negative = []
    start = 0
    for length in lengths.tolist():
        stop = start + int(length)
        local = y_batch[start:stop]
        pos = np.flatnonzero(local > 0.5) + start
        neg = np.flatnonzero(local <= 0.5) + start
        if pos.size and neg.size:
            pair_count = max(pos.size, neg.size)
            p = rng.choice(pos, size=pair_count, replace=pos.size < pair_count)
            n = rng.choice(neg, size=pair_count, replace=neg.size < pair_count)
            positive.append(p)
            negative.append(n)
        start = stop
    if not positive:
        return None, None
    return np.concatenate(positive).astype(np.int64), np.concatenate(negative).astype(np.int64)


def predict(model, x, users, centered, device, batch_size=131072):
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.as_tensor(x[start:start + batch_size], dtype=torch.long, device=device)
            chunks.append(model.raw_score(xb).detach().cpu().numpy())
    raw = np.concatenate(chunks).astype(np.float64)
    bias = float(model.global_bias.detach().cpu())
    if not centered:
        return raw + bias
    _, inverse = np.unique(users, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    sums = np.bincount(inverse, weights=raw).astype(np.float64)
    return raw - (sums / np.maximum(counts, 1.0))[inverse] + bias


def evaluate_scores(evaluator, users, labels, scores):
    return metric_values(evaluator(users, labels, scores))


def train_one(x_train, y_train, train_users, x_val, y_val, val_users, total_dim,
              centered, seed, epochs, evaluator, device):
    seed_everything(seed)
    rng = np.random.default_rng(seed + 193)
    prevalence = float(np.clip(y_train.mean(), 1e-5, 1.0 - 1e-5))
    initial_bias = math.log(prevalence / (1.0 - prevalence))
    model = FactorizationMachine(total_dim, embedding_dim=16, dropout=0.20,
                                 initial_bias=initial_bias).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 6, 8], gamma=0.35)
    sorted_rows, boundaries = build_user_slates(train_users)
    best_primary = -float("inf")
    best_metrics = None
    best_predictions = None
    best_epoch = 0
    epoch_history = []
    stale = 0

    for epoch in range(epochs):
        model.train()
        batches = make_slate_batches(sorted_rows, boundaries, rng)
        for row_indices, lengths in batches:
            xb = torch.as_tensor(x_train[row_indices], dtype=torch.long, device=device)
            y_np = y_train[row_indices]
            yb = torch.as_tensor(y_np, dtype=torch.float32, device=device)
            group_np = np.repeat(np.arange(len(lengths), dtype=np.int64), lengths)
            group_ids = torch.as_tensor(group_np, dtype=torch.long, device=device)

            optimizer.zero_grad(set_to_none=True)
            raw = model.raw_score(xb)
            if centered:
                point_logits = centered_logits(raw, group_ids, len(lengths), model.global_bias)
            else:
                point_logits = raw + model.global_bias
            bce = F.binary_cross_entropy_with_logits(point_logits, yb)

            pos_np, neg_np = pair_indices(y_np, lengths, rng)
            if pos_np is None:
                bpr = raw.sum() * 0.0
            else:
                pos = torch.as_tensor(pos_np, dtype=torch.long, device=device)
                neg = torch.as_tensor(neg_np, dtype=torch.long, device=device)
                bpr = F.softplus(-(raw[pos] - raw[neg])).mean()
            loss = 0.5 * bce + 0.5 * bpr
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        scheduler.step()
        predictions = predict(model, x_val, val_users, centered, device)
        metrics = evaluate_scores(evaluator, val_users, y_val, predictions)
        epoch_history.append({"epoch": epoch + 1, **metrics})
        if metrics["primary"] > best_primary + 1e-12:
            best_primary = metrics["primary"]
            best_metrics = metrics
            best_predictions = predictions.copy()
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
        if stale >= 3 and epoch + 1 >= 5:
            break

    return best_predictions, best_metrics, best_epoch, epoch_history


def append_progress(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.log"
    if progress_path.exists():
        progress_path.unlink()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = load_data(data_dir)
    x_train, y_train, train_users, x_val, y_val, val_users, val_videos, total_dim, fast_path = loaded

    if fast_path:
        from data.official.evaluate import evaluate as evaluator
    else:
        from harness.evaluate_provisional import evaluate as evaluator

    smoke_value = os.environ.get("SMOKE_EPOCHS")
    epochs = 10
    if smoke_value is not None:
        epochs = max(1, min(epochs, int(smoke_value)))
    seed_count = 7 if smoke_value is None or int(smoke_value) > 1 else 2
    seeds = [args.seed + 1009 * i for i in range(seed_count)]

    history = []
    centered_predictions = []
    paired_deltas = []
    run_start = time.time()

    for seed in seeds:
        ordinary_start = time.time()
        ordinary_pred, ordinary_metrics, ordinary_epoch, ordinary_epochs = train_one(
            x_train, y_train, train_users, x_val, y_val, val_users, total_dim,
            False, seed, epochs, evaluator, device
        )
        ordinary_record = {
            "config": "ordinary_bce_plus_bpr",
            "seed": seed,
            "centered_bce": False,
            "bce_weight": 0.5,
            "bpr_weight": 0.5,
            "embedding_dim": 16,
            "dropout": 0.20,
            "weight_decay": 1e-5,
            "best_epoch": ordinary_epoch,
            "runtime_seconds": time.time() - ordinary_start,
            **ordinary_metrics,
            "epochs": ordinary_epochs,
        }
        history.append(ordinary_record)
        append_progress(progress_path, {k: v for k, v in ordinary_record.items() if k != "epochs"})

        centered_start = time.time()
        centered_pred, centered_metrics, centered_epoch, centered_epochs = train_one(
            x_train, y_train, train_users, x_val, y_val, val_users, total_dim,
            True, seed, epochs, evaluator, device
        )
        if np.allclose(centered_pred, ordinary_pred, rtol=1e-7, atol=1e-8):
            raise RuntimeError("Centered and ordinary member predictions are identical")
        for previous in centered_predictions:
            if np.allclose(centered_pred, previous, rtol=1e-7, atol=1e-8):
                raise RuntimeError("Two centered seed members produced identical predictions")
        centered_predictions.append(centered_pred)
        delta = float(centered_metrics["primary"] - ordinary_metrics["primary"])
        paired_deltas.append(delta)
        centered_record = {
            "config": "gauge_fixed_bce_plus_bpr",
            "seed": seed,
            "centered_bce": True,
            "complete_user_slates": True,
            "bce_weight": 0.5,
            "bpr_weight": 0.5,
            "embedding_dim": 16,
            "dropout": 0.20,
            "weight_decay": 1e-5,
            "best_epoch": centered_epoch,
            "paired_primary_delta": delta,
            "runtime_seconds": time.time() - centered_start,
            **centered_metrics,
            "epochs": centered_epochs,
        }
        history.append(centered_record)
        append_progress(progress_path, {k: v for k, v in centered_record.items() if k != "epochs"})

    final_predictions = np.mean(np.stack(centered_predictions, axis=0), axis=0)
    final_metrics = evaluate_scores(evaluator, val_users, y_val, final_predictions)
    delta_array = np.asarray(paired_deltas, dtype=np.float64)
    mean_delta = float(delta_array.mean())
    if len(delta_array) > 1:
        standard_error = float(delta_array.std(ddof=1) / math.sqrt(len(delta_array)))
    else:
        standard_error = 0.0

    predictions_path = out_dir / "predictions.csv"
    with open(predictions_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for row_id, (user, video, score) in enumerate(zip(val_users, val_videos, final_predictions)):
            user_value = user.item() if isinstance(user, np.generic) else user
            video_value = video.item() if isinstance(video, np.generic) else video
            writer.writerow([row_id, user_value, video_value, format(float(score), ".12g")])

    metrics_payload = {
        "gauc": final_metrics["gauc"],
        "ndcg5": final_metrics["ndcg5"],
        "primary": final_metrics["primary"],
        "history": history,
        "paired_summary": {
            "seeds": seeds,
            "primary_deltas": paired_deltas,
            "mean_primary_delta": mean_delta,
            "standard_error": standard_error,
            "ci95_low": mean_delta - 1.96 * standard_error,
            "ci95_high": mean_delta + 1.96 * standard_error,
        },
        "final_model": {
            "type": "mean_of_gauge_fixed_seed_members",
            "member_count": len(centered_predictions),
            "complete_user_slates": True,
            "total_runtime_seconds": time.time() - run_start,
            "device": device.type,
            "fast_path": fast_path,
        },
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, sort_keys=True, indent=2)


if __name__ == "__main__":
    main()

```

## Parent learning curve (per epoch)
epoch 1: train_loss None, val_gauc 0.6568949222564697, val_primary 0.5934718251228333
epoch 2: train_loss None, val_gauc 0.6626065969467163, val_primary 0.5975932478904724
epoch 3: train_loss None, val_gauc 0.6662139892578125, val_primary 0.6006368398666382
epoch 4: train_loss None, val_gauc 0.6670058369636536, val_primary 0.6012954115867615
epoch 5: train_loss None, val_gauc 0.6682368516921997, val_primary 0.6023193597793579
epoch 6: train_loss None, val_gauc 0.6687070727348328, val_primary 0.6025843620300293
epoch 7: train_loss None, val_gauc 0.6690689325332642, val_primary 0.6028332114219666
epoch 8: train_loss None, val_gauc 0.6694426536560059, val_primary 0.6031064987182617
epoch 9: train_loss None, val_gauc 0.66941237449646, val_primary 0.6031171083450317
epoch 10: train_loss None, val_gauc 0.6693763136863708, val_primary 0.6031186580657959
DIAGNOSE before proposing: val peaks early then falls = overfit (attack with regularization/schedules); val still rising at stop = underfit (train longer); flat = the idea itself adds no signal (change direction, not dosage). State your diagnosis in the hypothesis.

Respond with the JSON object only.

## REPLY
{"execution_kind":"farm_close","hypothesis":"Because the parent validation primary is still rising at epoch 10, the system is underfit; rank-averaging complementary regularized DCN, temporal-kernel, causal sequential DeepFM, and recency-weighted FM families will improve validation primary by about 0.0030 over node_002.","expected_delta":0.003,"expected_delta_basis":"The selected diverse-family-farm-close card reports cross-family blends at 0.6058-0.6065, versus the current parent primary of 0.6029, while same-family ensembles add only about 0.0003.","action":"draft","parent":"node_002","timeout_s":7200,"ensemble_plan":{"probe_epochs":2,"full_member_limit":3,"min_probe_blend_gain":0.0,"members":[{"family":"regularized_dcn_package","code":"import argparse\nimport csv\nimport json\nimport math\nimport os\nimport random\nfrom pathlib import Path\n\nimport numpy as np\nimport torch\nfrom torch import nn\nimport torch.nn.functional as F\n\n\ndef seed_all(seed):\n    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)\n    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)\n    if hasattr(torch.backends, 'cudnn'):\n        torch.backends.cudnn.benchmark = False\n        torch.backends.cudnn.deterministic = True\n\n\ndef metrics_of(r):\n    def g(*names):\n        for n in names:\n            if n in r: return float(r[n])\n        raise KeyError(names[0])\n    return {'gauc': g('GAUC','gauc'), 'ndcg5': g('nDCG@5','ndcg5','NDCG@5'), 'primary': g('primary','PRIMARY')}\n\n\ndef scalar(v):\n    try: return int(v)\n    except (ValueError, TypeError): return v\n\n\ndef load_csv(dd):\n    def read(path, train):\n        rows=[]; y=[]; users=[]; videos=[]\n        with open(path, newline='', encoding='utf-8') as f:\n            rd=csv.DictReader(f); has_author='author_id' in (rd.fieldnames or [])\n            for r in rd:\n                u=r['user_id']; v=r['video_id']; a=r['author_id'] if has_author else '__author__'\n                rows.append((u,v,a,r.get('tab','0'),float(r.get('duration_ms','0') or 0)))\n                users.append(scalar(u)); videos.append(scalar(v)); y.append(float(r['long_view']))\n        return rows,np.asarray(y,np.float32),np.asarray(users),np.asarray(videos)\n    tr,yt,ut,_=read(dd/'train.csv',True); va,yv,uv,vv=read(dd/'val.csv',False)\n    q=np.quantile(np.asarray([r[4] for r in tr]),np.linspace(.1,.9,9)) if tr else np.zeros(9)\n    xt=[]; xv=[]; dims=[]; off=0\n    for j in range(4):\n        mp={}\n        for r in tr:\n            if r[j] not in mp: mp[r[j]]=len(mp)+1\n        d=len(mp)+1; xt.append(np.asarray([mp.get(r[j],0)+off for r in tr])); xv.append(np.asarray([mp.get(r[j],0)+off for r in va])); dims.append(d); off+=d\n    xt.append(np.searchsorted(q,[r[4] for r in tr],side='right')+off); xv.append(np.searchsorted(q,[r[4] for r in va],side='right')+off); dims.append(10)\n    return np.column_stack(xt).astype(np.int64),yt,ut,np.column_stack(xv).astype(np.int64),yv,uv,vv,int(sum(dims)),False\n\n\ndef load(dd):\n    if (dd/'train.npz').exists() and (dd/'val.npz').exists():\n        with np.load(dd/'train.npz',allow_pickle=False) as z:\n            xt=np.asarray(z['X'],np.int64); yt=np.asarray(z['y'],np.float32).reshape(-1); ut=np.asarray(z['user']).reshape(-1); dims=np.asarray(z['field_dims'],np.int64)\n        with np.load(dd/'val.npz',allow_pickle=False) as z:\n            xv=np.asarray(z['X'],np.int64); yv=np.asarray(z['y'],np.float32).reshape(-1); uv=np.asarray(z['user']).reshape(-1)\n        vv=xv[:,1]-int(dims[0]); total=max(int(dims.sum()),int(max(xt.max(initial=0),xv.max(initial=0)))+1)\n        return xt,yt,ut,xv,yv,uv,vv,total,True\n    return load_csv(dd)\n\n\nclass DCN(nn.Module):\n    def __init__(self,total,fields,bias):\n        super().__init__(); k=16; width=fields*k\n        self.e=nn.Embedding(total,k); self.lin=nn.Embedding(total,1); self.bias=nn.Parameter(torch.tensor(bias,dtype=torch.float32))\n        self.w1=nn.Parameter(torch.empty(width)); self.b1=nn.Parameter(torch.zeros(width)); self.w2=nn.Parameter(torch.empty(width)); self.b2=nn.Parameter(torch.zeros(width))\n        self.mlp=nn.Sequential(nn.Linear(width,128),nn.ReLU(),nn.Dropout(.30),nn.Linear(128,64),nn.ReLU(),nn.Dropout(.20),nn.Linear(64,1))\n        self.out=nn.Linear(width,1,bias=False); nn.init.normal_(self.e.weight,std=.01); nn.init.zeros_(self.lin.weight); nn.init.normal_(self.w1,std=.01); nn.init.normal_(self.w2,std=.01)\n    def forward(self,x):\n        x0=F.dropout(self.e(x),.20,self.training).flatten(1); z=x0\n        z=x0*(z*self.w1).sum(1,keepdim=True)+self.b1+z\n        z=x0*(z*self.w2).sum(1,keepdim=True)+self.b2+z\n        return self.lin(x).sum(1).squeeze(-1)+self.out(z).squeeze(-1)+self.mlp(x0).squeeze(-1)+self.bias\n\n\ndef predict(model,x,device):\n    model.eval(); out=[]\n    with torch.no_grad():\n        for s in range(0,len(x),131072): out.append(model(torch.as_tensor(x[s:s+131072],dtype=torch.long,device=device)).cpu().numpy())\n    return np.concatenate(out).astype(np.float64)\n\n\ndef main():\n    ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',required=True); ap.add_argument('--out-dir',required=True); ap.add_argument('--seed',type=int,default=42); a=ap.parse_args()\n    seed_all(a.seed); dd=Path(a.data_dir); od=Path(a.out_dir); od.mkdir(parents=True,exist_ok=True); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n    xt,yt,ut,xv,yv,uv,vv,total,fast=load(dd); evaluator=__import__('data.official.evaluate',fromlist=['evaluate']).evaluate if fast else __import__('harness.evaluate_provisional',fromlist=['evaluate']).evaluate\n    p=float(np.clip(yt.mean(),1e-5,1-1e-5)); model=DCN(total,xt.shape[1],math.log(p/(1-p))).to(device)\n    opt=torch.optim.AdamW(model.parameters(),lr=.0025,weight_decay=8e-5); sch=torch.optim.lr_scheduler.MultiStepLR(opt,[3,6,9,12],gamma=.4)\n    epochs=14; smoke=os.environ.get('SMOKE_EPOCHS'); epochs=min(epochs,max(1,int(smoke))) if smoke is not None else epochs\n    rng=np.random.default_rng(a.seed+17); best=None; bm=None; be=0; hist=[]\n    for ep in range(epochs):\n        model.train(); order=rng.permutation(len(xt))\n        for s in range(0,len(order),32768):\n            ix=order[s:s+32768]; xb=torch.as_tensor(xt[ix],dtype=torch.long,device=device); yb=torch.as_tensor(yt[ix],dtype=torch.float32,device=device)\n            opt.zero_grad(set_to_none=True); loss=F.binary_cross_entropy_with_logits(model(xb),yb,label_smoothing=.015); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); opt.step()\n        sch.step(); pr=predict(model,xv,device); m=metrics_of(evaluator(uv,yv,pr)); hist.append({'epoch':ep+1,**m})\n        if bm is None or m['primary']>bm['primary']: best=pr.copy(); bm=m; be=ep+1\n    with open(od/'predictions.csv','w',newline='',encoding='utf-8') as f:\n        w=csv.writer(f); w.writerow(['row_id','user_id','video_id','score'])\n        for i,(u,v,s) in enumerate(zip(uv,vv,best)): w.writerow([i,u.item() if isinstance(u,np.generic) else u,v.item() if isinstance(v,np.generic) else v,format(float(s),'.12g')])\n    payload={**bm,'history':hist,'family':'regularized_dcn_package','best_epoch':be,'seed':a.seed}\n    with open(od/'metrics.json','w',encoding='utf-8') as f: json.dump(payload,f,sort_keys=True,indent=2)\n    with open(od/'progress.log','a',encoding='utf-8') as f: f.write(json.dumps({'family':'regularized_dcn_package','seed':a.seed,**bm},sort_keys=True)+'\\n')\n\n\nif __name__=='__main__': main()\n","config":{"embedding_dim":16,"cross_layers":2,"dropout":0.3,"weight_decay":8e-05},"seed":42},{"family":"temporal_pair_kernel","code":"import argparse\nimport csv\nimport json\nimport math\nimport os\nimport random\nfrom pathlib import Path\n\nimport numpy as np\nimport torch\nfrom torch import nn\nimport torch.nn.functional as F\n\n\ndef seed_all(s):\n    random.seed(s); np.random.seed(s); torch.manual_seed(s)\n    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)\n    if hasattr(torch.backends,'cudnn'): torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True\n\n\ndef mv(r):\n    def g(*ns):\n        for n in ns:\n            if n in r: return float(r[n])\n        raise KeyError(ns[0])\n    return {'gauc':g('GAUC','gauc'),'ndcg5':g('nDCG@5','ndcg5','NDCG@5'),'primary':g('primary','PRIMARY')}\n\n\ndef sc(v):\n    try:return int(v)\n    except:return v\n\n\ndef contexts(hour,date):\n    h=np.asarray(hour).astype(np.int64); h=np.where(h>=100,h//100,h)%24\n    d=np.asarray(date).astype(np.int64); dow=(d%100+2*(d//100%100)+3)%7\n    return np.column_stack([h,dow]).astype(np.int64)\n\n\ndef load(dd):\n    if (dd/'train.npz').exists() and (dd/'val.npz').exists():\n        with np.load(dd/'train.npz',allow_pickle=False) as z: xt=np.asarray(z['X'],np.int64); yt=np.asarray(z['y'],np.float32).reshape(-1); ut=np.asarray(z['user']).reshape(-1); ct=contexts(z['hourmin'],z['date']); dims=np.asarray(z['field_dims'],np.int64)\n        with np.load(dd/'val.npz',allow_pickle=False) as z: xv=np.asarray(z['X'],np.int64); yv=np.asarray(z['y'],np.float32).reshape(-1); uv=np.asarray(z['user']).reshape(-1); cv=contexts(z['hourmin'],z['date'])\n        vv=xv[:,1]-int(dims[0]); total=max(int(dims.sum()),int(max(xt.max(initial=0),xv.max(initial=0)))+1); return xt,yt,ut,ct,xv,yv,uv,cv,vv,total,True\n    def read(path):\n        rows=[]; y=[]; us=[]; vs=[]; hm=[]; dt=[]\n        with open(path,newline='',encoding='utf-8') as f:\n            rd=csv.DictReader(f); ha='author_id' in (rd.fieldnames or [])\n            for r in rd:\n                u=r['user_id']; v=r['video_id']; rows.append((u,v,r['author_id'] if ha else '__author__',r.get('tab','0'),float(r.get('duration_ms','0') or 0))); y.append(float(r['long_view'])); us.append(sc(u)); vs.append(sc(v)); hm.append(int(float(r.get('hourmin','0') or 0))); dt.append(int(float(r.get('date','0') or 0)))\n        return rows,np.asarray(y,np.float32),np.asarray(us),np.asarray(vs),contexts(hm,dt)\n    tr,yt,ut,_,ct=read(dd/'train.csv'); va,yv,uv,vv,cv=read(dd/'val.csv'); q=np.quantile([r[4] for r in tr],np.linspace(.1,.9,9)); A=[]; B=[]; off=0; dims=[]\n    for j in range(4):\n        mp={}\n        for r in tr:\n            if r[j] not in mp:mp[r[j]]=len(mp)+1\n        d=len(mp)+1; A.append(np.asarray([mp.get(r[j],0)+off for r in tr])); B.append(np.asarray([mp.get(r[j],0)+off for r in va])); dims.append(d); off+=d\n    A.append(np.searchsorted(q,[r[4] for r in tr])+off); B.append(np.searchsorted(q,[r[4] for r in va])+off); dims.append(10)\n    return np.column_stack(A).astype(np.int64),yt,ut,ct,np.column_stack(B).astype(np.int64),yv,uv,cv,vv,int(sum(dims)),False\n\n\nclass TemporalFM(nn.Module):\n    def __init__(self,total,bias):\n        super().__init__(); k=16; self.e=nn.Embedding(total,k); self.l=nn.Embedding(total,1); self.hour=nn.Embedding(24,k); self.day=nn.Embedding(7,k); self.gate=nn.Sequential(nn.Linear(k*2,32),nn.ReLU(),nn.Dropout(.15),nn.Linear(32,1)); self.bias=nn.Parameter(torch.tensor(bias,dtype=torch.float32)); nn.init.normal_(self.e.weight,std=.01); nn.init.zeros_(self.l.weight)\n    def forward(self,x,c):\n        e=F.dropout(self.e(x),.18,self.training); sm=e.sum(1); fm=.5*(sm.square()-e.square().sum(1)).sum(1); ctx=self.hour(c[:,0])+self.day(c[:,1]); item=e[:,1]; author=e[:,2]; kernel=(item*ctx).sum(1)+(author*self.hour(c[:,0])).sum(1)*.5; gate=self.gate(torch.cat([item,ctx],1)).squeeze(1); return self.l(x).sum(1).squeeze(1)+fm+kernel+gate+self.bias\n\n\ndef pred(m,x,c,d):\n    m.eval(); z=[]\n    with torch.no_grad():\n        for s in range(0,len(x),131072): z.append(m(torch.as_tensor(x[s:s+131072],dtype=torch.long,device=d),torch.as_tensor(c[s:s+131072],dtype=torch.long,device=d)).cpu().numpy())\n    return np.concatenate(z).astype(np.float64)\n\n\ndef main():\n    ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',required=True); ap.add_argument('--out-dir',required=True); ap.add_argument('--seed',type=int,default=42); a=ap.parse_args(); seed_all(a.seed); od=Path(a.out_dir); od.mkdir(parents=True,exist_ok=True)\n    xt,yt,ut,ct,xv,yv,uv,cv,vv,total,fast=load(Path(a.data_dir)); ev=__import__('data.official.evaluate',fromlist=['evaluate']).evaluate if fast else __import__('harness.evaluate_provisional',fromlist=['evaluate']).evaluate; d=torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n    p=float(np.clip(yt.mean(),1e-5,1-1e-5)); m=TemporalFM(total,math.log(p/(1-p))).to(d); opt=torch.optim.AdamW(m.parameters(),lr=.0028,weight_decay=4e-5); sch=torch.optim.lr_scheduler.MultiStepLR(opt,[4,7,10],gamma=.35)\n    epochs=14; smoke=os.environ.get('SMOKE_EPOCHS'); epochs=min(epochs,max(1,int(smoke))) if smoke is not None else epochs; rng=np.random.default_rng(a.seed+71); best=None; bm=None; be=0; hist=[]\n    for ep in range(epochs):\n        m.train(); order=rng.permutation(len(xt))\n        for s in range(0,len(order),49152):\n            ix=order[s:s+49152]; xb=torch.as_tensor(xt[ix],dtype=torch.long,device=d); cb=torch.as_tensor(ct[ix],dtype=torch.long,device=d); yb=torch.as_tensor(yt[ix],dtype=torch.float32,device=d); opt.zero_grad(set_to_none=True); loss=F.binary_cross_entropy_with_logits(m(xb,cb),yb); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),5); opt.step()\n        sch.step(); pr=pred(m,xv,cv,d); mm=mv(ev(uv,yv,pr)); hist.append({'epoch':ep+1,**mm})\n        if bm is None or mm['primary']>bm['primary']: best=pr.copy(); bm=mm; be=ep+1\n    with open(od/'predictions.csv','w',newline='',encoding='utf-8') as f:\n        w=csv.writer(f); w.writerow(['row_id','user_id','video_id','score'])\n        for i,(u,v,s) in enumerate(zip(uv,vv,best)):w.writerow([i,u.item() if isinstance(u,np.generic) else u,v.item() if isinstance(v,np.generic) else v,format(float(s),'.12g')])\n    with open(od/'metrics.json','w',encoding='utf-8') as f:json.dump({**bm,'history':hist,'family':'temporal_pair_kernel','best_epoch':be,'seed':a.seed},f,sort_keys=True,indent=2)\n    with open(od/'progress.log','a',encoding='utf-8') as f:f.write(json.dumps({'family':'temporal_pair_kernel','seed':a.seed,**bm},sort_keys=True)+'\\n')\n\n\nif __name__=='__main__':main()\n","config":{"embedding_dim":16,"hour_kernel":true,"day_kernel":true,"weight_decay":4e-05},"seed":1051},{"family":"causal_seq_deepfm_composite","code":"import argparse\nimport csv\nimport json\nimport math\nimport os\nimport random\nfrom pathlib import Path\n\nimport numpy as np\nimport torch\nfrom torch import nn\nimport torch.nn.functional as F\n\n\ndef seed_all(s):\n    random.seed(s); np.random.seed(s); torch.manual_seed(s)\n    if torch.cuda.is_available():torch.cuda.manual_seed_all(s)\n    if hasattr(torch.backends,'cudnn'):torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True\n\n\ndef met(r):\n    def g(*ns):\n        for n in ns:\n            if n in r:return float(r[n])\n        raise KeyError(ns[0])\n    return {'gauc':g('GAUC','gauc'),'ndcg5':g('nDCG@5','ndcg5','NDCG@5'),'primary':g('primary','PRIMARY')}\n\n\ndef sc(v):\n    try:return int(v)\n    except:return v\n\n\ndef previous_items(xt,ut,xv,uv,unknown):\n    last={}; pt=np.empty(len(xt),np.int64)\n    for i,(u,row) in enumerate(zip(ut,xt)):\n        key=u.item() if isinstance(u,np.generic) else u; pt[i]=last.get(key,unknown); last[key]=int(row[1])\n    pv=np.empty(len(xv),np.int64); state=dict(last)\n    for i,(u,row) in enumerate(zip(uv,xv)):\n        key=u.item() if isinstance(u,np.generic) else u; pv[i]=state.get(key,unknown); state[key]=int(row[1])\n    return pt,pv\n\n\ndef load(dd):\n    if (dd/'train.npz').exists() and (dd/'val.npz').exists():\n        with np.load(dd/'train.npz',allow_pickle=False) as z:xt=np.asarray(z['X'],np.int64);yt=np.asarray(z['y'],np.float32).reshape(-1);ut=np.asarray(z['user']).reshape(-1);click=np.asarray(z['click'],np.float32).reshape(-1);dims=np.asarray(z['field_dims'],np.int64)\n        with np.load(dd/'val.npz',allow_pickle=False) as z:xv=np.asarray(z['X'],np.int64);yv=np.asarray(z['y'],np.float32).reshape(-1);uv=np.asarray(z['user']).reshape(-1)\n        vv=xv[:,1]-int(dims[0]);total=max(int(dims.sum()),int(max(xt.max(initial=0),xv.max(initial=0)))+1);pt,pv=previous_items(xt,ut,xv,uv,int(dims[0]));return xt,yt,click,ut,pt,xv,yv,uv,pv,vv,total,True\n    def read(path,train):\n        rows=[];y=[];us=[];vs=[];click=[]\n        with open(path,newline='',encoding='utf-8') as f:\n            rd=csv.DictReader(f);ha='author_id' in (rd.fieldnames or [])\n            for r in rd:\n                u=r['user_id'];v=r['video_id'];rows.append((u,v,r['author_id'] if ha else '__author__',r.get('tab','0'),float(r.get('duration_ms','0') or 0)));y.append(float(r['long_view']));us.append(sc(u));vs.append(sc(v));click.append(float(r.get('click','0') or 0) if train else 0)\n        return rows,np.asarray(y,np.float32),np.asarray(click,np.float32),np.asarray(us),np.asarray(vs)\n    tr,yt,click,ut,_=read(dd/'train.csv',True);va,yv,_,uv,vv=read(dd/'val.csv',False);q=np.quantile([r[4] for r in tr],np.linspace(.1,.9,9));A=[];B=[];off=0;dims=[]\n    for j in range(4):\n        mp={}\n        for r in tr:\n            if r[j] not in mp:mp[r[j]]=len(mp)+1\n        d=len(mp)+1;A.append(np.asarray([mp.get(r[j],0)+off for r in tr]));B.append(np.asarray([mp.get(r[j],0)+off for r in va]));dims.append(d);off+=d\n    A.append(np.searchsorted(q,[r[4] for r in tr])+off);B.append(np.searchsorted(q,[r[4] for r in va])+off);dims.append(10);xt=np.column_stack(A).astype(np.int64);xv=np.column_stack(B).astype(np.int64);pt,pv=previous_items(xt,ut,xv,uv,dims[0]);return xt,yt,click,ut,pt,xv,yv,uv,pv,vv,int(sum(dims)),False\n\n\nclass SeqDeepFM(nn.Module):\n    def __init__(self,total,fields,bias):\n        super().__init__();k=16;self.e=nn.Embedding(total,k);self.l=nn.Embedding(total,1);self.net=nn.Sequential(nn.Linear((fields+1)*k,128),nn.ReLU(),nn.Dropout(.25),nn.Linear(128,64),nn.ReLU(),nn.Dropout(.15));self.main=nn.Linear(65,1);self.aux=nn.Linear(64,1);self.bias=nn.Parameter(torch.tensor(bias,dtype=torch.float32));nn.init.normal_(self.e.weight,std=.01);nn.init.zeros_(self.l.weight)\n    def forward(self,x,prev):\n        e=F.dropout(self.e(x),.15,self.training);pe=self.e(prev);sm=e.sum(1);fm=.5*(sm.square()-e.square().sum(1)).sum(1);seq=(e[:,1]*pe).sum(1,keepdim=True);h=self.net(torch.cat([e.flatten(1),pe],1));base=self.l(x).sum(1).squeeze(1)+fm+self.bias;main=base+self.main(torch.cat([h,seq],1)).squeeze(1);return main,self.aux(h).squeeze(1)\n\n\ndef pred(m,x,p,d):\n    m.eval();z=[]\n    with torch.no_grad():\n        for s in range(0,len(x),131072):z.append(m(torch.as_tensor(x[s:s+131072],dtype=torch.long,device=d),torch.as_tensor(p[s:s+131072],dtype=torch.long,device=d))[0].cpu().numpy())\n    return np.concatenate(z).astype(np.float64)\n\n\ndef main():\n    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',required=True);ap.add_argument('--out-dir',required=True);ap.add_argument('--seed',type=int,default=42);a=ap.parse_args();seed_all(a.seed);od=Path(a.out_dir);od.mkdir(parents=True,exist_ok=True)\n    xt,yt,click,ut,pt,xv,yv,uv,pv,vv,total,fast=load(Path(a.data_dir));ev=__import__('data.official.evaluate',fromlist=['evaluate']).evaluate if fast else __import__('harness.evaluate_provisional',fromlist=['evaluate']).evaluate;d=torch.device('cuda' if torch.cuda.is_available() else 'cpu');p=float(np.clip(yt.mean(),1e-5,1-1e-5));m=SeqDeepFM(total,xt.shape[1],math.log(p/(1-p))).to(d);opt=torch.optim.AdamW(m.parameters(),lr=.0022,weight_decay=6e-5);sch=torch.optim.lr_scheduler.MultiStepLR(opt,[4,7,10],gamma=.4)\n    epochs=14;smoke=os.environ.get('SMOKE_EPOCHS');epochs=min(epochs,max(1,int(smoke))) if smoke is not None else epochs;rng=np.random.default_rng(a.seed+313);best=None;bm=None;be=0;hist=[]\n    for ep in range(epochs):\n        m.train();order=rng.permutation(len(xt))\n        for s in range(0,len(order),32768):\n            ix=order[s:s+32768];xb=torch.as_tensor(xt[ix],dtype=torch.long,device=d);pb=torch.as_tensor(pt[ix],dtype=torch.long,device=d);yb=torch.as_tensor(yt[ix],dtype=torch.float32,device=d);cb=torch.as_tensor(click[ix],dtype=torch.float32,device=d);opt.zero_grad(set_to_none=True);main,aux=m(xb,pb);loss=F.binary_cross_entropy_with_logits(main,yb)+.1*F.binary_cross_entropy_with_logits(aux,cb);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()\n        sch.step();pr=pred(m,xv,pv,d);mm=met(ev(uv,yv,pr));hist.append({'epoch':ep+1,**mm})\n        if bm is None or mm['primary']>bm['primary']:best=pr.copy();bm=mm;be=ep+1\n    with open(od/'predictions.csv','w',newline='',encoding='utf-8') as f:\n        w=csv.writer(f);w.writerow(['row_id','user_id','video_id','score'])\n        for i,(u,v,s) in enumerate(zip(uv,vv,best)):w.writerow([i,u.item() if isinstance(u,np.generic) else u,v.item() if isinstance(v,np.generic) else v,format(float(s),'.12g')])\n    with open(od/'metrics.json','w',encoding='utf-8') as f:json.dump({**bm,'history':hist,'family':'causal_seq_deepfm_composite','best_epoch':be,'seed':a.seed},f,sort_keys=True,indent=2)\n    with open(od/'progress.log','a',encoding='utf-8') as f:f.write(json.dumps({'family':'causal_seq_deepfm_composite','seed':a.seed,**bm},sort_keys=True)+'\\n')\n\n\nif __name__=='__main__':main()\n","config":{"embedding_dim":16,"hidden_dim":128,"aux_click_weight":0.1,"causal_previous_item":true},"seed":2060},{"family":"recency_weighted_bilinear_fm","code":"import argparse\nimport csv\nimport json\nimport math\nimport os\nimport random\nfrom pathlib import Path\n\nimport numpy as np\nimport torch\nfrom torch import nn\nimport torch.nn.functional as F\n\n\ndef seed_all(s):\n    random.seed(s);np.random.seed(s);torch.manual_seed(s)\n    if torch.cuda.is_available():torch.cuda.manual_seed_all(s)\n    if hasattr(torch.backends,'cudnn'):torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True\n\n\ndef met(r):\n    def g(*ns):\n        for n in ns:\n            if n in r:return float(r[n])\n        raise KeyError(ns[0])\n    return {'gauc':g('GAUC','gauc'),'ndcg5':g('nDCG@5','ndcg5','NDCG@5'),'primary':g('primary','PRIMARY')}\n\n\ndef sc(v):\n    try:return int(v)\n    except:return v\n\n\ndef day_number(values):\n    out=[]\n    for v in values:\n        s=str(int(v));y=int(s[:4]) if len(s)>=8 else 2022;m=int(s[4:6]) if len(s)>=8 else 1;d=int(s[6:8]) if len(s)>=8 else int(v)%100;out.append(y*372+m*31+d)\n    return np.asarray(out,np.float32)\n\n\ndef load(dd):\n    if (dd/'train.npz').exists() and (dd/'val.npz').exists():\n        with np.load(dd/'train.npz',allow_pickle=False) as z:xt=np.asarray(z['X'],np.int64);yt=np.asarray(z['y'],np.float32).reshape(-1);ut=np.asarray(z['user']).reshape(-1);dates=np.asarray(z['date']).reshape(-1);dims=np.asarray(z['field_dims'],np.int64)\n        with np.load(dd/'val.npz',allow_pickle=False) as z:xv=np.asarray(z['X'],np.int64);yv=np.asarray(z['y'],np.float32).reshape(-1);uv=np.asarray(z['user']).reshape(-1)\n        vv=xv[:,1]-int(dims[0]);total=max(int(dims.sum()),int(max(xt.max(initial=0),xv.max(initial=0)))+1);return xt,yt,ut,dates,xv,yv,uv,vv,total,True\n    def read(path):\n        rows=[];y=[];us=[];vs=[];dates=[]\n        with open(path,newline='',encoding='utf-8') as f:\n            rd=csv.DictReader(f);ha='author_id' in (rd.fieldnames or [])\n            for r in rd:\n                u=r['user_id'];v=r['video_id'];rows.append((u,v,r['author_id'] if ha else '__author__',r.get('tab','0'),float(r.get('duration_ms','0') or 0)));y.append(float(r['long_view']));us.append(sc(u));vs.append(sc(v));dates.append(int(float(r.get('date','0') or 0)))\n        return rows,np.asarray(y,np.float32),np.asarray(us),np.asarray(vs),np.asarray(dates)\n    tr,yt,ut,_,dates=read(dd/'train.csv');va,yv,uv,vv,_=read(dd/'val.csv');q=np.quantile([r[4] for r in tr],np.linspace(.1,.9,9));A=[];B=[];off=0;dims=[]\n    for j in range(4):\n        mp={}\n        for r in tr:\n            if r[j] not in mp:mp[r[j]]=len(mp)+1\n        d=len(mp)+1;A.append(np.asarray([mp.get(r[j],0)+off for r in tr]));B.append(np.asarray([mp.get(r[j],0)+off for r in va]));dims.append(d);off+=d\n    A.append(np.searchsorted(q,[r[4] for r in tr])+off);B.append(np.searchsorted(q,[r[4] for r in va])+off);dims.append(10);return np.column_stack(A).astype(np.int64),yt,ut,dates,np.column_stack(B).astype(np.int64),yv,uv,vv,int(sum(dims)),False\n\n\nclass BilinearFM(nn.Module):\n    def __init__(self,total,fields,bias):\n        super().__init__();k=16;self.e=nn.Embedding(total,k);self.l=nn.Embedding(total,1);self.transforms=nn.Parameter(torch.stack([torch.eye(k) for _ in range(fields)]));self.bias=nn.Parameter(torch.tensor(bias,dtype=torch.float32));self.norm=nn.LayerNorm(k);nn.init.normal_(self.e.weight,std=.01);nn.init.zeros_(self.l.weight)\n    def forward(self,x):\n        e=F.dropout(self.norm(self.e(x)),.12,self.training);z=torch.einsum('bfk,fkl->bfl',e,self.transforms);inter=torch.zeros(len(x),device=x.device)\n        for i in range(e.shape[1]):\n            for j in range(i+1,e.shape[1]):inter=inter+(z[:,i]*e[:,j]).sum(1)\n        return self.l(x).sum(1).squeeze(1)+inter+self.bias\n\n\ndef pred(m,x,d):\n    m.eval();z=[]\n    with torch.no_grad():\n        for s in range(0,len(x),131072):z.append(m(torch.as_tensor(x[s:s+131072],dtype=torch.long,device=d)).cpu().numpy())\n    return np.concatenate(z).astype(np.float64)\n\n\ndef main():\n    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',required=True);ap.add_argument('--out-dir',required=True);ap.add_argument('--seed',type=int,default=42);a=ap.parse_args();seed_all(a.seed);od=Path(a.out_dir);od.mkdir(parents=True,exist_ok=True)\n    xt,yt,ut,dates,xv,yv,uv,vv,total,fast=load(Path(a.data_dir));ev=__import__('data.official.evaluate',fromlist=['evaluate']).evaluate if fast else __import__('harness.evaluate_provisional',fromlist=['evaluate']).evaluate;d=torch.device('cuda' if torch.cuda.is_available() else 'cpu');dn=day_number(dates);weights=np.exp2(-(dn.max()-dn)/7.0).astype(np.float32);weights/=weights.mean();p=float(np.clip(np.average(yt,weights=weights),1e-5,1-1e-5));m=BilinearFM(total,xt.shape[1],math.log(p/(1-p))).to(d);opt=torch.optim.AdamW(m.parameters(),lr=.0025,weight_decay=5e-5);sch=torch.optim.lr_scheduler.MultiStepLR(opt,[3,6,9,12],gamma=.4)\n    epochs=14;smoke=os.environ.get('SMOKE_EPOCHS');epochs=min(epochs,max(1,int(smoke))) if smoke is not None else epochs;rng=np.random.default_rng(a.seed+911);best=None;bm=None;be=0;hist=[]\n    for ep in range(epochs):\n        m.train();order=rng.permutation(len(xt))\n        for s in range(0,len(order),49152):\n            ix=order[s:s+49152];xb=torch.as_tensor(xt[ix],dtype=torch.long,device=d);yb=torch.as_tensor(yt[ix],dtype=torch.float32,device=d);wb=torch.as_tensor(weights[ix],dtype=torch.float32,device=d);opt.zero_grad(set_to_none=True);loss=(F.binary_cross_entropy_with_logits(m(xb),yb,reduction='none')*wb).sum()/wb.sum();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()\n        sch.step();pr=pred(m,xv,d);mm=met(ev(uv,yv,pr));hist.append({'epoch':ep+1,**mm})\n        if bm is None or mm['primary']>bm['primary']:best=pr.copy();bm=mm;be=ep+1\n    with open(od/'predictions.csv','w',newline='',encoding='utf-8') as f:\n        w=csv.writer(f);w.writerow(['row_id','user_id','video_id','score'])\n        for i,(u,v,s) in enumerate(zip(uv,vv,best)):w.writerow([i,u.item() if isinstance(u,np.generic) else u,v.item() if isinstance(v,np.generic) else v,format(float(s),'.12g')])\n    with open(od/'metrics.json','w',encoding='utf-8') as f:json.dump({**bm,'history':hist,'family':'recency_weighted_bilinear_fm','best_epoch':be,'seed':a.seed,'half_life_days':7},f,sort_keys=True,indent=2)\n    with open(od/'progress.log','a',encoding='utf-8') as f:f.write(json.dumps({'family':'recency_weighted_bilinear_fm','seed':a.seed,**bm},sort_keys=True)+'\\n')\n\n\nif __name__=='__main__':main()\n","config":{"embedding_dim":16,"half_life_days":7,"bilinear_interactions":true,"weight_decay":5e-05},"seed":3069}],"blend":{"weights":"equal","aggregations":[{"method":"rank_average","scope":"per_user"},{"method":"rank_average","scope":"global"}]}}}
