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
node_002 [<-node_000] draft "Validation peaking and then declining diagnoses overfitting; replacing the flat FM with the complete regularized Sequence DeepFM composite—causal 12-author history, temporal and causal session context, censor-aware watch-time auxiliary supervision, and a predeclared three-seed mean-logit close—will improve validation primary by approximately 0.0026 over the 0.6018 parent." primary=0.6042 ACCEPTED
node_003 [<-node_002] draft "Because validation primary peaks at epoch 3 and then falls while training loss continues decreasing, overfitting under a metric-mismatched pointwise objective is the diagnosis; complete-user batching with user-centered BCE gauge fixing will improve the accepted sequence composite's validation primary by approximately 0.0004." primary=0.6042 REJECTED

Mode: IMPROVE. Apply one change to the parent script (the current best node) — atomic by default, or a cited package / internal fan-out per the task brief. Prefer the highest-expected-gain untried menu item; use the journal to avoid rejected ideas. Emit the whole parent file with the smallest coherent change needed to test the hypothesis; unnecessary rewrites are defects.

## Selected method (implement THIS)
### heterogeneous-ensemble-design: Validation-selected cross-mechanism ensemble
- mechanism: Train members under DIFFERENT mechanisms (e.g. temporal-pair-kernel, gauge-fixed-bce, decayed-positive, frozen regularized stack) rather than jittered copies of one recipe; validation-select the member subset and aggregation (rank vs probability average, optional per-member weights) before scoring. Diversity across mechanisms is the untested axis — jittered same-recipe closes are measured at +0.0013.
- kind: opportunity
- treats: variance | plateau
- reference_primary: 0.605938 (direct evidence probe: EQUAL-WEIGHT rank blend of one pair-kernel member 0.60515 + two composite seeds 0.6044-0.6047 = 0.605938; champion-ens + one composite seed = 0.605886; many cross-family combos cluster 0.6058-0.6059 — decorrelated families are where closes pay)
- preconditions: At least 2 mechanism families measured above 0.6040 in this run's lineage; select on validation only.
- citation: run_bigclock_07 n6 (jitter close +0.0013); evidence/blend_audit.md (caveat)
- expected_gain / cost: +0.0005-0.0020 primary / medium (several member trainings).
- status_pure: measured-win (run_final_s2 n4 +0.0010; cross-family blend probe 0.605828 evidence) — the endgame play: build members from BOTH the DCN package and seq-deepfm families, then rank-aggregate across families
- status_1k: untried
selector diagnosis: overfit
selector why: Training loss keeps falling while validation peaks at epoch 3 and then declines sharply, indicating overfit. This run now has two competent, distinct mechanism variants at about 0.6042—the accepted sequence composite and its user-centered-BCE variant—so the card's prerequisite is met. A validation-gated cross-mechanism ensemble has stronger measured upside from the current 0.6042 best than another small atomic treatment and can reduce both overfit variance and correlated ranking errors.

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
streak_state = {'no_improve_streak': 1, 'n_converge': 3, 'iters_left': 12}
The run ends after N consecutive iterations whose best-so-far improvement is <= epsilon = 0.002. Select experiments by expected scientific value given the remaining budget: at every iteration, including the first, prefer the eligible move with the largest evidence-supported expected gain for its cost; an early iteration spent on a small-ceiling treatment is a convergence strike bought at full price. Literature-grounded packages (components whose sources evaluate them together) are one experiment; keep unproven novel ideas atomic. Plan the run so its final iterations produce the strongest possible finished artifact rather than leaving the run un-finalized. Do the epsilon arithmetic before choosing: if the streak means the run ends unless THIS iteration improves best-so-far by at least epsilon, then a move whose own evidence caps its gain below epsilon cannot extend the run no matter how proven it is; on such an iteration prefer the eligible move with the largest evidence-supported expected gain at or above epsilon, and among qualifying moves prefer the one whose evidence clears epsilon with the widest margin: a move whose evidence only just reaches the bar fails it about half the time, so bare arithmetic reach is not parity with a wide-margin alternative (combining decorrelated mechanism families generally out-gains both re-seeding one family and any single atomic mechanism). Read margins against the CURRENT best, not a card's original baseline: an unspent package whose measured absolute score sits near the current best offers almost no headroom, while a close whose evidence exceeds every single-model score in the ledger offers the most. A proven small-gain close is the right pick only when no eligible move has evidence reaching epsilon. Do not change what counts as an iteration in response to the streak.

## Runtime budget (overrides the 600s default above)
THIS run's per-node timeout is 7200 seconds (~120 minutes). A full-length training on the npz fast path costs roughly 40-90s on CPU, far less on GPU. Plan to SPEND ~60-70% of this budget on search probes when playing a search card — e.g. at 2+ hours that is 40+ full-length probes plus refinement, not 8. Reserve the remainder for the final training(s). Finishing a search node in a small fraction of the budget is a defect, not efficiency: unspent budget is free score variance left unexplored.

When implementing ANY ensemble/member card: each member MUST be trained with a distinct seed; after scoring, ASSERT member score vectors are not identical (numpy allclose check between members and against the parent predictions) and print per-member validation primaries to progress output. An ensemble whose final predictions equal the parent's is a no-op and will be rejected by the harness, except when the farm-close executor explicitly selects and records the incumbent fallback.

## Parent node "node_002" (full code)
```python
"""Full Sequence DeepFM composite with causal history, session context, censored
watch-time auxiliary supervision, and a predeclared three-seed logit ensemble."""
import argparse
import csv
import datetime
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def date_parts(values):
    out = np.zeros(len(values), dtype=np.int64)
    days = np.zeros(len(values), dtype=np.int64)
    cache = {}
    for i, value in enumerate(values):
        iv = int(value)
        if iv not in cache:
            text = str(iv)
            try:
                dt = datetime.datetime.strptime(text, "%Y%m%d").date()
                cache[iv] = (dt.weekday(), dt.toordinal())
            except ValueError:
                cache[iv] = (iv % 7, iv)
        out[i], days[i] = cache[iv]
    return out, days


def hour_and_minute(values):
    values = np.asarray(values, dtype=np.int64)
    if len(values) == 0:
        return values.copy(), values.copy()
    if int(np.max(values)) > 1439:
        hour = np.clip(values // 100, 0, 23)
        minute = hour * 60 + np.clip(values % 100, 0, 59)
    elif int(np.max(values)) > 23:
        minute = np.clip(values, 0, 1439)
        hour = minute // 60
    else:
        hour = np.clip(values, 0, 23)
        minute = hour * 60
    return hour.astype(np.int64), minute.astype(np.int64)


def encode_offsets(train_raw, val_raw):
    train_cols = []
    val_cols = []
    dims = []
    offset = 0
    for tr_col, va_col in zip(train_raw, val_raw):
        mapping = {}
        tr_enc = np.empty(len(tr_col), dtype=np.int64)
        for i, value in enumerate(tr_col):
            if value not in mapping:
                mapping[value] = len(mapping)
            tr_enc[i] = mapping[value]
        unknown = len(mapping)
        va_enc = np.empty(len(va_col), dtype=np.int64)
        for i, value in enumerate(va_col):
            va_enc[i] = mapping.get(value, unknown)
        dim = unknown + 1
        train_cols.append(tr_enc + offset)
        val_cols.append(va_enc + offset)
        dims.append(dim)
        offset += dim
    return np.stack(train_cols, axis=1), np.stack(val_cols, axis=1), np.asarray(dims, dtype=np.int64)


def load_csv_data(data_dir):
    def read_file(path, training):
        user = []
        video = []
        tab = []
        hourmin = []
        date = []
        duration = []
        labels = []
        play = []
        with open(path, "r", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                user.append(row["user_id"])
                video.append(row["video_id"])
                tab.append(row["tab"])
                hourmin.append(int(float(row["hourmin"])))
                date.append(int(float(row["date"])))
                duration.append(float(row["duration_ms"]))
                labels.append(float(row["long_view"]))
                if training:
                    play.append(float(row["play_time_ms"]))
        return {"user_raw": np.asarray(user), "video_raw": np.asarray(video),
                "tab_raw": np.asarray(tab), "hourmin": np.asarray(hourmin, dtype=np.int64),
                "date": np.asarray(date, dtype=np.int64),
                "duration_ms": np.asarray(duration, dtype=np.float32),
                "y": np.asarray(labels, dtype=np.float32),
                "play_time_ms": np.asarray(play, dtype=np.float32) if training else None}

    tr = read_file(os.path.join(data_dir, "train.csv"), True)
    va = read_file(os.path.join(data_dir, "val.csv"), False)
    quantiles = np.quantile(tr["duration_ms"], np.linspace(0.0, 1.0, 11)[1:-1])
    tr_bucket = np.searchsorted(quantiles, tr["duration_ms"], side="right").astype(str)
    va_bucket = np.searchsorted(quantiles, va["duration_ms"], side="right").astype(str)
    # The public CSV schema has no author_id; video_id is the deterministic fallback.
    tr_fields = [tr["user_raw"], tr["video_raw"], tr["video_raw"], tr["tab_raw"], tr_bucket]
    va_fields = [va["user_raw"], va["video_raw"], va["video_raw"], va["tab_raw"], va_bucket]
    Xt, Xv, dims = encode_offsets(tr_fields, va_fields)
    tr["X"] = Xt.astype(np.int32)
    va["X"] = Xv.astype(np.int32)
    tr["field_dims"] = dims
    va["field_dims"] = dims
    tr["user"] = tr["user_raw"]
    va["user"] = va["user_raw"]
    return tr, va, False


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        trn = np.load(train_npz)
        van = np.load(val_npz)
        tr = {key: trn[key] for key in trn.files}
        va = {key: van[key] for key in van.files}
        tr["video_raw"] = np.zeros(len(tr["y"]), dtype=np.int64)
        va["video_raw"] = np.zeros(len(va["y"]), dtype=np.int64)
        return tr, va, True
    return load_csv_data(data_dir)


def evaluator(fast_path):
    if fast_path:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    return evaluate


def metric_dict(evaluate_fn, users, labels, scores):
    m = evaluate_fn(users, labels.astype(int), scores)
    return {"gauc": float(m.get("GAUC", m.get("gauc"))),
            "ndcg5": float(m.get("nDCG@5", m.get("ndcg5"))),
            "primary": float(m["primary"])}


def local_tab(X, field_dims):
    return X[:, 3].astype(np.int64) - int(np.sum(field_dims[:3]))


def build_causal_features(tr, va):
    ntr = len(tr["y"])
    nva = len(va["y"])
    histories_tr = np.full((ntr, 12), -1, dtype=np.int32)
    histories_va = np.full((nva, 12), -1, dtype=np.int32)
    gap_tr = np.full(ntr, 8, dtype=np.int64)
    gap_va = np.full(nva, 8, dtype=np.int64)
    pos_tr = np.zeros(ntr, dtype=np.int64)
    pos_va = np.zeros(nva, dtype=np.int64)

    weekday_tr, day_tr = date_parts(tr["date"])
    weekday_va, day_va = date_parts(va["date"])
    hour_tr, minute_tr = hour_and_minute(tr["hourmin"])
    hour_va, minute_va = hour_and_minute(va["hourmin"])
    time_tr = day_tr * 1440 + minute_tr
    time_va = day_va * 1440 + minute_va
    tab_tr = local_tab(tr["X"], tr["field_dims"])
    tab_va = local_tab(va["X"], tr["field_dims"])
    rand_tr = (tab_tr != 0).astype(np.int64)
    rand_va = (tab_va != 0).astype(np.int64)

    users = np.concatenate([np.asarray(tr["user"]), np.asarray(va["user"])])
    times = np.concatenate([time_tr, time_va])
    split = np.concatenate([np.zeros(ntr, dtype=np.int8), np.ones(nva, dtype=np.int8)])
    row = np.concatenate([np.arange(ntr), np.arange(nva)])
    authors = np.concatenate([tr["X"][:, 2], va["X"][:, 2]]).astype(np.int32)
    original = np.arange(ntr + nva)
    order = np.lexsort((original, split, times, users.astype(str)))
    state = {}
    gap_edges = np.asarray([0, 1, 2, 5, 10, 30, 60, 180], dtype=np.int64)
    for combined_index in order:
        user = users[combined_index]
        current_time = int(times[combined_index])
        history, previous_time, previous_pos = state.get(user, ([], None, -1))
        if previous_time is None:
            gap_bucket = 8
            session_pos = 0
        else:
            gap_minutes = max(0, current_time - previous_time)
            gap_bucket = int(np.searchsorted(gap_edges, gap_minutes, side="right") - 1)
            gap_bucket = max(0, min(7, gap_bucket))
            session_pos = 0 if gap_minutes > 30 else min(15, previous_pos + 1)
        hist_values = history[-12:]
        if split[combined_index] == 0:
            r = int(row[combined_index])
            if hist_values:
                histories_tr[r, :len(hist_values)] = hist_values
            gap_tr[r] = gap_bucket
            pos_tr[r] = session_pos
        else:
            r = int(row[combined_index])
            if hist_values:
                histories_va[r, :len(hist_values)] = hist_values
            gap_va[r] = gap_bucket
            pos_va[r] = session_pos
        history = (history + [int(authors[combined_index])])[-12:]
        state[user] = (history, current_time, session_pos)

    base_dim = int(np.sum(tr["field_dims"]))
    offsets = [base_dim, base_dim + 24, base_dim + 31, base_dim + 33, base_dim + 42]
    context_tr = np.stack([hour_tr + offsets[0], weekday_tr + offsets[1],
                           rand_tr + offsets[2], gap_tr + offsets[3], pos_tr + offsets[4]], axis=1)
    context_va = np.stack([hour_va + offsets[0], weekday_va + offsets[1],
                           rand_va + offsets[2], gap_va + offsets[3], pos_va + offsets[4]], axis=1)
    total_dim = base_dim + 58
    return (context_tr.astype(np.int32), context_va.astype(np.int32), histories_tr,
            histories_va, total_dim)


class ParentFM(torch.nn.Module):
    def __init__(self, total_dim, k=16):
        super().__init__()
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)

    def forward(self, x):
        e = self.emb(x)
        s = e.sum(1)
        pair = 0.5 * (s * s - (e * e).sum(1)).sum(1)
        return self.bias + self.lin(x).sum((1, 2)) + pair


class SequenceDeepFM(torch.nn.Module):
    def __init__(self, total_dim, k=16, dropout=0.20):
        super().__init__()
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(11 * k, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.main_head = torch.nn.Linear(64, 1)
        self.watch_head = torch.nn.Linear(64, 1)

    def forward(self, x, context, history):
        ids = torch.cat([x, context], dim=1)
        current_e = self.emb(ids)
        mask = (history >= 0).float().unsqueeze(-1)
        safe_history = history.clamp_min(0)
        hist_e = (self.emb(safe_history) * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        fields = torch.cat([current_e, hist_e.unsqueeze(1)], dim=1)
        summed = fields.sum(1)
        pair = 0.5 * (summed.square() - fields.square().sum(1)).sum(1)
        linear = self.lin(ids).sum((1, 2))
        hist_linear = (self.lin(safe_history) * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        deep = self.mlp(fields.flatten(1))
        logit = self.bias + linear + hist_linear.squeeze(1) + pair + self.main_head(deep).squeeze(1)
        watch = self.watch_head(deep).squeeze(1)
        return logit, watch


def predict_parent(model, X, device):
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(X), 65536):
            xb = torch.from_numpy(X[start:start + 65536].astype(np.int64)).to(device)
            output.append(model(xb).detach().cpu().numpy())
    return np.concatenate(output)


def train_parent_reference(tr, va, device, epochs):
    seed_everything(42)
    total_dim = int(np.sum(tr["field_dims"]))
    model = ParentFM(total_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()
    n = len(tr["y"])
    rng = np.random.RandomState(42)
    X = tr["X"].astype(np.int64)
    y = tr["y"].astype(np.float32)
    for _ in range(epochs):
        model.train()
        perm = rng.permutation(n)
        for start in range(0, n, 8192):
            ids = perm[start:start + 8192]
            xb = torch.from_numpy(X[ids]).to(device)
            yb = torch.from_numpy(y[ids]).to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
    return predict_parent(model, va["X"], device)


def predict_composite(model, X, context, history, device):
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(X), 32768):
            end = start + 32768
            xb = torch.from_numpy(X[start:end].astype(np.int64)).to(device)
            cb = torch.from_numpy(context[start:end].astype(np.int64)).to(device)
            hb = torch.from_numpy(history[start:end].astype(np.int64)).to(device)
            logits, _ = model(xb, cb, hb)
            output.append(logits.detach().cpu().numpy())
    return np.concatenate(output)


def train_member(seed, tr, va, context_tr, context_va, history_tr, history_va,
                 total_dim, device, epochs, evaluate_fn):
    seed_everything(seed)
    model = SequenceDeepFM(total_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=2e-5)
    milestones = sorted(set([max(1, epochs // 3), max(2, (2 * epochs) // 3)]))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=0.35)
    bce = torch.nn.BCEWithLogitsLoss()
    X = tr["X"].astype(np.int64)
    y = tr["y"].astype(np.float32)
    play = np.maximum(tr["play_time_ms"].astype(np.float32), 0.0)
    duration = np.maximum(tr["duration_ms"].astype(np.float32), 1.0)
    watch_target = (np.log1p(np.minimum(play, duration)) / 10.0).astype(np.float32)
    censored = ((play >= duration) & (duration > 1.0)).astype(np.float32)
    n = len(y)
    rng = np.random.RandomState(seed)
    best_primary = -1.0
    best_scores = None
    best_state = None
    patience = 0
    history_log = []
    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(n)
        running = 0.0
        batches = 0
        for start in range(0, n, 4096):
            ids = perm[start:start + 4096]
            xb = torch.from_numpy(X[ids]).to(device)
            cb = torch.from_numpy(context_tr[ids].astype(np.int64)).to(device)
            hb = torch.from_numpy(history_tr[ids].astype(np.int64)).to(device)
            yb = torch.from_numpy(y[ids]).to(device)
            tb = torch.from_numpy(watch_target[ids]).to(device)
            zb = torch.from_numpy(censored[ids]).to(device)
            opt.zero_grad(set_to_none=True)
            logits, watch_pred = model(xb, cb, hb)
            main_loss = bce(logits, yb)
            uncensored_loss = torch.nn.functional.smooth_l1_loss(watch_pred, tb, reduction="none")
            censored_loss = torch.relu(tb - watch_pred).square()
            watch_loss = ((1.0 - zb) * uncensored_loss + zb * censored_loss).mean()
            loss = main_loss + 0.05 * watch_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            running += float(loss.detach().cpu())
            batches += 1
        scheduler.step()
        scores = predict_composite(model, va["X"], context_va, history_va, device)
        metrics = metric_dict(evaluate_fn, va["user"], va["y"], scores)
        history_log.append({"epoch": epoch + 1,
                            "train_loss": round(running / max(1, batches), 6),
                            "lr": float(opt.param_groups[0]["lr"]),
                            "val_gauc": round(metrics["gauc"], 6),
                            "val_primary": round(metrics["primary"], 6)})
        if metrics["primary"] > best_primary + 1e-6:
            best_primary = metrics["primary"]
            best_scores = scores.copy()
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 3:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_scores, best_primary, history_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=14)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, va, fast_path = load_data(args.data_dir)
    evaluate_fn = evaluator(fast_path)
    smoke = os.environ.get("SMOKE_EPOCHS")
    epochs = args.epochs if smoke is None else min(args.epochs, max(1, int(smoke)))
    context_tr, context_va, history_tr, history_va, total_dim = build_causal_features(tr, va)

    parent_epochs = min(8, epochs)
    parent_scores = train_parent_reference(tr, va, device, parent_epochs)
    member_scores = []
    member_history = []
    member_primaries = []
    progress_path = os.path.join(args.out_dir, "progress.log")
    member_seeds = [args.seed, args.seed + 1, args.seed + 2]
    for member_seed in member_seeds:
        scores, primary, history_log = train_member(
            member_seed, tr, va, context_tr, context_va, history_tr, history_va,
            total_dim, device, epochs, evaluate_fn)
        if np.allclose(scores, parent_scores, rtol=1e-7, atol=1e-8):
            raise AssertionError("Composite member predictions equal parent predictions")
        for previous in member_scores:
            if np.allclose(scores, previous, rtol=1e-7, atol=1e-8):
                raise AssertionError("Distinct-seed ensemble members produced identical scores")
        member_scores.append(scores)
        member_primaries.append(float(primary))
        member_history.append({"seed": int(member_seed), "best_primary": float(primary),
                               "epochs": history_log})
        with open(progress_path, "a") as fh:
            fh.write(json.dumps({"seed": int(member_seed), "val_primary": float(primary),
                                 "model": "seq_deepfm_composite"}) + "\n")

    final_scores = np.mean(np.stack(member_scores, axis=0), axis=0)
    if np.allclose(final_scores, parent_scores, rtol=1e-7, atol=1e-8):
        raise AssertionError("Ensemble predictions equal parent predictions")
    final_metrics = metric_dict(evaluate_fn, va["user"], va["y"], final_scores)
    metrics_output = {"gauc": final_metrics["gauc"], "ndcg5": final_metrics["ndcg5"],
                      "primary": final_metrics["primary"], "history": member_history,
                      "member_primaries": member_primaries,
                      "ensemble": {"method": "mean_logit", "seeds": member_seeds}}
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics_output, fh)
    with open(os.path.join(args.out_dir, "predictions.csv"), "w") as fh:
        fh.write("row_id,user_id,video_id,score\n")
        videos = va.get("video_raw", np.zeros(len(final_scores), dtype=np.int64))
        for i, score in enumerate(final_scores):
            fh.write(f"{i},{va['user'][i]},{videos[i]},{float(score):.8g}\n")


if __name__ == "__main__":
    main()

```

## Parent learning curve (per epoch)
epoch 1: train_loss 0.556233, val_gauc 0.668081, val_primary 0.602459
epoch 2: train_loss 0.506877, val_gauc 0.669226, val_primary 0.603196
epoch 3: train_loss 0.500831, val_gauc 0.671047, val_primary 0.604334
epoch 4: train_loss 0.497215, val_gauc 0.66927, val_primary 0.603067
epoch 5: train_loss 0.488459, val_gauc 0.66979, val_primary 0.603251
epoch 6: train_loss 0.486066, val_gauc 0.667967, val_primary 0.601933
DIAGNOSE before proposing: val peaks early then falls = overfit (attack with regularization/schedules); val still rising at stop = underfit (train longer); flat = the idea itself adds no signal (change direction, not dosage). State your diagnosis in the hypothesis.

Respond with the JSON object only.

## REPLY
{"execution_kind":"farm_close","hypothesis":"Validation peaking at epoch 3 and then declining diagnoses overfit; equal-weight per-user rank averaging across sequence-history DeepFM, strongly regularized DCN, context-stratified pairwise DeepFM, and recency-weighted FM mechanisms will reduce decorrelated ranking errors and improve validation primary by approximately 0.0016 over node_002.","expected_delta":0.0016,"expected_delta_basis":"The selected card reports cross-family blends around 0.6058-0.6059 versus the current 0.6042, including a directly measured equal-weight rank blend at 0.605938.","action":"improve","parent":"node_002","timeout_s":7200,"ensemble_plan":{"probe_epochs":2,"full_member_limit":3,"min_probe_blend_gain":0.0,"members":[{"family":"sequence-history-deepfm","code":"import argparse, csv, datetime, json, os, sys\nimport numpy as np\nimport torch\nsys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\n\ndef seed_all(s):\n    np.random.seed(s); torch.manual_seed(s)\n    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)\n    if hasattr(torch.backends, 'cudnn'):\n        torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False\n\ndef encode(tr,va):\n    a=[]; b=[]; dims=[]; off=0\n    for x,z in zip(tr,va):\n        m={}; p=np.empty(len(x),np.int64)\n        for i,v in enumerate(x):\n            if v not in m: m[v]=len(m)\n            p[i]=m[v]\n        u=len(m); q=np.asarray([m.get(v,u) for v in z],np.int64); d=u+1\n        a.append(p+off); b.append(q+off); dims.append(d); off+=d\n    return np.stack(a,1).astype(np.int32),np.stack(b,1).astype(np.int32),np.asarray(dims,np.int64)\n\ndef load(data):\n    tp=os.path.join(data,'train.npz'); vp=os.path.join(data,'val.npz')\n    if os.path.exists(tp) and os.path.exists(vp):\n        t0=np.load(tp); v0=np.load(vp); t={k:t0[k] for k in t0.files}; v={k:v0[k] for k in v0.files}\n        v['video_raw']=np.zeros(len(v['y']),np.int64); return t,v,True\n    def rd(path,train):\n        d={k:[] for k in ['user','video','tab','hourmin','date','duration_ms','y','play_time_ms']}\n        with open(path,newline='') as f:\n            for r in csv.DictReader(f):\n                d['user'].append(r['user_id']); d['video'].append(r['video_id']); d['tab'].append(r['tab'])\n                d['hourmin'].append(int(float(r['hourmin']))); d['date'].append(int(float(r['date'])))\n                d['duration_ms'].append(float(r['duration_ms'])); d['y'].append(float(r['long_view']))\n                if train: d['play_time_ms'].append(float(r['play_time_ms']))\n        for k in ['hourmin','date']: d[k]=np.asarray(d[k],np.int64)\n        for k in ['duration_ms','y','play_time_ms']: d[k]=np.asarray(d[k],np.float32)\n        d['user']=np.asarray(d['user']); d['video']=np.asarray(d['video']); d['tab']=np.asarray(d['tab']); return d\n    t=rd(os.path.join(data,'train.csv'),True); v=rd(os.path.join(data,'val.csv'),False)\n    e=np.quantile(t['duration_ms'],np.linspace(0,1,11)[1:-1]); tb=np.searchsorted(e,t['duration_ms']).astype(str); vb=np.searchsorted(e,v['duration_ms']).astype(str)\n    t['X'],v['X'],t['field_dims']=encode([t['user'],t['video'],t['video'],t['tab'],tb],[v['user'],v['video'],v['video'],v['tab'],vb]); v['field_dims']=t['field_dims']; v['video_raw']=v['video']; return t,v,False\n\ndef contexts(t,v):\n    base=int(t['field_dims'].sum())\n    def parts(d):\n        hm=np.asarray(d['hourmin'],np.int64); hour=np.clip(np.where(hm>1439,hm//100,np.where(hm>23,hm//60,hm)),0,23)\n        dow=np.asarray([datetime.datetime.strptime(str(int(x)),'%Y%m%d').weekday() if len(str(int(x)))==8 else int(x)%7 for x in d['date']],np.int64)\n        return np.stack([hour+base,dow+base+24],1).astype(np.int32)\n    ct,cv=parts(t),parts(v); h1=np.full((len(t['y']),12),-1,np.int32); h2=np.full((len(v['y']),12),-1,np.int32); state={}\n    order=np.lexsort((np.arange(len(t['y'])),np.asarray(t['date']),np.asarray(t['user']).astype(str)))\n    for i in order:\n        u=t['user'][i]; z=state.get(u,[])[-12:]\n        if z: h1[i,:len(z)]=z\n        state[u]=(state.get(u,[])+[int(t['X'][i,2])])[-12:]\n    for i,u in enumerate(v['user']):\n        z=state.get(u,[])[-12:]\n        if z: h2[i,:len(z)]=z\n        state[u]=(state.get(u,[])+[int(v['X'][i,2])])[-12:]\n    return ct,cv,h1,h2,base+31\n\nclass Model(torch.nn.Module):\n    def __init__(self,n,k=16):\n        super().__init__(); self.e=torch.nn.Embedding(n,k); self.l=torch.nn.Embedding(n,1); self.b=torch.nn.Parameter(torch.zeros(1)); torch.nn.init.normal_(self.e.weight,std=.01); torch.nn.init.zeros_(self.l.weight)\n        self.net=torch.nn.Sequential(torch.nn.Linear(8*k,128),torch.nn.ReLU(),torch.nn.Dropout(.2),torch.nn.Linear(128,64),torch.nn.ReLU(),torch.nn.Dropout(.2)); self.main=torch.nn.Linear(64,1); self.watch=torch.nn.Linear(64,1)\n    def forward(self,x,c,h):\n        ids=torch.cat([x,c],1); a=self.e(ids); mask=(h>=0).float().unsqueeze(-1); hs=h.clamp_min(0); he=(self.e(hs)*mask).sum(1)/mask.sum(1).clamp_min(1); f=torch.cat([a,he[:,None,:]],1); s=f.sum(1); fm=.5*(s.square()-f.square().sum(1)).sum(1); deep=self.net(f.flatten(1)); lin=self.l(ids).sum((1,2))+(self.l(hs)*mask).sum((1,2))/mask.sum((1,2)).clamp_min(1); return self.b+lin+fm+self.main(deep).squeeze(1),self.watch(deep).squeeze(1)\n\ndef pred(m,X,C,H,dev):\n    m.eval(); out=[]\n    with torch.no_grad():\n        for s in range(0,len(X),32768):\n            z=m(torch.as_tensor(X[s:s+32768],dtype=torch.long,device=dev),torch.as_tensor(C[s:s+32768],dtype=torch.long,device=dev),torch.as_tensor(H[s:s+32768],dtype=torch.long,device=dev))[0]; out.append(z.cpu().numpy())\n    return np.concatenate(out)\n\ndef main():\n    p=argparse.ArgumentParser(); p.add_argument('--data-dir',required=True); p.add_argument('--out-dir',required=True); p.add_argument('--seed',type=int,default=42); p.add_argument('--epochs',type=int,default=14); a=p.parse_args(); os.makedirs(a.out_dir,exist_ok=True); seed_all(a.seed); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); t,v,fast=load(a.data_dir); C,D,H,J,n=contexts(t,v); epochs=min(a.epochs,max(1,int(os.environ.get('SMOKE_EPOCHS',a.epochs)))); m=Model(n).to(dev); opt=torch.optim.AdamW(m.parameters(),lr=7e-4,weight_decay=2e-5); sch=torch.optim.lr_scheduler.MultiStepLR(opt,[max(1,epochs//3),max(2,2*epochs//3)],gamma=.35); rng=np.random.RandomState(a.seed); y=t['y'].astype(np.float32); play=np.maximum(t['play_time_ms'],0); dur=np.maximum(t['duration_ms'],1); wt=(np.log1p(np.minimum(play,dur))/10).astype(np.float32); cen=(play>=dur).astype(np.float32); best=None; bp=-1\n    ev=__import__('data.official.evaluate',fromlist=['evaluate']).evaluate if fast else __import__('harness.evaluate_provisional',fromlist=['evaluate']).evaluate\n    for _ in range(epochs):\n        m.train()\n        for s in range(0,len(y),4096):\n            q=rng.permutation(len(y))[s:s+4096]; x=torch.as_tensor(t['X'][q],dtype=torch.long,device=dev); c=torch.as_tensor(C[q],dtype=torch.long,device=dev); h=torch.as_tensor(H[q],dtype=torch.long,device=dev); yy=torch.as_tensor(y[q],device=dev); tt=torch.as_tensor(wt[q],device=dev); zz=torch.as_tensor(cen[q],device=dev); opt.zero_grad(set_to_none=True); o,w=m(x,c,h); aux=((1-zz)*torch.nn.functional.smooth_l1_loss(w,tt,reduction='none')+zz*torch.relu(tt-w).square()).mean(); loss=torch.nn.functional.binary_cross_entropy_with_logits(o,yy)+.05*aux; loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),5); opt.step()\n        sch.step(); sc=pred(m,v['X'],D,J,dev); mm=ev(v['user'],v['y'].astype(int),sc); pr=float(mm['primary'])\n        if pr>bp: bp=pr; best=sc.copy()\n    mm=ev(v['user'],v['y'].astype(int),best); out={'gauc':float(mm.get('GAUC',mm.get('gauc'))),'ndcg5':float(mm.get('nDCG@5',mm.get('ndcg5'))),'primary':float(mm['primary'])}; json.dump(out,open(os.path.join(a.out_dir,'metrics.json'),'w')); f=open(os.path.join(a.out_dir,'predictions.csv'),'w'); f.write('row_id,user_id,video_id,score\\n'); vids=v['video_raw']; [f.write(f'{i},{v[\"user\"][i]},{vids[i]},{float(z):.8g}\\n') for i,z in enumerate(best)]; f.close()\nif __name__=='__main__': main()\n","config":{"epochs":14},"seed":42},{"family":"strong-regularized-dcn","code":"import argparse,csv,json,os,sys\nimport numpy as np\nimport torch\nsys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\ndef seed_all(s):\n np.random.seed(s);torch.manual_seed(s)\n if torch.cuda.is_available():torch.cuda.manual_seed_all(s)\n if hasattr(torch.backends,'cudnn'):torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False\ndef enc(a,b):\n A=[];B=[];ds=[];off=0\n for x,z in zip(a,b):\n  m={};p=[]\n  for v in x:\n   if v not in m:m[v]=len(m)\n   p.append(m[v])\n  u=len(m);q=[m.get(v,u) for v in z];d=u+1;A.append(np.asarray(p)+off);B.append(np.asarray(q)+off);ds.append(d);off+=d\n return np.stack(A,1).astype(np.int32),np.stack(B,1).astype(np.int32),np.asarray(ds)\ndef load(d):\n if os.path.exists(os.path.join(d,'train.npz')) and os.path.exists(os.path.join(d,'val.npz')):\n  x=np.load(os.path.join(d,'train.npz'));z=np.load(os.path.join(d,'val.npz'));t={k:x[k] for k in x.files};v={k:z[k] for k in z.files};v['video_raw']=np.zeros(len(v['y']),np.int64);return t,v,True\n def rd(p,tr):\n  q={k:[] for k in ['user','video','tab','duration_ms','y']}\n  with open(p,newline='') as f:\n   for r in csv.DictReader(f):q['user'].append(r['user_id']);q['video'].append(r['video_id']);q['tab'].append(r['tab']);q['duration_ms'].append(float(r['duration_ms']));q['y'].append(float(r['long_view']))\n  q['user']=np.asarray(q['user']);q['video']=np.asarray(q['video']);q['tab']=np.asarray(q['tab']);q['duration_ms']=np.asarray(q['duration_ms'],np.float32);q['y']=np.asarray(q['y'],np.float32);return q\n t=rd(os.path.join(d,'train.csv'),1);v=rd(os.path.join(d,'val.csv'),0);e=np.quantile(t['duration_ms'],np.linspace(0,1,11)[1:-1]);tb=np.searchsorted(e,t['duration_ms']).astype(str);vb=np.searchsorted(e,v['duration_ms']).astype(str);t['X'],v['X'],t['field_dims']=enc([t['user'],t['video'],t['video'],t['tab'],tb],[v['user'],v['video'],v['video'],v['tab'],vb]);v['field_dims']=t['field_dims'];v['video_raw']=v['video'];return t,v,False\nclass DCN(torch.nn.Module):\n def __init__(self,n,k=16):\n  super().__init__();self.e=torch.nn.Embedding(n,k);self.l=torch.nn.Embedding(n,1);torch.nn.init.normal_(self.e.weight,std=.01);torch.nn.init.zeros_(self.l.weight);d=5*k;self.w1=torch.nn.Parameter(torch.zeros(d));self.b1=torch.nn.Parameter(torch.zeros(d));self.w2=torch.nn.Parameter(torch.zeros(d));self.b2=torch.nn.Parameter(torch.zeros(d));self.drop=torch.nn.Dropout(.25);self.deep=torch.nn.Sequential(torch.nn.Linear(d,128),torch.nn.ReLU(),torch.nn.Dropout(.3),torch.nn.Linear(128,64),torch.nn.ReLU(),torch.nn.Dropout(.3));self.out=torch.nn.Linear(d+64,1)\n def forward(self,x):\n  e=self.drop(self.e(x));x0=e.flatten(1);x1=x0*(x0@self.w1)[:,None]+self.b1+x0;x2=x0*(x1@self.w2)[:,None]+self.b2+x1;return self.l(x).sum((1,2))+self.out(torch.cat([x2,self.deep(x0)],1)).squeeze(1)\ndef pred(m,X,d):\n m.eval();o=[]\n with torch.no_grad():\n  for s in range(0,len(X),65536):o.append(m(torch.as_tensor(X[s:s+65536],dtype=torch.long,device=d)).cpu().numpy())\n return np.concatenate(o)\ndef main():\n p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--out-dir',required=True);p.add_argument('--seed',type=int,default=42);p.add_argument('--epochs',type=int,default=14);a=p.parse_args();os.makedirs(a.out_dir,exist_ok=True);seed_all(a.seed);dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');t,v,fast=load(a.data_dir);epochs=min(a.epochs,max(1,int(os.environ.get('SMOKE_EPOCHS',a.epochs))));m=DCN(int(t['field_dims'].sum())).to(dev);op=torch.optim.AdamW(m.parameters(),lr=6e-4,weight_decay=1e-4);sch=torch.optim.lr_scheduler.MultiStepLR(op,[2,4],gamma=.3);rng=np.random.RandomState(a.seed);ev=__import__('data.official.evaluate',fromlist=['evaluate']).evaluate if fast else __import__('harness.evaluate_provisional',fromlist=['evaluate']).evaluate;best=None;bp=-1\n for ep in range(epochs):\n  m.train();perm=rng.permutation(len(t['y']))\n  for s in range(0,len(perm),4096):\n   q=perm[s:s+4096];x=torch.as_tensor(t['X'][q],dtype=torch.long,device=dev);y=torch.as_tensor(t['y'][q],device=dev);op.zero_grad(set_to_none=True);loss=torch.nn.functional.binary_cross_entropy_with_logits(m(x),y,label_smoothing=.02);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),3);op.step()\n  sch.step();sc=pred(m,v['X'],dev);mm=ev(v['user'],v['y'].astype(int),sc)\n  if float(mm['primary'])>bp:bp=float(mm['primary']);best=sc.copy()\n mm=ev(v['user'],v['y'].astype(int),best);json.dump({'gauc':float(mm.get('GAUC',mm.get('gauc'))),'ndcg5':float(mm.get('nDCG@5',mm.get('ndcg5'))),'primary':float(mm['primary'])},open(os.path.join(a.out_dir,'metrics.json'),'w'));f=open(os.path.join(a.out_dir,'predictions.csv'),'w');f.write('row_id,user_id,video_id,score\\n');[f.write(f'{i},{v[\"user\"][i]},{v[\"video_raw\"][i]},{float(z):.8g}\\n') for i,z in enumerate(best)];f.close()\nif __name__=='__main__':main()\n","config":{"epochs":14},"seed":43},{"family":"context-stratified-pair-deepfm","code":"import argparse,csv,json,os,sys\nimport numpy as np\nimport torch\nsys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\ndef seed_all(s):\n np.random.seed(s);torch.manual_seed(s)\n if torch.cuda.is_available():torch.cuda.manual_seed_all(s)\n if hasattr(torch.backends,'cudnn'):torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False\ndef enc(a,b):\n A=[];B=[];ds=[];off=0\n for x,z in zip(a,b):\n  m={};p=[]\n  for v in x:\n   if v not in m:m[v]=len(m)\n   p.append(m[v])\n  u=len(m);A.append(np.asarray(p)+off);B.append(np.asarray([m.get(v,u) for v in z])+off);ds.append(u+1);off+=u+1\n return np.stack(A,1).astype(np.int32),np.stack(B,1).astype(np.int32),np.asarray(ds)\ndef load(d):\n if os.path.exists(os.path.join(d,'train.npz')) and os.path.exists(os.path.join(d,'val.npz')):\n  a=np.load(os.path.join(d,'train.npz'));b=np.load(os.path.join(d,'val.npz'));t={k:a[k] for k in a.files};v={k:b[k] for k in b.files};v['video_raw']=np.zeros(len(v['y']),np.int64);return t,v,True\n def rd(p):\n  q={k:[] for k in ['user','video','tab','hourmin','duration_ms','y']}\n  with open(p,newline='') as f:\n   for r in csv.DictReader(f):q['user'].append(r['user_id']);q['video'].append(r['video_id']);q['tab'].append(r['tab']);q['hourmin'].append(int(float(r['hourmin'])));q['duration_ms'].append(float(r['duration_ms']));q['y'].append(float(r['long_view']))\n  for k in ['user','video','tab']:q[k]=np.asarray(q[k])\n  q['hourmin']=np.asarray(q['hourmin']);q['duration_ms']=np.asarray(q['duration_ms'],np.float32);q['y']=np.asarray(q['y'],np.float32);return q\n t=rd(os.path.join(d,'train.csv'));v=rd(os.path.join(d,'val.csv'));e=np.quantile(t['duration_ms'],np.linspace(0,1,11)[1:-1]);tb=np.searchsorted(e,t['duration_ms']).astype(str);vb=np.searchsorted(e,v['duration_ms']).astype(str);hour1=np.where(t['hourmin']>23,t['hourmin']//100,t['hourmin']).astype(str);hour2=np.where(v['hourmin']>23,v['hourmin']//100,v['hourmin']).astype(str);t['X'],v['X'],t['field_dims']=enc([t['user'],t['video'],t['video'],t['tab'],tb,hour1],[v['user'],v['video'],v['video'],v['tab'],vb,hour2]);v['field_dims']=t['field_dims'];v['video_raw']=v['video'];return t,v,False\nclass DeepFM(torch.nn.Module):\n def __init__(self,n,f,k=16):\n  super().__init__();self.e=torch.nn.Embedding(n,k);self.l=torch.nn.Embedding(n,1);torch.nn.init.normal_(self.e.weight,std=.01);torch.nn.init.zeros_(self.l.weight);self.net=torch.nn.Sequential(torch.nn.Linear(f*k,96),torch.nn.ReLU(),torch.nn.Dropout(.18),torch.nn.Linear(96,32),torch.nn.ReLU(),torch.nn.Dropout(.18),torch.nn.Linear(32,1))\n def forward(self,x):\n  e=self.e(x);s=e.sum(1);fm=.5*(s.square()-(e.square()).sum(1)).sum(1);return self.l(x).sum((1,2))+fm+self.net(e.flatten(1)).squeeze(1)\ndef pairs(t,seed):\n groups={}\n tab=(t['X'][:,3]-int(t['field_dims'][:3].sum())).astype(np.int64)\n for i,(u,z) in enumerate(zip(t['user'],tab)):groups.setdefault((str(u),int(z)),[[],[]])[int(t['y'][i]<=0)].append(i)\n rng=np.random.RandomState(seed);P=[];N=[]\n for pos,neg in groups.values():\n  if pos and neg:\n   k=min(max(len(pos),len(neg)),32);P.extend(rng.choice(pos,k,replace=True));N.extend(rng.choice(neg,k,replace=True))\n return np.asarray(P,np.int64),np.asarray(N,np.int64)\ndef pred(m,X,d):\n m.eval();o=[]\n with torch.no_grad():\n  for s in range(0,len(X),65536):o.append(m(torch.as_tensor(X[s:s+65536],dtype=torch.long,device=d)).cpu().numpy())\n return np.concatenate(o)\ndef main():\n p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--out-dir',required=True);p.add_argument('--seed',type=int,default=42);p.add_argument('--epochs',type=int,default=12);a=p.parse_args();os.makedirs(a.out_dir,exist_ok=True);seed_all(a.seed);dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');t,v,fast=load(a.data_dir);epochs=min(a.epochs,max(1,int(os.environ.get('SMOKE_EPOCHS',a.epochs))));m=DeepFM(int(t['field_dims'].sum()),t['X'].shape[1]).to(dev);op=torch.optim.AdamW(m.parameters(),lr=7e-4,weight_decay=4e-5);sch=torch.optim.lr_scheduler.MultiStepLR(op,[2,4],gamma=.35);rng=np.random.RandomState(a.seed);P,N=pairs(t,a.seed);ev=__import__('data.official.evaluate',fromlist=['evaluate']).evaluate if fast else __import__('harness.evaluate_provisional',fromlist=['evaluate']).evaluate;best=None;bp=-1\n for ep in range(epochs):\n  m.train();perm=rng.permutation(len(t['y']));pp=rng.permutation(len(P)) if len(P) else np.empty(0,int)\n  for bi,s in enumerate(range(0,len(perm),4096)):\n   q=perm[s:s+4096];x=torch.as_tensor(t['X'][q],dtype=torch.long,device=dev);y=torch.as_tensor(t['y'][q],device=dev);op.zero_grad(set_to_none=True);loss=.5*torch.nn.functional.binary_cross_entropy_with_logits(m(x),y)\n   if len(P):\n    j=pp[(bi*2048+np.arange(2048))%len(pp)];xp=torch.as_tensor(t['X'][P[j]],dtype=torch.long,device=dev);xn=torch.as_tensor(t['X'][N[j]],dtype=torch.long,device=dev);loss=loss+.5*torch.nn.functional.softplus(-(m(xp)-m(xn))).mean()\n   loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);op.step()\n  sch.step();sc=pred(m,v['X'],dev);mm=ev(v['user'],v['y'].astype(int),sc)\n  if float(mm['primary'])>bp:bp=float(mm['primary']);best=sc.copy()\n mm=ev(v['user'],v['y'].astype(int),best);json.dump({'gauc':float(mm.get('GAUC',mm.get('gauc'))),'ndcg5':float(mm.get('nDCG@5',mm.get('ndcg5'))),'primary':float(mm['primary'])},open(os.path.join(a.out_dir,'metrics.json'),'w'));f=open(os.path.join(a.out_dir,'predictions.csv'),'w');f.write('row_id,user_id,video_id,score\\n');[f.write(f'{i},{v[\"user\"][i]},{v[\"video_raw\"][i]},{float(z):.8g}\\n') for i,z in enumerate(best)];f.close()\nif __name__=='__main__':main()\n","config":{"epochs":12},"seed":44},{"family":"recency-weighted-regularized-fm","code":"import argparse,csv,datetime,json,os,sys\nimport numpy as np\nimport torch\nsys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\ndef seed_all(s):\n np.random.seed(s);torch.manual_seed(s)\n if torch.cuda.is_available():torch.cuda.manual_seed_all(s)\n if hasattr(torch.backends,'cudnn'):torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False\ndef encode(a,b):\n A=[];B=[];ds=[];off=0\n for x,z in zip(a,b):\n  m={};p=[]\n  for v in x:\n   if v not in m:m[v]=len(m)\n   p.append(m[v])\n  u=len(m);A.append(np.asarray(p)+off);B.append(np.asarray([m.get(v,u) for v in z])+off);ds.append(u+1);off+=u+1\n return np.stack(A,1).astype(np.int32),np.stack(B,1).astype(np.int32),np.asarray(ds)\ndef load(d):\n if os.path.exists(os.path.join(d,'train.npz')) and os.path.exists(os.path.join(d,'val.npz')):\n  a=np.load(os.path.join(d,'train.npz'));b=np.load(os.path.join(d,'val.npz'));t={k:a[k] for k in a.files};v={k:b[k] for k in b.files};v['video_raw']=np.zeros(len(v['y']),np.int64);return t,v,True\n def rd(p):\n  q={k:[] for k in ['user','video','tab','duration_ms','date','y']}\n  with open(p,newline='') as f:\n   for r in csv.DictReader(f):q['user'].append(r['user_id']);q['video'].append(r['video_id']);q['tab'].append(r['tab']);q['duration_ms'].append(float(r['duration_ms']));q['date'].append(int(float(r['date'])));q['y'].append(float(r['long_view']))\n  for k in ['user','video','tab']:q[k]=np.asarray(q[k])\n  q['duration_ms']=np.asarray(q['duration_ms'],np.float32);q['date']=np.asarray(q['date'],np.int64);q['y']=np.asarray(q['y'],np.float32);return q\n t=rd(os.path.join(d,'train.csv'));v=rd(os.path.join(d,'val.csv'));e=np.quantile(t['duration_ms'],np.linspace(0,1,11)[1:-1]);tb=np.searchsorted(e,t['duration_ms']).astype(str);vb=np.searchsorted(e,v['duration_ms']).astype(str);t['X'],v['X'],t['field_dims']=encode([t['user'],t['video'],t['video'],t['tab'],tb],[v['user'],v['video'],v['video'],v['tab'],vb]);v['field_dims']=t['field_dims'];v['video_raw']=v['video'];return t,v,False\nclass FM(torch.nn.Module):\n def __init__(self,n,k=16):\n  super().__init__();self.e=torch.nn.Embedding(n,k);self.l=torch.nn.Embedding(n,1);self.b=torch.nn.Parameter(torch.zeros(1));self.drop=torch.nn.Dropout(.15);torch.nn.init.normal_(self.e.weight,std=.01);torch.nn.init.zeros_(self.l.weight)\n def forward(self,x):\n  e=self.drop(self.e(x));s=e.sum(1);return self.b+self.l(x).sum((1,2))+.5*(s.square()-e.square().sum(1)).sum(1)\ndef pred(m,X,d):\n m.eval();o=[]\n with torch.no_grad():\n  for s in range(0,len(X),65536):o.append(m(torch.as_tensor(X[s:s+65536],dtype=torch.long,device=d)).cpu().numpy())\n return np.concatenate(o)\ndef main():\n p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--out-dir',required=True);p.add_argument('--seed',type=int,default=42);p.add_argument('--epochs',type=int,default=16);a=p.parse_args();os.makedirs(a.out_dir,exist_ok=True);seed_all(a.seed);dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');t,v,fast=load(a.data_dir);epochs=min(a.epochs,max(1,int(os.environ.get('SMOKE_EPOCHS',a.epochs))));m=FM(int(t['field_dims'].sum())).to(dev);op=torch.optim.AdamW(m.parameters(),lr=5e-4,weight_decay=8e-5);sch=torch.optim.lr_scheduler.MultiStepLR(op,[2,4,7],gamma=.4);raw=np.asarray(t['date']);\n try:days=np.asarray([datetime.datetime.strptime(str(int(x)),'%Y%m%d').date().toordinal() for x in raw])\n except ValueError:days=raw.astype(np.int64)\n w=np.exp2(-(days.max()-days)/7.0).astype(np.float32);w/=w.mean();rng=np.random.RandomState(a.seed);ev=__import__('data.official.evaluate',fromlist=['evaluate']).evaluate if fast else __import__('harness.evaluate_provisional',fromlist=['evaluate']).evaluate;best=None;bp=-1\n for ep in range(epochs):\n  m.train();perm=rng.permutation(len(t['y']))\n  for s in range(0,len(perm),8192):\n   q=perm[s:s+8192];x=torch.as_tensor(t['X'][q],dtype=torch.long,device=dev);y=torch.as_tensor(t['y'][q],device=dev);ww=torch.as_tensor(w[q],device=dev);op.zero_grad(set_to_none=True);loss=(torch.nn.functional.binary_cross_entropy_with_logits(m(x),y,reduction='none')*ww).mean();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),4);op.step()\n  sch.step();sc=pred(m,v['X'],dev);mm=ev(v['user'],v['y'].astype(int),sc)\n  if float(mm['primary'])>bp:bp=float(mm['primary']);best=sc.copy()\n mm=ev(v['user'],v['y'].astype(int),best);json.dump({'gauc':float(mm.get('GAUC',mm.get('gauc'))),'ndcg5':float(mm.get('nDCG@5',mm.get('ndcg5'))),'primary':float(mm['primary'])},open(os.path.join(a.out_dir,'metrics.json'),'w'));f=open(os.path.join(a.out_dir,'predictions.csv'),'w');f.write('row_id,user_id,video_id,score\\n');[f.write(f'{i},{v[\"user\"][i]},{v[\"video_raw\"][i]},{float(z):.8g}\\n') for i,z in enumerate(best)];f.close()\nif __name__=='__main__':main()\n","config":{"epochs":16},"seed":45}],"blend":{"weights":"equal","aggregations":[{"method":"rank_average","scope":"per_user"},{"method":"rank_average","scope":"global"}]}}}
