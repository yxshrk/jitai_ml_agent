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
For an IMPROVE proposal, do NOT re-emit the whole script. Replace "code" with
targeted edit blocks applied verbatim to the parent script:
 "edits": [{"search": "<exact contiguous snippet copied character-for-character
from the parent script; must occur exactly once>",
            "replace": "<the replacement text>"}, ...]
Edits are applied in order; untouched code stays byte-identical, so the parent's
debugged trainer, data pipeline, and evaluation scaffolding survive unchanged.
Copy search text EXACTLY (whitespace included) from the parent script you were
given. Use several small blocks rather than one giant block.


## Task context
Dataset: KuaiRand short-video recommendation (pure track).
Metrics: within-user GAUC, per-user nDCG@5, and their mean primary score.
Splits: train on the fixed training split and evaluate only on the fixed validation split.

Research doctrine (general methodology; you have no prior results on this benchmark):
- Screen before you bet: open with broad, cheap information-buying moves — one
  fan-out node probing many dial settings or mechanisms with short trainings and
  a full-fidelity final beats a single narrow architecture bet (random search:
  Bergstra & Bengio 2012; successive halving: Hyperband/ASHA).
- Budget arithmetic: probes are cheap relative to the node clock; spend most of
  a search node on probes, reserve the rest for the full-length final training.
- Keep your own ledger: in-run measurements on this dataset outrank literature
  priors. Compose and tune what your journal shows working; never retry a
  mechanism family your journal shows rejected twice.
- Compound cleared wins: build each change on the current champion.
- Honest telemetry: with no usable learning curve, diagnose
  insufficient-telemetry and pick a low-risk broad move; do not guess.
- Close with diversity: reserve the final iterations for an ensemble of the
  champion family — diverse members (seeds plus modest config variation)
  reduce correlated errors (Deep Ensembles, Lakshminarayanan et al. 2017).

## USER
## Prior runs (do not repeat failed openings)
(none recorded)

## Journal (one line per prior node)
node_000 [baseline] draft "baseline FM" primary=0.6018 ACCEPTED (sigma=0.0001)
node_001 [<-node_000] draft "Because validation peaks at epoch 8 while training loss continues falling, diagnosing overfit, a broad short-probe screen of regularization, recency weighting, pairwise ranking alignment, DeepFM interaction heads, and leakage-safe frequency aggregates will select a robust configuration that improves validation primary by at least 0.003." no-metric FAILED

Mode: DEBUG. The parent script failed. Preserve its approach and hypothesis; fix the failure shown in the traceback tail. Emit the corrected whole script.

## Selected method (implement THIS)
### mechanism-screen: Multi-mechanism screening matrix with short probes
- kind: opportunity
- treats: flat-signal | insufficient-telemetry
- mechanism: One fan-out node implements several small candidate mechanisms (for example a pairwise-loss term, an interaction head, recency weighting, item aggregates) as toggles, probes each alone and in a few pairs with short trainings, and promotes only the strongest combination to a full-fidelity final training. This converts uncertainty over which mechanism family fits the dataset into a cheap measured ranking.
- preconditions: Each toggle must be independently correct and leakage-safe; keep probes short and comparable; record the full probe matrix in metrics history.
- citation: Bergstra & Bengio, JMLR 2012; Hyperband/ASHA (Li et al. 2018); ablation-matrix practice in FuxiCTR/BARS benchmark literature
- expected_gain / cost: Benchmark-suite literature reports that measured screening across mechanism families avoids committing budget to a mismatched family and typically finds the dataset's dominant lever early / medium runtime, one node.
- status_pure: untried
- status_1k: untried
selector diagnosis: overfit
selector why: Validation peaks at epoch 8 (0.601838) and declines through epoch 10 while training loss trends downward, indicating overfit. However, with 15 iterations left, no prior experiments, and typical atomic gains near or below the 0.002 convergence threshold, a broad short-probe screen has greater scientific value than committing immediately to one small treatment. Screen coherent regularization, recency weighting, ranking alignment, and leakage-safe aggregates, then full-train only the strongest combination; this can identify a gain that clears epsilon while controlling small-data memorization.

## Convergence pressure
streak_state = {'no_improve_streak': 1, 'n_converge': 3, 'iters_left': 14}
The run ends after N consecutive iterations whose best-so-far improvement is <= epsilon = 0.002. Select experiments by expected scientific value given the remaining budget: at every iteration, including the first, prefer the eligible move with the largest evidence-supported expected gain for its cost; an early iteration spent on a small-ceiling treatment is a convergence strike bought at full price. Literature-grounded packages (components whose sources evaluate them together) are one experiment; keep unproven novel ideas atomic. Plan the run so its final iterations produce the strongest possible finished artifact rather than leaving the run un-finalized. Do the epsilon arithmetic before choosing: if the streak means the run ends unless THIS iteration improves best-so-far by at least epsilon, then a move whose own evidence caps its gain below epsilon cannot extend the run no matter how proven it is; on such an iteration prefer the eligible move with the largest evidence-supported expected gain at or above epsilon, and among qualifying moves prefer the one whose evidence clears epsilon with the widest margin: a move whose evidence only just reaches the bar fails it about half the time, so bare arithmetic reach is not parity with a wide-margin alternative (combining decorrelated mechanism families generally out-gains both re-seeding one family and any single atomic mechanism). Read margins against the CURRENT best, not a card's original baseline: an unspent package whose measured absolute score sits near the current best offers almost no headroom, while a close whose evidence exceeds every single-model score in the ledger offers the most. A proven small-gain close is the right pick only when no eligible move has evidence reaching epsilon. If a close was just REJECTED for a gain that did not repeat, its members were too close to the incumbent: re-rolling the same blend with new seeds is not a new experiment; the bottleneck a failed confirm reveals is MEMBER DIVERSITY, so the next node must add a NEW MECHANISM FAMILY the ledger has not yet given the blend (a measured package from another family), then close again. A dosage or regularization treatment on the existing champion does not qualify even if untried: it cannot decorrelate the next blend because it adds no new family. Strengthening means a NEW mechanism or family member: a component the champion stack ALREADY CONTAINS (check its accepted lineage) is not a strengthener, and re-applying it is a no-op, not a treatment. Do not change what counts as an iteration in response to the streak.

## Runtime budget (overrides the 600s default above)
THIS run's per-node timeout is 7200 seconds (~120 minutes). A full-length training on the npz fast path costs roughly 40-90s on CPU, far less on GPU. Plan to SPEND ~60-70% of this budget on search probes when playing a search card — e.g. at 2+ hours that is 40+ full-length probes plus refinement, not 8. Reserve the remainder for the final training(s). Finishing a search node in a small fraction of the budget is a defect, not efficiency: unspent budget is free score variance left unexplored.

When implementing ANY ensemble/member card: each member MUST be trained with a distinct seed; after scoring, ASSERT member score vectors are not identical (numpy allclose check between members and against the parent predictions) and print per-member validation primaries to progress output. An ensemble whose final predictions equal the parent's is a no-op and will be rejected by the harness, except when the farm-close executor explicitly selects and records the incumbent fallback.

## Parent node "node_001" (full code)
```python
"""Multi-mechanism screening over the official KuaiRand fast path.

Screens independently correct, leakage-safe regularization, recency weighting,
pairwise ranking, DeepFM interaction, and impression-frequency mechanisms before
full-fidelity training of the strongest robust configuration.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.official.evaluate import evaluate as official_evaluate


class ScreenFM(torch.nn.Module):
    def __init__(self, total_dim, k=16, dropout=0.0, deep_hidden=0, use_freq=False):
        super().__init__()
        self.k = int(k)
        self.dropout = float(dropout)
        self.deep_hidden = int(deep_hidden)
        self.use_freq = bool(use_freq)
        self.emb = torch.nn.Embedding(total_dim, self.k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)
        if self.deep_hidden > 0:
            self.deep1 = torch.nn.Linear(5 * self.k, self.deep_hidden)
            self.deep2 = torch.nn.Linear(self.deep_hidden, 1)
            torch.nn.init.xavier_uniform_(self.deep1.weight)
            torch.nn.init.zeros_(self.deep1.bias)
            torch.nn.init.normal_(self.deep2.weight, std=0.01)
            torch.nn.init.zeros_(self.deep2.bias)
        if self.use_freq:
            self.freq_lin = torch.nn.Linear(3, 1)
            torch.nn.init.zeros_(self.freq_lin.weight)
            torch.nn.init.zeros_(self.freq_lin.bias)

    def forward(self, x, freq=None):
        e = self.emb(x)
        if self.dropout > 0.0:
            e_used = F.dropout(e, p=self.dropout, training=self.training)
        else:
            e_used = e
        s = e_used.sum(1)
        pair = 0.5 * (s * s - (e_used * e_used).sum(1)).sum(1)
        out = self.bias + self.lin(x).sum((1, 2)) + pair
        if self.deep_hidden > 0:
            h = F.relu(self.deep1(e_used.reshape(e_used.shape[0], -1)))
            h = F.dropout(h, p=self.dropout, training=self.training)
            out = out + self.deep2(h).squeeze(1)
        if self.use_freq:
            out = out + self.freq_lin(freq).squeeze(1)
        return out


def _mapping(values):
    result = {}
    for value in values:
        if value not in result:
            result[value] = len(result) + 1
    return result


def load_csv_data(data_dir):
    train_rows = []
    train_path = os.path.join(data_dir, "train.csv")
    with open(train_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            train_rows.append((
                row["user_id"], row["video_id"], row["tab"],
                float(row["duration_ms"]), float(row["long_view"]),
                float(row["date"])
            ))

    val_rows = []
    val_path = os.path.join(data_dir, "val.csv")
    with open(val_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            val_rows.append((
                row["user_id"], row["video_id"], row["tab"],
                float(row["duration_ms"]), float(row["long_view"])
            ))

    user_map = _mapping([r[0] for r in train_rows])
    video_map = _mapping([r[1] for r in train_rows])
    tab_map = _mapping([r[2] for r in train_rows])
    field_dims = np.asarray([
        len(user_map) + 1,
        len(video_map) + 1,
        1,
        len(tab_map) + 1,
        32,
    ], dtype=np.int64)
    offsets = np.concatenate([
        np.zeros(1, dtype=np.int64),
        np.cumsum(field_dims[:-1]),
    ])

    def encode(rows):
        x = np.zeros((len(rows), 5), dtype=np.int64)
        for i, row in enumerate(rows):
            x[i, 0] = user_map.get(row[0], 0)
            x[i, 1] = video_map.get(row[1], 0)
            x[i, 2] = 0
            x[i, 3] = tab_map.get(row[2], 0)
            duration = max(float(row[3]), 0.0)
            x[i, 4] = min(31, int(np.log2(1.0 + duration / 1000.0)))
        return x + offsets.reshape(1, -1)

    tr = {
        "X": encode(train_rows),
        "y": np.asarray([r[4] for r in train_rows], dtype=np.float32),
        "user": np.asarray([r[0] for r in train_rows]),
        "date": np.asarray([r[5] for r in train_rows], dtype=np.float64),
        "field_dims": field_dims,
    }
    va = {
        "X": encode(val_rows),
        "y": np.asarray([r[4] for r in val_rows], dtype=np.float32),
        "user": np.asarray([r[0] for r in val_rows]),
    }
    video_out = np.asarray([r[1] for r in val_rows])
    return tr, va, video_out


def normalize_metrics(metrics):
    return {
        "gauc": float(metrics["GAUC"] if "GAUC" in metrics else metrics["gauc"]),
        "ndcg5": float(metrics.get("nDCG@5", metrics.get("ndcg5"))),
        "primary": float(metrics["primary"]),
    }


def make_frequency_features(x_train, x_val, total_dim):
    train_parts = []
    val_parts = []
    for field in range(3):
        counts = np.bincount(
            x_train[:, field], minlength=total_dim
        ).astype(np.float32)
        train_value = np.log1p(counts[x_train[:, field]])
        val_value = np.log1p(counts[x_val[:, field]])
        mean = float(train_value.mean())
        std = float(train_value.std())
        if std < 1e-6:
            std = 1.0
        train_parts.append((train_value - mean) / std)
        val_parts.append((val_value - mean) / std)
    return (
        np.stack(train_parts, axis=1).astype(np.float32),
        np.stack(val_parts, axis=1).astype(np.float32),
    )


def make_recency_base(dates):
    dates = np.asarray(dates)
    if len(dates) == 0:
        return np.empty(0, dtype=np.float32)
    _, inverse = np.unique(dates, return_inverse=True)
    denom = max(int(inverse.max()), 1)
    return inverse.astype(np.float32) / float(denom) - 0.5


def make_pair_pool(users, labels):
    users = np.asarray(users)
    labels = np.asarray(labels)
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    starts = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1]
    ends = np.r_[starts[1:], len(order)]
    positive = []
    negative = []
    for start, end in zip(starts, ends):
        idx = order[start:end]
        pos = idx[labels[idx] > 0.5]
        neg = idx[labels[idx] <= 0.5]
        if len(pos) and len(neg):
            positive.append(int(pos[0]))
            negative.append(int(neg[0]))
    return (
        torch.tensor(positive, dtype=torch.long),
        torch.tensor(negative, dtype=torch.long),
    )


def default_config():
    return {
        "k": 16,
        "lr": 0.001,
        "weight_decay": 0.0,
        "dropout": 0.0,
        "recency": 0.0,
        "pairwise": 0.0,
        "deep_hidden": 0,
        "use_freq": False,
        "cosine": False,
    }


def build_configs(count, seed):
    base = default_config()
    configs = [dict(base)]

    singles = [
        ("weight_decay", 1e-5), ("weight_decay", 5e-5),
        ("weight_decay", 2e-4), ("dropout", 0.1),
        ("dropout", 0.25), ("dropout", 0.35),
        ("k", 8), ("k", 24), ("k", 32),
        ("recency", 0.25), ("recency", 0.5), ("recency", 1.0),
        ("pairwise", 0.05), ("pairwise", 0.1), ("pairwise", 0.2),
        ("deep_hidden", 32), ("deep_hidden", 64),
        ("use_freq", True), ("cosine", True),
        ("lr", 0.0005), ("lr", 0.002),
    ]
    for key, value in singles:
        config = dict(base)
        config[key] = value
        configs.append(config)

    paired = [
        {"weight_decay": 5e-5, "dropout": 0.1},
        {"weight_decay": 5e-5, "k": 8},
        {"weight_decay": 1e-5, "recency": 0.5},
        {"weight_decay": 5e-5, "pairwise": 0.1},
        {"weight_decay": 5e-5, "deep_hidden": 32, "dropout": 0.1},
        {"weight_decay": 5e-5, "use_freq": True},
        {"dropout": 0.1, "deep_hidden": 64},
        {"pairwise": 0.1, "recency": 0.5},
        {"pairwise": 0.1, "use_freq": True},
        {"deep_hidden": 32, "use_freq": True},
        {"recency": 0.5, "use_freq": True},
        {"cosine": True, "weight_decay": 5e-5},
    ]
    for changes in paired:
        config = dict(base)
        config.update(changes)
        configs.append(config)

    rng = np.random.RandomState(seed + 1701)
    choices = {
        "k": [8, 16, 24, 32],
        "lr": [0.0005, 0.001, 0.002],
        "weight_decay": [0.0, 1e-6, 1e-5, 5e-5, 2e-4],
        "dropout": [0.0, 0.1, 0.2, 0.35],
        "recency": [0.0, 0.25, 0.5, 1.0],
        "pairwise": [0.0, 0.05, 0.1, 0.2],
        "deep_hidden": [0, 32, 64],
        "use_freq": [False, True],
        "cosine": [False, True],
    }
    seen = {json.dumps(c, sort_keys=True) for c in configs}
    while len(configs) < count:
        config = {
            key: values[int(rng.randint(len(values)))]
            for key, values in choices.items()
        }
        signature = json.dumps(config, sort_keys=True)
        if signature not in seen:
            seen.add(signature)
            configs.append(config)
    return configs[:count]


def run_training(config, epochs, run_seed, tensors, arrays, device, eval_fn,
                 early_stop=False):
    torch.manual_seed(run_seed)
    np.random.seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)

    xt, yt, xv, zt, zv, pair_pos, pair_neg = tensors
    users_val, labels_val, recency_base = arrays
    model = ScreenFM(
        int(config["total_dim"]),
        k=int(config["k"]),
        dropout=float(config["dropout"]),
        deep_hidden=int(config["deep_hidden"]),
        use_freq=bool(config["use_freq"]),
    ).to(device)

    if float(config["weight_decay"]) > 0.0:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["lr"]),
            weight_decay=float(config["weight_decay"]),
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=float(config["lr"])
        )

    scheduler = None
    if bool(config["cosine"]):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(int(epochs), 1),
            eta_min=float(config["lr"]) * 0.1,
        )

    recency = float(config["recency"])
    if recency > 0.0:
        weights_np = np.exp(recency * recency_base).astype(np.float32)
        weights_np /= max(float(weights_np.mean()), 1e-8)
        train_weights = torch.from_numpy(weights_np)
    else:
        train_weights = None

    n = len(yt)
    batch_size = 8192
    best_primary = -1.0
    best_scores = None
    best_metrics = None
    best_epoch = 0
    patience = 0
    curve = []
    generator = torch.Generator(device="cpu")
    generator.manual_seed(run_seed + 313)

    for epoch in range(int(epochs)):
        model.train()
        permutation = torch.randperm(n, generator=generator)
        loss_sum = 0.0
        rows_seen = 0

        for start in range(0, n, batch_size):
            idx = permutation[start:start + batch_size]
            xb = xt[idx].to(device, non_blocking=True)
            yb = yt[idx].to(device, non_blocking=True)
            zb = zt[idx].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb, zb)
            point_loss = F.binary_cross_entropy_with_logits(
                logits, yb, reduction="none"
            )
            if train_weights is not None:
                wb = train_weights[idx].to(device, non_blocking=True)
                loss = (point_loss * wb).mean()
            else:
                loss = point_loss.mean()

            pair_weight = float(config["pairwise"])
            if pair_weight > 0.0 and len(pair_pos) > 0:
                pair_count = min(max(256, len(idx) // 2), len(pair_pos))
                choice = torch.randint(
                    len(pair_pos), (pair_count,), generator=generator
                )
                pos_idx = pair_pos[choice]
                neg_idx = pair_neg[choice]
                both_idx = torch.cat([pos_idx, neg_idx])
                pair_x = xt[both_idx].to(device, non_blocking=True)
                pair_z = zt[both_idx].to(device, non_blocking=True)
                pair_logits = model(pair_x, pair_z)
                pos_logits = pair_logits[:pair_count]
                neg_logits = pair_logits[pair_count:]
                ranking_loss = F.softplus(
                    -(pos_logits - neg_logits)
                ).mean()
                loss = loss + pair_weight * ranking_loss

            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(idx)
            rows_seen += len(idx)

        if scheduler is not None:
            scheduler.step()

        model.eval()
        score_parts = []
        with torch.no_grad():
            for start in range(0, len(xv), 65536):
                xb = xv[start:start + 65536].to(
                    device, non_blocking=True
                )
                zb = zv[start:start + 65536].to(
                    device, non_blocking=True
                )
                score_parts.append(model(xb, zb).detach().cpu().numpy())

        scores = np.concatenate(score_parts).astype(np.float64, copy=False)
        current = normalize_metrics(eval_fn(users_val, labels_val, scores))
        curve.append({
            "epoch": epoch + 1,
            "train_loss": round(loss_sum / max(rows_seen, 1), 6),
            "val_gauc": round(current["gauc"], 6),
            "val_ndcg5": round(current["ndcg5"], 6),
            "val_primary": round(current["primary"], 6),
        })

        if current["primary"] > best_primary + 1e-8:
            best_primary = current["primary"]
            best_scores = scores.copy()
            best_metrics = current
            best_epoch = epoch + 1
            patience = 0
        else:
            patience += 1
            if early_stop and patience >= 2:
                break

    result = {
        "primary": float(best_metrics["primary"]),
        "gauc": float(best_metrics["gauc"]),
        "ndcg5": float(best_metrics["ndcg5"]),
        "best_epoch": int(best_epoch),
        "epochs_run": len(curve),
        "curve": curve,
    }
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, best_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    os.makedirs(args.out_dir, exist_ok=True)
    fast_path = (
        os.path.exists(os.path.join(args.data_dir, "train.npz"))
        and os.path.exists(os.path.join(args.data_dir, "val.npz"))
    )

    if fast_path:
        tr = np.load(os.path.join(args.data_dir, "train.npz"))
        va = np.load(os.path.join(args.data_dir, "val.npz"))
        video_out = (
            np.asarray(va["video"])
            if "video" in va.files
            else np.zeros(len(va["y"]), dtype=np.int64)
        )
        eval_fn = official_evaluate
    else:
        tr, va, video_out = load_csv_data(args.data_dir)
        from harness.evaluate_provisional import evaluate as provisional_evaluate
        eval_fn = provisional_evaluate

    field_dims = np.asarray(tr["field_dims"], dtype=np.int64)
    total_dim = int(field_dims.sum())
    x_train_np = np.asarray(tr["X"], dtype=np.int64)
    x_val_np = np.asarray(va["X"], dtype=np.int64)

    if fast_path:
        offsets = np.concatenate([
            np.zeros(1, dtype=np.int64),
            np.cumsum(field_dims[:-1]),
        ])
        x_train_np = x_train_np + offsets.reshape(1, -1)
        x_val_np = x_val_np + offsets.reshape(1, -1)

    y_train_np = np.asarray(tr["y"], dtype=np.float32)
    y_val_np = np.asarray(va["y"], dtype=np.float32).astype(int)
    users_train = np.asarray(tr["user"])
    users_val = np.asarray(va["user"])

    z_train_np, z_val_np = make_frequency_features(
        x_train_np, x_val_np, total_dim
    )
    recency_base = make_recency_base(np.asarray(tr["date"]))
    pair_pos, pair_neg = make_pair_pool(users_train, y_train_np)

    xt = torch.from_numpy(x_train_np)
    yt = torch.from_numpy(y_train_np)
    xv = torch.from_numpy(x_val_np)
    zt = torch.from_numpy(z_train_np)
    zv = torch.from_numpy(z_val_np)
    tensors = (xt, yt, xv, zt, zv, pair_pos, pair_neg)
    arrays = (users_val, y_val_np, recency_base)

    smoke_value = os.environ.get("SMOKE_EPOCHS")
    smoke_cap = int(smoke_value) if smoke_value is not None else None

    def capped(value):
        if smoke_cap is None:
            return max(1, int(value))
        return max(1, min(int(value), smoke_cap))

    if smoke_cap is not None:
        initial_count = 8
        refine_count = 2
        confirm_count = 1
    elif device.type == "cuda":
        initial_count = 192
        refine_count = 48
        confirm_count = 16
    else:
        initial_count = 64
        refine_count = 20
        confirm_count = 8

    probe_epochs = capped(min(6, args.epochs))
    full_epochs = capped(args.epochs)
    configs = build_configs(initial_count, args.seed)
    for config in configs:
        config["total_dim"] = total_dim

    history = []
    initial_results = []
    for index, config in enumerate(configs):
        result, _ = run_training(
            config, probe_epochs, args.seed, tensors,
            arrays, device, eval_fn, early_stop=False
        )
        record = {
            "stage": "probe",
            "probe_id": index,
            "seed": args.seed,
            "config": {
                k: v for k, v in config.items() if k != "total_dim"
            },
            **result,
        }
        history.append(record)
        initial_results.append((result["primary"], index, config))

    initial_results.sort(key=lambda item: (-item[0], item[1]))
    finalists = initial_results[:min(refine_count, len(initial_results))]
    refined_results = []

    for rank, (_, original_index, config) in enumerate(finalists):
        result, _ = run_training(
            config, full_epochs, args.seed, tensors,
            arrays, device, eval_fn, early_stop=False
        )
        record = {
            "stage": "refine",
            "probe_id": original_index,
            "refine_rank": rank,
            "seed": args.seed,
            "config": {
                k: v for k, v in config.items() if k != "total_dim"
            },
            **result,
        }
        history.append(record)
        refined_results.append((
            result["primary"], original_index, config, result
        ))

    refined_results.sort(key=lambda item: (-item[0], item[1]))
    confirmation_pool = refined_results[
        :min(confirm_count, len(refined_results))
    ]
    robust_results = []

    for rank, (
        refined_primary, original_index, config, refined_result
    ) in enumerate(confirmation_pool):
        confirm_seed = args.seed + 10000 + original_index
        result, _ = run_training(
            config, full_epochs, confirm_seed, tensors,
            arrays, device, eval_fn, early_stop=False
        )
        robust_primary = 0.5 * (refined_primary + result["primary"])
        record = {
            "stage": "confirm",
            "probe_id": original_index,
            "confirm_rank": rank,
            "seed": confirm_seed,
            "config": {
                k: v for k, v in config.items() if k != "total_dim"
            },
            "robust_primary": robust_primary,
            **result,
        }
        history.append(record)
        robust_results.append((
            robust_primary, refined_primary, original_index, config
        ))

    robust_results.sort(key=lambda item: (-item[0], -item[1], item[2]))
    winner_robust, _, winner_index, winner_config = robust_results[0]

    final_result, final_scores = run_training(
        winner_config, full_epochs, args.seed, tensors, arrays, device,
        eval_fn, early_stop=True
    )
    final_record = {
        "stage": "final",
        "probe_id": winner_index,
        "seed": args.seed,
        "robust_selection_primary": winner_robust,
        "config": {
            k: v for k, v in winner_config.items() if k != "total_dim"
        },
        **final_result,
    }
    history.append(final_record)

    final_metrics = normalize_metrics(
        eval_fn(users_val, y_val_np, final_scores)
    )
    metrics_payload = {
        "gauc": final_metrics["gauc"],
        "ndcg5": final_metrics["ndcg5"],
        "primary": final_metrics["primary"],
        "selected_probe_id": winner_index,
        "selected_config": {
            k: v for k, v in winner_config.items() if k != "total_dim"
        },
        "selection_robust_primary": winner_robust,
        "history": history,
    }

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics_payload, fh)

    with open(
        os.path.join(args.out_dir, "predictions.csv"), "w", newline=""
    ) as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for index, score in enumerate(final_scores):
            writer.writerow([
                index,
                users_val[index],
                video_out[index],
                format(float(score), ".6g"),
            ])


if __name__ == "__main__":
    main()

```

## Traceback tail
```
Traceback (most recent call last):
  File "/Users/pallav/mle-agent/logs/run_c9/nodes/001.py", line 650, in <module>
    main()
    ~~~~^^
  File "/Users/pallav/mle-agent/logs/run_c9/nodes/001.py", line 483, in main
    z_train_np, z_val_np = make_frequency_features(
                           ~~~~~~~~~~~~~~~~~~~~~~~^
        x_train_np, x_val_np, total_dim
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/pallav/mle-agent/logs/run_c9/nodes/001.py", line 151, in make_frequency_features
    val_value = np.log1p(counts[x_val[:, field]])
                         ~~~~~~^^^^^^^^^^^^^^^^^
IndexError: index 59960 is out of bounds for axis 0 with size 59960
```

Respond with the JSON object only.

## REPLY
{"execution_kind":"script","hypothesis":"Because validation peaks at epoch 8 while training loss continues falling, diagnosing overfit, a broad short-probe screen of regularization, recency weighting, pairwise ranking alignment, DeepFM interaction heads, and leakage-safe frequency aggregates will select a robust configuration that improves validation primary by at least 0.003.","expected_delta":0.003,"expected_delta_basis":"The selected mechanism-screen card reports that measured screening across mechanism families typically identifies the dataset's dominant lever early, while node_000's epoch-8 peak indicates actionable overfitting.","action":"debug","parent":"node_001","code":"\"\"\"Multi-mechanism screening over the official KuaiRand fast path.\n\nScreens independently correct, leakage-safe regularization, recency weighting,\npairwise ranking, DeepFM interaction, and impression-frequency mechanisms before\nfull-fidelity training of the strongest robust configuration.\n\"\"\"\nimport argparse\nimport csv\nimport json\nimport os\nimport sys\n\nimport numpy as np\nimport torch\nimport torch.nn.functional as F\n\nsys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\nfrom data.official.evaluate import evaluate as official_evaluate\n\n\nclass ScreenFM(torch.nn.Module):\n    def __init__(self, total_dim, k=16, dropout=0.0, deep_hidden=0, use_freq=False):\n        super().__init__()\n        self.k = int(k)\n        self.dropout = float(dropout)\n        self.deep_hidden = int(deep_hidden)\n        self.use_freq = bool(use_freq)\n        self.emb = torch.nn.Embedding(total_dim, self.k)\n        self.lin = torch.nn.Embedding(total_dim, 1)\n        self.bias = torch.nn.Parameter(torch.zeros(1))\n        torch.nn.init.normal_(self.emb.weight, std=0.01)\n        torch.nn.init.zeros_(self.lin.weight)\n        if self.deep_hidden > 0:\n            self.deep1 = torch.nn.Linear(5 * self.k, self.deep_hidden)\n            self.deep2 = torch.nn.Linear(self.deep_hidden, 1)\n            torch.nn.init.xavier_uniform_(self.deep1.weight)\n            torch.nn.init.zeros_(self.deep1.bias)\n            torch.nn.init.normal_(self.deep2.weight, std=0.01)\n            torch.nn.init.zeros_(self.deep2.bias)\n        if self.use_freq:\n            self.freq_lin = torch.nn.Linear(3, 1)\n            torch.nn.init.zeros_(self.freq_lin.weight)\n            torch.nn.init.zeros_(self.freq_lin.bias)\n\n    def forward(self, x, freq=None):\n        e = self.emb(x)\n        if self.dropout > 0.0:\n            e_used = F.dropout(e, p=self.dropout, training=self.training)\n        else:\n            e_used = e\n        s = e_used.sum(1)\n        pair = 0.5 * (s * s - (e_used * e_used).sum(1)).sum(1)\n        out = self.bias + self.lin(x).sum((1, 2)) + pair\n        if self.deep_hidden > 0:\n            h = F.relu(self.deep1(e_used.reshape(e_used.shape[0], -1)))\n            h = F.dropout(h, p=self.dropout, training=self.training)\n            out = out + self.deep2(h).squeeze(1)\n        if self.use_freq:\n            out = out + self.freq_lin(freq).squeeze(1)\n        return out\n\n\ndef _mapping(values):\n    result = {}\n    for value in values:\n        if value not in result:\n            result[value] = len(result) + 1\n    return result\n\n\ndef load_csv_data(data_dir):\n    train_rows = []\n    train_path = os.path.join(data_dir, \"train.csv\")\n    with open(train_path, \"r\", newline=\"\") as fh:\n        reader = csv.DictReader(fh)\n        for row in reader:\n            train_rows.append((\n                row[\"user_id\"], row[\"video_id\"], row[\"tab\"],\n                float(row[\"duration_ms\"]), float(row[\"long_view\"]),\n                float(row[\"date\"])\n            ))\n\n    val_rows = []\n    val_path = os.path.join(data_dir, \"val.csv\")\n    with open(val_path, \"r\", newline=\"\") as fh:\n        reader = csv.DictReader(fh)\n        for row in reader:\n            val_rows.append((\n                row[\"user_id\"], row[\"video_id\"], row[\"tab\"],\n                float(row[\"duration_ms\"]), float(row[\"long_view\"])\n            ))\n\n    user_map = _mapping([r[0] for r in train_rows])\n    video_map = _mapping([r[1] for r in train_rows])\n    tab_map = _mapping([r[2] for r in train_rows])\n    field_dims = np.asarray([\n        len(user_map) + 1,\n        len(video_map) + 1,\n        1,\n        len(tab_map) + 1,\n        32,\n    ], dtype=np.int64)\n    offsets = np.concatenate([\n        np.zeros(1, dtype=np.int64),\n        np.cumsum(field_dims[:-1]),\n    ])\n\n    def encode(rows):\n        x = np.zeros((len(rows), 5), dtype=np.int64)\n        for i, row in enumerate(rows):\n            x[i, 0] = user_map.get(row[0], 0)\n            x[i, 1] = video_map.get(row[1], 0)\n            x[i, 2] = 0\n            x[i, 3] = tab_map.get(row[2], 0)\n            duration = max(float(row[3]), 0.0)\n            x[i, 4] = min(31, int(np.log2(1.0 + duration / 1000.0)))\n        return x + offsets.reshape(1, -1)\n\n    tr = {\n        \"X\": encode(train_rows),\n        \"y\": np.asarray([r[4] for r in train_rows], dtype=np.float32),\n        \"user\": np.asarray([r[0] for r in train_rows]),\n        \"date\": np.asarray([r[5] for r in train_rows], dtype=np.float64),\n        \"field_dims\": field_dims,\n    }\n    va = {\n        \"X\": encode(val_rows),\n        \"y\": np.asarray([r[4] for r in val_rows], dtype=np.float32),\n        \"user\": np.asarray([r[0] for r in val_rows]),\n    }\n    video_out = np.asarray([r[1] for r in val_rows])\n    return tr, va, video_out\n\n\ndef normalize_metrics(metrics):\n    return {\n        \"gauc\": float(metrics[\"GAUC\"] if \"GAUC\" in metrics else metrics[\"gauc\"]),\n        \"ndcg5\": float(metrics.get(\"nDCG@5\", metrics.get(\"ndcg5\"))),\n        \"primary\": float(metrics[\"primary\"]),\n    }\n\n\ndef make_frequency_features(x_train, x_val, total_dim):\n    train_parts = []\n    val_parts = []\n    for field in range(3):\n        counts = np.bincount(\n            x_train[:, field], minlength=total_dim\n        ).astype(np.float32)\n        train_value = np.log1p(counts[x_train[:, field]])\n        val_value = np.log1p(counts[x_val[:, field]])\n        mean = float(train_value.mean())\n        std = float(train_value.std())\n        if std < 1e-6:\n            std = 1.0\n        train_parts.append((train_value - mean) / std)\n        val_parts.append((val_value - mean) / std)\n    return (\n        np.stack(train_parts, axis=1).astype(np.float32),\n        np.stack(val_parts, axis=1).astype(np.float32),\n    )\n\n\ndef make_recency_base(dates):\n    dates = np.asarray(dates)\n    if len(dates) == 0:\n        return np.empty(0, dtype=np.float32)\n    _, inverse = np.unique(dates, return_inverse=True)\n    denom = max(int(inverse.max()), 1)\n    return inverse.astype(np.float32) / float(denom) - 0.5\n\n\ndef make_pair_pool(users, labels):\n    users = np.asarray(users)\n    labels = np.asarray(labels)\n    order = np.argsort(users, kind=\"mergesort\")\n    sorted_users = users[order]\n    starts = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1]\n    ends = np.r_[starts[1:], len(order)]\n    positive = []\n    negative = []\n    for start, end in zip(starts, ends):\n        idx = order[start:end]\n        pos = idx[labels[idx] > 0.5]\n        neg = idx[labels[idx] <= 0.5]\n        if len(pos) and len(neg):\n            positive.append(int(pos[0]))\n            negative.append(int(neg[0]))\n    return (\n        torch.tensor(positive, dtype=torch.long),\n        torch.tensor(negative, dtype=torch.long),\n    )\n\n\ndef default_config():\n    return {\n        \"k\": 16,\n        \"lr\": 0.001,\n        \"weight_decay\": 0.0,\n        \"dropout\": 0.0,\n        \"recency\": 0.0,\n        \"pairwise\": 0.0,\n        \"deep_hidden\": 0,\n        \"use_freq\": False,\n        \"cosine\": False,\n    }\n\n\ndef build_configs(count, seed):\n    base = default_config()\n    configs = [dict(base)]\n\n    singles = [\n        (\"weight_decay\", 1e-5), (\"weight_decay\", 5e-5),\n        (\"weight_decay\", 2e-4), (\"dropout\", 0.1),\n        (\"dropout\", 0.25), (\"dropout\", 0.35),\n        (\"k\", 8), (\"k\", 24), (\"k\", 32),\n        (\"recency\", 0.25), (\"recency\", 0.5), (\"recency\", 1.0),\n        (\"pairwise\", 0.05), (\"pairwise\", 0.1), (\"pairwise\", 0.2),\n        (\"deep_hidden\", 32), (\"deep_hidden\", 64),\n        (\"use_freq\", True), (\"cosine\", True),\n        (\"lr\", 0.0005), (\"lr\", 0.002),\n    ]\n    for key, value in singles:\n        config = dict(base)\n        config[key] = value\n        configs.append(config)\n\n    paired = [\n        {\"weight_decay\": 5e-5, \"dropout\": 0.1},\n        {\"weight_decay\": 5e-5, \"k\": 8},\n        {\"weight_decay\": 1e-5, \"recency\": 0.5},\n        {\"weight_decay\": 5e-5, \"pairwise\": 0.1},\n        {\"weight_decay\": 5e-5, \"deep_hidden\": 32, \"dropout\": 0.1},\n        {\"weight_decay\": 5e-5, \"use_freq\": True},\n        {\"dropout\": 0.1, \"deep_hidden\": 64},\n        {\"pairwise\": 0.1, \"recency\": 0.5},\n        {\"pairwise\": 0.1, \"use_freq\": True},\n        {\"deep_hidden\": 32, \"use_freq\": True},\n        {\"recency\": 0.5, \"use_freq\": True},\n        {\"cosine\": True, \"weight_decay\": 5e-5},\n    ]\n    for changes in paired:\n        config = dict(base)\n        config.update(changes)\n        configs.append(config)\n\n    rng = np.random.RandomState(seed + 1701)\n    choices = {\n        \"k\": [8, 16, 24, 32],\n        \"lr\": [0.0005, 0.001, 0.002],\n        \"weight_decay\": [0.0, 1e-6, 1e-5, 5e-5, 2e-4],\n        \"dropout\": [0.0, 0.1, 0.2, 0.35],\n        \"recency\": [0.0, 0.25, 0.5, 1.0],\n        \"pairwise\": [0.0, 0.05, 0.1, 0.2],\n        \"deep_hidden\": [0, 32, 64],\n        \"use_freq\": [False, True],\n        \"cosine\": [False, True],\n    }\n    seen = {json.dumps(c, sort_keys=True) for c in configs}\n    while len(configs) < count:\n        config = {\n            key: values[int(rng.randint(len(values)))]\n            for key, values in choices.items()\n        }\n        signature = json.dumps(config, sort_keys=True)\n        if signature not in seen:\n            seen.add(signature)\n            configs.append(config)\n    return configs[:count]\n\n\ndef run_training(config, epochs, run_seed, tensors, arrays, device, eval_fn,\n                 early_stop=False):\n    torch.manual_seed(run_seed)\n    np.random.seed(run_seed)\n    if torch.cuda.is_available():\n        torch.cuda.manual_seed_all(run_seed)\n\n    xt, yt, xv, zt, zv, pair_pos, pair_neg = tensors\n    users_val, labels_val, recency_base = arrays\n    model = ScreenFM(\n        int(config[\"total_dim\"]),\n        k=int(config[\"k\"]),\n        dropout=float(config[\"dropout\"]),\n        deep_hidden=int(config[\"deep_hidden\"]),\n        use_freq=bool(config[\"use_freq\"]),\n    ).to(device)\n\n    if float(config[\"weight_decay\"]) > 0.0:\n        optimizer = torch.optim.AdamW(\n            model.parameters(),\n            lr=float(config[\"lr\"]),\n            weight_decay=float(config[\"weight_decay\"]),\n        )\n    else:\n        optimizer = torch.optim.Adam(\n            model.parameters(), lr=float(config[\"lr\"])\n        )\n\n    scheduler = None\n    if bool(config[\"cosine\"]):\n        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(\n            optimizer,\n            T_max=max(int(epochs), 1),\n            eta_min=float(config[\"lr\"]) * 0.1,\n        )\n\n    recency = float(config[\"recency\"])\n    if recency > 0.0:\n        weights_np = np.exp(recency * recency_base).astype(np.float32)\n        weights_np /= max(float(weights_np.mean()), 1e-8)\n        train_weights = torch.from_numpy(weights_np)\n    else:\n        train_weights = None\n\n    n = len(yt)\n    batch_size = 8192\n    best_primary = -1.0\n    best_scores = None\n    best_metrics = None\n    best_epoch = 0\n    patience = 0\n    curve = []\n    generator = torch.Generator(device=\"cpu\")\n    generator.manual_seed(run_seed + 313)\n\n    for epoch in range(int(epochs)):\n        model.train()\n        permutation = torch.randperm(n, generator=generator)\n        loss_sum = 0.0\n        rows_seen = 0\n\n        for start in range(0, n, batch_size):\n            idx = permutation[start:start + batch_size]\n            xb = xt[idx].to(device, non_blocking=True)\n            yb = yt[idx].to(device, non_blocking=True)\n            zb = zt[idx].to(device, non_blocking=True)\n\n            optimizer.zero_grad(set_to_none=True)\n            logits = model(xb, zb)\n            point_loss = F.binary_cross_entropy_with_logits(\n                logits, yb, reduction=\"none\"\n            )\n            if train_weights is not None:\n                wb = train_weights[idx].to(device, non_blocking=True)\n                loss = (point_loss * wb).mean()\n            else:\n                loss = point_loss.mean()\n\n            pair_weight = float(config[\"pairwise\"])\n            if pair_weight > 0.0 and len(pair_pos) > 0:\n                pair_count = min(max(256, len(idx) // 2), len(pair_pos))\n                choice = torch.randint(\n                    len(pair_pos), (pair_count,), generator=generator\n                )\n                pos_idx = pair_pos[choice]\n                neg_idx = pair_neg[choice]\n                both_idx = torch.cat([pos_idx, neg_idx])\n                pair_x = xt[both_idx].to(device, non_blocking=True)\n                pair_z = zt[both_idx].to(device, non_blocking=True)\n                pair_logits = model(pair_x, pair_z)\n                pos_logits = pair_logits[:pair_count]\n                neg_logits = pair_logits[pair_count:]\n                ranking_loss = F.softplus(\n                    -(pos_logits - neg_logits)\n                ).mean()\n                loss = loss + pair_weight * ranking_loss\n\n            loss.backward()\n            optimizer.step()\n            loss_sum += float(loss.detach().cpu()) * len(idx)\n            rows_seen += len(idx)\n\n        if scheduler is not None:\n            scheduler.step()\n\n        model.eval()\n        score_parts = []\n        with torch.no_grad():\n            for start in range(0, len(xv), 65536):\n                xb = xv[start:start + 65536].to(\n                    device, non_blocking=True\n                )\n                zb = zv[start:start + 65536].to(\n                    device, non_blocking=True\n                )\n                score_parts.append(model(xb, zb).detach().cpu().numpy())\n\n        scores = np.concatenate(score_parts).astype(np.float64, copy=False)\n        current = normalize_metrics(eval_fn(users_val, labels_val, scores))\n        curve.append({\n            \"epoch\": epoch + 1,\n            \"train_loss\": round(loss_sum / max(rows_seen, 1), 6),\n            \"val_gauc\": round(current[\"gauc\"], 6),\n            \"val_ndcg5\": round(current[\"ndcg5\"], 6),\n            \"val_primary\": round(current[\"primary\"], 6),\n        })\n\n        if current[\"primary\"] > best_primary + 1e-8:\n            best_primary = current[\"primary\"]\n            best_scores = scores.copy()\n            best_metrics = current\n            best_epoch = epoch + 1\n            patience = 0\n        else:\n            patience += 1\n            if early_stop and patience >= 2:\n                break\n\n    result = {\n        \"primary\": float(best_metrics[\"primary\"]),\n        \"gauc\": float(best_metrics[\"gauc\"]),\n        \"ndcg5\": float(best_metrics[\"ndcg5\"]),\n        \"best_epoch\": int(best_epoch),\n        \"epochs_run\": len(curve),\n        \"curve\": curve,\n    }\n    del model, optimizer\n    if device.type == \"cuda\":\n        torch.cuda.empty_cache()\n    return result, best_scores\n\n\ndef append_progress(path, record):\n    with open(path, \"a\") as fh:\n        fh.write(json.dumps(record, sort_keys=True) + \"\\n\")\n\n\ndef main():\n    parser = argparse.ArgumentParser()\n    parser.add_argument(\"--data-dir\", required=True)\n    parser.add_argument(\"--out-dir\", required=True)\n    parser.add_argument(\"--seed\", type=int, default=42)\n    parser.add_argument(\"--epochs\", type=int, default=12)\n    args = parser.parse_args()\n\n    device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n    torch.manual_seed(args.seed)\n    np.random.seed(args.seed)\n    if device.type == \"cuda\":\n        torch.cuda.manual_seed_all(args.seed)\n        torch.backends.cudnn.deterministic = True\n        torch.backends.cudnn.benchmark = False\n    try:\n        torch.use_deterministic_algorithms(True)\n    except Exception:\n        pass\n\n    os.makedirs(args.out_dir, exist_ok=True)\n    progress_path = os.path.join(args.out_dir, \"progress.log\")\n    with open(progress_path, \"w\"):\n        pass\n\n    fast_path = (\n        os.path.exists(os.path.join(args.data_dir, \"train.npz\"))\n        and os.path.exists(os.path.join(args.data_dir, \"val.npz\"))\n    )\n\n    if fast_path:\n        tr = np.load(os.path.join(args.data_dir, \"train.npz\"))\n        va = np.load(os.path.join(args.data_dir, \"val.npz\"))\n        video_out = (\n            np.asarray(va[\"video\"])\n            if \"video\" in va.files\n            else np.zeros(len(va[\"y\"]), dtype=np.int64)\n        )\n        eval_fn = official_evaluate\n    else:\n        tr, va, video_out = load_csv_data(args.data_dir)\n        from harness.evaluate_provisional import evaluate as provisional_evaluate\n        eval_fn = provisional_evaluate\n\n    field_dims = np.asarray(tr[\"field_dims\"], dtype=np.int64)\n    total_dim = int(field_dims.sum())\n    x_train_np = np.asarray(tr[\"X\"], dtype=np.int64)\n    x_val_np = np.asarray(va[\"X\"], dtype=np.int64)\n\n    if x_train_np.size and (\n        int(x_train_np.min()) < 0 or int(x_train_np.max()) >= total_dim\n    ):\n        raise ValueError(\"train X indices are outside offset-encoded field_dims\")\n    if x_val_np.size and (\n        int(x_val_np.min()) < 0 or int(x_val_np.max()) >= total_dim\n    ):\n        raise ValueError(\"validation X indices are outside offset-encoded field_dims\")\n\n    y_train_np = np.asarray(tr[\"y\"], dtype=np.float32)\n    y_val_np = np.asarray(va[\"y\"], dtype=np.float32).astype(int)\n    users_train = np.asarray(tr[\"user\"])\n    users_val = np.asarray(va[\"user\"])\n\n    z_train_np, z_val_np = make_frequency_features(\n        x_train_np, x_val_np, total_dim\n    )\n    recency_base = make_recency_base(np.asarray(tr[\"date\"]))\n    pair_pos, pair_neg = make_pair_pool(users_train, y_train_np)\n\n    xt = torch.from_numpy(x_train_np)\n    yt = torch.from_numpy(y_train_np)\n    xv = torch.from_numpy(x_val_np)\n    zt = torch.from_numpy(z_train_np)\n    zv = torch.from_numpy(z_val_np)\n    tensors = (xt, yt, xv, zt, zv, pair_pos, pair_neg)\n    arrays = (users_val, y_val_np, recency_base)\n\n    smoke_value = os.environ.get(\"SMOKE_EPOCHS\")\n    smoke_cap = int(smoke_value) if smoke_value is not None else None\n\n    def capped(value):\n        if smoke_cap is None:\n            return max(1, int(value))\n        return max(1, min(int(value), smoke_cap))\n\n    if smoke_cap is not None:\n        initial_count = 8\n        refine_count = 2\n        confirm_count = 1\n    elif device.type == \"cuda\":\n        initial_count = 192\n        refine_count = 48\n        confirm_count = 16\n    else:\n        initial_count = 64\n        refine_count = 20\n        confirm_count = 8\n\n    probe_epochs = capped(min(6, args.epochs))\n    full_epochs = capped(args.epochs)\n    configs = build_configs(initial_count, args.seed)\n    for config in configs:\n        config[\"total_dim\"] = total_dim\n\n    history = []\n    initial_results = []\n    for index, config in enumerate(configs):\n        result, _ = run_training(\n            config, probe_epochs, args.seed, tensors,\n            arrays, device, eval_fn, early_stop=False\n        )\n        record = {\n            \"stage\": \"probe\",\n            \"probe_id\": index,\n            \"seed\": args.seed,\n            \"config\": {\n                k: v for k, v in config.items() if k != \"total_dim\"\n            },\n            **result,\n        }\n        history.append(record)\n        initial_results.append((result[\"primary\"], index, config))\n        append_progress(progress_path, {\n            \"stage\": \"probe\",\n            \"probe_id\": index,\n            \"primary\": result[\"primary\"],\n            \"config\": record[\"config\"],\n        })\n\n    initial_results.sort(key=lambda item: (-item[0], item[1]))\n    finalists = initial_results[:min(refine_count, len(initial_results))]\n    refined_results = []\n\n    for rank, (_, original_index, config) in enumerate(finalists):\n        result, _ = run_training(\n            config, full_epochs, args.seed, tensors,\n            arrays, device, eval_fn, early_stop=False\n        )\n        record = {\n            \"stage\": \"refine\",\n            \"probe_id\": original_index,\n            \"refine_rank\": rank,\n            \"seed\": args.seed,\n            \"config\": {\n                k: v for k, v in config.items() if k != \"total_dim\"\n            },\n            **result,\n        }\n        history.append(record)\n        refined_results.append((\n            result[\"primary\"], original_index, config, result\n        ))\n        append_progress(progress_path, {\n            \"stage\": \"refine\",\n            \"probe_id\": original_index,\n            \"refine_rank\": rank,\n            \"primary\": result[\"primary\"],\n            \"config\": record[\"config\"],\n        })\n\n    refined_results.sort(key=lambda item: (-item[0], item[1]))\n    confirmation_pool = refined_results[\n        :min(confirm_count, len(refined_results))\n    ]\n    robust_results = []\n\n    for rank, (\n        refined_primary, original_index, config, refined_result\n    ) in enumerate(confirmation_pool):\n        confirm_seed = args.seed + 10000 + original_index\n        result, _ = run_training(\n            config, full_epochs, confirm_seed, tensors,\n            arrays, device, eval_fn, early_stop=False\n        )\n        robust_primary = 0.5 * (refined_primary + result[\"primary\"])\n        record = {\n            \"stage\": \"confirm\",\n            \"probe_id\": original_index,\n            \"confirm_rank\": rank,\n            \"seed\": confirm_seed,\n            \"config\": {\n                k: v for k, v in config.items() if k != \"total_dim\"\n            },\n            \"robust_primary\": robust_primary,\n            **result,\n        }\n        history.append(record)\n        robust_results.append((\n            robust_primary, refined_primary, original_index, config\n        ))\n        append_progress(progress_path, {\n            \"stage\": \"confirm\",\n            \"probe_id\": original_index,\n            \"confirm_rank\": rank,\n            \"primary\": result[\"primary\"],\n            \"robust_primary\": robust_primary,\n            \"config\": record[\"config\"],\n        })\n\n    robust_results.sort(key=lambda item: (-item[0], -item[1], item[2]))\n    winner_robust, _, winner_index, winner_config = robust_results[0]\n\n    final_result, final_scores = run_training(\n        winner_config, full_epochs, args.seed, tensors, arrays, device,\n        eval_fn, early_stop=True\n    )\n    final_record = {\n        \"stage\": \"final\",\n        \"probe_id\": winner_index,\n        \"seed\": args.seed,\n        \"robust_selection_primary\": winner_robust,\n        \"config\": {\n            k: v for k, v in winner_config.items() if k != \"total_dim\"\n        },\n        **final_result,\n    }\n    history.append(final_record)\n    append_progress(progress_path, {\n        \"stage\": \"final\",\n        \"probe_id\": winner_index,\n        \"primary\": final_result[\"primary\"],\n        \"robust_selection_primary\": winner_robust,\n        \"config\": final_record[\"config\"],\n    })\n\n    final_metrics = normalize_metrics(\n        eval_fn(users_val, y_val_np, final_scores)\n    )\n    metrics_payload = {\n        \"gauc\": final_metrics[\"gauc\"],\n        \"ndcg5\": final_metrics[\"ndcg5\"],\n        \"primary\": final_metrics[\"primary\"],\n        \"selected_probe_id\": winner_index,\n        \"selected_config\": {\n            k: v for k, v in winner_config.items() if k != \"total_dim\"\n        },\n        \"selection_robust_primary\": winner_robust,\n        \"history\": history,\n    }\n\n    with open(os.path.join(args.out_dir, \"metrics.json\"), \"w\") as fh:\n        json.dump(metrics_payload, fh)\n\n    with open(\n        os.path.join(args.out_dir, \"predictions.csv\"), \"w\", newline=\"\"\n    ) as fh:\n        writer = csv.writer(fh)\n        writer.writerow([\"row_id\", \"user_id\", \"video_id\", \"score\"])\n        for index, score in enumerate(final_scores):\n            writer.writerow([\n                index,\n                users_val[index],\n                video_out[index],\n                format(float(score), \".6g\"),\n            ])\n\n\nif __name__ == \"__main__\":\n    main()\n"}
