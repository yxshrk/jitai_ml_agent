import argparse
import csv
import datetime as dt
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def date_ordinal(value):
    s = str(value).strip()
    try:
        if "." in s:
            s = str(int(float(s)))
        s = s.replace("-", "")
        return dt.date(int(s[:4]), int(s[4:6]), int(s[6:8])).toordinal()
    except Exception:
        return 0


def load_fast(data_dir):
    tr = np.load(Path(data_dir) / "train.npz", allow_pickle=False)
    va = np.load(Path(data_dir) / "val.npz", allow_pickle=False)
    Xtr = np.asarray(tr["X"], dtype=np.int64)
    ytr = np.asarray(tr["y"], dtype=np.float32)
    utr = np.asarray(tr["user"])
    Xva = np.asarray(va["X"], dtype=np.int64)
    yva = np.asarray(va["y"], dtype=np.float32)
    uva = np.asarray(va["user"])
    field_dims = np.asarray(tr["field_dims"], dtype=np.int64)
    dates = np.asarray(tr["date"]) if "date" in tr.files else np.zeros(len(ytr), dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1]))).astype(np.int64)
    video_local = Xva[:, 1] - offsets[1]
    return {
        "Xtr": Xtr,
        "ytr": ytr,
        "utr": utr,
        "dates": dates,
        "Xva": Xva,
        "yva": yva,
        "uva": uva,
        "video_out": video_local,
        "field_dims": field_dims,
        "fast": True,
    }


def read_csv_rows(path):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "user_id": row["user_id"],
                "video_id": row["video_id"],
                "author_id": row.get("author_id", "__missing_author__"),
                "tab": row.get("tab", "0"),
                "duration_ms": float(row.get("duration_ms", 0) or 0),
                "date": row.get("date", "0"),
                "long_view": float(row["long_view"]),
            })
    return rows


def load_csv(data_dir):
    train_rows = read_csv_rows(Path(data_dir) / "train.csv")
    val_rows = read_csv_rows(Path(data_dir) / "val.csv")
    durations = np.asarray([r["duration_ms"] for r in train_rows], dtype=np.float64)
    quantiles = np.quantile(durations, np.linspace(0.1, 0.9, 9)) if len(durations) else np.zeros(9)
    raw_train = [
        [r["user_id"], r["video_id"], r["author_id"], r["tab"], str(int(np.searchsorted(quantiles, r["duration_ms"], side="right")))]
        for r in train_rows
    ]
    raw_val = [
        [r["user_id"], r["video_id"], r["author_id"], r["tab"], str(int(np.searchsorted(quantiles, r["duration_ms"], side="right")))]
        for r in val_rows
    ]
    maps = []
    dims = []
    for j in range(5):
        values = sorted({r[j] for r in raw_train})
        mapping = {v: i + 1 for i, v in enumerate(values)}
        maps.append(mapping)
        dims.append(len(mapping) + 1)
    offsets = np.concatenate(([0], np.cumsum(dims[:-1]))).astype(np.int64)

    def encode(raw):
        X = np.empty((len(raw), 5), dtype=np.int64)
        for i, row in enumerate(raw):
            for j in range(5):
                X[i, j] = maps[j].get(row[j], 0) + offsets[j]
        return X

    return {
        "Xtr": encode(raw_train),
        "ytr": np.asarray([r["long_view"] for r in train_rows], dtype=np.float32),
        "utr": np.asarray([r["user_id"] for r in train_rows]),
        "dates": np.asarray([r["date"] for r in train_rows]),
        "Xva": encode(raw_val),
        "yva": np.asarray([r["long_view"] for r in val_rows], dtype=np.float32),
        "uva": np.asarray([r["user_id"] for r in val_rows]),
        "video_out": np.asarray([r["video_id"] for r in val_rows]),
        "field_dims": np.asarray(dims, dtype=np.int64),
        "fast": False,
    }


def make_recency_weights(dates, half_life):
    if half_life <= 0:
        return np.ones(len(dates), dtype=np.float32)
    ords = np.asarray([date_ordinal(x) for x in dates], dtype=np.int64)
    valid = ords > 0
    if not np.any(valid):
        return np.ones(len(dates), dtype=np.float32)
    newest = int(np.max(ords[valid]))
    ages = np.maximum(0, newest - ords)
    weights = np.exp(-math.log(2.0) * ages / half_life).astype(np.float32)
    weights[~valid] = 1.0
    weights /= max(float(weights.mean()), 1e-8)
    return weights


def make_user_groups(users):
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    groups = []
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_users[end] == sorted_users[start]:
            end += 1
        groups.append(order[start:end].astype(np.int64, copy=False))
        start = end
    return groups


def iter_complete_user_batches(groups, rng, target_size):
    group_order = rng.permutation(len(groups))
    pending = []
    pending_size = 0
    for group_id in group_order:
        group = groups[int(group_id)]
        if pending and pending_size + len(group) > target_size:
            indices = np.concatenate(pending)
            group_ids = np.concatenate([
                np.full(len(g), j, dtype=np.int64) for j, g in enumerate(pending)
            ])
            yield indices, group_ids, len(pending)
            pending = []
            pending_size = 0
        pending.append(group)
        pending_size += len(group)
        if pending_size >= target_size:
            indices = np.concatenate(pending)
            group_ids = np.concatenate([
                np.full(len(g), j, dtype=np.int64) for j, g in enumerate(pending)
            ])
            yield indices, group_ids, len(pending)
            pending = []
            pending_size = 0
    if pending:
        indices = np.concatenate(pending)
        group_ids = np.concatenate([
            np.full(len(g), j, dtype=np.int64) for j, g in enumerate(pending)
        ])
        yield indices, group_ids, len(pending)


def make_pairs(users, labels, seed):
    groups = make_user_groups(users)
    rng = np.random.default_rng(seed)
    pos_parts = []
    neg_parts = []
    for idx in groups:
        pos = idx[labels[idx] > 0.5]
        neg = idx[labels[idx] <= 0.5]
        if len(pos) and len(neg):
            sampled = neg[rng.integers(0, len(neg), size=len(pos))]
            pos_parts.append(pos.astype(np.int64, copy=False))
            neg_parts.append(sampled.astype(np.int64, copy=False))
    if not pos_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(pos_parts), np.concatenate(neg_parts)


class RankModel(nn.Module):
    def __init__(self, n_vocab, n_fields, k, architecture, dropout):
        super().__init__()
        self.architecture = architecture
        self.n_fields = n_fields
        self.k = k
        self.embedding = nn.Embedding(n_vocab, k)
        self.linear = nn.Embedding(n_vocab, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        self.gauge_bias = nn.Parameter(torch.zeros(1))
        self.embed_dropout = nn.Dropout(dropout)
        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.zeros_(self.linear.weight)
        if architecture == "dcn-lite":
            d = n_fields * k
            self.cross_scalar = nn.Linear(d, 1, bias=False)
            self.cross_bias = nn.Parameter(torch.zeros(d))
            self.deep = nn.Sequential(
                nn.Linear(d, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.head = nn.Linear(d + 32, 1)

    def forward(self, x):
        emb = self.embed_dropout(self.embedding(x))
        linear = self.linear(x).sum(dim=1).squeeze(-1) + self.bias
        if self.architecture == "fm":
            summed = emb.sum(dim=1)
            interaction = 0.5 * (summed.square() - emb.square().sum(dim=1)).sum(dim=1)
            return linear + interaction
        x0 = emb.reshape(emb.shape[0], -1)
        cross = x0 * self.cross_scalar(x0) + x0 + self.cross_bias
        deep = self.deep(x0)
        return linear + self.head(torch.cat([cross, deep], dim=1)).squeeze(-1)


def center_logits_by_user(logits, group_ids, n_groups, global_bias):
    sums = torch.zeros(n_groups, dtype=logits.dtype, device=logits.device)
    counts = torch.zeros(n_groups, dtype=logits.dtype, device=logits.device)
    sums.scatter_add_(0, group_ids, logits)
    counts.scatter_add_(0, group_ids, torch.ones_like(logits))
    means = sums / counts.clamp_min(1.0)
    return logits - means[group_ids] + global_bias


def metric_function(fast):
    if fast:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    return evaluate


def score_model(model, Xva, batch_size, device):
    model.eval()
    pieces = []
    with torch.no_grad():
        for start in range(0, len(Xva), batch_size):
            xb = torch.as_tensor(Xva[start:start + batch_size], dtype=torch.long, device=device)
            pieces.append(torch.sigmoid(model(xb)).detach().cpu().numpy())
    return np.concatenate(pieces).astype(np.float64)


def normalize_metrics(result):
    return {
        "gauc": float(result.get("GAUC", result.get("gauc"))),
        "ndcg5": float(result.get("nDCG@5", result.get("ndcg5"))),
        "primary": float(result.get("primary")),
    }


def train_one(data, cfg, seed, epochs, device, evaluator, user_groups, pair_pos, pair_neg, keep_predictions=False):
    seed_all(seed)
    Xtr = data["Xtr"]
    ytr = data["ytr"]
    n_vocab = max(int(np.sum(data["field_dims"])), int(Xtr.max()) + 1, int(data["Xva"].max()) + 1)
    model = RankModel(n_vocab, Xtr.shape[1], 16, cfg["architecture"], cfg["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    batch_size = 16384 if device.type == "cuda" else 8192
    recency = data["recency7"] if cfg["weighting"] == "recency-7d" else data["uniform"]
    rng = np.random.default_rng(seed + 991)
    best_primary = -float("inf")
    best_metrics = None
    best_state = None
    best_predictions = None
    best_checkpoint = 0.0
    trajectory = []
    shuffled_pairs = np.arange(len(pair_pos), dtype=np.int64)
    pair_cursor = 0

    for epoch in range(epochs):
        model.train()
        if len(shuffled_pairs):
            shuffled_pairs = rng.permutation(len(pair_pos))
            pair_cursor = 0
        for idx, group_ids_np, n_groups in iter_complete_user_batches(user_groups, rng, batch_size):
            xb = torch.as_tensor(Xtr[idx], dtype=torch.long, device=device)
            yb = torch.as_tensor(ytr[idx], dtype=torch.float32, device=device)
            wb = torch.as_tensor(recency[idx], dtype=torch.float32, device=device)
            group_ids = torch.as_tensor(group_ids_np, dtype=torch.long, device=device)
            raw_logits = model(xb)
            centered_logits = center_logits_by_user(raw_logits, group_ids, n_groups, model.gauge_bias)
            point_loss = (F.binary_cross_entropy_with_logits(centered_logits, yb, reduction="none") * wb).mean()

            if cfg["loss"] == "bpr-hybrid" and len(pair_pos):
                need = len(idx)
                if pair_cursor + need > len(shuffled_pairs):
                    shuffled_pairs = rng.permutation(len(pair_pos))
                    pair_cursor = 0
                psel = shuffled_pairs[pair_cursor:min(pair_cursor + need, len(shuffled_pairs))]
                pair_cursor += len(psel)
                if len(psel):
                    pi = pair_pos[psel]
                    ni = pair_neg[psel]
                    xp = torch.as_tensor(Xtr[pi], dtype=torch.long, device=device)
                    xn = torch.as_tensor(Xtr[ni], dtype=torch.long, device=device)
                    pw = torch.as_tensor(0.5 * (recency[pi] + recency[ni]), dtype=torch.float32, device=device)
                    pair_loss = (F.softplus(-(model(xp) - model(xn))) * pw).mean()
                    loss = 0.5 * point_loss + 0.5 * pair_loss
                else:
                    loss = point_loss
            else:
                loss = point_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        predictions = score_model(model, data["Xva"], batch_size, device)
        metrics = normalize_metrics(evaluator(data["uva"], data["yva"], predictions))
        checkpoint = float(epoch + 1.0)
        trajectory.append({"checkpoint": checkpoint, **metrics})
        if metrics["primary"] > best_primary:
            best_primary = metrics["primary"]
            best_metrics = metrics
            best_checkpoint = checkpoint
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if keep_predictions:
                best_predictions = predictions.copy()

        for group in optimizer.param_groups:
            group["lr"] *= cfg["lr_decay"]

    if best_state is not None:
        model.load_state_dict(best_state)
    if keep_predictions and best_predictions is None:
        best_predictions = score_model(model, data["Xva"], batch_size, device)
    result = {
        "metrics": best_metrics,
        "best_checkpoint": float(best_checkpoint),
        "trajectory": trajectory,
    }
    if keep_predictions:
        result["predictions"] = best_predictions
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def append_progress(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def main():
    args = parse_args()
    seed_all(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.log"
    if progress_path.exists():
        progress_path.unlink()

    data_dir = Path(args.data_dir)
    fast = (data_dir / "train.npz").exists() and (data_dir / "val.npz").exists()
    data = load_fast(data_dir) if fast else load_csv(data_dir)
    data["uniform"] = np.ones(len(data["ytr"]), dtype=np.float32)
    data["recency7"] = make_recency_weights(data["dates"], 7.0)
    evaluator = metric_function(fast)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    user_groups = make_user_groups(data["utr"])
    pair_pos, pair_neg = make_pairs(data["utr"], data["ytr"], args.seed + 17)

    smoke_raw = os.environ.get("SMOKE_EPOCHS")
    smoke_cap = int(smoke_raw) if smoke_raw is not None else None
    full_epochs = 1 if smoke_cap is None else max(1, min(1, smoke_cap))

    matrix = []
    for architecture in ["fm", "dcn-lite"]:
        for loss in ["logloss", "bpr-hybrid"]:
            for weighting in ["uniform", "recency-7d"]:
                matrix.append({
                    "architecture": architecture,
                    "loss": loss,
                    "weighting": weighting,
                    "objective": "user-centered-bce",
                    "regularization": "mild",
                    "dropout": 0.21,
                    "weight_decay": 3.7e-5,
                    "lr_decay": 0.72,
                    "lr": 0.00168,
                })
    if smoke_cap is not None:
        matrix = matrix[:2]

    history = []
    grouped = []
    for cell_id, cfg in enumerate(matrix):
        result = train_one(data, cfg, args.seed, full_epochs, device, evaluator, user_groups, pair_pos, pair_neg, False)
        entry = {
            "phase": "matrix",
            "cell": cell_id,
            "seed": args.seed,
            "config": cfg,
            "best_checkpoint": result["best_checkpoint"],
            "metrics": result["metrics"],
            "trajectory": result["trajectory"],
        }
        history.append(entry)
        grouped.append((result["metrics"]["primary"], cell_id, cfg))
        append_progress(progress_path, {
            "phase": "matrix",
            "cell": cell_id,
            "seed": args.seed,
            "primary": result["metrics"]["primary"],
            "config": cfg,
        })

    grouped.sort(key=lambda x: (-x[0], x[1]))
    winning_cfg = dict(grouped[0][2])
    final_result = train_one(data, winning_cfg, args.seed, full_epochs, device, evaluator, user_groups, pair_pos, pair_neg, True)
    predictions = final_result["predictions"]
    final_entry = {
        "phase": "final",
        "seed": args.seed,
        "config": winning_cfg,
        "best_checkpoint": final_result["best_checkpoint"],
        "metrics": final_result["metrics"],
        "trajectory": final_result["trajectory"],
    }
    history.append(final_entry)
    append_progress(progress_path, {
        "phase": "final",
        "seed": args.seed,
        "primary": final_result["metrics"]["primary"],
        "config": winning_cfg,
    })

    final_metrics = normalize_metrics(evaluator(data["uva"], data["yva"], predictions))
    with open(out_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, (user, video, score) in enumerate(zip(data["uva"], data["video_out"], predictions)):
            writer.writerow([i, user, video, format(float(score), ".10g")])

    payload = {
        "gauc": final_metrics["gauc"],
        "ndcg5": final_metrics["ndcg5"],
        "primary": final_metrics["primary"],
        "selected_config": winning_cfg,
        "matrix_summary": [
            {"mean_primary": score, "std_primary": 0.0, "cell": cell, "config": cfg}
            for score, cell, cfg in grouped
        ],
        "history": history,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, sort_keys=True)


if __name__ == "__main__":
    main()
