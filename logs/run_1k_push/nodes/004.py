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


def parse_hourmin(value):
    try:
        v = int(float(str(value).strip()))
        hour = v // 100
        minute = v % 100
        if 0 <= hour < 24 and 0 <= minute < 60:
            return hour, minute
    except Exception:
        pass
    return 24, 0


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
    dates_tr = np.asarray(tr["date"]) if "date" in tr.files else np.zeros(len(ytr), dtype=np.int64)
    dates_va = np.asarray(va["date"]) if "date" in va.files else np.zeros(len(yva), dtype=np.int64)
    hourmin_tr = np.asarray(tr["hourmin"]) if "hourmin" in tr.files else np.zeros(len(ytr), dtype=np.int64)
    hourmin_va = np.asarray(va["hourmin"]) if "hourmin" in va.files else np.zeros(len(yva), dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1]))).astype(np.int64)
    video_local = Xva[:, 1] - offsets[1]
    return {
        "Xtr": Xtr,
        "ytr": ytr,
        "utr": utr,
        "dates": dates_tr,
        "hourmin_tr": hourmin_tr,
        "Xva": Xva,
        "yva": yva,
        "uva": uva,
        "dates_va": dates_va,
        "hourmin_va": hourmin_va,
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
                "hourmin": row.get("hourmin", "0"),
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
        "hourmin_tr": np.asarray([r["hourmin"] for r in train_rows]),
        "Xva": encode(raw_val),
        "yva": np.asarray([r["long_view"] for r in val_rows], dtype=np.float32),
        "uva": np.asarray([r["user_id"] for r in val_rows]),
        "dates_va": np.asarray([r["date"] for r in val_rows]),
        "hourmin_va": np.asarray([r["hourmin"] for r in val_rows]),
        "video_out": np.asarray([r["video_id"] for r in val_rows]),
        "field_dims": np.asarray(dims, dtype=np.int64),
        "fast": False,
    }


def add_session_time_features(data):
    base_dims = np.asarray(data["field_dims"], dtype=np.int64)
    base_offsets = np.concatenate(([0], np.cumsum(base_dims[:-1]))).astype(np.int64)
    tab_dim = int(base_dims[3])
    gap_edges = np.asarray([1, 5, 15, 30, 60, 180, 720], dtype=np.float64)
    position_edges = np.asarray([1, 2, 3, 5, 8, 16], dtype=np.int64)
    session_dims = np.asarray([9, 7, 25 * tab_dim, 8 * tab_dim], dtype=np.int64)

    date_cache = {}

    def ordinal_cached(value):
        key = str(value)
        if key not in date_cache:
            date_cache[key] = date_ordinal(value)
        return date_cache[key]

    def derive(users, dates, hourmins, X, state):
        n = len(users)
        local = np.empty((n, 4), dtype=np.int64)
        tab_local = np.clip(X[:, 3] - base_offsets[3], 0, tab_dim - 1).astype(np.int64)
        for i in range(n):
            user = users[i].item() if isinstance(users[i], np.generic) else users[i]
            ordinal = ordinal_cached(dates[i])
            hour, minute = parse_hourmin(hourmins[i])
            valid_time = ordinal > 0 and hour < 24
            timestamp = ordinal * 1440 + hour * 60 + minute if valid_time else None
            previous = state.get(user)
            gap_code = 0
            position = 1
            if previous is not None and timestamp is not None and previous[0] is not None:
                gap = timestamp - previous[0]
                if gap >= 0:
                    gap_code = 1 + int(np.searchsorted(gap_edges, float(gap), side="right"))
                    position = previous[1] + 1 if gap <= 30 else 1
            position_code = int(np.searchsorted(position_edges, position, side="right"))
            weekday = (ordinal - 1) % 7 if ordinal > 0 else 7
            local[i, 0] = gap_code
            local[i, 1] = position_code
            local[i, 2] = hour * tab_dim + tab_local[i]
            local[i, 3] = weekday * tab_dim + tab_local[i]
            state[user] = (timestamp, position)
        return local

    state = {}
    local_tr = derive(data["utr"], data["dates"], data["hourmin_tr"], data["Xtr"], state)
    local_va = derive(data["uva"], data["dates_va"], data["hourmin_va"], data["Xva"], state)
    appended_offsets = int(np.sum(base_dims)) + np.concatenate(([0], np.cumsum(session_dims[:-1]))).astype(np.int64)
    data["Xtr"] = np.concatenate([data["Xtr"], local_tr + appended_offsets], axis=1)
    data["Xva"] = np.concatenate([data["Xva"], local_va + appended_offsets], axis=1)
    data["field_dims"] = np.concatenate([base_dims, session_dims])


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


def make_pairs(users, labels, seed):
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    rng = np.random.default_rng(seed)
    pos_parts = []
    neg_parts = []
    start = 0
    n = len(order)
    while start < n:
        end = start + 1
        while end < n and sorted_users[end] == sorted_users[start]:
            end += 1
        idx = order[start:end]
        pos = idx[labels[idx] > 0.5]
        neg = idx[labels[idx] <= 0.5]
        if len(pos) and len(neg):
            sampled = neg[rng.integers(0, len(neg), size=len(pos))]
            pos_parts.append(pos.astype(np.int64, copy=False))
            neg_parts.append(sampled.astype(np.int64, copy=False))
        start = end
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


def train_one(data, cfg, seed, epochs, device, evaluator, pair_pos, pair_neg, keep_predictions=False):
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
    pair_cursor = 0
    shuffled_pairs = np.arange(len(pair_pos), dtype=np.int64)

    for epoch in range(epochs):
        model.train()
        point_order = rng.permutation(len(Xtr))
        if len(shuffled_pairs):
            shuffled_pairs = rng.permutation(len(pair_pos))
            pair_cursor = 0
        total_steps = max(1, math.ceil(len(point_order) / batch_size))
        for step in range(total_steps):
            idx = point_order[step * batch_size:(step + 1) * batch_size]
            xb = torch.as_tensor(Xtr[idx], dtype=torch.long, device=device)
            yb = torch.as_tensor(ytr[idx], dtype=torch.float32, device=device)
            wb = torch.as_tensor(recency[idx], dtype=torch.float32, device=device)
            logits = model(xb)
            point_loss = (F.binary_cross_entropy_with_logits(logits, yb, reduction="none") * wb).mean()
            if cfg["loss"] == "bpr-hybrid" and len(pair_pos):
                need = len(idx)
                if pair_cursor + need > len(shuffled_pairs):
                    shuffled_pairs = rng.permutation(len(pair_pos))
                    pair_cursor = 0
                psel = shuffled_pairs[pair_cursor:pair_cursor + need]
                pair_cursor += len(psel)
                pi = pair_pos[psel]
                ni = pair_neg[psel]
                xp = torch.as_tensor(Xtr[pi], dtype=torch.long, device=device)
                xn = torch.as_tensor(Xtr[ni], dtype=torch.long, device=device)
                pw = torch.as_tensor(0.5 * (recency[pi] + recency[ni]), dtype=torch.float32, device=device)
                pair_loss = (F.softplus(-(model(xp) - model(xn))) * pw).mean()
                loss = 0.5 * point_loss + 0.5 * pair_loss
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
    add_session_time_features(data)
    data["uniform"] = np.ones(len(data["ytr"]), dtype=np.float32)
    data["recency7"] = make_recency_weights(data["dates"], 7.0)
    evaluator = metric_function(fast)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pair_pos, pair_neg = make_pairs(data["utr"], data["ytr"], args.seed + 17)
    smoke_raw = os.environ.get("SMOKE_EPOCHS")
    smoke_cap = int(smoke_raw) if smoke_raw is not None else None
    full_epochs = 1 if smoke_cap is None else max(1, min(1, smoke_cap))

    architectures = ["fm", "dcn-lite"]
    losses = ["logloss", "bpr-hybrid"]
    weightings = ["uniform", "recency-7d"]
    matrix = []
    for architecture in architectures:
        for loss in losses:
            for weighting in weightings:
                matrix.append({
                    "architecture": architecture,
                    "loss": loss,
                    "weighting": weighting,
                    "regularization": "mild",
                    "dropout": 0.21,
                    "weight_decay": 3.7e-5,
                    "lr_decay": 0.72,
                    "lr": 0.00168,
                    "session_time_features": True,
                })
    if smoke_cap is not None:
        matrix = matrix[:2]
        matrix_seeds = [args.seed]
    else:
        matrix_seeds = [args.seed + i for i in range(8)]

    history = []
    grouped = []
    for cell_id, cfg in enumerate(matrix):
        scores = []
        for seed in matrix_seeds:
            result = train_one(data, cfg, seed, full_epochs, device, evaluator, pair_pos, pair_neg, False)
            entry = {
                "phase": "matrix",
                "cell": cell_id,
                "seed": seed,
                "config": cfg,
                "best_checkpoint": result["best_checkpoint"],
                "metrics": result["metrics"],
                "trajectory": result["trajectory"],
            }
            history.append(entry)
            scores.append(result["metrics"]["primary"])
            append_progress(progress_path, {"phase": "matrix", "cell": cell_id, "seed": seed, "primary": scores[-1], "config": cfg})
        grouped.append((float(np.mean(scores)), float(np.std(scores)), cell_id, cfg))
    grouped.sort(key=lambda x: (-x[0], x[1], x[2]))
    winning_cfg = dict(grouped[0][3])

    final_predictions = []
    final_runs = []
    for seed in [args.seed]:
        result = train_one(data, winning_cfg, seed, full_epochs, device, evaluator, pair_pos, pair_neg, True)
        final_predictions.append(result["predictions"])
        final_entry = {
            "phase": "final",
            "seed": seed,
            "config": winning_cfg,
            "best_checkpoint": result["best_checkpoint"],
            "metrics": result["metrics"],
            "trajectory": result["trajectory"],
        }
        history.append(final_entry)
        final_runs.append(final_entry)
        append_progress(progress_path, {"phase": "final", "seed": seed, "primary": result["metrics"]["primary"], "config": winning_cfg})
    predictions = np.mean(np.stack(final_predictions, axis=0), axis=0)
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
        "session_recipe": {
            "gap_edges_minutes": [1, 5, 15, 30, 60, 180, 720],
            "position_edges": [1, 2, 3, 5, 8, 16],
            "session_cut_minutes": 30,
            "crosses": ["hour_x_tab", "weekday_x_tab"],
            "causal": True,
        },
        "matrix_summary": [
            {"mean_primary": mean, "std_primary": std, "cell": cell, "config": cfg}
            for mean, std, cell, cfg in grouped
        ],
        "history": history,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, sort_keys=True)


if __name__ == "__main__":
    main()
