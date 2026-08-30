"""Regularized DeepFM with exponential training recency weighting."""
import argparse
import csv
import datetime
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RegularizedDeepFM(torch.nn.Module):
    def __init__(self, total_dim, fields=5, k=16, dropout=0.15):
        super().__init__()
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(fields * k, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(32, 1),
        )
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)
        for layer in self.mlp:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
        torch.nn.init.normal_(self.mlp[-1].weight, std=0.01)

    def forward(self, x):
        e = self.emb(x)
        summed = e.sum(1)
        pair = 0.5 * (summed * summed - (e * e).sum(1)).sum(1)
        fm = self.bias + self.lin(x).sum((1, 2)) + pair
        deep = self.mlp(e.reshape(e.shape[0], -1)).squeeze(1)
        return fm + deep

    def accessed_row_l2(self, x):
        rows = torch.unique(x)
        return self.emb(rows).square().sum() + self.lin(rows).square().sum()


def load_npz(data_dir):
    tr = np.load(os.path.join(data_dir, "train.npz"))
    va = np.load(os.path.join(data_dir, "val.npz"))
    xt = tr["X"].astype(np.int64)
    yt = tr["y"].astype(np.float32)
    xv = va["X"].astype(np.int64)
    yv = va["y"].astype(np.float32)
    users = va["user"]
    dims = tr["field_dims"].astype(np.int64)
    video_offset = int(dims[0])
    videos = xv[:, 1] - video_offset
    train_dates = np.asarray(tr["date"])
    val_dates = np.asarray(va["date"])
    return xt, yt, xv, yv, users, videos, dims, train_dates, val_dates


def load_csv(data_dir):
    maps = {"user_id": {}, "video_id": {}, "tab": {}, "dur": {}}
    train_values = []
    train_y = []
    train_dates = []

    def add_value(name, value):
        mapping = maps[name]
        if value not in mapping:
            mapping[value] = len(mapping) + 1
        return mapping[value]

    with open(os.path.join(data_dir, "train.csv"), "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            duration = int(float(row["duration_ms"])) if row["duration_ms"] else 0
            dur_bucket = str(min(max(duration // 5000, 0), 120))
            train_values.append((
                add_value("user_id", row["user_id"]),
                add_value("video_id", row["video_id"]),
                0,
                add_value("tab", row["tab"]),
                add_value("dur", dur_bucket),
            ))
            train_y.append(float(row["long_view"]))
            train_dates.append(row["date"])

    val_values = []
    val_y = []
    val_users = []
    val_videos = []
    val_dates = []
    with open(os.path.join(data_dir, "val.csv"), "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            duration = int(float(row["duration_ms"])) if row["duration_ms"] else 0
            dur_bucket = str(min(max(duration // 5000, 0), 120))
            val_values.append((
                maps["user_id"].get(row["user_id"], 0),
                maps["video_id"].get(row["video_id"], 0),
                0,
                maps["tab"].get(row["tab"], 0),
                maps["dur"].get(dur_bucket, 0),
            ))
            val_y.append(float(row["long_view"]))
            val_users.append(row["user_id"])
            val_videos.append(row["video_id"])
            val_dates.append(row["date"])

    dims = np.asarray([
        len(maps["user_id"]) + 1,
        len(maps["video_id"]) + 1,
        1,
        len(maps["tab"]) + 1,
        len(maps["dur"]) + 1,
    ], dtype=np.int64)
    offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(dims)[:-1]))
    xt = np.asarray(train_values, dtype=np.int64) + offsets
    xv = np.asarray(val_values, dtype=np.int64) + offsets
    return (xt, np.asarray(train_y, dtype=np.float32), xv,
            np.asarray(val_y, dtype=np.float32), np.asarray(val_users),
            np.asarray(val_videos), dims, np.asarray(train_dates),
            np.asarray(val_dates))


def date_value_to_day(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) >= 8:
        candidate = digits[:8]
        try:
            parsed = datetime.datetime.strptime(candidate, "%Y%m%d").date()
            return float(parsed.toordinal())
        except ValueError:
            pass
    try:
        return float(text)
    except ValueError:
        try:
            return float(datetime.datetime.fromisoformat(text).date().toordinal())
        except ValueError:
            return 0.0


def dates_to_days(values):
    flat = np.asarray(values).reshape(-1)
    result = np.empty(len(flat), dtype=np.float64)
    cache = {}
    for i, value in enumerate(flat):
        key = value.item() if isinstance(value, np.generic) else value
        if isinstance(key, bytes):
            cache_key = key.decode("utf-8")
        else:
            cache_key = str(key)
        if cache_key not in cache:
            cache[cache_key] = date_value_to_day(key)
        result[i] = cache[cache_key]
    return result


def make_recency_weights(train_dates, val_dates, half_life_days=7.0):
    train_days = dates_to_days(train_dates)
    val_days = dates_to_days(val_dates)
    finite_train = np.isfinite(train_days)
    finite_val = np.isfinite(val_days)
    if not finite_train.any():
        return np.ones(len(train_days), dtype=np.float32)
    if finite_val.any():
        boundary = float(np.min(val_days[finite_val]))
    else:
        boundary = float(np.max(train_days[finite_train]))
    replacement = float(np.min(train_days[finite_train]))
    train_days = np.where(finite_train, train_days, replacement)
    age = np.maximum(boundary - train_days, 0.0)
    raw = np.exp(-np.log(2.0) * age / half_life_days)
    mean_raw = float(raw.mean())
    if not np.isfinite(mean_raw) or mean_raw <= 0.0:
        return np.ones(len(train_days), dtype=np.float32)
    weights = raw / mean_raw
    weights = np.clip(weights, 0.25, 4.0)
    weights /= weights.mean()
    return weights.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)

    epochs = args.epochs
    if "SMOKE_EPOCHS" in os.environ:
        epochs = min(epochs, int(os.environ["SMOKE_EPOCHS"]))
    epochs = max(1, epochs)

    fast_path = (os.path.exists(os.path.join(args.data_dir, "train.npz")) and
                 os.path.exists(os.path.join(args.data_dir, "val.npz")))
    if fast_path:
        from data.official.evaluate import evaluate
        loaded = load_npz(args.data_dir)
    else:
        from harness.evaluate_provisional import evaluate
        loaded = load_csv(args.data_dir)

    (xt_np, yt_np, xv_np, yv, users, videos, field_dims,
     train_dates, val_dates) = loaded
    recency_np = make_recency_weights(train_dates, val_dates)

    xt = torch.from_numpy(xt_np)
    yt = torch.from_numpy(yt_np)
    xv = torch.from_numpy(xv_np)
    recency = torch.from_numpy(recency_np)

    model = RegularizedDeepFM(int(field_dims.sum()))
    embedding_params = [model.emb.weight, model.lin.weight, model.bias]
    dense_params = list(model.mlp.parameters())
    optimizer = torch.optim.AdamW([
        {"params": embedding_params, "weight_decay": 0.0},
        {"params": dense_params, "weight_decay": 1e-4},
    ], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

    n = len(yt)
    batch_size = 8192
    best_primary = -1.0
    best_scores = None
    patience = 0

    for _ in range(epochs):
        model.train()
        permutation = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = permutation[start:start + batch_size]
            xb = xt[idx]
            wb = recency[idx]
            optimizer.zero_grad()
            row_loss = criterion(model(xb), yt[idx])
            loss = (row_loss * wb).sum() / wb.sum().clamp_min(1e-12)
            loss = loss + 1e-6 * model.accessed_row_l2(xb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            scores = np.concatenate([
                model(xv[start:start + 65536]).cpu().numpy()
                for start in range(0, len(xv), 65536)
            ])
        metrics = evaluate(users, yv.astype(int), scores)
        primary = float(metrics["primary"])
        if primary > best_primary + 1e-6:
            best_primary = primary
            best_scores = scores.copy()
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break

    final_metrics = evaluate(users, yv.astype(int), best_scores)
    gauc = final_metrics.get("GAUC", final_metrics.get("gauc"))
    ndcg5 = final_metrics.get("nDCG@5", final_metrics.get("ndcg5"))

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump({
            "gauc": float(gauc),
            "ndcg5": float(ndcg5),
            "primary": float(final_metrics["primary"]),
        }, fh)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(best_scores):
            writer.writerow([i, users[i], videos[i], format(float(score), ".8g")])


if __name__ == "__main__":
    main()
