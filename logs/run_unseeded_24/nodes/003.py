import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import contextlib
import csv
import io
import json
import math
import random
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def date_to_ordinal(value):
    try:
        text = str(value.decode() if isinstance(value, bytes) else value).strip()
        if text.endswith(".0"):
            text = text[:-2]
        return datetime.strptime(text, "%Y%m%d").toordinal()
    except Exception:
        return 0


def ordinal_array(values):
    values = np.asarray(values)
    result = np.empty(len(values), dtype=np.int32)
    cache = {}
    for i, value in enumerate(values):
        key = value.item() if isinstance(value, np.generic) else value
        if key not in cache:
            cache[key] = date_to_ordinal(key)
        result[i] = cache[key]
    return result


def load_npz(data_dir):
    train_path = os.path.join(data_dir, "train.npz")
    val_path = os.path.join(data_dir, "val.npz")
    tr = np.load(train_path, allow_pickle=False)
    va = np.load(val_path, allow_pickle=False)
    x_train = np.asarray(tr["X"], dtype=np.int64)
    y_train = np.asarray(tr["y"], dtype=np.float32)
    x_val = np.asarray(va["X"], dtype=np.int64)
    y_val = np.asarray(va["y"], dtype=np.float32)
    train_users = np.asarray(tr["user"])
    val_users = np.asarray(va["user"])
    field_dims = np.asarray(tr["field_dims"], dtype=np.int64)
    dates = np.asarray(tr["date"]) if "date" in tr.files else np.zeros(len(y_train), dtype=np.int32)
    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1]))).astype(np.int64)
    val_video = x_val[:, 1] - offsets[1]
    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "train_users": train_users,
        "val_users": val_users,
        "val_video": val_video,
        "field_dims": field_dims,
        "dates": dates,
        "npz": True,
    }


def read_csv_rows(path, training):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = {
                "user": row["user_id"],
                "video": row["video_id"],
                "author": row.get("author_id", "0"),
                "tab": row.get("tab", "0"),
                "duration": float(row.get("duration_ms", 0.0) or 0.0),
                "label": float(row["long_view"]),
            }
            if training:
                item["date"] = row.get("date", "0")
            rows.append(item)
    return rows


def load_csv(data_dir):
    train_rows = read_csv_rows(os.path.join(data_dir, "train.csv"), True)
    val_rows = read_csv_rows(os.path.join(data_dir, "val.csv"), False)
    durations = np.asarray([r["duration"] for r in train_rows], dtype=np.float64)
    quantiles = np.quantile(durations, np.linspace(0.1, 0.9, 9))

    field_names = ("user", "video", "author", "tab")
    maps = []
    for name in field_names:
        values = sorted({r[name] for r in train_rows})
        maps.append({v: i + 1 for i, v in enumerate(values)})

    field_dims = np.asarray([len(m) + 1 for m in maps] + [10], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1]))).astype(np.int64)

    def encode(rows):
        x = np.empty((len(rows), 5), dtype=np.int64)
        for i, row in enumerate(rows):
            for j, name in enumerate(field_names):
                x[i, j] = maps[j].get(row[name], 0) + offsets[j]
            bucket = int(np.searchsorted(quantiles, row["duration"], side="right"))
            x[i, 4] = bucket + offsets[4]
        return x

    return {
        "x_train": encode(train_rows),
        "y_train": np.asarray([r["label"] for r in train_rows], dtype=np.float32),
        "x_val": encode(val_rows),
        "y_val": np.asarray([r["label"] for r in val_rows], dtype=np.float32),
        "train_users": np.asarray([r["user"] for r in train_rows]),
        "val_users": np.asarray([r["user"] for r in val_rows]),
        "val_video": np.asarray([r["video"] for r in val_rows]),
        "field_dims": field_dims,
        "dates": np.asarray([r["date"] for r in train_rows]),
        "npz": False,
    }


def make_recency_weights(dates):
    ordinals = ordinal_array(dates)
    valid = ordinals > 0
    if not np.any(valid):
        return np.ones(len(ordinals), dtype=np.float32)
    latest = int(ordinals[valid].max())
    ages = np.maximum(0, latest - ordinals)
    weights = np.power(2.0, -ages.astype(np.float64) / 7.0)
    weights[~valid] = 1.0
    weights /= max(float(weights.mean()), 1e-8)
    return weights.astype(np.float32)


def build_pair_pool(users, labels, seed):
    users = np.asarray(users)
    labels = np.asarray(labels) >= 0.5
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    rng = np.random.default_rng(seed)
    positive_parts = []
    negative_parts = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        group = order[left:right]
        pos = group[labels[group]]
        neg = group[~labels[group]]
        if len(pos) and len(neg):
            positive_parts.append(pos)
            negative_parts.append(rng.choice(neg, size=len(pos), replace=True))
    if not positive_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(positive_parts).astype(np.int64), np.concatenate(negative_parts).astype(np.int64)


class DCNLite(nn.Module):
    def __init__(self, total_features, embedding_dim, hidden_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(total_features, embedding_dim)
        input_dim = 5 * embedding_dim
        self.embedding_dropout = nn.Dropout(dropout)
        self.cross_weight = nn.Parameter(torch.empty(input_dim))
        self.cross_bias = nn.Parameter(torch.zeros(input_dim))
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.output = nn.Linear(input_dim + hidden_dim // 2, 1)
        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.normal_(self.cross_weight, std=0.01)

    def forward(self, x):
        base = self.embedding(x).flatten(1)
        base = self.embedding_dropout(base)
        cross_scalar = torch.sum(base * self.cross_weight, dim=1, keepdim=True)
        crossed = base + base * cross_scalar + self.cross_bias
        deep = self.mlp(base)
        return self.output(torch.cat((crossed, deep), dim=1)).squeeze(1)


def predict(model, x, device, batch_size):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.as_tensor(x[start:start + batch_size], dtype=torch.long, device=device)
            outputs.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(outputs).astype(np.float64)


def official_metrics(is_npz, users, labels, scores):
    if is_npz:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        result = evaluate(users, labels, scores)
    gauc = float(result.get("GAUC", result.get("gauc")))
    ndcg = float(result.get("nDCG@5", result.get("ndcg5")))
    primary = float(result.get("primary", 0.5 * (gauc + ndcg)))
    return {"gauc": gauc, "ndcg5": ndcg, "primary": primary}


def train_candidate(data, recency, pair_pos, pair_neg, config, epochs, seed, device):
    seed_everything(seed)
    x_train = data["x_train"]
    y_train = data["y_train"]
    n = len(y_train)
    batch_size = config["batch_size"]
    model = DCNLite(int(np.sum(data["field_dims"])), 16, 128, config["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    rng = np.random.default_rng(seed)
    best_gauc = -math.inf
    best_scores = None
    stale = 0
    checks_per_epoch = 2

    for epoch in range(epochs):
        model.train()
        impression_order = rng.permutation(n)
        pair_order = rng.permutation(len(pair_pos)) if len(pair_pos) else np.empty(0, dtype=np.int64)
        steps = int(math.ceil(n / batch_size))
        split_step = max(1, steps // checks_per_epoch)

        for step in range(steps):
            start = step * batch_size
            indices = impression_order[start:min(start + batch_size, n)]
            xb = torch.as_tensor(x_train[indices], dtype=torch.long, device=device)
            yb = torch.as_tensor(y_train[indices], dtype=torch.float32, device=device)
            wb = torch.as_tensor(recency[indices], dtype=torch.float32, device=device)
            logits = model(xb)
            point_losses = F.binary_cross_entropy_with_logits(logits, yb, reduction="none")
            point_loss = torch.sum(point_losses * wb) / torch.clamp(torch.sum(wb), min=1e-8)

            if len(pair_pos):
                count = len(indices)
                pair_slots = (np.arange(start, start + count, dtype=np.int64) % len(pair_pos))
                selected = pair_order[pair_slots]
                pos_idx = pair_pos[selected]
                neg_idx = pair_neg[selected]
                pos_x = torch.as_tensor(x_train[pos_idx], dtype=torch.long, device=device)
                neg_x = torch.as_tensor(x_train[neg_idx], dtype=torch.long, device=device)
                pos_score = model(pos_x)
                neg_score = model(neg_x)
                pair_loss_raw = F.softplus(-(pos_score - neg_score))
                pair_weights = torch.as_tensor(
                    0.5 * (recency[pos_idx] + recency[neg_idx]), dtype=torch.float32, device=device
                )
                pair_loss = torch.sum(pair_loss_raw * pair_weights) / torch.clamp(torch.sum(pair_weights), min=1e-8)
                loss = 0.5 * point_loss + 0.5 * pair_loss
            else:
                loss = point_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            should_check = (step + 1 == steps) or (step + 1 == split_step)
            if should_check:
                scores = predict(model, data["x_val"], device, batch_size * 2)
                metrics = official_metrics(data["npz"], data["val_users"], data["y_val"], scores)
                if metrics["gauc"] > best_gauc + 1e-8:
                    best_gauc = metrics["gauc"]
                    best_scores = scores.copy()
                    stale = 0
                else:
                    stale += 1
                model.train()

        if epoch in config["decay_epochs"]:
            for group in optimizer.param_groups:
                group["lr"] *= 0.5
        if stale >= 5:
            break

    return best_gauc, best_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    use_npz = os.path.exists(os.path.join(args.data_dir, "train.npz")) and os.path.exists(os.path.join(args.data_dir, "val.npz"))
    data = load_npz(args.data_dir) if use_npz else load_csv(args.data_dir)
    recency = make_recency_weights(data["dates"])
    pair_pos, pair_neg = build_pair_pool(data["train_users"], data["y_train"], args.seed + 913)

    epochs = 8
    smoke = os.environ.get("SMOKE_EPOCHS")
    if smoke is not None:
        epochs = min(epochs, max(1, int(smoke)))

    configs = [
        {"dropout": 0.25, "weight_decay": 5e-4, "lr": 0.0020, "decay_epochs": (1, 3, 5), "batch_size": 8192},
        {"dropout": 0.30, "weight_decay": 1e-3, "lr": 0.0015, "decay_epochs": (1, 3, 5), "batch_size": 8192},
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    overall_gauc = -math.inf
    best_scores = None
    for candidate_index, config in enumerate(configs):
        gauc, scores = train_candidate(
            data, recency, pair_pos, pair_neg, config, epochs, args.seed + candidate_index * 1009, device
        )
        if scores is not None and gauc > overall_gauc:
            overall_gauc = gauc
            best_scores = scores

    if best_scores is None:
        best_scores = np.full(len(data["y_val"]), float(np.mean(data["y_train"])), dtype=np.float64)

    metrics = official_metrics(data["npz"], data["val_users"], data["y_val"], best_scores)
    prediction_path = os.path.join(args.out_dir, "predictions.csv")
    with open(prediction_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, (user, video, score) in enumerate(zip(data["val_users"], data["val_video"], best_scores)):
            if isinstance(user, bytes):
                user = user.decode()
            if isinstance(video, bytes):
                video = video.decode()
            writer.writerow([i, user, video, format(float(score), ".10g")])

    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
