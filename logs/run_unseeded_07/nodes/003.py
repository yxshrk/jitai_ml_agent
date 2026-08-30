import argparse
import csv
import json
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_npz_data(data_dir):
    train_path = os.path.join(data_dir, "train.npz")
    val_path = os.path.join(data_dir, "val.npz")
    tr = np.load(train_path, allow_pickle=False)
    va = np.load(val_path, allow_pickle=False)

    x_train = np.asarray(tr["X"], dtype=np.int64)
    y_train = np.asarray(tr["y"], dtype=np.float32)
    train_user = np.asarray(tr["user"])
    play_train = np.asarray(tr["play_time_ms"], dtype=np.float32)
    duration_train = np.asarray(tr["duration_ms"], dtype=np.float32)
    train_date = np.asarray(tr["date"])
    field_dims = np.asarray(tr["field_dims"], dtype=np.int64)

    x_val = np.asarray(va["X"], dtype=np.int64)
    y_val = np.asarray(va["y"], dtype=np.float32)
    val_user = np.asarray(va["user"])

    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1]))).astype(np.int64)
    val_video = x_val[:, 1] - offsets[1]
    return {
        "x_train": x_train,
        "y_train": y_train,
        "user_train": train_user,
        "play_train": play_train,
        "duration_train": duration_train,
        "date_train": train_date,
        "x_val": x_val,
        "y_val": y_val,
        "user_val": val_user,
        "video_val": val_video,
        "field_dims": field_dims,
        "npz": True,
    }


def read_csv_rows(path, training):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = {
                "user_id": row["user_id"],
                "video_id": row["video_id"],
                "author_id": row.get("author_id", "__unknown_author__"),
                "tab": row["tab"],
                "duration_ms": float(row["duration_ms"]),
                "long_view": float(row["long_view"]),
            }
            if training:
                item["play_time_ms"] = float(row["play_time_ms"])
                item["date"] = row["date"]
            rows.append(item)
    return rows


def make_mapping(values):
    unique = sorted(set(values))
    return {value: i + 1 for i, value in enumerate(unique)}


def load_csv_data(data_dir):
    train_rows = read_csv_rows(os.path.join(data_dir, "train.csv"), True)
    val_rows = read_csv_rows(os.path.join(data_dir, "val.csv"), False)

    user_map = make_mapping([r["user_id"] for r in train_rows])
    video_map = make_mapping([r["video_id"] for r in train_rows])
    author_map = make_mapping([r["author_id"] for r in train_rows])
    tab_map = make_mapping([r["tab"] for r in train_rows])

    train_duration = np.asarray([r["duration_ms"] for r in train_rows], dtype=np.float32)
    quantiles = np.quantile(train_duration, np.linspace(0.1, 0.9, 9)).astype(np.float32)
    field_dims = np.asarray([
        len(user_map) + 1,
        len(video_map) + 1,
        len(author_map) + 1,
        len(tab_map) + 1,
        10,
    ], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1]))).astype(np.int64)

    def encode(rows):
        x = np.empty((len(rows), 5), dtype=np.int64)
        for i, row in enumerate(rows):
            x[i, 0] = user_map.get(row["user_id"], 0)
            x[i, 1] = video_map.get(row["video_id"], 0)
            x[i, 2] = author_map.get(row["author_id"], 0)
            x[i, 3] = tab_map.get(row["tab"], 0)
            x[i, 4] = int(np.searchsorted(quantiles, row["duration_ms"], side="right"))
        x += offsets.reshape(1, -1)
        return x

    x_train = encode(train_rows)
    x_val = encode(val_rows)
    return {
        "x_train": x_train,
        "y_train": np.asarray([r["long_view"] for r in train_rows], dtype=np.float32),
        "user_train": np.asarray([r["user_id"] for r in train_rows]),
        "play_train": np.asarray([r["play_time_ms"] for r in train_rows], dtype=np.float32),
        "duration_train": train_duration,
        "date_train": np.asarray([r["date"] for r in train_rows]),
        "x_val": x_val,
        "y_val": np.asarray([r["long_view"] for r in val_rows], dtype=np.float32),
        "user_val": np.asarray([r["user_id"] for r in val_rows]),
        "video_val": np.asarray([r["video_id"] for r in val_rows]),
        "field_dims": field_dims,
        "npz": False,
    }


def date_to_ordinal(value):
    text = str(value)
    if text.endswith(".0"):
        text = text[:-2]
    text = text.replace("-", "").replace("/", "")
    try:
        return datetime.strptime(text[:8], "%Y%m%d").toordinal()
    except ValueError:
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return 0


def recency_weights(dates):
    unique = np.unique(dates)
    mapping = {v: date_to_ordinal(v) for v in unique}
    ordinals = np.asarray([mapping[v] for v in dates], dtype=np.float32)
    latest = float(np.max(ordinals)) if len(ordinals) else 0.0
    return np.exp2(-(latest - ordinals) / 7.0).astype(np.float32)


def ordinal_targets(play_time, duration):
    denominator = np.minimum(np.maximum(duration, 1.0), 18000.0)
    ratio = np.clip(play_time / denominator, 0.0, 2.0)
    thresholds = np.asarray([0.25, 0.50, 0.75, 1.00], dtype=np.float32)
    return (ratio[:, None] >= thresholds[None, :]).astype(np.float32)


def build_pairs(users, labels, seed):
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    rng = np.random.default_rng(seed)
    positives = []
    negatives = []
    start = 0
    n = len(order)
    while start < n:
        end = start + 1
        while end < n and sorted_users[end] == sorted_users[start]:
            end += 1
        group = order[start:end]
        pos = group[labels[group] > 0.5]
        neg = group[labels[group] <= 0.5]
        if len(pos) and len(neg):
            positives.append(pos)
            negatives.append(neg[rng.integers(0, len(neg), size=len(pos))])
        start = end
    if not positives:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(positives), np.concatenate(negatives)


class DCNLite(nn.Module):
    def __init__(self, field_dims, embedding_dim=16, dropout=0.30):
        super().__init__()
        total = int(np.sum(field_dims))
        input_dim = len(field_dims) * embedding_dim
        self.embedding = nn.Embedding(total, embedding_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        self.cross_weight = nn.Parameter(torch.empty(input_dim))
        self.cross_bias = nn.Parameter(torch.zeros(input_dim))
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.main_head = nn.Linear(input_dim + 128, 1)
        self.ordinal_head = nn.Linear(input_dim + 128, 4)
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.normal_(self.cross_weight, std=0.01)
        nn.init.xavier_uniform_(self.main_head.weight)
        nn.init.zeros_(self.main_head.bias)
        nn.init.xavier_uniform_(self.ordinal_head.weight)
        nn.init.zeros_(self.ordinal_head.bias)

    def forward(self, x):
        x0 = self.embedding(x).flatten(1)
        x0 = self.embedding_dropout(x0)
        cross = x0 * torch.sum(x0 * self.cross_weight, dim=1, keepdim=True) + self.cross_bias + x0
        deep = self.mlp(x0)
        representation = torch.cat([cross, deep], dim=1)
        return self.main_head(representation).squeeze(1), self.ordinal_head(representation)


def predict(model, x, device, batch_size):
    model.eval()
    result = np.empty(len(x), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            xb = torch.from_numpy(x[start:end]).to(device)
            logits, _ = model(xb)
            result[start:end] = torch.sigmoid(logits).cpu().numpy()
    return result


def metric_values(result):
    gauc = result.get("GAUC", result.get("gauc"))
    ndcg = result.get("nDCG@5", result.get("ndcg5"))
    primary = result.get("primary")
    return float(gauc), float(ndcg), float(primary)


def main():
    args = parse_args()
    seed_everything(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    if os.path.exists(os.path.join(args.data_dir, "train.npz")) and os.path.exists(os.path.join(args.data_dir, "val.npz")):
        data = load_npz_data(args.data_dir)
    else:
        data = load_csv_data(args.data_dir)

    x_train = data["x_train"]
    y_train = data["y_train"]
    x_val = data["x_val"]
    y_val = data["y_val"]
    weights = recency_weights(data["date_train"])
    ord_targets = ordinal_targets(data["play_train"], data["duration_train"])
    pair_pos, pair_neg = build_pairs(data["user_train"], y_train, args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4096 if device.type == "cuda" else 8192
    eval_batch_size = batch_size * 2
    model = DCNLite(data["field_dims"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    epochs = 8
    smoke = os.environ.get("SMOKE_EPOCHS")
    if smoke is not None:
        epochs = min(epochs, max(1, int(smoke)))

    rng = np.random.default_rng(args.seed)
    best_gauc = -float("inf")
    best_state = None
    stale = 0

    if data["npz"]:
        from data.official.evaluate import evaluate as evaluator
    else:
        from harness.evaluate_provisional import evaluate as evaluator

    for _ in range(epochs):
        model.train()
        main_order = rng.permutation(len(x_train))
        pair_order = rng.permutation(len(pair_pos)) if len(pair_pos) else np.empty(0, dtype=np.int64)
        pair_cursor = 0

        for start in range(0, len(main_order), batch_size):
            idx = main_order[start:min(start + batch_size, len(main_order))]
            xb = torch.from_numpy(x_train[idx]).to(device)
            yb = torch.from_numpy(y_train[idx]).to(device)
            wb = torch.from_numpy(weights[idx]).to(device)
            ob = torch.from_numpy(ord_targets[idx]).to(device)

            logits, ordinal_logits = model(xb)
            point_loss = F.binary_cross_entropy_with_logits(logits, yb, reduction="none")
            point_loss = torch.sum(point_loss * wb) / torch.clamp(torch.sum(wb), min=1e-6)
            aux_loss = F.binary_cross_entropy_with_logits(ordinal_logits, ob, reduction="none").mean(dim=1)
            aux_loss = torch.sum(aux_loss * wb) / torch.clamp(torch.sum(wb), min=1e-6)

            if len(pair_order):
                needed = min(batch_size, len(pair_order))
                if pair_cursor + needed > len(pair_order):
                    pair_order = rng.permutation(len(pair_pos))
                    pair_cursor = 0
                selected = pair_order[pair_cursor:pair_cursor + needed]
                pair_cursor += needed
                pi = pair_pos[selected]
                ni = pair_neg[selected]
                px = torch.from_numpy(x_train[pi]).to(device)
                nx = torch.from_numpy(x_train[ni]).to(device)
                ps, _ = model(px)
                ns, _ = model(nx)
                pw_np = 0.5 * (weights[pi] + weights[ni])
                pw = torch.from_numpy(pw_np).to(device)
                pair_loss_values = F.softplus(-(ps - ns))
                pair_loss = torch.sum(pair_loss_values * pw) / torch.clamp(torch.sum(pw), min=1e-6)
            else:
                pair_loss = point_loss.new_zeros(())

            loss = 0.5 * point_loss + 0.5 * pair_loss + 0.3 * aux_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        scheduler.step()
        val_scores = predict(model, x_val, device, eval_batch_size)
        validation = evaluator(data["user_val"], y_val, val_scores)
        current_gauc = float(validation.get("GAUC", validation.get("gauc")))
        if current_gauc > best_gauc + 1e-7:
            best_gauc = current_gauc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 3:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    scores = predict(model, x_val, device, eval_batch_size)
    final_result = evaluator(data["user_val"], y_val, scores)
    gauc, ndcg5, primary = metric_values(final_result)

    prediction_path = os.path.join(args.out_dir, "predictions.csv")
    with open(prediction_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, (user_id, video_id, score) in enumerate(zip(data["user_val"], data["video_val"], scores)):
            writer.writerow([i, user_id, video_id, format(float(score), ".10f")])

    metrics_path = os.path.join(args.out_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"gauc": gauc, "ndcg5": ndcg5, "primary": primary}, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
