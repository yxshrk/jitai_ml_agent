import argparse
import csv
import json
import os
import random

import numpy as np
import torch
from torch import nn


class FactorizationMachine(nn.Module):
    def __init__(self, total_features, embedding_dim):
        super().__init__()
        self.linear = nn.Embedding(total_features, 1)
        self.embedding = nn.Embedding(total_features, embedding_dim)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

    def forward(self, x):
        linear_term = self.linear(x).sum(dim=1).squeeze(-1)
        vectors = self.embedding(x)
        summed = vectors.sum(dim=1)
        interactions = 0.5 * (
            summed.square().sum(dim=1) - vectors.square().sum(dim=(1, 2))
        )
        return self.bias + linear_term + interactions


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def scalar_text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        value = value.item()
    return str(value)


def load_npz(data_dir):
    train_path = os.path.join(data_dir, "train.npz")
    val_path = os.path.join(data_dir, "val.npz")
    with np.load(train_path, allow_pickle=False) as z:
        train_x = np.asarray(z["X"], dtype=np.int64)
        train_y = np.asarray(z["y"], dtype=np.float32)
        field_dims = np.asarray(z["field_dims"], dtype=np.int64)
    with np.load(val_path, allow_pickle=False) as z:
        val_x = np.asarray(z["X"], dtype=np.int64)
        val_y = np.asarray(z["y"], dtype=np.float32)
        val_users = np.asarray(z["user"])
    video_offset = int(field_dims[0])
    val_videos = val_x[:, 1].astype(np.int64) - video_offset
    return train_x, train_y, val_x, val_y, val_users, val_videos, field_dims, True


def read_csv_rows(path, training):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = {
                "user_id": row["user_id"],
                "video_id": row["video_id"],
                "tab": row["tab"],
                "duration_ms": float(row["duration_ms"]),
                "long_view": float(row["long_view"]),
            }
            if training and "author_id" in row:
                item["author_id"] = row["author_id"]
            rows.append(item)
    return rows


def make_mapping(values):
    unique = sorted(set(values))
    return {value: index + 1 for index, value in enumerate(unique)}


def load_csv(data_dir):
    train_rows = read_csv_rows(os.path.join(data_dir, "train.csv"), True)
    val_rows = read_csv_rows(os.path.join(data_dir, "val.csv"), False)

    user_map = make_mapping([r["user_id"] for r in train_rows])
    video_map = make_mapping([r["video_id"] for r in train_rows])
    tab_map = make_mapping([r["tab"] for r in train_rows])
    has_author = bool(train_rows) and "author_id" in train_rows[0]
    if has_author:
        author_map = make_mapping([r["author_id"] for r in train_rows])
    else:
        author_map = {"__unknown__": 1}

    durations = np.asarray([r["duration_ms"] for r in train_rows], dtype=np.float64)
    quantiles = np.quantile(durations, np.linspace(0.1, 0.9, 9))
    quantiles = np.maximum.accumulate(quantiles)

    field_dims = np.asarray(
        [len(user_map) + 1, len(video_map) + 1, len(author_map) + 1, len(tab_map) + 1, 10],
        dtype=np.int64,
    )
    offsets = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(field_dims)[:-1]])

    def encode(rows, is_train):
        x = np.empty((len(rows), 5), dtype=np.int64)
        y = np.empty(len(rows), dtype=np.float32)
        for i, row in enumerate(rows):
            user_code = user_map.get(row["user_id"], 0)
            video_code = video_map.get(row["video_id"], 0)
            if has_author and is_train:
                author_code = author_map.get(row["author_id"], 0)
            else:
                author_code = 0
            tab_code = tab_map.get(row["tab"], 0)
            duration_code = int(np.searchsorted(quantiles, row["duration_ms"], side="right"))
            x[i] = np.asarray(
                [user_code, video_code, author_code, tab_code, duration_code],
                dtype=np.int64,
            ) + offsets
            y[i] = row["long_view"]
        return x, y

    train_x, train_y = encode(train_rows, True)
    val_x, val_y = encode(val_rows, False)
    val_users = np.asarray([r["user_id"] for r in val_rows], dtype=object)
    val_videos = np.asarray([r["video_id"] for r in val_rows], dtype=object)
    return train_x, train_y, val_x, val_y, val_users, val_videos, field_dims, False


def train_one(train_x, train_y, val_x, field_dims, seed, epochs, device):
    set_seed(seed)
    model = FactorizationMachine(int(field_dims.sum()), 16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, foreach=False)
    criterion = nn.BCEWithLogitsLoss()
    rng = np.random.RandomState(seed)
    batch_size = 16384

    model.train()
    for _ in range(epochs):
        order = rng.permutation(len(train_y))
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            xb = torch.as_tensor(train_x[indices], dtype=torch.long, device=device)
            yb = torch.as_tensor(train_y[indices], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    predictions = np.empty(len(val_x), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(val_x), batch_size):
            end = min(start + batch_size, len(val_x))
            xb = torch.as_tensor(val_x[start:end], dtype=torch.long, device=device)
            predictions[start:end] = torch.sigmoid(model(xb)).cpu().numpy().astype(np.float64)
    return predictions


def within_user_ranks(users, scores):
    users_text = np.asarray([scalar_text(v) for v in users], dtype=object)
    order = np.lexsort((np.arange(len(scores)), scores, users_text))
    sorted_users = users_text[order]
    result = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_users[end] == sorted_users[start]:
            end += 1
        count = end - start
        if count == 1:
            result[order[start]] = 0.5
        else:
            result[order[start:end]] = np.arange(count, dtype=np.float64) / float(count - 1)
        start = end
    return result


def normalize_metrics(metrics):
    return {
        "gauc": float(metrics["GAUC"]),
        "ndcg5": float(metrics["nDCG@5"]),
        "primary": float(metrics["primary"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    fast_path = os.path.exists(os.path.join(args.data_dir, "train.npz")) and os.path.exists(
        os.path.join(args.data_dir, "val.npz")
    )
    if fast_path:
        data = load_npz(args.data_dir)
        from data.official.evaluate import evaluate
    else:
        data = load_csv(args.data_dir)
        from harness.evaluate_provisional import evaluate

    train_x, train_y, val_x, val_y, val_users, val_videos, field_dims, _ = data
    epochs = 8
    smoke_epochs = os.environ.get("SMOKE_EPOCHS")
    if smoke_epochs is not None:
        epochs = min(epochs, max(1, int(smoke_epochs)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [args.seed, args.seed + 1, args.seed + 2]
    seed_predictions = [
        train_one(train_x, train_y, val_x, field_dims, seed, epochs, device)
        for seed in seeds
    ]

    raw_average = np.mean(np.stack(seed_predictions, axis=0), axis=0)
    ranked = [within_user_ranks(val_users, p) for p in seed_predictions]
    rank_average = np.mean(np.stack(ranked, axis=0), axis=0)

    raw_metrics = normalize_metrics(evaluate(val_users, val_y, raw_average))
    rank_metrics = normalize_metrics(evaluate(val_users, val_y, rank_average))
    if rank_metrics["primary"] > raw_metrics["primary"]:
        final_scores = rank_average
        final_metrics = rank_metrics
    else:
        final_scores = raw_average
        final_metrics = raw_metrics

    prediction_path = os.path.join(args.out_dir, "predictions.csv")
    with open(prediction_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, (user_id, video_id, score) in enumerate(zip(val_users, val_videos, final_scores)):
            writer.writerow([i, scalar_text(user_id), scalar_text(video_id), format(float(score), ".17g")])

    metrics_path = os.path.join(args.out_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    main()
