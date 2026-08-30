import argparse
import csv
import json
import os
import random
import warnings
from contextlib import redirect_stderr, redirect_stdout

import numpy as np
import torch
from torch import nn


class FactorizationMachine(nn.Module):
    def __init__(self, num_embeddings, num_fields, dim=16):
        super().__init__()
        self.linear = nn.Embedding(num_embeddings, 1)
        self.embedding = nn.Embedding(num_embeddings, dim)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)
        self.num_fields = num_fields

    def forward(self, x):
        linear = self.linear(x).sum(dim=1).squeeze(1) + self.bias
        e = self.embedding(x)
        interaction = 0.5 * ((e.sum(dim=1) ** 2 - (e ** 2).sum(dim=1)).sum(dim=1))
        return linear + interaction


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))


def as_python_id(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def load_npz(data_dir):
    train_path = os.path.join(data_dir, "train.npz")
    val_path = os.path.join(data_dir, "val.npz")
    with np.load(train_path, allow_pickle=False) as z:
        x_train = np.asarray(z["X"], dtype=np.int64)
        y_train = np.asarray(z["y"], dtype=np.float32).reshape(-1)
        field_dims = np.asarray(z["field_dims"], dtype=np.int64).reshape(-1)
    with np.load(val_path, allow_pickle=False) as z:
        x_val = np.asarray(z["X"], dtype=np.int64)
        y_val = np.asarray(z["y"], dtype=np.float32).reshape(-1)
        users = np.asarray(z["user"]).reshape(-1)
        if "video" in z.files:
            videos = np.asarray(z["video"]).reshape(-1)
        else:
            video_offset = int(field_dims[0])
            videos = x_val[:, 1] - video_offset
    total_embeddings = max(int(field_dims.sum()), int(x_train.max()) + 1, int(x_val.max()) + 1)
    return x_train, y_train, x_val, y_val, users, videos, total_embeddings, True


def make_mapping(values):
    mapping = {}
    for value in values:
        if value not in mapping:
            mapping[value] = len(mapping) + 1
    return mapping


def load_csv(data_dir):
    train_rows = []
    val_rows = []
    with open(os.path.join(data_dir, "train.csv"), "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            train_rows.append((
                row["user_id"],
                row["video_id"],
                row.get("author_id", "__unknown_author__"),
                row["tab"],
                float(row["duration_ms"]),
                float(row["long_view"]),
            ))
    with open(os.path.join(data_dir, "val.csv"), "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            val_rows.append((
                row["user_id"],
                row["video_id"],
                row.get("author_id", "__unknown_author__"),
                row["tab"],
                float(row["duration_ms"]),
                float(row["long_view"]),
            ))

    user_map = make_mapping(row[0] for row in train_rows)
    video_map = make_mapping(row[1] for row in train_rows)
    author_map = make_mapping(row[2] for row in train_rows)
    tab_map = make_mapping(row[3] for row in train_rows)
    train_durations = np.asarray([row[4] for row in train_rows], dtype=np.float64)
    quantiles = np.quantile(train_durations, np.linspace(0.1, 0.9, 9))

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
        y = np.empty(len(rows), dtype=np.float32)
        for i, row in enumerate(rows):
            x[i, 0] = user_map.get(row[0], 0) + offsets[0]
            x[i, 1] = video_map.get(row[1], 0) + offsets[1]
            x[i, 2] = author_map.get(row[2], 0) + offsets[2]
            x[i, 3] = tab_map.get(row[3], 0) + offsets[3]
            x[i, 4] = int(np.searchsorted(quantiles, row[4], side="right")) + offsets[4]
            y[i] = row[5]
        return x, y

    x_train, y_train = encode(train_rows)
    x_val, y_val = encode(val_rows)
    users = np.asarray([row[0] for row in val_rows], dtype=object)
    videos = np.asarray([row[1] for row in val_rows], dtype=object)
    return x_train, y_train, x_val, y_val, users, videos, int(field_dims.sum()), False


def train_one(x_train, y_train, x_val, total_embeddings, seed, epochs):
    seed_everything(seed)
    device = torch.device("cpu")
    model = FactorizationMachine(total_embeddings, x_train.shape[1], dim=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    batch_size = 16384
    n = x_train.shape[0]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    model.train()
    for _ in range(epochs):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size].numpy()
            xb = torch.from_numpy(x_train[idx]).to(device)
            yb = torch.from_numpy(y_train[idx]).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    predictions = np.empty(x_val.shape[0], dtype=np.float64)
    with torch.no_grad():
        for start in range(0, x_val.shape[0], batch_size):
            end = min(start + batch_size, x_val.shape[0])
            xb = torch.from_numpy(x_val[start:end]).to(device)
            predictions[start:end] = torch.sigmoid(model(xb)).cpu().numpy()
    return predictions


def rank_transform(scores):
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.shape[0], dtype=np.float64)
    ranks[order] = np.arange(scores.shape[0], dtype=np.float64)
    if scores.shape[0] > 1:
        ranks /= float(scores.shape[0] - 1)
    return ranks


def run(args):
    os.makedirs(args.out_dir, exist_ok=True)
    npz_fast = os.path.exists(os.path.join(args.data_dir, "train.npz")) and os.path.exists(os.path.join(args.data_dir, "val.npz"))
    if npz_fast:
        x_train, y_train, x_val, y_val, users, videos, total_embeddings, used_npz = load_npz(args.data_dir)
    else:
        x_train, y_train, x_val, y_val, users, videos, total_embeddings, used_npz = load_csv(args.data_dir)

    epochs = 3
    smoke = os.environ.get("SMOKE_EPOCHS")
    if smoke is not None:
        epochs = min(epochs, max(1, int(smoke)))

    rank_sum = np.zeros(x_val.shape[0], dtype=np.float64)
    for ensemble_index in range(5):
        scores = train_one(
            x_train,
            y_train,
            x_val,
            total_embeddings,
            args.seed + ensemble_index,
            epochs,
        )
        rank_sum += rank_transform(scores)
    final_scores = rank_sum / 5.0

    if used_npz:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    result = evaluate(users, y_val, final_scores)
    metrics = {
        "gauc": float(result["GAUC"]),
        "ndcg5": float(result["nDCG@5"]),
        "primary": float(result["primary"]),
    }

    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i in range(final_scores.shape[0]):
            writer.writerow([i, as_python_id(users[i]), as_python_id(videos[i]), float(final_scores[i])])

    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, separators=(",", ":"), sort_keys=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
        main()
