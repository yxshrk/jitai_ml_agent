import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FM(torch.nn.Module):
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


def load_csv_data(data_dir):
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")

    with open(train_path, "r", newline="") as fh:
        train_rows = list(csv.DictReader(fh))
    with open(val_path, "r", newline="") as fh:
        val_rows = list(csv.DictReader(fh))

    train_duration = np.asarray(
        [float(row.get("duration_ms", 0) or 0) for row in train_rows],
        dtype=np.float64,
    )
    quantiles = np.quantile(train_duration, np.linspace(0.0, 1.0, 11))
    inner_edges = quantiles[1:-1]

    def raw_fields(row):
        duration = float(row.get("duration_ms", 0) or 0)
        author = row.get("author_id", "0") or "0"
        return (
            row.get("user_id", ""),
            row.get("video_id", ""),
            author,
            row.get("tab", ""),
            str(int(np.searchsorted(inner_edges, duration, side="right"))),
        )

    train_raw = [raw_fields(row) for row in train_rows]
    val_raw = [raw_fields(row) for row in val_rows]
    maps = []
    field_dims = []
    for field in range(5):
        values = sorted({row[field] for row in train_raw})
        mapping = {value: i for i, value in enumerate(values)}
        maps.append(mapping)
        field_dims.append(len(mapping) + 1)

    offsets = np.cumsum([0] + field_dims[:-1], dtype=np.int64)

    def encode(rows):
        x = np.empty((len(rows), 5), dtype=np.int64)
        for i, row in enumerate(rows):
            for field in range(5):
                local = maps[field].get(row[field], len(maps[field]))
                x[i, field] = local + offsets[field]
        return x

    xt = encode(train_raw)
    xv = encode(val_raw)
    yt = np.asarray([float(row["long_view"]) for row in train_rows], dtype=np.float32)
    yv = np.asarray([float(row["long_view"]) for row in val_rows], dtype=np.float32)
    users = np.asarray([row.get("user_id", "") for row in val_rows])
    videos = np.asarray([row.get("video_id", "") for row in val_rows])
    return xt, yt, xv, yv, users, videos, np.asarray(field_dims, dtype=np.int64), False


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        tr = np.load(train_npz)
        va = np.load(val_npz)
        xt = tr["X"].astype(np.int64, copy=False)
        yt = tr["y"].astype(np.float32, copy=False)
        xv = va["X"].astype(np.int64, copy=False)
        yv = va["y"].astype(np.float32, copy=False)
        users = va["user"]
        field_dims = tr["field_dims"].astype(np.int64, copy=False)
        video_offset = int(field_dims[0])
        videos = xv[:, 1] - video_offset
        return xt, yt, xv, yv, users, videos, field_dims, True
    return load_csv_data(data_dir)


def train_member(xt, yt, xv, yv, users, total_dim, seed, epochs, evaluate_fn):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = FM(total_dim, k=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()
    n = len(yt)
    batch_size = 8192
    best_primary = -1.0
    best_scores = None
    patience = 0

    for _ in range(epochs):
        model.train()
        permutation = torch.randperm(n)
        for start in range(0, n, batch_size):
            index = permutation[start:start + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(xt[index]), yt[index])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            scores = np.concatenate([
                model(xv[start:start + 65536]).cpu().numpy()
                for start in range(0, len(xv), 65536)
            ])
        metrics = evaluate_fn(users, yv.astype(int), scores)
        primary = float(metrics["primary"])
        if primary > best_primary + 1e-6:
            best_primary = primary
            best_scores = scores.copy()
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break
    return best_scores


def per_user_rank_average(users, member_scores):
    users_array = np.asarray(users)
    _, inverse = np.unique(users_array, return_inverse=True)
    groups = [[] for _ in range(int(inverse.max()) + 1)]
    for row_index, group_index in enumerate(inverse):
        groups[int(group_index)].append(row_index)

    result = np.zeros(len(users_array), dtype=np.float64)
    for scores in member_scores:
        ranked = np.zeros(len(users_array), dtype=np.float64)
        for indices_list in groups:
            indices = np.asarray(indices_list, dtype=np.int64)
            if len(indices) == 1:
                ranked[indices[0]] = 0.5
                continue
            order = np.argsort(scores[indices], kind="mergesort")
            ranks = np.empty(len(indices), dtype=np.float64)
            ranks[order] = np.arange(len(indices), dtype=np.float64)
            ranked[indices] = ranks / float(len(indices) - 1)
        result += ranked
    return result / float(len(member_scores))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    epochs = max(1, args.epochs)
    smoke_epochs = os.environ.get("SMOKE_EPOCHS")
    if smoke_epochs is not None:
        epochs = min(epochs, max(1, int(smoke_epochs)))

    xt_np, yt_np, xv_np, yv, users, videos, field_dims, fast_path = load_data(args.data_dir)
    xt = torch.from_numpy(xt_np)
    yt = torch.from_numpy(yt_np)
    xv = torch.from_numpy(xv_np)

    if fast_path:
        from data.official.evaluate import evaluate as evaluate_fn
    else:
        from harness.evaluate_provisional import evaluate as evaluate_fn

    total_dim = int(field_dims.sum())
    member_scores = []
    for member_index in range(3):
        scores = train_member(
            xt, yt, xv, yv, users, total_dim,
            args.seed + member_index, epochs, evaluate_fn,
        )
        member_scores.append(scores)

    ensemble_scores = per_user_rank_average(users, member_scores)
    metrics = evaluate_fn(users, yv.astype(int), ensemble_scores)
    output_metrics = {
        "gauc": float(metrics.get("GAUC", metrics.get("gauc"))),
        "ndcg5": float(metrics.get("nDCG@5", metrics.get("ndcg5"))),
        "primary": float(metrics["primary"]),
    }

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(output_metrics, fh)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(ensemble_scores):
            writer.writerow([i, users[i], videos[i], format(float(score), ".12g")])


if __name__ == "__main__":
    main()
