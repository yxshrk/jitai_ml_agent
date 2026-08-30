import argparse
import csv
import json
import os
import sys

import numpy as np
import torch


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

    train_rows = []
    durations = []
    with open(train_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            duration = float(row.get("duration_ms", 0) or 0)
            train_rows.append((
                row["user_id"],
                row["video_id"],
                row.get("author_id", "0"),
                row.get("tab", "0"),
                duration,
                float(row["long_view"]),
            ))
            durations.append(duration)

    duration_array = np.asarray(durations, dtype=np.float64)
    quantiles = np.quantile(duration_array, np.linspace(0.0, 1.0, 11))
    duration_edges = quantiles[1:-1]

    maps = [{}, {}, {}, {}]
    for row in train_rows:
        values = row[:4]
        for field, value in enumerate(values):
            if value not in maps[field]:
                maps[field][value] = len(maps[field]) + 1

    field_dims = np.asarray(
        [len(maps[0]) + 1, len(maps[1]) + 1, len(maps[2]) + 1,
         len(maps[3]) + 1, 10],
        dtype=np.int64,
    )
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1])).astype(np.int64)

    def encode(rows):
        x = np.empty((len(rows), 5), dtype=np.int64)
        y = np.empty(len(rows), dtype=np.float32)
        users = []
        videos = []
        for i, row in enumerate(rows):
            for field in range(4):
                x[i, field] = maps[field].get(row[field], 0) + offsets[field]
            bucket = int(np.searchsorted(duration_edges, row[4], side="right"))
            x[i, 4] = min(bucket, 9) + offsets[4]
            y[i] = row[5]
            users.append(row[0])
            videos.append(row[1])
        return x, y, np.asarray(users), np.asarray(videos)

    val_rows = []
    with open(val_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            val_rows.append((
                row["user_id"],
                row["video_id"],
                row.get("author_id", "0"),
                row.get("tab", "0"),
                float(row.get("duration_ms", 0) or 0),
                float(row["long_view"]),
            ))

    xt, yt, _, _ = encode(train_rows)
    xv, yv, users, videos = encode(val_rows)
    return xt, yt, xv, yv, users, videos, field_dims, False


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        tr = np.load(train_npz)
        va = np.load(val_npz)
        xv = va["X"].astype(np.int64)
        videos = xv[:, 1].copy()
        return (
            tr["X"].astype(np.int64),
            tr["y"].astype(np.float32),
            xv,
            va["y"].astype(np.float32),
            va["user"],
            videos,
            tr["field_dims"].astype(np.int64),
            True,
        )
    return load_csv_data(data_dir)


def per_user_ranks(users, scores):
    scores = np.asarray(scores, dtype=np.float64)
    ranks = np.empty(len(scores), dtype=np.float64)
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_users[end] == sorted_users[start]:
            end += 1
        indices = order[start:end]
        count = end - start
        if count == 1:
            ranks[indices[0]] = 0.5
        else:
            local_order = np.argsort(scores[indices], kind="stable")
            local_ranks = np.empty(count, dtype=np.float64)
            local_ranks[local_order] = np.arange(count, dtype=np.float64) / (count - 1)
            ranks[indices] = local_ranks
        start = end
    return ranks


def train_member(xt, yt, xv, users, yv, total_dim, epochs, seed, evaluate):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = FM(total_dim, k=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()
    generator = torch.Generator()
    generator.manual_seed(seed)
    n = len(yt)
    batch_size = 8192
    best_primary = -1.0
    best_scores = None
    patience = 0

    for _ in range(epochs):
        model.train()
        permutation = torch.randperm(n, generator=generator)
        for start in range(0, n, batch_size):
            idx = permutation[start:start + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(xt[idx]), yt[idx])
            loss.backward()
            optimizer.step()

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

    return best_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--n-members", type=int, default=5)
    args = parser.parse_args()

    epochs = args.epochs
    smoke_epochs = os.environ.get("SMOKE_EPOCHS")
    if smoke_epochs is not None:
        epochs = min(epochs, int(smoke_epochs))
    epochs = max(1, epochs)

    xt_np, yt_np, xv_np, yv, users, videos, field_dims, fast_path = load_data(args.data_dir)
    if fast_path:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate

    xt = torch.from_numpy(xt_np)
    yt = torch.from_numpy(yt_np)
    xv = torch.from_numpy(xv_np)
    total_dim = int(field_dims.sum())
    ensemble_ranks = np.zeros(len(xv_np), dtype=np.float64)

    for member in range(args.n_members):
        member_scores = train_member(
            xt, yt, xv, users, yv, total_dim, epochs,
            args.seed + member, evaluate,
        )
        ensemble_ranks += per_user_ranks(users, member_scores)

    ensemble_scores = ensemble_ranks / args.n_members
    metrics = evaluate(users, yv.astype(int), ensemble_scores)
    output_metrics = {
        "gauc": float(metrics["GAUC"] if "GAUC" in metrics else metrics["gauc"]),
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
            writer.writerow([i, users[i], videos[i], format(float(score), ".10g")])


if __name__ == "__main__":
    main()
