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

    train_rows = []
    with open(train_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            train_rows.append({
                "user_id": row["user_id"],
                "video_id": row["video_id"],
                "author_id": row.get("author_id", "__missing_author__"),
                "tab": row["tab"],
                "duration_ms": float(row["duration_ms"]),
                "long_view": float(row["long_view"]),
            })

    val_rows = []
    with open(val_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            val_rows.append({
                "user_id": row["user_id"],
                "video_id": row["video_id"],
                "author_id": row.get("author_id", "__missing_author__"),
                "tab": row["tab"],
                "duration_ms": float(row["duration_ms"]),
                "long_view": float(row["long_view"]),
            })

    train_durations = np.asarray([r["duration_ms"] for r in train_rows], dtype=np.float64)
    quantiles = np.quantile(train_durations, np.linspace(0.1, 0.9, 9))

    field_names = ("user_id", "video_id", "author_id", "tab")
    mappings = []
    for field in field_names:
        values = sorted({r[field] for r in train_rows})
        mappings.append({value: i + 1 for i, value in enumerate(values)})

    field_dims = np.asarray([len(m) + 1 for m in mappings] + [10], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1])).astype(np.int64)

    def encode(rows):
        x = np.empty((len(rows), 5), dtype=np.int64)
        for i, row in enumerate(rows):
            for j, field in enumerate(field_names):
                x[i, j] = mappings[j].get(row[field], 0) + offsets[j]
            bucket = int(np.searchsorted(quantiles, row["duration_ms"], side="right"))
            x[i, 4] = bucket + offsets[4]
        return x

    xt = encode(train_rows)
    xv = encode(val_rows)
    yt = np.asarray([r["long_view"] for r in train_rows], dtype=np.float32)
    yv = np.asarray([r["long_view"] for r in val_rows], dtype=np.float32)
    users = np.asarray([r["user_id"] for r in val_rows])
    videos = np.asarray([r["video_id"] for r in val_rows])
    return xt, yt, xv, yv, users, videos, field_dims


def per_user_ranks(users, scores):
    users = np.asarray(users)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    result = np.empty(n, dtype=np.float64)
    indices = np.arange(n, dtype=np.int64)
    order = np.lexsort((indices, scores, users))
    sorted_users = users[order]
    boundaries = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [n]))
    for start, end in zip(starts, ends):
        size = end - start
        if size == 1:
            result[order[start]] = 0.5
        else:
            result[order[start:end]] = np.arange(size, dtype=np.float64) / float(size - 1)
    return result


def train_member(xt, yt, xv, val_users, val_labels, total_dim, seed, epochs, evaluate):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = FM(total_dim, k=16)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = torch.nn.BCEWithLogitsLoss()
    n = len(yt)
    batch_size = 8192
    best_primary = -1.0
    best_scores = None
    patience = 0

    if epochs <= 0:
        model.eval()
        with torch.no_grad():
            return np.concatenate([
                model(xv[i:i + 65536]).numpy()
                for i in range(0, len(xv), 65536)
            ])

    for _ in range(epochs):
        model.train()
        permutation = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = permutation[i:i + batch_size]
            opt.zero_grad()
            loss = bce(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            scores = np.concatenate([
                model(xv[i:i + 65536]).numpy()
                for i in range(0, len(xv), 65536)
            ])
        metrics = evaluate(val_users, val_labels.astype(int), scores)
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
    args = parser.parse_args()

    epochs = args.epochs
    smoke_epochs = os.environ.get("SMOKE_EPOCHS")
    if smoke_epochs is not None:
        epochs = min(epochs, max(0, int(smoke_epochs)))

    train_npz = os.path.join(args.data_dir, "train.npz")
    val_npz = os.path.join(args.data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        from data.official.evaluate import evaluate

        tr = np.load(train_npz)
        va = np.load(val_npz)
        xt_np = tr["X"].astype(np.int64)
        yt_np = tr["y"].astype(np.float32)
        xv_np = va["X"].astype(np.int64)
        yv = va["y"].astype(np.float32)
        users = va["user"]
        field_dims = tr["field_dims"].astype(np.int64)
        video_offset = int(field_dims[0])
        videos = xv_np[:, 1] - video_offset
    else:
        from harness.evaluate_provisional import evaluate

        xt_np, yt_np, xv_np, yv, users, videos, field_dims = load_csv_data(args.data_dir)

    xt = torch.from_numpy(xt_np)
    yt = torch.from_numpy(yt_np)
    xv = torch.from_numpy(xv_np)
    total_dim = int(field_dims.sum())

    rank_sum = np.zeros(len(xv_np), dtype=np.float64)
    for member in range(5):
        scores = train_member(
            xt, yt, xv, users, yv, total_dim,
            args.seed + member, epochs, evaluate
        )
        rank_sum += per_user_ranks(users, scores)
    ensemble_scores = rank_sum / 5.0

    metrics = evaluate(users, yv.astype(int), ensemble_scores)
    gauc = metrics["GAUC"] if "GAUC" in metrics else metrics["gauc"]
    ndcg5 = metrics["nDCG@5"] if "nDCG@5" in metrics else metrics["ndcg5"]

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump({
            "gauc": float(gauc),
            "ndcg5": float(ndcg5),
            "primary": float(metrics["primary"]),
        }, fh)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(ensemble_scores):
            writer.writerow([i, users[i], videos[i], format(float(score), ".9g")])


if __name__ == "__main__":
    main()
