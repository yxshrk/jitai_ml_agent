"""Three-seed rank ensemble of the official-parity k=16 FM baseline."""
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
    with open(train_path, newline="") as fh:
        train_rows = list(csv.DictReader(fh))
    with open(val_path, newline="") as fh:
        val_rows = list(csv.DictReader(fh))

    fields = ("user_id", "video_id", "author_id", "tab")
    maps = {}
    for field in fields:
        values = []
        for row in train_rows:
            values.append(row.get(field, "__missing__"))
        maps[field] = {value: i + 1 for i, value in enumerate(sorted(set(values)))}

    train_duration = np.asarray(
        [float(row.get("duration_ms", 0.0) or 0.0) for row in train_rows],
        dtype=np.float64,
    )
    quantiles = np.quantile(train_duration, np.arange(1, 10) / 10.0)

    field_dims = np.asarray(
        [len(maps[field]) + 1 for field in fields] + [10], dtype=np.int64
    )
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1])).astype(np.int64)

    def encode(rows):
        x = np.empty((len(rows), 5), dtype=np.int64)
        for i, row in enumerate(rows):
            for j, field in enumerate(fields):
                x[i, j] = maps[field].get(row.get(field, "__missing__"), 0)
            duration = float(row.get("duration_ms", 0.0) or 0.0)
            x[i, 4] = int(np.searchsorted(quantiles, duration, side="right"))
        x += offsets[None, :]
        return x

    xt = encode(train_rows)
    xv = encode(val_rows)
    yt = np.asarray([float(row["long_view"]) for row in train_rows], dtype=np.float32)
    yv = np.asarray([float(row["long_view"]) for row in val_rows], dtype=np.float32)
    users = np.asarray([row["user_id"] for row in val_rows])
    videos = np.asarray([row["video_id"] for row in val_rows])
    return xt, yt, xv, yv, users, videos, field_dims


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        with np.load(train_npz) as tr:
            xt = tr["X"].astype(np.int64, copy=True)
            yt = tr["y"].astype(np.float32, copy=True)
            field_dims = tr["field_dims"].astype(np.int64, copy=True)
        with np.load(val_npz) as va:
            xv = va["X"].astype(np.int64, copy=True)
            yv = va["y"].astype(np.float32, copy=True)
            users = va["user"].copy()
        video_offset = int(field_dims[0])
        videos = xv[:, 1] - video_offset
        return xt, yt, xv, yv, users, videos, field_dims, True
    data = load_csv_data(data_dir)
    return (*data, False)


def metric_values(result):
    return (
        float(result["GAUC"] if "GAUC" in result else result["gauc"]),
        float(result["nDCG@5"] if "nDCG@5" in result else result["ndcg5"]),
        float(result["primary"]),
    )


def predict(model, xv):
    model.eval()
    with torch.no_grad():
        return np.concatenate(
            [model(xv[i:i + 65536]).cpu().numpy() for i in range(0, len(xv), 65536)]
        ).astype(np.float64)


def train_member(xt, yt, xv, yv, users, total_dim, seed, epochs, evaluator):
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

        scores = predict(model, xv)
        result = evaluator(users, yv.astype(int), scores)
        primary = float(result["primary"])
        if primary > best_primary + 1e-6:
            best_primary = primary
            best_scores = scores.copy()
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break

    if best_scores is None:
        best_scores = predict(model, xv)
    return best_scores


def per_user_ranks(users, scores):
    ranks = np.empty(len(scores), dtype=np.float64)
    groups = {}
    for i, user in enumerate(users):
        key = user.item() if isinstance(user, np.generic) else user
        groups.setdefault(key, []).append(i)
    for indices in groups.values():
        idx = np.asarray(indices, dtype=np.int64)
        if len(idx) == 1:
            ranks[idx[0]] = 0.5
        else:
            order = np.argsort(scores[idx], kind="mergesort")
            local = np.empty(len(idx), dtype=np.float64)
            local[order] = np.arange(len(idx), dtype=np.float64) / float(len(idx) - 1)
            ranks[idx] = local
    return ranks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()

    epochs = args.epochs
    smoke = os.environ.get("SMOKE_EPOCHS")
    if smoke is not None:
        epochs = min(epochs, int(smoke))
    epochs = max(1, epochs)

    xt_np, yt_np, xv_np, yv, users, videos, field_dims, fast_path = load_data(args.data_dir)
    xt = torch.from_numpy(xt_np)
    yt = torch.from_numpy(yt_np)
    xv = torch.from_numpy(xv_np)

    if fast_path:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate

    total_dim = int(field_dims.sum())
    rank_sum = np.zeros(len(yv), dtype=np.float64)
    for member in range(3):
        scores = train_member(
            xt, yt, xv, yv, users, total_dim, args.seed + member, epochs, evaluate
        )
        rank_sum += per_user_ranks(users, scores)
    ensemble_scores = rank_sum / 3.0

    result = evaluate(users, yv.astype(int), ensemble_scores)
    gauc, ndcg5, primary = metric_values(result)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump({"gauc": gauc, "ndcg5": ndcg5, "primary": primary}, fh)
    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(ensemble_scores):
            writer.writerow([i, users[i], videos[i], format(float(score), ".10g")])


if __name__ == "__main__":
    main()
