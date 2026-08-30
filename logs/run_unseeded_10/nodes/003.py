"""Five-seed rank ensemble of the official-parity FM baseline."""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.official.evaluate import evaluate as official_evaluate
from harness.evaluate_provisional import evaluate as provisional_evaluate


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
    with open(train_path, newline="") as fh:
        for row in csv.DictReader(fh):
            train_rows.append((row["user_id"], row["video_id"], row["tab"],
                               float(row["duration_ms"]), float(row["long_view"])))

    val_rows = []
    with open(val_path, newline="") as fh:
        for row in csv.DictReader(fh):
            val_rows.append((row["user_id"], row["video_id"], row["tab"],
                             float(row["duration_ms"]), float(row["long_view"])))

    user_map = {v: i for i, v in enumerate(sorted({r[0] for r in train_rows}))}
    video_map = {v: i for i, v in enumerate(sorted({r[1] for r in train_rows}))}
    tab_map = {v: i for i, v in enumerate(sorted({r[2] for r in train_rows}))}
    durations = np.asarray([r[3] for r in train_rows], dtype=np.float64)
    edges = np.quantile(durations, np.linspace(0.0, 1.0, 11))[1:-1]

    field_dims = np.asarray([len(user_map) + 1, len(video_map) + 1, 2,
                             len(tab_map) + 1, 10], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1])))

    def encode(rows):
        x = np.empty((len(rows), 5), dtype=np.int64)
        y = np.empty(len(rows), dtype=np.float32)
        for i, (user, video, tab, duration, label) in enumerate(rows):
            x[i, 0] = user_map.get(user, len(user_map))
            x[i, 1] = video_map.get(video, len(video_map))
            x[i, 2] = 0
            x[i, 3] = tab_map.get(tab, len(tab_map))
            x[i, 4] = np.searchsorted(edges, duration, side="right")
            y[i] = label
        x += offsets
        return x, y

    xt, yt = encode(train_rows)
    xv, yv = encode(val_rows)
    users = np.asarray([r[0] for r in val_rows])
    videos = np.asarray([r[1] for r in val_rows])
    return xt, yt, xv, yv, users, videos, field_dims, provisional_evaluate


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        tr = np.load(train_npz)
        va = np.load(val_npz)
        xt = tr["X"].astype(np.int64)
        yt = tr["y"].astype(np.float32)
        xv = va["X"].astype(np.int64)
        yv = va["y"].astype(np.float32)
        users = np.asarray(va["user"])
        videos = np.zeros(len(yv), dtype=np.int64)
        field_dims = np.asarray(tr["field_dims"], dtype=np.int64)
        return xt, yt, xv, yv, users, videos, field_dims, official_evaluate
    return load_csv_data(data_dir)


def train_member(xt, yt, xv, users, yv, total_dim, seed, epochs, evaluator):
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

    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            opt.zero_grad()
            loss = bce(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            scores = np.concatenate([
                model(xv[start:start + 65536]).numpy()
                for start in range(0, len(xv), 65536)
            ])
        metrics = evaluator(users, yv.astype(int), scores)
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


def per_user_ranks(users, scores):
    order = np.lexsort((scores, users))
    sorted_users = users[order]
    starts = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1]])
    ends = np.r_[starts[1:], len(order)]
    lengths = ends - starts
    repeated_starts = np.repeat(starts, lengths)
    repeated_lengths = np.repeat(lengths, lengths)
    positions = np.arange(len(order)) - repeated_starts
    ranked = (positions + 1.0) / (repeated_lengths + 1.0)
    result = np.empty(len(order), dtype=np.float64)
    result[order] = ranked
    return result


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
        epochs = min(epochs, int(smoke_epochs))

    xt_np, yt_np, xv_np, yv, users, videos, field_dims, evaluator = load_data(args.data_dir)
    xt = torch.from_numpy(xt_np)
    yt = torch.from_numpy(yt_np)
    xv = torch.from_numpy(xv_np)
    total_dim = int(field_dims.sum())

    rank_sum = np.zeros(len(yv), dtype=np.float64)
    for member in range(5):
        scores = train_member(xt, yt, xv, users, yv, total_dim,
                              args.seed + member, epochs, evaluator)
        rank_sum += per_user_ranks(users, scores)
    ensemble_scores = rank_sum / 5.0

    metrics = evaluator(users, yv.astype(int), ensemble_scores)
    output_metrics = {
        "gauc": float(metrics["GAUC"] if "GAUC" in metrics else metrics["gauc"]),
        "ndcg5": float(metrics.get("nDCG@5", metrics.get("ndcg5"))),
        "primary": float(metrics["primary"])
    }

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(output_metrics, fh)
    with open(os.path.join(args.out_dir, "predictions.csv"), "w") as fh:
        fh.write("row_id,user_id,video_id,score\n")
        for i, score in enumerate(ensemble_scores):
            fh.write(f"{i},{users[i]},{videos[i]},{score:.9g}\n")


if __name__ == "__main__":
    main()
