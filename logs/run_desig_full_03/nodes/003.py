"""FM capacity ablation using k=8 embeddings over the official five fields."""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FM(torch.nn.Module):
    def __init__(self, total_dim, k=8):
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

    with open(train_path, "r", newline="", encoding="utf-8") as fh:
        train_rows = list(csv.DictReader(fh))
    with open(val_path, "r", newline="", encoding="utf-8") as fh:
        val_rows = list(csv.DictReader(fh))

    durations = np.asarray(
        [float(row.get("duration_ms", 0) or 0) for row in train_rows],
        dtype=np.float64,
    )
    quantiles = np.quantile(durations, np.arange(1, 10) / 10.0)

    field_names = ["user_id", "video_id", "author_id", "tab"]
    mappings = []
    for name in field_names:
        values = []
        for row in train_rows:
            if name == "author_id" and name not in row:
                values.append("__missing_author__")
            else:
                values.append(row.get(name, ""))
        unique = sorted(set(values))
        mappings.append({value: i for i, value in enumerate(unique)})

    field_dims = np.asarray(
        [len(mapping) + 1 for mapping in mappings] + [10], dtype=np.int64
    )
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1])).astype(np.int64)

    def encode(rows):
        x = np.empty((len(rows), 5), dtype=np.int64)
        for i, row in enumerate(rows):
            for j, name in enumerate(field_names):
                if name == "author_id" and name not in row:
                    value = "__missing_author__"
                else:
                    value = row.get(name, "")
                x[i, j] = mappings[j].get(value, len(mappings[j])) + offsets[j]
            duration = float(row.get("duration_ms", 0) or 0)
            x[i, 4] = int(np.searchsorted(quantiles, duration, side="right")) + offsets[4]
        return x

    xt = encode(train_rows)
    xv = encode(val_rows)
    yt = np.asarray([float(row["long_view"]) for row in train_rows], dtype=np.float32)
    yv = np.asarray([float(row["long_view"]) for row in val_rows], dtype=np.float32)
    train_users = np.asarray([row.get("user_id", "") for row in train_rows])
    val_users = np.asarray([row.get("user_id", "") for row in val_rows])
    val_videos = np.asarray([row.get("video_id", "") for row in val_rows])
    return xt, yt, xv, yv, train_users, val_users, val_videos, field_dims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    epochs = args.epochs
    if "SMOKE_EPOCHS" in os.environ:
        epochs = min(epochs, int(os.environ["SMOKE_EPOCHS"]))

    train_npz = os.path.join(args.data_dir, "train.npz")
    val_npz = os.path.join(args.data_dir, "val.npz")
    use_fast_path = os.path.exists(train_npz) and os.path.exists(val_npz)

    if use_fast_path:
        tr = np.load(train_npz)
        va = np.load(val_npz)
        xt_np = tr["X"].astype(np.int64)
        yt_np = tr["y"].astype(np.float32)
        xv_np = va["X"].astype(np.int64)
        yv_np = va["y"].astype(np.float32)
        val_users = va["user"]
        field_dims = tr["field_dims"].astype(np.int64)
        video_offset = int(field_dims[0])
        val_videos = xv_np[:, 1] - video_offset
        from data.official.evaluate import evaluate
    else:
        (
            xt_np,
            yt_np,
            xv_np,
            yv_np,
            _train_users,
            val_users,
            val_videos,
            field_dims,
        ) = load_csv_data(args.data_dir)
        from harness.evaluate_provisional import evaluate

    total_dim = int(field_dims.sum())
    xt = torch.from_numpy(xt_np)
    yt = torch.from_numpy(yt_np)
    xv = torch.from_numpy(xv_np)

    model = FM(total_dim, k=8)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = torch.nn.BCEWithLogitsLoss()
    n = len(yt)
    batch_size = 8192
    best = -1.0
    best_scores = None
    patience = 0

    for _epoch in range(epochs):
        model.train()
        permutation = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = permutation[start:start + batch_size]
            opt.zero_grad()
            loss = bce(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            scores = np.concatenate(
                [
                    model(xv[start:start + 65536]).cpu().numpy()
                    for start in range(0, len(xv), 65536)
                ]
            )
        metrics = evaluate(val_users, yv_np.astype(int), scores)
        primary = float(metrics["primary"])
        if primary > best + 1e-6:
            best = primary
            best_scores = scores.copy()
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break

    if best_scores is None:
        model.eval()
        with torch.no_grad():
            best_scores = np.concatenate(
                [
                    model(xv[start:start + 65536]).cpu().numpy()
                    for start in range(0, len(xv), 65536)
                ]
            )

    metrics = evaluate(val_users, yv_np.astype(int), best_scores)
    result = {
        "gauc": float(metrics.get("GAUC", metrics.get("gauc"))),
        "ndcg5": float(metrics.get("nDCG@5", metrics.get("ndcg5"))),
        "primary": float(metrics["primary"]),
    }

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh)

    with open(
        os.path.join(args.out_dir, "predictions.csv"),
        "w",
        newline="",
        encoding="utf-8",
    ) as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(best_scores):
            writer.writerow([i, val_users[i], val_videos[i], format(float(score), ".6g")])


if __name__ == "__main__":
    main()
