"""Regularized FM with dropout, accessed-row L2, weight decay, and LR decay."""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FM(torch.nn.Module):
    def __init__(self, total_dim, k=16, dropout=0.05):
        super().__init__()
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.dropout = torch.nn.Dropout(dropout)
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)

    def forward(self, x):
        e = self.dropout(self.emb(x))
        s = e.sum(1)
        pair = 0.5 * (s * s - (e * e).sum(1)).sum(1)
        return self.bias + self.lin(x).sum((1, 2)) + pair


def duration_bucket(value):
    d = max(0, int(float(value)))
    if d < 10000:
        return "0"
    if d < 30000:
        return "1"
    if d < 60000:
        return "2"
    if d < 120000:
        return "3"
    if d < 300000:
        return "4"
    return "5"


def read_csv_rows(path, training):
    rows = []
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            item = {
                "user": row["user_id"],
                "video": row["video_id"],
                "tab": row["tab"],
                "dur": duration_bucket(row["duration_ms"]),
            }
            if training:
                item["y"] = float(row["long_view"])
            else:
                item["y"] = float(row["long_view"])
            rows.append(item)
    return rows


def encode_csv(train_rows, val_rows):
    maps = []
    for key in ("user", "video"):
        values = sorted({r[key] for r in train_rows})
        maps.append({v: i + 1 for i, v in enumerate(values)})
    maps.append({"constant": 0})
    for key in ("tab", "dur"):
        values = sorted({r[key] for r in train_rows})
        maps.append({v: i + 1 for i, v in enumerate(values)})

    field_dims = np.asarray([
        len(maps[0]) + 1,
        len(maps[1]) + 1,
        1,
        len(maps[3]) + 1,
        len(maps[4]) + 1,
    ], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1]))

    def encode(rows):
        x = np.empty((len(rows), 5), dtype=np.int64)
        for i, row in enumerate(rows):
            x[i, 0] = maps[0].get(row["user"], 0) + offsets[0]
            x[i, 1] = maps[1].get(row["video"], 0) + offsets[1]
            x[i, 2] = offsets[2]
            x[i, 3] = maps[3].get(row["tab"], 0) + offsets[3]
            x[i, 4] = maps[4].get(row["dur"], 0) + offsets[4]
        return x

    return encode(train_rows), encode(val_rows), field_dims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)

    epochs = args.epochs
    smoke = os.environ.get("SMOKE_EPOCHS")
    if smoke is not None:
        epochs = min(epochs, max(1, int(smoke)))

    train_npz = os.path.join(args.data_dir, "train.npz")
    val_npz = os.path.join(args.data_dir, "val.npz")
    use_npz = os.path.exists(train_npz) and os.path.exists(val_npz)

    if use_npz:
        from data.official.evaluate import evaluate

        tr = np.load(train_npz)
        va = np.load(val_npz)
        x_train_np = tr["X"].astype(np.int64)
        y_train_np = tr["y"].astype(np.float32)
        x_val_np = va["X"].astype(np.int64)
        y_val_np = va["y"].astype(np.int64)
        val_users = va["user"]
        field_dims = tr["field_dims"].astype(np.int64)
        video_offset = int(field_dims[0])
        val_videos = x_val_np[:, 1] - video_offset
    else:
        from harness.evaluate_provisional import evaluate

        train_rows = read_csv_rows(os.path.join(args.data_dir, "train.csv"), True)
        val_rows = read_csv_rows(os.path.join(args.data_dir, "val.csv"), False)
        x_train_np, x_val_np, field_dims = encode_csv(train_rows, val_rows)
        y_train_np = np.asarray([r["y"] for r in train_rows], dtype=np.float32)
        y_val_np = np.asarray([int(r["y"]) for r in val_rows], dtype=np.int64)
        val_users = np.asarray([r["user"] for r in val_rows])
        val_videos = np.asarray([r["video"] for r in val_rows])

    x_train = torch.from_numpy(x_train_np)
    y_train = torch.from_numpy(y_train_np)
    x_val = torch.from_numpy(x_val_np)

    model = FM(int(field_dims.sum()), k=16, dropout=0.05)
    optimizer = torch.optim.AdamW([
        {"params": [model.emb.weight], "weight_decay": 0.0},
        {"params": [model.lin.weight, model.bias], "weight_decay": 1e-4},
    ], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    bce = torch.nn.BCEWithLogitsLoss()

    n = len(y_train)
    batch_size = 8192
    best_primary = -1.0
    best_scores = None
    patience = 0

    for _ in range(epochs):
        model.train()
        permutation = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = permutation[start:start + batch_size]
            xb = x_train[idx]
            optimizer.zero_grad()
            logits = model(xb)
            row_l2 = model.emb(xb).square().sum(dim=2).mean()
            loss = bce(logits, y_train[idx]) + 1e-4 * row_l2
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            scores = np.concatenate([
                model(x_val[i:i + 65536]).cpu().numpy()
                for i in range(0, len(x_val), 65536)
            ])
        metrics = evaluate(val_users, y_val_np, scores)
        primary = float(metrics["primary"])
        if primary > best_primary + 1e-6:
            best_primary = primary
            best_scores = scores.copy()
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break

    os.makedirs(args.out_dir, exist_ok=True)
    metrics = evaluate(val_users, y_val_np, best_scores)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump({
            "gauc": float(metrics.get("GAUC", metrics.get("gauc"))),
            "ndcg5": float(metrics.get("nDCG@5", metrics.get("ndcg5"))),
            "primary": float(metrics["primary"]),
        }, fh)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(best_scores):
            writer.writerow([i, val_users[i], val_videos[i], format(float(score), ".9g")])


if __name__ == "__main__":
    main()
