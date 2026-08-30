import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RegularizedFM(torch.nn.Module):
    def __init__(self, total_dim, k=16, dropout=0.3):
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

    def accessed_row_l2(self, x):
        rows = torch.unique(x)
        return self.emb(rows).square().sum(1).mean()


def load_csv_data(data_dir):
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    with open(train_path, "r", newline="") as fh:
        train_rows = list(csv.DictReader(fh))
    with open(val_path, "r", newline="") as fh:
        val_rows = list(csv.DictReader(fh))

    train_durations = np.asarray(
        [float(r.get("duration_ms", 0) or 0) for r in train_rows], dtype=np.float64
    )
    quantiles = np.quantile(train_durations, np.linspace(0.0, 1.0, 11))
    duration_edges = quantiles[1:-1]

    def value(row, field):
        if field == "author_id":
            return row.get("author_id", "__UNKNOWN_AUTHOR__")
        return row.get(field, "")

    fields = ["user_id", "video_id", "author_id", "tab"]
    mappings = []
    for field in fields:
        values = sorted({value(r, field) for r in train_rows})
        mappings.append({v: i + 1 for i, v in enumerate(values)})

    field_dims = np.asarray([len(m) + 1 for m in mappings] + [10], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1])).astype(np.int64)

    def encode(rows):
        x = np.zeros((len(rows), 5), dtype=np.int64)
        for i, row in enumerate(rows):
            for j, field in enumerate(fields):
                x[i, j] = mappings[j].get(value(row, field), 0) + offsets[j]
            duration = float(row.get("duration_ms", 0) or 0)
            bucket = int(np.searchsorted(duration_edges, duration, side="right"))
            x[i, 4] = bucket + offsets[4]
        return x

    xt = encode(train_rows)
    xv = encode(val_rows)
    yt = np.asarray([float(r["long_view"]) for r in train_rows], dtype=np.float32)
    yv = np.asarray([float(r["long_view"]) for r in val_rows], dtype=np.float32)
    train_users = np.asarray([r.get("user_id", "") for r in train_rows])
    val_users = np.asarray([r.get("user_id", "") for r in val_rows])
    val_videos = np.asarray([r.get("video_id", "") for r in val_rows])
    return xt, yt, xv, yv, train_users, val_users, val_videos, field_dims


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
        train_users = tr["user"]
        val_users = va["user"]
        field_dims = tr["field_dims"].astype(np.int64)
        if "video" in va.files:
            val_videos = va["video"]
        else:
            video_offset = int(field_dims[0])
            val_videos = xv[:, 1] - video_offset
        return xt, yt, xv, yv, train_users, val_users, val_videos, field_dims, True
    data = load_csv_data(data_dir)
    return data + (False,)


def metric_value(metrics, upper, lower):
    return float(metrics[upper] if upper in metrics else metrics[lower])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    epochs = args.epochs
    smoke_epochs = os.environ.get("SMOKE_EPOCHS")
    if smoke_epochs is not None:
        epochs = min(epochs, int(smoke_epochs))
    epochs = max(1, epochs)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    xt_np, yt_np, xv_np, yv_np, _, val_users, val_videos, field_dims, fast_path = load_data(args.data_dir)

    if fast_path:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate

    xt = torch.from_numpy(xt_np)
    yt = torch.from_numpy(yt_np)
    xv = torch.from_numpy(xv_np)

    model = RegularizedFM(int(field_dims.sum()), k=16, dropout=0.3)
    optimizer = torch.optim.AdamW(
        [
            {"params": [model.emb.weight], "weight_decay": 0.0},
            {"params": [model.lin.weight, model.bias], "weight_decay": 1e-3},
        ],
        lr=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    bce = torch.nn.BCEWithLogitsLoss()

    n = len(yt)
    batch_size = 8192
    best_gauc = -1.0
    best_scores = None
    patience = 0

    for _ in range(epochs):
        model.train()
        permutation = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = permutation[start:start + batch_size]
            xb = xt[idx]
            optimizer.zero_grad()
            logits = model(xb)
            loss = bce(logits, yt[idx]) + 1e-3 * model.accessed_row_l2(xb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            scores = np.concatenate(
                [model(xv[i:i + 65536]).cpu().numpy() for i in range(0, len(xv), 65536)]
            )
        metrics = evaluate(val_users, yv_np.astype(int), scores)
        gauc = metric_value(metrics, "GAUC", "gauc")
        if gauc > best_gauc + 1e-6:
            best_gauc = gauc
            best_scores = scores.copy()
            patience = 0
        else:
            patience += 1
        scheduler.step()
        if patience >= 4:
            break

    final_metrics = evaluate(val_users, yv_np.astype(int), best_scores)
    output_metrics = {
        "gauc": metric_value(final_metrics, "GAUC", "gauc"),
        "ndcg5": metric_value(final_metrics, "nDCG@5", "ndcg5"),
        "primary": float(final_metrics["primary"]),
    }

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(output_metrics, fh)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(best_scores):
            writer.writerow([i, val_users[i], val_videos[i], format(float(score), ".8g")])


if __name__ == "__main__":
    main()
