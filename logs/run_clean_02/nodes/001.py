"""FM baseline with compound regularization for late-epoch overfitting."""
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
    try:
        value = max(0, int(float(value)))
    except (TypeError, ValueError):
        return 0
    if value <= 0:
        return 0
    return min(16, int(np.log2(value)) + 1)


def build_mapping(rows, column):
    values = sorted({row[column] for row in rows})
    return {value: i + 1 for i, value in enumerate(values)}


def encode_csv_rows(rows, mappings, field_dims):
    offsets = np.cumsum(np.asarray([0] + field_dims[:-1], dtype=np.int64))
    x = np.empty((len(rows), 5), dtype=np.int64)
    users = np.empty(len(rows), dtype=object)
    videos = np.empty(len(rows), dtype=object)
    labels = np.empty(len(rows), dtype=np.float32)
    for i, row in enumerate(rows):
        raw = np.asarray([
            mappings["user_id"].get(row["user_id"], 0),
            mappings["video_id"].get(row["video_id"], 0),
            0,
            mappings["tab"].get(row["tab"], 0),
            duration_bucket(row["duration_ms"]),
        ], dtype=np.int64)
        x[i] = raw + offsets
        users[i] = row["user_id"]
        videos[i] = row["video_id"]
        labels[i] = float(row["long_view"])
    return x, labels, users, videos


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        tr_file = np.load(train_npz)
        va_file = np.load(val_npz)
        field_dims = tr_file["field_dims"].astype(np.int64)
        train_x = tr_file["X"].astype(np.int64)
        train_y = tr_file["y"].astype(np.float32)
        val_x = va_file["X"].astype(np.int64)
        val_y = va_file["y"].astype(np.float32)
        val_users = va_file["user"].copy()
        video_offset = int(field_dims[0])
        val_videos = (val_x[:, 1] - video_offset).copy()
        from data.official.evaluate import evaluate
        return train_x, train_y, val_x, val_y, val_users, val_videos, field_dims, evaluate

    with open(os.path.join(data_dir, "train.csv"), newline="") as fh:
        train_rows = list(csv.DictReader(fh))
    with open(os.path.join(data_dir, "val.csv"), newline="") as fh:
        val_rows = list(csv.DictReader(fh))
    mappings = {
        "user_id": build_mapping(train_rows, "user_id"),
        "video_id": build_mapping(train_rows, "video_id"),
        "tab": build_mapping(train_rows, "tab"),
    }
    field_dims = np.asarray([
        len(mappings["user_id"]) + 1,
        len(mappings["video_id"]) + 1,
        1,
        len(mappings["tab"]) + 1,
        17,
    ], dtype=np.int64)
    train_x, train_y, _, _ = encode_csv_rows(train_rows, mappings, field_dims.tolist())
    val_x, val_y, val_users, val_videos = encode_csv_rows(val_rows, mappings, field_dims.tolist())
    from harness.evaluate_provisional import evaluate
    return train_x, train_y, val_x, val_y, val_users, val_videos, field_dims, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()

    epochs = args.epochs
    if "SMOKE_EPOCHS" in os.environ:
        epochs = min(epochs, max(1, int(os.environ["SMOKE_EPOCHS"])))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)

    train_x, train_y, val_x, val_y, val_users, val_videos, field_dims, evaluate = load_data(args.data_dir)
    xt = torch.from_numpy(train_x)
    yt = torch.from_numpy(train_y)
    xv = torch.from_numpy(val_x)

    model = FM(int(field_dims.sum()), k=16, dropout=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[6, 8], gamma=0.5)
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
            idx = permutation[start:start + batch_size]
            batch_x = xt[idx]
            optimizer.zero_grad()
            logits = model(batch_x)
            rows = torch.unique(batch_x)
            row_l2 = (model.emb.weight[rows].square().sum() +
                      model.lin.weight[rows].square().sum())
            loss = criterion(logits, yt[idx]) + 1e-7 * row_l2
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            scores = np.concatenate([
                model(xv[start:start + 65536]).cpu().numpy()
                for start in range(0, len(xv), 65536)
            ])
        metrics = evaluate(val_users, val_y.astype(int), scores)
        primary = float(metrics["primary"])
        if primary > best_primary + 1e-6:
            best_primary = primary
            best_scores = scores.copy()
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break

    final_metrics = evaluate(val_users, val_y.astype(int), best_scores)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump({
            "gauc": float(final_metrics.get("GAUC", final_metrics.get("gauc"))),
            "ndcg5": float(final_metrics.get("nDCG@5", final_metrics.get("ndcg5"))),
            "primary": float(final_metrics["primary"]),
        }, fh)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(best_scores):
            writer.writerow([i, val_users[i], val_videos[i], format(float(score), ".6g")])


if __name__ == "__main__":
    main()
