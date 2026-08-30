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


def load_csv_data(data_dir):
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    train_rows = []
    with open(train_path, "r", newline="") as fh:
        for row in csv.DictReader(fh):
            train_rows.append((
                row["user_id"],
                row["video_id"],
                row.get("author_id", "__missing_author__"),
                row["tab"],
                float(row["duration_ms"]),
                float(row["long_view"]),
            ))
    durations = np.asarray([r[4] for r in train_rows], dtype=np.float64)
    quantiles = np.quantile(durations, np.arange(1, 10) / 10.0)
    field_values = [set() for _ in range(4)]
    for row in train_rows:
        for j in range(4):
            field_values[j].add(row[j])
    mappings = []
    field_dims = []
    for values in field_values:
        mapping = {v: i for i, v in enumerate(sorted(values))}
        mappings.append(mapping)
        field_dims.append(len(mapping) + 1)
    field_dims.append(10)
    offsets = np.cumsum([0] + field_dims[:-1], dtype=np.int64)

    def encode(rows):
        x = np.empty((len(rows), 5), dtype=np.int64)
        for i, row in enumerate(rows):
            for j in range(4):
                x[i, j] = offsets[j] + mappings[j].get(row[j], field_dims[j] - 1)
            bucket = int(np.searchsorted(quantiles, row[4], side="right"))
            x[i, 4] = offsets[4] + min(bucket, 9)
        return x

    Xt = encode(train_rows)
    yt = np.asarray([r[5] for r in train_rows], dtype=np.float32)
    val_rows = []
    val_users = []
    val_videos = []
    val_labels = []
    with open(val_path, "r", newline="") as fh:
        for row in csv.DictReader(fh):
            user_id = row["user_id"]
            video_id = row["video_id"]
            val_rows.append((
                user_id,
                video_id,
                row.get("author_id", "__missing_author__"),
                row["tab"],
                float(row["duration_ms"]),
                float(row["long_view"]),
            ))
            val_users.append(user_id)
            val_videos.append(video_id)
            val_labels.append(float(row["long_view"]))
    Xv = encode(val_rows)
    return {
        "Xt": Xt,
        "yt": yt,
        "Xv": Xv,
        "yv": np.asarray(val_labels, dtype=np.float32),
        "users": np.asarray(val_users),
        "videos": np.asarray(val_videos),
        "field_dims": np.asarray(field_dims, dtype=np.int64),
        "npz": False,
    }


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        tr = np.load(train_npz)
        va = np.load(val_npz)
        field_dims = tr["field_dims"].astype(np.int64)
        video_offset = int(field_dims[0])
        return {
            "Xt": tr["X"].astype(np.int64),
            "yt": tr["y"].astype(np.float32),
            "Xv": va["X"].astype(np.int64),
            "yv": va["y"].astype(np.float32),
            "users": va["user"],
            "videos": va["X"][:, 1].astype(np.int64) - video_offset,
            "field_dims": field_dims,
            "npz": True,
        }
    return load_csv_data(data_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)

    data = load_data(args.data_dir)
    Xt = torch.from_numpy(data["Xt"])
    yt = torch.from_numpy(data["yt"])
    Xv = torch.from_numpy(data["Xv"])
    total_dim = int(data["field_dims"].sum())

    if data["npz"]:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate

    model = RegularizedFM(total_dim, k=16, dropout=0.3)
    optimizer = torch.optim.AdamW([
        {"params": [model.emb.weight], "weight_decay": 0.0},
        {"params": [model.lin.weight], "weight_decay": 1e-3},
        {"params": [model.bias], "weight_decay": 0.0},
    ], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1, min_lr=3.125e-5
    )
    bce = torch.nn.BCEWithLogitsLoss()
    row_l2_weight = 1e-3
    batch_size = 8192
    n = len(yt)
    best_primary = -1.0
    best_scores = None
    patience = 0

    for _ in range(args.epochs):
        model.train()
        permutation = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = permutation[start:start + batch_size]
            xb = Xt[idx]
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            rows = torch.unique(xb)
            row_l2 = model.emb.weight[rows].pow(2).sum(1).mean()
            loss = bce(logits, yt[idx]) + row_l2_weight * row_l2
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            scores = np.concatenate([
                model(Xv[start:start + 65536]).cpu().numpy()
                for start in range(0, len(Xv), 65536)
            ])
        metrics = evaluate(data["users"], data["yv"].astype(int), scores)
        primary = float(metrics["primary"])
        scheduler.step(primary)
        if primary > best_primary + 1e-6:
            best_primary = primary
            best_scores = scores.copy()
            patience = 0
        else:
            patience += 1
            if patience >= 5:
                break

    final_metrics = evaluate(data["users"], data["yv"].astype(int), best_scores)
    gauc = final_metrics.get("GAUC", final_metrics.get("gauc"))
    ndcg5 = final_metrics.get("nDCG@5", final_metrics.get("ndcg5"))
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump({
            "gauc": float(gauc),
            "ndcg5": float(ndcg5),
            "primary": float(final_metrics["primary"]),
        }, fh)
    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(best_scores):
            writer.writerow([i, data["users"][i], data["videos"][i], format(float(score), ".8g")])


if __name__ == "__main__":
    main()
