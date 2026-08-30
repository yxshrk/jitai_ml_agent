"""Regularized FM with dropout, accessed-row L2, AdamW, and rapid LR decay."""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RegularizedFM(torch.nn.Module):
    def __init__(self, total_dim, k=16, dropout=0.30):
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

    train_fields = []
    train_y = []
    train_durations = []
    with open(train_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        has_author = "author_id" in (reader.fieldnames or [])
        for row in reader:
            video = row["video_id"]
            author = row["author_id"] if has_author else video
            train_fields.append((row["user_id"], video, author, row["tab"]))
            train_durations.append(float(row["duration_ms"]))
            train_y.append(float(row["long_view"]))

    train_durations = np.asarray(train_durations, dtype=np.float64)
    quantiles = np.quantile(train_durations, np.arange(1, 10) / 10.0)
    quantiles = np.maximum.accumulate(quantiles)

    val_fields = []
    val_y = []
    val_durations = []
    val_users = []
    val_videos = []
    with open(val_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        has_author = "author_id" in (reader.fieldnames or [])
        for row in reader:
            video = row["video_id"]
            author = row["author_id"] if has_author else video
            val_fields.append((row["user_id"], video, author, row["tab"]))
            val_durations.append(float(row["duration_ms"]))
            val_y.append(float(row["long_view"]))
            val_users.append(row["user_id"])
            val_videos.append(video)

    train_columns = list(zip(*train_fields))
    val_columns = list(zip(*val_fields))
    encoded_train = []
    encoded_val = []
    field_dims = []
    for train_col, val_col in zip(train_columns, val_columns):
        values = sorted(set(train_col))
        mapping = {value: idx for idx, value in enumerate(values)}
        unknown = len(values)
        encoded_train.append(np.asarray([mapping[v] for v in train_col], dtype=np.int64))
        encoded_val.append(np.asarray([mapping.get(v, unknown) for v in val_col], dtype=np.int64))
        field_dims.append(len(values) + 1)

    train_bucket = np.searchsorted(quantiles, train_durations, side="right").astype(np.int64)
    val_bucket = np.searchsorted(quantiles, np.asarray(val_durations), side="right").astype(np.int64)
    encoded_train.append(train_bucket)
    encoded_val.append(val_bucket)
    field_dims.append(10)

    offsets = np.cumsum([0] + field_dims[:-1], dtype=np.int64)
    x_train = np.stack(encoded_train, axis=1) + offsets
    x_val = np.stack(encoded_val, axis=1) + offsets
    return {
        "X_train": x_train,
        "y_train": np.asarray(train_y, dtype=np.float32),
        "X_val": x_val,
        "y_val": np.asarray(val_y, dtype=np.float32),
        "users": np.asarray(val_users),
        "videos": np.asarray(val_videos),
        "field_dims": np.asarray(field_dims, dtype=np.int64),
        "fast_path": False,
    }


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        tr = np.load(train_npz)
        va = np.load(val_npz)
        return {
            "X_train": tr["X"].astype(np.int64),
            "y_train": tr["y"].astype(np.float32),
            "X_val": va["X"].astype(np.int64),
            "y_val": va["y"].astype(np.float32),
            "users": va["user"],
            "videos": va["X"][:, 1],
            "field_dims": tr["field_dims"].astype(np.int64),
            "fast_path": True,
        }
    return load_csv_data(data_dir)


def metric_values(metrics):
    return (
        float(metrics.get("GAUC", metrics.get("gauc"))),
        float(metrics.get("nDCG@5", metrics.get("ndcg5"))),
        float(metrics["primary"]),
    )


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

    epochs = args.epochs
    smoke_epochs = os.environ.get("SMOKE_EPOCHS")
    if smoke_epochs is not None:
        epochs = min(epochs, max(1, int(smoke_epochs)))

    data = load_data(args.data_dir)
    if data["fast_path"]:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate

    x_train = torch.from_numpy(data["X_train"])
    y_train = torch.from_numpy(data["y_train"])
    x_val = torch.from_numpy(data["X_val"])
    total_dim = int(data["field_dims"].sum())

    model = RegularizedFM(total_dim=total_dim, k=16, dropout=0.30)
    optimizer = torch.optim.AdamW(
        [
            {"params": [model.emb.weight], "weight_decay": 0.0},
            {"params": [model.lin.weight, model.bias], "weight_decay": 1e-3},
        ],
        lr=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    bce = torch.nn.BCEWithLogitsLoss()

    n = len(y_train)
    batch_size = 8192
    row_l2_weight = 3e-3
    best_gauc = -1.0
    best_scores = None
    patience = 0

    for _ in range(epochs):
        model.train()
        permutation = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = permutation[start:start + batch_size]
            xb = x_train[idx]
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            accessed_rows = torch.unique(xb)
            row_l2 = model.emb(accessed_rows).pow(2).sum(1).mean()
            loss = bce(logits, y_train[idx]) + row_l2_weight * row_l2
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            scores = np.concatenate([
                model(x_val[start:start + 65536]).cpu().numpy()
                for start in range(0, len(x_val), 65536)
            ])
        metrics = evaluate(data["users"], data["y_val"].astype(int), scores)
        gauc, _, _ = metric_values(metrics)
        if gauc > best_gauc + 1e-7:
            best_gauc = gauc
            best_scores = scores.copy()
            patience = 0
        else:
            patience += 1
        scheduler.step()
        if patience >= 4:
            break

    final_metrics = evaluate(data["users"], data["y_val"].astype(int), best_scores)
    gauc, ndcg5, primary = metric_values(final_metrics)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump({"gauc": gauc, "ndcg5": ndcg5, "primary": primary}, fh)
    with open(os.path.join(args.out_dir, "predictions.csv"), "w") as fh:
        fh.write("row_id,user_id,video_id,score\n")
        for row_id, (user, video, score) in enumerate(zip(data["users"], data["videos"], best_scores)):
            fh.write(f"{row_id},{user},{video},{float(score):.9g}\n")


if __name__ == "__main__":
    main()
