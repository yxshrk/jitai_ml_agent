"""DCNv2-lite with embedding dropout to reduce early overfitting."""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DCNv2Lite(torch.nn.Module):
    def __init__(self, total_dim, num_fields=5, k=16, hidden=128, dropout=0.15):
        super().__init__()
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.embedding_dropout = torch.nn.Dropout(dropout)
        input_dim = num_fields * k
        self.cross = torch.nn.Linear(input_dim, input_dim)
        self.deep = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Linear(input_dim + hidden, 1, bias=False)
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)
        torch.nn.init.xavier_uniform_(self.cross.weight)
        torch.nn.init.zeros_(self.cross.bias)
        torch.nn.init.xavier_uniform_(self.deep[0].weight)
        torch.nn.init.zeros_(self.deep[0].bias)
        torch.nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x):
        x0 = self.embedding_dropout(self.emb(x).flatten(1))
        cross = x0 * self.cross(x0) + x0
        deep = self.deep(x0)
        interaction = self.out(torch.cat((cross, deep), dim=1)).squeeze(1)
        first_order = self.lin(x).sum((1, 2))
        return self.bias + first_order + interaction


def read_csv_split(path, training):
    rows = []
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            item = {
                "user_id": row["user_id"],
                "video_id": row["video_id"],
                "author_id": row.get("author_id", "__constant_author__"),
                "tab": row["tab"],
                "duration_ms": float(row["duration_ms"]),
                "long_view": float(row["long_view"]),
            }
            rows.append(item)
    return rows


def encode_csv(train_rows, val_rows):
    categorical = ["user_id", "video_id", "author_id", "tab"]
    mappings = []
    for name in categorical:
        mapping = {}
        for row in train_rows:
            value = row[name]
            if value not in mapping:
                mapping[value] = len(mapping) + 1
        mappings.append(mapping)

    train_duration = np.asarray(
        [row["duration_ms"] for row in train_rows], dtype=np.float64
    )
    edges = np.quantile(train_duration, np.linspace(0.1, 0.9, 9))

    field_dims = np.asarray(
        [len(mapping) + 1 for mapping in mappings] + [10], dtype=np.int64
    )
    offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(field_dims)[:-1]))

    def transform(rows):
        x = np.empty((len(rows), 5), dtype=np.int64)
        for i, row in enumerate(rows):
            for j, name in enumerate(categorical):
                x[i, j] = mappings[j].get(row[name], 0) + offsets[j]
            bucket = int(np.searchsorted(edges, row["duration_ms"], side="right"))
            x[i, 4] = bucket + offsets[4]
        return x

    return transform(train_rows), transform(val_rows), field_dims


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        tr = np.load(train_npz)
        va = np.load(val_npz)
        field_dims = tr["field_dims"].astype(np.int64)
        video_offset = int(field_dims[0])
        return {
            "X_train": tr["X"].astype(np.int64),
            "y_train": tr["y"].astype(np.float32),
            "X_val": va["X"].astype(np.int64),
            "y_val": va["y"].astype(np.int64),
            "users": va["user"],
            "videos": va["X"][:, 1].astype(np.int64) - video_offset,
            "field_dims": field_dims,
            "npz": True,
        }

    train_rows = read_csv_split(os.path.join(data_dir, "train.csv"), True)
    val_rows = read_csv_split(os.path.join(data_dir, "val.csv"), False)
    xt, xv, field_dims = encode_csv(train_rows, val_rows)
    return {
        "X_train": xt,
        "y_train": np.asarray([r["long_view"] for r in train_rows], dtype=np.float32),
        "X_val": xv,
        "y_val": np.asarray([r["long_view"] for r in val_rows], dtype=np.int64),
        "users": np.asarray([r["user_id"] for r in val_rows]),
        "videos": np.asarray([r["video_id"] for r in val_rows]),
        "field_dims": field_dims,
        "npz": False,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = load_data(args.data_dir)
    if data["npz"]:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate

    xt = torch.from_numpy(data["X_train"])
    yt = torch.from_numpy(data["y_train"])
    xv = torch.from_numpy(data["X_val"])
    total_dim = int(data["field_dims"].sum())

    model = DCNv2Lite(total_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()
    batch_size = 8192
    n = len(yt)
    best_primary = -1.0
    best_scores = None
    patience = 0

    for _ in range(args.epochs):
        model.train()
        permutation = torch.randperm(n)
        for start in range(0, n, batch_size):
            indices = permutation[start:start + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(xt[indices]), yt[indices])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            scores = np.concatenate([
                model(xv[start:start + 65536]).numpy()
                for start in range(0, len(xv), 65536)
            ])
        metrics = evaluate(data["users"], data["y_val"], scores)
        primary = float(metrics["primary"])
        if primary > best_primary + 1e-6:
            best_primary = primary
            best_scores = scores.copy()
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break

    metrics = evaluate(data["users"], data["y_val"], best_scores)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump({
            "gauc": float(metrics["GAUC"] if "GAUC" in metrics else metrics["gauc"]),
            "ndcg5": float(metrics["nDCG@5"] if "nDCG@5" in metrics else metrics["ndcg5"]),
            "primary": float(metrics["primary"]),
        }, fh)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(best_scores):
            writer.writerow([i, data["users"][i], data["videos"][i], format(float(score), ".6g")])


if __name__ == "__main__":
    main()
