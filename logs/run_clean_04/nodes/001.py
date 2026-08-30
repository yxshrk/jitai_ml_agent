"""Regularized DeepFM variant using the official NPZ fast path when available."""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RegularizedDeepFM(torch.nn.Module):
    def __init__(self, total_dim, fields=5, k=16, dropout=0.20):
        super().__init__()
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(fields * k, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(32, 1),
        )
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)
        for layer in self.mlp:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
        torch.nn.init.zeros_(self.mlp[-1].weight)

    def forward(self, x, return_embeddings=False):
        e = self.emb(x)
        summed = e.sum(1)
        pair = 0.5 * (summed.square() - e.square().sum(1)).sum(1)
        deep = self.mlp(e.flatten(1)).squeeze(1)
        logits = self.bias + self.lin(x).sum((1, 2)) + pair + deep
        if return_embeddings:
            return logits, e
        return logits


def load_csv_data(data_dir):
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    with open(train_path, "r", newline="") as fh:
        train_rows = list(csv.DictReader(fh))
    with open(val_path, "r", newline="") as fh:
        val_rows = list(csv.DictReader(fh))

    def duration_bucket(row):
        return str(min(60, max(0, int(float(row.get("duration_ms", 0) or 0)) // 10000)))

    extractors = [
        lambda r: str(r["user_id"]),
        lambda r: str(r["video_id"]),
        lambda r: "__unknown_author__",
        lambda r: str(r.get("tab", "")),
        duration_bucket,
    ]
    mappings = []
    for extractor in extractors:
        values = sorted({extractor(row) for row in train_rows})
        mappings.append({value: i + 1 for i, value in enumerate(values)})
    field_dims = np.asarray([len(mapping) + 1 for mapping in mappings], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1]))

    def encode(rows):
        x = np.empty((len(rows), 5), dtype=np.int64)
        for i, row in enumerate(rows):
            for j, (extractor, mapping) in enumerate(zip(extractors, mappings)):
                x[i, j] = mapping.get(extractor(row), 0) + offsets[j]
        return x

    xt = encode(train_rows)
    xv = encode(val_rows)
    yt = np.asarray([float(row["long_view"]) for row in train_rows], dtype=np.float32)
    yv = np.asarray([float(row["long_view"]) for row in val_rows], dtype=np.float32)
    val_users = np.asarray([row["user_id"] for row in val_rows])
    val_videos = np.asarray([row["video_id"] for row in val_rows])
    return xt, yt, xv, yv, val_users, val_videos, field_dims, False


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        tr = np.load(train_npz)
        va = np.load(val_npz)
        field_dims = tr["field_dims"].astype(np.int64)
        xv = va["X"].astype(np.int64)
        video_offset = int(field_dims[0])
        val_videos = xv[:, 1] - video_offset
        return (tr["X"].astype(np.int64), tr["y"].astype(np.float32), xv,
                va["y"].astype(np.float32), va["user"], val_videos,
                field_dims, True)
    return load_csv_data(data_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(1)

    epochs = args.epochs
    smoke_epochs = os.environ.get("SMOKE_EPOCHS")
    if smoke_epochs is not None:
        epochs = min(epochs, int(smoke_epochs))

    xt_np, yt_np, xv_np, yv, val_users, val_videos, field_dims, fast_path = load_data(args.data_dir)
    xt = torch.from_numpy(xt_np)
    yt = torch.from_numpy(yt_np)
    xv = torch.from_numpy(xv_np)

    model = RegularizedDeepFM(int(field_dims.sum()))
    dense_params = list(model.mlp.parameters())
    dense_ids = {id(parameter) for parameter in dense_params}
    sparse_fm_params = [parameter for parameter in model.parameters() if id(parameter) not in dense_ids]
    optimizer = torch.optim.Adam([
        {"params": sparse_fm_params, "weight_decay": 0.0},
        {"params": dense_params, "weight_decay": 1e-5},
    ], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.90)
    criterion = torch.nn.BCEWithLogitsLoss()

    if fast_path:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate

    n = len(yt)
    batch_size = 8192
    best_primary = -1.0
    best_scores = None
    patience = 0

    for _ in range(epochs):
        model.train()
        permutation = torch.randperm(n)
        for start in range(0, n, batch_size):
            indices = permutation[start:start + batch_size]
            optimizer.zero_grad()
            logits, accessed_embeddings = model(xt[indices], return_embeddings=True)
            row_l2 = 1e-6 * accessed_embeddings.square().sum()
            loss = criterion(logits, yt[indices]) + row_l2
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            scores = np.concatenate([
                model(xv[start:start + 65536]).numpy()
                for start in range(0, len(xv), 65536)
            ])
        metrics = evaluate(val_users, yv.astype(int), scores)
        primary = float(metrics["primary"])
        if primary > best_primary + 1e-6:
            best_primary = primary
            best_scores = scores.copy()
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break

    if best_scores is None:
        model.eval()
        with torch.no_grad():
            best_scores = np.concatenate([
                model(xv[start:start + 65536]).numpy()
                for start in range(0, len(xv), 65536)
            ])

    final_metrics = evaluate(val_users, yv.astype(int), best_scores)
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
        for row_id, (user_id, video_id, score) in enumerate(zip(val_users, val_videos, best_scores)):
            writer.writerow([row_id, user_id, video_id, format(float(score), ".6g")])


if __name__ == "__main__":
    main()
