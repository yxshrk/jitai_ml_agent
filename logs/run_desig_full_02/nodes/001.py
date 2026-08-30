"""Regularized FM with embedding dropout, accessed-row L2, AdamW, and LR decay."""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch


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
        summed = e.sum(dim=1)
        pair = 0.5 * (summed.square() - e.square().sum(dim=1)).sum(dim=1)
        return self.bias + self.lin(x).sum(dim=(1, 2)) + pair

    def accessed_row_l2(self, x):
        rows = torch.unique(x)
        emb_penalty = self.emb.weight[rows].square().sum(dim=1).mean()
        lin_penalty = self.lin.weight[rows].square().sum(dim=1).mean()
        return emb_penalty + lin_penalty


def load_npz(data_dir):
    tr = np.load(os.path.join(data_dir, "train.npz"))
    va = np.load(os.path.join(data_dir, "val.npz"))
    train_x = tr["X"].astype(np.int64)
    train_y = tr["y"].astype(np.float32)
    val_x = va["X"].astype(np.int64)
    val_y = va["y"].astype(np.int64)
    val_users = va["user"]
    field_dims = tr["field_dims"].astype(np.int64)
    video_offset = int(field_dims[0])
    val_videos = val_x[:, 1] - video_offset
    return train_x, train_y, val_x, val_y, val_users, val_videos, field_dims, True


def read_csv_rows(path):
    with open(path, "r", newline="") as fh:
        return list(csv.DictReader(fh))


def load_csv(data_dir):
    train_rows = read_csv_rows(os.path.join(data_dir, "train.csv"))
    val_rows = read_csv_rows(os.path.join(data_dir, "val.csv"))

    def build_mapping(name):
        values = sorted({row[name] for row in train_rows})
        return {value: i + 1 for i, value in enumerate(values)}

    user_map = build_mapping("user_id")
    video_map = build_mapping("video_id")
    tab_map = build_mapping("tab")
    train_durations = np.asarray(
        [float(row["duration_ms"]) for row in train_rows], dtype=np.float64
    )
    quantiles = np.quantile(train_durations, np.linspace(0.1, 0.9, 9))
    field_dims = np.asarray(
        [len(user_map) + 1, len(video_map) + 1, 1, len(tab_map) + 1, 10],
        dtype=np.int64,
    )
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1])).astype(np.int64)

    def encode(rows):
        x = np.empty((len(rows), 5), dtype=np.int64)
        for i, row in enumerate(rows):
            duration = float(row["duration_ms"])
            x[i, 0] = user_map.get(row["user_id"], 0) + offsets[0]
            x[i, 1] = video_map.get(row["video_id"], 0) + offsets[1]
            x[i, 2] = offsets[2]
            x[i, 3] = tab_map.get(row["tab"], 0) + offsets[3]
            x[i, 4] = int(np.searchsorted(quantiles, duration, side="right")) + offsets[4]
        return x

    train_x = encode(train_rows)
    val_x = encode(val_rows)
    train_y = np.asarray([float(row["long_view"]) for row in train_rows], dtype=np.float32)
    val_y = np.asarray([int(float(row["long_view"])) for row in val_rows], dtype=np.int64)
    val_users = np.asarray([row["user_id"] for row in val_rows])
    val_videos = np.asarray([row["video_id"] for row in val_rows])
    return train_x, train_y, val_x, val_y, val_users, val_videos, field_dims, False


def metric_values(metrics):
    return {
        "gauc": float(metrics.get("GAUC", metrics.get("gauc"))),
        "ndcg5": float(metrics.get("nDCG@5", metrics.get("ndcg5"))),
        "primary": float(metrics["primary"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)

    fast_path = os.path.exists(os.path.join(args.data_dir, "train.npz")) and os.path.exists(
        os.path.join(args.data_dir, "val.npz")
    )
    if fast_path:
        train_x, train_y, val_x, val_y, val_users, val_videos, field_dims, official = load_npz(
            args.data_dir
        )
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from data.official.evaluate import evaluate
    else:
        train_x, train_y, val_x, val_y, val_users, val_videos, field_dims, official = load_csv(
            args.data_dir
        )
        from harness.evaluate_provisional import evaluate

    epochs = args.epochs
    smoke_epochs = os.environ.get("SMOKE_EPOCHS")
    if smoke_epochs is not None:
        epochs = min(epochs, max(1, int(smoke_epochs)))

    xt = torch.from_numpy(train_x)
    yt = torch.from_numpy(train_y)
    xv = torch.from_numpy(val_x)
    model = RegularizedFM(int(field_dims.sum()), k=16, dropout=0.30)
    optimizer = torch.optim.AdamW(
        [
            {"params": [model.emb.weight, model.lin.weight], "weight_decay": 0.0},
            {"params": [model.bias], "weight_decay": 1e-3},
        ],
        lr=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.70)
    bce = torch.nn.BCEWithLogitsLoss()
    batch_size = 8192
    row_l2_weight = 1e-4
    best_gauc = -1.0
    best_scores = None
    patience = 0

    for _ in range(epochs):
        model.train()
        permutation = torch.randperm(len(yt))
        for start in range(0, len(yt), batch_size):
            idx = permutation[start:start + batch_size]
            xb = xt[idx]
            optimizer.zero_grad(set_to_none=True)
            loss = bce(model(xb), yt[idx]) + row_l2_weight * model.accessed_row_l2(xb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            scores = np.concatenate(
                [
                    model(xv[start:start + 65536]).cpu().numpy()
                    for start in range(0, len(xv), 65536)
                ]
            )
        metrics = evaluate(val_users, val_y, scores)
        gauc = float(metrics.get("GAUC", metrics.get("gauc")))
        if gauc > best_gauc + 1e-6:
            best_gauc = gauc
            best_scores = scores.copy()
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break

    final_metrics = metric_values(evaluate(val_users, val_y, best_scores))
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(final_metrics, fh)
    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(best_scores):
            writer.writerow([i, val_users[i], val_videos[i], format(float(score), ".8g")])


if __name__ == "__main__":
    main()
