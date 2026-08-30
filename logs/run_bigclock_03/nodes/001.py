"""Rank-16 FM with an aggressive anti-overfitting regularization schedule."""
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


def load_npz(data_dir):
    tr = np.load(os.path.join(data_dir, "train.npz"))
    va = np.load(os.path.join(data_dir, "val.npz"))
    train = {
        "X": tr["X"].astype(np.int64),
        "y": tr["y"].astype(np.float32),
        "field_dims": tr["field_dims"].astype(np.int64),
    }
    val = {
        "X": va["X"].astype(np.int64),
        "y": va["y"].astype(np.float32),
        "user": va["user"],
    }
    video_offset = int(train["field_dims"][0])
    val["video"] = val["X"][:, 1] - video_offset
    return train, val


def read_selected_csv(path, is_train):
    rows = []
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        has_author = "author_id" in (reader.fieldnames or [])
        for row in reader:
            selected = {
                "user_id": row["user_id"],
                "video_id": row["video_id"],
                "author_id": row["author_id"] if has_author else "__missing_author__",
                "tab": row["tab"],
                "duration_ms": float(row["duration_ms"]),
                "long_view": float(row["long_view"]),
            }
            rows.append(selected)
    return rows


def make_mapping(values):
    mapping = {}
    for value in values:
        if value not in mapping:
            mapping[value] = len(mapping) + 1
    return mapping


def load_csv(data_dir):
    train_rows = read_selected_csv(os.path.join(data_dir, "train.csv"), True)
    val_rows = read_selected_csv(os.path.join(data_dir, "val.csv"), False)

    durations = np.asarray([r["duration_ms"] for r in train_rows], dtype=np.float64)
    quantiles = np.quantile(durations, np.linspace(0.1, 0.9, 9))

    field_names = ["user_id", "video_id", "author_id", "tab"]
    mappings = {name: make_mapping(r[name] for r in train_rows) for name in field_names}
    field_dims = [len(mappings[name]) + 1 for name in field_names] + [10]
    offsets = np.cumsum([0] + field_dims[:-1], dtype=np.int64)

    def encode(rows):
        X = np.empty((len(rows), 5), dtype=np.int64)
        for i, row in enumerate(rows):
            for j, name in enumerate(field_names):
                X[i, j] = mappings[name].get(row[name], 0) + offsets[j]
            bucket = int(np.searchsorted(quantiles, row["duration_ms"], side="right"))
            X[i, 4] = bucket + offsets[4]
        return X

    train = {
        "X": encode(train_rows),
        "y": np.asarray([r["long_view"] for r in train_rows], dtype=np.float32),
        "field_dims": np.asarray(field_dims, dtype=np.int64),
    }
    val = {
        "X": encode(val_rows),
        "y": np.asarray([r["long_view"] for r in val_rows], dtype=np.float32),
        "user": np.asarray([r["user_id"] for r in val_rows]),
        "video": np.asarray([r["video_id"] for r in val_rows]),
    }
    return train, val


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

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    fast_path = (
        os.path.exists(os.path.join(args.data_dir, "train.npz"))
        and os.path.exists(os.path.join(args.data_dir, "val.npz"))
    )
    if fast_path:
        train, val = load_npz(args.data_dir)
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from data.official.evaluate import evaluate
    else:
        train, val = load_csv(args.data_dir)
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from harness.evaluate_provisional import evaluate

    epochs = args.epochs
    smoke_epochs = os.environ.get("SMOKE_EPOCHS")
    if smoke_epochs is not None:
        epochs = min(epochs, max(1, int(smoke_epochs)))

    Xt = torch.from_numpy(train["X"])
    yt = torch.from_numpy(train["y"])
    Xv = torch.from_numpy(val["X"])
    total_dim = int(train["field_dims"].sum())

    model = RegularizedFM(total_dim=total_dim, k=16, dropout=0.30)
    optimizer = torch.optim.AdamW(
        [
            {"params": [model.emb.weight], "weight_decay": 0.0},
            {"params": [model.lin.weight], "weight_decay": 3e-4},
            {"params": [model.bias], "weight_decay": 0.0},
        ],
        lr=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    bce = torch.nn.BCEWithLogitsLoss()

    n = len(yt)
    batch_size = 8192
    row_l2_weight = 3e-4
    best_gauc = -1.0
    best_scores = None
    patience = 0
    history = []

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(n)
        loss_sum = 0.0
        examples = 0

        for start in range(0, n, batch_size):
            idx = permutation[start:start + batch_size]
            xb = Xt[idx]
            yb = yt[idx]
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            unique_rows = torch.unique(xb)
            accessed_row_l2 = model.emb(unique_rows).square().sum(dim=1).mean()
            loss = bce(logits, yb) + row_l2_weight * accessed_row_l2
            loss.backward()
            optimizer.step()
            count = len(idx)
            loss_sum += float(loss.detach()) * count
            examples += count

        scheduler.step()
        model.eval()
        with torch.no_grad():
            score_parts = [
                model(Xv[start:start + 65536]).cpu().numpy()
                for start in range(0, len(Xv), 65536)
            ]
            scores = np.concatenate(score_parts).astype(np.float64)

        metrics = evaluate(val["user"], val["y"].astype(int), scores)
        gauc, ndcg5, primary = metric_values(metrics)
        history.append({
            "epoch": epoch + 1,
            "train_loss": round(loss_sum / max(examples, 1), 6),
            "lr": optimizer.param_groups[0]["lr"],
            "val_gauc": round(gauc, 6),
            "val_ndcg5": round(ndcg5, 6),
            "val_primary": round(primary, 6),
        })

        if gauc > best_gauc + 1e-6:
            best_gauc = gauc
            best_scores = scores.copy()
            patience = 0
        else:
            patience += 1
            if patience >= 5:
                break

    final_metrics = evaluate(val["user"], val["y"].astype(int), best_scores)
    final_gauc, final_ndcg5, final_primary = metric_values(final_metrics)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump({
            "gauc": final_gauc,
            "ndcg5": final_ndcg5,
            "primary": final_primary,
            "history": history,
            "config": {
                "model": "FM",
                "rank": 16,
                "embedding_dropout": 0.30,
                "accessed_row_l2": row_l2_weight,
                "adamw_weight_decay": 3e-4,
                "lr": 1e-3,
                "lr_step_size": 3,
                "lr_gamma": 0.5,
                "selection_metric": "GAUC",
            },
        }, fh)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(best_scores):
            writer.writerow([i, val["user"][i], val["video"][i], format(float(score), ".8g")])


if __name__ == "__main__":
    main()
