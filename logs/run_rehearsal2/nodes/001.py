"""Aggressively regularized FM with dropout, row L2, AdamW, and LR decay."""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FM(torch.nn.Module):
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

    def accessed_row_l2(self, x):
        rows = torch.unique(x)
        emb_penalty = self.emb(rows).pow(2).sum(1).mean()
        lin_penalty = self.lin(rows).pow(2).sum(1).mean()
        return emb_penalty + lin_penalty


def _as_number(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def load_csv_data(data_dir):
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")

    train_rows = []
    with open(train_path, "r", newline="") as fh:
        for row in csv.DictReader(fh):
            train_rows.append({
                "user": row["user_id"],
                "video": row["video_id"],
                "author": row.get("author_id", "__missing_author__"),
                "tab": row["tab"],
                "duration": float(row["duration_ms"]),
                "y": float(row["long_view"]),
            })

    val_rows = []
    with open(val_path, "r", newline="") as fh:
        for row in csv.DictReader(fh):
            val_rows.append({
                "user": row["user_id"],
                "video": row["video_id"],
                "author": row.get("author_id", "__missing_author__"),
                "tab": row["tab"],
                "duration": float(row["duration_ms"]),
                "y": float(row["long_view"]),
            })

    field_names = ("user", "video", "author", "tab")
    mappings = {}
    dims = []
    for field in field_names:
        values = sorted({r[field] for r in train_rows})
        mappings[field] = {value: i for i, value in enumerate(values)}
        dims.append(len(values) + 1)

    durations = np.asarray([r["duration"] for r in train_rows], dtype=np.float64)
    edges = np.quantile(durations, np.linspace(0.1, 0.9, 9))
    dims.append(10)
    offsets = np.cumsum([0] + dims[:-1], dtype=np.int64)

    def encode(rows):
        x = np.empty((len(rows), 5), dtype=np.int64)
        for i, row in enumerate(rows):
            for j, field in enumerate(field_names):
                mapping = mappings[field]
                x[i, j] = offsets[j] + mapping.get(row[field], len(mapping))
            x[i, 4] = offsets[4] + int(np.searchsorted(edges, row["duration"], side="right"))
        return x

    return {
        "Xt": encode(train_rows),
        "yt": np.asarray([r["y"] for r in train_rows], dtype=np.float32),
        "Xv": encode(val_rows),
        "yv": np.asarray([r["y"] for r in val_rows], dtype=np.float32),
        "val_user": np.asarray([_as_number(r["user"]) for r in val_rows]),
        "val_video": np.asarray([_as_number(r["video"]) for r in val_rows]),
        "field_dims": np.asarray(dims, dtype=np.int64),
        "official": False,
    }


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        tr = np.load(train_npz)
        va = np.load(val_npz)
        return {
            "Xt": tr["X"].astype(np.int64),
            "yt": tr["y"].astype(np.float32),
            "Xv": va["X"].astype(np.int64),
            "yv": va["y"].astype(np.float32),
            "val_user": va["user"],
            "val_video": np.zeros(len(va["y"]), dtype=np.int64),
            "field_dims": tr["field_dims"].astype(np.int64),
            "official": True,
        }
    return load_csv_data(data_dir)


def run_evaluator(official, user_ids, labels, scores):
    if official:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    return evaluate(user_ids, labels, scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    data = load_data(args.data_dir)
    Xt = torch.from_numpy(data["Xt"])
    yt = torch.from_numpy(data["yt"])
    Xv = torch.from_numpy(data["Xv"])
    total_dim = int(data["field_dims"].sum())

    model = FM(total_dim, k=16, dropout=0.30)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    bce = torch.nn.BCEWithLogitsLoss()

    n = len(yt)
    batch_size = 8192
    row_l2_weight = 1e-3
    best_primary = -1.0
    best_scores = None
    patience = 0
    history = []

    for epoch in range(args.epochs):
        model.train()
        permutation = torch.randperm(n)
        epoch_loss = 0.0
        seen = 0
        for start in range(0, n, batch_size):
            idx = permutation[start:start + batch_size]
            xb = Xt[idx]
            yb = yt[idx]
            optimizer.zero_grad()
            logits = model(xb)
            loss = bce(logits, yb) + row_l2_weight * model.accessed_row_l2(xb)
            loss.backward()
            optimizer.step()
            count = len(idx)
            epoch_loss += float(loss.item()) * count
            seen += count

        model.eval()
        with torch.no_grad():
            scores = np.concatenate([
                model(Xv[start:start + 65536]).cpu().numpy()
                for start in range(0, len(Xv), 65536)
            ])

        metrics = run_evaluator(
            data["official"], data["val_user"], data["yv"].astype(int), scores
        )
        primary = float(metrics["primary"])
        history.append({
            "epoch": epoch + 1,
            "train_loss": round(epoch_loss / max(seen, 1), 5),
            "val_gauc": round(float(metrics.get("GAUC", metrics.get("gauc", 0.0))), 6),
            "val_primary": round(primary, 6),
            "lr": optimizer.param_groups[0]["lr"],
        })

        if primary > best_primary + 1e-6:
            best_primary = primary
            best_scores = scores.copy()
            patience = 0
        else:
            patience += 1

        scheduler.step()
        if patience >= 4:
            break

    final_metrics = run_evaluator(
        data["official"], data["val_user"], data["yv"].astype(int), best_scores
    )
    gauc = final_metrics.get("GAUC", final_metrics.get("gauc"))
    ndcg5 = final_metrics.get("nDCG@5", final_metrics.get("ndcg5"))

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump({
            "gauc": float(gauc),
            "ndcg5": float(ndcg5),
            "primary": float(final_metrics["primary"]),
            "history": history,
        }, fh)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(best_scores):
            writer.writerow([i, data["val_user"][i], data["val_video"][i], f"{score:.6g}"])


if __name__ == "__main__":
    main()
