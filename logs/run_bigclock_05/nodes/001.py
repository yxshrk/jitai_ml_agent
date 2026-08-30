"""Aggressively regularized rank-16 FM with GAUC checkpoint selection."""
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
        e = self.emb(x)
        e = self.dropout(e)
        s = e.sum(1)
        pair = 0.5 * (s * s - (e * e).sum(1)).sum(1)
        return self.bias + self.lin(x).sum((1, 2)) + pair


def make_map(values):
    mapping = {}
    for value in values:
        if value not in mapping:
            mapping[value] = len(mapping)
    return mapping


def load_csv_data(data_dir):
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    with open(train_path, "r", newline="") as fh:
        train_rows = list(csv.DictReader(fh))
    with open(val_path, "r", newline="") as fh:
        val_rows = list(csv.DictReader(fh))

    user_map = make_map(row["user_id"] for row in train_rows)
    video_map = make_map(row["video_id"] for row in train_rows)
    tab_map = make_map(row["tab"] for row in train_rows)

    train_duration = np.asarray(
        [float(row.get("duration_ms", 0) or 0) for row in train_rows],
        dtype=np.float64,
    )
    quantiles = np.quantile(train_duration, np.linspace(0.1, 0.9, 9))
    quantiles = np.maximum.accumulate(quantiles)

    field_dims = np.asarray(
        [len(user_map) + 1, len(video_map) + 1, 1, len(tab_map) + 1, 10],
        dtype=np.int64,
    )
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1])).astype(np.int64)

    def encode(rows):
        x = np.empty((len(rows), 5), dtype=np.int64)
        raw_users = []
        raw_videos = []
        for i, row in enumerate(rows):
            duration = float(row.get("duration_ms", 0) or 0)
            x[i, 0] = user_map.get(row["user_id"], len(user_map))
            x[i, 1] = video_map.get(row["video_id"], len(video_map))
            x[i, 2] = 0
            x[i, 3] = tab_map.get(row["tab"], len(tab_map))
            x[i, 4] = int(np.searchsorted(quantiles, duration, side="right"))
            raw_users.append(row["user_id"])
            raw_videos.append(row["video_id"])
        x += offsets[None, :]
        return x, raw_users, raw_videos

    xt, _, _ = encode(train_rows)
    xv, raw_users, raw_videos = encode(val_rows)
    yt = np.asarray([float(row["long_view"]) for row in train_rows], dtype=np.float32)
    yv = np.asarray([int(float(row["long_view"])) for row in val_rows], dtype=np.int64)
    eval_users = np.asarray(
        [user_map.get(row["user_id"], len(user_map)) for row in val_rows],
        dtype=np.int64,
    )
    return xt, yt, xv, yv, eval_users, field_dims, raw_users, raw_videos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

    smoke = os.environ.get("SMOKE_EPOCHS")
    epochs = args.epochs
    if smoke is not None:
        epochs = min(epochs, max(1, int(smoke)))

    train_npz = os.path.join(args.data_dir, "train.npz")
    val_npz = os.path.join(args.data_dir, "val.npz")
    fast_path = os.path.exists(train_npz) and os.path.exists(val_npz)

    if fast_path:
        from data.official.evaluate import evaluate

        tr = np.load(train_npz)
        va = np.load(val_npz)
        xt_np = tr["X"].astype(np.int64)
        yt_np = tr["y"].astype(np.float32)
        xv_np = va["X"].astype(np.int64)
        yv = va["y"].astype(np.int64)
        eval_users = va["user"]
        field_dims = tr["field_dims"].astype(np.int64)
        raw_users = va["user"].tolist()
        if "video" in va.files:
            raw_videos = va["video"].tolist()
        else:
            raw_videos = (xv_np[:, 1] - int(field_dims[0])).tolist()
    else:
        from harness.evaluate_provisional import evaluate

        (xt_np, yt_np, xv_np, yv, eval_users, field_dims,
         raw_users, raw_videos) = load_csv_data(args.data_dir)

    xt = torch.from_numpy(xt_np)
    yt = torch.from_numpy(yt_np)
    xv = torch.from_numpy(xv_np)
    total_dim = int(field_dims.sum())

    model = FM(total_dim=total_dim, k=16, dropout=0.30)
    optimizer = torch.optim.AdamW(
        [
            {"params": [model.emb.weight], "weight_decay": 0.0},
            {"params": [model.lin.weight, model.bias], "weight_decay": 3e-4},
        ],
        lr=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    bce = torch.nn.BCEWithLogitsLoss()

    n = len(yt)
    batch_size = 8192
    row_l2_weight = 3e-4
    best_gauc = -1.0
    best_primary = -1.0
    best_scores = None
    patience = 0
    history = []

    for epoch in range(epochs):
        model.train()
        generator = torch.Generator()
        generator.manual_seed(args.seed + epoch)
        permutation = torch.randperm(n, generator=generator)
        loss_sum = 0.0
        seen = 0

        for start in range(0, n, batch_size):
            idx = permutation[start:start + batch_size]
            xb = xt[idx]
            yb = yt[idx]
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            prediction_loss = bce(logits, yb)
            accessed = torch.unique(xb.reshape(-1))
            row_l2 = model.emb.weight[accessed].pow(2).sum(1).mean()
            loss = prediction_loss + row_l2_weight * row_l2
            loss.backward()
            optimizer.step()
            count = len(idx)
            loss_sum += float(loss.detach()) * count
            seen += count

        model.eval()
        with torch.no_grad():
            scores = np.concatenate([
                model(xv[start:start + 65536]).cpu().numpy()
                for start in range(0, len(xv), 65536)
            ])
        metrics = evaluate(eval_users, yv, scores)
        gauc = float(metrics.get("GAUC", metrics.get("gauc")))
        primary = float(metrics["primary"])
        lr_used = float(optimizer.param_groups[0]["lr"])
        history.append({
            "epoch": epoch + 1,
            "train_loss": round(loss_sum / max(1, seen), 6),
            "val_gauc": round(gauc, 6),
            "val_primary": round(primary, 6),
            "lr": lr_used,
            "dropout": 0.30,
            "row_l2": row_l2_weight,
            "weight_decay": 3e-4,
        })

        if gauc > best_gauc + 1e-7 or (
            abs(gauc - best_gauc) <= 1e-7 and primary > best_primary
        ):
            best_gauc = gauc
            best_primary = primary
            best_scores = scores.copy()
            patience = 0
        else:
            patience += 1

        scheduler.step()
        if patience >= 4:
            break

    if best_scores is None:
        model.eval()
        with torch.no_grad():
            best_scores = np.concatenate([
                model(xv[start:start + 65536]).cpu().numpy()
                for start in range(0, len(xv), 65536)
            ])

    final_metrics = evaluate(eval_users, yv, best_scores)
    output_metrics = {
        "gauc": float(final_metrics.get("GAUC", final_metrics.get("gauc"))),
        "ndcg5": float(final_metrics.get("nDCG@5", final_metrics.get("ndcg5"))),
        "primary": float(final_metrics["primary"]),
        "history": history,
        "config": {
            "embedding_dim": 16,
            "dropout": 0.30,
            "row_l2": row_l2_weight,
            "adamw_dense_weight_decay": 3e-4,
            "initial_lr": 1e-3,
            "lr_step_gamma": 0.5,
            "selection_metric": "GAUC",
        },
    }

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(output_metrics, fh)
    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(best_scores):
            writer.writerow([i, raw_users[i], raw_videos[i], format(float(score), ".8g")])


if __name__ == "__main__":
    main()
