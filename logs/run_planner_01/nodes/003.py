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


def read_csv_rows(path):
    with open(path, "r", newline="") as fh:
        return list(csv.DictReader(fh))


def raw_feature(row, name):
    if name == "author_id" and name not in row:
        return "__missing_author__"
    return row.get(name, "")


def prepare_csv(data_dir):
    train_rows = read_csv_rows(os.path.join(data_dir, "train.csv"))
    val_rows = read_csv_rows(os.path.join(data_dir, "val.csv"))
    categorical = ["user_id", "video_id", "author_id", "tab"]
    maps = []
    dims = []
    for name in categorical:
        values = sorted({raw_feature(row, name) for row in train_rows})
        mapping = {value: i + 1 for i, value in enumerate(values)}
        maps.append(mapping)
        dims.append(len(mapping) + 1)

    train_duration = np.asarray(
        [float(row.get("duration_ms", 0) or 0) for row in train_rows],
        dtype=np.float64,
    )
    if len(train_duration):
        cuts = np.quantile(train_duration, np.linspace(0.1, 0.9, 9))
    else:
        cuts = np.zeros(9, dtype=np.float64)
    dims.append(10)
    offsets = np.cumsum([0] + dims[:-1], dtype=np.int64)

    def encode(rows):
        x = np.empty((len(rows), 5), dtype=np.int64)
        for j, (name, mapping) in enumerate(zip(categorical, maps)):
            x[:, j] = np.asarray(
                [mapping.get(raw_feature(row, name), 0) for row in rows],
                dtype=np.int64,
            ) + offsets[j]
        durations = np.asarray(
            [float(row.get("duration_ms", 0) or 0) for row in rows],
            dtype=np.float64,
        )
        x[:, 4] = np.searchsorted(cuts, durations, side="right") + offsets[4]
        return x

    xt = encode(train_rows)
    xv = encode(val_rows)
    yt = np.asarray([float(row["long_view"]) for row in train_rows], dtype=np.float32)
    yv = np.asarray([float(row["long_view"]) for row in val_rows], dtype=np.float32)
    val_users_raw = [row["user_id"] for row in val_rows]
    val_videos_raw = [row["video_id"] for row in val_rows]
    try:
        eval_users = np.asarray(val_users_raw, dtype=np.int64)
    except ValueError:
        user_eval_map = {v: i for i, v in enumerate(sorted(set(val_users_raw)))}
        eval_users = np.asarray([user_eval_map[v] for v in val_users_raw], dtype=np.int64)
    return xt, yt, xv, yv, eval_users, val_users_raw, val_videos_raw, int(sum(dims)), False


def prepare_npz(data_dir):
    tr = np.load(os.path.join(data_dir, "train.npz"))
    va = np.load(os.path.join(data_dir, "val.npz"))
    xt = tr["X"].astype(np.int64)
    yt = tr["y"].astype(np.float32)
    xv = va["X"].astype(np.int64)
    yv = va["y"].astype(np.float32)
    users = va["user"]
    user_output = [str(v) for v in users]
    video_output = ["0"] * len(xv)
    return xt, yt, xv, yv, users, user_output, video_output, int(tr["field_dims"].sum()), True


def metric_value(metrics, upper, lower):
    return float(metrics[upper] if upper in metrics else metrics[lower])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True)

    epochs = args.epochs
    smoke = os.environ.get("SMOKE_EPOCHS")
    if smoke is not None:
        epochs = min(epochs, max(1, int(smoke)))

    fast_path = os.path.exists(os.path.join(args.data_dir, "train.npz")) and os.path.exists(
        os.path.join(args.data_dir, "val.npz")
    )
    if fast_path:
        xt_np, yt_np, xv_np, yv, eval_users, output_users, output_videos, total_dim, official = prepare_npz(args.data_dir)
        from data.official.evaluate import evaluate
    else:
        xt_np, yt_np, xv_np, yv, eval_users, output_users, output_videos, total_dim, official = prepare_csv(args.data_dir)
        from harness.evaluate_provisional import evaluate

    xt = torch.from_numpy(xt_np)
    yt = torch.from_numpy(yt_np)
    xv = torch.from_numpy(xv_np)

    model = RegularizedFM(total_dim=total_dim, k=16, dropout=0.30)
    optimizer = torch.optim.AdamW(
        [
            {"params": [model.emb.weight], "weight_decay": 0.0},
            {"params": [model.lin.weight], "weight_decay": 1e-3},
            {"params": [model.bias], "weight_decay": 0.0},
        ],
        lr=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    bce = torch.nn.BCEWithLogitsLoss()
    batch_size = 8192
    row_l2_weight = 1e-5
    n = len(yt)
    best_gauc = -np.inf
    best_scores = None
    patience = 0

    for _ in range(epochs):
        model.train()
        permutation = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = permutation[start:start + batch_size]
            xb = xt[idx]
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            accessed = torch.unique(xb)
            row_l2 = model.emb.weight[accessed].square().sum()
            loss = bce(logits, yt[idx]) + row_l2_weight * row_l2
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            scores = np.concatenate(
                [model(xv[start:start + 65536]).cpu().numpy() for start in range(0, len(xv), 65536)]
            )
        metrics = evaluate(eval_users, yv.astype(int), scores)
        gauc = metric_value(metrics, "GAUC", "gauc")
        if gauc > best_gauc + 1e-6:
            best_gauc = gauc
            best_scores = scores.copy()
            patience = 0
        else:
            patience += 1
            if patience >= 4:
                break

    if best_scores is None:
        model.eval()
        with torch.no_grad():
            best_scores = np.concatenate(
                [model(xv[start:start + 65536]).cpu().numpy() for start in range(0, len(xv), 65536)]
            )

    final_metrics = evaluate(eval_users, yv.astype(int), best_scores)
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
            writer.writerow([i, output_users[i], output_videos[i], format(float(score), ".9g")])


if __name__ == "__main__":
    main()
