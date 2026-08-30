"""Regularized FM with dropout, accessed-row L2, weight decay, and LR decay."""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FM(torch.nn.Module):
    def __init__(self, total_dim, k=16, dropout=0.08):
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
        return self.emb(rows).pow(2).sum(1).mean()


def read_csv_split(path, need_label):
    users = []
    videos = []
    tabs = []
    durations = []
    labels = []
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            users.append(row["user_id"])
            videos.append(row["video_id"])
            tabs.append(row["tab"])
            duration = int(float(row["duration_ms"])) if row["duration_ms"] else 0
            durations.append(str(max(0, duration) // 10000))
            if need_label:
                labels.append(float(row["long_view"]))
    return {
        "user": users,
        "video": videos,
        "tab": tabs,
        "dur": durations,
        "y": np.asarray(labels, dtype=np.float32),
    }


def make_mapping(values):
    mapping = {}
    for value in values:
        if value not in mapping:
            mapping[value] = len(mapping) + 1
    return mapping


def encode_csv(train, val):
    user_map = make_mapping(train["user"])
    video_map = make_mapping(train["video"])
    tab_map = make_mapping(train["tab"])
    dur_map = make_mapping(train["dur"])
    field_dims = np.asarray([
        len(user_map) + 1,
        len(video_map) + 1,
        1,
        len(tab_map) + 1,
        len(dur_map) + 1,
    ], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1]))

    def encode(split):
        n = len(split["user"])
        x = np.empty((n, 5), dtype=np.int64)
        x[:, 0] = [user_map.get(v, 0) for v in split["user"]]
        x[:, 1] = [video_map.get(v, 0) for v in split["video"]]
        x[:, 2] = 0
        x[:, 3] = [tab_map.get(v, 0) for v in split["tab"]]
        x[:, 4] = [dur_map.get(v, 0) for v in split["dur"]]
        x += offsets
        return x

    return encode(train), encode(val), field_dims


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        tr_file = np.load(train_npz)
        va_file = np.load(val_npz)
        tr = {key: tr_file[key] for key in tr_file.files}
        va = {key: va_file[key] for key in va_file.files}
        field_dims = tr["field_dims"].astype(np.int64)
        video_offset = int(field_dims[0])
        output_users = va["user"]
        output_videos = va["X"][:, 1].astype(np.int64) - video_offset
        return (
            tr["X"].astype(np.int64),
            tr["y"].astype(np.float32),
            va["X"].astype(np.int64),
            va["y"].astype(np.float32),
            va["user"],
            field_dims,
            output_users,
            output_videos,
            True,
        )

    tr_raw = read_csv_split(os.path.join(data_dir, "train.csv"), True)
    va_raw = read_csv_split(os.path.join(data_dir, "val.csv"), True)
    xt, xv, field_dims = encode_csv(tr_raw, va_raw)
    return (
        xt,
        tr_raw["y"],
        xv,
        va_raw["y"],
        np.asarray(va_raw["user"]),
        field_dims,
        va_raw["user"],
        va_raw["video"],
        False,
    )


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

    epochs = args.epochs
    if "SMOKE_EPOCHS" in os.environ:
        epochs = min(epochs, int(os.environ["SMOKE_EPOCHS"]))
    epochs = max(1, epochs)

    xt_np, yt_np, xv_np, yv_np, eval_users, field_dims, out_users, out_videos, fast = load_data(args.data_dir)
    xt = torch.from_numpy(xt_np)
    yt = torch.from_numpy(yt_np)
    xv = torch.from_numpy(xv_np)

    model = FM(int(field_dims.sum()))
    optimizer = torch.optim.AdamW([
        {"params": [model.emb.weight], "weight_decay": 0.0},
        {"params": [model.lin.weight], "weight_decay": 1e-5},
        {"params": [model.bias], "weight_decay": 0.0},
    ], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.92)
    bce = torch.nn.BCEWithLogitsLoss()

    if fast:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate

    n = len(yt)
    batch_size = 8192
    best = -1.0
    best_scores = None
    patience = 0

    for _ in range(epochs):
        model.train()
        permutation = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = permutation[start:start + batch_size]
            xb = xt[idx]
            optimizer.zero_grad()
            loss = bce(model(xb), yt[idx]) + 1e-3 * model.accessed_row_l2(xb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            scores = np.concatenate([
                model(xv[start:start + 65536]).cpu().numpy()
                for start in range(0, len(xv), 65536)
            ])
        metrics = evaluate(eval_users, yv_np.astype(int), scores)
        primary = float(metrics["primary"])
        if primary > best + 1e-6:
            best = primary
            best_scores = scores.copy()
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break

    final_metrics = evaluate(eval_users, yv_np.astype(int), best_scores)
    gauc = final_metrics["GAUC"] if "GAUC" in final_metrics else final_metrics["gauc"]
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
            writer.writerow([i, out_users[i], out_videos[i], format(float(score), ".8g")])


if __name__ == "__main__":
    main()
