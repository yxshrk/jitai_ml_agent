"""FM baseline with a hybrid pointwise BCE and within-user BPR objective."""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FM(torch.nn.Module):
    def __init__(self, total_dim, k=16):
        super().__init__()
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)

    def forward(self, x):
        e = self.emb(x)
        s = e.sum(1)
        pair = 0.5 * (s * s - (e * e).sum(1)).sum(1)
        return self.bias + self.lin(x).sum((1, 2)) + pair


def read_csv_split(path, need_label):
    rows = []
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            item = {
                "user_id": row["user_id"],
                "video_id": row["video_id"],
                "tab": row["tab"],
                "duration_ms": int(float(row["duration_ms"])),
            }
            if need_label:
                item["long_view"] = float(row["long_view"])
            rows.append(item)
    return rows


def encode_csv(train_rows, val_rows):
    def duration_bucket(ms):
        return str(min(60, max(0, ms // 10000)))

    train_values = [
        [r["user_id"] for r in train_rows],
        [r["video_id"] for r in train_rows],
        ["0" for _ in train_rows],
        [r["tab"] for r in train_rows],
        [duration_bucket(r["duration_ms"]) for r in train_rows],
    ]
    maps = []
    field_dims = []
    for values in train_values:
        unique = sorted(set(values))
        mapping = {value: i + 1 for i, value in enumerate(unique)}
        maps.append(mapping)
        field_dims.append(len(mapping) + 1)
    field_dims = np.asarray(field_dims, dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1]))

    def transform(rows):
        x = np.empty((len(rows), 5), dtype=np.int32)
        for i, row in enumerate(rows):
            values = [row["user_id"], row["video_id"], "0", row["tab"],
                      duration_bucket(row["duration_ms"])]
            for j, value in enumerate(values):
                x[i, j] = maps[j].get(value, 0) + offsets[j]
        return x

    return transform(train_rows), transform(val_rows), field_dims


def make_within_user_pairs(users, labels):
    users = np.asarray(users)
    labels = np.asarray(labels) > 0.5
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    boundaries = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(order)]))
    positive_parts = []
    negative_parts = []
    for start, end in zip(starts, ends):
        group = order[start:end]
        positive = group[labels[group]]
        negative = group[~labels[group]]
        if len(positive) == 0 or len(negative) == 0:
            continue
        count = max(len(positive), len(negative))
        positive_parts.append(np.resize(positive, count))
        negative_parts.append(np.resize(negative, count))
    if not positive_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return (np.concatenate(positive_parts).astype(np.int64),
            np.concatenate(negative_parts).astype(np.int64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=12)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.use_deterministic_algorithms(True)

    epochs = a.epochs
    if "SMOKE_EPOCHS" in os.environ:
        epochs = min(epochs, int(os.environ["SMOKE_EPOCHS"]))

    train_npz = os.path.join(a.data_dir, "train.npz")
    val_npz = os.path.join(a.data_dir, "val.npz")
    use_npz = os.path.exists(train_npz) and os.path.exists(val_npz)

    if use_npz:
        from data.official.evaluate import evaluate
        tr = np.load(train_npz)
        va = np.load(val_npz)
        xt_np = tr["X"].astype(np.int64)
        yt_np = tr["y"].astype(np.float32)
        xv_np = va["X"].astype(np.int64)
        yv_np = va["y"].astype(np.int64)
        train_users = tr["user"]
        val_users = va["user"]
        field_dims = tr["field_dims"].astype(np.int64)
        output_users = val_users
        output_videos = np.zeros(len(xv_np), dtype=np.int64)
    else:
        from harness.evaluate_provisional import evaluate
        train_rows = read_csv_split(os.path.join(a.data_dir, "train.csv"), True)
        val_rows = read_csv_split(os.path.join(a.data_dir, "val.csv"), True)
        xt_np, xv_np, field_dims = encode_csv(train_rows, val_rows)
        yt_np = np.asarray([r["long_view"] for r in train_rows], dtype=np.float32)
        yv_np = np.asarray([r["long_view"] for r in val_rows], dtype=np.int64)
        train_users = np.asarray([r["user_id"] for r in train_rows])
        val_users = np.asarray([r["user_id"] for r in val_rows])
        output_users = val_users
        output_videos = np.asarray([r["video_id"] for r in val_rows])

    total_dim = int(field_dims.sum())
    Xt = torch.from_numpy(xt_np)
    yt = torch.from_numpy(yt_np)
    Xv = torch.from_numpy(xv_np)
    pair_pos_np, pair_neg_np = make_within_user_pairs(train_users, yt_np)
    pair_pos = torch.from_numpy(pair_pos_np)
    pair_neg = torch.from_numpy(pair_neg_np)

    model = FM(total_dim)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = torch.nn.BCEWithLogitsLoss()
    n = len(yt)
    bs = 8192
    best = -1.0
    best_scores = None
    patience = 0
    history = []

    for epoch in range(epochs):
        model.train()
        row_perm = torch.randperm(n)
        if len(pair_pos) > 0:
            pair_perm = torch.randperm(len(pair_pos))
        last_loss = 0.0
        for i in range(0, n, bs):
            idx = row_perm[i:i + bs]
            opt.zero_grad()
            point_loss = bce(model(Xt[idx]), yt[idx])
            if len(pair_pos) > 0:
                positions = torch.arange(i, i + len(idx), dtype=torch.long) % len(pair_pos)
                selected = pair_perm[positions]
                positive_scores = model(Xt[pair_pos[selected]])
                negative_scores = model(Xt[pair_neg[selected]])
                rank_loss = torch.nn.functional.softplus(
                    negative_scores - positive_scores).mean()
                loss = point_loss + 0.25 * rank_loss
            else:
                loss = point_loss
            loss.backward()
            opt.step()
            last_loss = float(loss.item())

        model.eval()
        with torch.no_grad():
            scores = np.concatenate([
                model(Xv[i:i + 65536]).numpy()
                for i in range(0, len(Xv), 65536)
            ])
        m = evaluate(val_users, yv_np, scores)
        primary = m["primary"]
        history.append({
            "epoch": epoch + 1,
            "train_loss": round(last_loss, 5),
            "val_gauc": round(m.get("GAUC", m.get("gauc", 0.0)), 6),
            "val_primary": round(primary, 6),
        })
        if primary > best + 1e-6:
            best = primary
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
                model(Xv[i:i + 65536]).numpy()
                for i in range(0, len(Xv), 65536)
            ])

    os.makedirs(a.out_dir, exist_ok=True)
    m = evaluate(val_users, yv_np, best_scores)
    with open(os.path.join(a.out_dir, "metrics.json"), "w") as fh:
        json.dump({
            "gauc": m["GAUC"] if "GAUC" in m else m["gauc"],
            "ndcg5": m.get("nDCG@5", m.get("ndcg5")),
            "primary": m["primary"],
            "history": history,
        }, fh)

    with open(os.path.join(a.out_dir, "predictions.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(best_scores):
            writer.writerow([i, output_users[i], output_videos[i], f"{score:.6g}"])


if __name__ == "__main__":
    main()
