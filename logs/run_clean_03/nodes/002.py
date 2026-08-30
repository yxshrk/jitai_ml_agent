"""FM with a hybrid pointwise BCE and within-user BPR objective.

Uses complete user groups in each training batch, forms positive/negative pairs only
within users, and retains all impressions (including one-class users) for BCE.
"""
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


def _duration_bucket(value):
    try:
        value = max(0, int(float(value)))
    except (TypeError, ValueError):
        value = 0
    return int(np.floor(np.log2(value + 1)))


def _load_csv(data_dir):
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")

    with open(train_path, newline="") as fh:
        train_rows = list(csv.DictReader(fh))
    with open(val_path, newline="") as fh:
        val_rows = list(csv.DictReader(fh))

    def make_map(values):
        unique = sorted(set(values), key=lambda x: str(x))
        return {value: i for i, value in enumerate(unique)}

    user_map = make_map(row["user_id"] for row in train_rows)
    video_map = make_map(row["video_id"] for row in train_rows)
    tab_map = make_map(row["tab"] for row in train_rows)
    dur_map = make_map(_duration_bucket(row["duration_ms"]) for row in train_rows)

    field_dims = np.asarray([
        len(user_map) + 1,
        len(video_map) + 1,
        1,
        len(tab_map) + 1,
        len(dur_map) + 1,
    ], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1])).astype(np.int64)

    def encode(rows):
        x = np.empty((len(rows), 5), dtype=np.int64)
        users = []
        videos = []
        labels = np.empty(len(rows), dtype=np.float32)
        for i, row in enumerate(rows):
            raw = np.asarray([
                user_map.get(row["user_id"], len(user_map)),
                video_map.get(row["video_id"], len(video_map)),
                0,
                tab_map.get(row["tab"], len(tab_map)),
                dur_map.get(_duration_bucket(row["duration_ms"]), len(dur_map)),
            ], dtype=np.int64)
            x[i] = raw + offsets
            users.append(row["user_id"])
            videos.append(row["video_id"])
            labels[i] = float(row["long_view"])
        return x, labels, np.asarray(users), np.asarray(videos)

    xt, yt, train_users, _ = encode(train_rows)
    xv, yv, val_users, val_videos = encode(val_rows)
    return xt, yt, train_users, xv, yv, val_users, val_videos, field_dims


def _load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        tr = np.load(train_npz)
        va = np.load(val_npz)
        field_dims = tr["field_dims"].astype(np.int64)
        xv = va["X"].astype(np.int64)
        video_offset = int(field_dims[0])
        val_videos = xv[:, 1] - video_offset
        return (
            tr["X"].astype(np.int64),
            tr["y"].astype(np.float32),
            np.asarray(tr["user"]),
            xv,
            va["y"].astype(np.float32),
            np.asarray(va["user"]),
            val_videos,
            field_dims,
            True,
        )

    xt, yt, train_users, xv, yv, val_users, val_videos, field_dims = _load_csv(data_dir)
    return xt, yt, train_users, xv, yv, val_users, val_videos, field_dims, False


def _make_user_groups(users):
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    if len(order) == 0:
        return order, np.asarray([0], dtype=np.int64)
    boundaries = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
    boundaries = np.concatenate(([0], boundaries, [len(order)])).astype(np.int64)
    return order.astype(np.int64), boundaries


def _group_batches(group_order, sorted_indices, boundaries, batch_size):
    current = []
    current_size = 0
    for group_id in group_order:
        start = int(boundaries[group_id])
        end = int(boundaries[group_id + 1])
        group_size = end - start
        if current and current_size + group_size > batch_size:
            yield current
            current = []
            current_size = 0
        current.append((start, end))
        current_size += group_size
    if current:
        yield current


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    epochs = args.epochs
    smoke_epochs = os.environ.get("SMOKE_EPOCHS")
    if smoke_epochs is not None:
        epochs = min(epochs, max(1, int(smoke_epochs)))

    xt_np, yt_np, train_users, xv_np, yv_np, val_users, val_videos, field_dims, fast_path = _load_data(args.data_dir)

    xt = torch.from_numpy(xt_np)
    yt = torch.from_numpy(yt_np)
    xv = torch.from_numpy(xv_np)

    model = FM(int(field_dims.sum()))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = torch.nn.BCEWithLogitsLoss()
    softplus = torch.nn.Softplus()

    sorted_indices, boundaries = _make_user_groups(train_users)
    group_count = len(boundaries) - 1
    batch_size = 8192

    if fast_path:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate

    best = -1.0
    best_scores = None
    patience = 0

    for _epoch in range(epochs):
        model.train()
        group_order = np.random.permutation(group_count)

        for spans in _group_batches(group_order, sorted_indices, boundaries, batch_size):
            global_parts = [sorted_indices[start:end] for start, end in spans]
            global_indices = np.concatenate(global_parts)
            idx = torch.from_numpy(global_indices)

            opt.zero_grad()
            logits = model(xt[idx])
            point_loss = bce(logits, yt[idx])

            positive_positions = []
            negative_positions = []
            local_offset = 0
            for start, end in spans:
                group_global = sorted_indices[start:end]
                group_labels = yt_np[group_global]
                positives = np.flatnonzero(group_labels > 0.5)
                negatives = np.flatnonzero(group_labels <= 0.5)
                if len(positives) and len(negatives):
                    sampled_negatives = negatives[np.random.randint(0, len(negatives), size=len(positives))]
                    positive_positions.append(positives + local_offset)
                    negative_positions.append(sampled_negatives + local_offset)
                local_offset += end - start

            if positive_positions:
                pos_idx = torch.from_numpy(np.concatenate(positive_positions).astype(np.int64))
                neg_idx = torch.from_numpy(np.concatenate(negative_positions).astype(np.int64))
                rank_loss = softplus(-(logits[pos_idx] - logits[neg_idx])).mean()
                loss = point_loss + 0.5 * rank_loss
            else:
                loss = point_loss

            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            scores = np.concatenate([
                model(xv[i:i + 65536]).numpy()
                for i in range(0, len(xv), 65536)
            ])
        metrics = evaluate(val_users, yv_np.astype(int), scores)
        primary = float(metrics["primary"])
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
                model(xv[i:i + 65536]).numpy()
                for i in range(0, len(xv), 65536)
            ])

    final_metrics = evaluate(val_users, yv_np.astype(int), best_scores)
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
            writer.writerow([i, val_users[i], val_videos[i], format(float(score), ".6g")])


if __name__ == "__main__":
    main()
