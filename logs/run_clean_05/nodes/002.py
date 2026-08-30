"""Regularized DeepFM trained with a within-user BPR and pointwise BCE hybrid."""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RegularizedDeepFM(torch.nn.Module):
    def __init__(self, total_dim, fields=5, k=16, dropout=0.15):
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
        torch.nn.init.normal_(self.mlp[-1].weight, std=0.01)

    def forward(self, x):
        e = self.emb(x)
        summed = e.sum(1)
        pair = 0.5 * (summed * summed - (e * e).sum(1)).sum(1)
        fm = self.bias + self.lin(x).sum((1, 2)) + pair
        deep = self.mlp(e.reshape(e.shape[0], -1)).squeeze(1)
        return fm + deep

    def accessed_row_l2(self, x):
        rows = torch.unique(x)
        return self.emb(rows).square().sum() + self.lin(rows).square().sum()


def load_npz(data_dir):
    tr = np.load(os.path.join(data_dir, "train.npz"))
    va = np.load(os.path.join(data_dir, "val.npz"))
    xt = tr["X"].astype(np.int64)
    yt = tr["y"].astype(np.float32)
    xv = va["X"].astype(np.int64)
    yv = va["y"].astype(np.float32)
    users = va["user"]
    dims = tr["field_dims"].astype(np.int64)
    video_offset = int(dims[0])
    videos = xv[:, 1] - video_offset
    return xt, yt, xv, yv, users, videos, dims


def load_csv(data_dir):
    maps = {"user_id": {}, "video_id": {}, "tab": {}, "dur": {}}
    train_values = []
    train_y = []

    def add_value(name, value):
        mapping = maps[name]
        if value not in mapping:
            mapping[value] = len(mapping) + 1
        return mapping[value]

    with open(os.path.join(data_dir, "train.csv"), "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            duration = int(float(row["duration_ms"])) if row["duration_ms"] else 0
            dur_bucket = str(min(max(duration // 5000, 0), 120))
            train_values.append((
                add_value("user_id", row["user_id"]),
                add_value("video_id", row["video_id"]),
                0,
                add_value("tab", row["tab"]),
                add_value("dur", dur_bucket),
            ))
            train_y.append(float(row["long_view"]))

    val_values = []
    val_y = []
    val_users = []
    val_videos = []
    with open(os.path.join(data_dir, "val.csv"), "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            duration = int(float(row["duration_ms"])) if row["duration_ms"] else 0
            dur_bucket = str(min(max(duration // 5000, 0), 120))
            val_values.append((
                maps["user_id"].get(row["user_id"], 0),
                maps["video_id"].get(row["video_id"], 0),
                0,
                maps["tab"].get(row["tab"], 0),
                maps["dur"].get(dur_bucket, 0),
            ))
            val_y.append(float(row["long_view"]))
            val_users.append(row["user_id"])
            val_videos.append(row["video_id"])

    dims = np.asarray([
        len(maps["user_id"]) + 1,
        len(maps["video_id"]) + 1,
        1,
        len(maps["tab"]) + 1,
        len(maps["dur"]) + 1,
    ], dtype=np.int64)
    offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(dims)[:-1]))
    xt = np.asarray(train_values, dtype=np.int64) + offsets
    xv = np.asarray(val_values, dtype=np.int64) + offsets
    return (xt, np.asarray(train_y, dtype=np.float32), xv,
            np.asarray(val_y, dtype=np.float32), np.asarray(val_users),
            np.asarray(val_videos), dims)


def make_user_groups(encoded_users):
    order = np.argsort(encoded_users, kind="stable")
    sorted_users = encoded_users[order]
    if len(order) == 0:
        return []
    boundaries = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
    return [part.astype(np.int64, copy=False)
            for part in np.split(order, boundaries)]


def make_group_batches(groups, labels, batch_size, rng):
    group_order = rng.permutation(len(groups))
    packed = []
    packed_size = 0

    def construct(group_ids):
        row_parts = []
        positive_parts = []
        negative_parts = []
        offset = 0
        for group_id in group_ids:
            rows = groups[int(group_id)]
            row_parts.append(rows)
            group_labels = labels[rows]
            positives = np.flatnonzero(group_labels > 0.5)
            negatives = np.flatnonzero(group_labels <= 0.5)
            if len(positives) and len(negatives):
                positives = positives[rng.permutation(len(positives))]
                negatives = negatives[rng.permutation(len(negatives))]
                pair_count = max(len(positives), len(negatives))
                positive_parts.append(
                    offset + positives[np.arange(pair_count) % len(positives)])
                negative_parts.append(
                    offset + negatives[np.arange(pair_count) % len(negatives)])
            offset += len(rows)
        batch_rows = np.concatenate(row_parts)
        if positive_parts:
            pair_pos = np.concatenate(positive_parts).astype(np.int64, copy=False)
            pair_neg = np.concatenate(negative_parts).astype(np.int64, copy=False)
        else:
            pair_pos = np.empty(0, dtype=np.int64)
            pair_neg = np.empty(0, dtype=np.int64)
        return batch_rows, pair_pos, pair_neg

    for group_id in group_order:
        group_size = len(groups[int(group_id)])
        if packed and packed_size + group_size > batch_size:
            yield construct(packed)
            packed = []
            packed_size = 0
        packed.append(int(group_id))
        packed_size += group_size
    if packed:
        yield construct(packed)


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

    fast_path = (os.path.exists(os.path.join(args.data_dir, "train.npz")) and
                 os.path.exists(os.path.join(args.data_dir, "val.npz")))
    if fast_path:
        from data.official.evaluate import evaluate
        xt_np, yt_np, xv_np, yv, users, videos, field_dims = load_npz(args.data_dir)
    else:
        from harness.evaluate_provisional import evaluate
        xt_np, yt_np, xv_np, yv, users, videos, field_dims = load_csv(args.data_dir)

    xt = torch.from_numpy(xt_np)
    yt = torch.from_numpy(yt_np)
    xv = torch.from_numpy(xv_np)
    user_groups = make_user_groups(xt_np[:, 0])

    model = RegularizedDeepFM(int(field_dims.sum()))
    embedding_params = [model.emb.weight, model.lin.weight, model.bias]
    dense_params = list(model.mlp.parameters())
    optimizer = torch.optim.AdamW([
        {"params": embedding_params, "weight_decay": 0.0},
        {"params": dense_params, "weight_decay": 1e-4},
    ], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
    criterion = torch.nn.BCEWithLogitsLoss()

    batch_size = 8192
    bpr_weight = 0.3
    best_primary = -1.0
    best_scores = None
    patience = 0

    for epoch in range(epochs):
        model.train()
        rng = np.random.default_rng(args.seed + epoch)
        for idx_np, pos_np, neg_np in make_group_batches(
                user_groups, yt_np, batch_size, rng):
            idx = torch.from_numpy(idx_np)
            xb = xt[idx]
            targets = yt[idx]
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, targets)
            if len(pos_np):
                pos_idx = torch.from_numpy(pos_np)
                neg_idx = torch.from_numpy(neg_np)
                bpr_loss = torch.nn.functional.softplus(
                    -(logits[pos_idx] - logits[neg_idx])).mean()
                loss = loss + bpr_weight * bpr_loss
            loss = loss + 1e-6 * model.accessed_row_l2(xb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            scores = np.concatenate([
                model(xv[start:start + 65536]).cpu().numpy()
                for start in range(0, len(xv), 65536)
            ])
        metrics = evaluate(users, yv.astype(int), scores)
        primary = float(metrics["primary"])
        if primary > best_primary + 1e-6:
            best_primary = primary
            best_scores = scores.copy()
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break

    final_metrics = evaluate(users, yv.astype(int), best_scores)
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
        for i, score in enumerate(best_scores):
            writer.writerow([i, users[i], videos[i], format(float(score), ".8g")])


if __name__ == "__main__":
    main()
