"""FM with a hybrid pointwise BCE and within-user BPR objective."""
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
        summed = e.sum(1)
        pair = 0.5 * (summed * summed - (e * e).sum(1)).sum(1)
        return self.bias + self.lin(x).sum((1, 2)) + pair


def make_within_user_pairs(users, labels, rng, max_pairs=200000):
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    cuts = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
    starts = np.concatenate((np.array([0], dtype=np.int64), cuts))
    ends = np.concatenate((cuts, np.array([len(order)], dtype=np.int64)))
    positives = []
    negatives = []
    total = 0
    for start, end in zip(starts, ends):
        group = order[start:end]
        pos = group[labels[group] > 0.5]
        neg = group[labels[group] <= 0.5]
        if len(pos) == 0 or len(neg) == 0:
            continue
        count = min(max(len(pos), len(neg)), max_pairs - total)
        if count <= 0:
            break
        positives.append(rng.choice(pos, size=count, replace=len(pos) < count))
        negatives.append(rng.choice(neg, size=count, replace=len(neg) < count))
        total += count
    if not positives:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    return (
        np.concatenate(positives).astype(np.int64, copy=False),
        np.concatenate(negatives).astype(np.int64, copy=False),
    )


def duration_bucket(value):
    try:
        duration = max(0, int(float(value)))
    except (TypeError, ValueError):
        duration = 0
    return str(min(duration // 10000, 120))


def read_csv_data(data_dir):
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")

    train_values = [[], [], [], [], []]
    train_y = []
    train_users = []
    with open(train_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            user = row["user_id"]
            video = row["video_id"]
            train_values[0].append(user)
            train_values[1].append(video)
            train_values[2].append("0")
            train_values[3].append(row["tab"])
            train_values[4].append(duration_bucket(row["duration_ms"]))
            train_y.append(float(row["long_view"]))
            train_users.append(user)

    mappings = []
    field_dims = []
    for values in train_values:
        mapping = {}
        for value in values:
            if value not in mapping:
                mapping[value] = len(mapping)
        mappings.append(mapping)
        field_dims.append(len(mapping) + 1)

    offsets = np.cumsum(np.array([0] + field_dims[:-1], dtype=np.int64))
    train_x = np.empty((len(train_y), 5), dtype=np.int64)
    for field in range(5):
        mapping = mappings[field]
        unknown = len(mapping)
        train_x[:, field] = np.fromiter(
            (mapping.get(value, unknown) + offsets[field] for value in train_values[field]),
            dtype=np.int64,
            count=len(train_y),
        )

    val_values = [[], [], [], [], []]
    val_y = []
    val_users = []
    val_videos = []
    with open(val_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            user = row["user_id"]
            video = row["video_id"]
            val_values[0].append(user)
            val_values[1].append(video)
            val_values[2].append("0")
            val_values[3].append(row["tab"])
            val_values[4].append(duration_bucket(row["duration_ms"]))
            val_y.append(float(row["long_view"]))
            val_users.append(user)
            val_videos.append(video)

    val_x = np.empty((len(val_y), 5), dtype=np.int64)
    for field in range(5):
        mapping = mappings[field]
        unknown = len(mapping)
        val_x[:, field] = np.fromiter(
            (mapping.get(value, unknown) + offsets[field] for value in val_values[field]),
            dtype=np.int64,
            count=len(val_y),
        )

    return {
        "train_x": train_x,
        "train_y": np.asarray(train_y, dtype=np.float32),
        "train_users": np.asarray(train_users),
        "val_x": val_x,
        "val_y": np.asarray(val_y, dtype=np.float32),
        "val_users": np.asarray(val_users),
        "val_videos": np.asarray(val_videos),
        "field_dims": np.asarray(field_dims, dtype=np.int64),
        "npz": False,
    }


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.isfile(train_npz) and os.path.isfile(val_npz):
        with np.load(train_npz) as tr, np.load(val_npz) as va:
            field_dims = np.asarray(tr["field_dims"], dtype=np.int64)
            val_x = va["X"].astype(np.int64)
            video_offset = int(field_dims[0])
            return {
                "train_x": tr["X"].astype(np.int64),
                "train_y": tr["y"].astype(np.float32),
                "train_users": np.asarray(tr["user"]).copy(),
                "val_x": val_x,
                "val_y": va["y"].astype(np.float32),
                "val_users": np.asarray(va["user"]).copy(),
                "val_videos": (val_x[:, 1] - video_offset).copy(),
                "field_dims": field_dims,
                "npz": True,
            }
    return read_csv_data(data_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    epochs = args.epochs
    if "SMOKE_EPOCHS" in os.environ:
        epochs = min(epochs, int(os.environ["SMOKE_EPOCHS"]))
    epochs = max(1, epochs)

    data = load_data(args.data_dir)
    if data["npz"]:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate

    train_x_np = data["train_x"]
    train_y_np = data["train_y"]
    train_users = data["train_users"]
    val_x_np = data["val_x"]
    val_y_np = data["val_y"]
    val_users = data["val_users"]
    val_videos = data["val_videos"]

    train_x = torch.from_numpy(train_x_np)
    train_y = torch.from_numpy(train_y_np)
    val_x = torch.from_numpy(val_x_np)

    model = FM(int(data["field_dims"].sum()))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = torch.nn.BCEWithLogitsLoss()
    n = len(train_y)
    batch_size = 8192
    pair_batch_size = 1024
    best_primary = -1.0
    best_scores = None
    patience = 0

    for epoch in range(epochs):
        rng = np.random.RandomState(args.seed + epoch)
        pos_np, neg_np = make_within_user_pairs(train_users, train_y_np, rng)
        pos_idx = torch.from_numpy(pos_np)
        neg_idx = torch.from_numpy(neg_np)
        pair_count = len(pos_idx)

        model.train()
        permutation = torch.randperm(n)
        pair_permutation = torch.randperm(pair_count) if pair_count else None
        pair_cursor = 0

        for start in range(0, n, batch_size):
            idx = permutation[start:start + batch_size]
            optimizer.zero_grad()
            loss = bce(model(train_x[idx]), train_y[idx])

            if pair_count:
                take = min(pair_batch_size, pair_count)
                if pair_cursor + take <= pair_count:
                    selected = pair_permutation[pair_cursor:pair_cursor + take]
                    pair_cursor += take
                else:
                    first = pair_permutation[pair_cursor:]
                    pair_permutation = torch.randperm(pair_count)
                    remaining = take - len(first)
                    selected = torch.cat((first, pair_permutation[:remaining]))
                    pair_cursor = remaining

                pair_x = torch.cat(
                    (train_x[pos_idx[selected]], train_x[neg_idx[selected]]), dim=0
                )
                pair_scores = model(pair_x)
                score_difference = pair_scores[:take] - pair_scores[take:]
                bpr_loss = torch.nn.functional.softplus(-score_difference).mean()
                loss = loss + 0.3 * bpr_loss

            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            scores = np.concatenate([
                model(val_x[start:start + 65536]).numpy()
                for start in range(0, len(val_x), 65536)
            ])
        metrics = evaluate(val_users, val_y_np.astype(int), scores)
        primary = float(metrics["primary"])
        if best_scores is None or primary > best_primary + 1e-6:
            best_primary = primary
            best_scores = scores.copy()
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break

    final_metrics = evaluate(val_users, val_y_np.astype(int), best_scores)
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
        for row_id, (user, video, score) in enumerate(
            zip(val_users, val_videos, best_scores)
        ):
            writer.writerow([row_id, user, video, format(float(score), ".9g")])


if __name__ == "__main__":
    main()
