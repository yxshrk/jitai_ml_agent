import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.official.evaluate import evaluate


class DCNLite(torch.nn.Module):
    def __init__(self, total_dim, num_fields=5, k=8, hidden=128):
        super().__init__()
        self.emb = torch.nn.Embedding(total_dim, k)
        self.linear = torch.nn.Embedding(total_dim, 1)
        input_dim = num_fields * k
        self.cross_weight = torch.nn.Parameter(torch.empty(input_dim))
        self.cross_bias = torch.nn.Parameter(torch.zeros(input_dim))
        self.deep = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden),
            torch.nn.ReLU(),
        )
        self.output = torch.nn.Linear(input_dim + hidden, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.normal_(self.cross_weight, std=0.01)
        torch.nn.init.xavier_uniform_(self.deep[0].weight)
        torch.nn.init.zeros_(self.deep[0].bias)
        torch.nn.init.xavier_uniform_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)

    def forward(self, x):
        x0 = self.emb(x).flatten(1)
        cross_scale = torch.matmul(x0, self.cross_weight).unsqueeze(1)
        cross = x0 + x0 * cross_scale + self.cross_bias
        deep = self.deep(x0)
        interaction = self.output(torch.cat((cross, deep), dim=1)).squeeze(1)
        linear = self.linear(x).sum((1, 2))
        return self.bias + linear + interaction


def load_csv_data(data_dir):
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")

    train_rows = []
    with open(train_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            train_rows.append({
                "user": row["user_id"],
                "video": row["video_id"],
                "author": row.get("author_id", "0"),
                "tab": row["tab"],
                "duration": float(row["duration_ms"]),
                "y": float(row["long_view"]),
            })

    durations = np.asarray([r["duration"] for r in train_rows], dtype=np.float64)
    edges = np.quantile(durations, np.linspace(0.1, 0.9, 9))
    edges = np.unique(edges)

    raw_train = [
        [r["user"], r["video"], r["author"], r["tab"], str(int(np.searchsorted(edges, r["duration"], side="right")))]
        for r in train_rows
    ]
    mappings = []
    for field in range(5):
        values = sorted({row[field] for row in raw_train})
        mappings.append({value: i for i, value in enumerate(values)})

    field_dims = np.asarray([len(mapping) + 1 for mapping in mappings], dtype=np.int64)
    offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(field_dims)[:-1]))

    def encode(rows):
        out = np.empty((len(rows), 5), dtype=np.int64)
        for i, row in enumerate(rows):
            for field in range(5):
                local = mappings[field].get(row[field], len(mappings[field]))
                out[i, field] = local + offsets[field]
        return out

    val_rows = []
    with open(val_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            duration = float(row["duration_ms"])
            val_rows.append({
                "raw": [
                    row["user_id"],
                    row["video_id"],
                    row.get("author_id", "0"),
                    row["tab"],
                    str(int(np.searchsorted(edges, duration, side="right"))),
                ],
                "user": row["user_id"],
                "video": row["video_id"],
                "y": float(row["long_view"]),
            })

    train_raw = raw_train
    val_raw = [r["raw"] for r in val_rows]
    return {
        "train_X": encode(train_raw),
        "train_y": np.asarray([r["y"] for r in train_rows], dtype=np.float32),
        "train_user": np.asarray([r["user"] for r in train_rows]),
        "val_X": encode(val_raw),
        "val_y": np.asarray([r["y"] for r in val_rows], dtype=np.float32),
        "val_user": np.asarray([r["user"] for r in val_rows]),
        "val_video": np.asarray([r["video"] for r in val_rows]),
        "field_dims": field_dims,
    }


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        with np.load(train_npz) as tr, np.load(val_npz) as va:
            field_dims = tr["field_dims"].astype(np.int64)
            video_offset = int(field_dims[0])
            return {
                "train_X": tr["X"].astype(np.int64),
                "train_y": tr["y"].astype(np.float32),
                "train_user": tr["user"].copy(),
                "val_X": va["X"].astype(np.int64),
                "val_y": va["y"].astype(np.float32),
                "val_user": va["user"].copy(),
                "val_video": (va["X"][:, 1].astype(np.int64) - video_offset),
                "field_dims": field_dims,
            }
    return load_csv_data(data_dir)


def make_pair_arrays(users, labels):
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    sorted_labels = labels[order] > 0.5
    if len(order) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    starts = np.concatenate(([0], np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1))
    ends = np.concatenate((starts[1:], [len(order)]))
    group_ids = np.repeat(np.arange(len(starts), dtype=np.int64), ends - starts)
    pos_counts = np.bincount(group_ids[sorted_labels], minlength=len(starts))
    neg_counts = np.bincount(group_ids[~sorted_labels], minlength=len(starts))
    eligible = (pos_counts > 0) & (neg_counts > 0)

    pos_mask = sorted_labels & eligible[group_ids]
    pair_pos = order[pos_mask].astype(np.int64)
    pair_group = group_ids[pos_mask]
    negative_indices = order[~sorted_labels].astype(np.int64)
    neg_starts = np.cumsum(neg_counts) - neg_counts
    return pair_pos, pair_group, negative_indices, neg_starts, neg_counts


def metric_value(metrics, upper, lower):
    return float(metrics[upper] if upper in metrics else metrics[lower])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)

    data = load_data(args.data_dir)
    train_X = torch.from_numpy(data["train_X"])
    train_y = torch.from_numpy(data["train_y"])
    val_X = torch.from_numpy(data["val_X"])
    total_dim = int(data["field_dims"].sum())

    model = DCNLite(total_dim=total_dim, num_fields=5, k=8, hidden=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = torch.nn.BCEWithLogitsLoss()

    pair_pos, pair_group, negative_indices, neg_starts, neg_counts = make_pair_arrays(
        data["train_user"], data["train_y"]
    )
    pair_pos_t = torch.from_numpy(pair_pos)
    negative_indices_t = torch.from_numpy(negative_indices)

    n = len(train_y)
    batch_size = 8192
    best_gauc = -1.0
    best_scores = None
    patience = 0
    rng = np.random.RandomState(args.seed)

    for _ in range(args.epochs):
        if len(pair_pos) > 0:
            sampled_offsets = (rng.random(len(pair_pos)) * neg_counts[pair_group]).astype(np.int64)
            pair_neg = negative_indices[neg_starts[pair_group] + sampled_offsets]
            pair_neg_t = torch.from_numpy(pair_neg)
        else:
            pair_neg_t = torch.empty(0, dtype=torch.int64)

        model.train()
        permutation = torch.randperm(n)
        for start in range(0, n, batch_size):
            batch_idx = permutation[start:start + batch_size]
            current_size = len(batch_idx)

            optimizer.zero_grad()
            if len(pair_pos_t) > 0:
                sampled_pairs = torch.randint(0, len(pair_pos_t), (current_size,))
                pos_idx = pair_pos_t[sampled_pairs]
                neg_idx = pair_neg_t[sampled_pairs]
                all_X = torch.cat((train_X[batch_idx], train_X[pos_idx], train_X[neg_idx]), dim=0)
                all_scores = model(all_X)
                main_scores = all_scores[:current_size]
                pos_scores = all_scores[current_size:2 * current_size]
                neg_scores = all_scores[2 * current_size:]
                point_loss = bce(main_scores, train_y[batch_idx])
                pair_loss = torch.nn.functional.softplus(-(pos_scores - neg_scores)).mean()
                loss = 0.5 * point_loss + 0.5 * pair_loss
            else:
                loss = bce(model(train_X[batch_idx]), train_y[batch_idx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            scores = np.concatenate([
                model(val_X[i:i + 65536]).cpu().numpy()
                for i in range(0, len(val_X), 65536)
            ])
        metrics = evaluate(data["val_user"], data["val_y"].astype(int), scores)
        gauc = metric_value(metrics, "GAUC", "gauc")
        if gauc > best_gauc + 1e-6:
            best_gauc = gauc
            best_scores = scores.copy()
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break

    final_metrics = evaluate(data["val_user"], data["val_y"].astype(int), best_scores)
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
            writer.writerow([i, data["val_user"][i], data["val_video"][i], format(float(score), ".8g")])


if __name__ == "__main__":
    main()
