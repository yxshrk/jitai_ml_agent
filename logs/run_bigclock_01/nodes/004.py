import argparse
import csv
import datetime
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_npz(data_dir):
    from data.official.evaluate import evaluate
    tr = np.load(os.path.join(data_dir, "train.npz"))
    va = np.load(os.path.join(data_dir, "val.npz"))
    data = {
        "Xt": tr["X"].astype(np.int64),
        "yt": tr["y"].astype(np.float32),
        "ut": tr["user"],
        "dates": tr["date"],
        "Xv": va["X"].astype(np.int64),
        "yv": va["y"].astype(np.int64),
        "uv": va["user"],
        "video_v": np.zeros(len(va["y"]), dtype=np.int64),
        "field_dims": tr["field_dims"].astype(np.int64),
    }
    return data, evaluate


def scalar_value(value):
    try:
        return int(value)
    except ValueError:
        return value


def load_csv_data(data_dir):
    from harness.evaluate_provisional import evaluate

    train_rows = []
    with open(os.path.join(data_dir, "train.csv"), newline="") as fh:
        for row in csv.DictReader(fh):
            train_rows.append({
                "user": row["user_id"],
                "video": row["video_id"],
                "tab": row["tab"],
                "duration": float(row["duration_ms"]),
                "date": row["date"],
                "label": float(row["long_view"]),
            })

    val_rows = []
    with open(os.path.join(data_dir, "val.csv"), newline="") as fh:
        for row in csv.DictReader(fh):
            val_rows.append({
                "user": row["user_id"],
                "video": row["video_id"],
                "tab": row["tab"],
                "duration": float(row["duration_ms"]),
                "label": int(float(row["long_view"])),
            })

    def make_map(values):
        return {value: i + 1 for i, value in enumerate(sorted(set(values)))}

    user_map = make_map([row["user"] for row in train_rows])
    video_map = make_map([row["video"] for row in train_rows])
    tab_map = make_map([row["tab"] for row in train_rows])
    durations = np.asarray([row["duration"] for row in train_rows], dtype=np.float64)
    edges = np.unique(np.quantile(durations, np.linspace(0.1, 0.9, 9)))
    field_dims = np.asarray([
        len(user_map) + 1,
        len(video_map) + 1,
        1,
        len(tab_map) + 1,
        len(edges) + 2,
    ], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1]))

    def encode(rows):
        x = np.zeros((len(rows), 5), dtype=np.int64)
        for i, row in enumerate(rows):
            x[i, 0] = user_map.get(row["user"], 0)
            x[i, 1] = video_map.get(row["video"], 0)
            x[i, 2] = 0
            x[i, 3] = tab_map.get(row["tab"], 0)
            x[i, 4] = int(np.searchsorted(edges, row["duration"], side="right")) + 1
        return x + offsets[None, :]

    data = {
        "Xt": encode(train_rows),
        "yt": np.asarray([row["label"] for row in train_rows], dtype=np.float32),
        "ut": np.asarray([scalar_value(row["user"]) for row in train_rows]),
        "dates": np.asarray([row["date"] for row in train_rows]),
        "Xv": encode(val_rows),
        "yv": np.asarray([row["label"] for row in val_rows], dtype=np.int64),
        "uv": np.asarray([scalar_value(row["user"]) for row in val_rows]),
        "video_v": np.asarray([scalar_value(row["video"]) for row in val_rows]),
        "field_dims": field_dims,
    }
    return data, evaluate


def date_ordinal(value):
    text = str(value.decode() if isinstance(value, bytes) else value)
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(char for char in text if char.isdigit())
    if len(digits) >= 8:
        try:
            return datetime.datetime.strptime(digits[:8], "%Y%m%d").date().toordinal()
        except ValueError:
            pass
    try:
        return int(float(text))
    except ValueError:
        return 0


def recency_weights(dates, half_life):
    unique_dates = np.unique(dates)
    mapping = {value: date_ordinal(value) for value in unique_dates}
    ordinals = np.asarray([mapping[value] for value in dates], dtype=np.float32)
    ages = ordinals.max() - ordinals
    weights = np.exp(-math.log(2.0) * ages / float(half_life)).astype(np.float32)
    return weights / max(float(weights.mean()), 1e-6)


def make_pairs(users, labels, seed):
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    rng = np.random.RandomState(seed)
    positive_parts = []
    negative_parts = []
    for j in range(len(boundaries) - 1):
        rows = order[boundaries[j]:boundaries[j + 1]]
        positives = rows[labels[rows] > 0.5]
        negatives = rows[labels[rows] <= 0.5]
        if len(positives) and len(negatives):
            positive_parts.append(positives)
            negative_parts.append(rng.choice(negatives, size=len(positives), replace=True))
    if not positive_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return (np.concatenate(positive_parts).astype(np.int64),
            np.concatenate(negative_parts).astype(np.int64))


class DCNLite(torch.nn.Module):
    def __init__(self, total_dim, fields=5, k=16, hidden=128, dropout=0.23):
        super().__init__()
        width = fields * k
        self.embedding = torch.nn.Embedding(total_dim, k)
        self.linear = torch.nn.Embedding(total_dim, 1)
        self.embedding_dropout = torch.nn.Dropout(dropout)
        self.cross_weight = torch.nn.Parameter(torch.empty(width))
        self.cross_bias = torch.nn.Parameter(torch.zeros(width))
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(width, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, hidden // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.head = torch.nn.Linear(width + hidden // 2, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        torch.nn.init.normal_(self.embedding.weight, std=0.01)
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.normal_(self.cross_weight, std=0.01)
        torch.nn.init.xavier_uniform_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)

    def forward(self, x):
        embedded = self.embedding_dropout(self.embedding(x)).flatten(1)
        cross = (embedded * torch.sum(embedded * self.cross_weight, dim=1, keepdim=True)
                 + self.cross_bias + embedded)
        deep = self.mlp(embedded)
        linear = self.linear(x).sum((1, 2))
        return self.bias + linear + self.head(torch.cat((cross, deep), dim=1)).squeeze(1)


def metric_values(evaluator, users, labels, scores):
    result = evaluator(users, labels, scores)
    return {
        "gauc": float(result.get("GAUC", result.get("gauc"))),
        "ndcg5": float(result.get("nDCG@5", result.get("ndcg5"))),
        "primary": float(result["primary"]),
    }


def predict(model, x, batch_size=65536):
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            parts.append(model(x[start:start + batch_size]).cpu().numpy())
    return np.concatenate(parts)


def rank_transform(users, scores):
    transformed = np.empty(len(scores), dtype=np.float64)
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    for j in range(len(boundaries) - 1):
        rows = order[boundaries[j]:boundaries[j + 1]]
        local_order = np.argsort(scores[rows], kind="mergesort")
        ranks = np.empty(len(rows), dtype=np.float64)
        ranks[local_order] = np.arange(len(rows), dtype=np.float64)
        transformed[rows] = ranks / max(len(rows) - 1, 1)
    return transformed


def train_member(config, epochs, seed, tensors, evaluator, history, member_index):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    model = DCNLite(tensors["total_dim"], k=16, hidden=128, dropout=config["dropout"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    weights = torch.from_numpy(recency_weights(tensors["dates_np"], config["half_life"]))
    train_indices = np.arange(len(tensors["yt"]), dtype=np.int64)
    pair_indices = np.arange(len(tensors["pos"]), dtype=np.int64)
    batch_size = 8192
    best_primary = -1.0
    best_scores = None
    best_epoch = 0
    best_half = 0
    global_step = 0

    for epoch in range(epochs):
        shuffled_rows = train_indices[rng.permutation(len(train_indices))]
        if len(pair_indices):
            shuffled_pairs = pair_indices[rng.permutation(len(pair_indices))]
        else:
            shuffled_pairs = pair_indices
        row_splits = np.array_split(shuffled_rows, 2)
        pair_splits = np.array_split(shuffled_pairs, 2)

        for half in range(2):
            model.train()
            row_part = row_splits[half]
            pair_part = pair_splits[half]
            steps = max(1, int(math.ceil(max(len(row_part), len(pair_part), 1) /
                                         float(batch_size))))
            last_loss = 0.0
            for step in range(steps):
                optimizer.zero_grad()
                losses = []
                batch_rows = row_part[step * batch_size:(step + 1) * batch_size]
                if len(batch_rows):
                    batch_tensor = torch.from_numpy(batch_rows)
                    logits = model(tensors["Xt"][batch_tensor])
                    raw_bce = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits, tensors["yt"][batch_tensor], reduction="none")
                    losses.append(0.5 * (raw_bce * weights[batch_tensor]).mean())

                batch_pairs = pair_part[step * batch_size:(step + 1) * batch_size]
                if len(batch_pairs):
                    pair_tensor = torch.from_numpy(batch_pairs)
                    positive_indices = tensors["pos"][pair_tensor]
                    negative_indices = tensors["neg"][pair_tensor]
                    difference = (model(tensors["Xt"][positive_indices])
                                  - model(tensors["Xt"][negative_indices]))
                    pair_weights = 0.5 * (weights[positive_indices] + weights[negative_indices])
                    bpr = (torch.nn.functional.softplus(-difference) * pair_weights).mean()
                    losses.append(0.5 * bpr)

                if losses:
                    loss = sum(losses)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    last_loss = float(loss.detach())
                    global_step += 1

            for group in optimizer.param_groups:
                group["lr"] *= config["decay"]

            scores = predict(model, tensors["Xv"])
            metrics = metric_values(
                evaluator, tensors["uv_np"], tensors["yv_np"], scores)
            history.append({
                "stage": "ensemble_member",
                "member": member_index,
                "seed": seed,
                "config": config,
                "epoch": epoch + 1,
                "half": half + 1,
                "step": global_step,
                "train_loss": round(last_loss, 6),
                "lr": optimizer.param_groups[0]["lr"],
                "val_gauc": round(metrics["gauc"], 6),
                "val_ndcg5": round(metrics["ndcg5"], 6),
                "val_primary": round(metrics["primary"], 6),
            })
            if metrics["primary"] > best_primary + 1e-12:
                best_primary = metrics["primary"]
                best_scores = scores.copy()
                best_epoch = epoch + 1
                best_half = half + 1

    return best_scores, {
        "member": member_index,
        "seed": seed,
        "config": config,
        "best_primary": best_primary,
        "best_epoch": best_epoch,
        "best_half": best_half,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    torch.use_deterministic_algorithms(True)

    fast_path = (os.path.exists(os.path.join(args.data_dir, "train.npz")) and
                 os.path.exists(os.path.join(args.data_dir, "val.npz")))
    if fast_path:
        data, evaluator = load_npz(args.data_dir)
    else:
        data, evaluator = load_csv_data(args.data_dir)

    positive_pairs, negative_pairs = make_pairs(data["ut"], data["yt"], args.seed + 91)
    tensors = {
        "Xt": torch.from_numpy(data["Xt"]),
        "yt": torch.from_numpy(data["yt"]),
        "Xv": torch.from_numpy(data["Xv"]),
        "pos": torch.from_numpy(positive_pairs),
        "neg": torch.from_numpy(negative_pairs),
        "dates_np": data["dates"],
        "uv_np": data["uv"],
        "yv_np": data["yv"],
        "total_dim": int(data["field_dims"].sum()),
    }

    smoke = os.environ.get("SMOKE_EPOCHS")
    epochs = args.epochs
    if smoke is not None:
        epochs = min(epochs, int(smoke))

    member_configs = [
        {
            "dropout": 0.20,
            "weight_decay": 0.00030,
            "lr": 0.00095,
            "decay": 0.59,
            "half_life": 3.5,
        },
        {
            "dropout": 0.23,
            "weight_decay": 0.00030,
            "lr": 0.00090,
            "decay": 0.57,
            "half_life": 3.5,
        },
        {
            "dropout": 0.26,
            "weight_decay": 0.00030,
            "lr": 0.00085,
            "decay": 0.55,
            "half_life": 4.0,
        },
    ]

    history = []
    member_records = []
    member_scores = []
    for member_index, config in enumerate(member_configs):
        member_seed = args.seed + member_index
        scores, record = train_member(
            config, epochs, member_seed, tensors, evaluator, history, member_index)
        member_scores.append(scores)
        member_records.append(record)

    ranked_members = [rank_transform(data["uv"], scores) for scores in member_scores]
    ensemble_history = []
    for count in range(1, len(ranked_members) + 1):
        prefix_scores = np.mean(np.stack(ranked_members[:count], axis=0), axis=0)
        prefix_metrics = metric_values(evaluator, data["uv"], data["yv"], prefix_scores)
        ensemble_history.append({
            "members": count,
            "seeds": [args.seed + i for i in range(count)],
            "gauc": prefix_metrics["gauc"],
            "ndcg5": prefix_metrics["ndcg5"],
            "primary": prefix_metrics["primary"],
        })

    final_scores = np.mean(np.stack(ranked_members, axis=0), axis=0)
    final_metrics = metric_values(evaluator, data["uv"], data["yv"], final_scores)

    os.makedirs(args.out_dir, exist_ok=True)
    metrics_output = {
        "gauc": final_metrics["gauc"],
        "ndcg5": final_metrics["ndcg5"],
        "primary": final_metrics["primary"],
        "ensemble_method": "per_user_rank_average",
        "member_count": len(member_configs),
        "members": member_records,
        "ensemble_history": ensemble_history,
        "history": history,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics_output, fh)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w") as fh:
        fh.write("row_id,user_id,video_id,score\n")
        for row_id, score in enumerate(final_scores):
            fh.write("%d,%s,%s,%.8g\n" % (
                row_id, data["uv"][row_id], data["video_v"][row_id], score))


if __name__ == "__main__":
    main()
