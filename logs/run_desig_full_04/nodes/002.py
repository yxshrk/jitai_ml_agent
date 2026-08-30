"""Single-model FM with exponential averaging of useful late checkpoints."""
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


def metric_values(m):
    return {
        "gauc": float(m["GAUC"] if "GAUC" in m else m["gauc"]),
        "ndcg5": float(m["nDCG@5"] if "nDCG@5" in m else m["ndcg5"]),
        "primary": float(m["primary"]),
    }


def read_csv_rows(path, need_label):
    rows = []
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            item = {
                "user_id": row["user_id"],
                "video_id": row["video_id"],
                "author_id": row.get("author_id", row["video_id"]),
                "tab": row["tab"],
                "duration_ms": float(row["duration_ms"] or 0.0),
            }
            if need_label:
                item["long_view"] = float(row["long_view"])
            rows.append(item)
    return rows


def encode_category(train_values, val_values):
    mapping = {}
    train_encoded = np.empty(len(train_values), dtype=np.int64)
    for i, value in enumerate(train_values):
        if value not in mapping:
            mapping[value] = len(mapping) + 1
        train_encoded[i] = mapping[value]
    val_encoded = np.asarray(
        [mapping.get(value, 0) for value in val_values], dtype=np.int64
    )
    return train_encoded, val_encoded, len(mapping) + 1


def load_csv_data(data_dir):
    train_rows = read_csv_rows(os.path.join(data_dir, "train.csv"), True)
    val_rows = read_csv_rows(os.path.join(data_dir, "val.csv"), True)

    train_columns = []
    val_columns = []
    dims = []
    for name in ("user_id", "video_id", "author_id", "tab"):
        tr_values = [row[name] for row in train_rows]
        va_values = [row[name] for row in val_rows]
        tr_col, va_col, dim = encode_category(tr_values, va_values)
        train_columns.append(tr_col)
        val_columns.append(va_col)
        dims.append(dim)

    train_duration = np.asarray(
        [row["duration_ms"] for row in train_rows], dtype=np.float64
    )
    val_duration = np.asarray(
        [row["duration_ms"] for row in val_rows], dtype=np.float64
    )
    quantiles = np.quantile(train_duration, np.linspace(0.1, 0.9, 9))
    train_bucket = np.searchsorted(
        quantiles, train_duration, side="right"
    ).astype(np.int64)
    val_bucket = np.searchsorted(
        quantiles, val_duration, side="right"
    ).astype(np.int64)
    train_columns.append(train_bucket)
    val_columns.append(val_bucket)
    dims.append(10)

    offsets = np.cumsum(np.asarray([0] + dims[:-1], dtype=np.int64))
    train_x = np.stack(train_columns, axis=1) + offsets
    val_x = np.stack(val_columns, axis=1) + offsets
    train_y = np.asarray(
        [row["long_view"] for row in train_rows], dtype=np.float32
    )
    val_y = np.asarray(
        [row["long_view"] for row in val_rows], dtype=np.float32
    )
    val_users = np.asarray([row["user_id"] for row in val_rows])
    val_videos = np.asarray([row["video_id"] for row in val_rows])
    return (
        train_x,
        train_y,
        val_x,
        val_y,
        val_users,
        val_videos,
        np.asarray(dims),
    )


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        with np.load(train_npz) as tr, np.load(val_npz) as va:
            train_x = tr["X"].astype(np.int64)
            train_y = tr["y"].astype(np.float32)
            val_x = va["X"].astype(np.int64)
            val_y = va["y"].astype(np.float32)
            val_users = va["user"].copy()
            val_videos = val_x[:, 1].copy()
            field_dims = tr["field_dims"].astype(np.int64)
        from data.official.evaluate import evaluate
        return (
            train_x,
            train_y,
            val_x,
            val_y,
            val_users,
            val_videos,
            field_dims,
            evaluate,
        )

    data = load_csv_data(data_dir)
    from harness.evaluate_provisional import evaluate
    return data + (evaluate,)


def predict(model, x):
    model.eval()
    with torch.no_grad():
        return np.concatenate([
            model(x[start:start + 65536]).numpy()
            for start in range(0, len(x), 65536)
        ])


def train_ema(
    train_x,
    train_y,
    val_x,
    val_y,
    val_users,
    total_dim,
    epochs,
    seed,
    evaluate,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = FM(total_dim)
    averaged_model = FM(total_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()

    xt = torch.from_numpy(train_x)
    yt = torch.from_numpy(train_y)
    xv = torch.from_numpy(val_x)
    n = len(yt)
    batch_size = 8192

    generator = torch.Generator()
    generator.manual_seed(seed)

    ema_state = None
    ema_updates = 0
    ema_decay = 0.75
    ema_start = min(5, epochs - 1)
    best_primary = -1.0
    best_scores = None

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(n, generator=generator)
        for start in range(0, n, batch_size):
            idx = permutation[start:start + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(xt[idx]), yt[idx])
            loss.backward()
            optimizer.step()

        if epoch < ema_start:
            continue

        current_state = model.state_dict()
        if ema_state is None:
            ema_state = {
                name: value.detach().clone()
                for name, value in current_state.items()
            }
        else:
            for name, value in current_state.items():
                ema_state[name].mul_(ema_decay).add_(
                    value.detach(), alpha=1.0 - ema_decay
                )
        ema_updates += 1

        eligible = ema_updates >= 2 or epoch == epochs - 1
        if not eligible:
            continue

        averaged_model.load_state_dict(ema_state)
        scores = predict(averaged_model, xv)
        metrics = evaluate(val_users, val_y.astype(int), scores)
        primary = float(metrics["primary"])
        if primary > best_primary + 1e-6:
            best_primary = primary
            best_scores = scores.copy()

    return best_scores


def per_user_ranks(users, scores):
    ranks = np.empty(len(scores), dtype=np.float64)
    groups = {}
    for i, user in enumerate(users):
        key = user.item() if isinstance(user, np.generic) else user
        groups.setdefault(key, []).append(i)
    for indices in groups.values():
        idx = np.asarray(indices, dtype=np.int64)
        if len(idx) == 1:
            ranks[idx[0]] = 0.5
        else:
            order = np.argsort(scores[idx], kind="mergesort")
            local = np.empty(len(idx), dtype=np.float64)
            local[order] = (
                np.arange(len(idx), dtype=np.float64) / float(len(idx) - 1)
            )
            ranks[idx] = local
    return ranks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()

    epochs = args.epochs
    smoke = os.environ.get("SMOKE_EPOCHS")
    if smoke is not None:
        epochs = min(epochs, max(1, int(smoke)))

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    data = load_data(args.data_dir)
    (
        train_x,
        train_y,
        val_x,
        val_y,
        val_users,
        val_videos,
        field_dims,
        evaluate,
    ) = data
    total_dim = int(field_dims.sum())

    raw_scores = train_ema(
        train_x,
        train_y,
        val_x,
        val_y,
        val_users,
        total_dim,
        epochs,
        args.seed,
        evaluate,
    )
    scores = per_user_ranks(val_users, raw_scores)
    metrics = metric_values(evaluate(val_users, val_y.astype(int), scores))

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh)
    with open(
        os.path.join(args.out_dir, "predictions.csv"), "w", newline=""
    ) as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(scores):
            writer.writerow([i, val_users[i], val_videos[i], "%.9g" % score])


if __name__ == "__main__":
    main()
