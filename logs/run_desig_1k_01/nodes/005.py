import argparse
import contextlib
import csv
import datetime
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))


def date_ordinal(value):
    text = str(value)
    if text.endswith(".0"):
        text = text[:-2]
    text = text.replace("-", "")
    if len(text) >= 8 and text[:8].isdigit():
        try:
            return datetime.date(
                int(text[:4]), int(text[4:6]), int(text[6:8])
            ).toordinal()
        except ValueError:
            pass
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def recency_weights(dates):
    ordinals = np.asarray([date_ordinal(x) for x in dates], dtype=np.int64)
    if ordinals.size == 0:
        return np.empty(0, dtype=np.float32)
    latest = int(ordinals.max())
    ages = np.maximum(latest - ordinals, 0)
    weights = np.exp2(-ages.astype(np.float64) / 7.0)
    weights /= max(float(weights.mean()), 1e-12)
    return weights.astype(np.float32)


def load_npz(data_dir):
    with np.load(data_dir / "train.npz", allow_pickle=False) as data:
        train_x = np.asarray(data["X"], dtype=np.int64)
        train_y = np.asarray(data["y"], dtype=np.float32)
        train_users = np.asarray(data["user"])
        train_dates = np.asarray(data["date"])
        field_dims = np.asarray(data["field_dims"], dtype=np.int64)
    with np.load(data_dir / "val.npz", allow_pickle=False) as data:
        val_x = np.asarray(data["X"], dtype=np.int64)
        val_y = np.asarray(data["y"], dtype=np.float32)
        val_users = np.asarray(data["user"])
    video_offset = int(field_dims[0])
    val_videos = val_x[:, 1] - video_offset
    weights = recency_weights(train_dates)
    return (
        train_x,
        train_y,
        train_users,
        weights,
        val_x,
        val_y,
        val_users,
        val_videos,
        field_dims,
        True,
    )


def read_csv_columns(path, training):
    users = []
    videos = []
    tabs = []
    durations = []
    dates = []
    labels = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            users.append(row["user_id"])
            videos.append(row["video_id"])
            tabs.append(row["tab"])
            durations.append(float(row["duration_ms"]))
            labels.append(float(row["long_view"]))
            if training:
                dates.append(row["date"])
    return users, videos, tabs, durations, dates, labels


def make_mapping(values):
    unique = sorted(set(values))
    return {value: index + 1 for index, value in enumerate(unique)}


def load_csv(data_dir):
    tr = read_csv_columns(data_dir / "train.csv", True)
    va = read_csv_columns(data_dir / "val.csv", False)
    user_map = make_mapping(tr[0])
    video_map = make_mapping(tr[1])
    tab_map = make_mapping(tr[2])

    train_duration = np.asarray(tr[3], dtype=np.float64)
    quantiles = np.quantile(train_duration, np.linspace(0.1, 0.9, 9))
    quantiles = np.maximum.accumulate(quantiles)

    field_dims = np.asarray(
        [len(user_map) + 1, len(video_map) + 1, 1, len(tab_map) + 1, 10],
        dtype=np.int64,
    )
    offsets = np.r_[0, np.cumsum(field_dims[:-1])]

    def encode(source):
        n = len(source[0])
        x = np.zeros((n, 5), dtype=np.int64)
        x[:, 0] = np.asarray([user_map.get(v, 0) for v in source[0]])
        x[:, 1] = np.asarray([video_map.get(v, 0) for v in source[1]])
        x[:, 2] = 0
        x[:, 3] = np.asarray([tab_map.get(v, 0) for v in source[2]])
        x[:, 4] = np.searchsorted(
            quantiles, np.asarray(source[3], dtype=np.float64), side="right"
        )
        x += offsets.reshape(1, -1)
        return x

    train_x = encode(tr)
    val_x = encode(va)
    return (
        train_x,
        np.asarray(tr[5], dtype=np.float32),
        np.asarray(tr[0], dtype=str),
        recency_weights(tr[4]),
        val_x,
        np.asarray(va[5], dtype=np.float32),
        np.asarray(va[0], dtype=str),
        np.asarray(va[1], dtype=str),
        field_dims,
        False,
    )


def build_pairs(users, labels, seed):
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    boundaries = np.flatnonzero(
        np.r_[True, sorted_users[1:] != sorted_users[:-1], True]
    )
    rng = np.random.default_rng(seed)
    positive_parts = []
    negative_parts = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        rows = order[left:right]
        positives = rows[labels[rows] > 0.5]
        negatives = rows[labels[rows] <= 0.5]
        if positives.size and negatives.size:
            chosen = negatives[rng.integers(0, negatives.size, size=positives.size)]
            positive_parts.append(positives)
            negative_parts.append(chosen)
    if not positive_parts:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    return np.concatenate(positive_parts), np.concatenate(negative_parts)


class RegularizedDCN(nn.Module):
    def __init__(self, field_dims, embedding_dim=16, dropout=0.30):
        super().__init__()
        width = int(len(field_dims) * embedding_dim)
        self.embedding = nn.Embedding(int(np.sum(field_dims)), embedding_dim)
        self.cross_w1 = nn.Parameter(torch.empty(width))
        self.cross_b1 = nn.Parameter(torch.zeros(width))
        self.cross_w2 = nn.Parameter(torch.empty(width))
        self.cross_b2 = nn.Parameter(torch.zeros(width))
        self.mlp = nn.Sequential(
            nn.Linear(width, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.output = nn.Linear(width + 64, 1)
        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.normal_(self.cross_w1, std=0.01)
        nn.init.normal_(self.cross_w2, std=0.01)

    def forward(self, x, return_embeddings=False):
        embeddings = self.embedding(x)
        base = embeddings.flatten(1)
        crossed = base * torch.sum(base * self.cross_w1, dim=1, keepdim=True)
        crossed = crossed + self.cross_b1 + base
        crossed = base * torch.sum(crossed * self.cross_w2, dim=1, keepdim=True)
        crossed = crossed + self.cross_b2 + crossed
        deep = self.mlp(base)
        logits = self.output(torch.cat([crossed, deep], dim=1)).squeeze(1)
        if return_embeddings:
            return logits, embeddings
        return logits


def predict(model, x, batch_size):
    model.eval()
    result = np.empty(x.shape[0], dtype=np.float32)
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            end = min(start + batch_size, x.shape[0])
            xb = torch.from_numpy(x[start:end])
            result[start:end] = model(xb).cpu().numpy()
    return result


def measured_metrics(users, labels, scores, fast_path):
    if fast_path:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    return evaluate(users, labels, scores)


def train_model(
    train_x,
    train_y,
    train_users,
    train_weights,
    val_x,
    val_y,
    val_users,
    field_dims,
    fast_path,
    seed,
):
    model = RegularizedDCN(field_dims)
    embedding_parameters = list(model.embedding.parameters())
    embedding_ids = {id(parameter) for parameter in embedding_parameters}
    dense_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in embedding_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": embedding_parameters, "weight_decay": 0.0},
            {"params": dense_parameters, "weight_decay": 1e-3},
        ],
        lr=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1, min_lr=2e-5
    )
    bce = nn.BCEWithLogitsLoss(reduction="none")
    pair_positive, pair_negative = build_pairs(train_users, train_y, seed)
    rng = np.random.default_rng(seed)
    batch_size = 4096
    epochs = 15
    smoke = os.environ.get("SMOKE_EPOCHS")
    if smoke is not None:
        try:
            epochs = min(epochs, max(1, int(smoke)))
        except ValueError:
            pass

    best_gauc = -np.inf
    best_state = None
    stale = 0
    for _ in range(epochs):
        model.train()
        point_order = rng.permutation(train_x.shape[0])
        pair_order = rng.permutation(pair_positive.size)
        steps = max(
            (point_order.size + batch_size - 1) // batch_size,
            (pair_order.size + batch_size - 1) // batch_size,
        )
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            total_loss = None

            point_start = step * batch_size
            if point_start < point_order.size:
                point_rows = point_order[point_start : point_start + batch_size]
                xb = torch.from_numpy(train_x[point_rows])
                yb = torch.from_numpy(train_y[point_rows])
                wb = torch.from_numpy(train_weights[point_rows])
                logits, embeddings = model(xb, return_embeddings=True)
                point_loss = (bce(logits, yb) * wb).mean()
                row_l2 = embeddings.square().mean()
                total_loss = 0.5 * point_loss + 1e-4 * row_l2

            pair_start = step * batch_size
            if pair_start < pair_order.size:
                selected = pair_order[pair_start : pair_start + batch_size]
                pos_rows = pair_positive[selected]
                neg_rows = pair_negative[selected]
                pos_x = torch.from_numpy(train_x[pos_rows])
                neg_x = torch.from_numpy(train_x[neg_rows])
                pos_logits, pos_embeddings = model(pos_x, return_embeddings=True)
                neg_logits, neg_embeddings = model(neg_x, return_embeddings=True)
                pair_weight = torch.from_numpy(
                    0.5 * (train_weights[pos_rows] + train_weights[neg_rows])
                )
                pair_loss = (
                    torch.nn.functional.softplus(-(pos_logits - neg_logits)) * pair_weight
                ).mean()
                pair_l2 = 0.5 * (
                    pos_embeddings.square().mean() + neg_embeddings.square().mean()
                )
                pair_term = 0.5 * pair_loss + 1e-4 * pair_l2
                total_loss = pair_term if total_loss is None else total_loss + pair_term

            if total_loss is not None:
                total_loss.backward()
                optimizer.step()

        val_scores = predict(model, val_x, batch_size=8192)
        metrics = measured_metrics(val_users, val_y, val_scores, fast_path)
        gauc = float(metrics["GAUC"])
        scheduler.step(gauc)
        if gauc > best_gauc + 1e-7:
            best_gauc = gauc
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= 4:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args, _ = parser.parse_known_args()

    seed_everything(args.seed)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if (data_dir / "train.npz").exists() and (data_dir / "val.npz").exists():
        loaded = load_npz(data_dir)
    else:
        loaded = load_csv(data_dir)
    (
        train_x,
        train_y,
        train_users,
        train_weights,
        val_x,
        val_y,
        val_users,
        val_videos,
        field_dims,
        fast_path,
    ) = loaded

    model = train_model(
        train_x,
        train_y,
        train_users,
        train_weights,
        val_x,
        val_y,
        val_users,
        field_dims,
        fast_path,
        args.seed,
    )
    scores = predict(model, val_x, batch_size=8192).astype(np.float64)

    with (out_dir / "predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for row_id, (user, video, score) in enumerate(
            zip(val_users, val_videos, scores)
        ):
            writer.writerow([row_id, user, video, "%.10g" % score])

    metrics = measured_metrics(val_users, val_y, scores, fast_path)
    output = {
        "gauc": float(metrics["GAUC"]),
        "ndcg5": float(metrics["nDCG@5"]),
        "primary": float(metrics["primary"]),
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(output, handle, separators=(",", ":"))


if __name__ == "__main__":
    with open(os.devnull, "w") as sink:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            main()
