import argparse
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def read_csv_rows(path):
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def make_mapping(values):
    unique = sorted(set(values))
    return {value: i + 1 for i, value in enumerate(unique)}


def encode_with_mapping(values, mapping):
    return np.asarray([mapping.get(value, 0) for value in values], dtype=np.int64)


def load_csv_data(data_dir):
    train_rows = read_csv_rows(Path(data_dir) / "train.csv")
    val_rows = read_csv_rows(Path(data_dir) / "val.csv")

    train_users_raw = [row["user_id"] for row in train_rows]
    val_users_raw = [row["user_id"] for row in val_rows]
    train_videos_raw = [row["video_id"] for row in train_rows]
    val_videos_raw = [row["video_id"] for row in val_rows]

    if train_rows and "author_id" in train_rows[0]:
        train_authors_raw = [row["author_id"] for row in train_rows]
        val_authors_raw = [row.get("author_id", "") for row in val_rows]
    else:
        train_authors_raw = ["__unknown_author__"] * len(train_rows)
        val_authors_raw = ["__unknown_author__"] * len(val_rows)

    train_tabs_raw = [row["tab"] for row in train_rows]
    val_tabs_raw = [row["tab"] for row in val_rows]

    user_map = make_mapping(train_users_raw)
    video_map = make_mapping(train_videos_raw)
    author_map = make_mapping(train_authors_raw)
    tab_map = make_mapping(train_tabs_raw)

    train_duration = np.asarray([float(row["duration_ms"]) for row in train_rows], dtype=np.float64)
    val_duration = np.asarray([float(row["duration_ms"]) for row in val_rows], dtype=np.float64)
    quantiles = np.quantile(train_duration, np.linspace(0.1, 0.9, 9))
    quantiles = np.maximum.accumulate(quantiles)
    train_bucket = np.searchsorted(quantiles, train_duration, side="right").astype(np.int64) + 1
    val_bucket = np.searchsorted(quantiles, val_duration, side="right").astype(np.int64) + 1

    train_columns = [
        encode_with_mapping(train_users_raw, user_map),
        encode_with_mapping(train_videos_raw, video_map),
        encode_with_mapping(train_authors_raw, author_map),
        encode_with_mapping(train_tabs_raw, tab_map),
        train_bucket,
    ]
    val_columns = [
        encode_with_mapping(val_users_raw, user_map),
        encode_with_mapping(val_videos_raw, video_map),
        encode_with_mapping(val_authors_raw, author_map),
        encode_with_mapping(val_tabs_raw, tab_map),
        val_bucket,
    ]

    field_dims = np.asarray([
        len(user_map) + 1,
        len(video_map) + 1,
        len(author_map) + 1,
        len(tab_map) + 1,
        11,
    ], dtype=np.int64)
    offsets = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(field_dims)[:-1]))
    train_x = np.stack(train_columns, axis=1) + offsets[None, :]
    val_x = np.stack(val_columns, axis=1) + offsets[None, :]

    train_y = np.asarray([float(row["long_view"]) for row in train_rows], dtype=np.float32)
    val_y = np.asarray([float(row["long_view"]) for row in val_rows], dtype=np.float32)
    train_user = np.asarray(train_users_raw)
    val_user = np.asarray(val_users_raw)
    val_video = np.asarray(val_videos_raw)
    row_ids = np.arange(len(val_rows), dtype=np.int64)

    return {
        "train_x": train_x.astype(np.int64),
        "train_y": train_y,
        "train_user": train_user,
        "val_x": val_x.astype(np.int64),
        "val_y": val_y,
        "val_user": val_user,
        "val_video": val_video,
        "row_ids": row_ids,
        "field_dims": field_dims,
        "fast": False,
    }


def load_npz_data(data_dir):
    train_npz = np.load(Path(data_dir) / "train.npz", allow_pickle=False)
    val_npz = np.load(Path(data_dir) / "val.npz", allow_pickle=False)
    train_x = np.asarray(train_npz["X"], dtype=np.int64)
    val_x = np.asarray(val_npz["X"], dtype=np.int64)
    train_user = np.asarray(train_npz["user"])
    val_user = np.asarray(val_npz["user"])
    field_dims = np.asarray(train_npz["field_dims"], dtype=np.int64)
    offsets = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(field_dims)[:-1]))
    val_video = val_x[:, 1] - offsets[1]
    return {
        "train_x": train_x,
        "train_y": np.asarray(train_npz["y"], dtype=np.float32),
        "train_user": train_user,
        "val_x": val_x,
        "val_y": np.asarray(val_npz["y"], dtype=np.float32),
        "val_user": val_user,
        "val_video": val_video,
        "row_ids": np.arange(len(val_x), dtype=np.int64),
        "field_dims": field_dims,
        "fast": True,
    }


def load_data(data_dir):
    if (Path(data_dir) / "train.npz").exists() and (Path(data_dir) / "val.npz").exists():
        return load_npz_data(data_dir)
    return load_csv_data(data_dir)


def build_pairs(users, labels):
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    positive_parts = []
    negative_parts = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        group_indices = order[left:right]
        positives = group_indices[labels[group_indices] > 0.5]
        negatives = group_indices[labels[group_indices] <= 0.5]
        if len(positives) == 0 or len(negatives) == 0:
            continue
        count = max(len(positives), len(negatives))
        positive_parts.append(np.resize(positives, count))
        negative_parts.append(np.resize(negatives, count))
    if not positive_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(positive_parts), np.concatenate(negative_parts)


class DCNLite(nn.Module):
    def __init__(self, field_dims, embedding_dim=16, hidden_dim=128, dropout=0.2):
        super().__init__()
        total_dim = int(np.sum(field_dims))
        input_dim = len(field_dims) * embedding_dim
        self.embedding = nn.Embedding(total_dim, embedding_dim)
        self.linear = nn.Embedding(total_dim, 1)
        self.cross_weight = nn.Parameter(torch.empty(input_dim))
        self.cross_bias = nn.Parameter(torch.zeros(input_dim))
        self.cross_output = nn.Linear(input_dim, 1, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.cross_weight, std=0.01)
        nn.init.zeros_(self.cross_output.weight)

    def forward(self, x):
        embedded = self.embedding(x).flatten(1)
        cross_scale = torch.sum(embedded * self.cross_weight, dim=1, keepdim=True)
        crossed = embedded + embedded * cross_scale + self.cross_bias
        first_order = self.linear(x).sum(dim=1).squeeze(1)
        return first_order + self.cross_output(crossed).squeeze(1) + self.mlp(embedded).squeeze(1)


def predict(model, x, device, batch_size):
    model.eval()
    outputs = np.empty(len(x), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            xb = torch.from_numpy(x[start:end]).to(device=device, dtype=torch.long)
            outputs[start:end] = torch.sigmoid(model(xb)).cpu().numpy()
    return outputs


def train_member(data, member_seed, epochs, device):
    seed_everything(member_seed)
    model = DCNLite(data["field_dims"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    x = data["train_x"]
    y = data["train_y"]
    pair_pos, pair_neg = build_pairs(data["train_user"], y)
    batch_size = 16384 if device.type == "cuda" else 8192
    pair_batch_size = 2048
    rng = np.random.default_rng(member_seed)

    for _ in range(epochs):
        model.train()
        permutation = rng.permutation(len(x))
        for start in range(0, len(x), batch_size):
            batch_indices = permutation[start:start + batch_size]
            xb = torch.from_numpy(x[batch_indices]).to(device=device, dtype=torch.long)
            yb = torch.from_numpy(y[batch_indices]).to(device=device, dtype=torch.float32)
            point_loss = F.binary_cross_entropy_with_logits(model(xb), yb)

            if len(pair_pos) > 0:
                selected = rng.integers(0, len(pair_pos), size=min(pair_batch_size, len(pair_pos)))
                pos_x = torch.from_numpy(x[pair_pos[selected]]).to(device=device, dtype=torch.long)
                neg_x = torch.from_numpy(x[pair_neg[selected]]).to(device=device, dtype=torch.long)
                pair_loss = F.softplus(-(model(pos_x) - model(neg_x))).mean()
                loss = 0.5 * point_loss + 0.5 * pair_loss
            else:
                loss = point_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scheduler.step()

    scores = predict(model, data["val_x"], device, batch_size)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return scores


def within_user_ranks(users, scores):
    order = np.lexsort((scores, users))
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    ranked = np.empty(len(scores), dtype=np.float64)
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        size = right - left
        if size == 1:
            ranked[order[left]] = 0.5
        else:
            ranked[order[left:right]] = np.arange(size, dtype=np.float64) / float(size - 1)
    return ranked


def write_predictions(path, row_ids, users, videos, scores):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for row_id, user, video, score in zip(row_ids, users, videos, scores):
            writer.writerow([row_id, user, video, format(float(score), ".10f")])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seed_everything(args.seed)
    data = load_data(args.data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 8
    smoke_epochs = os.environ.get("SMOKE_EPOCHS")
    if smoke_epochs is not None:
        epochs = min(epochs, max(1, int(smoke_epochs)))

    member_seeds = [args.seed + offset for offset in (0, 101, 211, 307, 419)]
    rank_sum = np.zeros(len(data["val_x"]), dtype=np.float64)
    for member_seed in member_seeds:
        member_scores = train_member(data, member_seed, epochs, device)
        rank_sum += within_user_ranks(data["val_user"], member_scores)
    ensemble_scores = rank_sum / float(len(member_seeds))

    predictions_path = Path(args.out_dir) / "predictions.csv"
    metrics_path = Path(args.out_dir) / "metrics.json"
    write_predictions(
        predictions_path,
        data["row_ids"],
        data["val_user"],
        data["val_video"],
        ensemble_scores,
    )

    if data["fast"]:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    result = evaluate(data["val_user"], data["val_y"], ensemble_scores)
    metrics = {
        "gauc": float(result["GAUC"]),
        "ndcg5": float(result["nDCG@5"]),
        "primary": float(result["primary"]),
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    main()
