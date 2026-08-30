import argparse
import copy
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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


def load_npz(data_dir):
    train = np.load(Path(data_dir) / "train.npz", allow_pickle=False)
    val = np.load(Path(data_dir) / "val.npz", allow_pickle=False)
    x_train = np.asarray(train["X"], dtype=np.int64)
    y_train = np.asarray(train["y"], dtype=np.float32)
    users_train = np.asarray(train["user"])
    play_train = np.asarray(train["play_time_ms"], dtype=np.float32)
    duration_train = np.asarray(train["duration_ms"], dtype=np.float32)
    field_dims = np.asarray(train["field_dims"], dtype=np.int64)
    x_val = np.asarray(val["X"], dtype=np.int64)
    y_val = np.asarray(val["y"], dtype=np.float32)
    users_val = np.asarray(val["user"])
    video_offset = int(field_dims[0])
    videos_val = x_val[:, 1] - video_offset
    return x_train, y_train, users_train, play_train, duration_train, x_val, y_val, users_val, videos_val, field_dims, True


def read_csv_rows(path, training):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = {
                "user_id": row["user_id"],
                "video_id": row["video_id"],
                "author_id": row.get("author_id", "__missing_author__"),
                "tab": row["tab"],
                "duration_ms": float(row["duration_ms"] or 0.0),
                "long_view": float(row["long_view"] or 0.0),
            }
            if training:
                item["play_time_ms"] = float(row["play_time_ms"] or 0.0)
            rows.append(item)
    return rows


def make_mapping(values):
    unique = sorted(set(values))
    return {value: i + 1 for i, value in enumerate(unique)}


def load_csv(data_dir):
    train_rows = read_csv_rows(Path(data_dir) / "train.csv", True)
    val_rows = read_csv_rows(Path(data_dir) / "val.csv", False)
    user_map = make_mapping([r["user_id"] for r in train_rows])
    video_map = make_mapping([r["video_id"] for r in train_rows])
    author_map = make_mapping([r["author_id"] for r in train_rows])
    tab_map = make_mapping([r["tab"] for r in train_rows])
    train_duration = np.asarray([r["duration_ms"] for r in train_rows], dtype=np.float64)
    duration_edges = np.quantile(train_duration, np.arange(1, 10) / 10.0)
    field_dims = np.asarray([
        len(user_map) + 1,
        len(video_map) + 1,
        len(author_map) + 1,
        len(tab_map) + 1,
        10,
    ], dtype=np.int64)
    offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(field_dims)[:-1]))

    def encode(rows):
        x = np.empty((len(rows), 5), dtype=np.int64)
        for i, row in enumerate(rows):
            x[i, 0] = user_map.get(row["user_id"], 0) + offsets[0]
            x[i, 1] = video_map.get(row["video_id"], 0) + offsets[1]
            x[i, 2] = author_map.get(row["author_id"], 0) + offsets[2]
            x[i, 3] = tab_map.get(row["tab"], 0) + offsets[3]
            x[i, 4] = int(np.searchsorted(duration_edges, row["duration_ms"], side="right")) + offsets[4]
        return x

    x_train = encode(train_rows)
    x_val = encode(val_rows)
    y_train = np.asarray([r["long_view"] for r in train_rows], dtype=np.float32)
    y_val = np.asarray([r["long_view"] for r in val_rows], dtype=np.float32)
    users_train = np.asarray([r["user_id"] for r in train_rows])
    users_val = np.asarray([r["user_id"] for r in val_rows])
    videos_val = np.asarray([r["video_id"] for r in val_rows])
    play_train = np.asarray([r["play_time_ms"] for r in train_rows], dtype=np.float32)
    duration_train = np.asarray([r["duration_ms"] for r in train_rows], dtype=np.float32)
    return x_train, y_train, users_train, play_train, duration_train, x_val, y_val, users_val, videos_val, field_dims, False


def make_pairs(users, labels, seed):
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    rng = np.random.default_rng(seed)
    positive_parts = []
    negative_parts = []
    for j in range(len(boundaries) - 1):
        idx = order[boundaries[j]:boundaries[j + 1]]
        pos = idx[labels[idx] > 0.5]
        neg = idx[labels[idx] <= 0.5]
        if len(pos) == 0 or len(neg) == 0:
            continue
        count = max(len(pos), len(neg))
        positive_parts.append(rng.choice(pos, size=count, replace=len(pos) < count))
        negative_parts.append(rng.choice(neg, size=count, replace=len(neg) < count))
    if not positive_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(positive_parts).astype(np.int64), np.concatenate(negative_parts).astype(np.int64)


class DCNLite(nn.Module):
    def __init__(self, field_dims, embedding_dim=16, hidden_dim=128, dropout=0.3):
        super().__init__()
        total_dim = int(np.sum(field_dims))
        input_dim = len(field_dims) * embedding_dim
        self.embedding = nn.Embedding(total_dim, embedding_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        self.cross_weight = nn.Parameter(torch.empty(input_dim))
        self.cross_bias = nn.Parameter(torch.zeros(input_dim))
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        representation_dim = input_dim + hidden_dim
        self.main_head = nn.Linear(representation_dim, 1)
        self.watch_head = nn.Linear(representation_dim, 1)
        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.normal_(self.cross_weight, std=0.01)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x0 = self.embedding_dropout(self.embedding(x)).flatten(1)
        cross_scalar = torch.sum(x0 * self.cross_weight, dim=1, keepdim=True)
        cross = x0 + x0 * cross_scalar + self.cross_bias
        deep = self.mlp(x0)
        representation = torch.cat((cross, deep), dim=1)
        logits = self.main_head(representation).squeeze(1)
        watch = self.watch_head(representation).squeeze(1)
        return logits, watch


def predict(model, x, device, batch_size=16384):
    model.eval()
    result = np.empty(len(x), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            xb = torch.as_tensor(x[start:end], dtype=torch.long, device=device)
            logits, _ = model(xb)
            result[start:end] = torch.sigmoid(logits).cpu().numpy()
    return result


def get_evaluator(fast_path):
    if fast_path:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    return evaluate


def train_model(x_train, y_train, users_train, play_train, duration_train, x_val, y_val, users_val, field_dims, seed, evaluate_fn):
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DCNLite(field_dims).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    epochs = 8
    smoke = os.environ.get("SMOKE_EPOCHS")
    if smoke is not None:
        epochs = min(epochs, max(1, int(smoke)))
    batch_size = 4096
    pair_pos, pair_neg = make_pairs(users_train, y_train, seed + 991)
    rng = np.random.default_rng(seed + 17)
    best_state = None
    best_gauc = -1.0
    stale = 0
    n = len(x_train)
    for epoch in range(epochs):
        model.train()
        point_order = rng.permutation(n)
        pair_order = rng.permutation(len(pair_pos)) if len(pair_pos) else np.empty(0, dtype=np.int64)
        point_batches = (n + batch_size - 1) // batch_size
        pair_batches = max(1, (len(pair_pos) + batch_size - 1) // batch_size)
        for step in range(point_batches):
            point_idx = point_order[step * batch_size:min((step + 1) * batch_size, n)]
            xb = torch.as_tensor(x_train[point_idx], dtype=torch.long, device=device)
            yb = torch.as_tensor(y_train[point_idx], dtype=torch.float32, device=device)
            observed_ms = np.minimum(np.maximum(play_train[point_idx], 0.0), np.maximum(duration_train[point_idx], 0.0))
            watch_target = np.log1p(observed_ms / 1000.0).astype(np.float32)
            completed = ((duration_train[point_idx] > 0.0) & (play_train[point_idx] >= duration_train[point_idx])).astype(np.float32)
            watch_target_t = torch.as_tensor(watch_target, dtype=torch.float32, device=device)
            completed_t = torch.as_tensor(completed, dtype=torch.float32, device=device)
            logits, watch_prediction = model(xb)
            bce = F.binary_cross_entropy_with_logits(logits, yb)
            exact_loss = F.smooth_l1_loss(watch_prediction, watch_target_t, reduction="none")
            censored_loss = torch.square(F.relu(watch_target_t - watch_prediction))
            watch_loss = torch.mean((1.0 - completed_t) * exact_loss + completed_t * censored_loss)
            if len(pair_pos):
                pair_batch = step % pair_batches
                selected = pair_order[pair_batch * batch_size:min((pair_batch + 1) * batch_size, len(pair_order))]
                if len(selected) == 0:
                    selected = pair_order[:min(batch_size, len(pair_order))]
                pos_x = torch.as_tensor(x_train[pair_pos[selected]], dtype=torch.long, device=device)
                neg_x = torch.as_tensor(x_train[pair_neg[selected]], dtype=torch.long, device=device)
                pos_logits, _ = model(pos_x)
                neg_logits, _ = model(neg_x)
                bpr = -F.logsigmoid(pos_logits - neg_logits).mean()
            else:
                bpr = torch.zeros((), device=device)
            loss = 0.5 * bce + 0.5 * bpr + 0.15 * watch_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        val_scores = predict(model, x_val, device)
        epoch_metrics = evaluate_fn(users_val, y_val, val_scores)
        gauc = float(epoch_metrics["GAUC"])
        if gauc > best_gauc + 1e-7:
            best_gauc = gauc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        scheduler.step()
        if stale >= 2:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return predict(model, x_val, device)


def write_outputs(out_dir, users, videos, scores, metrics):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, (user, video, score) in enumerate(zip(users, videos, scores)):
            if isinstance(user, np.generic):
                user = user.item()
            if isinstance(video, np.generic):
                video = video.item()
            writer.writerow([i, user, video, float(score)])
    payload = {
        "gauc": float(metrics["GAUC"]),
        "ndcg5": float(metrics["nDCG@5"]),
        "primary": float(metrics["primary"]),
    }
    with open(out_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed_everything(args.seed)
    fast_path = (Path(args.data_dir) / "train.npz").exists() and (Path(args.data_dir) / "val.npz").exists()
    if fast_path:
        data = load_npz(args.data_dir)
    else:
        data = load_csv(args.data_dir)
    x_train, y_train, users_train, play_train, duration_train, x_val, y_val, users_val, videos_val, field_dims, loaded_fast = data
    evaluate_fn = get_evaluator(loaded_fast)
    scores = train_model(
        x_train, y_train, users_train, play_train, duration_train,
        x_val, y_val, users_val, field_dims, args.seed, evaluate_fn
    )
    metrics = evaluate_fn(users_val, y_val, scores)
    write_outputs(args.out_dir, users_val, videos_val, scores, metrics)


if __name__ == "__main__":
    main()
