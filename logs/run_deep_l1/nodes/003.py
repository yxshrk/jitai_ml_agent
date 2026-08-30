import argparse
import csv
import json
import math
import os
import random
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_date_value(value):
    text = str(value)
    if text.endswith(".0"):
        text = text[:-2]
    text = text.replace("-", "")
    try:
        return datetime.strptime(text[:8], "%Y%m%d").toordinal()
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return 0


def recency_weights(dates):
    ordinals = np.asarray([parse_date_value(x) for x in dates], dtype=np.int64)
    maximum = int(ordinals.max()) if len(ordinals) else 0
    age = np.maximum(maximum - ordinals, 0)
    weights = np.exp(-math.log(2.0) * age / 7.0)
    weights = weights / max(float(weights.mean()), 1e-8)
    return weights.astype(np.float32)


def load_npz(data_dir):
    train_file = np.load(os.path.join(data_dir, "train.npz"), allow_pickle=False)
    val_file = np.load(os.path.join(data_dir, "val.npz"), allow_pickle=False)
    train = {
        "X": np.asarray(train_file["X"], dtype=np.int64),
        "y": np.asarray(train_file["y"], dtype=np.float32),
        "user": np.asarray(train_file["user"]),
        "play": np.asarray(train_file["play_time_ms"], dtype=np.float32),
        "duration": np.asarray(train_file["duration_ms"], dtype=np.float32),
        "date": np.asarray(train_file["date"]),
    }
    field_dims = np.asarray(train_file["field_dims"], dtype=np.int64)
    val_x = np.asarray(val_file["X"], dtype=np.int64)
    val_user = np.asarray(val_file["user"])
    video_offset = int(field_dims[0])
    val = {
        "X": val_x,
        "y": np.asarray(val_file["y"], dtype=np.float32),
        "user": val_user,
        "video": val_x[:, 1] - video_offset,
    }
    return train, val, field_dims, True


def read_csv_rows(path, training):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        has_author = "author_id" in (reader.fieldnames or [])
        for row in reader:
            item = {
                "user": row["user_id"],
                "video": row["video_id"],
                "author": row["author_id"] if has_author else "__unknown_author__",
                "tab": row["tab"],
                "duration": float(row["duration_ms"]),
                "date": row["date"],
                "y": float(row["long_view"]),
            }
            if training:
                item["play"] = float(row["play_time_ms"])
            rows.append(item)
    return rows


def make_mapping(values):
    mapping = {}
    for value in values:
        if value not in mapping:
            mapping[value] = len(mapping) + 1
    return mapping


def encode_with_unknown(values, mapping):
    return np.asarray([mapping.get(value, 0) for value in values], dtype=np.int64)


def load_csv(data_dir):
    train_rows = read_csv_rows(os.path.join(data_dir, "train.csv"), True)
    val_rows = read_csv_rows(os.path.join(data_dir, "val.csv"), False)
    train_duration = np.asarray([r["duration"] for r in train_rows], dtype=np.float64)
    if len(train_duration):
        boundaries = np.unique(np.quantile(train_duration, np.linspace(0.1, 0.9, 9)))
    else:
        boundaries = np.asarray([], dtype=np.float64)
    fields = ["user", "video", "author", "tab"]
    train_columns = []
    val_columns = []
    field_dims = []
    for field in fields:
        mapping = make_mapping([r[field] for r in train_rows])
        train_columns.append(encode_with_unknown([r[field] for r in train_rows], mapping))
        val_columns.append(encode_with_unknown([r[field] for r in val_rows], mapping))
        field_dims.append(len(mapping) + 1)
    train_bucket = np.searchsorted(boundaries, train_duration, side="right").astype(np.int64)
    val_duration = np.asarray([r["duration"] for r in val_rows], dtype=np.float64)
    val_bucket = np.searchsorted(boundaries, val_duration, side="right").astype(np.int64)
    train_columns.append(train_bucket)
    val_columns.append(val_bucket)
    field_dims.append(int(len(boundaries) + 1))
    offsets = np.cumsum([0] + field_dims[:-1], dtype=np.int64)
    train_x = np.stack(train_columns, axis=1) + offsets[None, :]
    val_x = np.stack(val_columns, axis=1) + offsets[None, :]
    train = {
        "X": train_x.astype(np.int64),
        "y": np.asarray([r["y"] for r in train_rows], dtype=np.float32),
        "user": np.asarray([r["user"] for r in train_rows]),
        "play": np.asarray([r["play"] for r in train_rows], dtype=np.float32),
        "duration": train_duration.astype(np.float32),
        "date": np.asarray([r["date"] for r in train_rows]),
    }
    val = {
        "X": val_x.astype(np.int64),
        "y": np.asarray([r["y"] for r in val_rows], dtype=np.float32),
        "user": np.asarray([r["user"] for r in val_rows]),
        "video": np.asarray([r["video"] for r in val_rows]),
    }
    return train, val, np.asarray(field_dims, dtype=np.int64), False


def build_pair_pool(users, labels):
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    starts = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1]
    ends = np.r_[starts[1:], len(order)]
    positive_parts = []
    negative_parts = []
    for start, end in zip(starts, ends):
        indices = order[start:end]
        positives = indices[labels[indices] > 0.5]
        negatives = indices[labels[indices] <= 0.5]
        if len(positives) and len(negatives):
            positive_parts.append(positives)
            negative_parts.append(negatives)
    if not positive_parts:
        return np.empty(0, dtype=np.int64), []
    return np.concatenate(positive_parts).astype(np.int64), negative_parts


def sample_negative_pool(positive_pool, negative_groups, rng):
    sampled = np.empty(len(positive_pool), dtype=np.int64)
    cursor = 0
    for negatives in negative_groups:
        next_cursor = cursor
        while next_cursor < len(positive_pool):
            next_cursor += 1
            if next_cursor == len(positive_pool):
                break
            if positive_pool[next_cursor] < positive_pool[next_cursor - 1]:
                break
        cursor = next_cursor
    cursor = 0
    for negatives in negative_groups:
        count = 0
        if cursor < len(positive_pool):
            user_marker = None
            while cursor + count < len(positive_pool):
                index = positive_pool[cursor + count]
                marker = index
                if user_marker is None:
                    user_marker = marker
                count += 1
                if count >= len(negatives) * 1000000:
                    break
                if cursor + count >= len(positive_pool):
                    break
                if count >= 1 and len(negative_groups) == 1:
                    continue
                break
        if count == 0:
            continue
        sampled[cursor:cursor + count] = negatives[rng.integers(0, len(negatives), size=count)]
        cursor += count
    if cursor != len(positive_pool):
        all_negatives = np.concatenate(negative_groups)
        sampled[cursor:] = all_negatives[rng.integers(0, len(all_negatives), size=len(positive_pool) - cursor)]
    return sampled


def build_pairs(users, labels, rng):
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    starts = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1]
    ends = np.r_[starts[1:], len(order)]
    positives_out = []
    negatives_out = []
    for start, end in zip(starts, ends):
        indices = order[start:end]
        positives = indices[labels[indices] > 0.5]
        negatives = indices[labels[indices] <= 0.5]
        if len(positives) and len(negatives):
            positives_out.append(positives)
            negatives_out.append(negatives[rng.integers(0, len(negatives), size=len(positives))])
    if not positives_out:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    return np.concatenate(positives_out), np.concatenate(negatives_out)


class DCNCensored(nn.Module):
    def __init__(self, total_categories, fields=5, embedding_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(total_categories, embedding_dim)
        dimension = fields * embedding_dim
        self.cross_weights = nn.ParameterList([nn.Parameter(torch.empty(dimension)) for _ in range(2)])
        self.cross_biases = nn.ParameterList([nn.Parameter(torch.zeros(dimension)) for _ in range(2)])
        self.mlp = nn.Sequential(
            nn.Linear(dimension, 128),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
        )
        self.main_head = nn.Linear(dimension + 64, 1)
        self.watch_head = nn.Linear(dimension + 64, 1)
        nn.init.normal_(self.embedding.weight, std=0.01)
        for weight in self.cross_weights:
            nn.init.normal_(weight, std=0.01)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = F.dropout(embedded, p=0.15, training=self.training)
        x0 = embedded.flatten(1)
        crossed = x0
        for weight, bias in zip(self.cross_weights, self.cross_biases):
            scalar = torch.sum(crossed * weight, dim=1, keepdim=True)
            crossed = x0 * scalar + bias + crossed
        hidden = self.mlp(x0)
        representation = torch.cat([crossed, hidden], dim=1)
        return self.main_head(representation).squeeze(1), self.watch_head(representation).squeeze(1)


def censored_watch_loss(prediction, play_ms, duration_ms):
    play_seconds = torch.clamp(play_ms, min=0.0) / 1000.0
    duration_seconds = torch.clamp(duration_ms, min=0.001) / 1000.0
    observed = torch.log1p(torch.minimum(play_seconds, duration_seconds)) / 4.0
    lower_bound = torch.log1p(duration_seconds) / 4.0
    completed = play_seconds >= duration_seconds
    uncensored = F.smooth_l1_loss(prediction, observed, reduction="none")
    censored = torch.square(F.relu(lower_bound - prediction))
    return torch.where(completed, censored, uncensored)


def metric_values(result):
    gauc = result.get("GAUC", result.get("gauc"))
    ndcg = result.get("nDCG@5", result.get("ndcg5", result.get("NDCG@5")))
    primary = result.get("primary")
    return float(gauc), float(ndcg), float(primary)


def predict(model, x, device, batch_size):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.as_tensor(x[start:start + batch_size], dtype=torch.long, device=device)
            logits, _ = model(batch)
            outputs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(outputs).astype(np.float64) if outputs else np.empty(0, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fast_path = os.path.exists(os.path.join(args.data_dir, "train.npz")) and os.path.exists(os.path.join(args.data_dir, "val.npz"))
    if fast_path:
        train, val, field_dims, used_npz = load_npz(args.data_dir)
        from data.official.evaluate import evaluate
    else:
        train, val, field_dims, used_npz = load_csv(args.data_dir)
        from harness.evaluate_provisional import evaluate
    weights = recency_weights(train["date"])
    rng = np.random.default_rng(args.seed)
    pair_positive, pair_negative = build_pairs(train["user"], train["y"], rng)
    model = DCNCensored(int(field_dims.sum())).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    epochs = 8
    smoke = os.environ.get("SMOKE_EPOCHS")
    if smoke is not None:
        epochs = max(1, min(epochs, int(smoke)))
    batch_size = 8192 if device.type == "cuda" else 4096
    n = len(train["y"])
    best_gauc = -1.0
    best_state = None
    history = []
    stale = 0
    for epoch in range(epochs):
        model.train()
        permutation = rng.permutation(n)
        pair_permutation = rng.permutation(len(pair_positive)) if len(pair_positive) else np.empty(0, dtype=np.int64)
        pair_cursor = 0
        for start in range(0, n, batch_size):
            indices = permutation[start:start + batch_size]
            xb = torch.as_tensor(train["X"][indices], dtype=torch.long, device=device)
            yb = torch.as_tensor(train["y"][indices], dtype=torch.float32, device=device)
            wb = torch.as_tensor(weights[indices], dtype=torch.float32, device=device)
            play = torch.as_tensor(train["play"][indices], dtype=torch.float32, device=device)
            duration = torch.as_tensor(train["duration"][indices], dtype=torch.float32, device=device)
            logits, watch_prediction = model(xb)
            point_loss = (F.binary_cross_entropy_with_logits(logits, yb, reduction="none") * wb).mean()
            watch_loss = (censored_watch_loss(watch_prediction, play, duration) * wb).mean()
            if len(pair_positive):
                wanted = min(len(indices), len(pair_positive))
                if pair_cursor + wanted > len(pair_permutation):
                    pair_permutation = rng.permutation(len(pair_positive))
                    pair_cursor = 0
                selected = pair_permutation[pair_cursor:pair_cursor + wanted]
                pair_cursor += wanted
                pos_x = torch.as_tensor(train["X"][pair_positive[selected]], dtype=torch.long, device=device)
                neg_x = torch.as_tensor(train["X"][pair_negative[selected]], dtype=torch.long, device=device)
                pos_logits, _ = model(pos_x)
                neg_logits, _ = model(neg_x)
                pair_loss = F.softplus(-(pos_logits - neg_logits)).mean()
            else:
                pair_loss = point_loss.new_zeros(())
            loss = 0.5 * point_loss + 0.5 * pair_loss + 0.3 * watch_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        scheduler.step()
        scores = predict(model, val["X"], device, batch_size * 2)
        result = evaluate(val["user"], val["y"], scores)
        gauc, ndcg, primary = metric_values(result)
        history.append({"epoch": epoch + 1, "gauc": gauc, "ndcg5": ndcg, "primary": primary, "lr": float(optimizer.param_groups[0]["lr"])})
        if gauc > best_gauc + 1e-7:
            best_gauc = gauc
            best_state = deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
        if stale >= 2 and epoch + 1 >= 3:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    final_scores = predict(model, val["X"], device, batch_size * 2)
    final_result = evaluate(val["user"], val["y"], final_scores)
    gauc, ndcg, primary = metric_values(final_result)
    predictions_path = os.path.join(args.out_dir, "predictions.csv")
    with open(predictions_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for row_id, (user_id, video_id, score) in enumerate(zip(val["user"], val["video"], final_scores)):
            writer.writerow([row_id, user_id.item() if isinstance(user_id, np.generic) else user_id, video_id.item() if isinstance(video_id, np.generic) else video_id, "%.10f" % float(score)])
    metrics = {
        "gauc": gauc,
        "ndcg5": ndcg,
        "primary": primary,
        "history": history,
        "config": {
            "model": "dcn_lite_censored_watch",
            "embedding_dim": 16,
            "cross_layers": 2,
            "bpr_weight": 0.5,
            "logloss_weight": 0.5,
            "censored_watch_weight": 0.3,
            "recency_half_life_days": 7.0,
            "seed": args.seed
        }
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, sort_keys=True)


if __name__ == "__main__":
    main()
