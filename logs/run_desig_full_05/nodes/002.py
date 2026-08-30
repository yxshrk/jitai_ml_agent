import argparse
import csv
import json
import math
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def date_ordinals(values):
    values = np.asarray(values)
    result = np.zeros(len(values), dtype=np.float32)
    cache = {}
    for value in np.unique(values):
        text = str(int(value)) if np.issubdtype(values.dtype, np.number) else str(value)
        text = text.replace("-", "")[:8]
        try:
            ordinal = datetime.strptime(text, "%Y%m%d").toordinal()
        except Exception:
            ordinal = 0
        cache[value.item() if hasattr(value, "item") else value] = ordinal
    for i, value in enumerate(values):
        key = value.item() if hasattr(value, "item") else value
        result[i] = cache[key]
    return result


def normalize_field_dims(raw_dims, x_train, x_val):
    dims = np.asarray(raw_dims).reshape(-1).astype(np.int64)
    if len(dims) != x_train.shape[1]:
        dims = np.empty(x_train.shape[1], dtype=np.int64)
        for j in range(x_train.shape[1]):
            dims[j] = max(int(x_train[:, j].max(initial=0)), int(x_val[:, j].max(initial=0))) + 1
        return dims
    total = int(dims.sum())
    max_value = max(int(x_train.max(initial=0)), int(x_val.max(initial=0)))
    if max_value >= total:
        dims = np.empty(x_train.shape[1], dtype=np.int64)
        for j in range(x_train.shape[1]):
            dims[j] = max(int(x_train[:, j].max(initial=0)), int(x_val[:, j].max(initial=0))) + 1
    return dims


def ensure_offsets(x_train, x_val, dims):
    x_train = np.asarray(x_train, dtype=np.int64).copy()
    x_val = np.asarray(x_val, dtype=np.int64).copy()
    offsets = np.concatenate(([0], np.cumsum(dims[:-1]))).astype(np.int64)
    already_offset = True
    for j in range(x_train.shape[1]):
        lo = int(offsets[j])
        hi = lo + int(dims[j])
        train_ok = bool(np.all((x_train[:, j] >= lo) & (x_train[:, j] < hi)))
        val_ok = bool(np.all((x_val[:, j] >= lo) & (x_val[:, j] < hi)))
        if not train_ok or not val_ok:
            already_offset = False
            break
    if not already_offset:
        for j in range(x_train.shape[1]):
            x_train[:, j] = np.clip(x_train[:, j], 0, dims[j] - 1) + offsets[j]
            x_val[:, j] = np.clip(x_val[:, j], 0, dims[j] - 1) + offsets[j]
    return x_train, x_val


def load_npz(data_dir):
    train_path = os.path.join(data_dir, "train.npz")
    val_path = os.path.join(data_dir, "val.npz")
    with np.load(train_path, allow_pickle=False) as tr, np.load(val_path, allow_pickle=False) as va:
        x_train = np.asarray(tr["X"], dtype=np.int64)
        x_val = np.asarray(va["X"], dtype=np.int64)
        y_train = np.asarray(tr["y"], dtype=np.float32)
        y_val = np.asarray(va["y"], dtype=np.float32)
        train_user = np.asarray(tr["user"])
        val_user = np.asarray(va["user"])
        play = np.asarray(tr["play_time_ms"], dtype=np.float32)
        duration = np.asarray(tr["duration_ms"], dtype=np.float32)
        dates = np.asarray(tr["date"])
        raw_dims = np.asarray(tr["field_dims"] if "field_dims" in tr.files else va["field_dims"])
        if "video" in va.files:
            val_video = np.asarray(va["video"])
        else:
            first_offset = int(np.asarray(raw_dims).reshape(-1)[0]) if len(np.asarray(raw_dims).reshape(-1)) else 0
            val_video = x_val[:, 1].astype(np.int64) - first_offset
    dims = normalize_field_dims(raw_dims, x_train, x_val)
    x_train, x_val = ensure_offsets(x_train, x_val, dims)
    return x_train, y_train, train_user, play, duration, dates, x_val, y_val, val_user, val_video, dims, True


def read_csv_rows(path, training):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            item = {
                "user": row["user_id"],
                "video": row["video_id"],
                "author": row.get("author_id", "__unknown_author__"),
                "tab": row.get("tab", "0"),
                "duration": float(row.get("duration_ms", 0) or 0),
                "label": float(row["long_view"]),
            }
            if training:
                item["play"] = float(row.get("play_time_ms", 0) or 0)
                item["date"] = row.get("date", "0")
            rows.append(item)
    return rows


def make_mapping(values):
    unique = sorted(set(values))
    return {value: i + 1 for i, value in enumerate(unique)}


def load_csv(data_dir):
    train_rows = read_csv_rows(os.path.join(data_dir, "train.csv"), True)
    val_rows = read_csv_rows(os.path.join(data_dir, "val.csv"), False)
    user_map = make_mapping([r["user"] for r in train_rows])
    video_map = make_mapping([r["video"] for r in train_rows])
    author_map = make_mapping([r["author"] for r in train_rows])
    tab_map = make_mapping([r["tab"] for r in train_rows])
    train_duration = np.asarray([r["duration"] for r in train_rows], dtype=np.float32)
    if len(train_duration):
        boundaries = np.unique(np.quantile(train_duration, np.linspace(0.1, 0.9, 9)))
    else:
        boundaries = np.asarray([], dtype=np.float32)

    dims = np.asarray([len(user_map) + 1, len(video_map) + 1, len(author_map) + 1, len(tab_map) + 1, 11], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(dims[:-1]))).astype(np.int64)

    def encode(rows):
        x = np.zeros((len(rows), 5), dtype=np.int64)
        for i, row in enumerate(rows):
            x[i, 0] = user_map.get(row["user"], 0) + offsets[0]
            x[i, 1] = video_map.get(row["video"], 0) + offsets[1]
            x[i, 2] = author_map.get(row["author"], 0) + offsets[2]
            x[i, 3] = tab_map.get(row["tab"], 0) + offsets[3]
            x[i, 4] = int(np.digitize(row["duration"], boundaries, right=False)) + 1 + offsets[4]
        return x

    x_train = encode(train_rows)
    x_val = encode(val_rows)
    y_train = np.asarray([r["label"] for r in train_rows], dtype=np.float32)
    y_val = np.asarray([r["label"] for r in val_rows], dtype=np.float32)
    train_user = np.asarray([r["user"] for r in train_rows])
    val_user = np.asarray([r["user"] for r in val_rows])
    val_video = np.asarray([r["video"] for r in val_rows])
    play = np.asarray([r["play"] for r in train_rows], dtype=np.float32)
    dates = np.asarray([r["date"] for r in train_rows])
    return x_train, y_train, train_user, play, train_duration, dates, x_val, y_val, val_user, val_video, dims, False


def build_pairs(users, labels, seed):
    users = np.asarray(users)
    labels = np.asarray(labels)
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    rng = np.random.default_rng(seed)
    positives = []
    negatives = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        group = order[start:end]
        pos = group[labels[group] > 0.5]
        neg = group[labels[group] <= 0.5]
        if len(pos) == 0 or len(neg) == 0:
            continue
        count = max(len(pos), len(neg))
        positives.append(rng.choice(pos, size=count, replace=len(pos) < count))
        negatives.append(rng.choice(neg, size=count, replace=len(neg) < count))
    if not positives:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(positives).astype(np.int64), np.concatenate(negatives).astype(np.int64)


class DCNLite(nn.Module):
    def __init__(self, total_categories, fields=5, embed_dim=16, hidden=128, dropout=0.25):
        super().__init__()
        self.embedding = nn.Embedding(total_categories, embed_dim)
        width = fields * embed_dim
        self.cross_weight = nn.Parameter(torch.empty(width))
        self.cross_bias = nn.Parameter(torch.zeros(width))
        self.mlp = nn.Sequential(
            nn.Linear(width, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.main_head = nn.Linear(width + hidden // 2, 1)
        self.watch_head = nn.Linear(width + hidden // 2, 1)
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.cross_weight, std=0.01)

    def forward(self, x):
        base = self.embedding(x).flatten(1)
        crossed = base + base * torch.sum(base * self.cross_weight, dim=1, keepdim=True) + self.cross_bias
        deep = self.mlp(base)
        shared = torch.cat([crossed, deep], dim=1)
        return self.main_head(shared).squeeze(1), self.watch_head(shared).squeeze(1)


def evaluate_scores(user_ids, labels, scores, fast_path):
    if fast_path:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    raw = evaluate(user_ids, labels, scores)
    gauc = raw.get("GAUC", raw.get("gauc"))
    ndcg = raw.get("nDCG@5", raw.get("ndcg5"))
    primary = raw.get("primary")
    return {"gauc": float(gauc), "ndcg5": float(ndcg), "primary": float(primary)}


def predict(model, x, device, batch_size):
    model.eval()
    output = np.empty(len(x), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            xb = torch.as_tensor(x[start:end], dtype=torch.long, device=device)
            logits, _ = model(xb)
            output[start:end] = logits.detach().cpu().numpy().astype(np.float32)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    fast_path = os.path.exists(os.path.join(args.data_dir, "train.npz")) and os.path.exists(os.path.join(args.data_dir, "val.npz"))
    if fast_path:
        data = load_npz(args.data_dir)
    else:
        data = load_csv(args.data_dir)
    x_train, y_train, train_user, play, duration, dates, x_val, y_val, val_user, val_video, dims, fast_path = data

    duration_safe = np.maximum(duration.astype(np.float32), 1.0)
    threshold = np.minimum(duration_safe, 18000.0)
    watch_target = np.clip(play.astype(np.float32) / np.maximum(threshold, 1.0), 0.0, 1.0)
    completed = play.astype(np.float32) >= duration_safe

    ordinals = date_ordinals(dates)
    if len(ordinals) and np.max(ordinals) > 0:
        recency = np.exp(-math.log(2.0) * (np.max(ordinals) - ordinals) / 7.0).astype(np.float32)
        recency /= max(float(recency.mean()), 1e-6)
    else:
        recency = np.ones(len(y_train), dtype=np.float32)

    pair_pos, pair_neg = build_pairs(train_user, y_train, args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DCNLite(int(np.sum(dims))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.75)

    epochs = 8
    smoke = os.environ.get("SMOKE_EPOCHS")
    if smoke is not None:
        epochs = min(epochs, max(1, int(smoke)))
    batch_size = 8192 if device.type == "cuda" else 4096
    rng = np.random.default_rng(args.seed)
    best_gauc = -float("inf")
    best_state = None
    stale = 0

    for epoch in range(epochs):
        model.train()
        impression_order = rng.permutation(len(y_train))
        if len(pair_pos):
            pair_order = rng.permutation(len(pair_pos))
        else:
            pair_order = np.empty(0, dtype=np.int64)
        pair_cursor = 0

        for start in range(0, len(impression_order), batch_size):
            idx = impression_order[start:start + batch_size]
            xb = torch.as_tensor(x_train[idx], dtype=torch.long, device=device)
            yb = torch.as_tensor(y_train[idx], dtype=torch.float32, device=device)
            wb = torch.as_tensor(recency[idx], dtype=torch.float32, device=device)
            target_watch = torch.as_tensor(watch_target[idx], dtype=torch.float32, device=device)
            censor = torch.as_tensor(completed[idx], dtype=torch.bool, device=device)

            logits, watch_logits = model(xb)
            point_loss = F.binary_cross_entropy_with_logits(logits, yb, reduction="none")
            point_loss = torch.sum(point_loss * wb) / torch.clamp(torch.sum(wb), min=1e-6)

            watch_pred = torch.sigmoid(watch_logits)
            exact_error = (watch_pred - target_watch) ** 2
            censored_error = F.relu(target_watch - watch_pred) ** 2
            watch_error = torch.where(censor, censored_error, exact_error)
            watch_loss = torch.sum(watch_error * wb) / torch.clamp(torch.sum(wb), min=1e-6)

            if len(pair_order):
                need = len(idx)
                if pair_cursor + need <= len(pair_order):
                    chosen = pair_order[pair_cursor:pair_cursor + need]
                    pair_cursor += need
                else:
                    first = pair_order[pair_cursor:]
                    pair_order = rng.permutation(len(pair_pos))
                    remaining = need - len(first)
                    chosen = np.concatenate([first, pair_order[:remaining]])
                    pair_cursor = remaining
                pos_idx = pair_pos[chosen]
                neg_idx = pair_neg[chosen]
                pos_x = torch.as_tensor(x_train[pos_idx], dtype=torch.long, device=device)
                neg_x = torch.as_tensor(x_train[neg_idx], dtype=torch.long, device=device)
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
        val_scores = predict(model, x_val, device, batch_size)
        epoch_metrics = evaluate_scores(val_user, y_val, val_scores, fast_path)
        if epoch_metrics["gauc"] > best_gauc + 1e-7:
            best_gauc = epoch_metrics["gauc"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 2:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    scores = predict(model, x_val, device, batch_size)
    metrics = evaluate_scores(val_user, y_val, scores, fast_path)

    prediction_path = os.path.join(args.out_dir, "predictions.csv")
    with open(prediction_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, (user_id, video_id, score) in enumerate(zip(val_user, val_video, scores)):
            user_value = user_id.item() if hasattr(user_id, "item") else user_id
            video_value = video_id.item() if hasattr(video_id, "item") else video_id
            writer.writerow([i, user_value, video_value, float(score)])

    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, separators=(",", ":"), sort_keys=True)


if __name__ == "__main__":
    main()
