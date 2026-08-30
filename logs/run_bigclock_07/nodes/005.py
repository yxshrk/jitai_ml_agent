import argparse
import csv
import json
import math
import os
import random
import warnings
from datetime import datetime

import numpy as np
import torch
from torch import nn

warnings.filterwarnings("ignore")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    torch.use_deterministic_algorithms(True)


def encode_dates(values):
    out = np.zeros(len(values), dtype=np.float32)
    cache = {}
    for i, value in enumerate(values):
        text = str(value)
        if text.endswith(".0"):
            text = text[:-2]
        text = text.replace("-", "")
        if text not in cache:
            try:
                cache[text] = datetime.strptime(text[:8], "%Y%m%d").toordinal()
            except Exception:
                try:
                    cache[text] = int(float(value))
                except Exception:
                    cache[text] = 0
        out[i] = cache[text]
    return out


def make_recency_weights(dates):
    ordinals = encode_dates(dates)
    latest = float(np.max(ordinals)) if len(ordinals) else 0.0
    weights = np.exp2(-(latest - ordinals) / 7.0).astype(np.float32)
    mean = float(weights.mean()) if len(weights) else 1.0
    if mean > 0:
        weights /= mean
    return np.clip(weights, 0.15, 5.0).astype(np.float32)


def load_npz(data_dir):
    tr = np.load(os.path.join(data_dir, "train.npz"), allow_pickle=False)
    va = np.load(os.path.join(data_dir, "val.npz"), allow_pickle=False)
    x_train = np.asarray(tr["X"], dtype=np.int64)
    x_val = np.asarray(va["X"], dtype=np.int64)
    y_train = np.asarray(tr["y"], dtype=np.float32)
    y_val = np.asarray(va["y"], dtype=np.float32)
    users_val = np.asarray(va["user"])
    field_dims = np.asarray(tr["field_dims"], dtype=np.int64)
    play = np.asarray(tr["play_time_ms"], dtype=np.float32)
    duration = np.asarray(tr["duration_ms"], dtype=np.float32)
    dates = np.asarray(tr["date"])
    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1]))).astype(np.int64)
    video_encoded = x_val[:, 1] - offsets[1]
    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "users_val": users_val,
        "videos_val": video_encoded,
        "field_dims": field_dims,
        "play": play,
        "duration": duration,
        "dates": dates,
        "fast": True,
    }


def read_csv_rows(path, validation=False):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = {
                "user_id": row["user_id"],
                "video_id": row["video_id"],
                "author_id": row.get("author_id", "__unknown_author__"),
                "tab": row["tab"],
                "duration_ms": float(row["duration_ms"] or 0),
                "date": row["date"],
                "long_view": float(row["long_view"] or 0),
            }
            if not validation:
                item["play_time_ms"] = float(row["play_time_ms"] or 0)
            rows.append(item)
    return rows


def build_mapping(values):
    unique = sorted(set(values))
    return {value: i + 1 for i, value in enumerate(unique)}


def load_csv(data_dir):
    train_rows = read_csv_rows(os.path.join(data_dir, "train.csv"), validation=False)
    val_rows = read_csv_rows(os.path.join(data_dir, "val.csv"), validation=True)
    user_map = build_mapping([r["user_id"] for r in train_rows])
    video_map = build_mapping([r["video_id"] for r in train_rows])
    author_map = build_mapping([r["author_id"] for r in train_rows])
    tab_map = build_mapping([r["tab"] for r in train_rows])
    train_durations = np.asarray([r["duration_ms"] for r in train_rows], dtype=np.float32)
    quantiles = np.unique(np.quantile(train_durations, np.linspace(0.1, 0.9, 9)))
    field_dims = np.asarray([
        len(user_map) + 1,
        len(video_map) + 1,
        len(author_map) + 1,
        len(tab_map) + 1,
        len(quantiles) + 2,
    ], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1]))).astype(np.int64)

    def transform(rows):
        x = np.empty((len(rows), 5), dtype=np.int64)
        for i, row in enumerate(rows):
            x[i, 0] = user_map.get(row["user_id"], 0) + offsets[0]
            x[i, 1] = video_map.get(row["video_id"], 0) + offsets[1]
            x[i, 2] = author_map.get(row["author_id"], 0) + offsets[2]
            x[i, 3] = tab_map.get(row["tab"], 0) + offsets[3]
            bucket = int(np.searchsorted(quantiles, row["duration_ms"], side="right")) + 1
            x[i, 4] = bucket + offsets[4]
        return x

    return {
        "x_train": transform(train_rows),
        "y_train": np.asarray([r["long_view"] for r in train_rows], dtype=np.float32),
        "x_val": transform(val_rows),
        "y_val": np.asarray([r["long_view"] for r in val_rows], dtype=np.float32),
        "users_val": np.asarray([r["user_id"] for r in val_rows]),
        "videos_val": np.asarray([r["video_id"] for r in val_rows]),
        "field_dims": field_dims,
        "play": np.asarray([r["play_time_ms"] for r in train_rows], dtype=np.float32),
        "duration": train_durations,
        "dates": np.asarray([r["date"] for r in train_rows]),
        "fast": False,
    }


def make_ordinal_targets(play, duration):
    denominator = np.maximum(1.0, np.minimum(duration, 18000.0))
    ratio = np.maximum(play, 0.0) / denominator
    thresholds = np.asarray([0.25, 0.50, 0.75, 1.00], dtype=np.float32)
    return (ratio[:, None] >= thresholds[None, :]).astype(np.float32)


def make_pairs(x, y, seed, max_pairs=500000):
    rng = np.random.default_rng(seed)
    order = np.argsort(x[:, 0], kind="stable")
    sorted_users = x[order, 0]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    positives = []
    negatives = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        idx = order[start:end]
        pos = idx[y[idx] >= 0.5]
        neg = idx[y[idx] < 0.5]
        if len(pos) and len(neg):
            positives.append(pos)
            negatives.append(rng.choice(neg, size=len(pos), replace=True))
    if not positives:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    pos = np.concatenate(positives).astype(np.int64)
    neg = np.concatenate(negatives).astype(np.int64)
    if len(pos) > max_pairs:
        chosen = rng.choice(len(pos), size=max_pairs, replace=False)
        pos = pos[chosen]
        neg = neg[chosen]
    return pos, neg


class DCNLite(nn.Module):
    def __init__(self, field_dims, embed_dim=16, hidden_dim=128, dropout=0.30):
        super().__init__()
        total = int(np.sum(field_dims))
        self.embedding = nn.Embedding(total, embed_dim)
        flat_dim = len(field_dims) * embed_dim
        self.cross_w = nn.Parameter(torch.empty(flat_dim))
        self.cross_b = nn.Parameter(torch.zeros(flat_dim))
        self.mlp = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.main_head = nn.Linear(hidden_dim // 2, 1)
        self.ordinal_head = nn.Linear(hidden_dim // 2, 4)
        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.normal_(self.cross_w, std=0.01)

    def forward(self, x):
        base = self.embedding(x).flatten(1)
        crossed = base + base * torch.sum(base * self.cross_w, dim=1, keepdim=True) + self.cross_b
        hidden = self.mlp(crossed)
        return self.main_head(hidden).squeeze(1), self.ordinal_head(hidden)


def predict(model, x, batch_size=32768):
    model.eval()
    output = np.empty(len(x), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            xb = torch.from_numpy(x[start:end])
            logits, _ = model(xb)
            output[start:end] = torch.sigmoid(logits).cpu().numpy()
    return output


def normalize_metrics(result):
    return {
        "gauc": float(result.get("GAUC", result.get("gauc"))),
        "ndcg5": float(result.get("nDCG@5", result.get("ndcg5"))),
        "primary": float(result.get("primary")),
    }


def train_variant(data, evaluator, aux_weight, epochs, seed):
    set_seed(seed)
    x = data["x_train"]
    y = data["y_train"]
    weights = make_recency_weights(data["dates"])
    ordinal = make_ordinal_targets(data["play"], data["duration"])
    pair_pos, pair_neg = make_pairs(x, y, seed + 193)
    model = DCNLite(data["field_dims"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.65)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    batch_size = 8192
    rng = np.random.default_rng(seed + 701)
    best_state = None
    best_gauc = -1.0
    best_metrics = None
    stale = 0

    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(x))
        pair_order = rng.permutation(len(pair_pos)) if len(pair_pos) else None
        pair_cursor = 0
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            if len(pair_pos):
                need = len(batch)
                if pair_cursor + need > len(pair_order):
                    pair_order = rng.permutation(len(pair_pos))
                    pair_cursor = 0
                selected = pair_order[pair_cursor:pair_cursor + need]
                pair_cursor += need
                pos_idx = pair_pos[selected]
                neg_idx = pair_neg[selected]
                joined = np.concatenate((x[batch], x[pos_idx], x[neg_idx]), axis=0)
                logits, ordinal_logits = model(torch.from_numpy(joined))
                n = len(batch)
                main_logits = logits[:n]
                pos_logits = logits[n:2 * n]
                neg_logits = logits[2 * n:]
                bpr_loss = -torch.nn.functional.logsigmoid(pos_logits - neg_logits)
                pair_w = torch.from_numpy(0.5 * (weights[pos_idx] + weights[neg_idx]))
                bpr_value = torch.sum(bpr_loss * pair_w) / torch.sum(pair_w)
                ordinal_logits = ordinal_logits[:n]
            else:
                main_logits, ordinal_logits = model(torch.from_numpy(x[batch]))
                bpr_value = torch.zeros((), dtype=torch.float32)
            target = torch.from_numpy(y[batch])
            sample_w = torch.from_numpy(weights[batch])
            main_loss = bce(main_logits, target)
            main_value = torch.sum(main_loss * sample_w) / torch.sum(sample_w)
            ordinal_target = torch.from_numpy(ordinal[batch])
            ordinal_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                ordinal_logits, ordinal_target, reduction="none"
            ).mean(dim=1)
            ordinal_value = torch.sum(ordinal_loss * sample_w) / torch.sum(sample_w)
            loss = 0.5 * main_value + 0.5 * bpr_value + aux_weight * ordinal_value
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        scheduler.step()
        scores = predict(model, data["x_val"])
        metrics = normalize_metrics(evaluator(data["users_val"], data["y_val"], scores))
        if metrics["gauc"] > best_gauc + 1.0e-7:
            best_gauc = metrics["gauc"]
            best_metrics = metrics
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 2:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_metrics


def scalar_text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        value = value.item()
    return str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    fast = os.path.exists(os.path.join(args.data_dir, "train.npz")) and os.path.exists(os.path.join(args.data_dir, "val.npz"))
    if fast:
        data = load_npz(args.data_dir)
        from data.official.evaluate import evaluate as evaluator
    else:
        data = load_csv(args.data_dir)
        from harness.evaluate_provisional import evaluate as evaluator

    smoke = os.environ.get("SMOKE_EPOCHS")
    smoke_cap = int(smoke) if smoke is not None else None
    probe_epochs = 2 if smoke_cap is None else max(1, min(2, smoke_cap))
    final_epochs = 10 if smoke_cap is None else max(1, min(10, smoke_cap))
    candidates = [0.0, 0.2, 0.35, 0.5]
    history = []
    best_weight = candidates[0]
    best_primary = -1.0

    for i, weight in enumerate(candidates):
        _, metrics = train_variant(data, evaluator, weight, probe_epochs, args.seed + i * 1009)
        history.append({
            "phase": "probe",
            "config": {"ordinal_aux_weight": weight, "epochs": probe_epochs},
            "gauc": metrics["gauc"],
            "ndcg5": metrics["ndcg5"],
            "primary": metrics["primary"],
        })
        if metrics["primary"] > best_primary:
            best_primary = metrics["primary"]
            best_weight = weight

    final_model, _ = train_variant(data, evaluator, best_weight, final_epochs, args.seed)
    final_scores = predict(final_model, data["x_val"])
    final_metrics = normalize_metrics(evaluator(data["users_val"], data["y_val"], final_scores))
    history.append({
        "phase": "final",
        "config": {"ordinal_aux_weight": best_weight, "epochs": final_epochs},
        "gauc": final_metrics["gauc"],
        "ndcg5": final_metrics["ndcg5"],
        "primary": final_metrics["primary"],
    })

    prediction_path = os.path.join(args.out_dir, "predictions.csv")
    with open(prediction_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(final_scores):
            writer.writerow([i, scalar_text(data["users_val"][i]), scalar_text(data["videos_val"][i]), "%.10f" % float(score)])

    payload = {
        "gauc": final_metrics["gauc"],
        "ndcg5": final_metrics["ndcg5"],
        "primary": final_metrics["primary"],
        "selected_ordinal_aux_weight": best_weight,
        "history": history,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, sort_keys=True)


if __name__ == "__main__":
    main()
