import argparse
import csv
import json
import math
import os
import random
import time
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
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def metric_values(result):
    def get(*names):
        for name in names:
            if name in result:
                return float(result[name])
        raise KeyError(names[0])
    return {
        "gauc": get("GAUC", "gauc"),
        "ndcg5": get("nDCG@5", "ndcg5", "NDCG@5"),
        "primary": get("primary", "PRIMARY"),
    }


def parse_scalar(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def read_csv_rows(path, training):
    feature_rows = []
    labels = []
    users = []
    videos = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        has_author = reader.fieldnames is not None and "author_id" in reader.fieldnames
        for row in reader:
            user = row["user_id"]
            video = row["video_id"]
            author = row["author_id"] if has_author else "__unknown_author__"
            tab = row.get("tab", "0")
            duration = float(row.get("duration_ms", "0") or 0.0)
            feature_rows.append((user, video, author, tab, duration))
            users.append(parse_scalar(user))
            videos.append(parse_scalar(video))
            if training:
                labels.append(float(row["long_view"]))
            else:
                labels.append(float(row["long_view"]))
    return feature_rows, np.asarray(labels, dtype=np.float32), np.asarray(users), np.asarray(videos)


def encode_csv(train_rows, val_rows):
    duration_train = np.asarray([r[4] for r in train_rows], dtype=np.float64)
    if duration_train.size:
        quantiles = np.quantile(duration_train, np.linspace(0.1, 0.9, 9))
        quantiles = np.maximum.accumulate(quantiles)
    else:
        quantiles = np.zeros(9, dtype=np.float64)

    train_columns = [
        [r[0] for r in train_rows],
        [r[1] for r in train_rows],
        [r[2] for r in train_rows],
        [r[3] for r in train_rows],
    ]
    val_columns = [
        [r[0] for r in val_rows],
        [r[1] for r in val_rows],
        [r[2] for r in val_rows],
        [r[3] for r in val_rows],
    ]
    encoded_train = []
    encoded_val = []
    field_dims = []
    offset = 0
    for train_col, val_col in zip(train_columns, val_columns):
        mapping = {}
        for value in train_col:
            if value not in mapping:
                mapping[value] = len(mapping) + 1
        dim = len(mapping) + 1
        encoded_train.append(np.asarray([mapping.get(v, 0) + offset for v in train_col], dtype=np.int64))
        encoded_val.append(np.asarray([mapping.get(v, 0) + offset for v in val_col], dtype=np.int64))
        field_dims.append(dim)
        offset += dim

    train_bucket = np.searchsorted(quantiles, duration_train, side="right").astype(np.int64)
    val_duration = np.asarray([r[4] for r in val_rows], dtype=np.float64)
    val_bucket = np.searchsorted(quantiles, val_duration, side="right").astype(np.int64)
    encoded_train.append(train_bucket + offset)
    encoded_val.append(val_bucket + offset)
    field_dims.append(10)

    return (
        np.column_stack(encoded_train).astype(np.int64),
        np.column_stack(encoded_val).astype(np.int64),
        np.asarray(field_dims, dtype=np.int64),
    )


def load_data(data_dir):
    train_npz = data_dir / "train.npz"
    val_npz = data_dir / "val.npz"
    if train_npz.exists() and val_npz.exists():
        with np.load(train_npz, allow_pickle=False) as tr:
            x_train = np.asarray(tr["X"], dtype=np.int64)
            y_train = np.asarray(tr["y"], dtype=np.float32).reshape(-1)
            train_users = np.asarray(tr["user"]).reshape(-1)
            field_dims = np.asarray(tr["field_dims"], dtype=np.int64).reshape(-1)
        with np.load(val_npz, allow_pickle=False) as va:
            x_val = np.asarray(va["X"], dtype=np.int64)
            y_val = np.asarray(va["y"], dtype=np.float32).reshape(-1)
            val_users = np.asarray(va["user"]).reshape(-1)
        video_offset = int(field_dims[0]) if field_dims.size > 1 else 0
        val_videos = x_val[:, 1].astype(np.int64) - video_offset
        total_dim = max(int(field_dims.sum()), int(max(x_train.max(initial=0), x_val.max(initial=0))) + 1)
        return x_train, y_train, train_users, x_val, y_val, val_users, val_videos, total_dim, True

    train_rows, y_train, train_users, _ = read_csv_rows(data_dir / "train.csv", True)
    val_rows, y_val, val_users, val_videos = read_csv_rows(data_dir / "val.csv", False)
    x_train, x_val, field_dims = encode_csv(train_rows, val_rows)
    return x_train, y_train, train_users, x_val, y_val, val_users, val_videos, int(field_dims.sum()), False


class FactorizationMachine(nn.Module):
    def __init__(self, total_dim, embedding_dim=16, dropout=0.20, initial_bias=0.0):
        super().__init__()
        self.linear = nn.Embedding(total_dim, 1)
        self.embedding = nn.Embedding(total_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.global_bias = nn.Parameter(torch.tensor(float(initial_bias), dtype=torch.float32))
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

    def raw_score(self, x):
        linear = self.linear(x).sum(dim=1).squeeze(-1)
        emb = self.dropout(self.embedding(x))
        summed = emb.sum(dim=1)
        interaction = 0.5 * (summed.square() - emb.square().sum(dim=1)).sum(dim=1)
        return linear + interaction


def build_user_slates(users):
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    if len(order) == 0:
        return order, np.asarray([0], dtype=np.int64)
    starts = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1]])
    boundaries = np.r_[starts, len(order)].astype(np.int64)
    return order.astype(np.int64), boundaries


def make_slate_batches(sorted_rows, boundaries, rng, target_rows=65536):
    n_users = len(boundaries) - 1
    user_order = np.arange(n_users, dtype=np.int64)
    rng.shuffle(user_order)
    batches = []
    pieces = []
    lengths = []
    count = 0
    for user_index in user_order:
        lo = int(boundaries[user_index])
        hi = int(boundaries[user_index + 1])
        length = hi - lo
        if pieces and count + length > target_rows:
            batches.append((np.concatenate(pieces), np.asarray(lengths, dtype=np.int64)))
            pieces = []
            lengths = []
            count = 0
        pieces.append(sorted_rows[lo:hi])
        lengths.append(length)
        count += length
    if pieces:
        batches.append((np.concatenate(pieces), np.asarray(lengths, dtype=np.int64)))
    return batches


def centered_logits(raw, group_ids, group_count, global_bias):
    sums = torch.zeros(group_count, dtype=raw.dtype, device=raw.device)
    sums.index_add_(0, group_ids, raw)
    counts = torch.bincount(group_ids, minlength=group_count).to(raw.dtype)
    means = sums / counts.clamp_min(1.0)
    return raw - means[group_ids] + global_bias


def pair_indices(y_batch, lengths, rng):
    positive = []
    negative = []
    start = 0
    for length in lengths.tolist():
        stop = start + int(length)
        local = y_batch[start:stop]
        pos = np.flatnonzero(local > 0.5) + start
        neg = np.flatnonzero(local <= 0.5) + start
        if pos.size and neg.size:
            pair_count = max(pos.size, neg.size)
            p = rng.choice(pos, size=pair_count, replace=pos.size < pair_count)
            n = rng.choice(neg, size=pair_count, replace=neg.size < pair_count)
            positive.append(p)
            negative.append(n)
        start = stop
    if not positive:
        return None, None
    return np.concatenate(positive).astype(np.int64), np.concatenate(negative).astype(np.int64)


def predict(model, x, users, centered, device, batch_size=131072):
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.as_tensor(x[start:start + batch_size], dtype=torch.long, device=device)
            chunks.append(model.raw_score(xb).detach().cpu().numpy())
    raw = np.concatenate(chunks).astype(np.float64)
    bias = float(model.global_bias.detach().cpu())
    if not centered:
        return raw + bias
    _, inverse = np.unique(users, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    sums = np.bincount(inverse, weights=raw).astype(np.float64)
    return raw - (sums / np.maximum(counts, 1.0))[inverse] + bias


def evaluate_scores(evaluator, users, labels, scores):
    return metric_values(evaluator(users, labels, scores))


def train_one(x_train, y_train, train_users, x_val, y_val, val_users, total_dim,
              centered, seed, epochs, evaluator, device):
    seed_everything(seed)
    rng = np.random.default_rng(seed + 193)
    prevalence = float(np.clip(y_train.mean(), 1e-5, 1.0 - 1e-5))
    initial_bias = math.log(prevalence / (1.0 - prevalence))
    model = FactorizationMachine(total_dim, embedding_dim=16, dropout=0.20,
                                 initial_bias=initial_bias).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 6, 8], gamma=0.35)
    sorted_rows, boundaries = build_user_slates(train_users)
    best_primary = -float("inf")
    best_metrics = None
    best_predictions = None
    best_epoch = 0
    epoch_history = []
    stale = 0

    for epoch in range(epochs):
        model.train()
        batches = make_slate_batches(sorted_rows, boundaries, rng)
        for row_indices, lengths in batches:
            xb = torch.as_tensor(x_train[row_indices], dtype=torch.long, device=device)
            y_np = y_train[row_indices]
            yb = torch.as_tensor(y_np, dtype=torch.float32, device=device)
            group_np = np.repeat(np.arange(len(lengths), dtype=np.int64), lengths)
            group_ids = torch.as_tensor(group_np, dtype=torch.long, device=device)

            optimizer.zero_grad(set_to_none=True)
            raw = model.raw_score(xb)
            if centered:
                point_logits = centered_logits(raw, group_ids, len(lengths), model.global_bias)
            else:
                point_logits = raw + model.global_bias
            bce = F.binary_cross_entropy_with_logits(point_logits, yb)

            pos_np, neg_np = pair_indices(y_np, lengths, rng)
            if pos_np is None:
                bpr = raw.sum() * 0.0
            else:
                pos = torch.as_tensor(pos_np, dtype=torch.long, device=device)
                neg = torch.as_tensor(neg_np, dtype=torch.long, device=device)
                bpr = F.softplus(-(raw[pos] - raw[neg])).mean()
            loss = 0.5 * bce + 0.5 * bpr
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        scheduler.step()
        predictions = predict(model, x_val, val_users, centered, device)
        metrics = evaluate_scores(evaluator, val_users, y_val, predictions)
        epoch_history.append({"epoch": epoch + 1, **metrics})
        if metrics["primary"] > best_primary + 1e-12:
            best_primary = metrics["primary"]
            best_metrics = metrics
            best_predictions = predictions.copy()
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
        if stale >= 3 and epoch + 1 >= 5:
            break

    return best_predictions, best_metrics, best_epoch, epoch_history


def append_progress(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.log"
    if progress_path.exists():
        progress_path.unlink()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = load_data(data_dir)
    x_train, y_train, train_users, x_val, y_val, val_users, val_videos, total_dim, fast_path = loaded

    if fast_path:
        from data.official.evaluate import evaluate as evaluator
    else:
        from harness.evaluate_provisional import evaluate as evaluator

    smoke_value = os.environ.get("SMOKE_EPOCHS")
    epochs = 10
    if smoke_value is not None:
        epochs = max(1, min(epochs, int(smoke_value)))
    seed_count = 7 if smoke_value is None or int(smoke_value) > 1 else 2
    seeds = [args.seed + 1009 * i for i in range(seed_count)]

    history = []
    centered_predictions = []
    paired_deltas = []
    run_start = time.time()

    for seed in seeds:
        ordinary_start = time.time()
        ordinary_pred, ordinary_metrics, ordinary_epoch, ordinary_epochs = train_one(
            x_train, y_train, train_users, x_val, y_val, val_users, total_dim,
            False, seed, epochs, evaluator, device
        )
        ordinary_record = {
            "config": "ordinary_bce_plus_bpr",
            "seed": seed,
            "centered_bce": False,
            "bce_weight": 0.5,
            "bpr_weight": 0.5,
            "embedding_dim": 16,
            "dropout": 0.20,
            "weight_decay": 1e-5,
            "best_epoch": ordinary_epoch,
            "runtime_seconds": time.time() - ordinary_start,
            **ordinary_metrics,
            "epochs": ordinary_epochs,
        }
        history.append(ordinary_record)
        append_progress(progress_path, {k: v for k, v in ordinary_record.items() if k != "epochs"})

        centered_start = time.time()
        centered_pred, centered_metrics, centered_epoch, centered_epochs = train_one(
            x_train, y_train, train_users, x_val, y_val, val_users, total_dim,
            True, seed, epochs, evaluator, device
        )
        if np.allclose(centered_pred, ordinary_pred, rtol=1e-7, atol=1e-8):
            raise RuntimeError("Centered and ordinary member predictions are identical")
        for previous in centered_predictions:
            if np.allclose(centered_pred, previous, rtol=1e-7, atol=1e-8):
                raise RuntimeError("Two centered seed members produced identical predictions")
        centered_predictions.append(centered_pred)
        delta = float(centered_metrics["primary"] - ordinary_metrics["primary"])
        paired_deltas.append(delta)
        centered_record = {
            "config": "gauge_fixed_bce_plus_bpr",
            "seed": seed,
            "centered_bce": True,
            "complete_user_slates": True,
            "bce_weight": 0.5,
            "bpr_weight": 0.5,
            "embedding_dim": 16,
            "dropout": 0.20,
            "weight_decay": 1e-5,
            "best_epoch": centered_epoch,
            "paired_primary_delta": delta,
            "runtime_seconds": time.time() - centered_start,
            **centered_metrics,
            "epochs": centered_epochs,
        }
        history.append(centered_record)
        append_progress(progress_path, {k: v for k, v in centered_record.items() if k != "epochs"})

    final_predictions = np.mean(np.stack(centered_predictions, axis=0), axis=0)
    final_metrics = evaluate_scores(evaluator, val_users, y_val, final_predictions)
    delta_array = np.asarray(paired_deltas, dtype=np.float64)
    mean_delta = float(delta_array.mean())
    if len(delta_array) > 1:
        standard_error = float(delta_array.std(ddof=1) / math.sqrt(len(delta_array)))
    else:
        standard_error = 0.0

    predictions_path = out_dir / "predictions.csv"
    with open(predictions_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for row_id, (user, video, score) in enumerate(zip(val_users, val_videos, final_predictions)):
            user_value = user.item() if isinstance(user, np.generic) else user
            video_value = video.item() if isinstance(video, np.generic) else video
            writer.writerow([row_id, user_value, video_value, format(float(score), ".12g")])

    metrics_payload = {
        "gauc": final_metrics["gauc"],
        "ndcg5": final_metrics["ndcg5"],
        "primary": final_metrics["primary"],
        "history": history,
        "paired_summary": {
            "seeds": seeds,
            "primary_deltas": paired_deltas,
            "mean_primary_delta": mean_delta,
            "standard_error": standard_error,
            "ci95_low": mean_delta - 1.96 * standard_error,
            "ci95_high": mean_delta + 1.96 * standard_error,
        },
        "final_model": {
            "type": "mean_of_gauge_fixed_seed_members",
            "member_count": len(centered_predictions),
            "complete_user_slates": True,
            "total_runtime_seconds": time.time() - run_start,
            "device": device.type,
            "fast_path": fast_path,
        },
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, sort_keys=True, indent=2)


if __name__ == "__main__":
    main()
