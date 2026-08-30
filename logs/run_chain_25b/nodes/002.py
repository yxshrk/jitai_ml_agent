import argparse
import contextlib
import csv
import datetime as dt
import json
import os
import random

import numpy as np
import torch
from torch import nn


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def quiet_evaluate(fn, user, labels, scores):
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        return fn(user, labels, scores)


def load_npz(data_dir):
    tr = np.load(os.path.join(data_dir, "train.npz"), allow_pickle=False)
    va = np.load(os.path.join(data_dir, "val.npz"), allow_pickle=False)
    x_train = np.asarray(tr["X"], dtype=np.int64)
    y_train = np.asarray(tr["y"], dtype=np.float32)
    train_user = np.asarray(tr["user"])
    train_date = np.asarray(tr["date"]) if "date" in tr.files else np.zeros(len(y_train), dtype=np.int64)
    x_val = np.asarray(va["X"], dtype=np.int64)
    y_val = np.asarray(va["y"], dtype=np.float32)
    val_user = np.asarray(va["user"])
    field_dims = np.asarray(tr["field_dims"], dtype=np.int64).reshape(-1)
    if len(field_dims) != x_train.shape[1]:
        field_dims = np.maximum(x_train.max(axis=0), x_val.max(axis=0)) + 1
    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1]))).astype(np.int64)
    video_out = x_val[:, 1] - offsets[1]
    return x_train, y_train, train_user, train_date, x_val, y_val, val_user, video_out, field_dims


def parse_date_value(value):
    text = str(value).strip()
    try:
        return dt.datetime.strptime(text[:8], "%Y%m%d").date().toordinal()
    except Exception:
        try:
            return int(float(text))
        except Exception:
            return 0


def load_csv(data_dir):
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    input_fields = ["user_id", "video_id", "author_id", "tab"]
    maps = {name: {} for name in input_fields}
    train_raw = []
    train_y = []
    train_dates = []
    durations = []
    with open(train_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            values = []
            for name in input_fields:
                raw = row.get(name, "0")
                if name == "author_id" and name not in row:
                    raw = "0"
                mapping = maps[name]
                if raw not in mapping:
                    mapping[raw] = len(mapping) + 1
                values.append(mapping[raw])
            duration = float(row.get("duration_ms", 0) or 0)
            train_raw.append(values)
            durations.append(duration)
            train_y.append(float(row["long_view"]))
            train_dates.append(parse_date_value(row.get("date", "0")))
    durations_np = np.asarray(durations, dtype=np.float64)
    quantiles = np.quantile(durations_np, np.linspace(0.1, 0.9, 9)) if len(durations_np) else np.zeros(9)
    train_bucket = np.searchsorted(quantiles, durations_np, side="right") + 1
    val_raw = []
    val_y = []
    val_user = []
    video_out = []
    val_durations = []
    with open(val_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            values = []
            for name in input_fields:
                raw = row.get(name, "0")
                if name == "author_id" and name not in row:
                    raw = "0"
                values.append(maps[name].get(raw, 0))
            val_raw.append(values)
            val_durations.append(float(row.get("duration_ms", 0) or 0))
            val_y.append(float(row["long_view"]))
            val_user.append(row["user_id"])
            video_out.append(row["video_id"])
    train_raw = np.asarray(train_raw, dtype=np.int64)
    val_raw = np.asarray(val_raw, dtype=np.int64)
    val_bucket = np.searchsorted(quantiles, np.asarray(val_durations), side="right") + 1
    train_local = np.column_stack((train_raw[:, 0], train_raw[:, 1], train_raw[:, 2], train_raw[:, 3], train_bucket))
    val_local = np.column_stack((val_raw[:, 0], val_raw[:, 1], val_raw[:, 2], val_raw[:, 3], val_bucket))
    field_dims = np.asarray([len(maps[n]) + 1 for n in input_fields] + [11], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1]))).astype(np.int64)
    x_train = train_local + offsets
    x_val = val_local + offsets
    return (x_train.astype(np.int64), np.asarray(train_y, dtype=np.float32), train_raw[:, 0],
            np.asarray(train_dates, dtype=np.int64), x_val.astype(np.int64),
            np.asarray(val_y, dtype=np.float32), np.asarray(val_user),
            np.asarray(video_out), field_dims)


def recency_weights(date_values):
    values = np.asarray(date_values)
    if len(values) == 0:
        return np.empty(0, dtype=np.float32)
    converted = np.empty(len(values), dtype=np.float64)
    cache = {}
    for i, value in enumerate(values):
        key = str(value)
        if key not in cache:
            cache[key] = parse_date_value(value)
        converted[i] = cache[key]
    newest = converted.max()
    age = np.maximum(newest - converted, 0.0)
    weights = np.power(0.5, age / 7.0)
    weights /= max(weights.mean(), 1e-8)
    return weights.astype(np.float32)


def make_pairs(users, labels, seed, limit=600000):
    users = np.asarray(users)
    labels = np.asarray(labels) >= 0.5
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    rng = np.random.default_rng(seed)
    positives = []
    negatives = []
    for j in range(len(boundaries) - 1):
        idx = order[boundaries[j]:boundaries[j + 1]]
        pos = idx[labels[idx]]
        neg = idx[~labels[idx]]
        if len(pos) and len(neg):
            positives.append(pos)
            negatives.append(rng.choice(neg, size=len(pos), replace=True))
    if not positives:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    pos = np.concatenate(positives).astype(np.int64)
    neg = np.concatenate(negatives).astype(np.int64)
    if len(pos) > limit:
        chosen = rng.choice(len(pos), size=limit, replace=False)
        pos = pos[chosen]
        neg = neg[chosen]
    return pos, neg


class DCNLite(nn.Module):
    def __init__(self, field_dims, embed_dim=16, hidden=128, dropout=0.1):
        super().__init__()
        total = int(np.sum(field_dims))
        width = len(field_dims) * embed_dim
        self.embedding = nn.Embedding(total, embed_dim)
        self.linear = nn.Embedding(total, 1)
        self.cross_weight = nn.Parameter(torch.empty(width))
        self.cross_bias = nn.Parameter(torch.zeros(width))
        self.mlp = nn.Sequential(
            nn.Linear(width, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.cross_out = nn.Linear(width, 1, bias=False)
        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.cross_weight, std=0.01)

    def forward(self, x):
        emb = self.embedding(x).flatten(1)
        cross = emb * (torch.sum(emb * self.cross_weight, dim=1, keepdim=True) + self.cross_bias) + emb
        return self.linear(x).sum(dim=1).squeeze(-1) + self.cross_out(cross).squeeze(-1) + self.mlp(emb).squeeze(-1)


def predict(model, x, device, batch_size=32768):
    model.eval()
    result = np.empty(len(x), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            xb = torch.as_tensor(x[start:end], dtype=torch.long, device=device)
            result[start:end] = torch.sigmoid(model(xb)).cpu().numpy()
    return result


def train_one(x_train, y_train, users, sample_weights, x_val, y_val, val_user,
              field_dims, seed, epochs, evaluate_fn, device):
    seed_everything(seed)
    model = DCNLite(field_dims).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.7)
    pos_idx, neg_idx = make_pairs(users, y_train, seed)
    rng = np.random.default_rng(seed)
    batch_size = 8192
    best_gauc = -1.0
    best_state = None
    stale = 0
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(x_train))
        if len(pos_idx):
            pair_order = rng.permutation(len(pos_idx))
        else:
            pair_order = np.empty(0, dtype=np.int64)
        pair_cursor = 0
        for start in range(0, len(order), batch_size):
            ids = order[start:start + batch_size]
            xb = torch.as_tensor(x_train[ids], dtype=torch.long, device=device)
            yb = torch.as_tensor(y_train[ids], dtype=torch.float32, device=device)
            wb = torch.as_tensor(sample_weights[ids], dtype=torch.float32, device=device)
            logits = model(xb)
            point_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yb, reduction="none")
            point_loss = (point_loss * wb).mean()
            if len(pair_order):
                count = len(ids)
                if pair_cursor + count > len(pair_order):
                    pair_order = rng.permutation(len(pos_idx))
                    pair_cursor = 0
                selected = pair_order[pair_cursor:pair_cursor + count]
                pair_cursor += len(selected)
                p = torch.as_tensor(x_train[pos_idx[selected]], dtype=torch.long, device=device)
                n = torch.as_tensor(x_train[neg_idx[selected]], dtype=torch.long, device=device)
                pair_loss = -torch.nn.functional.logsigmoid(model(p) - model(n)).mean()
                loss = 0.5 * point_loss + 0.5 * pair_loss
            else:
                loss = point_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        scheduler.step()
        val_score = predict(model, x_val, device)
        metrics = quiet_evaluate(evaluate_fn, val_user, y_val, val_score)
        gauc = float(metrics.get("GAUC", metrics.get("gauc", 0.0)))
        if gauc > best_gauc + 1e-7:
            best_gauc = gauc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch >= 5 and stale >= 3:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return predict(model, x_val, device)


def within_user_ranks(users, scores):
    users = np.asarray(users)
    scores = np.asarray(scores)
    result = np.empty(len(scores), dtype=np.float64)
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    for j in range(len(boundaries) - 1):
        idx = order[boundaries[j]:boundaries[j + 1]]
        local_order = np.argsort(scores[idx], kind="stable")
        ranks = np.empty(len(idx), dtype=np.float64)
        ranks[local_order] = np.arange(len(idx), dtype=np.float64)
        if len(idx) > 1:
            ranks /= len(idx) - 1
        else:
            ranks[0] = 0.5
        result[idx] = ranks
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    seed_everything(args.seed)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    fast = os.path.exists(os.path.join(args.data_dir, "train.npz")) and os.path.exists(os.path.join(args.data_dir, "val.npz"))
    if fast:
        data = load_npz(args.data_dir)
        from data.official.evaluate import evaluate as evaluate_fn
    else:
        data = load_csv(args.data_dir)
        from harness.evaluate_provisional import evaluate as evaluate_fn
    x_train, y_train, train_user, train_date, x_val, y_val, val_user, video_out, field_dims = data
    weights = recency_weights(train_date)
    requested_epochs = 10
    smoke = os.environ.get("SMOKE_EPOCHS")
    if smoke is not None:
        requested_epochs = min(requested_epochs, max(1, int(smoke)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [args.seed, args.seed + 1009, args.seed + 2017, args.seed + 3023, args.seed + 4021]
    member_scores = []
    for model_seed in seeds:
        scores = train_one(x_train, y_train, train_user, weights, x_val, y_val, val_user,
                           field_dims, model_seed, requested_epochs, evaluate_fn, device)
        member_scores.append(within_user_ranks(val_user, scores))
    final_scores = np.mean(np.stack(member_scores, axis=0), axis=0)
    metrics = quiet_evaluate(evaluate_fn, val_user, y_val, final_scores)
    gauc = float(metrics.get("GAUC", metrics.get("gauc")))
    ndcg = float(metrics.get("nDCG@5", metrics.get("ndcg5")))
    primary = float(metrics.get("primary", 0.5 * (gauc + ndcg)))
    prediction_path = os.path.join(args.out_dir, "predictions.csv")
    with open(prediction_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i in range(len(final_scores)):
            writer.writerow([i, val_user[i], video_out[i], format(float(final_scores[i]), ".10g")])
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump({"gauc": gauc, "ndcg5": ndcg, "primary": primary}, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
