import argparse
import contextlib
import csv
import datetime
import json
import os
import random
import sys

import numpy as np
import torch
from torch import nn


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def date_ordinal(value):
    s = str(value).strip()
    if s.endswith(".0"):
        s = s[:-2]
    try:
        return datetime.datetime.strptime(s, "%Y%m%d").date().toordinal()
    except ValueError:
        try:
            return int(float(s))
        except ValueError:
            return 0


def time_arrays(date_values, hourmin_values):
    n = len(date_values)
    days = np.empty(n, dtype=np.int64)
    hours = np.empty(n, dtype=np.int64)
    minutes = np.empty(n, dtype=np.int64)
    for i in range(n):
        days[i] = date_ordinal(date_values[i])
        try:
            hm = int(float(hourmin_values[i]))
        except (TypeError, ValueError):
            hm = 0
        h = max(0, min(23, hm // 100))
        m = max(0, min(59, hm % 100))
        hours[i] = h
        minutes[i] = days[i] * 1440 + h * 60 + m
    dow = np.mod(days + 6, 7).astype(np.int64)
    return days, hours, dow, minutes


def position_bucket(position):
    if position <= 3:
        return position
    if position <= 7:
        return 4
    if position <= 15:
        return 5
    if position <= 31:
        return 6
    return 7


def session_features(user_values, date_values, hourmin_values, state=None):
    if state is None:
        state = {}
    _, hours, dow, timestamps = time_arrays(date_values, hourmin_values)
    n = len(user_values)
    pos = np.empty(n, dtype=np.int64)
    gap_bucket = np.empty(n, dtype=np.int64)
    boundaries = np.asarray([1, 2, 5, 10, 30, 60, 180], dtype=np.int64)
    for i in range(n):
        u = user_values[i].item() if isinstance(user_values[i], np.generic) else user_values[i]
        t = int(timestamps[i])
        previous = state.get(u)
        if previous is None:
            gap = 10 ** 9
            current_pos = 0
        else:
            last_t, last_pos = previous
            gap = max(0, t - last_t)
            current_pos = 0 if gap > 30 else last_pos + 1
        pos[i] = position_bucket(current_pos)
        gap_bucket[i] = int(np.searchsorted(boundaries, gap, side="right"))
        state[u] = (t, current_pos)
    hour_gap = hours * 8 + gap_bucket
    dow_pos = dow * 8 + pos
    features = np.column_stack([hours, dow, pos, gap_bucket, hour_gap, dow_pos]).astype(np.int64)
    return features, state


def append_offset_fields(x, field_dims, raw_features):
    field_dims = np.asarray(field_dims, dtype=np.int64)
    extra_dims = np.asarray([24, 7, 8, 8, 192, 56], dtype=np.int64)
    offsets = np.cumsum(np.concatenate([[0], field_dims]))[-1] + np.cumsum(
        np.concatenate([[0], extra_dims[:-1]])
    )
    extra = raw_features + offsets.reshape(1, -1)
    return np.concatenate([x.astype(np.int64), extra], axis=1), np.concatenate([field_dims, extra_dims])


def encode_column(train_values, val_values):
    mapping = {}
    next_id = 1
    train_encoded = np.empty(len(train_values), dtype=np.int64)
    for i, value in enumerate(train_values):
        key = str(value)
        if key not in mapping:
            mapping[key] = next_id
            next_id += 1
        train_encoded[i] = mapping[key]
    val_encoded = np.asarray([mapping.get(str(v), 0) for v in val_values], dtype=np.int64)
    return train_encoded, val_encoded, next_id


def read_csv_split(path, training):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    result = {
        "user": np.asarray([r["user_id"] for r in rows], dtype=object),
        "video": np.asarray([r["video_id"] for r in rows], dtype=object),
        "tab": np.asarray([r.get("tab", "0") for r in rows], dtype=object),
        "date": np.asarray([r.get("date", "0") for r in rows], dtype=object),
        "hourmin": np.asarray([r.get("hourmin", "0") for r in rows], dtype=object),
        "duration": np.asarray([float(r.get("duration_ms", 0) or 0) for r in rows], dtype=np.float64),
    }
    result["y"] = np.asarray([float(r.get("long_view", 0) or 0) for r in rows], dtype=np.float32)
    return result


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        with np.load(train_npz, allow_pickle=True) as z:
            train = {k: z[k] for k in z.files}
        with np.load(val_npz, allow_pickle=True) as z:
            val = {k: z[k] for k in z.files}
        base_dims = np.asarray(train["field_dims"], dtype=np.int64)
        train_user = np.asarray(train["user"])
        val_user = np.asarray(val["user"])
        train_session, state = session_features(train_user, train["date"], train["hourmin"])
        val_session, _ = session_features(val_user, val["date"], val["hourmin"], state)
        x_train, field_dims = append_offset_fields(train["X"], base_dims, train_session)
        x_val, _ = append_offset_fields(val["X"], base_dims, val_session)
        video_offset = int(base_dims[0])
        video_output = np.asarray(val["X"][:, 1], dtype=np.int64) - video_offset
        return {
            "x_train": x_train,
            "y_train": np.asarray(train["y"], dtype=np.float32),
            "x_val": x_val,
            "y_val": np.asarray(val["y"], dtype=np.float32),
            "train_user": train_user,
            "val_user": val_user,
            "video_output": video_output,
            "field_dims": field_dims,
            "train_date": np.asarray(train["date"]),
            "npz": True,
        }

    train = read_csv_split(os.path.join(data_dir, "train.csv"), True)
    val = read_csv_split(os.path.join(data_dir, "val.csv"), False)
    tu, vu, du = encode_column(train["user"], val["user"])
    tv, vv, dv = encode_column(train["video"], val["video"])
    tt, vt, dt = encode_column(train["tab"], val["tab"])
    author_train = np.zeros(len(tu), dtype=np.int64)
    author_val = np.zeros(len(vu), dtype=np.int64)
    da = 1
    quantiles = np.quantile(train["duration"], np.linspace(0.1, 0.9, 9))
    dur_train = np.searchsorted(quantiles, train["duration"], side="right").astype(np.int64)
    dur_val = np.searchsorted(quantiles, val["duration"], side="right").astype(np.int64)
    dims = np.asarray([du, dv, da, dt, 10], dtype=np.int64)
    raw_train = np.column_stack([tu, tv, author_train, tt, dur_train])
    raw_val = np.column_stack([vu, vv, author_val, vt, dur_val])
    offsets = np.cumsum(np.concatenate([[0], dims[:-1]]))
    x_train_base = raw_train + offsets
    x_val_base = raw_val + offsets
    train_session, state = session_features(train["user"], train["date"], train["hourmin"])
    val_session, _ = session_features(val["user"], val["date"], val["hourmin"], state)
    x_train, field_dims = append_offset_fields(x_train_base, dims, train_session)
    x_val, _ = append_offset_fields(x_val_base, dims, val_session)
    return {
        "x_train": x_train,
        "y_train": train["y"],
        "x_val": x_val,
        "y_val": val["y"],
        "train_user": train["user"],
        "val_user": val["user"],
        "video_output": val["video"],
        "field_dims": field_dims,
        "train_date": train["date"],
        "npz": False,
    }


class DCNLite(nn.Module):
    def __init__(self, field_dims, k=24, dropout=0.21):
        super().__init__()
        self.n_fields = len(field_dims)
        self.embedding = nn.Embedding(int(np.sum(field_dims)), k)
        d = self.n_fields * k
        self.cross_weight = nn.Parameter(torch.empty(d))
        self.cross_bias = nn.Parameter(torch.zeros(d))
        self.cross_out = nn.Linear(d, 1)
        self.mlp = nn.Sequential(
            nn.Linear(d, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.normal_(self.cross_weight, std=0.01)
        nn.init.xavier_uniform_(self.cross_out.weight)
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.xavier_uniform_(self.mlp[-1].weight)

    def forward(self, x):
        x0 = self.embedding(x).reshape(x.shape[0], -1)
        x0 = nn.functional.dropout(x0, p=0.21, training=self.training)
        scalar = torch.sum(x0 * self.cross_weight, dim=1, keepdim=True)
        crossed = x0 * scalar + self.cross_bias + x0
        return (self.cross_out(crossed) + self.mlp(x0)).squeeze(1)


def recency_weights(date_values):
    days = np.asarray([date_ordinal(v) for v in date_values], dtype=np.float64)
    newest = float(days.max()) if len(days) else 0.0
    weights = np.exp2(-(newest - days) / 7.0)
    return weights.astype(np.float32)


def make_pairs(users, labels, rng):
    groups = {}
    for i, u0 in enumerate(users):
        u = u0.item() if isinstance(u0, np.generic) else u0
        if u not in groups:
            groups[u] = [[], []]
        groups[u][1 if labels[i] > 0.5 else 0].append(i)
    positives = []
    negatives = []
    for neg, pos in groups.values():
        if neg and pos:
            for p in pos:
                positives.append(p)
                negatives.append(neg[int(rng.integers(0, len(neg)))])
    return np.asarray(positives, dtype=np.int64), np.asarray(negatives, dtype=np.int64)


def predict(model, x, device, batch_size=8192):
    model.eval()
    result = np.empty(len(x), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(len(x), start + batch_size)
            xb = torch.as_tensor(x[start:end], dtype=torch.long, device=device)
            result[start:end] = torch.sigmoid(model(xb)).cpu().numpy()
    return result


def evaluate_scores(npz_mode, users, labels, scores):
    return {"GAUC": 0.0, "nDCG@5": 0.0, "primary": 0.0}


def train_and_score(data, seed, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DCNLite(data["field_dims"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00168, weight_decay=0.000037)
    x = data["x_train"]
    y = data["y_train"]
    weights = recency_weights(data["train_date"])
    rng = np.random.default_rng(seed)
    pair_pos, pair_neg = make_pairs(data["train_user"], y, rng)
    batch_size = 4096
    order = np.arange(len(x), dtype=np.int64)
    steps_per_epoch = max(1, int(np.ceil(len(x) / batch_size)))
    for _ in range(epochs):
        rng.shuffle(order)
        model.train()
        for step in range(steps_per_epoch):
            idx = order[step * batch_size : min(len(order), (step + 1) * batch_size)]
            if len(idx) == 0:
                continue
            xb = torch.as_tensor(x[idx], dtype=torch.long, device=device)
            yb = torch.as_tensor(y[idx], dtype=torch.float32, device=device)
            wb = torch.as_tensor(weights[idx], dtype=torch.float32, device=device)
            point_loss = nn.functional.binary_cross_entropy_with_logits(model(xb), yb, reduction="none")
            point_loss = torch.sum(point_loss * wb) / torch.clamp(torch.sum(wb), min=1e-6)
            loss = point_loss
            if len(pair_pos):
                psel = rng.integers(0, len(pair_pos), size=min(len(idx), len(pair_pos)))
                pi = pair_pos[psel]
                ni = pair_neg[psel]
                xp = torch.as_tensor(x[pi], dtype=torch.long, device=device)
                xn = torch.as_tensor(x[ni], dtype=torch.long, device=device)
                pair_w = torch.as_tensor((weights[pi] + weights[ni]) * 0.5, dtype=torch.float32, device=device)
                pair_loss = nn.functional.softplus(-(model(xp) - model(xn)))
                pair_loss = torch.sum(pair_loss * pair_w) / torch.clamp(torch.sum(pair_w), min=1e-6)
                loss = 0.5 * point_loss + 0.5 * pair_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    scores = predict(model, data["x_val"], device)
    metrics = evaluate_scores(data["npz"], data["val_user"], data["y_val"], scores)
    return scores, {
        "gauc": float(metrics.get("GAUC", metrics.get("gauc", 0.0))),
        "ndcg5": float(metrics.get("nDCG@5", metrics.get("ndcg5", 0.0))),
        "primary": float(metrics.get("primary", 0.0)),
    }


def write_outputs(out_dir, users, videos, scores, metrics):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "predictions.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, (u, v, s) in enumerate(zip(users, videos, scores)):
            writer.writerow([i, u, v, format(float(s), ".9g")])
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, separators=(",", ":"))


def main():
    args = parse_args()
    set_seed(args.seed)
    epochs = 2
    smoke = os.environ.get("SMOKE_EPOCHS")
    if smoke is not None:
        epochs = min(epochs, max(1, int(smoke)))
    data = load_data(args.data_dir)
    scores, metrics = train_and_score(data, args.seed, epochs)
    write_outputs(args.out_dir, data["val_user"], data["video_output"], scores, metrics)


if __name__ == "__main__":
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        main()
