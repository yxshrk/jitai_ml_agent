import argparse
import csv
import datetime as dt
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def date_ordinal(value):
    value = int(value)
    try:
        return dt.date(value // 10000, (value // 100) % 100, value % 100).toordinal()
    except Exception:
        return value


def absolute_minutes(dates, hourmins):
    out = np.empty(len(dates), dtype=np.int64)
    cache = {}
    for i, (date_value, hm_value) in enumerate(zip(dates, hourmins)):
        key = int(date_value)
        day = cache.get(key)
        if day is None:
            day = date_ordinal(key)
            cache[key] = day
        hm = int(hm_value)
        hour = max(0, min(23, hm // 100))
        minute = max(0, min(59, hm % 100))
        out[i] = day * 1440 + hour * 60 + minute
    return out


def session_features(users, dates, hourmins):
    times = absolute_minutes(dates, hourmins)
    gap_bucket = np.zeros(len(users), dtype=np.int64)
    position = np.zeros(len(users), dtype=np.int64)
    state = {}
    for i in range(len(users)):
        key = users[i].item() if isinstance(users[i], np.generic) else users[i]
        now = int(times[i])
        previous = state.get(key)
        if previous is None:
            gap_bucket[i] = 0
            position[i] = 0
            state[key] = (now, 0)
            continue
        last_time, last_position = previous
        gap = now - last_time
        if gap < 0 or gap > 30:
            position_value = 0
        else:
            position_value = min(last_position + 1, 15)
        if gap < 0:
            bucket = 7
        elif gap <= 1:
            bucket = 1
        elif gap <= 3:
            bucket = 2
        elif gap <= 7:
            bucket = 3
        elif gap <= 15:
            bucket = 4
        elif gap <= 30:
            bucket = 5
        elif gap <= 120:
            bucket = 6
        else:
            bucket = 7
        gap_bucket[i] = bucket
        position[i] = position_value
        state[key] = (now, position_value)
    return gap_bucket, position


def append_session_fields(train_x, val_x, train_user, val_user, train_date, val_date,
                          train_hourmin, val_hourmin, field_dims):
    users = np.concatenate([train_user, val_user])
    dates = np.concatenate([train_date, val_date])
    hourmins = np.concatenate([train_hourmin, val_hourmin])
    gaps, positions = session_features(users, dates, hourmins)
    split = len(train_x)
    base_offset = int(np.sum(field_dims))
    gap_offset = base_offset
    position_offset = base_offset + 8
    train_extra = np.stack([gaps[:split] + gap_offset,
                            positions[:split] + position_offset], axis=1)
    val_extra = np.stack([gaps[split:] + gap_offset,
                          positions[split:] + position_offset], axis=1)
    train_out = np.concatenate([train_x.astype(np.int64), train_extra], axis=1)
    val_out = np.concatenate([val_x.astype(np.int64), val_extra], axis=1)
    dims = np.concatenate([np.asarray(field_dims, dtype=np.int64),
                           np.asarray([8, 16], dtype=np.int64)])
    return train_out, val_out, dims


def load_npz(data_dir):
    train = np.load(Path(data_dir) / "train.npz", allow_pickle=False)
    val = np.load(Path(data_dir) / "val.npz", allow_pickle=False)
    train_x = np.asarray(train["X"], dtype=np.int64)
    val_x = np.asarray(val["X"], dtype=np.int64)
    train_y = np.asarray(train["y"], dtype=np.float32)
    val_y = np.asarray(val["y"], dtype=np.float32)
    train_user = np.asarray(train["user"])
    val_user = np.asarray(val["user"])
    train_date = np.asarray(train["date"])
    val_date = np.asarray(val["date"])
    train_hourmin = np.asarray(train["hourmin"])
    val_hourmin = np.asarray(val["hourmin"])
    field_dims = np.asarray(train["field_dims"], dtype=np.int64)
    train_x, val_x, field_dims = append_session_fields(
        train_x, val_x, train_user, val_user, train_date, val_date,
        train_hourmin, val_hourmin, field_dims)
    video_offset = int(field_dims[0])
    val_video = val_x[:, 1] - video_offset
    return {
        "train_x": train_x,
        "train_y": train_y,
        "train_user": train_user,
        "train_date": train_date,
        "val_x": val_x,
        "val_y": val_y,
        "val_user": val_user,
        "val_video": val_video,
        "field_dims": field_dims,
        "fast": True,
    }


def read_csv_columns(path, validation=False):
    allowed = {"user_id", "video_id", "tab", "hourmin", "date", "duration_ms", "long_view"}
    result = {name: [] for name in allowed}
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        indexes = {name: header.index(name) for name in allowed if name in header}
        for row in reader:
            for name, index in indexes.items():
                result[name].append(row[index])
    return result


def make_mapping(values):
    mapping = {}
    for value in values:
        if value not in mapping:
            mapping[value] = len(mapping) + 1
    return mapping


def encode(values, mapping):
    return np.asarray([mapping.get(value, 0) for value in values], dtype=np.int64)


def load_csv(data_dir):
    train = read_csv_columns(Path(data_dir) / "train.csv")
    val = read_csv_columns(Path(data_dir) / "val.csv", validation=True)
    user_map = make_mapping(train["user_id"])
    video_map = make_mapping(train["video_id"])
    tab_map = make_mapping(train["tab"])
    train_duration = np.asarray(train["duration_ms"], dtype=np.float64)
    val_duration = np.asarray(val["duration_ms"], dtype=np.float64)
    quantiles = np.quantile(train_duration, np.linspace(0.1, 0.9, 9))
    train_local = np.stack([
        encode(train["user_id"], user_map),
        encode(train["video_id"], video_map),
        np.zeros(len(train["user_id"]), dtype=np.int64),
        encode(train["tab"], tab_map),
        np.searchsorted(quantiles, train_duration, side="right").astype(np.int64),
    ], axis=1)
    val_local = np.stack([
        encode(val["user_id"], user_map),
        encode(val["video_id"], video_map),
        np.zeros(len(val["user_id"]), dtype=np.int64),
        encode(val["tab"], tab_map),
        np.searchsorted(quantiles, val_duration, side="right").astype(np.int64),
    ], axis=1)
    dims = np.asarray([len(user_map) + 1, len(video_map) + 1, 1,
                       len(tab_map) + 1, 10], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(dims)[:-1]])
    train_x = train_local + offsets
    val_x = val_local + offsets
    train_user = np.asarray(train["user_id"])
    val_user = np.asarray(val["user_id"])
    train_date = np.asarray(train["date"], dtype=np.int64)
    val_date = np.asarray(val["date"], dtype=np.int64)
    train_hourmin = np.asarray(train["hourmin"], dtype=np.int64)
    val_hourmin = np.asarray(val["hourmin"], dtype=np.int64)
    train_x, val_x, dims = append_session_fields(
        train_x, val_x, train_user, val_user, train_date, val_date,
        train_hourmin, val_hourmin, dims)
    return {
        "train_x": train_x,
        "train_y": np.asarray(train["long_view"], dtype=np.float32),
        "train_user": train_user,
        "train_date": train_date,
        "val_x": val_x,
        "val_y": np.asarray(val["long_view"], dtype=np.float32),
        "val_user": val_user,
        "val_video": np.asarray(val["video_id"]),
        "field_dims": dims,
        "fast": False,
    }


class CrossLayer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(width))
        self.bias = nn.Parameter(torch.zeros(width))
        nn.init.normal_(self.weight, std=0.01)

    def forward(self, x0, x):
        scale = torch.sum(x * self.weight, dim=1, keepdim=True)
        return x0 * scale + self.bias + x


class SessionDCN(nn.Module):
    def __init__(self, field_dims, embedding_dim=24, dropout=0.21):
        super().__init__()
        total = int(np.sum(field_dims))
        fields = len(field_dims)
        width = fields * embedding_dim
        self.embedding = nn.Embedding(total, embedding_dim)
        self.linear = nn.Embedding(total, 1)
        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.zeros_(self.linear.weight)
        self.embedding_dropout = nn.Dropout(dropout)
        self.cross1 = CrossLayer(width)
        self.cross2 = CrossLayer(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self.cross_out = nn.Linear(width, 1)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        embedded = self.embedding_dropout(self.embedding(x)).flatten(1)
        crossed = self.cross1(embedded, embedded)
        crossed = self.cross2(embedded, crossed)
        first_order = self.linear(x).sum(dim=1).squeeze(1)
        return first_order + self.cross_out(crossed).squeeze(1) + self.mlp(embedded).squeeze(1) + self.bias


def recency_weights(dates, half_life=7.0):
    ordinals = np.asarray([date_ordinal(v) for v in dates], dtype=np.float64)
    age = np.max(ordinals) - ordinals
    weights = np.exp2(-age / half_life)
    weights /= max(float(np.mean(weights)), 1e-8)
    return weights.astype(np.float32)


def opposite_pairs(users, labels, seed):
    groups = {}
    for i, user in enumerate(users):
        key = user.item() if isinstance(user, np.generic) else user
        if key not in groups:
            groups[key] = [[], []]
        groups[key][int(labels[i] >= 0.5)].append(i)
    rng = np.random.default_rng(seed)
    pairs = np.arange(len(labels), dtype=np.int64)
    valid = np.zeros(len(labels), dtype=np.float32)
    for i, user in enumerate(users):
        key = user.item() if isinstance(user, np.generic) else user
        candidates = groups[key][1 - int(labels[i] >= 0.5)]
        if candidates:
            pairs[i] = candidates[int(rng.integers(0, len(candidates)))]
            valid[i] = 1.0
    return pairs, valid


def metric_values(users, labels, scores, fast):
    if fast:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    result = evaluate(users, labels, scores)
    return {
        "gauc": float(result["GAUC"]),
        "ndcg5": float(result["nDCG@5"]),
        "primary": float(result["primary"]),
    }


def predict(model, x, device, batch_size=8192):
    model.eval()
    pieces = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.as_tensor(x[start:start + batch_size], dtype=torch.long, device=device)
            pieces.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(pieces).astype(np.float64)


def train_model(data, seed, device, epochs):
    x = data["train_x"]
    y = data["train_y"]
    weights = recency_weights(data["train_date"], 7.0)
    pairs, pair_valid = opposite_pairs(data["train_user"], y, seed)
    dataset = TensorDataset(
        torch.from_numpy(x.astype(np.int64)),
        torch.from_numpy(y.astype(np.float32)),
        torch.from_numpy(weights),
        torch.from_numpy(pairs),
        torch.from_numpy(pair_valid),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    batch_size = 4096 if device.type == "cuda" else 2048
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator,
                        num_workers=0, drop_last=False)
    model = SessionDCN(data["field_dims"], embedding_dim=24, dropout=0.21).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00168, weight_decay=0.000037)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    best_state = None
    best_primary = -float("inf")
    history = []
    global_step = 0
    half_boundary = max(1, math.ceil(len(loader) / 2))
    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        seen = 0
        for step, (xb, yb, wb, pair_index, valid_pair) in enumerate(loader, start=1):
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)
            valid_pair = valid_pair.to(device)
            pair_x = torch.as_tensor(x[pair_index.numpy()], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            paired_logits = model(pair_x)
            point_loss = (bce(logits, yb) * wb).mean()
            direction = torch.where(yb >= 0.5, logits - paired_logits, paired_logits - logits)
            pair_terms = torch.nn.functional.softplus(-direction) * valid_pair * wb
            pair_loss = pair_terms.sum() / valid_pair.mul(wb).sum().clamp_min(1.0)
            loss = 0.5 * point_loss + 0.5 * pair_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            count = len(xb)
            loss_sum += float(loss.detach().cpu()) * count
            seen += count
            global_step += 1
            if step == half_boundary or step == len(loader):
                val_scores = predict(model, data["val_x"], device)
                metrics = metric_values(data["val_user"], data["val_y"], val_scores, data["fast"])
                record = {
                    "epoch": float(epoch + step / len(loader)),
                    "train_loss": float(loss_sum / max(seen, 1)),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    **metrics,
                }
                history.append(record)
                if metrics["primary"] > best_primary:
                    best_primary = metrics["primary"]
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                model.train()
        scheduler.step()
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def write_predictions(path, users, videos, scores):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, (user, video, score) in enumerate(zip(users, videos, scores)):
            user_value = user.item() if isinstance(user, np.generic) else user
            video_value = video.item() if isinstance(video, np.generic) else video
            writer.writerow([i, user_value, video_value, format(float(score), ".10g")])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if (Path(args.data_dir) / "train.npz").exists() and (Path(args.data_dir) / "val.npz").exists():
        data = load_npz(args.data_dir)
    else:
        data = load_csv(args.data_dir)
    epochs = 6
    smoke = os.environ.get("SMOKE_EPOCHS")
    if smoke is not None:
        epochs = min(epochs, max(1, int(smoke)))
    model, history = train_model(data, args.seed, device, epochs)
    scores = predict(model, data["val_x"], device)
    metrics = metric_values(data["val_user"], data["val_y"], scores, data["fast"])
    metrics["history"] = history
    metrics["best_epoch"] = float(max(history, key=lambda row: row["primary"])["epoch"])
    metrics["seed"] = int(args.seed)
    metrics["configuration"] = {
        "change": "session_time_features",
        "gap_bins_minutes": [1, 3, 7, 15, 30, 120],
        "session_boundary_minutes": 30,
        "position_cap": 15,
        "embedding_dim": 24,
        "dropout": 0.21,
        "weight_decay": 0.000037,
        "learning_rate": 0.00168,
        "recency_half_life_days": 7.0,
        "objective": "0.5_bce_0.5_bpr",
    }
    write_predictions(out_dir / "predictions.csv", data["val_user"], data["val_video"], scores)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, separators=(",", ":"), sort_keys=True)


if __name__ == "__main__":
    main()
