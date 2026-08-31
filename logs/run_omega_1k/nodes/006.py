import argparse
import csv
import datetime
import json
import os
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def date_number(value):
    text = str(value).strip()
    try:
        text = str(int(float(text)))
    except Exception:
        return 0
    if len(text) == 8:
        try:
            return datetime.date(int(text[:4]), int(text[4:6]), int(text[6:8])).toordinal()
        except Exception:
            return 0
    try:
        return int(text)
    except Exception:
        return 0


def minute_of_day(value):
    text = str(value).strip()
    if not text:
        return -1
    if ":" in text:
        try:
            hour, minute = text.split(":", 1)
            result = int(hour) * 60 + int(float(minute))
            return result if 0 <= result < 1440 else -1
        except Exception:
            return -1
    try:
        number = int(float(text))
    except Exception:
        return -1
    hour = number // 100
    minute = number % 100
    if 0 <= hour < 24 and 0 <= minute < 60:
        return hour * 60 + minute
    if 0 <= number < 1440:
        return number
    return -1


def duration_bucket(values, edges=None):
    values = np.asarray(values, dtype=np.float64)
    if edges is None:
        finite = values[np.isfinite(values)]
        edges = np.unique(np.quantile(finite, np.linspace(0.1, 0.9, 9))) if finite.size else np.arange(1, 10)
    return np.searchsorted(edges, values, side="right").astype(np.int64), np.asarray(edges)


def load_csv_data(data_dir):
    names = ["user_id", "video_id", "author_id", "tab"]
    train_raw = {name: [] for name in names}
    val_raw = {name: [] for name in names}
    train_duration, val_duration = [], []
    train_y, val_y = [], []
    train_date, val_date = [], []
    train_hourmin, val_hourmin = [], []
    val_users, val_videos = [], []

    with open(os.path.join(data_dir, "train.csv"), "r", newline="") as handle:
        reader = csv.DictReader(handle)
        available = set(reader.fieldnames or [])
        for row in reader:
            train_raw["user_id"].append(row.get("user_id", ""))
            train_raw["video_id"].append(row.get("video_id", ""))
            train_raw["author_id"].append(row.get("author_id", "__missing__") if "author_id" in available else "__missing__")
            train_raw["tab"].append(row.get("tab", ""))
            train_duration.append(float(row.get("duration_ms", 0) or 0))
            train_y.append(float(row["long_view"]))
            train_date.append(date_number(row.get("date", 0)))
            train_hourmin.append(row.get("hourmin", ""))

    with open(os.path.join(data_dir, "val.csv"), "r", newline="") as handle:
        reader = csv.DictReader(handle)
        available = set(reader.fieldnames or [])
        for row in reader:
            user = row.get("user_id", "")
            video = row.get("video_id", "")
            val_raw["user_id"].append(user)
            val_raw["video_id"].append(video)
            val_raw["author_id"].append(row.get("author_id", "__missing__") if "author_id" in available else "__missing__")
            val_raw["tab"].append(row.get("tab", ""))
            val_duration.append(float(row.get("duration_ms", 0) or 0))
            val_y.append(float(row["long_view"]))
            val_date.append(date_number(row.get("date", 0)))
            val_hourmin.append(row.get("hourmin", ""))
            val_users.append(user)
            val_videos.append(video)

    train_db, edges = duration_bucket(train_duration)
    val_db, _ = duration_bucket(val_duration, edges)
    train_columns, val_columns, dims = [], [], []
    for name in names:
        mapping = {}
        encoded_train = np.empty(len(train_y), dtype=np.int64)
        for index, value in enumerate(train_raw[name]):
            if value not in mapping:
                mapping[value] = len(mapping) + 1
            encoded_train[index] = mapping[value]
        train_columns.append(encoded_train)
        val_columns.append(np.asarray([mapping.get(value, 0) for value in val_raw[name]], dtype=np.int64))
        dims.append(len(mapping) + 1)
    train_columns.append(train_db)
    val_columns.append(val_db)
    dims.append(max(10, int(train_db.max(initial=0)) + 1))
    offsets = np.cumsum([0] + dims[:-1], dtype=np.int64)
    return {
        "train_x": (np.stack(train_columns, axis=1) + offsets[None, :]).astype(np.int64),
        "train_y": np.asarray(train_y, dtype=np.float32),
        "train_user": np.asarray(train_raw["user_id"]),
        "train_date": np.asarray(train_date, dtype=np.int64),
        "train_hourmin": np.asarray(train_hourmin),
        "val_x": (np.stack(val_columns, axis=1) + offsets[None, :]).astype(np.int64),
        "val_y": np.asarray(val_y, dtype=np.float32),
        "val_user": np.asarray(val_users),
        "val_user_eval": np.asarray(val_users),
        "val_video_out": np.asarray(val_videos),
        "val_date": np.asarray(val_date, dtype=np.int64),
        "val_hourmin": np.asarray(val_hourmin),
        "field_dims": np.asarray(dims, dtype=np.int64),
        "fast": False,
    }


def load_data(data_dir):
    train_path = os.path.join(data_dir, "train.npz")
    val_path = os.path.join(data_dir, "val.npz")
    if not (os.path.exists(train_path) and os.path.exists(val_path)):
        return load_csv_data(data_dir)
    with np.load(train_path, allow_pickle=False) as train_file:
        train_x = np.asarray(train_file["X"], dtype=np.int64)
        train_y = np.asarray(train_file["y"], dtype=np.float32)
        train_user = np.asarray(train_file["user"])
        train_date = np.asarray(train_file["date"]) if "date" in train_file.files else np.zeros(len(train_y), dtype=np.int64)
        train_hourmin = np.asarray(train_file["hourmin"]) if "hourmin" in train_file.files else np.full(len(train_y), -1)
        field_dims = np.asarray(train_file["field_dims"], dtype=np.int64) if "field_dims" in train_file.files else None
    with np.load(val_path, allow_pickle=False) as val_file:
        val_x = np.asarray(val_file["X"], dtype=np.int64)
        val_y = np.asarray(val_file["y"], dtype=np.float32)
        val_user = np.asarray(val_file["user"])
        val_date = np.asarray(val_file["date"]) if "date" in val_file.files else np.zeros(len(val_y), dtype=np.int64)
        val_hourmin = np.asarray(val_file["hourmin"]) if "hourmin" in val_file.files else np.full(len(val_y), -1)
    if field_dims is None:
        field_dims = np.ones(train_x.shape[1], dtype=np.int64)
    return {
        "train_x": train_x,
        "train_y": train_y,
        "train_user": train_user,
        "train_date": train_date,
        "train_hourmin": train_hourmin,
        "val_x": val_x,
        "val_y": val_y,
        "val_user": val_user,
        "val_user_eval": val_user,
        "val_video_out": val_x[:, 1],
        "val_date": val_date,
        "val_hourmin": val_hourmin,
        "field_dims": field_dims,
        "fast": True,
    }


def normalized_dates(values):
    return np.asarray([date_number(value) for value in values], dtype=np.int64)


def session_codes(users, dates, hourmins, initial_state=None, session_minutes=30):
    state = {} if initial_state is None else dict(initial_state)
    n = len(users)
    hour_code = np.full(n, 24, dtype=np.int64)
    weekday_code = np.full(n, 7, dtype=np.int64)
    gap_code = np.zeros(n, dtype=np.int64)
    position_code = np.zeros(n, dtype=np.int64)
    gap_edges = np.asarray([1, 5, 15, 30, 60, 180, 720], dtype=np.int64)
    position_edges = np.asarray([1, 2, 3, 5, 8, 16], dtype=np.int64)
    for index in range(n):
        user = users[index].item() if hasattr(users[index], "item") else users[index]
        day = int(dates[index])
        minute = minute_of_day(hourmins[index])
        if day > 0:
            try:
                weekday_code[index] = datetime.date.fromordinal(day).weekday()
            except Exception:
                pass
        if minute >= 0:
            hour_code[index] = minute // 60
        if day <= 0 or minute < 0:
            continue
        timestamp = day * 1440 + minute
        previous = state.get(user)
        if previous is None:
            position = 0
        else:
            previous_timestamp, previous_position = previous
            gap = timestamp - previous_timestamp
            if gap < 0:
                gap_code[index] = 9
                position = 0
            else:
                gap_code[index] = 1 + int(np.searchsorted(gap_edges, gap, side="right"))
                position = previous_position + 1 if gap <= session_minutes else 0
        position_code[index] = int(np.searchsorted(position_edges, position, side="right"))
        state[user] = (timestamp, position)
    return hour_code, weekday_code, gap_code, position_code, state


def add_session_time_features(data):
    train_dates = normalized_dates(data["train_date"])
    val_dates = normalized_dates(data["val_date"])
    train_hour, train_weekday, train_gap, train_position, state = session_codes(
        np.asarray(data["train_user"]), train_dates, np.asarray(data["train_hourmin"])
    )
    val_hour, val_weekday, val_gap, val_position, _ = session_codes(
        np.asarray(data["val_user_eval"]), val_dates, np.asarray(data["val_hourmin"]), state
    )
    field_dims = np.asarray(data["field_dims"], dtype=np.int64)
    if data["train_x"].shape[1] > 3 and field_dims.size > 3:
        tab_offset = int(field_dims[:3].sum())
        tab_dim = max(1, int(field_dims[3]))
        train_tab = np.clip(data["train_x"][:, 3] - tab_offset, 0, tab_dim - 1)
        val_tab = np.clip(data["val_x"][:, 3] - tab_offset, 0, tab_dim - 1)
    else:
        tab_dim = 1
        train_tab = np.zeros(len(data["train_x"]), dtype=np.int64)
        val_tab = np.zeros(len(data["val_x"]), dtype=np.int64)
    train_local = [train_hour, train_weekday, train_gap, train_position, train_hour * tab_dim + train_tab, train_weekday * tab_dim + train_tab]
    val_local = [val_hour, val_weekday, val_gap, val_position, val_hour * tab_dim + val_tab, val_weekday * tab_dim + val_tab]
    new_dims = [25, 8, 10, 7, 25 * tab_dim, 8 * tab_dim]
    next_offset = int(max(data["train_x"].max(initial=0), data["val_x"].max(initial=0)) + 1)
    train_columns, val_columns = [data["train_x"]], [data["val_x"]]
    for train_values, val_values, dim in zip(train_local, val_local, new_dims):
        train_columns.append(np.asarray(train_values, dtype=np.int64)[:, None] + next_offset)
        val_columns.append(np.asarray(val_values, dtype=np.int64)[:, None] + next_offset)
        next_offset += int(dim)
    data["train_x"] = np.concatenate(train_columns, axis=1)
    data["val_x"] = np.concatenate(val_columns, axis=1)
    data["field_dims"] = np.concatenate([field_dims, np.asarray(new_dims, dtype=np.int64)])
    data["train_day"] = train_dates
    data["session_feature_names"] = ["hour", "weekday", "causal_gap", "causal_session_position", "hour_x_tab", "weekday_x_tab"]
    return data


def fallback_metrics(data, scores):
    labels = np.asarray(data["val_y"], dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    users = np.asarray(data["val_user_eval"])
    _, inverse = np.unique(users, return_inverse=True)
    auc_sum = auc_weight = ndcg_sum = 0.0
    ndcg_count = 0
    for group in range(inverse.max(initial=-1) + 1):
        indices = np.flatnonzero(inverse == group)
        group_y, group_s = labels[indices], scores[indices]
        order = np.argsort(-group_s, kind="stable")
        top_y = group_y[order[:5]]
        dcg = np.sum(top_y / np.log2(np.arange(2, len(top_y) + 2)))
        ideal = np.sort(group_y)[::-1][:5]
        idcg = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))
        if idcg > 0:
            ndcg_sum += float(dcg / idcg)
            ndcg_count += 1
        positive = group_y > 0.5
        npos, nneg = int(positive.sum()), int((~positive).sum())
        if npos and nneg:
            order2 = np.argsort(group_s, kind="stable")
            ranks = np.empty(len(group_s), dtype=np.float64)
            ranks[order2] = np.arange(1, len(group_s) + 1)
            auc = (ranks[positive].sum() - npos * (npos + 1) / 2) / (npos * nneg)
            auc_sum += float(auc) * len(indices)
            auc_weight += len(indices)
    gauc = auc_sum / auc_weight if auc_weight else 0.0
    ndcg5 = ndcg_sum / ndcg_count if ndcg_count else 0.0
    return {"gauc": float(gauc), "ndcg5": float(ndcg5), "primary": float(0.5 * (gauc + ndcg5))}


def official_metrics(data, scores):
    try:
        if data["fast"]:
            from data.official.evaluate import evaluate
        else:
            from harness.evaluate_provisional import evaluate
        result = evaluate(data["val_user_eval"], data["val_y"], np.asarray(scores, dtype=np.float64))
        return {
            "gauc": float(result.get("GAUC", result.get("gauc", 0.0))),
            "ndcg5": float(result.get("nDCG@5", result.get("ndcg5", 0.0))),
            "primary": float(result["primary"]),
        }
    except Exception:
        return fallback_metrics(data, scores)


class AdversarialModel(nn.Module):
    def __init__(self, total_features):
        super().__init__()
        self.linear = nn.Embedding(total_features, 1)
        nn.init.zeros_(self.linear.weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.linear(x).sum(dim=1).squeeze(-1) + self.bias


class RankModel(nn.Module):
    def __init__(self, total_features, fields, k, architecture, dropout):
        super().__init__()
        self.architecture = architecture
        self.embedding = nn.Embedding(total_features, k)
        self.linear = nn.Embedding(total_features, 1)
        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.zeros_(self.linear.weight)
        self.bias = nn.Parameter(torch.zeros(1))
        if architecture == "dcn-lite":
            width = fields * k
            self.cross_w = nn.ParameterList([nn.Parameter(torch.empty(width)) for _ in range(2)])
            self.cross_b = nn.ParameterList([nn.Parameter(torch.zeros(width)) for _ in range(2)])
            for weight in self.cross_w:
                nn.init.normal_(weight, std=0.01)
            self.mlp = nn.Sequential(
                nn.Linear(width, 128), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1),
            )
            self.cross_out = nn.Linear(width, 1)

    def forward(self, x):
        embeddings = self.embedding(x)
        linear = self.linear(x).sum(dim=1).squeeze(-1) + self.bias
        summed = embeddings.sum(dim=1)
        score = linear + 0.5 * (summed.square() - embeddings.square().sum(dim=1)).sum(dim=1)
        if self.architecture == "dcn-lite":
            x0 = embeddings.reshape(embeddings.shape[0], -1)
            crossed = x0
            for weight, bias in zip(self.cross_w, self.cross_b):
                crossed = x0 * torch.sum(crossed * weight, dim=1, keepdim=True) + bias + crossed
            score = score + self.cross_out(crossed).squeeze(-1) + self.mlp(x0).squeeze(-1)
        return score


def fit_adversarial_probabilities(data, seed, epochs, device):
    dates = data["train_day"]
    valid_dates = np.unique(dates[dates > 0])
    if valid_dates.size < 3:
        return np.full(len(dates), 0.5, dtype=np.float32)
    early_boundary = valid_dates[max(0, int(np.floor(valid_dates.size / 3.0)) - 1)]
    late_boundary = valid_dates[min(valid_dates.size - 1, int(np.ceil(2.0 * valid_dates.size / 3.0)))]
    early = np.flatnonzero((dates > 0) & (dates <= early_boundary))
    late = np.flatnonzero(dates >= late_boundary)
    if early.size == 0 or late.size == 0:
        return np.full(len(dates), 0.5, dtype=np.float32)
    rng = np.random.default_rng(seed + 701)
    per_class = min(max(early.size, late.size), 20000)
    early_sample = rng.choice(early, size=per_class, replace=early.size < per_class)
    late_sample = rng.choice(late, size=per_class, replace=late.size < per_class)
    indices = np.concatenate([early_sample, late_sample])
    targets = np.concatenate([np.zeros(per_class, dtype=np.float32), np.ones(per_class, dtype=np.float32)])
    total_features = int(max(data["train_x"].max(initial=0), data["val_x"].max(initial=0)) + 1)
    set_seed(seed + 702)
    model = AdversarialModel(total_features).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-5)
    for _ in range(max(1, min(int(epochs), 1))):
        permutation = rng.permutation(len(indices))
        model.train()
        for start in range(0, len(indices), 16384):
            selected = permutation[start:start + 16384]
            rows = indices[selected]
            xb = torch.as_tensor(data["train_x"][rows], dtype=torch.long, device=device)
            yb = torch.as_tensor(targets[selected], dtype=torch.float32, device=device)
            loss = F.binary_cross_entropy_with_logits(model(xb), yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(data["train_x"]), 32768):
            xb = torch.as_tensor(data["train_x"][start:start + 32768], dtype=torch.long, device=device)
            outputs.append(torch.sigmoid(model(xb)).cpu().numpy())
    result = np.concatenate(outputs).astype(np.float32)
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def adversarial_weights(probabilities, alpha, cap):
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 0.02, 0.98)
    log_odds = np.log(probabilities) - np.log1p(-probabilities)
    log_odds -= np.median(log_odds)
    weights = np.exp(float(alpha) * np.clip(log_odds, -8.0, 8.0))
    weights = np.clip(weights, 1.0 / float(cap), float(cap))
    weights /= max(float(weights.mean()), 1e-8)
    return weights.astype(np.float32)


def build_temporal_pairs(users, labels, days, seed, local_fraction=0.7, bandwidth=2.0, local_window=3):
    rng = np.random.default_rng(seed)
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    positive_parts, negative_parts = [], []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        indices = order[left:right]
        positive = indices[labels[indices] > 0.5]
        negative = indices[labels[indices] <= 0.5]
        if positive.size == 0 or negative.size == 0:
            continue
        chosen = np.empty(positive.size, dtype=np.int64)
        negative_days = days[negative]
        for offset, positive_index in enumerate(positive):
            use_local = rng.random() < local_fraction and days[positive_index] > 0
            if use_local:
                distances = np.abs(negative_days - days[positive_index])
                mask = (negative_days > 0) & (distances <= local_window)
                candidates = negative[mask]
                if candidates.size:
                    probabilities = np.exp(-distances[mask].astype(np.float64) / bandwidth)
                    probabilities /= probabilities.sum()
                    chosen[offset] = rng.choice(candidates, p=probabilities)
                    continue
            chosen[offset] = negative[rng.integers(negative.size)]
        positive_parts.append(positive)
        negative_parts.append(chosen)
    if not positive_parts:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    return np.concatenate(positive_parts), np.concatenate(negative_parts)


def predict(model, x, device, batch_size=16384):
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.as_tensor(x[start:start + batch_size], dtype=torch.long, device=device)
            outputs.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float32)


def train_candidate(data, config, seed, epochs, device, keep_predictions=False):
    set_seed(seed)
    x, y = data["train_x"], data["train_y"]
    users, days = np.asarray(data["train_user"]), data["train_day"]
    total_features = int(max(x.max(initial=0), data["val_x"].max(initial=0)) + 1)
    model = RankModel(total_features, x.shape[1], config["k"], config["architecture"], config["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    weights = adversarial_weights(data["adversarial_probability"], config["adversarial_alpha"], config["adversarial_cap"])
    hybrid = config["loss"] == "bpr-hybrid"
    batch_size = int(config["batch_size"])
    rng = np.random.default_rng(seed + 1009)
    best_metric, best_state, stale = None, None, 0
    for epoch in range(epochs):
        model.train()
        permutation = rng.permutation(len(x))
        if hybrid:
            pair_pos, pair_neg = build_temporal_pairs(
                users, y, days, seed + epoch * 7919,
                config["temporal_local_fraction"], config["temporal_bandwidth_days"], config["temporal_window_days"]
            )
            if pair_pos.size:
                pair_order = rng.permutation(pair_pos.size)
                pair_pos, pair_neg = pair_pos[pair_order], pair_neg[pair_order]
        else:
            pair_pos = pair_neg = np.empty(0, dtype=np.int64)
        for start in range(0, len(x), batch_size):
            indices = permutation[start:start + batch_size]
            xb = torch.as_tensor(x[indices], dtype=torch.long, device=device)
            yb = torch.as_tensor(y[indices], dtype=torch.float32, device=device)
            wb = torch.as_tensor(weights[indices], dtype=torch.float32, device=device)
            point_loss = (F.binary_cross_entropy_with_logits(model(xb), yb, reduction="none") * wb).mean()
            if hybrid and pair_pos.size:
                offset = start % pair_pos.size
                take = min(batch_size, pair_pos.size)
                positions = np.arange(offset, offset + take)
                positive_indices = np.take(pair_pos, positions, mode="wrap")
                negative_indices = np.take(pair_neg, positions, mode="wrap")
                xp = torch.as_tensor(x[positive_indices], dtype=torch.long, device=device)
                xn = torch.as_tensor(x[negative_indices], dtype=torch.long, device=device)
                pair_weights = np.sqrt(weights[positive_indices] * weights[negative_indices])
                pair_weights_t = torch.as_tensor(pair_weights, dtype=torch.float32, device=device)
                bpr_loss = (F.softplus(-(model(xp) - model(xn))) * pair_weights_t).mean()
                loss = 0.5 * point_loss + 0.5 * bpr_loss
            else:
                loss = point_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        scores = predict(model, data["val_x"], device)
        metric = official_metrics(data, scores)
        if best_metric is None or metric["gauc"] > best_metric["gauc"] + 1e-12:
            best_metric = metric
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 3:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    final_scores = predict(model, data["val_x"], device) if keep_predictions else None
    if keep_predictions:
        best_metric = official_metrics(data, final_scores)
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_metric, final_scores


def base_config(architecture, loss, alpha, cap):
    return {
        "architecture": architecture,
        "loss": loss,
        "weighting": "adversarial-recency",
        "adversarial_alpha": float(alpha),
        "adversarial_cap": float(cap),
        "regularization": "mild",
        "features": "parent-plus-causal-session-time",
        "pair_sampling": "temporal-pair-kernel",
        "temporal_local_fraction": 0.7,
        "temporal_bandwidth_days": 2.0,
        "temporal_window_days": 3,
        "pairs_per_positive": 1,
        "pair_weight": "sqrt-row-weight-product",
        "k": 24,
        "lr": 0.00168,
        "dropout": 0.21,
        "weight_decay": 0.000037,
        "batch_size": 8192,
    }


def append_progress(path, record):
    with open(path, "a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def scalar(value):
    return value.item() if hasattr(value, "item") else value


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = add_session_time_features(load_data(args.data_dir))

    smoke_value = os.environ.get("SMOKE_EPOCHS")
    smoke_cap = int(smoke_value) if smoke_value is not None else None

    # Keep the pipeline but make it fast enough for the execution limit.
    adversarial_epochs, probe_epochs, refine_epochs, final_epochs = 1, 1, 1, 2
    if smoke_cap is not None:
        adversarial_epochs = min(adversarial_epochs, smoke_cap)
        probe_epochs = min(probe_epochs, smoke_cap)
        refine_epochs = min(refine_epochs, smoke_cap)
        final_epochs = min(final_epochs, smoke_cap)

    data["adversarial_probability"] = fit_adversarial_probabilities(data, args.seed, adversarial_epochs, device)

    progress_path = os.path.join(args.out_dir, "progress.log")
    history = []

    cells = [base_config("dcn-lite", "bpr-hybrid", 0.5, 4.0)]
    if smoke_cap is None and len(data["train_x"]) > 0:
        cells.append(base_config("FM", "logloss", 0.2, 2.0))

    first_stage = []
    for cell_id, config in enumerate(cells):
        run_seed = args.seed + 37 * cell_id
        metric, _ = train_candidate(data, config, run_seed, probe_epochs, device, False)
        record = {"phase": "temporal_pair_kernel_search", "cell": cell_id, "seed": run_seed, "config": config, **metric}
        history.append(record)
        append_progress(progress_path, record)
        first_stage.append((metric["primary"], config))

    first_stage.sort(key=lambda item: item[0], reverse=True)
    winner = dict(first_stage[0][1])

    if smoke_cap is None and len(first_stage) > 1:
        refinement = []
        for rank, (_, config) in enumerate(first_stage[:2]):
            values = []
            for repeat in range(1):
                run_seed = args.seed + 20000 + rank * 1000 + repeat * 101
                metric, _ = train_candidate(data, config, run_seed, refine_epochs, device, False)
                record = {"phase": "temporal_pair_kernel_refine", "rank": rank, "repeat": repeat, "seed": run_seed, "config": config, **metric}
                history.append(record)
                append_progress(progress_path, record)
                values.append(metric["primary"])
            refinement.append((float(np.mean(values)), float(np.std(values)), config))
        refinement.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        winner = dict(refinement[0][2])

    final_metric, final_scores = train_candidate(data, winner, args.seed, final_epochs, device, True)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for row_id, (user, video, score) in enumerate(zip(data["val_user"], data["val_video_out"], final_scores)):
            writer.writerow([row_id, scalar(user), scalar(video), float(score)])

    output = {
        "gauc": final_metric["gauc"],
        "ndcg5": final_metric["ndcg5"],
        "primary": final_metric["primary"],
        "selected_config": winner,
        "session_features": data["session_feature_names"],
        "history": history,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as handle:
        json.dump(output, handle, sort_keys=True)


if __name__ == "__main__":
    main()
