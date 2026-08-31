import argparse
import csv
import datetime
import json
import math
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


def duration_bucket(values, edges=None):
    values = np.asarray(values, dtype=np.float64)
    if edges is None:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            edges = np.arange(1, 10, dtype=np.float64)
        else:
            edges = np.unique(np.quantile(finite, np.linspace(0.1, 0.9, 9)))
    return np.searchsorted(edges, values, side="right").astype(np.int64), np.asarray(edges)


def load_csv_data(data_dir):
    feature_names = ["user_id", "video_id", "author_id", "tab"]
    train_raw = {name: [] for name in feature_names}
    train_duration = []
    train_y = []
    train_date = []

    with open(os.path.join(data_dir, "train.csv"), "r", newline="") as handle:
        reader = csv.DictReader(handle)
        names = set(reader.fieldnames or [])
        for row in reader:
            train_raw["user_id"].append(row.get("user_id", ""))
            train_raw["video_id"].append(row.get("video_id", ""))
            train_raw["author_id"].append(row.get("author_id", "__missing__") if "author_id" in names else "__missing__")
            train_raw["tab"].append(row.get("tab", ""))
            train_duration.append(float(row.get("duration_ms", 0) or 0))
            train_y.append(float(row["long_view"]))
            train_date.append(date_number(row.get("date", 0)))

    val_raw = {name: [] for name in feature_names}
    val_duration = []
    val_y = []
    val_users_out = []
    val_videos_out = []

    with open(os.path.join(data_dir, "val.csv"), "r", newline="") as handle:
        reader = csv.DictReader(handle)
        names = set(reader.fieldnames or [])
        for row in reader:
            user = row.get("user_id", "")
            video = row.get("video_id", "")
            val_raw["user_id"].append(user)
            val_raw["video_id"].append(video)
            val_raw["author_id"].append(row.get("author_id", "__missing__") if "author_id" in names else "__missing__")
            val_raw["tab"].append(row.get("tab", ""))
            val_duration.append(float(row.get("duration_ms", 0) or 0))
            val_y.append(float(row["long_view"]))
            val_users_out.append(user)
            val_videos_out.append(video)

    train_duration_bucket, edges = duration_bucket(train_duration)
    val_duration_bucket, _ = duration_bucket(val_duration, edges)
    train_columns = []
    val_columns = []
    field_dims = []

    for name in feature_names:
        mapping = {}
        encoded_train = np.empty(len(train_y), dtype=np.int64)
        for index, value in enumerate(train_raw[name]):
            if value not in mapping:
                mapping[value] = len(mapping) + 1
            encoded_train[index] = mapping[value]
        encoded_val = np.asarray([mapping.get(value, 0) for value in val_raw[name]], dtype=np.int64)
        train_columns.append(encoded_train)
        val_columns.append(encoded_val)
        field_dims.append(len(mapping) + 1)

    train_columns.append(train_duration_bucket)
    val_columns.append(val_duration_bucket)
    field_dims.append(max(10, int(train_duration_bucket.max()) + 1 if train_duration_bucket.size else 10))
    offsets = np.cumsum([0] + field_dims[:-1], dtype=np.int64)
    train_x = np.stack(train_columns, axis=1) + offsets[None, :]
    val_x = np.stack(val_columns, axis=1) + offsets[None, :]

    return {
        "train_x": train_x.astype(np.int64),
        "train_y": np.asarray(train_y, dtype=np.float32),
        "train_user": train_x[:, 0].astype(np.int64),
        "train_date": np.asarray(train_date, dtype=np.int64),
        "val_x": val_x.astype(np.int64),
        "val_y": np.asarray(val_y, dtype=np.float32),
        "val_user": np.asarray(val_users_out),
        "val_user_eval": np.asarray(val_users_out),
        "val_video_out": np.asarray(val_videos_out),
        "field_dims": np.asarray(field_dims, dtype=np.int64),
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
        field_dims = np.asarray(train_file["field_dims"], dtype=np.int64) if "field_dims" in train_file.files else None

    with np.load(val_path, allow_pickle=False) as val_file:
        val_x = np.asarray(val_file["X"], dtype=np.int64)
        val_y = np.asarray(val_file["y"], dtype=np.float32)
        val_user = np.asarray(val_file["user"])

    if field_dims is None:
        field_dims = np.ones(train_x.shape[1], dtype=np.int64)

    return {
        "train_x": train_x,
        "train_y": train_y,
        "train_user": train_user,
        "train_date": train_date,
        "val_x": val_x,
        "val_y": val_y,
        "val_user": val_user,
        "val_user_eval": val_user,
        "val_video_out": val_x[:, 1],
        "field_dims": field_dims,
        "fast": True,
    }


def fallback_metrics(data, scores):
    labels = np.asarray(data["val_y"], dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    users = np.asarray(data["val_user_eval"])
    _, inverse = np.unique(users, return_inverse=True)
    auc_sum = 0.0
    auc_weight = 0.0
    ndcg_sum = 0.0
    ndcg_count = 0

    max_group = int(inverse.max()) if inverse.size else -1
    for group in range(max_group + 1):
        indices = np.flatnonzero(inverse == group)
        if indices.size < 2:
            continue
        y = labels[indices]
        s = scores[indices]
        positive = y > 0.5
        negative = ~positive
        n_positive = int(positive.sum())
        n_negative = int(negative.sum())
        if n_positive == 0 or n_negative == 0:
            continue

        order = np.argsort(-s, kind="stable")
        ranked = y[order]
        top = ranked[:5]
        discounts = np.log2(np.arange(2, len(top) + 2))
        dcg = float(np.sum((2.0 ** top - 1.0) / discounts))
        ideal = np.sort(y)[::-1][:5]
        ideal_discounts = np.log2(np.arange(2, len(ideal) + 2))
        idcg = float(np.sum((2.0 ** ideal - 1.0) / ideal_discounts))
        ndcg_sum += dcg / idcg if idcg > 0 else 0.0
        ndcg_count += 1

        rank_order = np.argsort(s, kind="stable")
        ranks = np.empty(len(s), dtype=np.float64)
        ranks[rank_order] = np.arange(1, len(s) + 1, dtype=np.float64)
        auc = (ranks[positive].sum() - n_positive * (n_positive + 1.0) / 2.0) / (n_positive * n_negative)
        auc_sum += auc * len(indices)
        auc_weight += len(indices)

    gauc = auc_sum / auc_weight if auc_weight > 0 else 0.0
    ndcg5 = ndcg_sum / ndcg_count if ndcg_count > 0 else 0.0
    return {"gauc": float(gauc), "ndcg5": float(ndcg5), "primary": float((gauc + ndcg5) / 2.0)}


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


class RankModel(nn.Module):
    def __init__(self, total_features, fields, k, architecture, dropout):
        super().__init__()
        self.architecture = architecture
        self.embedding = nn.Embedding(total_features, k)
        self.linear = nn.Embedding(total_features, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.zeros_(self.linear.weight)

        if architecture == "dcn-lite":
            width = fields * k
            self.cross_w = nn.ParameterList([nn.Parameter(torch.empty(width)) for _ in range(2)])
            self.cross_b = nn.ParameterList([nn.Parameter(torch.zeros(width)) for _ in range(2)])
            for weight in self.cross_w:
                nn.init.normal_(weight, std=0.01)
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

    def forward(self, x):
        embeddings = self.embedding(x)
        linear = self.linear(x).sum(dim=1).squeeze(-1)
        summed = embeddings.sum(dim=1)
        fm = 0.5 * (summed.square() - embeddings.square().sum(dim=1)).sum(dim=1)
        score = linear + fm + self.bias
        if self.architecture == "dcn-lite":
            x0 = embeddings.reshape(embeddings.shape[0], -1)
            crossed = x0
            for weight, bias in zip(self.cross_w, self.cross_b):
                crossed = x0 * torch.sum(crossed * weight, dim=1, keepdim=True) + bias + crossed
            score = score + self.cross_out(crossed).squeeze(-1) + self.mlp(x0).squeeze(-1)
        return score


def recency_weights(dates, enabled):
    if not enabled:
        return np.ones(len(dates), dtype=np.float32)
    converted = np.asarray([date_number(value) for value in dates], dtype=np.int64)
    valid = converted[converted > 0]
    if valid.size == 0:
        return np.ones(len(converted), dtype=np.float32)
    age = np.maximum(0, int(valid.max()) - converted)
    weights = np.exp(-math.log(2.0) * age / 7.0)
    weights[converted <= 0] = 1.0
    weights /= max(float(weights.mean()), 1e-8)
    return weights.astype(np.float32)


def build_pairs(users, labels, seed):
    rng = np.random.default_rng(seed)
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    positive_parts = []
    negative_parts = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        indices = order[left:right]
        positive = indices[labels[indices] > 0.5]
        negative = indices[labels[indices] <= 0.5]
        if len(positive) and len(negative):
            count = max(len(positive), len(negative))
            positive_parts.append(rng.choice(positive, size=count, replace=len(positive) < count))
            negative_parts.append(rng.choice(negative, size=count, replace=len(negative) < count))
    if not positive_parts:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    return np.concatenate(positive_parts), np.concatenate(negative_parts)


def make_complete_user_batches(users, target_size, rng):
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    groups = [order[left:right] for left, right in zip(boundaries[:-1], boundaries[1:])]
    group_order = rng.permutation(len(groups))
    batches = []
    current = []
    current_size = 0
    for group_index in group_order:
        group = groups[int(group_index)]
        if current and current_size + len(group) > target_size:
            batches.append(np.concatenate(current))
            current = []
            current_size = 0
        current.append(group)
        current_size += len(group)
        if current_size >= target_size:
            batches.append(np.concatenate(current))
            current = []
            current_size = 0
    if current:
        batches.append(np.concatenate(current))
    return batches


def centered_bce_logits(logits, users, global_bias):
    _, inverse = torch.unique(users, sorted=False, return_inverse=True)
    group_count = int(inverse.max().item()) + 1
    sums = torch.zeros(group_count, dtype=logits.dtype, device=logits.device)
    counts = torch.zeros(group_count, dtype=logits.dtype, device=logits.device)
    sums.scatter_add_(0, inverse, logits)
    counts.scatter_add_(0, inverse, torch.ones_like(logits))
    means = sums / counts.clamp_min(1.0)
    return logits - means[inverse] + global_bias


def predict(model, x, device, batch_size=65536):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.as_tensor(x[start:start + batch_size], dtype=torch.long, device=device)
            outputs.append(torch.sigmoid(model(batch)).cpu().numpy())
    return np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float32)


def train_candidate(data, config, seed, epochs, device, keep_predictions=False):
    set_seed(seed)
    x = data["train_x"]
    y = data["train_y"]
    users = np.asarray(data["train_user"])
    train_max = int(x.max()) if x.size else 0
    val_max = int(data["val_x"].max()) if data["val_x"].size else 0
    total_features = max(train_max, val_max) + 1
    model = RankModel(total_features, x.shape[1], config["k"], config["architecture"], config["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    weights = recency_weights(data["train_date"], config["weighting"] == "recency-7d")
    hybrid = config["loss"] == "bpr-hybrid"
    batch_size = int(config.get("batch_size", 32768))
    rng = np.random.default_rng(seed + 1009)

    for epoch in range(epochs):
        model.train()
        batches = make_complete_user_batches(users, batch_size, rng)
        if hybrid:
            pair_positive, pair_negative = build_pairs(users, y, seed + epoch * 7919)
            if len(pair_positive):
                pair_order = rng.permutation(len(pair_positive))
                pair_positive = pair_positive[pair_order]
                pair_negative = pair_negative[pair_order]
        else:
            pair_positive = pair_negative = np.empty(0, dtype=np.int64)

        pair_cursor = 0
        for indices in batches:
            xb = torch.as_tensor(x[indices], dtype=torch.long, device=device)
            yb = torch.as_tensor(y[indices], dtype=torch.float32, device=device)
            wb = torch.as_tensor(weights[indices], dtype=torch.float32, device=device)
            ub = torch.as_tensor(users[indices], dtype=torch.long, device=device)
            raw_logits = model(xb)
            gauge_logits = centered_bce_logits(raw_logits, ub, model.bias)
            point = (F.binary_cross_entropy_with_logits(gauge_logits, yb, reduction="none") * wb).mean()

            if hybrid and len(pair_positive):
                take = min(len(indices), len(pair_positive))
                positions = np.arange(pair_cursor, pair_cursor + take)
                positive_indices = np.take(pair_positive, positions, mode="wrap")
                negative_indices = np.take(pair_negative, positions, mode="wrap")
                pair_cursor = (pair_cursor + take) % len(pair_positive)
                xp = torch.as_tensor(x[positive_indices], dtype=torch.long, device=device)
                xn = torch.as_tensor(x[negative_indices], dtype=torch.long, device=device)
                pair_weight = torch.as_tensor(
                    0.5 * (weights[positive_indices] + weights[negative_indices]),
                    dtype=torch.float32,
                    device=device,
                )
                bpr = (F.softplus(-(model(xp) - model(xn))) * pair_weight).mean()
                loss = 0.5 * point + 0.5 * bpr
            else:
                loss = point

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    scores = predict(model, data["val_x"], device)
    metric = official_metrics(data, scores)
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metric, scores if keep_predictions else None


def base_config(architecture, loss, weighting, regularization):
    strong = regularization == "strong"
    return {
        "architecture": architecture,
        "loss": loss,
        "weighting": weighting,
        "regularization": regularization,
        "point_objective": "user-centered-bce",
        "complete_user_slates": True,
        "k": 24,
        "lr": 0.00135 if strong else 0.00168,
        "dropout": 0.32 if strong else 0.21,
        "weight_decay": 0.0005 if strong else 0.000037,
        "batch_size": 32768,
    }


def append_progress(path, record):
    with open(path, "a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_data(args.data_dir)

    smoke_value = os.environ.get("SMOKE_EPOCHS")
    smoke_cap = int(smoke_value) if smoke_value is not None else None
    probe_epochs = 1 if smoke_cap is None else max(1, min(1, smoke_cap))
    final_epochs = 1 if smoke_cap is None else max(1, min(1, smoke_cap))

    history = []
    cells = [
        base_config(architecture, loss, "uniform", "mild")
        for architecture in ["FM", "dcn-lite"]
        for loss in ["logloss", "bpr-hybrid"]
    ]
    cell_results = []
    progress_path = os.path.join(args.out_dir, "progress.log")

    for cell_id, config in enumerate(cells):
        run_seed = args.seed + 37 * cell_id
        metric, _ = train_candidate(data, config, run_seed, probe_epochs, device, False)
        record = {"phase": "matrix", "cell": cell_id, "repeat": 0, "seed": run_seed, "config": config, **metric}
        history.append(record)
        append_progress(progress_path, record)
        cell_results.append((metric["primary"], 0.0, config))

    cell_results.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    winner = dict(cell_results[0][2])

    final_metric, final_scores = train_candidate(data, winner, args.seed, final_epochs, device, True)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for row_id, (user, video, score) in enumerate(zip(data["val_user"], data["val_video_out"], final_scores)):
            user_value = user.item() if hasattr(user, "item") else user
            video_value = video.item() if hasattr(video, "item") else video
            writer.writerow([row_id, user_value, video_value, float(score)])

    output = {
        "gauc": final_metric["gauc"],
        "ndcg5": final_metric["ndcg5"],
        "primary": final_metric["primary"],
        "selected_config": winner,
        "history": history,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as handle:
        json.dump(output, handle, sort_keys=True)


if __name__ == "__main__":
    main()
