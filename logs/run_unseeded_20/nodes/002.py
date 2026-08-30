import argparse
import csv
import json
import os
import random
import warnings
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def parse_date_value(value):
    text = str(value)
    if text.endswith(".0"):
        text = text[:-2]
    text = text.replace("-", "").replace("/", "")
    try:
        return datetime.strptime(text[:8], "%Y%m%d").toordinal()
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return 0


def recency_weights(dates):
    vals = np.asarray([parse_date_value(x) for x in dates], dtype=np.int64)
    if len(vals) == 0 or np.max(vals) == np.min(vals):
        return np.ones(len(vals), dtype=np.float32)
    latest = int(np.max(vals))
    delta = np.maximum(0, latest - vals).astype(np.float32)
    weights = np.exp2(-delta / 7.0).astype(np.float32)
    weights /= max(float(weights.mean()), 1e-6)
    return weights


def load_npz_data(data_dir):
    train_file = np.load(os.path.join(data_dir, "train.npz"), allow_pickle=False)
    val_file = np.load(os.path.join(data_dir, "val.npz"), allow_pickle=False)

    x_train = np.asarray(train_file["X"], dtype=np.int64)
    y_train = np.asarray(train_file["y"], dtype=np.float32).reshape(-1)
    train_users = np.asarray(train_file["user"]).reshape(-1)
    play_train = np.asarray(train_file["play_time_ms"], dtype=np.float32).reshape(-1)
    duration_train = np.asarray(train_file["duration_ms"], dtype=np.float32).reshape(-1)
    dates_train = np.asarray(train_file["date"]).reshape(-1)
    field_dims = np.asarray(train_file["field_dims"], dtype=np.int64).reshape(-1)

    x_val = np.asarray(val_file["X"], dtype=np.int64)
    y_val = np.asarray(val_file["y"], dtype=np.float32).reshape(-1)
    val_users = np.asarray(val_file["user"]).reshape(-1)

    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1]))).astype(np.int64)
    video_ids = x_val[:, 1] - offsets[1]
    row_ids = np.arange(len(x_val), dtype=np.int64)
    weights = recency_weights(dates_train)

    denom = np.maximum(np.minimum(duration_train, 18000.0), 1.0)
    ratio = np.clip(play_train / denom, 0.0, 1.0)
    thresholds = np.asarray([0.2, 0.4, 0.6, 0.8], dtype=np.float32)
    ordinal = (ratio[:, None] >= thresholds[None, :]).astype(np.float32)

    return {
        "x_train": x_train,
        "y_train": y_train,
        "users_train": train_users,
        "weights": weights,
        "ordinal": ordinal,
        "x_val": x_val,
        "y_val": y_val,
        "users_val": val_users,
        "video_val": video_ids,
        "row_val": row_ids,
        "field_dims": field_dims,
        "npz": True,
    }


def read_csv_rows(path, training):
    result = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_number, row in enumerate(reader):
            record = {
                "row_id": row.get("row_id", str(row_number)),
                "user_id": row.get("user_id", ""),
                "video_id": row.get("video_id", ""),
                "author_id": row.get("author_id", "__missing_author__"),
                "tab": row.get("tab", ""),
                "duration_ms": float(row.get("duration_ms", 0) or 0),
                "long_view": float(row.get("long_view", 0) or 0),
            }
            if training:
                record["play_time_ms"] = float(row.get("play_time_ms", 0) or 0)
                record["date"] = row.get("date", "0")
            result.append(record)
    return result


def load_csv_data(data_dir):
    train_rows = read_csv_rows(os.path.join(data_dir, "train.csv"), True)
    val_rows = read_csv_rows(os.path.join(data_dir, "val.csv"), False)

    train_duration = np.asarray([r["duration_ms"] for r in train_rows], dtype=np.float32)
    quantiles = np.linspace(0.1, 0.9, 9)
    duration_cuts = np.unique(np.quantile(train_duration, quantiles)) if len(train_duration) else np.asarray([])

    def raw_fields(rows):
        output = []
        for r in rows:
            dur_bucket = str(int(np.searchsorted(duration_cuts, r["duration_ms"], side="right")))
            output.append((r["user_id"], r["video_id"], r["author_id"], r["tab"], dur_bucket))
        return output

    train_fields = raw_fields(train_rows)
    val_fields = raw_fields(val_rows)
    maps = []
    field_dims = []
    for j in range(5):
        values = sorted({fields[j] for fields in train_fields})
        mapping = {value: i + 1 for i, value in enumerate(values)}
        maps.append(mapping)
        field_dims.append(len(mapping) + 1)
    field_dims = np.asarray(field_dims, dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1]))).astype(np.int64)

    def encode(fields):
        x = np.empty((len(fields), 5), dtype=np.int64)
        for i, values in enumerate(fields):
            for j in range(5):
                x[i, j] = maps[j].get(values[j], 0) + offsets[j]
        return x

    x_train = encode(train_fields)
    x_val = encode(val_fields)
    y_train = np.asarray([r["long_view"] for r in train_rows], dtype=np.float32)
    y_val = np.asarray([r["long_view"] for r in val_rows], dtype=np.float32)
    play = np.asarray([r["play_time_ms"] for r in train_rows], dtype=np.float32)
    denom = np.maximum(np.minimum(train_duration, 18000.0), 1.0)
    ratio = np.clip(play / denom, 0.0, 1.0)
    thresholds = np.asarray([0.2, 0.4, 0.6, 0.8], dtype=np.float32)
    ordinal = (ratio[:, None] >= thresholds[None, :]).astype(np.float32)
    weights = recency_weights([r["date"] for r in train_rows])

    return {
        "x_train": x_train,
        "y_train": y_train,
        "users_train": np.asarray([r["user_id"] for r in train_rows], dtype=object),
        "weights": weights,
        "ordinal": ordinal,
        "x_val": x_val,
        "y_val": y_val,
        "users_val": np.asarray([r["user_id"] for r in val_rows], dtype=object),
        "video_val": np.asarray([r["video_id"] for r in val_rows], dtype=object),
        "row_val": np.asarray([r["row_id"] for r in val_rows], dtype=object),
        "field_dims": field_dims,
        "npz": False,
    }


def prepare_pair_groups(users, labels):
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    if len(order) == 0:
        return []
    boundaries = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
    groups = np.split(order, boundaries)
    pair_groups = []
    for group in groups:
        positives = group[labels[group] > 0.5]
        negatives = group[labels[group] <= 0.5]
        if len(positives) and len(negatives):
            pair_groups.append((positives.astype(np.int64), negatives.astype(np.int64)))
    return pair_groups


def sample_pairs(pair_groups, rng):
    if not pair_groups:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    positives = []
    negatives = []
    for pos, neg in pair_groups:
        positives.append(pos)
        negatives.append(neg[rng.integers(0, len(neg), size=len(pos))])
    return np.concatenate(positives), np.concatenate(negatives)


class DCNLite(nn.Module):
    def __init__(self, field_dims, embed_dim=16, hidden_dim=128, dropout=0.3):
        super().__init__()
        total_dim = int(np.sum(field_dims))
        input_dim = len(field_dims) * embed_dim
        self.embedding = nn.Embedding(total_dim, embed_dim)
        self.cross_weights = nn.ModuleList([nn.Linear(input_dim, 1, bias=False) for _ in range(2)])
        self.cross_biases = nn.ParameterList([nn.Parameter(torch.zeros(input_dim)) for _ in range(2)])
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        joined_dim = input_dim + hidden_dim // 2
        self.main_head = nn.Linear(joined_dim, 1)
        self.ordinal_head = nn.Linear(joined_dim, 4)
        nn.init.normal_(self.embedding.weight, std=0.01)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        x0 = self.embedding(x).flatten(1)
        cross = x0
        for weight, bias in zip(self.cross_weights, self.cross_biases):
            cross = x0 * weight(cross) + bias + cross
        deep = self.mlp(x0)
        joined = torch.cat((cross, deep), dim=1)
        return self.main_head(joined).squeeze(1), self.ordinal_head(joined)


def predict(model, x, device, batch_size):
    model.eval()
    scores = np.empty(len(x), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            xb = torch.as_tensor(x[start:end], dtype=torch.long, device=device)
            logits, _ = model(xb)
            scores[start:end] = torch.sigmoid(logits).cpu().numpy()
    return scores


def official_metrics(data, scores):
    if data["npz"]:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    result = evaluate(data["users_val"], data["y_val"], scores)
    return {
        "gauc": float(result["GAUC"]),
        "ndcg5": float(result["nDCG@5"]),
        "primary": float(result["primary"]),
    }


def train_model(data, seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4096 if device.type == "cuda" else 2048
    epochs = 8
    smoke = os.environ.get("SMOKE_EPOCHS")
    if smoke is not None:
        epochs = min(epochs, max(1, int(smoke)))

    model = DCNLite(data["field_dims"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    x = data["x_train"]
    y = data["y_train"]
    ordinal = data["ordinal"]
    weights = data["weights"]
    n = len(x)
    pair_groups = prepare_pair_groups(data["users_train"], y)
    rng = np.random.default_rng(seed)

    best_state = None
    best_gauc = -float("inf")
    stale = 0

    for epoch in range(epochs):
        model.train()
        order = rng.permutation(n)
        pair_pos, pair_neg = sample_pairs(pair_groups, rng)
        if len(pair_pos):
            pair_order = rng.permutation(len(pair_pos))
            pair_pos = pair_pos[pair_order]
            pair_neg = pair_neg[pair_order]

        for start in range(0, n, batch_size):
            main_idx = order[start:min(start + batch_size, n)]
            count = len(main_idx)
            if len(pair_pos):
                positions = np.arange(start, start + count, dtype=np.int64) % len(pair_pos)
                pos_idx = pair_pos[positions]
                neg_idx = pair_neg[positions]
                all_idx = np.concatenate((main_idx, pos_idx, neg_idx))
            else:
                pos_idx = None
                all_idx = main_idx

            xb = torch.as_tensor(x[all_idx], dtype=torch.long, device=device)
            logits, ordinal_logits = model(xb)
            main_logits = logits[:count]
            main_ordinal_logits = ordinal_logits[:count]
            yb = torch.as_tensor(y[main_idx], dtype=torch.float32, device=device)
            ob = torch.as_tensor(ordinal[main_idx], dtype=torch.float32, device=device)
            wb = torch.as_tensor(weights[main_idx], dtype=torch.float32, device=device)

            bce_each = F.binary_cross_entropy_with_logits(main_logits, yb, reduction="none")
            bce_loss = (bce_each * wb).sum() / wb.sum().clamp_min(1e-6)
            ordinal_each = F.binary_cross_entropy_with_logits(main_ordinal_logits, ob, reduction="none").mean(dim=1)
            ordinal_loss = (ordinal_each * wb).sum() / wb.sum().clamp_min(1e-6)

            if pos_idx is not None:
                pos_logits = logits[count:2 * count]
                neg_logits = logits[2 * count:3 * count]
                pair_weights = torch.as_tensor(weights[pos_idx], dtype=torch.float32, device=device)
                bpr_each = F.softplus(-(pos_logits - neg_logits))
                bpr_loss = (bpr_each * pair_weights).sum() / pair_weights.sum().clamp_min(1e-6)
                base_loss = 0.5 * bce_loss + 0.5 * bpr_loss
            else:
                base_loss = bce_loss

            loss = base_loss + 0.3 * ordinal_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        scores = predict(model, data["x_val"], device, batch_size * 2)
        metrics = official_metrics(data, scores)
        current_gauc = metrics["gauc"]
        if current_gauc > best_gauc + 1e-7:
            best_gauc = current_gauc
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        scheduler.step()
        if stale >= 2:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, device, batch_size


def write_outputs(out_dir, data, scores, metrics):
    os.makedirs(out_dir, exist_ok=True)
    prediction_path = os.path.join(out_dir, "predictions.csv")
    with open(prediction_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for row_id, user_id, video_id, score in zip(
            data["row_val"], data["users_val"], data["video_val"], scores
        ):
            writer.writerow([row_id, user_id, video_id, format(float(score), ".10g")])
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, separators=(",", ":"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    fast_train = os.path.join(args.data_dir, "train.npz")
    fast_val = os.path.join(args.data_dir, "val.npz")
    if os.path.exists(fast_train) and os.path.exists(fast_val):
        data = load_npz_data(args.data_dir)
    else:
        data = load_csv_data(args.data_dir)

    model, device, batch_size = train_model(data, args.seed)
    scores = predict(model, data["x_val"], device, batch_size * 2)
    metrics = official_metrics(data, scores)
    write_outputs(args.out_dir, data, scores, metrics)


if __name__ == "__main__":
    main()
