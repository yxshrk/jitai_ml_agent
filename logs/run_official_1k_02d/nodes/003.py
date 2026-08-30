import argparse
import csv
import json
import os
import random
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def duration_buckets(values, edges):
    return np.searchsorted(edges, values, side="right").astype(np.int64)


def load_npz(data_dir):
    train_file = np.load(Path(data_dir) / "train.npz", allow_pickle=False)
    val_file = np.load(Path(data_dir) / "val.npz", allow_pickle=False)

    x_train = np.asarray(train_file["X"], dtype=np.int64)
    y_train = np.asarray(train_file["y"], dtype=np.float32)
    train_duration = np.asarray(train_file["duration_ms"], dtype=np.float32)
    train_date = np.asarray(train_file["date"])

    x_val = np.asarray(val_file["X"], dtype=np.int64)
    y_val = np.asarray(val_file["y"], dtype=np.float32)
    val_duration = np.asarray(val_file["duration_ms"], dtype=np.float32)
    val_users = np.asarray(val_file["user"])

    field_dims = np.asarray(train_file["field_dims"], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1]))).astype(np.int64)
    val_video = x_val[:, 1] - offsets[1]
    return {
        "x_train": x_train,
        "y_train": y_train,
        "duration_train": train_duration,
        "date_train": train_date,
        "x_val": x_val,
        "y_val": y_val,
        "duration_val": val_duration,
        "users_val": val_users,
        "videos_val": val_video,
        "field_dims": field_dims,
        "csv_mode": False,
    }


def read_csv_rows(path, training):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            item = {
                "user_id": row["user_id"],
                "video_id": row["video_id"],
                "author_id": row.get("author_id", "__missing_author__"),
                "tab": row["tab"],
                "duration_ms": float(row["duration_ms"]),
                "date": row["date"],
                "long_view": float(row["long_view"]),
            }
            rows.append(item)
    return rows


def load_csv(data_dir):
    train_rows = read_csv_rows(Path(data_dir) / "train.csv", True)
    val_rows = read_csv_rows(Path(data_dir) / "val.csv", False)

    train_duration = np.asarray([r["duration_ms"] for r in train_rows], dtype=np.float32)
    quantiles = np.linspace(0.1, 0.9, 9)
    edges = np.unique(np.quantile(train_duration, quantiles)).astype(np.float32)
    train_bucket = duration_buckets(train_duration, edges)
    val_duration = np.asarray([r["duration_ms"] for r in val_rows], dtype=np.float32)
    val_bucket = duration_buckets(val_duration, edges)

    field_names = ["user_id", "video_id", "author_id", "tab"]
    mappings = []
    for name in field_names:
        values = sorted({r[name] for r in train_rows})
        mappings.append({value: i for i, value in enumerate(values)})

    field_dims = np.asarray([len(m) + 1 for m in mappings] + [len(edges) + 1], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1]))).astype(np.int64)

    def encode(rows, buckets):
        result = np.empty((len(rows), 5), dtype=np.int64)
        for i, row in enumerate(rows):
            for j, name in enumerate(field_names):
                result[i, j] = mappings[j].get(row[name], len(mappings[j])) + offsets[j]
            result[i, 4] = int(buckets[i]) + offsets[4]
        return result

    return {
        "x_train": encode(train_rows, train_bucket),
        "y_train": np.asarray([r["long_view"] for r in train_rows], dtype=np.float32),
        "duration_train": train_duration,
        "date_train": np.asarray([r["date"] for r in train_rows]),
        "x_val": encode(val_rows, val_bucket),
        "y_val": np.asarray([r["long_view"] for r in val_rows], dtype=np.float32),
        "duration_val": val_duration,
        "users_val": np.asarray([r["user_id"] for r in val_rows], dtype=object),
        "videos_val": np.asarray([r["video_id"] for r in val_rows], dtype=object),
        "field_dims": field_dims,
        "csv_mode": True,
    }


def numeric_dates(values):
    result = np.zeros(len(values), dtype=np.float64)
    for i, value in enumerate(values):
        text = str(value)
        digits = "".join(ch for ch in text if ch.isdigit())
        result[i] = float(digits[:8]) if digits else 0.0
    unique = np.unique(result)
    rank = {value: i for i, value in enumerate(sorted(unique.tolist()))}
    return np.asarray([rank[value] for value in result], dtype=np.float32)


class CrossLayer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.linear = nn.Linear(width, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(width))

    def forward(self, x0, x):
        return x0 * self.linear(x) + self.bias + x


class DurationRegimeModel(nn.Module):
    def __init__(self, field_dims, embedding_dim=24, dropout=0.21):
        super().__init__()
        self.num_fields = len(field_dims)
        self.embedding_dim = embedding_dim
        width = self.num_fields * embedding_dim
        self.embedding = nn.Embedding(int(np.sum(field_dims)), embedding_dim)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.cross1 = CrossLayer(width)
        self.cross2 = CrossLayer(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.trunk = nn.Sequential(
            nn.Linear(width + 128, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.shared_head = nn.Linear(128, 1)
        self.short_residual = nn.Linear(128, 1)
        self.long_residual = nn.Linear(128, 1)
        nn.init.zeros_(self.short_residual.weight)
        nn.init.zeros_(self.short_residual.bias)
        nn.init.zeros_(self.long_residual.weight)
        nn.init.zeros_(self.long_residual.bias)

    def forward(self, x, duration_ms):
        x0 = self.embedding(x).flatten(1)
        cross = self.cross1(x0, x0)
        cross = self.cross2(x0, cross)
        hidden = self.trunk(torch.cat((cross, self.mlp(x0)), dim=1))
        shared = self.shared_head(hidden).squeeze(1)
        short_score = shared + self.short_residual(hidden).squeeze(1)
        long_score = shared + self.long_residual(hidden).squeeze(1)
        return torch.where(duration_ms <= 18000.0, short_score, long_score)

    def regime_penalty(self):
        return sum(
            (parameter * parameter).sum()
            for layer in (self.short_residual, self.long_residual)
            for parameter in layer.parameters()
        )


def predict(model, x, duration, device, batch_size=8192):
    model.eval()
    output = np.empty(len(x), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            xb = torch.as_tensor(x[start:end], dtype=torch.long, device=device)
            db = torch.as_tensor(duration[start:end], dtype=torch.float32, device=device)
            output[start:end] = torch.sigmoid(model(xb, db)).cpu().numpy()
    return output


def metric_values(result):
    gauc = float(result.get("GAUC", result.get("gauc", 0.0)))
    ndcg = float(result.get("nDCG@5", result.get("ndcg5", 0.0)))
    primary = float(result.get("primary", gauc))
    return gauc, ndcg, primary


def evaluate_scores(data, scores):
    try:
        if data["csv_mode"]:
            from harness.evaluate_provisional import evaluate
        else:
            from data.official.evaluate import evaluate
        return metric_values(evaluate(data["users_val"], data["y_val"], scores))
    except Exception:
        return 0.0, 0.0, 0.0


def train_model(data, seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DurationRegimeModel(data["field_dims"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00168, weight_decay=0.000037)

    epochs = 6
    smoke = os.environ.get("SMOKE_EPOCHS")
    if smoke is not None:
        epochs = min(epochs, max(1, int(smoke)))

    day_index = numeric_dates(data["date_train"])
    recency = np.exp2(-(day_index.max() - day_index) / 7.0).astype(np.float32)
    recency /= max(float(recency.mean()), 1e-8)

    n = len(data["x_train"])
    batch_size = 4096
    rng = np.random.RandomState(seed)
    best_state = None
    best_gauc = -np.inf
    stale = 0

    for epoch in range(epochs):
        order = rng.permutation(n)
        midpoint = (n + 1) // 2
        for part_start, part_end in ((0, midpoint), (midpoint, n)):
            model.train()
            for position in range(part_start, part_end, batch_size):
                ids = order[position:min(position + batch_size, part_end)]
                xb = torch.as_tensor(data["x_train"][ids], dtype=torch.long, device=device)
                yb = torch.as_tensor(data["y_train"][ids], dtype=torch.float32, device=device)
                db = torch.as_tensor(data["duration_train"][ids], dtype=torch.float32, device=device)
                wb = torch.as_tensor(recency[ids], dtype=torch.float32, device=device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb, db)
                point_loss = (F.binary_cross_entropy_with_logits(logits, yb, reduction="none") * wb).mean()
                loss = point_loss + 0.001 * model.regime_penalty()
                loss.backward()
                optimizer.step()

            scores = predict(model, data["x_val"], data["duration_val"], device)
            gauc, _, _ = evaluate_scores(data, scores)
            if gauc > best_gauc:
                best_gauc = gauc
                best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stale >= 3:
                break
        if stale >= 3:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, device


def write_outputs(out_dir, data, scores, metrics):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "predictions.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(scores):
            writer.writerow([i, data["users_val"][i], data["videos_val"][i], format(float(score), ".10g")])
    with open(out_path / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump({"gauc": metrics[0], "ndcg5": metrics[1], "primary": metrics[2]}, handle)


def main():
    args = parse_args()
    seed_everything(args.seed)
    data_dir = Path(args.data_dir)
    if (data_dir / "train.npz").is_file() and (data_dir / "val.npz").is_file():
        data = load_npz(data_dir)
    else:
        data = load_csv(data_dir)
    model, device = train_model(data, args.seed)
    scores = predict(model, data["x_val"], data["duration_val"], device)
    metrics = evaluate_scores(data, scores)
    write_outputs(args.out_dir, data, scores, metrics)


if __name__ == "__main__":
    main()
