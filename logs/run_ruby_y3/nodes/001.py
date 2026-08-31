"""Causal pooled author-history DeepFM over the official five fields and context.

Uses strictly preceding rows from each user's impression stream to construct a
mean-pooled history of the previous 12 authors. Validation histories begin with
the terminal training histories and are then updated causally in validation
file order. Outcomes are never used as model inputs.
"""
import argparse
import csv
import datetime
import json
import os
import sys
from collections import defaultdict, deque

import numpy as np
import torch


class CausalHistoryDeepFM(torch.nn.Module):
    def __init__(self, total_dim, num_current_fields=8, embedding_dim=12,
                 hidden_dim=48, dropout=0.05):
        super().__init__()
        self.num_current_fields = num_current_fields
        self.embedding = torch.nn.Embedding(total_dim, embedding_dim)
        self.linear = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.deep = torch.nn.Sequential(
            torch.nn.Linear((num_current_fields + 1) * embedding_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        )
        torch.nn.init.normal_(self.embedding.weight, std=0.01)
        torch.nn.init.zeros_(self.linear.weight)
        for layer in self.deep:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

    def forward(self, x, history):
        current = self.embedding(x)
        mask = history.ge(0)
        safe_history = history.clamp_min(0)
        history_embeddings = self.embedding(safe_history)
        history_embeddings = history_embeddings * mask.unsqueeze(-1).to(history_embeddings.dtype)
        denominator = mask.sum(1, keepdim=True).clamp_min(1).to(history_embeddings.dtype)
        pooled_history = history_embeddings.sum(1) / denominator
        pooled_history = pooled_history * mask.any(1, keepdim=True).to(pooled_history.dtype)

        fields = torch.cat((current, pooled_history.unsqueeze(1)), dim=1)
        summed = fields.sum(1)
        fm = 0.5 * (summed.square() - fields.square().sum(1)).sum(1)
        linear = self.linear(x).sum((1, 2))
        deep = self.deep(fields.flatten(1)).squeeze(1)
        return self.bias + linear + fm + deep


def weekday_from_dates(values):
    result = np.zeros(len(values), dtype=np.int64)
    cache = {}
    for i, value in enumerate(values):
        key = int(value)
        if key not in cache:
            text = str(key)
            try:
                cache[key] = datetime.date(int(text[:4]), int(text[4:6]), int(text[6:8])).weekday()
            except (ValueError, IndexError):
                cache[key] = 0
        result[i] = cache[key]
    return result


def hour_from_hourmin(values):
    values = np.asarray(values, dtype=np.int64)
    if len(values) == 0:
        return values.copy()
    maximum = int(values.max())
    if maximum <= 2359:
        hours = values // 100
    else:
        hours = values // 60
    return np.clip(hours, 0, 23).astype(np.int64)


def add_context_fields(x, field_dims, hourmin, dates):
    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1]))).astype(np.int64)
    tab_raw = x[:, 3].astype(np.int64) - offsets[3]
    hour = hour_from_hourmin(hourmin)
    weekday = weekday_from_dates(dates)
    is_rand = (tab_raw == 1).astype(np.int64)
    base_total = int(np.sum(field_dims))
    context = np.column_stack((
        hour + base_total,
        weekday + base_total + 24,
        is_rand + base_total + 31,
    )).astype(np.int64)
    return np.concatenate((x.astype(np.int64), context), axis=1), base_total + 33


def build_causal_histories(train_users, train_authors, val_users, val_authors, length=12):
    state = defaultdict(lambda: deque(maxlen=length))
    train_history = np.full((len(train_users), length), -1, dtype=np.int32)
    val_history = np.full((len(val_users), length), -1, dtype=np.int32)

    for i in range(len(train_users)):
        user = train_users[i].item() if isinstance(train_users[i], np.generic) else train_users[i]
        prior = state[user]
        if prior:
            train_history[i, :len(prior)] = np.fromiter(prior, dtype=np.int32, count=len(prior))
        prior.append(int(train_authors[i]))

    for i in range(len(val_users)):
        user = val_users[i].item() if isinstance(val_users[i], np.generic) else val_users[i]
        prior = state[user]
        if prior:
            val_history[i, :len(prior)] = np.fromiter(prior, dtype=np.int32, count=len(prior))
        prior.append(int(val_authors[i]))

    return train_history, val_history


def read_csv_rows(path, training):
    rows = []
    with open(path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            item = {
                "user_id": row["user_id"],
                "video_id": row["video_id"],
                "author_id": row.get("author_id", row["video_id"]),
                "tab": row["tab"],
                "hourmin": int(float(row["hourmin"])),
                "date": int(float(row["date"])),
                "duration_ms": float(row["duration_ms"]),
                "long_view": float(row["long_view"]),
            }
            rows.append(item)
    return rows


def prepare_csv_data(data_dir):
    train_rows = read_csv_rows(os.path.join(data_dir, "train.csv"), True)
    val_rows = read_csv_rows(os.path.join(data_dir, "val.csv"), False)
    fields = ("user_id", "video_id", "author_id", "tab")
    mappings = {}
    dimensions = []
    for field in fields:
        mapping = {}
        for row in train_rows:
            value = row[field]
            if value not in mapping:
                mapping[value] = len(mapping)
        mappings[field] = mapping
        dimensions.append(len(mapping) + 1)

    durations = np.asarray([r["duration_ms"] for r in train_rows], dtype=np.float64)
    quantiles = np.quantile(durations, np.linspace(0.1, 0.9, 9))
    dimensions.append(10)
    offsets = np.concatenate(([0], np.cumsum(dimensions[:-1]))).astype(np.int64)

    def encode(rows):
        x = np.empty((len(rows), 5), dtype=np.int32)
        for i, row in enumerate(rows):
            for j, field in enumerate(fields):
                mapping = mappings[field]
                x[i, j] = offsets[j] + mapping.get(row[field], len(mapping))
            x[i, 4] = offsets[4] + int(np.searchsorted(quantiles, row["duration_ms"], side="right"))
        return {
            "X": x,
            "y": np.asarray([r["long_view"] for r in rows], dtype=np.float32),
            "user": np.asarray([r["user_id"] for r in rows]),
            "video": np.asarray([r["video_id"] for r in rows]),
            "hourmin": np.asarray([r["hourmin"] for r in rows], dtype=np.int32),
            "date": np.asarray([r["date"] for r in rows], dtype=np.int32),
            "field_dims": np.asarray(dimensions, dtype=np.int64),
        }

    return encode(train_rows), encode(val_rows)


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        train_file = np.load(train_npz)
        val_file = np.load(val_npz)
        train = {key: train_file[key] for key in train_file.files}
        val = {key: val_file[key] for key in val_file.files}
        field_dims = np.asarray(train["field_dims"], dtype=np.int64)
        video_offset = int(field_dims[0])
        val["video"] = val["X"][:, 1].astype(np.int64) - video_offset
        return train, val, True
    train, val = prepare_csv_data(data_dir)
    return train, val, False


def predict(model, x, history, device, batch_size=65536):
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            stop = min(start + batch_size, len(x))
            xb = torch.from_numpy(x[start:stop]).to(device=device, dtype=torch.long)
            hb = torch.from_numpy(history[start:stop]).to(device=device, dtype=torch.long)
            parts.append(model(xb, hb).detach().cpu().numpy())
    return np.concatenate(parts).astype(np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()

    seed = int(args.seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = int(args.epochs)
    smoke_epochs = os.environ.get("SMOKE_EPOCHS")
    if smoke_epochs is not None:
        epochs = min(epochs, max(1, int(smoke_epochs)))

    train, val, fast_path = load_data(args.data_dir)
    field_dims = np.asarray(train["field_dims"], dtype=np.int64)
    train_x, total_dim = add_context_fields(
        np.asarray(train["X"]), field_dims, train["hourmin"], train["date"])
    val_x, _ = add_context_fields(
        np.asarray(val["X"]), field_dims, val["hourmin"], val["date"])

    train_author = np.asarray(train["X"][:, 2], dtype=np.int64)
    val_author = np.asarray(val["X"][:, 2], dtype=np.int64)
    train_history, val_history = build_causal_histories(
        np.asarray(train["user"]), train_author,
        np.asarray(val["user"]), val_author, length=12)

    if fast_path:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if root not in sys.path:
            sys.path.insert(0, root)
        from data.official.evaluate import evaluate
    else:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if root not in sys.path:
            sys.path.insert(0, root)
        from harness.evaluate_provisional import evaluate

    model = CausalHistoryDeepFM(
        total_dim=total_dim,
        num_current_fields=8,
        embedding_dim=12,
        hidden_dim=48,
        dropout=0.05,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00115, weight_decay=3e-6)
    criterion = torch.nn.BCEWithLogitsLoss()
    labels = np.asarray(train["y"], dtype=np.float32)
    val_labels = np.asarray(val["y"], dtype=np.int64)
    n = len(labels)
    batch_size = 8192
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    best_primary = -1.0
    best_scores = None
    history = []
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(n, generator=generator).numpy()
        loss_sum = 0.0
        seen = 0
        for start in range(0, n, batch_size):
            indices = permutation[start:start + batch_size]
            xb = torch.from_numpy(train_x[indices]).to(device=device, dtype=torch.long)
            hb = torch.from_numpy(train_history[indices]).to(device=device, dtype=torch.long)
            yb = torch.from_numpy(labels[indices]).to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb, hb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            count = len(indices)
            loss_sum += float(loss.detach().cpu()) * count
            seen += count

        scores = predict(model, val_x, val_history, device)
        metrics = evaluate(np.asarray(val["user"]), val_labels, scores)
        primary = float(metrics["primary"])
        history.append({
            "epoch": epoch + 1,
            "train_loss": round(loss_sum / max(seen, 1), 6),
            "val_gauc": round(float(metrics.get("GAUC", metrics.get("gauc", 0.0))), 6),
            "val_primary": round(primary, 6),
        })
        if primary > best_primary + 1e-8:
            best_primary = primary
            best_scores = scores.copy()

    final_metrics = evaluate(np.asarray(val["user"]), val_labels, best_scores)
    gauc = float(final_metrics.get("GAUC", final_metrics.get("gauc")))
    ndcg5 = float(final_metrics.get("nDCG@5", final_metrics.get("ndcg5")))
    primary = float(final_metrics["primary"])

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as handle:
        json.dump({
            "gauc": gauc,
            "ndcg5": ndcg5,
            "primary": primary,
            "history": history,
            "configuration": {
                "method": "causal_pooled_author_history_deepfm",
                "embedding_dim": 12,
                "hidden_dim": 48,
                "history_length": 12,
                "learning_rate": 0.00115,
                "weight_decay": 3e-6,
                "dropout": 0.05,
                "epochs": epochs,
                "seed": seed,
            },
        }, handle)

    users = np.asarray(val["user"])
    videos = np.asarray(val["video"])
    with open(os.path.join(args.out_dir, "predictions.csv"), "w") as handle:
        handle.write("row_id,user_id,video_id,score\n")
        for i, score in enumerate(best_scores):
            handle.write(f"{i},{users[i]},{videos[i]},{score:.8g}\n")


if __name__ == "__main__":
    main()
