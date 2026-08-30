import argparse
import csv
import json
import os
import random
import warnings

import numpy as np
import torch
from torch import nn

warnings.filterwarnings("ignore")


class FactorizationMachine(nn.Module):
    def __init__(self, total_features, embedding_dim):
        super().__init__()
        self.linear = nn.Embedding(total_features, 1)
        self.embedding = nn.Embedding(total_features, embedding_dim)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

    def forward(self, x):
        linear_term = self.linear(x).sum(dim=1).squeeze(-1)
        vectors = self.embedding(x)
        summed = vectors.sum(dim=1)
        interaction = 0.5 * (
            summed.square().sum(dim=1) - vectors.square().sum(dim=(1, 2))
        )
        return self.bias + linear_term + interaction


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_mapping(values):
    unique = sorted(set(values))
    return {value: index + 1 for index, value in enumerate(unique)}


def load_csv_split(path, include_training_outcomes):
    records = []
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        has_author = "author_id" in (reader.fieldnames or [])
        for row in reader:
            record = {
                "user": row["user_id"],
                "video": row["video_id"],
                "author": row["author_id"] if has_author else "__missing_author__",
                "tab": row["tab"],
                "duration": float(row["duration_ms"]),
                "label": float(row["long_view"]),
            }
            records.append(record)
    return records


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        train_data = np.load(train_npz, allow_pickle=False)
        val_data = np.load(val_npz, allow_pickle=False)
        x_train = np.asarray(train_data["X"], dtype=np.int64)
        y_train = np.asarray(train_data["y"], dtype=np.float32)
        x_val = np.asarray(val_data["X"], dtype=np.int64)
        y_val = np.asarray(val_data["y"], dtype=np.float32)
        val_users = np.asarray(val_data["user"])
        field_dims = np.asarray(train_data["field_dims"], dtype=np.int64)
        video_offset = int(field_dims[0])
        val_videos = x_val[:, 1].astype(np.int64) - video_offset
        return {
            "x_train": x_train,
            "y_train": y_train,
            "x_val": x_val,
            "y_val": y_val,
            "val_users": val_users,
            "val_videos": val_videos,
            "field_dims": field_dims,
            "fast": True,
        }

    train_records = load_csv_split(os.path.join(data_dir, "train.csv"), True)
    val_records = load_csv_split(os.path.join(data_dir, "val.csv"), False)

    user_map = build_mapping([r["user"] for r in train_records])
    video_map = build_mapping([r["video"] for r in train_records])
    author_map = build_mapping([r["author"] for r in train_records])
    tab_map = build_mapping([r["tab"] for r in train_records])

    train_durations = np.asarray([r["duration"] for r in train_records], dtype=np.float64)
    quantiles = np.quantile(train_durations, np.linspace(0.0, 1.0, 11))
    duration_edges = np.unique(quantiles[1:-1])

    field_dims = np.asarray(
        [len(user_map) + 1, len(video_map) + 1, len(author_map) + 1, len(tab_map) + 1, 10],
        dtype=np.int64,
    )
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1])).astype(np.int64)

    def encode(records):
        x = np.empty((len(records), 5), dtype=np.int64)
        y = np.empty(len(records), dtype=np.float32)
        for i, record in enumerate(records):
            values = [
                user_map.get(record["user"], 0),
                video_map.get(record["video"], 0),
                author_map.get(record["author"], 0),
                tab_map.get(record["tab"], 0),
                min(9, int(np.searchsorted(duration_edges, record["duration"], side="right"))),
            ]
            x[i] = np.asarray(values, dtype=np.int64) + offsets
            y[i] = record["label"]
        return x, y

    x_train, y_train = encode(train_records)
    x_val, y_val = encode(val_records)
    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "val_users": np.asarray([r["user"] for r in val_records]),
        "val_videos": np.asarray([r["video"] for r in val_records]),
        "field_dims": field_dims,
        "fast": False,
    }


def metric_values(result):
    gauc = result["GAUC"] if "GAUC" in result else result["gauc"]
    ndcg = result["nDCG@5"] if "nDCG@5" in result else result["ndcg5"]
    primary = result["primary"]
    return float(gauc), float(ndcg), float(primary)


def evaluate_scores(fast, users, labels, scores):
    if fast:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    return metric_values(evaluate(users, labels, scores))


def predict(model, x, device, batch_size):
    model.eval()
    outputs = np.empty(len(x), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            batch = torch.from_numpy(x[start:end]).to(device=device, dtype=torch.long)
            outputs[start:end] = torch.sigmoid(model(batch)).cpu().numpy()
    return outputs


def train_one(data, seed, epochs, device):
    set_seed(seed)
    total_features = int(np.sum(data["field_dims"]))
    model = FactorizationMachine(total_features, 16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    batch_size = 8192
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    best_primary = -float("inf")
    best_scores = None
    stale_epochs = 0

    x_train = data["x_train"]
    y_train = data["y_train"]
    for _ in range(epochs):
        model.train()
        permutation = torch.randperm(len(x_train), generator=generator).numpy()
        for start in range(0, len(x_train), batch_size):
            indices = permutation[start:start + batch_size]
            xb = torch.from_numpy(x_train[indices]).to(device=device, dtype=torch.long)
            yb = torch.from_numpy(y_train[indices]).to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        scores = predict(model, data["x_val"], device, batch_size * 2)
        _, _, primary = evaluate_scores(
            data["fast"], data["val_users"], data["y_val"], scores
        )
        if primary > best_primary:
            best_primary = primary
            best_scores = scores.copy()
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= 2:
            break

    return best_scores


def per_user_ranks(users, scores):
    users = np.asarray(users)
    scores = np.asarray(scores)
    order = np.lexsort((scores, users))
    ranked = np.empty(len(scores), dtype=np.float64)
    sorted_users = users[order]
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_users[end] == sorted_users[start]:
            end += 1
        count = end - start
        if count == 1:
            ranked[order[start]] = 0.5
        else:
            ranked[order[start:end]] = np.arange(count, dtype=np.float64) / float(count - 1)
        start = end
    return ranked


def scalar_text(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    data = load_data(args.data_dir)
    epochs = 10
    smoke_epochs = os.environ.get("SMOKE_EPOCHS")
    if smoke_epochs is not None:
        epochs = min(epochs, max(1, int(smoke_epochs)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [args.seed, args.seed + 1009, args.seed + 2018]
    model_scores = [train_one(data, seed, epochs, device) for seed in seeds]
    rank_scores = [per_user_ranks(data["val_users"], scores) for scores in model_scores]
    final_scores = np.mean(np.stack(rank_scores, axis=0), axis=0)

    gauc, ndcg5, primary = evaluate_scores(
        data["fast"], data["val_users"], data["y_val"], final_scores
    )

    predictions_path = os.path.join(args.out_dir, "predictions.csv")
    with open(predictions_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i in range(len(final_scores)):
            writer.writerow([
                i,
                scalar_text(data["val_users"][i]),
                scalar_text(data["val_videos"][i]),
                repr(float(final_scores[i])),
            ])

    metrics_path = os.path.join(args.out_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(
            {"gauc": gauc, "ndcg5": ndcg5, "primary": primary},
            handle,
            separators=(",", ":"),
        )


if __name__ == "__main__":
    main()
