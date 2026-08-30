import argparse
import csv
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def scalar_string(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        value = value.item()
    return str(value)


def load_npz_data(data_dir):
    train_path = os.path.join(data_dir, "train.npz")
    val_path = os.path.join(data_dir, "val.npz")
    if not (os.path.exists(train_path) and os.path.exists(val_path)):
        return None

    with np.load(train_path, allow_pickle=False) as tr:
        train_x = np.asarray(tr["X"], dtype=np.int64)
        train_y = np.asarray(tr["y"], dtype=np.float32)
        field_dims = np.asarray(tr["field_dims"], dtype=np.int64).reshape(-1)

    with np.load(val_path, allow_pickle=False) as va:
        val_x = np.asarray(va["X"], dtype=np.int64)
        val_y = np.asarray(va["y"], dtype=np.float32)
        val_user_eval = np.asarray(va["user"])

    if field_dims.size != 5:
        raise ValueError("Expected exactly five encoded fields")

    offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(field_dims)[:-1]))
    val_video = val_x[:, 1] - offsets[1]
    val_user_out = val_user_eval
    return {
        "train_x": train_x,
        "train_y": train_y,
        "val_x": val_x,
        "val_y": val_y,
        "eval_users": val_user_eval,
        "out_users": val_user_out,
        "out_videos": val_video,
        "field_dims": field_dims,
        "fast": True,
    }


def read_csv_rows(path, training):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = {
                "user_id": row["user_id"],
                "video_id": row["video_id"],
                "tab": row["tab"],
                "duration_ms": float(row["duration_ms"]),
                "long_view": float(row["long_view"]),
            }
            rows.append(item)
    return rows


def make_mapping(values):
    mapping = {}
    for value in values:
        if value not in mapping:
            mapping[value] = len(mapping)
    return mapping


def duration_edges(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return np.zeros(9, dtype=np.float64)
    return np.quantile(arr, np.arange(1, 10, dtype=np.float64) / 10.0)


def encode_csv_rows(rows, user_map, video_map, tab_map, edges, field_dims):
    n = len(rows)
    local = np.empty((n, 5), dtype=np.int64)
    users_out = []
    videos_out = []
    user_unk = len(user_map)
    video_unk = len(video_map)
    tab_unk = len(tab_map)
    labels = np.empty(n, dtype=np.float32)

    for i, row in enumerate(rows):
        user = row["user_id"]
        video = row["video_id"]
        tab = row["tab"]
        local[i, 0] = user_map.get(user, user_unk)
        local[i, 1] = video_map.get(video, video_unk)
        local[i, 2] = 0
        local[i, 3] = tab_map.get(tab, tab_unk)
        local[i, 4] = int(np.searchsorted(edges, row["duration_ms"], side="right"))
        labels[i] = row["long_view"]
        users_out.append(user)
        videos_out.append(video)

    offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(field_dims)[:-1]))
    encoded = local + offsets.reshape(1, -1)
    return encoded, labels, np.asarray(users_out), np.asarray(videos_out)


def load_csv_data(data_dir):
    train_rows = read_csv_rows(os.path.join(data_dir, "train.csv"), True)
    val_rows = read_csv_rows(os.path.join(data_dir, "val.csv"), False)

    user_map = make_mapping(row["user_id"] for row in train_rows)
    video_map = make_mapping(row["video_id"] for row in train_rows)
    tab_map = make_mapping(row["tab"] for row in train_rows)
    edges = duration_edges([row["duration_ms"] for row in train_rows])
    field_dims = np.asarray(
        [len(user_map) + 1, len(video_map) + 1, 1, len(tab_map) + 1, 10],
        dtype=np.int64,
    )

    train_x, train_y, _, _ = encode_csv_rows(
        train_rows, user_map, video_map, tab_map, edges, field_dims
    )
    val_x, val_y, val_users, val_videos = encode_csv_rows(
        val_rows, user_map, video_map, tab_map, edges, field_dims
    )
    return {
        "train_x": train_x,
        "train_y": train_y,
        "val_x": val_x,
        "val_y": val_y,
        "eval_users": val_users,
        "out_users": val_users,
        "out_videos": val_videos,
        "field_dims": field_dims,
        "fast": False,
    }


class FactorizationMachine(nn.Module):
    def __init__(self, total_features, rank):
        super().__init__()
        self.linear = nn.Embedding(total_features, 1)
        self.embedding = nn.Embedding(total_features, rank)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

    def forward(self, x):
        linear_term = self.linear(x).sum(dim=1).squeeze(-1)
        embedded = self.embedding(x)
        summed = embedded.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - embedded.square().sum(dim=1)
        ).sum(dim=1)
        return self.bias + linear_term + interaction


def predict(model, x, device, batch_size):
    model.eval()
    result = np.empty(x.shape[0], dtype=np.float32)
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            end = min(start + batch_size, x.shape[0])
            xb = torch.from_numpy(x[start:end]).to(device=device, dtype=torch.long)
            result[start:end] = torch.sigmoid(model(xb)).cpu().numpy()
    return result


def evaluate_scores(fast, users, labels, scores):
    if fast:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    raw = evaluate(users, labels, scores)
    return {
        "gauc": float(raw.get("GAUC", raw.get("gauc"))),
        "ndcg5": float(raw.get("nDCG@5", raw.get("ndcg5"))),
        "primary": float(raw.get("primary")),
    }


def train_model(data, seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FactorizationMachine(int(data["field_dims"].sum()), 16).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=3e-4)

    epochs = 6
    smoke = os.environ.get("SMOKE_EPOCHS")
    if smoke is not None:
        epochs = min(epochs, max(1, int(smoke)))

    batch_size = 8192
    train_x = data["train_x"]
    train_y = data["train_y"]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    best_scores = None
    best_metrics = None
    best_gauc = -float("inf")
    history = []

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(train_x.shape[0], generator=generator).numpy()
        for start in range(0, train_x.shape[0], batch_size):
            idx = permutation[start:start + batch_size]
            xb = torch.from_numpy(train_x[idx]).to(device=device, dtype=torch.long)
            yb = torch.from_numpy(train_y[idx]).to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            optimizer.step()

        scores = predict(model, data["val_x"], device, batch_size * 2)
        metrics = evaluate_scores(
            data["fast"], data["eval_users"], data["val_y"], scores
        )
        history.append({
            "epoch": epoch + 1,
            "rank": 16,
            "weight_decay": 0.0003,
            "learning_rate": 0.001,
            "gauc": metrics["gauc"],
            "ndcg5": metrics["ndcg5"],
            "primary": metrics["primary"],
        })
        if metrics["gauc"] > best_gauc:
            best_gauc = metrics["gauc"]
            best_scores = scores.copy()
            best_metrics = metrics

    best_metrics["history"] = history
    best_metrics["selected_epoch"] = int(
        max(history, key=lambda item: item["gauc"])["epoch"]
    )
    best_metrics["config"] = {
        "model": "fm",
        "rank": 16,
        "optimizer": "AdamW",
        "weight_decay": 0.0003,
        "learning_rate": 0.001,
    }
    return best_scores, best_metrics


def write_outputs(out_dir, data, scores, metrics):
    os.makedirs(out_dir, exist_ok=True)
    prediction_path = os.path.join(out_dir, "predictions.csv")
    with open(prediction_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, (user, video, score) in enumerate(
            zip(data["out_users"], data["out_videos"], scores)
        ):
            writer.writerow([i, scalar_string(user), scalar_string(video), float(score)])

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, sort_keys=True)


def main():
    args = parse_args()
    set_seed(args.seed)
    data = load_npz_data(args.data_dir)
    if data is None:
        data = load_csv_data(args.data_dir)
    scores, metrics = train_model(data, args.seed)
    write_outputs(args.out_dir, data, scores, metrics)


if __name__ == "__main__":
    main()
