import argparse
import csv
import json
import math
import os
import random
import warnings

import numpy as np
import torch
from torch import nn

warnings.filterwarnings("ignore")


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_npz(data_dir):
    train_path = os.path.join(data_dir, "train.npz")
    val_path = os.path.join(data_dir, "val.npz")
    if not (os.path.exists(train_path) and os.path.exists(val_path)):
        return None
    tr = np.load(train_path, allow_pickle=False)
    va = np.load(val_path, allow_pickle=False)
    x_train = np.asarray(tr["X"], dtype=np.int64)
    y_train = np.asarray(tr["y"], dtype=np.float32).reshape(-1)
    x_val = np.asarray(va["X"], dtype=np.int64)
    y_val = np.asarray(va["y"], dtype=np.float32).reshape(-1)
    val_users = np.asarray(va["user"]).reshape(-1)
    field_dims = np.asarray(tr["field_dims"], dtype=np.int64).reshape(-1)
    if len(field_dims) != 5:
        raise ValueError("Expected exactly five field dimensions")
    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1]))).astype(np.int64)
    video_encoded = x_val[:, 1] - offsets[1]
    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "val_users": val_users,
        "val_videos": video_encoded,
        "field_dims": field_dims,
        "npz": True,
    }


def read_csv_rows(path, training):
    rows = []
    durations = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            duration = float(row.get("duration_ms", 0) or 0)
            item = {
                "user": row.get("user_id", "0"),
                "video": row.get("video_id", "0"),
                "author": row.get("author_id", "0"),
                "tab": row.get("tab", "0"),
                "duration": duration,
                "label": float(row.get("long_view", 0) or 0),
            }
            rows.append(item)
            if training:
                durations.append(duration)
    return rows, durations


def make_mapping(values):
    mapping = {}
    for value in values:
        if value not in mapping:
            mapping[value] = len(mapping) + 1
    return mapping


def load_csv(data_dir):
    train_rows, durations = read_csv_rows(os.path.join(data_dir, "train.csv"), True)
    val_rows, _ = read_csv_rows(os.path.join(data_dir, "val.csv"), False)
    user_map = make_mapping(r["user"] for r in train_rows)
    video_map = make_mapping(r["video"] for r in train_rows)
    author_map = make_mapping(r["author"] for r in train_rows)
    tab_map = make_mapping(r["tab"] for r in train_rows)
    duration_array = np.asarray(durations, dtype=np.float64)
    if len(duration_array):
        boundaries = np.unique(np.quantile(duration_array, np.linspace(0.1, 0.9, 9)))
    else:
        boundaries = np.asarray([], dtype=np.float64)
    field_dims = np.asarray([
        len(user_map) + 1,
        len(video_map) + 1,
        len(author_map) + 1,
        len(tab_map) + 1,
        len(boundaries) + 1,
    ], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1]))).astype(np.int64)

    def encode(rows):
        x = np.empty((len(rows), 5), dtype=np.int64)
        y = np.empty(len(rows), dtype=np.float32)
        for i, row in enumerate(rows):
            x[i, 0] = user_map.get(row["user"], 0) + offsets[0]
            x[i, 1] = video_map.get(row["video"], 0) + offsets[1]
            x[i, 2] = author_map.get(row["author"], 0) + offsets[2]
            x[i, 3] = tab_map.get(row["tab"], 0) + offsets[3]
            x[i, 4] = int(np.searchsorted(boundaries, row["duration"], side="right")) + offsets[4]
            y[i] = row["label"]
        return x, y

    x_train, y_train = encode(train_rows)
    x_val, y_val = encode(val_rows)
    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "val_users": np.asarray([r["user"] for r in val_rows], dtype=object),
        "val_videos": np.asarray([r["video"] for r in val_rows], dtype=object),
        "field_dims": field_dims,
        "npz": False,
    }


class FinalMLP(nn.Module):
    def __init__(self, field_dims, embed_dim=16, hidden_dim=128):
        super().__init__()
        total_dim = int(np.sum(field_dims))
        input_dim = len(field_dims) * embed_dim
        self.embedding = nn.Embedding(total_dim, embed_dim)
        self.linear = nn.Embedding(total_dim, 1)
        self.gate1 = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid(),
        )
        self.gate2 = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid(),
        )
        self.branch1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
        )
        self.branch2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
        )
        self.fusion = nn.Linear(64, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.zeros_(self.linear.weight)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        emb = self.embedding(x).flatten(1)
        selected1 = emb * (2.0 * self.gate1(emb))
        selected2 = emb * (2.0 * self.gate2(emb))
        h1 = self.branch1(selected1)
        h2 = self.branch2(selected2)
        interaction = self.fusion(h1 * h2).squeeze(1)
        linear_term = self.linear(x).sum(dim=1).squeeze(1)
        return interaction + linear_term + self.bias


def predict(model, x, device, batch_size):
    model.eval()
    result = np.empty(len(x), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            xb = torch.as_tensor(x[start:end], dtype=torch.long, device=device)
            result[start:end] = torch.sigmoid(model(xb)).cpu().numpy()
    return result


def official_metrics(npz_mode, users, labels, scores):
    if npz_mode:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    raw = evaluate(users, labels, scores)
    return {
        "gauc": float(raw.get("GAUC", raw.get("gauc"))),
        "ndcg5": float(raw.get("nDCG@5", raw.get("ndcg5"))),
        "primary": float(raw.get("primary")),
    }


def train_model(data, seed, epochs):
    x_train = data["x_train"]
    y_train = data["y_train"]
    x_val = data["x_val"]
    y_val = data["y_val"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FinalMLP(data["field_dims"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    batch_size = 8192 if device.type == "cuda" else 4096
    rng = np.random.default_rng(seed)
    best_state = None
    best_gauc = -float("inf")
    stale = 0

    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(x_train))
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            xb = torch.as_tensor(x_train[idx], dtype=torch.long, device=device)
            yb = torch.as_tensor(y_train[idx], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        val_scores = predict(model, x_val, device, batch_size * 2)
        metrics = official_metrics(data["npz"], data["val_users"], y_val, val_scores)
        if metrics["gauc"] > best_gauc + 1e-8:
            best_gauc = metrics["gauc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 2:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return predict(model, x_val, device, batch_size * 2)


def write_outputs(out_dir, data, scores, metrics):
    os.makedirs(out_dir, exist_ok=True)
    pred_path = os.path.join(out_dir, "predictions.csv")
    with open(pred_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, (user, video, score) in enumerate(zip(data["val_users"], data["val_videos"], scores)):
            if isinstance(user, np.generic):
                user = user.item()
            if isinstance(video, np.generic):
                video = video.item()
            writer.writerow([i, user, video, format(float(score), ".10g")])
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, separators=(",", ":"), allow_nan=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed_everything(args.seed)
    data = load_npz(args.data_dir)
    if data is None:
        data = load_csv(args.data_dir)
    epochs = 10
    smoke = os.environ.get("SMOKE_EPOCHS")
    if smoke is not None:
        epochs = min(epochs, max(1, int(smoke)))
    scores = train_model(data, args.seed, epochs)
    metrics = official_metrics(data["npz"], data["val_users"], data["y_val"], scores)
    write_outputs(args.out_dir, data, scores, metrics)


if __name__ == "__main__":
    main()
