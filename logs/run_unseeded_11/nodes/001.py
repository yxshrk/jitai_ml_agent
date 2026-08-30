import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FM(torch.nn.Module):
    def __init__(self, total_dim, k=16):
        super().__init__()
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)

    def forward(self, x):
        e = self.emb(x)
        s = e.sum(dim=1)
        pair = 0.5 * (s * s - (e * e).sum(dim=1)).sum(dim=1)
        return self.bias + self.lin(x).sum(dim=(1, 2)) + pair


def load_csv_data(data_dir):
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")

    train_rows = []
    durations = []
    with open(train_path, "r", newline="") as fh:
        for row in csv.DictReader(fh):
            user = row["user_id"]
            video = row["video_id"]
            tab = row["tab"]
            duration = float(row["duration_ms"])
            label = float(row["long_view"])
            train_rows.append((user, video, tab, duration, label))
            durations.append(duration)

    duration_array = np.asarray(durations, dtype=np.float64)
    boundaries = np.quantile(duration_array, np.arange(1, 10) / 10.0)

    user_values = sorted({r[0] for r in train_rows})
    video_values = sorted({r[1] for r in train_rows})
    tab_values = sorted({r[2] for r in train_rows})
    user_map = {v: i + 1 for i, v in enumerate(user_values)}
    video_map = {v: i + 1 for i, v in enumerate(video_values)}
    tab_map = {v: i + 1 for i, v in enumerate(tab_values)}

    field_dims = np.asarray(
        [len(user_map) + 1, len(video_map) + 1, 1, len(tab_map) + 1, 10],
        dtype=np.int64,
    )
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1]))

    def encode(user, video, tab, duration):
        local = np.asarray(
            [
                user_map.get(user, 0),
                video_map.get(video, 0),
                0,
                tab_map.get(tab, 0),
                int(np.searchsorted(boundaries, duration, side="right")),
            ],
            dtype=np.int64,
        )
        return local + offsets

    train_x = np.stack([encode(r[0], r[1], r[2], r[3]) for r in train_rows])
    train_y = np.asarray([r[4] for r in train_rows], dtype=np.float32)

    val_encoded = []
    val_y = []
    val_users = []
    val_videos = []
    with open(val_path, "r", newline="") as fh:
        for row in csv.DictReader(fh):
            user = row["user_id"]
            video = row["video_id"]
            tab = row["tab"]
            duration = float(row["duration_ms"])
            val_encoded.append(encode(user, video, tab, duration))
            val_y.append(float(row["long_view"]))
            val_users.append(user)
            val_videos.append(video)

    return {
        "train_x": train_x,
        "train_y": train_y,
        "val_x": np.stack(val_encoded),
        "val_y": np.asarray(val_y, dtype=np.float32),
        "val_users": np.asarray(val_users),
        "val_videos": np.asarray(val_videos),
        "field_dims": field_dims,
        "npz": False,
    }


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        tr = np.load(train_npz)
        va = np.load(val_npz)
        field_dims = tr["field_dims"].astype(np.int64)
        video_offset = int(field_dims[0])
        return {
            "train_x": tr["X"].astype(np.int64),
            "train_y": tr["y"].astype(np.float32),
            "val_x": va["X"].astype(np.int64),
            "val_y": va["y"].astype(np.float32),
            "val_users": va["user"],
            "val_videos": va["X"][:, 1].astype(np.int64) - video_offset,
            "field_dims": field_dims,
            "npz": True,
        }
    return load_csv_data(data_dir)


def fit_member(train_x, train_y, val_x, total_dim, seed, epochs):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = FM(total_dim, k=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()
    n = len(train_y)
    batch_size = 8192

    model.train()
    for _ in range(epochs):
        permutation = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = permutation[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(train_x[idx])
            loss = criterion(logits, train_y[idx])
            loss.backward()
            optimizer.step()

    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(val_x), 65536):
            parts.append(model(val_x[start:start + 65536]).cpu().numpy())
    return np.concatenate(parts).astype(np.float64)


def per_user_ranks(users, scores):
    groups = {}
    for index, user in enumerate(users):
        key = user.item() if isinstance(user, np.generic) else user
        groups.setdefault(key, []).append(index)

    ranked = np.empty(len(scores), dtype=np.float64)
    for indices in groups.values():
        idx = np.asarray(indices, dtype=np.int64)
        if len(idx) == 1:
            ranked[idx[0]] = 0.5
            continue
        order = np.argsort(scores[idx], kind="mergesort")
        values = np.empty(len(idx), dtype=np.float64)
        values[order] = np.arange(len(idx), dtype=np.float64) / float(len(idx) - 1)
        ranked[idx] = values
    return ranked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    data = load_data(args.data_dir)

    epochs = 8
    smoke_epochs = os.environ.get("SMOKE_EPOCHS")
    if smoke_epochs is not None:
        epochs = min(epochs, max(1, int(smoke_epochs)))

    train_x = torch.from_numpy(data["train_x"])
    train_y = torch.from_numpy(data["train_y"])
    val_x = torch.from_numpy(data["val_x"])
    total_dim = int(data["field_dims"].sum())

    rank_sum = np.zeros(len(data["val_y"]), dtype=np.float64)
    for member in range(3):
        raw_scores = fit_member(
            train_x,
            train_y,
            val_x,
            total_dim,
            args.seed + member,
            epochs,
        )
        rank_sum += per_user_ranks(data["val_users"], raw_scores)
    scores = rank_sum / 3.0

    if data["npz"]:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate

    metrics = evaluate(data["val_users"], data["val_y"].astype(int), scores)
    gauc = metrics["GAUC"] if "GAUC" in metrics else metrics["gauc"]
    ndcg5 = metrics["nDCG@5"] if "nDCG@5" in metrics else metrics["ndcg5"]

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(
            {"gauc": float(gauc), "ndcg5": float(ndcg5), "primary": float(metrics["primary"])},
            fh,
        )

    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(scores):
            writer.writerow([i, data["val_users"][i], data["val_videos"][i], format(float(score), ".9g")])


if __name__ == "__main__":
    main()
