"""FM baseline with a within-user BPR and pointwise BCE hybrid objective."""
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
        s = e.sum(1)
        pair = 0.5 * (s * s - (e * e).sum(1)).sum(1)
        return self.bias + self.lin(x).sum((1, 2)) + pair


def duration_bucket(value):
    return str(min(max(int(float(value)) // 10000, 0), 30))


def read_csv_rows(path, training):
    features = []
    labels = []
    users = []
    videos = []
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            user = row["user_id"]
            video = row["video_id"]
            features.append((user, video, "0", row["tab"], duration_bucket(row["duration_ms"])))
            labels.append(float(row["long_view"]))
            users.append(user)
            videos.append(video)
    return features, np.asarray(labels, dtype=np.float32), np.asarray(users), np.asarray(videos)


def load_csv_data(data_dir):
    train_features, train_y, train_users, train_videos = read_csv_rows(
        os.path.join(data_dir, "train.csv"), True
    )
    val_features, val_y, val_users, val_videos = read_csv_rows(
        os.path.join(data_dir, "val.csv"), False
    )
    maps = []
    field_dims = []
    for field in range(5):
        values = sorted({row[field] for row in train_features})
        mapping = {value: i + 1 for i, value in enumerate(values)}
        maps.append(mapping)
        field_dims.append(len(mapping) + 1)
    offsets = np.cumsum([0] + field_dims[:-1], dtype=np.int64)

    def encode(rows):
        x = np.empty((len(rows), 5), dtype=np.int64)
        for i, row in enumerate(rows):
            for field in range(5):
                x[i, field] = offsets[field] + maps[field].get(row[field], 0)
        return x

    return {
        "Xt": encode(train_features),
        "yt": train_y,
        "train_users": train_users,
        "Xv": encode(val_features),
        "yv": val_y,
        "val_users": val_users,
        "val_videos": val_videos,
        "field_dims": np.asarray(field_dims, dtype=np.int64),
        "fast": False,
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
            "Xt": tr["X"].astype(np.int64),
            "yt": tr["y"].astype(np.float32),
            "train_users": np.asarray(tr["user"]),
            "Xv": va["X"].astype(np.int64),
            "yv": va["y"].astype(np.float32),
            "val_users": np.asarray(va["user"]),
            "val_videos": va["X"][:, 1].astype(np.int64) - video_offset,
            "field_dims": field_dims,
            "fast": True,
        }
    return load_csv_data(data_dir)


def make_mixed_user_groups(users, labels):
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    groups = []
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_users[end] == sorted_users[start]:
            end += 1
        idx = order[start:end]
        pos = idx[labels[idx] > 0.5]
        neg = idx[labels[idx] <= 0.5]
        if len(pos) and len(neg):
            groups.append((pos.astype(np.int64), neg.astype(np.int64)))
        start = end
    return groups


def sample_pairs(groups, rng):
    positives = []
    negatives = []
    for pos, neg in groups:
        positives.append(pos)
        negatives.append(neg[rng.integers(0, len(neg), size=len(pos))])
    if not positives:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(positives), np.concatenate(negatives)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()

    epochs = args.epochs
    if "SMOKE_EPOCHS" in os.environ:
        epochs = min(epochs, max(1, int(os.environ["SMOKE_EPOCHS"])))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    data = load_data(args.data_dir)
    Xt = torch.from_numpy(data["Xt"])
    yt = torch.from_numpy(data["yt"])
    Xv = torch.from_numpy(data["Xv"])
    total_dim = int(data["field_dims"].sum())

    if data["fast"]:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate

    groups = make_mixed_user_groups(data["train_users"], data["yt"])
    model = FM(total_dim)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = torch.nn.BCEWithLogitsLoss()
    n = len(yt)
    bs = 8192
    best = -1.0
    best_scores = None
    patience = 0

    for _ in range(epochs):
        model.train()
        row_perm = torch.randperm(n)
        pair_pos, pair_neg = sample_pairs(groups, rng)
        if len(pair_pos):
            pair_perm = rng.permutation(len(pair_pos))
            pair_pos = pair_pos[pair_perm]
            pair_neg = pair_neg[pair_perm]

        for step, begin in enumerate(range(0, n, bs)):
            idx = row_perm[begin:begin + bs]
            opt.zero_grad()
            point_loss = bce(model(Xt[idx]), yt[idx])
            loss = point_loss
            if len(pair_pos):
                pair_begin = (step * bs) % len(pair_pos)
                pair_indices = (np.arange(bs, dtype=np.int64) + pair_begin) % len(pair_pos)
                pos_idx = torch.from_numpy(pair_pos[pair_indices])
                neg_idx = torch.from_numpy(pair_neg[pair_indices])
                pos_score = model(Xt[pos_idx])
                neg_score = model(Xt[neg_idx])
                pair_loss = torch.nn.functional.softplus(-(pos_score - neg_score)).mean()
                loss = point_loss + 0.25 * pair_loss
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            scores = np.concatenate([
                model(Xv[i:i + 65536]).numpy()
                for i in range(0, len(Xv), 65536)
            ])
        metrics = evaluate(data["val_users"], data["yv"].astype(int), scores)
        primary = float(metrics["primary"])
        if primary > best + 1e-6:
            best = primary
            best_scores = scores.copy()
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break

    os.makedirs(args.out_dir, exist_ok=True)
    metrics = evaluate(data["val_users"], data["yv"].astype(int), best_scores)
    output_metrics = {
        "gauc": float(metrics["GAUC"] if "GAUC" in metrics else metrics["gauc"]),
        "ndcg5": float(metrics.get("nDCG@5", metrics.get("ndcg5"))),
        "primary": float(metrics["primary"]),
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(output_metrics, fh)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(best_scores):
            writer.writerow([i, data["val_users"][i], data["val_videos"][i], f"{score:.6g}"])


if __name__ == "__main__":
    main()
