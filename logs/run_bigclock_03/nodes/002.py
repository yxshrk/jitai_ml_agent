"""Three-member seed ensemble of the official-parity rank-16 FM baseline."""
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


def read_csv_data(data_dir):
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")

    train_rows = []
    with open(train_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            train_rows.append({
                "user_id": row["user_id"],
                "video_id": row["video_id"],
                "tab": row["tab"],
                "duration_ms": float(row["duration_ms"]),
                "long_view": float(row["long_view"]),
            })

    val_rows = []
    with open(val_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            val_rows.append({
                "user_id": row["user_id"],
                "video_id": row["video_id"],
                "tab": row["tab"],
                "duration_ms": float(row["duration_ms"]),
                "long_view": float(row["long_view"]),
            })

    def make_map(values):
        return {value: i + 1 for i, value in enumerate(sorted(set(values)))}

    user_map = make_map([r["user_id"] for r in train_rows])
    video_map = make_map([r["video_id"] for r in train_rows])
    tab_map = make_map([r["tab"] for r in train_rows])
    train_duration = np.asarray([r["duration_ms"] for r in train_rows], dtype=np.float64)
    edges = np.quantile(train_duration, np.linspace(0.1, 0.9, 9))

    field_dims = np.asarray([
        len(user_map) + 1,
        len(video_map) + 1,
        1,
        len(tab_map) + 1,
        10,
    ], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1]))

    def encode(rows):
        x = np.empty((len(rows), 5), dtype=np.int64)
        for i, row in enumerate(rows):
            x[i, 0] = user_map.get(row["user_id"], 0)
            x[i, 1] = video_map.get(row["video_id"], 0)
            x[i, 2] = 0
            x[i, 3] = tab_map.get(row["tab"], 0)
            x[i, 4] = int(np.searchsorted(edges, row["duration_ms"], side="right"))
        x += offsets[None, :]
        return x

    return {
        "Xt": encode(train_rows),
        "yt": np.asarray([r["long_view"] for r in train_rows], dtype=np.float32),
        "Xv": encode(val_rows),
        "yv": np.asarray([r["long_view"] for r in val_rows], dtype=np.float32),
        "users": np.asarray([r["user_id"] for r in val_rows]),
        "videos": np.asarray([r["video_id"] for r in val_rows]),
        "field_dims": field_dims,
        "fast": False,
    }


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        tr = np.load(train_npz)
        va = np.load(val_npz)
        return {
            "Xt": tr["X"].astype(np.int64),
            "yt": tr["y"].astype(np.float32),
            "Xv": va["X"].astype(np.int64),
            "yv": va["y"].astype(np.float32),
            "users": va["user"],
            "videos": va["X"][:, 1],
            "field_dims": tr["field_dims"].astype(np.int64),
            "fast": True,
        }
    return read_csv_data(data_dir)


def rank_within_user(users, scores):
    _, inverse = np.unique(users, return_inverse=True)
    n = len(scores)
    order = np.lexsort((np.arange(n), scores, inverse))
    sorted_groups = inverse[order]
    boundaries = np.empty(n, dtype=bool)
    boundaries[0] = True
    boundaries[1:] = sorted_groups[1:] != sorted_groups[:-1]
    starts = np.flatnonzero(boundaries)
    lengths = np.diff(np.append(starts, n))
    group_starts = np.repeat(starts, lengths)
    ranks = np.arange(n, dtype=np.float64) - group_starts
    denominators = np.maximum(lengths - 1, 1)
    normalized = ranks / np.repeat(denominators, lengths)
    normalized[np.repeat(lengths, lengths) == 1] = 0.5
    result = np.empty(n, dtype=np.float64)
    result[order] = normalized
    return result


def train_member(seed, epochs, total_dim, xt, yt, xv, users, yv, evaluate):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = FM(total_dim, k=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()
    n = len(yt)
    batch_size = 8192
    best_primary = -1.0
    best_scores = None
    patience = 0
    history = []

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(n)
        last_loss = 0.0
        for start in range(0, n, batch_size):
            indices = permutation[start:start + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(xt[indices]), yt[indices])
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())

        model.eval()
        with torch.no_grad():
            scores = np.concatenate([
                model(xv[start:start + 65536]).cpu().numpy()
                for start in range(0, len(xv), 65536)
            ])
        metrics = evaluate(users, yv.astype(int), scores)
        primary = float(metrics["primary"])
        history.append({
            "epoch": epoch + 1,
            "train_loss": round(last_loss, 5),
            "val_gauc": round(float(metrics.get("GAUC", metrics.get("gauc", 0.0))), 6),
            "val_primary": round(primary, 6),
        })
        if primary > best_primary + 1e-6:
            best_primary = primary
            best_scores = scores.copy()
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break

    return best_scores, best_primary, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--n-members", type=int, default=3)
    args = parser.parse_args()

    epochs = args.epochs
    smoke_epochs = os.environ.get("SMOKE_EPOCHS")
    if smoke_epochs is not None:
        epochs = min(epochs, max(1, int(smoke_epochs)))
    n_members = max(1, args.n_members)

    data = load_data(args.data_dir)
    if data["fast"]:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate

    xt = torch.from_numpy(data["Xt"])
    yt = torch.from_numpy(data["yt"])
    xv = torch.from_numpy(data["Xv"])
    total_dim = int(data["field_dims"].sum())

    member_scores = []
    member_history = []
    for member_index in range(n_members):
        member_seed = args.seed + member_index
        scores, primary, epochs_history = train_member(
            member_seed,
            epochs,
            total_dim,
            xt,
            yt,
            xv,
            data["users"],
            data["yv"],
            evaluate,
        )
        member_scores.append(scores)
        member_history.append({
            "member": member_index + 1,
            "seed": member_seed,
            "best_primary": primary,
            "epochs": epochs_history,
        })

    ranked = [rank_within_user(data["users"], scores) for scores in member_scores]
    ensemble_scores = np.mean(np.stack(ranked, axis=0), axis=0)
    metrics = evaluate(data["users"], data["yv"].astype(int), ensemble_scores)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump({
            "gauc": float(metrics.get("GAUC", metrics.get("gauc"))),
            "ndcg5": float(metrics.get("nDCG@5", metrics.get("ndcg5"))),
            "primary": float(metrics["primary"]),
            "history": member_history,
            "ensemble": {
                "method": "within_user_rank_average",
                "n_members": n_members,
                "seeds": [args.seed + i for i in range(n_members)],
            },
        }, fh)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w") as fh:
        fh.write("row_id,user_id,video_id,score\n")
        for i, score in enumerate(ensemble_scores):
            fh.write(f"{i},{data['users'][i]},{data['videos'][i]},{score:.9g}\n")


if __name__ == "__main__":
    main()
