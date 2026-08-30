"""Five-seed FM ensemble with short/long duration regime heads."""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DurationRegimeFM(torch.nn.Module):
    def __init__(self, total_dim, num_fields=5, k=16):
        super().__init__()
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.regime_heads = torch.nn.Embedding(2, num_fields * k + 1)
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)
        torch.nn.init.zeros_(self.regime_heads.weight)

    def forward(self, x, regime):
        e = self.emb(x)
        summed = e.sum(1)
        pair = 0.5 * (summed * summed - (e * e).sum(1)).sum(1)
        shared_logit = self.bias + self.lin(x).sum((1, 2)) + pair
        representation = e.reshape(e.shape[0], -1)
        head = self.regime_heads(regime)
        residual = (representation * head[:, :-1]).sum(1) + head[:, -1]
        return shared_logit + residual

    def head_penalty(self):
        return self.regime_heads.weight.square().mean()


def load_csv_data(data_dir):
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")

    with open(train_path, "r", newline="") as fh:
        train_rows = list(csv.DictReader(fh))
    with open(val_path, "r", newline="") as fh:
        val_rows = list(csv.DictReader(fh))

    def make_map(rows, name):
        values = sorted({row[name] for row in rows})
        return {value: i + 1 for i, value in enumerate(values)}

    user_map = make_map(train_rows, "user_id")
    video_map = make_map(train_rows, "video_id")
    tab_map = make_map(train_rows, "tab")

    train_duration = np.asarray(
        [float(row.get("duration_ms", 0.0) or 0.0) for row in train_rows],
        dtype=np.float64,
    )
    val_duration = np.asarray(
        [float(row.get("duration_ms", 0.0) or 0.0) for row in val_rows],
        dtype=np.float64,
    )
    quantiles = np.quantile(train_duration, np.linspace(0.1, 0.9, 9))

    field_dims = np.asarray(
        [len(user_map) + 1, len(video_map) + 1, 2, len(tab_map) + 1, 10],
        dtype=np.int64,
    )
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1]))

    def encode(rows):
        x = np.zeros((len(rows), 5), dtype=np.int64)
        for i, row in enumerate(rows):
            duration = float(row.get("duration_ms", 0.0) or 0.0)
            x[i, 0] = user_map.get(row["user_id"], 0)
            x[i, 1] = video_map.get(row["video_id"], 0)
            x[i, 2] = 0
            x[i, 3] = tab_map.get(row["tab"], 0)
            x[i, 4] = int(np.searchsorted(quantiles, duration, side="right"))
        x += offsets[None, :]
        return x

    return {
        "Xt": encode(train_rows),
        "yt": np.asarray([float(row["long_view"]) for row in train_rows], dtype=np.float32),
        "rt": (train_duration > 18000.0).astype(np.int64),
        "Xv": encode(val_rows),
        "yv": np.asarray([float(row["long_view"]) for row in val_rows], dtype=np.float32),
        "rv": (val_duration > 18000.0).astype(np.int64),
        "users": np.asarray([row["user_id"] for row in val_rows]),
        "videos": np.asarray([row["video_id"] for row in val_rows]),
        "field_dims": field_dims,
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
            "rt": (tr["duration_ms"].astype(np.float64) > 18000.0).astype(np.int64),
            "Xv": va["X"].astype(np.int64),
            "yv": va["y"].astype(np.float32),
            "rv": (va["duration_ms"].astype(np.float64) > 18000.0).astype(np.int64),
            "users": va["user"],
            "videos": va["X"][:, 1].astype(np.int64) - video_offset,
            "field_dims": field_dims,
            "fast": True,
        }
    return load_csv_data(data_dir)


def metric_evaluator(fast):
    if fast:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    return evaluate


def train_member(Xt, yt, rt, Xv, rv, users, yv, total_dim, seed, epochs, evaluate):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DurationRegimeFM(total_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()
    n = len(yt)
    batch_size = 8192
    best_primary = -1.0
    best_scores = None
    patience = 0
    head_l2 = 1e-2

    for _ in range(epochs):
        model.train()
        permutation = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = permutation[start:start + batch_size]
            optimizer.zero_grad()
            logits = model(Xt[idx], rt[idx])
            loss = criterion(logits, yt[idx]) + head_l2 * model.head_penalty()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            scores = np.concatenate([
                model(Xv[start:start + 65536], rv[start:start + 65536]).cpu().numpy()
                for start in range(0, len(Xv), 65536)
            ])
        metrics = evaluate(users, yv.astype(int), scores)
        primary = float(metrics["primary"])
        if primary > best_primary + 1e-6:
            best_primary = primary
            best_scores = scores.copy()
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break
    return best_scores


def per_user_ranks(users, scores):
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    result = np.empty(len(scores), dtype=np.float64)
    boundaries = np.flatnonzero(
        np.concatenate(([True], sorted_users[1:] != sorted_users[:-1], [True]))
    )
    for j in range(len(boundaries) - 1):
        positions = order[boundaries[j]:boundaries[j + 1]]
        count = len(positions)
        if count == 1:
            result[positions[0]] = 0.5
            continue
        local_order = np.argsort(scores[positions], kind="mergesort")
        local_ranks = np.empty(count, dtype=np.float64)
        local_ranks[local_order] = np.arange(count, dtype=np.float64) / float(count - 1)
        result[positions] = local_ranks
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()

    epochs = args.epochs
    smoke_epochs = os.environ.get("SMOKE_EPOCHS")
    if smoke_epochs is not None:
        epochs = min(epochs, int(smoke_epochs))
    epochs = max(1, epochs)

    data = load_data(args.data_dir)
    evaluate = metric_evaluator(data["fast"])
    Xt = torch.from_numpy(data["Xt"])
    yt = torch.from_numpy(data["yt"])
    rt = torch.from_numpy(data["rt"])
    Xv = torch.from_numpy(data["Xv"])
    rv = torch.from_numpy(data["rv"])
    total_dim = int(data["field_dims"].sum())

    rank_sum = np.zeros(len(data["yv"]), dtype=np.float64)
    for member_index in range(5):
        scores = train_member(
            Xt, yt, rt, Xv, rv, data["users"], data["yv"], total_dim,
            args.seed + member_index, epochs, evaluate,
        )
        rank_sum += per_user_ranks(data["users"], scores)
    ensemble_scores = rank_sum / 5.0

    metrics = evaluate(data["users"], data["yv"].astype(int), ensemble_scores)
    gauc = metrics["GAUC"] if "GAUC" in metrics else metrics["gauc"]
    ndcg5 = metrics.get("nDCG@5", metrics.get("ndcg5"))

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump({
            "gauc": float(gauc),
            "ndcg5": float(ndcg5),
            "primary": float(metrics["primary"]),
        }, fh)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(ensemble_scores):
            writer.writerow([i, data["users"][i], data["videos"][i], format(float(score), ".10g")])


if __name__ == "__main__":
    main()
