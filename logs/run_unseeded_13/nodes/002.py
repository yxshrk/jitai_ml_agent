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


def load_csv_split(path):
    with open(path, "r", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        pos = {name: i for i, name in enumerate(header)}
        required = ["user_id", "video_id", "tab", "duration_ms", "long_view"]
        for name in required:
            if name not in pos:
                raise ValueError("missing required column")
        users, videos, tabs, durations, labels = [], [], [], [], []
        for row in reader:
            users.append(row[pos["user_id"]])
            videos.append(row[pos["video_id"]])
            tabs.append(row[pos["tab"]])
            durations.append(float(row[pos["duration_ms"]] or 0.0))
            labels.append(float(row[pos["long_view"]] or 0.0))
    return {
        "user_raw": np.asarray(users),
        "video_raw": np.asarray(videos),
        "tab_raw": np.asarray(tabs),
        "duration": np.asarray(durations, dtype=np.float64),
        "y": np.asarray(labels, dtype=np.float32),
    }


def make_map(values):
    return {v: i + 1 for i, v in enumerate(sorted(set(values.tolist())))}


def encode(values, mapping):
    return np.fromiter(
        (mapping.get(v, 0) for v in values),
        dtype=np.int64,
        count=len(values),
    )


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        with np.load(train_npz) as trf:
            xt = trf["X"].astype(np.int64, copy=False)
            yt = trf["y"].astype(np.float32, copy=False)
            field_dims = trf["field_dims"].copy()
        with np.load(val_npz) as vaf:
            xv = vaf["X"].astype(np.int64, copy=False)
            yv = vaf["y"].astype(np.int32, copy=False)
            users = vaf["user"].copy()
        return {
            "Xt": xt,
            "yt": yt,
            "Xv": xv,
            "yv": yv,
            "users": users,
            "videos": np.zeros(len(yv), dtype=np.int64),
            "total_dim": int(np.asarray(field_dims).sum()),
            "official": True,
        }

    tr = load_csv_split(os.path.join(data_dir, "train.csv"))
    va = load_csv_split(os.path.join(data_dir, "val.csv"))
    user_map = make_map(tr["user_raw"])
    video_map = make_map(tr["video_raw"])
    tab_map = make_map(tr["tab_raw"])
    quantiles = np.linspace(0.0, 1.0, 11)
    cuts = np.unique(np.quantile(tr["duration"], quantiles)[1:-1])

    def build(split):
        user = encode(split["user_raw"], user_map)
        video = encode(split["video_raw"], video_map)
        author = np.zeros(len(user), dtype=np.int64)
        tab = encode(split["tab_raw"], tab_map)
        duration = np.searchsorted(
            cuts, split["duration"], side="right"
        ).astype(np.int64)
        return np.column_stack((user, video, author, tab, duration))

    dims = np.asarray(
        [
            len(user_map) + 1,
            len(video_map) + 1,
            1,
            len(tab_map) + 1,
            len(cuts) + 1,
        ],
        dtype=np.int64,
    )
    offsets = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(dims)[:-1])
    )
    return {
        "Xt": build(tr) + offsets,
        "yt": tr["y"],
        "Xv": build(va) + offsets,
        "yv": va["y"].astype(np.int32),
        "users": va["user_raw"],
        "videos": va["video_raw"],
        "total_dim": int(dims.sum()),
        "official": False,
    }


def metric_values(result):
    return {
        "gauc": float(result["GAUC"] if "GAUC" in result else result["gauc"]),
        "ndcg5": float(
            result["nDCG@5"] if "nDCG@5" in result else result["ndcg5"]
        ),
        "primary": float(result["primary"]),
    }


def per_user_ranks(scores, users):
    scores = np.asarray(scores, dtype=np.float64)
    users = np.asarray(users)
    if scores.ndim != 1 or len(scores) != len(users):
        raise ValueError("prediction and user arrays are not aligned")
    if not np.all(np.isfinite(scores)):
        raise ValueError("non-finite prediction")
    n = len(scores)
    if n == 0:
        return scores.copy()

    _, inverse = np.unique(users, return_inverse=True)
    order = np.lexsort((-scores, inverse))
    sorted_groups = inverse[order]
    sorted_scores = scores[order]
    positions = np.arange(n, dtype=np.int64)

    group_starts_mask = np.empty(n, dtype=bool)
    group_starts_mask[0] = True
    group_starts_mask[1:] = sorted_groups[1:] != sorted_groups[:-1]
    group_starts = np.maximum.accumulate(
        np.where(group_starts_mask, positions, 0)
    )

    tie_starts_mask = np.empty(n, dtype=bool)
    tie_starts_mask[0] = True
    tie_starts_mask[1:] = (
        (sorted_groups[1:] != sorted_groups[:-1])
        | (sorted_scores[1:] != sorted_scores[:-1])
    )
    tie_starts = positions[tie_starts_mask]
    tie_ends = np.concatenate((tie_starts[1:], np.asarray([n], dtype=np.int64)))
    tie_lengths = tie_ends - tie_starts
    tie_midpoints = (tie_starts + tie_ends - 1).astype(np.float64) * 0.5
    midpoint_by_position = np.repeat(tie_midpoints, tie_lengths)

    local_midpoints = midpoint_by_position - group_starts
    counts = np.bincount(inverse)
    group_counts = counts[sorted_groups]
    denominators = np.maximum(group_counts - 1, 1)

    ranked_sorted = 1.0 - local_midpoints / denominators
    ranked_sorted[group_counts == 1] = 0.5
    ranked = np.empty(n, dtype=np.float64)
    ranked[order] = ranked_sorted
    return ranked


def rank_ensemble(predictions, users):
    if len(predictions) == 0:
        raise ValueError("ensemble has no members")
    if len(predictions) == 1:
        return np.asarray(predictions[0], dtype=np.float64).copy()
    ranked = [per_user_ranks(p, users) for p in predictions]
    return np.mean(np.stack(ranked, axis=0), axis=0)


def train_member(seed, epochs, total_dim, xt, yt, xv, users, yv, evaluate):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = FM(total_dim)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = torch.nn.BCEWithLogitsLoss()
    n = len(yt)
    batch_size = 8192
    best_primary = -1.0
    best_scores = None
    patience = 0

    for _ in range(epochs):
        model.train()
        permutation = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = permutation[start:start + batch_size]
            opt.zero_grad()
            loss = bce(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            scores = np.concatenate(
                [
                    model(xv[start:start + 65536]).cpu().numpy()
                    for start in range(0, len(xv), 65536)
                ]
            )
        result = metric_values(evaluate(users, yv, scores))
        if result["primary"] > best_primary + 1e-6:
            best_primary = result["primary"]
            best_scores = scores.copy()
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break

    if best_scores is None:
        raise RuntimeError("member training produced no checkpoint")
    return best_scores, best_primary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    epochs = args.epochs
    if "SMOKE_EPOCHS" in os.environ:
        epochs = min(epochs, max(1, int(os.environ["SMOKE_EPOCHS"])))

    data = load_data(args.data_dir)
    if data["official"]:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate

    xt = torch.from_numpy(np.asarray(data["Xt"], dtype=np.int64))
    yt = torch.from_numpy(np.asarray(data["yt"], dtype=np.float32))
    xv = torch.from_numpy(np.asarray(data["Xv"], dtype=np.int64))

    predictions = []
    primaries = []
    for member in range(4):
        scores, primary = train_member(
            args.seed + member,
            epochs,
            data["total_dim"],
            xt,
            yt,
            xv,
            data["users"],
            data["yv"],
            evaluate,
        )
        predictions.append(scores)
        primaries.append(primary)

    best_member = max(primaries)
    retained = [
        predictions[i]
        for i in range(4)
        if primaries[i] >= best_member - 0.002
    ]
    if not retained:
        raise RuntimeError("member filtering retained no predictions")

    probe = retained[0]
    probe_metric = metric_values(evaluate(data["users"], data["yv"], probe))
    singleton = rank_ensemble([probe], data["users"])
    duplicate = rank_ensemble([probe, probe], data["users"])
    singleton_metric = metric_values(
        evaluate(data["users"], data["yv"], singleton)
    )
    duplicate_metric = metric_values(
        evaluate(data["users"], data["yv"], duplicate)
    )
    if not np.array_equal(singleton, np.asarray(probe, dtype=np.float64)):
        raise RuntimeError("singleton prediction invariance failed")
    if abs(singleton_metric["primary"] - probe_metric["primary"]) > 1e-10:
        raise RuntimeError("singleton ensemble invariance failed")
    if abs(duplicate_metric["primary"] - probe_metric["primary"]) > 1e-10:
        raise RuntimeError("duplicate ensemble invariance failed")

    scores = rank_ensemble(retained, data["users"])
    reversed_scores = rank_ensemble(list(reversed(retained)), data["users"])
    if not np.allclose(scores, reversed_scores, rtol=0.0, atol=1e-12):
        raise RuntimeError("member ordering invariance failed")

    metrics = metric_values(evaluate(data["users"], data["yv"], scores))
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh)
    with open(
        os.path.join(args.out_dir, "predictions.csv"), "w", newline=""
    ) as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(scores):
            writer.writerow(
                [
                    i,
                    data["users"][i],
                    data["videos"][i],
                    format(float(score), ".10g"),
                ]
            )


if __name__ == "__main__":
    main()
