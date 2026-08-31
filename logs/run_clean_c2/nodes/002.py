"""FM baseline with temporally local within-user pair sampling.

Adds a hybrid BCE/BPR objective to the official five-field FM. Each epoch draws one
negative per positive; 70% of draws use an exp(-day_distance/2) kernel among
opposite-label impressions within three days, with uniform fallback, and 30% are
uniform. Validation checkpointing uses the official primary metric.
"""
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


def load_csv_data(data_dir):
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")

    def read_rows(path, training):
        rows = []
        with open(path, "r", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                item = {
                    "user": row["user_id"],
                    "video": row["video_id"],
                    "tab": row["tab"],
                    "duration": float(row["duration_ms"]),
                    "date": row["date"],
                    "y": float(row["long_view"]),
                }
                rows.append(item)
        return rows

    train_rows = read_rows(train_path, True)
    val_rows = read_rows(val_path, False)
    durations = np.asarray([r["duration"] for r in train_rows], dtype=np.float64)
    quantiles = np.quantile(durations, np.linspace(0.1, 0.9, 9))

    maps = {}
    for field in ("user", "video", "tab"):
        values = sorted({r[field] for r in train_rows})
        maps[field] = {v: i for i, v in enumerate(values)}

    field_dims = np.asarray([
        len(maps["user"]) + 1,
        len(maps["video"]) + 1,
        1,
        len(maps["tab"]) + 1,
        10,
    ], dtype=np.int64)
    offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(field_dims)[:-1]))

    def encode(rows):
        x = np.empty((len(rows), 5), dtype=np.int64)
        for i, row in enumerate(rows):
            x[i, 0] = maps["user"].get(row["user"], len(maps["user"]))
            x[i, 1] = maps["video"].get(row["video"], len(maps["video"]))
            x[i, 2] = 0
            x[i, 3] = maps["tab"].get(row["tab"], len(maps["tab"]))
            x[i, 4] = int(np.searchsorted(quantiles, row["duration"], side="right"))
        x += offsets[None, :]
        return {
            "X": x.astype(np.int32),
            "y": np.asarray([r["y"] for r in rows], dtype=np.float32),
            "user": np.asarray([r["user"] for r in rows]),
            "video": np.asarray([r["video"] for r in rows]),
            "date": np.asarray([r["date"] for r in rows]),
            "field_dims": field_dims,
        }

    return encode(train_rows), encode(val_rows), False


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        tr_file = np.load(train_npz)
        va_file = np.load(val_npz)
        tr = {k: tr_file[k] for k in tr_file.files}
        va = {k: va_file[k] for k in va_file.files}
        tr_file.close()
        va_file.close()
        return tr, va, True
    return load_csv_data(data_dir)


def ordinal_days(values):
    values = np.asarray(values)
    unique = np.unique(values)
    unique = np.sort(unique)
    mapping = {v: i for i, v in enumerate(unique.tolist())}
    return np.asarray([mapping[v] for v in values.tolist()], dtype=np.int32)


def build_pair_groups(user_codes, labels, days):
    order = np.argsort(user_codes, kind="stable")
    sorted_users = user_codes[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    groups = []
    for j in range(len(boundaries) - 1):
        idx = order[boundaries[j]:boundaries[j + 1]]
        pos = idx[labels[idx] >= 0.5]
        neg = idx[labels[idx] < 0.5]
        if len(pos) and len(neg):
            groups.append((pos.astype(np.int64), neg.astype(np.int64), days[neg]))
    return groups


def sample_temporal_pairs(groups, days, row_weights, rng):
    total = sum(len(g[0]) for g in groups)
    pos_out = np.empty(total, dtype=np.int64)
    neg_out = np.empty(total, dtype=np.int64)
    weight_out = np.empty(total, dtype=np.float32)
    cursor = 0
    for positives, negatives, negative_days in groups:
        count = len(positives)
        uniform_choices = rng.integers(0, len(negatives), size=count)
        chosen = negatives[uniform_choices].copy()
        local_draw = rng.random(count) < 0.70
        for j in np.flatnonzero(local_draw):
            pos_day = days[positives[j]]
            mask = np.abs(negative_days - pos_day) <= 3
            candidate_locs = np.flatnonzero(mask)
            if len(candidate_locs):
                distances = np.abs(negative_days[candidate_locs] - pos_day).astype(np.float64)
                probabilities = np.exp(-distances / 2.0)
                probabilities /= probabilities.sum()
                chosen[j] = negatives[rng.choice(candidate_locs, p=probabilities)]
        end = cursor + count
        pos_out[cursor:end] = positives
        neg_out[cursor:end] = chosen
        weight_out[cursor:end] = np.sqrt(row_weights[positives] * row_weights[chosen])
        cursor = end
    return pos_out, neg_out, weight_out


def evaluate_scores(evaluate_fn, users, labels, scores):
    result = evaluate_fn(users, labels.astype(int), scores)
    return {
        "GAUC": float(result.get("GAUC", result.get("gauc"))),
        "nDCG@5": float(result.get("nDCG@5", result.get("ndcg5"))),
        "primary": float(result["primary"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()

    epochs = args.epochs
    smoke = os.environ.get("SMOKE_EPOCHS")
    if smoke is not None:
        epochs = min(epochs, max(1, int(smoke)))

    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr, va, fast_path = load_data(args.data_dir)
    if fast_path:
        from data.official.evaluate import evaluate as evaluate_fn
    else:
        from harness.evaluate_provisional import evaluate as evaluate_fn

    x_train_np = tr["X"].astype(np.int64, copy=False)
    y_train_np = tr["y"].astype(np.float32, copy=False)
    x_val_np = va["X"].astype(np.int64, copy=False)
    y_val_np = va["y"].astype(np.float32, copy=False)
    field_dims = tr["field_dims"].astype(np.int64, copy=False)
    total_dim = int(field_dims.sum())

    train_days = ordinal_days(tr["date"])
    user_codes = x_train_np[:, 0]
    row_weights = np.ones(len(y_train_np), dtype=np.float32)
    groups = build_pair_groups(user_codes, y_train_np, train_days)

    x_train = torch.from_numpy(x_train_np).to(device)
    y_train = torch.from_numpy(y_train_np).to(device)
    x_val = torch.from_numpy(x_val_np).to(device)

    model = FM(total_dim, k=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = torch.nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(args.seed)
    batch_size = 8192
    n = len(y_train_np)

    best_primary = -1.0
    best_scores = None
    patience = 0
    history = []

    for epoch in range(epochs):
        pos_np, neg_np, pair_weight_np = sample_temporal_pairs(
            groups, train_days, row_weights, rng
        )
        row_perm = torch.randperm(n, device=device)
        pair_order_np = rng.permutation(len(pos_np))
        pos_np = pos_np[pair_order_np]
        neg_np = neg_np[pair_order_np]
        pair_weight_np = pair_weight_np[pair_order_np]

        row_batches = (n + batch_size - 1) // batch_size
        pair_batches = (len(pos_np) + batch_size - 1) // batch_size
        steps = max(row_batches, pair_batches)
        epoch_loss = 0.0
        updates = 0
        model.train()

        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = None

            row_start = step * batch_size
            if row_start < n:
                idx = row_perm[row_start:row_start + batch_size]
                point_loss = bce(model(x_train[idx]), y_train[idx])
                loss = 0.5 * point_loss

            pair_start = step * batch_size
            if pair_start < len(pos_np):
                pair_end = min(pair_start + batch_size, len(pos_np))
                pidx = torch.from_numpy(pos_np[pair_start:pair_end]).to(device)
                nidx = torch.from_numpy(neg_np[pair_start:pair_end]).to(device)
                pweight = torch.from_numpy(pair_weight_np[pair_start:pair_end]).to(device)
                difference = model(x_train[pidx]) - model(x_train[nidx])
                pair_loss = (torch.nn.functional.softplus(-difference) * pweight).sum() / pweight.sum().clamp_min(1e-8)
                if loss is None:
                    loss = 0.5 * pair_loss
                else:
                    loss = loss + 0.5 * pair_loss

            if loss is not None:
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach().cpu())
                updates += 1

        model.eval()
        score_parts = []
        with torch.no_grad():
            for start in range(0, len(x_val_np), 65536):
                score_parts.append(model(x_val[start:start + 65536]).detach().cpu().numpy())
        scores = np.concatenate(score_parts).astype(np.float64, copy=False)
        metrics = evaluate_scores(evaluate_fn, va["user"], y_val_np, scores)
        history.append({
            "epoch": epoch + 1,
            "train_loss": round(epoch_loss / max(updates, 1), 6),
            "val_gauc": round(metrics["GAUC"], 6),
            "val_primary": round(metrics["primary"], 6),
            "pairs": int(len(pos_np)),
            "local_probability": 0.70,
            "kernel_half_scale_days": 2.0,
        })

        if metrics["primary"] > best_primary + 1e-6:
            best_primary = metrics["primary"]
            best_scores = scores.copy()
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break

    final_metrics = evaluate_scores(evaluate_fn, va["user"], y_val_np, best_scores)
    output_metrics = {
        "gauc": final_metrics["GAUC"],
        "ndcg5": final_metrics["nDCG@5"],
        "primary": final_metrics["primary"],
        "history": history,
        "config": {
            "model": "FM",
            "embedding_dim": 16,
            "objective": "0.5_bce_plus_0.5_temporal_bpr",
            "local_draw_probability": 0.70,
            "local_window_days": 3,
            "temporal_kernel": "exp(-abs(day_pos-day_neg)/2)",
            "pairs_per_positive": 1,
            "redraw_each_epoch": True,
            "seed": args.seed,
        },
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(output_metrics, fh)

    if "video" in va:
        val_videos = va["video"]
    else:
        val_videos = np.zeros(len(best_scores), dtype=np.int64)
    with open(os.path.join(args.out_dir, "predictions.csv"), "w") as fh:
        fh.write("row_id,user_id,video_id,score\n")
        for i, score in enumerate(best_scores):
            fh.write(f"{i},{va['user'][i]},{val_videos[i]},{score:.8g}\n")


if __name__ == "__main__":
    main()
