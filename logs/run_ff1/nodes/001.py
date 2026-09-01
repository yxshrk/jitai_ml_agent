"""Leakage-safe multi-mechanism screen followed by full-fidelity FM training."""
import argparse
import csv
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ScreenFM(torch.nn.Module):
    def __init__(self, total_dim, fields=5, k=16, dropout=0.0, deep=False,
                 aggregate=False):
        super().__init__()
        self.fields = fields
        self.k = k
        self.aggregate = aggregate
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.drop = torch.nn.Dropout(dropout)
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)
        self.deep = None
        if deep:
            width = 64
            self.deep = torch.nn.Sequential(
                torch.nn.Linear(fields * k, width),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(width, 1),
            )
            for layer in self.deep:
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(layer.weight)
                    torch.nn.init.zeros_(layer.bias)
        if aggregate:
            self.aggregate_weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x, aggregate_value=None):
        e0 = self.emb(x)
        e = self.drop(e0)
        s = e.sum(1)
        pair = 0.5 * (s * s - (e * e).sum(1)).sum(1)
        out = self.bias + self.lin(x).sum((1, 2)) + pair
        if self.deep is not None:
            out = out + self.deep(e.reshape(len(x), -1)).squeeze(1)
        if self.aggregate and aggregate_value is not None:
            out = out + self.aggregate_weight * aggregate_value
        return out


def encode_csv(train_path, val_path):
    feature_names = ["user_id", "video_id", "tab", "duration_ms"]

    def read_rows(path, validation):
        rows = []
        with open(path, "r", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                item = {
                    "user_id": row["user_id"],
                    "video_id": row["video_id"],
                    "tab": row["tab"],
                    "duration_ms": float(row["duration_ms"]),
                    "date": row.get("date", "0"),
                    "hourmin": row.get("hourmin", "0"),
                    "long_view": float(row["long_view"]),
                }
                rows.append(item)
        return rows

    tr_rows = read_rows(train_path, False)
    va_rows = read_rows(val_path, True)
    all_rows = tr_rows + va_rows
    maps = {}
    for name in feature_names[:3]:
        vals = sorted({r[name] for r in all_rows})
        maps[name] = {v: i for i, v in enumerate(vals)}
    durations = np.asarray([r["duration_ms"] for r in tr_rows], dtype=np.float64)
    positive = durations[durations > 0]
    if len(positive):
        edges = np.unique(np.quantile(np.log1p(positive), np.linspace(0, 1, 33)[1:-1]))
    else:
        edges = np.asarray([], dtype=np.float64)

    field_dims = np.asarray([
        len(maps["user_id"]), len(maps["video_id"]), len(maps["video_id"]),
        len(maps["tab"]), len(edges) + 1
    ], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1]))

    def convert(rows):
        x = np.empty((len(rows), 5), dtype=np.int64)
        for i, r in enumerate(rows):
            video = maps["video_id"][r["video_id"]]
            bucket = int(np.searchsorted(edges, math.log1p(max(0.0, r["duration_ms"]))))
            local = [maps["user_id"][r["user_id"]], video, video,
                     maps["tab"][r["tab"]], bucket]
            x[i] = np.asarray(local, dtype=np.int64) + offsets
        return {
            "X": x,
            "y": np.asarray([r["long_view"] for r in rows], dtype=np.float32),
            "user": np.asarray([r["user_id"] for r in rows]),
            "video": np.asarray([r["video_id"] for r in rows]),
            "date": np.asarray([r["date"] for r in rows]),
            "hourmin": np.asarray([r["hourmin"] for r in rows]),
            "field_dims": field_dims,
        }

    return convert(tr_rows), convert(va_rows), False


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        tr_file = np.load(train_npz)
        va_file = np.load(val_npz)
        tr = {k: tr_file[k] for k in tr_file.files}
        va = {k: va_file[k] for k in va_file.files}
        va["video"] = va["X"][:, 1]
        return tr, va, True
    return encode_csv(os.path.join(data_dir, "train.csv"),
                      os.path.join(data_dir, "val.csv"))


def metric_function(fast_path):
    if fast_path:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    return evaluate


def time_axis(date_values, hour_values):
    date_text = np.asarray(date_values).astype(str)
    unique = sorted(set(date_text.tolist()))
    rank = {v: i for i, v in enumerate(unique)}
    day = np.asarray([rank[v] for v in date_text], dtype=np.float32)
    hour = np.zeros(len(day), dtype=np.float32)
    for i, value in enumerate(np.asarray(hour_values).astype(str)):
        try:
            number = int(float(value))
            hh = number // 100 if number >= 100 else number
            mm = number % 100 if number >= 100 else 0
            hour[i] = min(23, max(0, hh)) / 24.0 + min(59, max(0, mm)) / 1440.0
        except ValueError:
            hour[i] = 0.0
    return day + hour


def make_aggregate(train_x, train_y, val_x, total_dim, alpha):
    ids = train_x[:, 1].astype(np.int64)
    val_ids = val_x[:, 1].astype(np.int64)
    count = np.bincount(ids, minlength=total_dim).astype(np.float64)
    sums = np.bincount(ids, weights=train_y, minlength=total_dim).astype(np.float64)
    prior = float(np.mean(train_y))
    train_rate = (sums[ids] - train_y + alpha * prior) / np.maximum(
        count[ids] - 1.0 + alpha, 1e-8)
    val_rate = (sums[val_ids] + alpha * prior) / np.maximum(
        count[val_ids] + alpha, 1e-8)
    eps = 1e-4
    base = math.log((prior + eps) / (1.0 - prior + eps))
    train_logit = np.log(np.clip(train_rate, eps, 1.0 - eps) /
                         np.clip(1.0 - train_rate, eps, 1.0)) - base
    val_logit = np.log(np.clip(val_rate, eps, 1.0 - eps) /
                       np.clip(1.0 - val_rate, eps, 1.0)) - base
    return np.clip(train_logit, -5, 5).astype(np.float32), \
        np.clip(val_logit, -5, 5).astype(np.float32)


def make_pairs(users, labels, seed, cap=250000):
    rng = np.random.RandomState(seed)
    order = np.argsort(users, kind="mergesort")
    sorted_users = np.asarray(users)[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    positives = []
    negatives = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        idx = order[left:right]
        pos = idx[labels[idx] > 0.5]
        neg = idx[labels[idx] <= 0.5]
        if len(pos) == 0 or len(neg) == 0:
            continue
        amount = min(8, max(len(pos), len(neg)))
        positives.extend(rng.choice(pos, amount, replace=True).tolist())
        negatives.extend(rng.choice(neg, amount, replace=True).tolist())
    if len(positives) > cap:
        chosen = rng.choice(len(positives), cap, replace=False)
        positives = np.asarray(positives, dtype=np.int64)[chosen]
        negatives = np.asarray(negatives, dtype=np.int64)[chosen]
    else:
        positives = np.asarray(positives, dtype=np.int64)
        negatives = np.asarray(negatives, dtype=np.int64)
    return positives, negatives


def predict(model, x, aggregate, device):
    model.eval()
    outputs = []
    with torch.no_grad():
        for left in range(0, len(x), 65536):
            right = min(len(x), left + 65536)
            xb = torch.as_tensor(x[left:right], dtype=torch.long, device=device)
            ab = None
            if aggregate is not None:
                ab = torch.as_tensor(aggregate[left:right], dtype=torch.float32, device=device)
            outputs.append(model(xb, ab).detach().cpu().numpy())
    return np.concatenate(outputs)


def train_candidate(config, epochs, seed, arrays, evaluate, device, track_epochs=False):
    x_train, y_train, x_val, val_user, val_y, total_dim, recency, aggregates, pairs = arrays
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = ScreenFM(total_dim, k=int(config["k"]), dropout=float(config["dropout"]),
                     deep=bool(config["deep"]), aggregate=bool(config["aggregate"])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["lr"]),
                                 weight_decay=float(config["weight_decay"]))
    rng = np.random.RandomState(seed + 177)
    n = len(y_train)
    batch_size = 8192
    aggregate_train, aggregate_val = aggregates[float(config["alpha"])]
    if not config["aggregate"]:
        aggregate_train = None
        aggregate_val = None
    weights = np.ones(n, dtype=np.float32)
    half_life = float(config["half_life"])
    if half_life > 0:
        age = float(np.max(recency)) - recency
        weights = np.exp(-math.log(2.0) * age / half_life).astype(np.float32)
        weights /= max(1e-8, float(np.mean(weights)))
    pos_pairs, neg_pairs = pairs
    best_primary = -1.0
    best_scores = None
    epoch_history = []
    patience = 0
    for epoch in range(epochs):
        model.train()
        permutation = rng.permutation(n)
        last_loss = 0.0
        for left in range(0, n, batch_size):
            ids = permutation[left:left + batch_size]
            xb = torch.as_tensor(x_train[ids], dtype=torch.long, device=device)
            yb = torch.as_tensor(y_train[ids], dtype=torch.float32, device=device)
            wb = torch.as_tensor(weights[ids], dtype=torch.float32, device=device)
            ab = None
            if aggregate_train is not None:
                ab = torch.as_tensor(aggregate_train[ids], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb, ab)
            point_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, yb, reduction="none")
            loss = (point_loss * wb).mean()
            rank_lambda = float(config["rank_lambda"])
            if rank_lambda > 0 and len(pos_pairs):
                amount = min(len(ids), len(pos_pairs))
                selected = rng.randint(0, len(pos_pairs), size=amount)
                pi = pos_pairs[selected]
                ni = neg_pairs[selected]
                px = torch.as_tensor(x_train[pi], dtype=torch.long, device=device)
                nx = torch.as_tensor(x_train[ni], dtype=torch.long, device=device)
                pa = na = None
                if aggregate_train is not None:
                    pa = torch.as_tensor(aggregate_train[pi], dtype=torch.float32, device=device)
                    na = torch.as_tensor(aggregate_train[ni], dtype=torch.float32, device=device)
                difference = model(px, pa) - model(nx, na)
                loss = loss + rank_lambda * torch.nn.functional.softplus(-difference).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            last_loss = float(loss.detach().cpu())
        if track_epochs:
            scores = predict(model, x_val, aggregate_val, device)
            measured = evaluate(val_user, val_y.astype(int), scores)
            primary = float(measured["primary"])
            epoch_history.append({"epoch": epoch + 1, "train_loss": round(last_loss, 6),
                                  "val_primary": round(primary, 6)})
            if primary > best_primary + 1e-7:
                best_primary = primary
                best_scores = scores.copy()
                patience = 0
            else:
                patience += 1
                if patience >= 3:
                    break
    if not track_epochs:
        best_scores = predict(model, x_val, aggregate_val, device)
        measured = evaluate(val_user, val_y.astype(int), best_scores)
        best_primary = float(measured["primary"])
    return best_primary, best_scores, epoch_history


def config(name, k=16, dropout=0.0, weight_decay=0.0, deep=False,
           half_life=0.0, rank_lambda=0.0, aggregate=False, alpha=20.0, lr=1e-3):
    return {"name": name, "k": k, "dropout": dropout, "weight_decay": weight_decay,
            "deep": deep, "half_life": half_life, "rank_lambda": rank_lambda,
            "aggregate": aggregate, "alpha": alpha, "lr": lr}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    tr, va, fast_path = load_data(args.data_dir)
    evaluate = metric_function(fast_path)
    x_train = np.asarray(tr["X"], dtype=np.int64)
    y_train = np.asarray(tr["y"], dtype=np.float32)
    x_val = np.asarray(va["X"], dtype=np.int64)
    val_y = np.asarray(va["y"], dtype=np.float32)
    val_user = np.asarray(va["user"])
    total_dim = int(np.asarray(tr["field_dims"]).sum())
    recency = time_axis(tr["date"], tr["hourmin"])
    aggregates = {}
    for alpha in (5.0, 20.0, 100.0):
        aggregates[alpha] = make_aggregate(x_train, y_train, x_val, total_dim, alpha)
    pairs = make_pairs(np.asarray(tr["user"]), y_train, args.seed + 991)
    arrays = (x_train, y_train, x_val, val_user, val_y, total_dim,
              recency, aggregates, pairs)

    candidates = [
        config("baseline"),
        config("wd_1e-6", weight_decay=1e-6),
        config("wd_1e-5", weight_decay=1e-5),
        config("wd_1e-4", weight_decay=1e-4),
        config("drop_005", dropout=0.05),
        config("drop_015", dropout=0.15),
        config("k8", k=8),
        config("k32_reg", k=32, weight_decay=1e-5),
        config("deep_drop", deep=True, dropout=0.10, weight_decay=1e-6),
        config("deep_strong_reg", deep=True, dropout=0.25, weight_decay=1e-5),
        config("recent_3", half_life=3.0),
        config("recent_7", half_life=7.0),
        config("rank_005", rank_lambda=0.05),
        config("rank_015", rank_lambda=0.15),
        config("agg_a5", aggregate=True, alpha=5.0),
        config("agg_a20", aggregate=True, alpha=20.0),
        config("agg_a100", aggregate=True, alpha=100.0),
        config("wd_drop", dropout=0.10, weight_decay=1e-5),
        config("agg_rank", aggregate=True, alpha=20.0, rank_lambda=0.05),
        config("deep_agg_reg", deep=True, aggregate=True, alpha=20.0,
               dropout=0.15, weight_decay=1e-5),
        config("recent_agg_reg", aggregate=True, alpha=20.0, half_life=7.0,
               weight_decay=1e-5),
        config("rank_drop_reg", rank_lambda=0.05, dropout=0.10, weight_decay=1e-5),
        config("k8_agg_reg", k=8, aggregate=True, alpha=20.0, weight_decay=1e-5),
        config("k32_deep_reg", k=32, deep=True, dropout=0.20, weight_decay=1e-5),
    ]

    smoke = os.environ.get("SMOKE_EPOCHS")
    cap = int(smoke) if smoke is not None else None
    probe_epochs = 3 if cap is None else min(3, cap)
    refine_epochs = 6 if cap is None else min(6, cap)
    final_epochs = args.epochs if cap is None else min(args.epochs, cap)

    # In ultra-short smoke runs, skip search and keep the baseline path stable.
    if cap is not None and cap <= 1:
        winner = candidates[0]
        final_primary, final_scores, epoch_history = train_candidate(
            winner, final_epochs, args.seed + 1000, arrays, evaluate, device, True)
        measured = evaluate(val_user, val_y.astype(int), final_scores)
        gauc = measured["GAUC"] if "GAUC" in measured else measured["gauc"]
        ndcg = measured["nDCG@5"] if "nDCG@5" in measured else measured["ndcg5"]
        metrics = {"gauc": float(gauc), "ndcg5": float(ndcg),
                   "primary": float(measured["primary"]),
                   "history": [{"stage": "final", "epochs_requested": final_epochs,
                                "config": winner, "best_primary": round(final_primary, 6),
                                "epochs": epoch_history}],
                   "selected_config": winner}
        with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
            json.dump(metrics, fh)

        videos = np.asarray(va.get("video", x_val[:, 1]))
        with open(os.path.join(args.out_dir, "predictions.csv"), "w") as fh:
            fh.write("row_id,user_id,video_id,score\n")
            for i, score in enumerate(final_scores):
                fh.write(f"{i},{val_user[i]},{videos[i]},{float(score):.8g}\n")
        return

    if cap is not None and cap <= 1:
        candidates = candidates[:6]

    history = []
    progress_path = os.path.join(args.out_dir, "progress.log")
    probe_results = []
    for index, candidate in enumerate(candidates):
        primary, _, _ = train_candidate(candidate, probe_epochs,
                                        args.seed + 1000 + index, arrays,
                                        evaluate, device, False)
        record = {"stage": "probe", "epochs": probe_epochs,
                  "config": candidate, "primary": round(primary, 6)}
        history.append(record)
        probe_results.append((primary, candidate, index))
        with open(progress_path, "a") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")

    probe_results.sort(key=lambda value: value[0], reverse=True)
    top_count = min(4, len(probe_results))
    refined = []
    for rank, (_, candidate, original_index) in enumerate(probe_results[:top_count]):
        primary, _, _ = train_candidate(candidate, refine_epochs,
                                        args.seed + 5000 + original_index, arrays,
                                        evaluate, device, False)
        record = {"stage": "refine", "epochs": refine_epochs,
                  "config": candidate, "primary": round(primary, 6)}
        history.append(record)
        refined.append((primary, candidate, original_index))
        with open(progress_path, "a") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
    refined.sort(key=lambda value: value[0], reverse=True)
    winner = refined[0][1]

    final_primary, final_scores, epoch_history = train_candidate(
        winner, final_epochs, args.seed + 9000, arrays, evaluate, device, True)
    history.append({"stage": "final", "epochs_requested": final_epochs,
                    "config": winner, "best_primary": round(final_primary, 6),
                    "epochs": epoch_history})
    measured = evaluate(val_user, val_y.astype(int), final_scores)
    gauc = measured["GAUC"] if "GAUC" in measured else measured["gauc"]
    ndcg = measured["nDCG@5"] if "nDCG@5" in measured else measured["ndcg5"]
    metrics = {"gauc": float(gauc), "ndcg5": float(ndcg),
               "primary": float(measured["primary"]), "history": history,
               "selected_config": winner}
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh)

    videos = np.asarray(va.get("video", x_val[:, 1]))
    with open(os.path.join(args.out_dir, "predictions.csv"), "w") as fh:
        fh.write("row_id,user_id,video_id,score\n")
        for i, score in enumerate(final_scores):
            fh.write(f"{i},{val_user[i]},{videos[i]},{float(score):.8g}\n")


if __name__ == "__main__":
    main()
