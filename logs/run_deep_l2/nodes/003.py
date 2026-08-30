import argparse
import csv
import datetime
import itertools
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RankModel(torch.nn.Module):
    def __init__(self, total_dim, architecture, strong, k=16):
        super().__init__()
        self.architecture = architecture
        self.dropout_p = 0.30 if strong else 0.0
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)
        if architecture == "dcn-lite":
            d = 5 * k
            self.cross_w = torch.nn.Linear(d, 1, bias=False)
            self.cross_b = torch.nn.Parameter(torch.zeros(d))
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(d, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout_p),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout_p),
                torch.nn.Linear(64, 1),
            )
            self.deep_scale = torch.nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        e = self.emb(x)
        if self.training and self.dropout_p > 0:
            e = torch.nn.functional.dropout(e, p=self.dropout_p, training=True)
        s = e.sum(1)
        pair = 0.5 * (s * s - (e * e).sum(1)).sum(1)
        fm = self.bias + self.lin(x).sum((1, 2)) + pair
        if self.architecture == "fm":
            return fm
        x0 = e.flatten(1)
        cross = x0 * self.cross_w(x0) + self.cross_b + x0
        deep = self.mlp(cross).squeeze(1)
        return fm + self.deep_scale * deep


def date_to_day(values):
    values = np.asarray(values)
    out = np.zeros(len(values), dtype=np.float32)
    cache = {}
    for i, raw in enumerate(values):
        key = str(raw.decode() if isinstance(raw, bytes) else raw).strip()
        key = key.split(".")[0]
        if key not in cache:
            try:
                dt = datetime.datetime.strptime(key, "%Y%m%d").date()
                cache[key] = float(dt.toordinal())
            except Exception:
                try:
                    cache[key] = float(key)
                except Exception:
                    cache[key] = 0.0
        out[i] = cache[key]
    return out


def encode_csv(train_path, val_path):
    needed = ["user_id", "video_id", "tab", "duration_ms", "date", "long_view"]

    def read_rows(path):
        result = {k: [] for k in needed}
        with open(path, "r", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                for key in needed:
                    result[key].append(row.get(key, "0"))
        return result

    tr = read_rows(train_path)
    va = read_rows(val_path)
    user_values = sorted(set(tr["user_id"]))
    video_values = sorted(set(tr["video_id"]))
    tab_values = sorted(set(tr["tab"]))
    user_map = {v: i + 1 for i, v in enumerate(user_values)}
    video_map = {v: i + 1 for i, v in enumerate(video_values)}
    tab_map = {v: i + 1 for i, v in enumerate(tab_values)}
    tr_dur = np.asarray([float(x or 0) for x in tr["duration_ms"]], dtype=np.float64)
    quantiles = np.quantile(tr_dur, np.linspace(0.1, 0.9, 9))
    field_dims = np.asarray([
        len(user_map) + 1,
        len(video_map) + 1,
        1,
        len(tab_map) + 1,
        10,
    ], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1]))

    def transform(rows):
        n = len(rows["user_id"])
        x = np.zeros((n, 5), dtype=np.int64)
        x[:, 0] = [user_map.get(v, 0) for v in rows["user_id"]]
        x[:, 1] = [video_map.get(v, 0) for v in rows["video_id"]]
        x[:, 2] = 0
        x[:, 3] = [tab_map.get(v, 0) for v in rows["tab"]]
        durations = np.asarray([float(v or 0) for v in rows["duration_ms"]])
        x[:, 4] = np.searchsorted(quantiles, durations, side="right")
        x += offsets[None, :]
        return {
            "X": x.astype(np.int32),
            "y": np.asarray([float(v or 0) for v in rows["long_view"]], dtype=np.float32),
            "user": np.asarray(rows["user_id"]),
            "video": np.asarray(rows["video_id"]),
            "date": np.asarray(rows["date"]),
            "field_dims": field_dims,
        }

    return transform(tr), transform(va), False


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        tr_file = np.load(train_npz)
        va_file = np.load(val_npz)
        tr = {k: tr_file[k] for k in tr_file.files}
        va = {k: va_file[k] for k in va_file.files}
        return tr, va, True
    return encode_csv(os.path.join(data_dir, "train.csv"), os.path.join(data_dir, "val.csv"))


def make_pairs(indices, users, labels, seed):
    indices = np.asarray(indices, dtype=np.int64)
    order = np.argsort(users[indices], kind="mergesort")
    ordered = indices[order]
    ordered_users = users[ordered]
    boundaries = np.r_[0, np.flatnonzero(ordered_users[1:] != ordered_users[:-1]) + 1, len(ordered)]
    rng = np.random.RandomState(seed)
    pos_parts = []
    neg_parts = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        group = ordered[left:right]
        pos = group[labels[group] > 0.5]
        neg = group[labels[group] <= 0.5]
        if len(pos) == 0 or len(neg) == 0:
            continue
        count = max(len(pos), len(neg))
        pos_parts.append(rng.choice(pos, count, replace=len(pos) < count))
        neg_parts.append(rng.choice(neg, count, replace=len(neg) < count))
    if not pos_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(pos_parts), np.concatenate(neg_parts)


def metric_values(metric):
    return {
        "gauc": float(metric.get("GAUC", metric.get("gauc", 0.0))),
        "ndcg5": float(metric.get("nDCG@5", metric.get("ndcg5", 0.0))),
        "primary": float(metric["primary"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    smoke = os.environ.get("SMOKE_EPOCHS")
    smoke_cap = max(1, int(smoke)) if smoke is not None else None
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    tr, va, fast_path = load_data(args.data_dir)
    if fast_path:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate

    X_train = torch.from_numpy(np.asarray(tr["X"], dtype=np.int64))
    y_train_np = np.asarray(tr["y"], dtype=np.float32)
    y_train = torch.from_numpy(y_train_np)
    X_val = torch.from_numpy(np.asarray(va["X"], dtype=np.int64))
    val_labels = np.asarray(va["y"], dtype=np.int64)
    train_users = np.asarray(tr["user"])
    val_users = np.asarray(va["user"])
    field_dims = np.asarray(tr["field_dims"], dtype=np.int64)
    total_dim = int(field_dims.sum())
    n = len(y_train_np)

    if "date" in tr:
        day = date_to_day(tr["date"])
        age = np.maximum(0.0, float(np.max(day)) - day)
        recency_np = np.exp(-math.log(2.0) * age / 7.0).astype(np.float32)
        recency_np /= max(float(recency_np.mean()), 1e-8)
    else:
        recency_np = np.ones(n, dtype=np.float32)
    recency = torch.from_numpy(recency_np)

    fixed_rng = np.random.RandomState(seed + 9001)
    fixed_order = fixed_rng.permutation(n)
    probe_count = max(1, int(n * (0.45 if device.type == "cpu" else 0.60)))
    refine_count = max(1, int(n * (0.70 if device.type == "cpu" else 0.85)))
    probe_indices = fixed_order[:probe_count]
    refine_indices = fixed_order[:refine_count]
    full_indices = np.arange(n, dtype=np.int64)
    pair_cache = {}

    def pairs_for(name, indices):
        if name not in pair_cache:
            pair_cache[name] = make_pairs(indices, train_users, y_train_np, seed + len(pair_cache) * 101)
        return pair_cache[name]

    def score_model(model):
        model.eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, len(X_val), 65536):
                xb = X_val[start:start + 65536].to(device, non_blocking=True)
                chunks.append(model(xb).detach().cpu().numpy())
        scores = np.concatenate(chunks)
        return scores, metric_values(evaluate(val_users, val_labels, scores))

    def fit(config, indices, pair_name, epochs, run_seed, half_checkpoints=False):
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(run_seed)
        strong = config["regularization"] == "strong"
        model = RankModel(total_dim, config["architecture"], strong).to(device)
        if strong:
            optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-3)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        pos_idx, neg_idx = pairs_for(pair_name, indices) if config["loss"] == "bpr-hybrid" else (None, None)
        batch_size = 8192 if config["architecture"] == "fm" else 4096
        if device.type == "cuda":
            batch_size *= 2
        local_rng = np.random.RandomState(run_seed + 17)
        best_primary = -1.0
        best_scores = None
        best_metrics = None
        curve = []
        steps_done = 0
        epochs = max(1, epochs)
        for epoch in range(epochs):
            model.train()
            shuffled = np.asarray(indices)[local_rng.permutation(len(indices))]
            if pos_idx is not None and len(pos_idx):
                pair_order = local_rng.permutation(len(pos_idx))
            else:
                pair_order = None
            split_points = [len(shuffled)]
            if half_checkpoints and len(shuffled) > 1:
                split_points = [len(shuffled) // 2, len(shuffled)]
            begin = 0
            for checkpoint_number, end in enumerate(split_points, 1):
                for start in range(begin, end, batch_size):
                    ids_np = shuffled[start:min(start + batch_size, end)]
                    ids = torch.from_numpy(ids_np).to(device)
                    xb = X_train[ids_np].to(device, non_blocking=True)
                    yb = y_train[ids_np].to(device, non_blocking=True)
                    logits = model(xb)
                    point_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yb, reduction="none")
                    if config["weighting"] == "recency-7d":
                        weights = recency[ids_np].to(device, non_blocking=True)
                        point_loss = (point_loss * weights).sum() / weights.sum().clamp_min(1e-8)
                    else:
                        point_loss = point_loss.mean()
                    if config["loss"] == "bpr-hybrid" and pair_order is not None and len(pair_order):
                        pstart = (steps_done * batch_size) % len(pair_order)
                        take = np.arange(pstart, pstart + len(ids_np)) % len(pair_order)
                        selected = pair_order[take]
                        p_np = pos_idx[selected]
                        q_np = neg_idx[selected]
                        p = X_train[p_np].to(device, non_blocking=True)
                        q = X_train[q_np].to(device, non_blocking=True)
                        pair_loss_raw = torch.nn.functional.softplus(-(model(p) - model(q)))
                        if config["weighting"] == "recency-7d":
                            pw = 0.5 * (recency[p_np] + recency[q_np])
                            pw = pw.to(device, non_blocking=True)
                            pair_loss = (pair_loss_raw * pw).sum() / pw.sum().clamp_min(1e-8)
                        else:
                            pair_loss = pair_loss_raw.mean()
                        loss = 0.5 * point_loss + 0.5 * pair_loss
                    else:
                        loss = point_loss
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    steps_done += 1
                begin = end
                if half_checkpoints or checkpoint_number == len(split_points):
                    scores, metrics = score_model(model)
                    position = epoch + checkpoint_number / len(split_points)
                    curve.append({
                        "epoch": round(float(position), 2),
                        "train_loss": round(float(loss.detach().cpu()), 6),
                        "val_gauc": round(metrics["gauc"], 6),
                        "val_primary": round(metrics["primary"], 6),
                    })
                    if metrics["primary"] > best_primary:
                        best_primary = metrics["primary"]
                        best_scores = scores.copy()
                        best_metrics = metrics
            if strong:
                for group in optimizer.param_groups:
                    group["lr"] *= 0.5
        return best_primary, best_scores, best_metrics, curve

    cells = [
        {"architecture": "fm", "loss": "logloss", "weighting": "uniform", "regularization": "mild"},
        {"architecture": "fm", "loss": "logloss", "weighting": "uniform", "regularization": "strong"},
        {"architecture": "fm", "loss": "logloss", "weighting": "recency-7d", "regularization": "mild"},
        {"architecture": "fm", "loss": "bpr-hybrid", "weighting": "uniform", "regularization": "mild"},
        {"architecture": "fm", "loss": "bpr-hybrid", "weighting": "recency-7d", "regularization": "strong"},
        {"architecture": "dcn-lite", "loss": "logloss", "weighting": "uniform", "regularization": "mild"},
        {"architecture": "dcn-lite", "loss": "logloss", "weighting": "uniform", "regularization": "strong"},
        {"architecture": "dcn-lite", "loss": "logloss", "weighting": "recency-7d", "regularization": "mild"},
        {"architecture": "dcn-lite", "loss": "logloss", "weighting": "recency-7d", "regularization": "strong"},
        {"architecture": "dcn-lite", "loss": "bpr-hybrid", "weighting": "uniform", "regularization": "mild"},
        {"architecture": "dcn-lite", "loss": "bpr-hybrid", "weighting": "uniform", "regularization": "strong"},
        {"architecture": "dcn-lite", "loss": "bpr-hybrid", "weighting": "recency-7d", "regularization": "mild"},
        {"architecture": "dcn-lite", "loss": "bpr-hybrid", "weighting": "recency-7d", "regularization": "strong"},
    ]
    probe_epochs = min(2, smoke_cap) if smoke_cap is not None else 2
    history = []
    progress_path = os.path.join(args.out_dir, "progress.log")
    probe_results = []
    for cell_id, config in enumerate(cells):
        primary, _, metrics, curve = fit(
            config, probe_indices, "probe", probe_epochs, seed + 1000 + cell_id, False
        )
        record = {
            "stage": "matrix_probe",
            "cell": cell_id,
            "config": config,
            "rows": int(len(probe_indices)),
            "epochs": probe_epochs,
            "gauc": round(metrics["gauc"], 6),
            "ndcg5": round(metrics["ndcg5"], 6),
            "primary": round(primary, 6),
            "curve": curve,
        }
        history.append(record)
        probe_results.append((primary, cell_id, config))
        with open(progress_path, "a") as fh:
            fh.write(json.dumps({"cell": cell_id, "config": config, "primary": round(primary, 6)}) + "\n")

    probe_results.sort(key=lambda x: (-x[0], x[1]))
    refine_epochs = min(3, smoke_cap) if smoke_cap is not None else 3
    refine_results = []
    for rank, (_, cell_id, config) in enumerate(probe_results[:2]):
        primary, _, metrics, curve = fit(
            config, refine_indices, "refine", refine_epochs, seed + 3000 + cell_id, False
        )
        record = {
            "stage": "refinement",
            "probe_rank": rank + 1,
            "cell": cell_id,
            "config": config,
            "rows": int(len(refine_indices)),
            "epochs": refine_epochs,
            "gauc": round(metrics["gauc"], 6),
            "ndcg5": round(metrics["ndcg5"], 6),
            "primary": round(primary, 6),
            "curve": curve,
        }
        history.append(record)
        refine_results.append((primary, cell_id, config))
        with open(progress_path, "a") as fh:
            fh.write(json.dumps({"refinement": rank + 1, "cell": cell_id, "config": config, "primary": round(primary, 6)}) + "\n")

    refine_results.sort(key=lambda x: (-x[0], x[1]))
    winning_cell = refine_results[0][1]
    winning_config = refine_results[0][2]
    final_epochs = min(args.epochs, smoke_cap) if smoke_cap is not None else args.epochs
    final_primary, best_scores, final_metrics, final_curve = fit(
        winning_config, full_indices, "full", final_epochs, seed + 7000 + winning_cell, True
    )
    history.append({
        "stage": "final",
        "cell": winning_cell,
        "config": winning_config,
        "rows": int(n),
        "epochs": int(final_epochs),
        "gauc": round(final_metrics["gauc"], 6),
        "ndcg5": round(final_metrics["ndcg5"], 6),
        "primary": round(final_primary, 6),
        "curve": final_curve,
    })

    output_metrics = {
        "gauc": final_metrics["gauc"],
        "ndcg5": final_metrics["ndcg5"],
        "primary": final_metrics["primary"],
        "selected_cell": winning_cell,
        "selected_config": winning_config,
        "history": history,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(output_metrics, fh)

    if "video" in va:
        video_ids = np.asarray(va["video"])
    elif fast_path:
        video_ids = np.asarray(va["X"][:, 1], dtype=np.int64) - int(field_dims[0])
    else:
        video_ids = np.zeros(len(best_scores), dtype=np.int64)
    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(best_scores):
            writer.writerow([i, val_users[i], video_ids[i], format(float(score), ".8g")])


if __name__ == "__main__":
    main()
