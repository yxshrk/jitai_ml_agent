import argparse
import csv
import datetime
import json
import math
import os
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def date_number(v):
    s = str(v).strip()
    try:
        s = str(int(float(s)))
    except Exception:
        return 0
    if len(s) == 8:
        try:
            return datetime.date(int(s[:4]), int(s[4:6]), int(s[6:8])).toordinal()
        except Exception:
            return 0
    try:
        return int(s)
    except Exception:
        return 0


def duration_bucket(values, edges=None):
    x = np.asarray(values, dtype=np.float64)
    if edges is None:
        finite = x[np.isfinite(x)]
        if finite.size == 0:
            edges = np.arange(1, 10, dtype=np.float64)
        else:
            edges = np.unique(np.quantile(finite, np.linspace(0.1, 0.9, 9)))
    return np.searchsorted(edges, x, side="right").astype(np.int64), np.asarray(edges)


def load_csv_data(data_dir):
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    feature_names = ["user_id", "video_id", "author_id", "tab"]

    train_raw = {k: [] for k in feature_names}
    train_duration = []
    train_y = []
    train_date = []
    with open(train_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        names = set(reader.fieldnames or [])
        for row in reader:
            train_raw["user_id"].append(row.get("user_id", ""))
            train_raw["video_id"].append(row.get("video_id", ""))
            train_raw["author_id"].append(row.get("author_id", "__missing__") if "author_id" in names else "__missing__")
            train_raw["tab"].append(row.get("tab", ""))
            train_duration.append(float(row.get("duration_ms", 0) or 0))
            train_y.append(float(row["long_view"]))
            train_date.append(date_number(row.get("date", 0)))

    val_raw = {k: [] for k in feature_names}
    val_duration = []
    val_y = []
    val_date = []
    val_users_out = []
    val_videos_out = []
    with open(val_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        names = set(reader.fieldnames or [])
        for row in reader:
            uid = row.get("user_id", "")
            vid = row.get("video_id", "")
            val_raw["user_id"].append(uid)
            val_raw["video_id"].append(vid)
            val_raw["author_id"].append(row.get("author_id", "__missing__") if "author_id" in names else "__missing__")
            val_raw["tab"].append(row.get("tab", ""))
            val_duration.append(float(row.get("duration_ms", 0) or 0))
            val_y.append(float(row["long_view"]))
            val_date.append(date_number(row.get("date", 0)))
            val_users_out.append(uid)
            val_videos_out.append(vid)

    train_db, edges = duration_bucket(train_duration)
    val_db, _ = duration_bucket(val_duration, edges)

    train_columns = []
    val_columns = []
    dims = []
    for name in feature_names:
        mapping = {}
        encoded_train = np.empty(len(train_y), dtype=np.int64)
        for i, value in enumerate(train_raw[name]):
            if value not in mapping:
                mapping[value] = len(mapping) + 1
            encoded_train[i] = mapping[value]
        encoded_val = np.asarray([mapping.get(value, 0) for value in val_raw[name]], dtype=np.int64)
        train_columns.append(encoded_train)
        val_columns.append(encoded_val)
        dims.append(len(mapping) + 1)

    train_columns.append(train_db)
    val_columns.append(val_db)
    dims.append(max(10, int(train_db.max(initial=0)) + 1))

    offsets = np.cumsum([0] + dims[:-1], dtype=np.int64)
    train_x = np.stack(train_columns, axis=1) + offsets[None, :]
    val_x = np.stack(val_columns, axis=1) + offsets[None, :]

    return {
        "train_x": train_x.astype(np.int64),
        "train_y": np.asarray(train_y, dtype=np.float32),
        "train_user": train_x[:, 0].astype(np.int64),
        "train_date": np.asarray(train_date, dtype=np.int64),
        "val_x": val_x.astype(np.int64),
        "val_y": np.asarray(val_y, dtype=np.float32),
        "val_user": np.asarray(val_users_out),
        "val_user_eval": np.asarray(val_users_out),
        "val_video_out": np.asarray(val_videos_out),
        "field_dims": np.asarray(dims, dtype=np.int64),
        "fast": False,
    }


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if not (os.path.exists(train_npz) and os.path.exists(val_npz)):
        return load_csv_data(data_dir)

    with np.load(train_npz, allow_pickle=False) as tr:
        train_x = np.asarray(tr["X"], dtype=np.int64)
        train_y = np.asarray(tr["y"], dtype=np.float32)
        train_user = np.asarray(tr["user"])
        train_date = np.asarray(tr["date"]) if "date" in tr.files else np.zeros(len(train_y), dtype=np.int64)
        field_dims = np.asarray(tr["field_dims"], dtype=np.int64) if "field_dims" in tr.files else None

    with np.load(val_npz, allow_pickle=False) as va:
        val_x = np.asarray(va["X"], dtype=np.int64)
        val_y = np.asarray(va["y"], dtype=np.float32)
        val_user = np.asarray(va["user"])

    if field_dims is None:
        field_dims = np.ones(train_x.shape[1], dtype=np.int64)

    return {
        "train_x": train_x,
        "train_y": train_y,
        "train_user": train_user,
        "train_date": train_date,
        "val_x": val_x,
        "val_y": val_y,
        "val_user": val_user,
        "val_user_eval": val_user,
        "val_video_out": val_x[:, 1],
        "field_dims": field_dims,
        "fast": True,
    }


def _fallback_metrics(data, scores):
    y = np.asarray(data["val_y"], dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    users = np.asarray(data["val_user_eval"])
    _, inv = np.unique(users, return_inverse=True)

    gauc_num = 0.0
    gauc_den = 0.0
    ndcg_total = 0.0
    ndcg_cnt = 0

    for g in range(inv.max(initial=-1) + 1):
        idx = np.where(inv == g)[0]
        if idx.size < 2:
            continue
        yy = y[idx]
        ss = s[idx]
        pos = yy > 0.5
        neg = ~pos
        npos = int(pos.sum())
        nneg = int(neg.sum())
        if npos == 0 or nneg == 0:
            continue

        order = np.argsort(-ss, kind="stable")
        ranked = yy[order]
        rank_pos = np.flatnonzero(ranked > 0.5) + 1
        dcg = np.sum((2.0 ** ranked[order[:5]] - 1.0) / np.log2(np.arange(2, min(5, ranked.size) + 2)))
        ideal = np.sort(yy)[::-1][:5]
        idcg = np.sum((2.0 ** ideal - 1.0) / np.log2(np.arange(2, min(5, ideal.size) + 2)))
        ndcg_total += float(dcg / idcg) if idcg > 0 else 0.0
        ndcg_cnt += 1

        # AUC via rank statistics
        ranks = np.argsort(np.argsort(ss, kind="stable"), kind="stable").astype(np.float64) + 1.0
        auc = (ranks[pos].sum() - npos * (npos + 1.0) / 2.0) / (npos * nneg)
        gauc_num += auc * (npos + nneg)
        gauc_den += (npos + nneg)

    gauc = gauc_num / gauc_den if gauc_den > 0 else 0.0
    ndcg5 = ndcg_total / ndcg_cnt if ndcg_cnt > 0 else 0.0
    return {"gauc": float(gauc), "ndcg5": float(ndcg5), "primary": float(gauc)}


def official_metrics(data, scores):
    try:
        if data["fast"]:
            from data.official.evaluate import evaluate
        else:
            from harness.evaluate_provisional import evaluate
        result = evaluate(data["val_user_eval"], data["val_y"], np.asarray(scores, dtype=np.float64))
        return {
            "gauc": float(result.get("GAUC", result.get("gauc", 0.0))),
            "ndcg5": float(result.get("nDCG@5", result.get("ndcg5", 0.0))),
            "primary": float(result["primary"]),
        }
    except Exception:
        return _fallback_metrics(data, scores)


class RankModel(nn.Module):
    def __init__(self, total_features, fields, k, architecture, dropout):
        super().__init__()
        self.architecture = architecture
        self.fields = fields
        self.k = k
        self.embedding = nn.Embedding(total_features, k)
        self.linear = nn.Embedding(total_features, 1)
        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.zeros_(self.linear.weight)
        self.bias = nn.Parameter(torch.zeros(1))
        if architecture == "dcn-lite":
            width = fields * k
            self.cross_w = nn.ParameterList([nn.Parameter(torch.empty(width)) for _ in range(2)])
            self.cross_b = nn.ParameterList([nn.Parameter(torch.zeros(width)) for _ in range(2)])
            for w in self.cross_w:
                nn.init.normal_(w, std=0.01)
            self.mlp = nn.Sequential(
                nn.Linear(width, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )
            self.cross_out = nn.Linear(width, 1)

    def forward(self, x):
        e = self.embedding(x)
        linear = self.linear(x).sum(dim=1).squeeze(-1) + self.bias
        summed = e.sum(dim=1)
        fm = 0.5 * (summed.square() - e.square().sum(dim=1)).sum(dim=1)
        score = linear + fm
        if self.architecture == "dcn-lite":
            x0 = e.reshape(e.shape[0], -1)
            z = x0
            for w, b in zip(self.cross_w, self.cross_b):
                z = x0 * torch.sum(z * w, dim=1, keepdim=True) + b + z
            score = score + self.cross_out(z).squeeze(-1) + self.mlp(x0).squeeze(-1)
        return score


def recency_weights(dates, enabled):
    if not enabled:
        return np.ones(len(dates), dtype=np.float32)
    d = np.asarray([date_number(v) for v in dates], dtype=np.int64)
    valid = d[d > 0]
    if valid.size == 0:
        return np.ones(len(d), dtype=np.float32)
    age = np.maximum(0, int(valid.max()) - d)
    w = np.exp(-math.log(2.0) * age / 7.0)
    w[d <= 0] = 1.0
    w = w / max(float(w.mean()), 1e-8)
    return w.astype(np.float32)


def build_pairs(users, labels, seed):
    rng = np.random.default_rng(seed)
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    pos_parts = []
    neg_parts = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        idx = order[a:b]
        pos = idx[labels[idx] > 0.5]
        neg = idx[labels[idx] <= 0.5]
        if len(pos) and len(neg):
            count = max(len(pos), len(neg))
            pos_parts.append(rng.choice(pos, size=count, replace=len(pos) < count))
            neg_parts.append(rng.choice(neg, size=count, replace=len(neg) < count))
    if not pos_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(pos_parts), np.concatenate(neg_parts)


def predict(model, x, device, batch_size=16384):
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.as_tensor(x[start:start + batch_size], dtype=torch.long, device=device)
            out.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(out) if out else np.empty(0, dtype=np.float32)


def train_candidate(data, config, seed, epochs, device, keep_predictions=False):
    set_seed(seed)
    x = data["train_x"]
    y = data["train_y"]
    total_features = int(max(x.max(initial=0), data["val_x"].max(initial=0)) + 1)
    model = RankModel(total_features, x.shape[1], config["k"], config["architecture"], config["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    weights = recency_weights(data["train_date"], config["weighting"] == "recency-7d")
    hybrid = config["loss"] == "bpr-hybrid"
    batch_size = int(config.get("batch_size", 8192))
    rng = np.random.default_rng(seed + 1009)
    best_metric = None
    best_state = None
    best_scores = None
    stale = 0

    for epoch in range(epochs):
        model.train()
        permutation = rng.permutation(len(x))
        if hybrid:
            pair_pos, pair_neg = build_pairs(np.asarray(data["train_user"]), y, seed + epoch * 7919)
            if len(pair_pos):
                pair_perm = rng.permutation(len(pair_pos))
                pair_pos = pair_pos[pair_perm]
                pair_neg = pair_neg[pair_perm]
        else:
            pair_pos = pair_neg = np.empty(0, dtype=np.int64)

        for start in range(0, len(x), batch_size):
            idx = permutation[start:start + batch_size]
            xb = torch.as_tensor(x[idx], dtype=torch.long, device=device)
            yb = torch.as_tensor(y[idx], dtype=torch.float32, device=device)
            wb = torch.as_tensor(weights[idx], dtype=torch.float32, device=device)
            logits = model(xb)
            point = (F.binary_cross_entropy_with_logits(logits, yb, reduction="none") * wb).mean()
            if hybrid and len(pair_pos):
                ps = start % len(pair_pos)
                take = min(batch_size, len(pair_pos))
                pi = np.take(pair_pos, np.arange(ps, ps + take), mode="wrap")
                ni = np.take(pair_neg, np.arange(ps, ps + take), mode="wrap")
                xp = torch.as_tensor(x[pi], dtype=torch.long, device=device)
                xn = torch.as_tensor(x[ni], dtype=torch.long, device=device)
                pair_w = torch.as_tensor(0.5 * (weights[pi] + weights[ni]), dtype=torch.float32, device=device)
                bpr = (F.softplus(-(model(xp) - model(xn))) * pair_w).mean()
                loss = 0.5 * point + 0.5 * bpr
            else:
                loss = point
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        scores = predict(model, data["val_x"], device)
        metric = official_metrics(data, scores)
        if best_metric is None or metric["gauc"] > best_metric["gauc"] + 1e-12:
            best_metric = metric
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_scores = scores.copy() if keep_predictions else None
            stale = 0
        else:
            stale += 1
        if stale >= 3:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    if keep_predictions:
        best_scores = predict(model, data["val_x"], device)
        best_metric = official_metrics(data, best_scores)
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_metric, best_scores


def base_config(architecture, loss, weighting, regularization):
    strong = regularization == "strong"
    return {
        "architecture": architecture,
        "loss": loss,
        "weighting": weighting,
        "regularization": regularization,
        "k": 24,
        "lr": 0.00135 if strong else 0.00168,
        "dropout": 0.32 if strong else 0.21,
        "weight_decay": 0.0005 if strong else 0.000037,
        "batch_size": 8192,
    }


def append_progress(path, record):
    return None


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_data(args.data_dir)

    smoke = os.environ.get("SMOKE_EPOCHS")
    smoke_cap = int(smoke) if smoke is not None else None

    # Lightweight search to keep runtime well under the timeout.
    probe_epochs = 2
    refine_epochs = 2
    final_epochs = 3
    if smoke_cap is not None:
        probe_epochs = min(probe_epochs, smoke_cap)
        refine_epochs = min(refine_epochs, smoke_cap)
        final_epochs = min(final_epochs, smoke_cap)

    history = []
    architectures = ["FM", "dcn-lite"]
    losses = ["logloss", "bpr-hybrid"]
    weightings = ["uniform"]
    regularizations = ["mild"]

    cells = [base_config(a, l, w, r) for a in architectures for l in losses for w in weightings for r in regularizations]
    repeats = 1

    cell_results = []
    for cell_id, config in enumerate(cells):
        values = []
        for repeat in range(repeats):
            run_seed = args.seed + 1000 * repeat + 37 * cell_id
            metric, _ = train_candidate(data, config, run_seed, probe_epochs, device, False)
            record = {"phase": "matrix", "cell": cell_id, "repeat": repeat, "seed": run_seed, "config": config, **metric}
            history.append(record)
            append_progress(os.path.join(args.out_dir, "progress.log"), record)
            values.append(metric["primary"])
        cell_results.append((float(np.mean(values)), float(np.std(values)), config))

    cell_results.sort(key=lambda z: (z[0], -z[1]), reverse=True)
    winner = dict(cell_results[0][2])

    if smoke_cap is None:
        top = cell_results[:2]
        refinement = []
        multipliers = [
            (0.85, 1.00, 1.00),
            (1.00, 1.00, 1.00),
            (1.15, 1.00, 1.00),
        ]
        for rank, (_, _, original) in enumerate(top):
            for dial, (lr_mult, drop_mult, wd_mult) in enumerate(multipliers):
                cfg = dict(original)
                cfg["lr"] = float(original["lr"] * lr_mult)
                cfg["dropout"] = float(min(0.45, original["dropout"] * drop_mult))
                cfg["weight_decay"] = float(original["weight_decay"] * wd_mult)
                vals = []
                for repeat in range(1):
                    run_seed = args.seed + 20000 + rank * 1000 + dial * 100 + repeat
                    metric, _ = train_candidate(data, cfg, run_seed, refine_epochs, device, False)
                    record = {"phase": "refine", "rank": rank, "dial": dial, "repeat": repeat, "seed": run_seed, "config": cfg, **metric}
                    history.append(record)
                    append_progress(os.path.join(args.out_dir, "progress.log"), record)
                    vals.append(metric["primary"])
                refinement.append((float(np.mean(vals)), float(np.std(vals)), cfg))
        refinement.sort(key=lambda z: (z[0], -z[1]), reverse=True)
        winner = dict(refinement[0][2])

    final_metric, final_scores = train_candidate(data, winner, args.seed, final_epochs, device, True)

    pred_path = os.path.join(args.out_dir, "predictions.csv")
    with open(pred_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, (u, v, s) in enumerate(zip(data["val_user"], data["val_video_out"], final_scores)):
            writer.writerow([i, u.item() if hasattr(u, "item") else u, v.item() if hasattr(v, "item") else v, float(s)])

    output = {
        "gauc": final_metric["gauc"],
        "ndcg5": final_metric["ndcg5"],
        "primary": final_metric["primary"],
        "selected_config": winner,
        "history": history,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(output, f, sort_keys=True)


if __name__ == "__main__":
    main()
