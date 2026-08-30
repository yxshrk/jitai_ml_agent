import argparse
import csv
import datetime
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DCNLite(torch.nn.Module):
    def __init__(self, total_dim, fields=5, k=16, hidden=128, dropout=0.2):
        super().__init__()
        self.fields = fields
        self.k = k
        width = fields * k
        self.emb = torch.nn.Embedding(total_dim, k)
        self.emb_dropout = torch.nn.Dropout(dropout)
        self.cross_w = torch.nn.Parameter(torch.empty(width))
        self.cross_b = torch.nn.Parameter(torch.zeros(width))
        self.cross_out = torch.nn.Linear(width, 1, bias=False)
        self.deep = torch.nn.Sequential(
            torch.nn.Linear(width, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, hidden // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden // 2, 1, bias=False),
        )
        self.bias = torch.nn.Parameter(torch.zeros(1))
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.normal_(self.cross_w, std=0.01)
        torch.nn.init.xavier_uniform_(self.cross_out.weight)
        for layer in self.deep:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)

    def forward(self, x):
        x0 = self.emb_dropout(self.emb(x)).reshape(x.shape[0], -1)
        scale = torch.sum(x0 * self.cross_w, dim=1, keepdim=True)
        cross = x0 + x0 * scale + self.cross_b
        return self.bias + self.cross_out(cross).squeeze(1) + self.deep(x0).squeeze(1)


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def encode_dates_as_days(values):
    values = np.asarray(values)
    out = np.empty(len(values), dtype=np.float32)
    cache = {}
    for value in np.unique(values):
        text = str(value)
        if text.endswith(".0"):
            text = text[:-2]
        text = text.replace("-", "")
        try:
            if len(text) >= 8:
                day = datetime.date(int(text[:4]), int(text[4:6]), int(text[6:8])).toordinal()
            else:
                day = int(float(text))
        except Exception:
            day = 0
        cache[value.item() if hasattr(value, "item") else value] = day
    for i, value in enumerate(values):
        key = value.item() if hasattr(value, "item") else value
        out[i] = cache[key]
    return out


def load_npz(data_dir):
    tr = np.load(os.path.join(data_dir, "train.npz"), allow_pickle=False)
    va = np.load(os.path.join(data_dir, "val.npz"), allow_pickle=False)
    dims = tr["field_dims"].astype(np.int64)
    video_offset = int(dims[0])
    bundle = {
        "X_train": tr["X"].astype(np.int64),
        "y_train": tr["y"].astype(np.float32),
        "users_train": np.asarray(tr["user"]),
        "dates_train": np.asarray(tr["date"]),
        "X_val": va["X"].astype(np.int64),
        "y_val": va["y"].astype(np.int64),
        "users_val": np.asarray(va["user"]),
        "videos_val": va["X"][:, 1].astype(np.int64) - video_offset,
        "field_dims": dims,
        "fast": True,
    }
    return bundle


def load_csv_fallback(data_dir):
    train_rows = []
    with open(os.path.join(data_dir, "train.csv"), newline="") as fh:
        for row in csv.DictReader(fh):
            train_rows.append({
                "user_id": row["user_id"],
                "video_id": row["video_id"],
                "tab": row["tab"],
                "duration_ms": float(row["duration_ms"]),
                "date": row["date"],
                "long_view": float(row["long_view"]),
            })
    val_rows = []
    with open(os.path.join(data_dir, "val.csv"), newline="") as fh:
        for row in csv.DictReader(fh):
            val_rows.append({
                "user_id": row["user_id"],
                "video_id": row["video_id"],
                "tab": row["tab"],
                "duration_ms": float(row["duration_ms"]),
                "date": row["date"],
                "long_view": float(row["long_view"]),
            })

    def make_map(values):
        return {v: i + 1 for i, v in enumerate(sorted(set(values)))}

    user_map = make_map([r["user_id"] for r in train_rows])
    video_map = make_map([r["video_id"] for r in train_rows])
    tab_map = make_map([r["tab"] for r in train_rows])
    durations = np.asarray([r["duration_ms"] for r in train_rows], dtype=np.float64)
    edges = np.unique(np.quantile(durations, np.linspace(0.1, 0.9, 9)))
    dims = np.asarray([len(user_map) + 2, len(video_map) + 2, 1, len(tab_map) + 2, 10], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(dims)[:-1]))

    def transform(rows):
        x = np.zeros((len(rows), 5), dtype=np.int64)
        for i, row in enumerate(rows):
            raw = [
                user_map.get(row["user_id"], 0),
                video_map.get(row["video_id"], 0),
                0,
                tab_map.get(row["tab"], 0),
                int(np.searchsorted(edges, row["duration_ms"], side="right")),
            ]
            x[i] = np.asarray(raw, dtype=np.int64) + offsets
        return x

    return {
        "X_train": transform(train_rows),
        "y_train": np.asarray([r["long_view"] for r in train_rows], dtype=np.float32),
        "users_train": np.asarray([r["user_id"] for r in train_rows]),
        "dates_train": np.asarray([r["date"] for r in train_rows]),
        "X_val": transform(val_rows),
        "y_val": np.asarray([r["long_view"] for r in val_rows], dtype=np.int64),
        "users_val": np.asarray([r["user_id"] for r in val_rows]),
        "videos_val": np.asarray([r["video_id"] for r in val_rows]),
        "field_dims": dims,
        "fast": False,
    }


def build_pairs(users, labels):
    users = np.asarray(users)
    labels = np.asarray(labels) > 0.5
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    positives = []
    negatives = []
    for j in range(len(boundaries) - 1):
        group = order[boundaries[j]:boundaries[j + 1]]
        pos = group[labels[group]]
        neg = group[~labels[group]]
        if len(pos) and len(neg):
            positives.append(pos)
            negatives.append(np.resize(neg, len(pos)))
    if not positives:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(positives).astype(np.int64), np.concatenate(negatives).astype(np.int64)


def metric_values(metric):
    return {
        "gauc": float(metric.get("GAUC", metric.get("gauc", 0.0))),
        "ndcg5": float(metric.get("nDCG@5", metric.get("ndcg5", 0.0))),
        "primary": float(metric["primary"]),
    }


def evaluate_scores(evaluate_fn, users, labels, scores):
    return metric_values(evaluate_fn(users, labels, scores))


def predict(model, x_val, batch_size):
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(x_val), batch_size):
            chunks.append(model(x_val[start:start + batch_size]).detach().cpu().numpy())
    return np.concatenate(chunks).astype(np.float64)


def train_phase(config, epochs, phase_seed, tensors, pair_pos, pair_neg, evaluate_fn,
                users_val, labels_val, device, half_epoch_checks=False):
    seed_all(phase_seed)
    x_train, y_train, recency_base, x_val = tensors
    model = DCNLite(
        int(config["total_dim"]), k=16, hidden=128, dropout=float(config["dropout"])
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=int(config["step_size"]), gamma=float(config["gamma"])
    )
    half_life = float(config["half_life"])
    recency = torch.exp(-np.log(2.0) * recency_base / half_life)
    bce_none = torch.nn.BCEWithLogitsLoss(reduction="none")
    n = len(y_train)
    bs = int(config["batch_size"])
    eval_bs = max(bs, 32768)
    pair_count = len(pair_pos)
    best_primary = -1.0
    best_scores = None
    best_metrics = None
    best_checkpoint = 0.0
    curve = []

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(n, device=device)
        if pair_count:
            pair_permutation = torch.randperm(pair_count, device=device)
        segments = 2 if half_epoch_checks else 1
        segment_size = (n + segments - 1) // segments
        last_loss = 0.0
        for segment in range(segments):
            lower = segment * segment_size
            upper = min(n, (segment + 1) * segment_size)
            for start in range(lower, upper, bs):
                idx = permutation[start:min(start + bs, upper)]
                optimizer.zero_grad(set_to_none=True)
                point_logits = model(x_train[idx])
                point_losses = bce_none(point_logits, y_train[idx])
                point_weights = recency[idx]
                point_loss = torch.sum(point_losses * point_weights) / torch.clamp(point_weights.sum(), min=1e-8)
                if pair_count:
                    positions = torch.arange(start, start + len(idx), device=device) % pair_count
                    pair_ids = pair_permutation[positions]
                    pos_idx = pair_pos[pair_ids]
                    neg_idx = pair_neg[pair_ids]
                    margin = model(x_train[pos_idx]) - model(x_train[neg_idx])
                    pair_losses = torch.nn.functional.softplus(-margin)
                    pair_weights = 0.5 * (recency[pos_idx] + recency[neg_idx])
                    pair_loss = torch.sum(pair_losses * pair_weights) / torch.clamp(pair_weights.sum(), min=1e-8)
                    loss = 0.5 * point_loss + 0.5 * pair_loss
                else:
                    loss = point_loss
                loss.backward()
                optimizer.step()
                last_loss = float(loss.detach().cpu())
            scores = predict(model, x_val, eval_bs)
            metrics = evaluate_scores(evaluate_fn, users_val, labels_val, scores)
            checkpoint = epoch + float(segment + 1) / segments
            curve.append({
                "checkpoint": checkpoint,
                "train_loss": round(last_loss, 6),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "val_gauc": round(metrics["gauc"], 6),
                "val_primary": round(metrics["primary"], 6),
            })
            if metrics["primary"] > best_primary + 1e-8:
                best_primary = metrics["primary"]
                best_scores = scores.copy()
                best_metrics = metrics
                best_checkpoint = checkpoint
            model.train()
        scheduler.step()

    return {
        "primary": best_primary,
        "metrics": best_metrics,
        "scores": best_scores,
        "best_checkpoint": best_checkpoint,
        "curve": curve,
    }


def append_progress(path, record):
    with open(path, "a") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    progress_path = os.path.join(args.out_dir, "progress.log")
    if os.path.exists(progress_path):
        os.remove(progress_path)

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fast = os.path.exists(os.path.join(args.data_dir, "train.npz")) and os.path.exists(os.path.join(args.data_dir, "val.npz"))
    data = load_npz(args.data_dir) if fast else load_csv_fallback(args.data_dir)
    if fast:
        from data.official.evaluate import evaluate as evaluate_fn
    else:
        from harness.evaluate_provisional import evaluate as evaluate_fn

    x_train_np = data["X_train"]
    y_train_np = data["y_train"]
    x_val_np = data["X_val"]
    day_values = encode_dates_as_days(data["dates_train"])
    recency_age_np = np.max(day_values) - day_values
    pos_np, neg_np = build_pairs(data["users_train"], y_train_np)

    x_train = torch.from_numpy(x_train_np).to(device)
    y_train = torch.from_numpy(y_train_np).to(device)
    recency_age = torch.from_numpy(recency_age_np.astype(np.float32)).to(device)
    x_val = torch.from_numpy(x_val_np).to(device)
    pair_pos = torch.from_numpy(pos_np).to(device)
    pair_neg = torch.from_numpy(neg_np).to(device)
    tensors = (x_train, y_train, recency_age, x_val)

    smoke = os.environ.get("SMOKE_EPOCHS")
    smoke_cap = int(smoke) if smoke is not None else None
    coarse_epochs = 2 if smoke_cap is None else min(2, smoke_cap)
    refine_epochs = 4 if smoke_cap is None else min(4, smoke_cap)
    final_epochs = args.epochs if smoke_cap is None else min(args.epochs, smoke_cap)
    total_dim = int(np.sum(data["field_dims"]))
    batch_size = 8192 if device.type == "cuda" else 32768

    coarse_configs = [
        {"dropout": 0.15, "weight_decay": 0.00003, "lr": 0.00120, "gamma": 0.50, "step_size": 1, "half_life": 3.5},
        {"dropout": 0.19, "weight_decay": 0.00008, "lr": 0.00095, "gamma": 0.62, "step_size": 1, "half_life": 7.0},
        {"dropout": 0.23, "weight_decay": 0.00020, "lr": 0.00135, "gamma": 0.45, "step_size": 2, "half_life": 14.0},
        {"dropout": 0.27, "weight_decay": 0.00045, "lr": 0.00075, "gamma": 0.70, "step_size": 1, "half_life": 3.5},
        {"dropout": 0.31, "weight_decay": 0.00090, "lr": 0.00105, "gamma": 0.55, "step_size": 2, "half_life": 7.0},
        {"dropout": 0.35, "weight_decay": 0.00160, "lr": 0.00065, "gamma": 0.72, "step_size": 1, "half_life": 14.0},
        {"dropout": 0.40, "weight_decay": 0.00300, "lr": 0.00085, "gamma": 0.40, "step_size": 2, "half_life": 7.0},
        {"dropout": 0.22, "weight_decay": 0.00032, "lr": 0.00150, "gamma": 0.35, "step_size": 1, "half_life": 14.0},
    ]

    rng = np.random.RandomState(args.seed + 91)
    coarse_limit = min(len(x_train_np), 350000 if device.type == "cuda" else 180000)
    coarse_indices_np = rng.choice(len(x_train_np), size=coarse_limit, replace=False)
    coarse_indices = torch.from_numpy(coarse_indices_np.astype(np.int64)).to(device)
    coarse_set = set(coarse_indices_np.tolist())
    pair_mask = np.asarray([(int(p) in coarse_set and int(n) in coarse_set) for p, n in zip(pos_np, neg_np)], dtype=bool)
    old_to_new = {int(old): i for i, old in enumerate(coarse_indices_np)}
    selected_pos = pos_np[pair_mask]
    selected_neg = neg_np[pair_mask]
    coarse_pos_np = np.asarray([old_to_new[int(v)] for v in selected_pos], dtype=np.int64)
    coarse_neg_np = np.asarray([old_to_new[int(v)] for v in selected_neg], dtype=np.int64)
    coarse_pair_pos = torch.from_numpy(coarse_pos_np).to(device)
    coarse_pair_neg = torch.from_numpy(coarse_neg_np).to(device)
    coarse_tensors = (
        x_train[coarse_indices], y_train[coarse_indices], recency_age[coarse_indices], x_val
    )

    history = []
    coarse_results = []
    for probe_id, base_config in enumerate(coarse_configs):
        config = dict(base_config)
        config.update({"total_dim": total_dim, "batch_size": batch_size})
        result = train_phase(
            config, coarse_epochs, args.seed + 100 + probe_id, coarse_tensors,
            coarse_pair_pos, coarse_pair_neg, evaluate_fn, data["users_val"],
            data["y_val"], device, half_epoch_checks=False
        )
        record = {
            "stage": "coarse", "probe": probe_id + 1,
            "config": base_config, "epochs": coarse_epochs,
            "primary": result["primary"], "gauc": result["metrics"]["gauc"],
            "ndcg5": result["metrics"]["ndcg5"],
            "best_checkpoint": result["best_checkpoint"], "curve": result["curve"],
        }
        history.append(record)
        coarse_results.append((result["primary"], base_config))
        append_progress(progress_path, {k: record[k] for k in ("stage", "probe", "config", "primary")})

    winner = dict(max(coarse_results, key=lambda item: item[0])[1])
    refinements = [
        (-0.035, 0.55, 0.82, -0.08, 0.75),
        (-0.018, 0.78, 0.92, -0.04, 0.90),
        (0.000, 1.00, 1.00, 0.00, 1.00),
        (0.018, 1.28, 1.08, 0.04, 1.10),
        (0.035, 1.75, 1.18, 0.08, 1.25),
        (0.008, 0.88, 0.88, 0.06, 1.15),
    ]
    refine_configs = []
    for drop_delta, wd_factor, lr_factor, gamma_delta, half_factor in refinements:
        config = {
            "dropout": float(np.clip(winner["dropout"] + drop_delta, 0.12, 0.45)),
            "weight_decay": float(np.clip(winner["weight_decay"] * wd_factor, 2e-5, 5e-3)),
            "lr": float(np.clip(winner["lr"] * lr_factor, 4e-4, 1.8e-3)),
            "gamma": float(np.clip(winner["gamma"] + gamma_delta, 0.30, 0.82)),
            "step_size": int(winner["step_size"]),
            "half_life": float(np.clip(winner["half_life"] * half_factor, 3.0, 16.0)),
        }
        refine_configs.append(config)

    refine_results = []
    for probe_id, base_config in enumerate(refine_configs):
        config = dict(base_config)
        config.update({"total_dim": total_dim, "batch_size": batch_size})
        result = train_phase(
            config, refine_epochs, args.seed + 300 + probe_id, tensors,
            pair_pos, pair_neg, evaluate_fn, data["users_val"], data["y_val"],
            device, half_epoch_checks=False
        )
        record = {
            "stage": "refine", "probe": probe_id + 1,
            "config": base_config, "epochs": refine_epochs,
            "primary": result["primary"], "gauc": result["metrics"]["gauc"],
            "ndcg5": result["metrics"]["ndcg5"],
            "best_checkpoint": result["best_checkpoint"], "curve": result["curve"],
        }
        history.append(record)
        refine_results.append((result["primary"], base_config))
        append_progress(progress_path, {k: record[k] for k in ("stage", "probe", "config", "primary")})

    final_base_config = dict(max(refine_results, key=lambda item: item[0])[1])
    final_config = dict(final_base_config)
    final_config.update({"total_dim": total_dim, "batch_size": batch_size})
    final_result = train_phase(
        final_config, final_epochs, args.seed, tensors, pair_pos, pair_neg,
        evaluate_fn, data["users_val"], data["y_val"], device,
        half_epoch_checks=True
    )
    final_record = {
        "stage": "final", "config": final_base_config, "epochs": final_epochs,
        "primary": final_result["primary"], "gauc": final_result["metrics"]["gauc"],
        "ndcg5": final_result["metrics"]["ndcg5"],
        "best_checkpoint": final_result["best_checkpoint"], "curve": final_result["curve"],
    }
    history.append(final_record)
    append_progress(progress_path, {
        "stage": "final", "config": final_base_config,
        "primary": final_result["primary"]
    })

    scores = final_result["scores"]
    metrics = evaluate_scores(evaluate_fn, data["users_val"], data["y_val"], scores)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump({
            "gauc": metrics["gauc"],
            "ndcg5": metrics["ndcg5"],
            "primary": metrics["primary"],
            "selected_config": final_base_config,
            "best_checkpoint": final_result["best_checkpoint"],
            "history": history,
        }, fh)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w") as fh:
        fh.write("row_id,user_id,video_id,score\n")
        for i, score in enumerate(scores):
            fh.write(f"{i},{data['users_val'][i]},{data['videos_val'][i]},{score:.8g}\n")


if __name__ == "__main__":
    main()
