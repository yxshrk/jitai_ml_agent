import argparse
import csv
import datetime
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DCNHybrid(torch.nn.Module):
    def __init__(self, total_dim, n_fields=5, k=16, hidden=96, dropout=0.25):
        super().__init__()
        self.n_fields = n_fields
        self.k = k
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        dim = n_fields * k
        self.emb_drop = torch.nn.Dropout(dropout)
        self.cross_w = torch.nn.Parameter(torch.empty(dim))
        self.cross_b = torch.nn.Parameter(torch.zeros(dim))
        self.cross_out = torch.nn.Linear(dim, 1, bias=False)
        self.deep = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, hidden // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden // 2, 1),
        )
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)
        torch.nn.init.normal_(self.cross_w, std=0.01)
        torch.nn.init.xavier_uniform_(self.cross_out.weight)
        for layer in self.deep:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

    def forward(self, x):
        e = self.emb(x)
        e = self.emb_drop(e)
        summed = e.sum(1)
        pair = 0.5 * (summed.square() - e.square().sum(1)).sum(1)
        linear = self.lin(x).sum((1, 2))
        x0 = e.reshape(e.shape[0], -1)
        cross = x0 * (x0 @ self.cross_w).unsqueeze(1) + self.cross_b + x0
        return self.bias + linear + pair + self.cross_out(cross).squeeze(1) + self.deep(x0).squeeze(1)


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def date_ordinals(values):
    arr = np.asarray(values)
    unique, inv = np.unique(arr.astype(str), return_inverse=True)
    converted = np.zeros(len(unique), dtype=np.int64)
    for i, value in enumerate(unique):
        text = ''.join(ch for ch in str(value) if ch.isdigit())
        try:
            if len(text) >= 8:
                converted[i] = datetime.date(int(text[:4]), int(text[4:6]), int(text[6:8])).toordinal()
            else:
                converted[i] = int(text) if text else 0
        except (ValueError, OverflowError):
            converted[i] = 0
    return converted[inv]


def load_npz(data_dir):
    tr = np.load(os.path.join(data_dir, "train.npz"), allow_pickle=False)
    va = np.load(os.path.join(data_dir, "val.npz"), allow_pickle=False)
    field_dims = tr["field_dims"].astype(np.int64)
    video_codes = va["X"][:, 1].astype(np.int64) - int(field_dims[0])
    return {
        "Xt": tr["X"].astype(np.int64),
        "yt": tr["y"].astype(np.float32),
        "ut": np.asarray(tr["user"]),
        "dates": np.asarray(tr["date"]),
        "Xv": va["X"].astype(np.int64),
        "yv": va["y"].astype(np.int64),
        "uv": np.asarray(va["user"]),
        "video_out": video_codes,
        "field_dims": field_dims,
        "official": True,
    }


def read_csv_rows(path, training):
    rows = []
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            item = {
                "user": row["user_id"],
                "video": row["video_id"],
                "tab": row["tab"],
                "duration": float(row["duration_ms"]),
                "label": float(row["long_view"]),
            }
            if training:
                item["date"] = row["date"]
            rows.append(item)
    return rows


def load_csv(data_dir):
    train_rows = read_csv_rows(os.path.join(data_dir, "train.csv"), True)
    val_rows = read_csv_rows(os.path.join(data_dir, "val.csv"), False)
    durations = np.asarray([r["duration"] for r in train_rows], dtype=np.float64)
    edges = np.unique(np.quantile(durations, np.linspace(0.1, 0.9, 9)))
    user_map = {v: i + 1 for i, v in enumerate(sorted({r["user"] for r in train_rows}))}
    video_map = {v: i + 1 for i, v in enumerate(sorted({r["video"] for r in train_rows}))}
    tab_map = {v: i + 1 for i, v in enumerate(sorted({r["tab"] for r in train_rows}))}
    dims = np.asarray([len(user_map) + 1, len(video_map) + 1, len(video_map) + 1,
                       len(tab_map) + 1, len(edges) + 2], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(dims)[:-1]))

    def encode(rows):
        x = np.zeros((len(rows), 5), dtype=np.int64)
        for i, r in enumerate(rows):
            v = video_map.get(r["video"], 0)
            x[i] = [user_map.get(r["user"], 0), v, v, tab_map.get(r["tab"], 0),
                    int(np.searchsorted(edges, r["duration"], side="right")) + 1]
        x += offsets
        return x

    return {
        "Xt": encode(train_rows),
        "yt": np.asarray([r["label"] for r in train_rows], dtype=np.float32),
        "ut": np.asarray([r["user"] for r in train_rows]),
        "dates": np.asarray([r["date"] for r in train_rows]),
        "Xv": encode(val_rows),
        "yv": np.asarray([r["label"] for r in val_rows], dtype=np.int64),
        "uv": np.asarray([r["user"] for r in val_rows]),
        "video_out": np.asarray([r["video"] for r in val_rows]),
        "field_dims": dims,
        "official": False,
    }


def get_evaluator(official):
    if official:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    return evaluate


def normalize_metrics(result):
    return {
        "gauc": float(result.get("GAUC", result.get("gauc"))),
        "ndcg5": float(result.get("nDCG@5", result.get("ndcg5"))),
        "primary": float(result["primary"]),
    }


def build_pairs(users, labels, indices, seed):
    rng = np.random.RandomState(seed)
    order = indices[np.argsort(users[indices], kind="stable")]
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    pos_parts = []
    neg_parts = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        group = order[left:right]
        pos = group[labels[group] > 0.5]
        neg = group[labels[group] <= 0.5]
        if len(pos) == 0 or len(neg) == 0:
            continue
        count = max(len(pos), len(neg))
        pos_parts.append(rng.choice(pos, size=count, replace=len(pos) < count))
        neg_parts.append(rng.choice(neg, size=count, replace=len(neg) < count))
    if not pos_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(pos_parts), np.concatenate(neg_parts)


def recency_weights(dates, half_life):
    ords = date_ordinals(dates)
    age = np.maximum(int(ords.max()) - ords, 0)
    weights = np.exp(-math.log(2.0) * age / float(half_life)).astype(np.float32)
    weights /= max(float(weights.mean()), 1e-8)
    return weights


def predict(model, x, device, batch_size=65536):
    model.eval()
    pieces = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start:start + batch_size]).to(device)
            pieces.append(model(xb).detach().cpu().numpy())
    return np.concatenate(pieces)


def train_run(data, indices, pairs, config, epochs, seed, device, evaluate_fn,
              half_epoch_checks=False, retain_scores=False):
    seed_everything(seed)
    model = DCNHybrid(int(data["field_dims"].sum()), dropout=float(config["dropout"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["lr"]),
                                  weight_decay=float(config["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(config["step_every"]),
                                                 gamma=float(config["gamma"]))
    weights = recency_weights(data["dates"], float(config["half_life"]))
    pair_pos, pair_neg = pairs
    batch_size = 8192 if device.type == "cuda" else 4096
    pair_batch = batch_size // 2
    rng = np.random.RandomState(seed + 7919)
    best_primary = -1.0
    best_scores = None
    checkpoints = []
    last_loss = 0.0
    checks_per_epoch = 2 if half_epoch_checks else 1
    for epoch in range(epochs):
        model.train()
        permutation = rng.permutation(indices)
        split_points = np.linspace(0, len(permutation), checks_per_epoch + 1, dtype=int)
        for segment in range(checks_per_epoch):
            segment_indices = permutation[split_points[segment]:split_points[segment + 1]]
            for start in range(0, len(segment_indices), batch_size):
                idx = segment_indices[start:start + batch_size]
                xb = torch.from_numpy(data["Xt"][idx]).to(device)
                yb = torch.from_numpy(data["yt"][idx]).to(device)
                wb = torch.from_numpy(weights[idx]).to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                point_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, yb, reduction="none")
                point_loss = (point_loss * wb).sum() / wb.sum().clamp_min(1e-8)
                if len(pair_pos):
                    selected = rng.randint(0, len(pair_pos), size=min(pair_batch, len(pair_pos)))
                    pi = pair_pos[selected]
                    ni = pair_neg[selected]
                    px = torch.from_numpy(data["Xt"][pi]).to(device)
                    nx = torch.from_numpy(data["Xt"][ni]).to(device)
                    diff = model(px) - model(nx)
                    pw_np = 0.5 * (weights[pi] + weights[ni])
                    pw = torch.from_numpy(pw_np).to(device)
                    rank_each = torch.nn.functional.softplus(-diff)
                    rank_loss = (rank_each * pw).sum() / pw.sum().clamp_min(1e-8)
                    loss = 0.5 * point_loss + 0.5 * rank_loss
                else:
                    loss = point_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                last_loss = float(loss.detach().cpu())
            if half_epoch_checks or segment == checks_per_epoch - 1:
                scores = predict(model, data["Xv"], device)
                metrics = normalize_metrics(evaluate_fn(data["uv"], data["yv"], scores))
                checkpoint = {
                    "epoch": epoch + (segment + 1) / checks_per_epoch,
                    "train_loss": round(last_loss, 6),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "val_gauc": round(metrics["gauc"], 6),
                    "val_primary": round(metrics["primary"], 6),
                }
                checkpoints.append(checkpoint)
                if metrics["primary"] > best_primary + 1e-8:
                    best_primary = metrics["primary"]
                    if retain_scores:
                        best_scores = scores.copy()
        scheduler.step()
    return best_primary, best_scores, checkpoints


def make_coarse_configs(count, seed):
    rng = np.random.RandomState(seed + 101)
    configs = []
    half_lives = [3.5, 5.0, 7.0, 10.0, 14.0]
    step_choices = [1, 1, 2, 2, 3]
    for _ in range(count):
        configs.append({
            "dropout": float(rng.uniform(0.13, 0.42)),
            "weight_decay": float(10 ** rng.uniform(math.log10(3e-5), math.log10(3e-3))),
            "lr": float(10 ** rng.uniform(math.log10(4e-4), math.log10(1.8e-3))),
            "gamma": float(rng.uniform(0.18, 0.62)),
            "step_every": int(step_choices[rng.randint(len(step_choices))]),
            "half_life": float(half_lives[rng.randint(len(half_lives))]),
        })
    return configs


def make_refine_configs(winner, count, seed):
    rng = np.random.RandomState(seed + 303)
    configs = [dict(winner)]
    for _ in range(count - 1):
        configs.append({
            "dropout": float(np.clip(winner["dropout"] + rng.uniform(-0.065, 0.065), 0.08, 0.48)),
            "weight_decay": float(np.clip(winner["weight_decay"] * math.exp(rng.uniform(-0.7, 0.7)),
                                            1e-5, 8e-3)),
            "lr": float(np.clip(winner["lr"] * math.exp(rng.uniform(-0.35, 0.35)), 2e-4, 2.5e-3)),
            "gamma": float(np.clip(winner["gamma"] + rng.uniform(-0.12, 0.12), 0.12, 0.75)),
            "step_every": int(max(1, min(3, winner["step_every"] + rng.choice([-1, 0, 0, 1])))),
            "half_life": float(np.clip(winner["half_life"] * math.exp(rng.uniform(-0.35, 0.35)),
                                           2.5, 18.0)),
        })
    return configs


def append_progress(path, stage, probe, config, score):
    record = {"stage": stage, "probe": probe, "config": config, "primary": score}
    with open(path, "a") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=14)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    progress_path = os.path.join(args.out_dir, "progress.log")
    if os.path.exists(progress_path):
        os.remove(progress_path)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.exists(os.path.join(args.data_dir, "train.npz")) and os.path.exists(os.path.join(args.data_dir, "val.npz")):
        data = load_npz(args.data_dir)
    else:
        data = load_csv(args.data_dir)
    evaluate_fn = get_evaluator(data["official"])
    smoke = os.environ.get("SMOKE_EPOCHS")
    smoke_cap = int(smoke) if smoke is not None else None
    coarse_epochs = min(2, smoke_cap) if smoke_cap is not None else 2
    refine_epochs = min(5, smoke_cap) if smoke_cap is not None else 5
    final_epochs = min(args.epochs, smoke_cap) if smoke_cap is not None else args.epochs
    if smoke_cap is not None:
        coarse_count, refine_count = 4, 2
    elif device.type == "cuda":
        coarse_count, refine_count = 56, 24
    else:
        coarse_count, refine_count = 42, 18
    n = len(data["yt"])
    rng = np.random.RandomState(args.seed + 17)
    coarse_size = max(1, int(0.65 * n))
    coarse_indices = np.sort(rng.choice(n, size=coarse_size, replace=False)).astype(np.int64)
    full_indices = np.arange(n, dtype=np.int64)
    coarse_pairs = build_pairs(data["ut"], data["yt"], coarse_indices, args.seed + 29)
    full_pairs = build_pairs(data["ut"], data["yt"], full_indices, args.seed + 31)
    history = []
    coarse_results = []
    for probe, config in enumerate(make_coarse_configs(coarse_count, args.seed), 1):
        primary, _, checkpoints = train_run(
            data, coarse_indices, coarse_pairs, config, coarse_epochs,
            args.seed + 1000 + probe, device, evaluate_fn, False, False)
        entry = {"stage": "coarse", "probe": probe, "config": config,
                 "best_primary": round(float(primary), 6), "checkpoints": checkpoints}
        history.append(entry)
        coarse_results.append((primary, config))
        append_progress(progress_path, "coarse", probe, config, primary)
    coarse_results.sort(key=lambda item: item[0], reverse=True)
    coarse_winner = coarse_results[0][1]
    refine_results = []
    for probe, config in enumerate(make_refine_configs(coarse_winner, refine_count, args.seed), 1):
        primary, _, checkpoints = train_run(
            data, full_indices, full_pairs, config, refine_epochs,
            args.seed + 3000 + probe, device, evaluate_fn, False, False)
        entry = {"stage": "refine", "probe": probe, "config": config,
                 "best_primary": round(float(primary), 6), "checkpoints": checkpoints}
        history.append(entry)
        refine_results.append((primary, config))
        append_progress(progress_path, "refine", probe, config, primary)
    refine_results.sort(key=lambda item: item[0], reverse=True)
    winning_config = refine_results[0][1]
    final_primary, best_scores, final_checkpoints = train_run(
        data, full_indices, full_pairs, winning_config, final_epochs,
        args.seed, device, evaluate_fn, True, True)
    append_progress(progress_path, "final", 1, winning_config, final_primary)
    final_metrics = normalize_metrics(evaluate_fn(data["uv"], data["yv"], best_scores))
    history.append({
        "stage": "final",
        "probe": 1,
        "config": winning_config,
        "best_primary": round(float(final_primary), 6),
        "checkpoints": final_checkpoints,
    })
    metrics_payload = {
        "gauc": final_metrics["gauc"],
        "ndcg5": final_metrics["ndcg5"],
        "primary": final_metrics["primary"],
        "winning_config": winning_config,
        "history": history,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics_payload, fh)
    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(best_scores):
            writer.writerow([i, data["uv"][i], data["video_out"][i], format(float(score), ".8g")])


if __name__ == "__main__":
    main()
