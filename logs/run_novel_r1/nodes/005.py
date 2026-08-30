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


class DCNLite(torch.nn.Module):
    def __init__(self, total_dim, fields=5, k=16, hidden=128, dropout=0.2, cross_layers=2):
        super().__init__()
        self.emb = torch.nn.Embedding(total_dim, k)
        self.linear = torch.nn.Embedding(total_dim, 1)
        width = fields * k
        self.cross_w = torch.nn.ParameterList([
            torch.nn.Parameter(torch.empty(width)) for _ in range(cross_layers)
        ])
        self.cross_b = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros(width)) for _ in range(cross_layers)
        ])
        self.cross_out = torch.nn.Linear(width, 1, bias=False)
        self.deep = torch.nn.Sequential(
            torch.nn.Linear(width, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, hidden // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden // 2, 1),
        )
        self.input_dropout = torch.nn.Dropout(dropout)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.linear.weight)
        for w in self.cross_w:
            torch.nn.init.normal_(w, std=0.01)

    def forward(self, x):
        linear = self.linear(x).sum((1, 2))
        x0 = self.input_dropout(self.emb(x).flatten(1))
        crossed = x0
        for w, b in zip(self.cross_w, self.cross_b):
            scale = torch.sum(crossed * w, dim=1, keepdim=True)
            crossed = x0 * scale + b + crossed
        return linear + self.cross_out(crossed).squeeze(1) + self.deep(x0).squeeze(1) + self.bias


def fit_map(values):
    mapping = {}
    encoded = np.empty(len(values), dtype=np.int64)
    for i, value in enumerate(values):
        if value not in mapping:
            mapping[value] = len(mapping) + 1
        encoded[i] = mapping[value]
    return mapping, encoded


def apply_map(values, mapping):
    return np.asarray([mapping.get(value, 0) for value in values], dtype=np.int64)


def display_value(value):
    if isinstance(value, np.generic):
        value = value.item()
    return value


def load_csv_data(data_dir):
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    tr_user, tr_video, tr_tab, tr_duration, tr_date, tr_y = [], [], [], [], [], []
    with open(train_path, newline="") as fh:
        for row in csv.DictReader(fh):
            tr_user.append(row["user_id"])
            tr_video.append(row["video_id"])
            tr_tab.append(row["tab"])
            tr_duration.append(float(row["duration_ms"]))
            tr_date.append(row["date"])
            tr_y.append(float(row["long_view"]))
    va_user, va_video, va_tab, va_duration, va_y = [], [], [], [], []
    with open(val_path, newline="") as fh:
        for row in csv.DictReader(fh):
            va_user.append(row["user_id"])
            va_video.append(row["video_id"])
            va_tab.append(row["tab"])
            va_duration.append(float(row["duration_ms"]))
            va_y.append(float(row["long_view"]))

    user_map, tr_u = fit_map(tr_user)
    video_map, tr_v = fit_map(tr_video)
    tab_map, tr_t = fit_map(tr_tab)
    va_u = apply_map(va_user, user_map)
    va_v = apply_map(va_video, video_map)
    va_t = apply_map(va_tab, tab_map)
    tr_duration = np.asarray(tr_duration, dtype=np.float64)
    va_duration = np.asarray(va_duration, dtype=np.float64)
    edges = np.unique(np.quantile(tr_duration, np.linspace(0.1, 0.9, 9)))
    tr_d = np.searchsorted(edges, tr_duration, side="right").astype(np.int64)
    va_d = np.searchsorted(edges, va_duration, side="right").astype(np.int64)
    dims = np.asarray([
        len(user_map) + 1,
        len(video_map) + 1,
        1,
        len(tab_map) + 1,
        max(10, len(edges) + 1),
    ], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(dims)[:-1]))
    Xt = np.stack((tr_u, tr_v, np.zeros(len(tr_u), dtype=np.int64), tr_t, tr_d), axis=1)
    Xv = np.stack((va_u, va_v, np.zeros(len(va_u), dtype=np.int64), va_t, va_d), axis=1)
    Xt += offsets
    Xv += offsets
    return {
        "Xt": Xt,
        "yt": np.asarray(tr_y, dtype=np.float32),
        "train_user": np.asarray(tr_u),
        "train_date": np.asarray(tr_date),
        "Xv": Xv,
        "yv": np.asarray(va_y, dtype=np.float32),
        "val_user": np.asarray(va_user),
        "val_video": np.asarray(va_video),
        "field_dims": dims,
        "fast": False,
    }


def load_data(data_dir):
    train_path = os.path.join(data_dir, "train.npz")
    val_path = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_path) and os.path.exists(val_path):
        tr = np.load(train_path)
        va = np.load(val_path)
        dims = tr["field_dims"].astype(np.int64)
        Xv = va["X"].astype(np.int64)
        video_offset = int(dims[0])
        val_video = Xv[:, 1] - video_offset
        return {
            "Xt": tr["X"].astype(np.int64),
            "yt": tr["y"].astype(np.float32),
            "train_user": tr["user"],
            "train_date": tr["date"],
            "Xv": Xv,
            "yv": va["y"].astype(np.float32),
            "val_user": va["user"],
            "val_video": val_video,
            "field_dims": dims,
            "fast": True,
        }
    return load_csv_data(data_dir)


def dates_to_days(values):
    result = np.empty(len(values), dtype=np.float64)
    fallback = {}
    next_fallback = 0
    for i, value in enumerate(values):
        text = str(display_value(value)).strip()
        parsed = None
        for fmt in ("%Y%m%d", "%Y-%m-%d", "%Y/%m/%d"):
            try:
                parsed = datetime.datetime.strptime(text, fmt).date().toordinal()
                break
            except ValueError:
                pass
        if parsed is None:
            try:
                parsed = float(text)
            except ValueError:
                if text not in fallback:
                    fallback[text] = next_fallback
                    next_fallback += 1
                parsed = fallback[text]
        result[i] = parsed
    return result


def recency_weights(dates, half_life):
    days = dates_to_days(dates)
    age = np.max(days) - days
    weights = np.exp2(-age / max(float(half_life), 1e-3))
    weights /= max(float(np.mean(weights)), 1e-8)
    return weights.astype(np.float32)


def make_pairs(users, labels, seed):
    order = np.argsort(users, kind="stable")
    sorted_users = np.asarray(users)[order]
    cuts = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
    groups = np.split(order, cuts)
    rng = np.random.default_rng(seed)
    positives = []
    negatives = []
    for idx in groups:
        pos = idx[labels[idx] > 0.5]
        neg = idx[labels[idx] <= 0.5]
        if len(pos) and len(neg):
            positives.append(pos)
            negatives.append(rng.choice(neg, size=len(pos), replace=True))
    if not positives:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(positives), np.concatenate(negatives)


def predict(model, X, device):
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(X), 65536):
            xb = torch.as_tensor(X[start:start + 65536], dtype=torch.long, device=device)
            parts.append(model(xb).detach().cpu().numpy())
    return np.concatenate(parts).astype(np.float32)


def metric_values(evaluate_fn, users, labels, scores):
    result = evaluate_fn(users, labels.astype(int), scores)
    return {
        "gauc": float(result.get("GAUC", result.get("gauc"))),
        "ndcg5": float(result.get("nDCG@5", result.get("ndcg5"))),
        "primary": float(result["primary"]),
    }


def train_config(config, seed, epochs, data, device, evaluate_fn, patience_halves):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = DCNLite(
        int(data["field_dims"].sum()),
        fields=data["Xt"].shape[1],
        k=16,
        hidden=128,
        dropout=float(config["dropout"]),
        cross_layers=2,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"])
    )
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    Xt = data["Xt"]
    yt = data["yt"]
    weights = recency_weights(data["train_date"], config["half_life"])
    pair_pos, pair_neg = make_pairs(data["train_user"], yt, seed + 31)
    rng = np.random.default_rng(seed + 7919)
    n = len(yt)
    batch_size = 8192
    best_primary = -1.0
    best_scores = None
    best_metrics = None
    curve = []
    stale = 0
    checkpoint = 0
    stop = False
    for epoch in range(epochs):
        row_order = rng.permutation(n)
        pair_order = rng.permutation(len(pair_pos)) if len(pair_pos) else pair_pos
        row_halves = np.array_split(row_order, 2)
        pair_halves = np.array_split(pair_order, 2) if len(pair_pos) else [pair_pos, pair_pos]
        for half in range(2):
            decay_power = checkpoint // max(1, int(config["step_halves"]))
            current_lr = float(config["lr"]) * (float(config["gamma"]) ** decay_power)
            for group in optimizer.param_groups:
                group["lr"] = current_lr
            model.train()
            idx_half = row_halves[half]
            p_half = pair_halves[half]
            point_loss_sum = 0.0
            pair_loss_sum = 0.0
            seen = 0
            for start in range(0, len(idx_half), batch_size):
                idx = idx_half[start:start + batch_size]
                if len(idx) == 0:
                    continue
                xb = torch.as_tensor(Xt[idx], dtype=torch.long, device=device)
                yb = torch.as_tensor(yt[idx], dtype=torch.float32, device=device)
                wb = torch.as_tensor(weights[idx], dtype=torch.float32, device=device)
                point_losses = bce(model(xb), yb)
                point_loss = torch.sum(point_losses * wb) / torch.clamp(torch.sum(wb), min=1e-8)
                if len(p_half):
                    psel = p_half[start % len(p_half):(start % len(p_half)) + len(idx)]
                    if len(psel) < len(idx):
                        psel = np.concatenate((psel, p_half[:len(idx) - len(psel)]))
                    pi = pair_pos[psel]
                    ni = pair_neg[psel]
                    both = np.concatenate((pi, ni))
                    pair_x = torch.as_tensor(Xt[both], dtype=torch.long, device=device)
                    pair_scores = model(pair_x)
                    pos_scores = pair_scores[:len(pi)]
                    neg_scores = pair_scores[len(pi):]
                    pair_w_np = 0.5 * (weights[pi] + weights[ni])
                    pair_w = torch.as_tensor(pair_w_np, dtype=torch.float32, device=device)
                    pair_losses = torch.nn.functional.softplus(-(pos_scores - neg_scores))
                    pair_loss = torch.sum(pair_losses * pair_w) / torch.clamp(torch.sum(pair_w), min=1e-8)
                else:
                    pair_loss = point_loss * 0.0
                loss = 0.5 * point_loss + 0.5 * pair_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                point_loss_sum += float(point_loss.detach().cpu()) * len(idx)
                pair_loss_sum += float(pair_loss.detach().cpu()) * len(idx)
                seen += len(idx)
            scores = predict(model, data["Xv"], device)
            metrics = metric_values(evaluate_fn, data["val_user"], data["yv"], scores)
            curve.append({
                "epoch": epoch + 1,
                "half": half + 1,
                "lr": round(current_lr, 9),
                "point_loss": round(point_loss_sum / max(seen, 1), 6),
                "bpr_loss": round(pair_loss_sum / max(seen, 1), 6),
                "gauc": round(metrics["gauc"], 6),
                "primary": round(metrics["primary"], 6),
            })
            if metrics["primary"] > best_primary + 1e-7:
                best_primary = metrics["primary"]
                best_scores = scores.copy()
                best_metrics = metrics
                stale = 0
            else:
                stale += 1
            checkpoint += 1
            if stale >= patience_halves:
                stop = True
                break
        if stop:
            break
    return best_scores, best_metrics, curve


def broad_configs(count, seed):
    rng = np.random.default_rng(seed)
    anchors = [
        {"dropout": 0.24, "weight_decay": 3e-4, "lr": 8e-4, "gamma": 0.55, "step_halves": 2, "half_life": 7.0},
        {"dropout": 0.34, "weight_decay": 1e-3, "lr": 6e-4, "gamma": 0.68, "step_halves": 2, "half_life": 7.0},
        {"dropout": 0.18, "weight_decay": 1e-4, "lr": 1.2e-3, "gamma": 0.42, "step_halves": 4, "half_life": 14.0},
    ]
    configs = list(anchors[:count])
    half_lives = np.asarray([3.5, 5.0, 7.0, 10.0, 14.0])
    gammas = np.asarray([0.38, 0.48, 0.58, 0.68, 0.78])
    steps = np.asarray([1, 2, 3, 4])
    while len(configs) < count:
        configs.append({
            "dropout": float(rng.uniform(0.15, 0.42)),
            "weight_decay": float(10 ** rng.uniform(math.log10(3e-5), math.log10(3e-3))),
            "lr": float(10 ** rng.uniform(math.log10(3e-4), math.log10(1.8e-3))),
            "gamma": float(rng.choice(gammas)),
            "step_halves": int(rng.choice(steps)),
            "half_life": float(rng.choice(half_lives)),
        })
    return configs


def refined_configs(winner, count, seed):
    rng = np.random.default_rng(seed)
    configs = [dict(winner)]
    while len(configs) < count:
        half_life = float(np.clip(winner["half_life"] * math.exp(rng.normal(0.0, 0.25)), 3.0, 16.0))
        configs.append({
            "dropout": float(np.clip(winner["dropout"] + rng.normal(0.0, 0.035), 0.12, 0.45)),
            "weight_decay": float(np.clip(winner["weight_decay"] * math.exp(rng.normal(0.0, 0.45)), 2e-5, 5e-3)),
            "lr": float(np.clip(winner["lr"] * math.exp(rng.normal(0.0, 0.25)), 2e-4, 2e-3)),
            "gamma": float(np.clip(winner["gamma"] + rng.normal(0.0, 0.065), 0.3, 0.85)),
            "step_halves": int(np.clip(winner["step_halves"] + rng.integers(-1, 2), 1, 5)),
            "half_life": half_life,
        })
    return configs


def validation_groups(users):
    order = np.argsort(users, kind="stable")
    sorted_users = np.asarray(users)[order]
    cuts = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
    return np.split(order, cuts)


def rank_transform(scores, groups):
    result = np.empty(len(scores), dtype=np.float32)
    for idx in groups:
        if len(idx) == 1:
            result[idx[0]] = 0.5
        else:
            order = np.argsort(scores[idx], kind="stable")
            ranks = np.empty(len(idx), dtype=np.float32)
            ranks[order] = np.arange(len(idx), dtype=np.float32) / float(len(idx) - 1)
            result[idx] = ranks
    return result


def clean_config(config):
    return {
        "dropout": round(float(config["dropout"]), 7),
        "weight_decay": round(float(config["weight_decay"]), 9),
        "lr": round(float(config["lr"]), 9),
        "gamma": round(float(config["gamma"]), 7),
        "step_halves": int(config["step_halves"]),
        "half_life": round(float(config["half_life"]), 5),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        device = torch.device("cpu")
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

    data = load_data(args.data_dir)
    if data["fast"]:
        from data.official.evaluate import evaluate as evaluate_fn
    else:
        from harness.evaluate_provisional import evaluate as evaluate_fn

    smoke_value = os.environ.get("SMOKE_EPOCHS")
    smoke = smoke_value is not None
    smoke_cap = max(1, int(smoke_value)) if smoke else None
    coarse_epochs = min(3, smoke_cap) if smoke else 3
    refine_epochs = min(6, smoke_cap) if smoke else 6
    final_epochs = min(args.epochs, smoke_cap) if smoke else args.epochs
    if smoke:
        coarse_count, refine_count, final_members = 1, 1, 1
    elif device.type == "cuda":
        coarse_count, refine_count, final_members = 160, 96, 5
    else:
        coarse_count, refine_count, final_members = 96, 64, 5

    history = []
    progress_path = os.path.join(args.out_dir, "progress.log")

    stage1 = broad_configs(coarse_count, args.seed + 1001)
    best_stage1 = None
    best_stage1_primary = -1.0
    for i, config in enumerate(stage1):
        probe_seed = args.seed + (i % 3) * 1009
        _, metrics, curve = train_config(
            config, probe_seed, coarse_epochs, data, device, evaluate_fn, patience_halves=4
        )
        record = {
            "stage": 1,
            "probe": i + 1,
            "seed": probe_seed,
            "config": clean_config(config),
            "gauc": round(metrics["gauc"], 6),
            "ndcg5": round(metrics["ndcg5"], 6),
            "primary": round(metrics["primary"], 6),
            "curve": curve,
        }
        history.append(record)
        with open(progress_path, "a") as fh:
            fh.write(json.dumps({"stage": 1, "probe": i + 1, "config": record["config"], "primary": record["primary"]}, sort_keys=True) + "\n")
        if metrics["primary"] > best_stage1_primary:
            best_stage1_primary = metrics["primary"]
            best_stage1 = dict(config)

    stage2 = refined_configs(best_stage1, refine_count, args.seed + 2003)
    best_config = None
    best_refine_primary = -1.0
    for i, config in enumerate(stage2):
        probe_seed = args.seed + (i % 4) * 1013
        _, metrics, curve = train_config(
            config, probe_seed, refine_epochs, data, device, evaluate_fn, patience_halves=6
        )
        record = {
            "stage": 2,
            "probe": i + 1,
            "seed": probe_seed,
            "config": clean_config(config),
            "gauc": round(metrics["gauc"], 6),
            "ndcg5": round(metrics["ndcg5"], 6),
            "primary": round(metrics["primary"], 6),
            "curve": curve,
        }
        history.append(record)
        with open(progress_path, "a") as fh:
            fh.write(json.dumps({"stage": 2, "probe": i + 1, "config": record["config"], "primary": record["primary"]}, sort_keys=True) + "\n")
        if metrics["primary"] > best_refine_primary:
            best_refine_primary = metrics["primary"]
            best_config = dict(config)

    member_scores = []
    final_history = []
    for member in range(final_members):
        member_seed = args.seed + member
        scores, metrics, curve = train_config(
            best_config, member_seed, final_epochs, data, device, evaluate_fn, patience_halves=8
        )
        member_scores.append(scores)
        entry = {
            "member": member + 1,
            "seed": member_seed,
            "gauc": round(metrics["gauc"], 6),
            "ndcg5": round(metrics["ndcg5"], 6),
            "primary": round(metrics["primary"], 6),
            "curve": curve,
        }
        final_history.append(entry)
        with open(progress_path, "a") as fh:
            fh.write(json.dumps({"stage": "final", "member": member + 1, "seed": member_seed, "primary": entry["primary"]}, sort_keys=True) + "\n")

    groups = validation_groups(data["val_user"])
    ranked = [rank_transform(scores, groups) for scores in member_scores]
    ensemble_history = []
    best_scores = None
    best_metrics = None
    best_count = None
    for count in range(1, final_members + 1):
        scores = np.mean(np.stack(ranked[:count]), axis=0).astype(np.float32)
        metrics = metric_values(evaluate_fn, data["val_user"], data["yv"], scores)
        ensemble_history.append({
            "member_count": count,
            "gauc": round(metrics["gauc"], 6),
            "ndcg5": round(metrics["ndcg5"], 6),
            "primary": round(metrics["primary"], 6),
        })
        if best_metrics is None or metrics["primary"] > best_metrics["primary"] + 1e-12:
            best_metrics = metrics
            best_scores = scores.copy()
            best_count = count

    output = {
        "gauc": best_metrics["gauc"],
        "ndcg5": best_metrics["ndcg5"],
        "primary": best_metrics["primary"],
        "selected_config": clean_config(best_config),
        "selected_ensemble_members": best_count,
        "history": history,
        "final_history": final_history,
        "ensemble_history": ensemble_history,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(output, fh)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w") as fh:
        fh.write("row_id,user_id,video_id,score\n")
        for i, score in enumerate(best_scores):
            fh.write(f"{i},{display_value(data['val_user'][i])},{display_value(data['val_video'][i])},{float(score):.7g}\n")


if __name__ == "__main__":
    main()
