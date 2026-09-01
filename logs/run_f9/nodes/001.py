"""Two-stage dial search for a regularized DCN-lite/BPR/recency package.

Uses the official five-field NPZ fast path when available, searches coarse and
refined dial grids with repeated seeds, then rank-averages five consecutive-seed
full-length members selected by half-epoch validation checkpoints.
"""
import argparse
import csv
import datetime
import gc
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.official.evaluate import evaluate as official_evaluate


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


class CrossLayer(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.proj = torch.nn.Linear(width, width)

    def forward(self, x0, x):
        return x + x0 * self.proj(x)


class DCNHybrid(torch.nn.Module):
    def __init__(self, total_dim, fields, k, hidden, cross_layers, dropout):
        super().__init__()
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        width = fields * k
        self.emb_drop = torch.nn.Dropout(dropout)
        self.cross = torch.nn.ModuleList([CrossLayer(width) for _ in range(cross_layers)])
        self.cross_head = torch.nn.Linear(width, 1)
        lower = max(32, hidden // 2)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(width, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, lower),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(lower, 1),
        )
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)

    def forward(self, x):
        e = self.emb_drop(self.emb(x))
        summed = e.sum(1)
        fm = 0.5 * (summed * summed - (e * e).sum(1)).sum(1)
        flat = e.reshape(e.shape[0], -1)
        crossed = flat
        for layer in self.cross:
            crossed = layer(flat, crossed)
        deep = self.mlp(flat).squeeze(1)
        cross_score = self.cross_head(crossed).squeeze(1)
        linear = self.lin(x).sum((1, 2))
        return self.bias + linear + fm + cross_score + deep


def metric_dict(evaluator, users, labels, scores):
    raw = evaluator(users, labels.astype(int), scores)
    return {
        "gauc": float(raw.get("GAUC", raw.get("gauc"))),
        "ndcg5": float(raw.get("nDCG@5", raw.get("ndcg5"))),
        "primary": float(raw["primary"]),
    }


def parse_date_ordinal(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    text = str(value).strip()
    try:
        text = str(int(float(text)))
    except ValueError:
        return 0
    if len(text) != 8:
        return 0
    try:
        return datetime.date(int(text[:4]), int(text[4:6]), int(text[6:8])).toordinal()
    except ValueError:
        return 0


def days_before_latest(values):
    unique = np.unique(values)
    mapping = {v: parse_date_ordinal(v) for v in unique}
    ordinals = np.asarray([mapping[v] for v in values], dtype=np.int32)
    positive = ordinals[ordinals > 0]
    if len(positive) == 0:
        return np.zeros(len(values), dtype=np.float32)
    latest = int(positive.max())
    ordinals[ordinals <= 0] = latest
    return (latest - ordinals).astype(np.float32)


def load_csv_split(path, need_train_outcomes):
    users, videos, tabs, durations, labels, dates = [], [], [], [], [], []
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            users.append(row["user_id"])
            videos.append(row["video_id"])
            tabs.append(row["tab"])
            durations.append(float(row["duration_ms"]))
            labels.append(float(row["long_view"]))
            if need_train_outcomes:
                dates.append(row["date"])
    return {
        "user_raw": np.asarray(users),
        "video_raw": np.asarray(videos),
        "tab_raw": np.asarray(tabs),
        "duration": np.asarray(durations, dtype=np.float32),
        "y": np.asarray(labels, dtype=np.float32),
        "date": np.asarray(dates) if need_train_outcomes else None,
    }


def encode_csv(train, val):
    user_values = np.unique(train["user_raw"])
    video_values = np.unique(train["video_raw"])
    tab_values = np.unique(train["tab_raw"])
    user_map = {v: i + 1 for i, v in enumerate(user_values)}
    video_map = {v: i + 1 for i, v in enumerate(video_values)}
    tab_map = {v: i + 1 for i, v in enumerate(tab_values)}
    quantiles = np.quantile(train["duration"], np.arange(1, 10) / 10.0)

    field_dims = np.asarray([
        len(user_map) + 1,
        len(video_map) + 1,
        1,
        len(tab_map) + 1,
        10,
    ], dtype=np.int64)
    offsets = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(field_dims)[:-1]])

    def transform(split):
        n = len(split["y"])
        x = np.zeros((n, 5), dtype=np.int64)
        x[:, 0] = np.asarray([user_map.get(v, 0) for v in split["user_raw"]])
        x[:, 1] = np.asarray([video_map.get(v, 0) for v in split["video_raw"]])
        x[:, 2] = 0
        x[:, 3] = np.asarray([tab_map.get(v, 0) for v in split["tab_raw"]])
        x[:, 4] = np.searchsorted(quantiles, split["duration"], side="right")
        x += offsets
        return x

    return transform(train), transform(val), field_dims


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        tr_file = np.load(train_npz)
        va_file = np.load(val_npz)
        field_dims = tr_file["field_dims"].astype(np.int64)
        train = {
            "X": tr_file["X"].astype(np.int64),
            "y": tr_file["y"].astype(np.float32),
            "user": np.asarray(tr_file["user"]),
            "days": days_before_latest(np.asarray(tr_file["date"])),
        }
        video_offset = int(field_dims[0])
        val = {
            "X": va_file["X"].astype(np.int64),
            "y": va_file["y"].astype(np.float32),
            "user": np.asarray(va_file["user"]),
            "video_out": va_file["X"][:, 1].astype(np.int64) - video_offset,
        }
        return train, val, field_dims, official_evaluate, "npz"

    from harness.evaluate_provisional import evaluate as provisional_evaluate
    tr_raw = load_csv_split(os.path.join(data_dir, "train.csv"), True)
    va_raw = load_csv_split(os.path.join(data_dir, "val.csv"), False)
    xt, xv, field_dims = encode_csv(tr_raw, va_raw)
    train = {
        "X": xt,
        "y": tr_raw["y"],
        "user": tr_raw["user_raw"],
        "days": days_before_latest(tr_raw["date"]),
    }
    val = {
        "X": xv,
        "y": va_raw["y"],
        "user": va_raw["user_raw"],
        "video_out": va_raw["video_raw"],
    }
    return train, val, field_dims, provisional_evaluate, "csv"


def make_pair_pool(users, labels):
    _, inverse = np.unique(users, return_inverse=True)
    pos = np.flatnonzero(labels > 0.5).astype(np.int64)
    neg = np.flatnonzero(labels <= 0.5).astype(np.int64)
    group_count = int(inverse.max()) + 1 if len(inverse) else 0
    neg_counts = np.bincount(inverse[neg], minlength=group_count).astype(np.int64)
    neg_order = neg[np.argsort(inverse[neg], kind="stable")]
    offsets = np.zeros(group_count, dtype=np.int64)
    if group_count > 1:
        offsets[1:] = np.cumsum(neg_counts[:-1])
    eligible_pos = pos[neg_counts[inverse[pos]] > 0]
    return inverse.astype(np.int64), eligible_pos, neg_order, offsets, neg_counts


def predict_model(model, xv, device):
    model.eval()
    pieces = []
    with torch.no_grad():
        for start in range(0, len(xv), 65536):
            xb = xv[start:start + 65536].to(device, non_blocking=True)
            pieces.append(model(xb).detach().cpu().numpy())
    return np.concatenate(pieces).astype(np.float64)


def train_parent_reference(xt, yt, xv, val, total_dim, evaluator, device, seed, epochs):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = FM(total_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_primary, best_scores, patience = -1.0, None, 0
    n = len(yt)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for start in range(0, n, 8192):
            idx = perm[start:start + 8192]
            xb = xt[idx].to(device, non_blocking=True)
            yb = yt[idx].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(xb), yb)
            loss.backward()
            opt.step()
        scores = predict_model(model, xv, device)
        metrics = metric_dict(evaluator, val["user"], val["y"], scores)
        if metrics["primary"] > best_primary + 1e-6:
            best_primary = metrics["primary"]
            best_scores = scores.copy()
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break
    del model, opt
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_scores, metric_dict(evaluator, val["user"], val["y"], best_scores)


def train_package(xt, yt, xv, val, total_dim, pair_pool, recency_weights,
                  evaluator, device, config, seed, epochs, half_epoch_checkpoints):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = DCNHybrid(
        total_dim=total_dim,
        fields=xt.shape[1],
        k=16,
        hidden=int(config["hidden"]),
        cross_layers=int(config["cross_layers"]),
        dropout=float(config["dropout"]),
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"])
    )
    inverse, eligible_pos, neg_order, neg_offsets, neg_counts = pair_pool
    rng = np.random.RandomState(seed + 9173)
    n = len(yt)
    batch_size = 16384
    best_primary = -1.0
    best_scores = None
    best_metrics = None
    best_checkpoint = 0.0
    curve = []
    perm = None
    halves = max(2, 2 * int(epochs))

    for half in range(halves):
        completed_epochs = half * 0.5
        decays = int(math.floor((completed_epochs + 1e-9) / float(config["step_every"])))
        current_lr = float(config["lr"]) * (float(config["gamma"]) ** decays)
        for group in opt.param_groups:
            group["lr"] = current_lr
        if half % 2 == 0:
            perm = torch.randperm(n)
        midpoint = (n + 1) // 2
        phase_idx = perm[:midpoint] if half % 2 == 0 else perm[midpoint:]
        model.train()
        loss_sum = 0.0
        steps = 0
        for start in range(0, len(phase_idx), batch_size):
            idx = phase_idx[start:start + batch_size]
            b = len(idx)
            q = max(1, b // 2) if len(eligible_pos) else 0
            if q:
                selected = rng.randint(0, len(eligible_pos), size=q)
                pidx = eligible_pos[selected]
                groups = inverse[pidx]
                offsets = (rng.random_sample(q) * neg_counts[groups]).astype(np.int64)
                nidx = neg_order[neg_offsets[groups] + offsets]
                ptorch = torch.from_numpy(pidx)
                ntorch = torch.from_numpy(nidx)
                all_x = torch.cat((xt[idx], xt[ptorch], xt[ntorch]), dim=0).to(
                    device, non_blocking=True
                )
            else:
                ptorch = ntorch = None
                all_x = xt[idx].to(device, non_blocking=True)
            logits = model(all_x)
            main_logits = logits[:b]
            yb = yt[idx].to(device, non_blocking=True)
            wb = recency_weights[idx].to(device, non_blocking=True)
            point = F.binary_cross_entropy_with_logits(main_logits, yb, reduction="none")
            point_loss = (point * wb).sum() / wb.sum().clamp_min(1e-8)
            if q:
                pos_logits = logits[b:b + q]
                neg_logits = logits[b + q:b + 2 * q]
                pw = recency_weights[ptorch].to(device, non_blocking=True)
                pair = F.softplus(-(pos_logits - neg_logits))
                pair_loss = (pair * pw).sum() / pw.sum().clamp_min(1e-8)
                loss = 0.5 * point_loss + 0.5 * pair_loss
            else:
                loss = point_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            loss_sum += float(loss.detach().cpu())
            steps += 1

        checkpoint = (half + 1) * 0.5
        should_evaluate = half_epoch_checkpoints or (half % 2 == 1) or half == halves - 1
        if should_evaluate:
            scores = predict_model(model, xv, device)
            metrics = metric_dict(evaluator, val["user"], val["y"], scores)
            curve.append({
                "checkpoint": checkpoint,
                "train_loss": round(loss_sum / max(steps, 1), 6),
                "lr": round(current_lr, 9),
                "gauc": round(metrics["gauc"], 6),
                "ndcg5": round(metrics["ndcg5"], 6),
                "primary": round(metrics["primary"], 6),
            })
            if metrics["primary"] > best_primary + 1e-8:
                best_primary = metrics["primary"]
                best_scores = scores.copy()
                best_metrics = metrics
                best_checkpoint = checkpoint

    del model, opt
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "scores": best_scores,
        "metrics": best_metrics,
        "best_checkpoint": best_checkpoint,
        "curve": curve,
    }


def coarse_configs(seed):
    rng = np.random.RandomState(seed + 73)
    configs = []
    hidden_choices = [64, 96, 128]
    step_choices = [0.75, 1.0, 1.5, 2.0]
    gamma_choices = [0.35, 0.48, 0.62, 0.74]
    half_lives = [3.5, 7.0, 14.0]
    for _ in range(11):
        configs.append({
            "dropout": float(rng.uniform(0.15, 0.40)),
            "weight_decay": float(10.0 ** rng.uniform(math.log10(3e-5), math.log10(3e-3))),
            "lr": float(10.0 ** rng.uniform(math.log10(4e-4), math.log10(1.8e-3))),
            "step_every": float(step_choices[rng.randint(len(step_choices))]),
            "gamma": float(gamma_choices[rng.randint(len(gamma_choices))]),
            "half_life": float(half_lives[rng.randint(len(half_lives))]),
            "hidden": int(hidden_choices[rng.randint(len(hidden_choices))]),
            "cross_layers": int(1 + rng.randint(2)),
        })
    configs.append({
        "dropout": 0.30,
        "weight_decay": 3e-4,
        "lr": 9e-4,
        "step_every": 1.0,
        "gamma": 0.52,
        "half_life": 7.0,
        "hidden": 128,
        "cross_layers": 1,
    })
    return configs


def refined_configs(winner):
    drop_delta = [-0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06, 0.08]
    log_wd_delta = [-0.45, -0.25, -0.12, 0.0, 0.12, 0.25, 0.40, 0.60]
    log_lr_delta = [0.18, -0.12, 0.08, 0.0, -0.06, 0.12, -0.18, 0.04]
    half_mult = [0.65, 0.80, 0.90, 1.0, 1.10, 1.20, 1.35, 1.50]
    step_delta = [-0.50, -0.25, 0.0, 0.0, 0.25, 0.50, -0.25, 0.25]
    gamma_delta = [-0.10, -0.05, 0.03, 0.0, -0.03, 0.05, 0.10, -0.08]
    result = []
    for j in range(8):
        config = {
            "dropout": float(np.clip(winner["dropout"] + drop_delta[j], 0.10, 0.48)),
            "weight_decay": float(np.clip(winner["weight_decay"] * (10.0 ** log_wd_delta[j]), 1e-5, 5e-3)),
            "lr": float(np.clip(winner["lr"] * (10.0 ** log_lr_delta[j]), 2e-4, 2.5e-3)),
            "step_every": float(np.clip(winner["step_every"] + step_delta[j], 0.5, 2.5)),
            "gamma": float(np.clip(winner["gamma"] + gamma_delta[j], 0.25, 0.82)),
            "half_life": float(np.clip(winner["half_life"] * half_mult[j], 2.5, 18.0)),
            "hidden": int(winner["hidden"]),
            "cross_layers": int(winner["cross_layers"]),
        }
        if j == 0 and winner["hidden"] > 64:
            config["hidden"] = int(winner["hidden"] - 32)
        if j == 7:
            config["cross_layers"] = 2 if int(winner["cross_layers"]) == 1 else 1
        result.append(config)
    return result


def within_user_ranks(users, scores):
    users = np.asarray(users)
    scores = np.asarray(scores)
    order = np.lexsort((scores, users))
    sorted_users = users[order]
    n = len(order)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    starts_mask = np.empty(n, dtype=bool)
    starts_mask[0] = True
    starts_mask[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.flatnonzero(starts_mask)
    ends = np.concatenate((starts[1:], np.asarray([n])))
    counts = ends - starts
    start_at_position = np.maximum.accumulate(np.where(starts_mask, np.arange(n), 0))
    local_rank = np.arange(n) - start_at_position
    denom = np.repeat(np.maximum(counts - 1, 1), counts)
    ranked = np.empty(n, dtype=np.float64)
    ranked[order] = local_rank / denom
    return ranked


def append_progress(path, payload):
    with open(path, "a") as fh:
        fh.write(json.dumps(payload, sort_keys=True) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=14)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    progress_path = os.path.join(args.out_dir, "progress.log")
    with open(progress_path, "w"):
        pass

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        device = torch.device("cpu")

    train, val, field_dims, evaluator, source = load_data(args.data_dir)
    total_dim = int(field_dims.sum())
    xt = torch.from_numpy(train["X"].astype(np.int64, copy=False))
    yt = torch.from_numpy(train["y"].astype(np.float32, copy=False))
    xv = torch.from_numpy(val["X"].astype(np.int64, copy=False))
    pair_pool = make_pair_pool(train["user"], train["y"])
    recency_cache = {}

    def recency_tensor(half_life):
        key = round(float(half_life), 8)
        if key not in recency_cache:
            weights = np.exp2(-train["days"] / float(half_life)).astype(np.float32)
            weights /= max(float(weights.mean()), 1e-8)
            recency_cache[key] = torch.from_numpy(weights)
        return recency_cache[key]

    smoke_text = os.environ.get("SMOKE_EPOCHS")
    smoke_cap = int(smoke_text) if smoke_text is not None else None
    smoke = smoke_cap is not None
    coarse_epochs = 3 if smoke_cap is None else max(1, min(3, smoke_cap))
    refine_epochs = 5 if smoke_cap is None else max(1, min(5, smoke_cap))
    final_epochs = max(1, int(args.epochs))
    parent_epochs = 12
    if smoke_cap is not None:
        final_epochs = max(1, min(final_epochs, smoke_cap))
        parent_epochs = max(1, min(parent_epochs, smoke_cap))

    parent_scores, parent_metrics = train_parent_reference(
        xt, yt, xv, val, total_dim, evaluator, device, args.seed, parent_epochs
    )
    append_progress(progress_path, {
        "stage": "parent_reference", "seed": args.seed,
        "primary": round(parent_metrics["primary"], 8)
    })

    history = []
    summaries = []
    coarse = coarse_configs(args.seed)
    if smoke:
        coarse = coarse[:3]
    coarse_repeats = 1 if smoke else 2
    coarse_means = []
    for candidate_id, config in enumerate(coarse):
        primaries = []
        for repeat in range(coarse_repeats):
            seed = args.seed + 1000 + candidate_id * 17 + repeat
            result = train_package(
                xt, yt, xv, val, total_dim, pair_pool,
                recency_tensor(config["half_life"]), evaluator, device,
                config, seed, coarse_epochs, False
            )
            record = {
                "stage": "coarse", "candidate": candidate_id,
                "replicate": repeat, "seed": seed, "epochs": coarse_epochs,
                "config": config, "best_checkpoint": result["best_checkpoint"],
                "gauc": result["metrics"]["gauc"],
                "ndcg5": result["metrics"]["ndcg5"],
                "primary": result["metrics"]["primary"],
                "curve": result["curve"],
            }
            history.append(record)
            primaries.append(result["metrics"]["primary"])
            append_progress(progress_path, {
                "stage": "coarse", "candidate": candidate_id,
                "replicate": repeat, "seed": seed, "config": config,
                "primary": round(result["metrics"]["primary"], 8)
            })
        mean_primary = float(np.mean(primaries))
        coarse_means.append(mean_primary)
        summaries.append({
            "stage": "coarse_summary", "candidate": candidate_id,
            "config": config, "mean_primary": mean_primary,
            "replicate_primaries": primaries,
        })

    coarse_winner_id = int(np.argmax(coarse_means))
    coarse_winner = coarse[coarse_winner_id]
    refined = refined_configs(coarse_winner)
    if smoke:
        refined = refined[:2]
    refine_repeats = 1 if smoke else 3
    refined_means = []
    for candidate_id, config in enumerate(refined):
        primaries = []
        for repeat in range(refine_repeats):
            seed = args.seed + 5000 + candidate_id * 19 + repeat
            result = train_package(
                xt, yt, xv, val, total_dim, pair_pool,
                recency_tensor(config["half_life"]), evaluator, device,
                config, seed, refine_epochs, False
            )
            record = {
                "stage": "refine", "candidate": candidate_id,
                "replicate": repeat, "seed": seed, "epochs": refine_epochs,
                "config": config, "best_checkpoint": result["best_checkpoint"],
                "gauc": result["metrics"]["gauc"],
                "ndcg5": result["metrics"]["ndcg5"],
                "primary": result["metrics"]["primary"],
                "curve": result["curve"],
            }
            history.append(record)
            primaries.append(result["metrics"]["primary"])
            append_progress(progress_path, {
                "stage": "refine", "candidate": candidate_id,
                "replicate": repeat, "seed": seed, "config": config,
                "primary": round(result["metrics"]["primary"], 8)
            })
        mean_primary = float(np.mean(primaries))
        refined_means.append(mean_primary)
        summaries.append({
            "stage": "refine_summary", "candidate": candidate_id,
            "config": config, "mean_primary": mean_primary,
            "replicate_primaries": primaries,
        })

    winning_id = int(np.argmax(refined_means))
    winning_config = refined[winning_id]
    member_count = 1 if smoke else 5
    member_scores = []
    member_ranks = []
    member_records = []
    parent_ranks = within_user_ranks(val["user"], parent_scores)
    for member in range(member_count):
        seed = args.seed + member
        result = train_package(
            xt, yt, xv, val, total_dim, pair_pool,
            recency_tensor(winning_config["half_life"]), evaluator, device,
            winning_config, seed, final_epochs, True
        )
        scores = result["scores"]
        ranks = within_user_ranks(val["user"], scores)
        assert not np.allclose(ranks, parent_ranks), "final member equals parent predictions"
        for prior_scores, prior_ranks in zip(member_scores, member_ranks):
            assert not np.allclose(scores, prior_scores), "final member raw scores are identical"
            assert not np.allclose(ranks, prior_ranks), "final member ranked scores are identical"
        member_scores.append(scores)
        member_ranks.append(ranks)
        member_record = {
            "member": member, "seed": seed,
            "best_checkpoint": result["best_checkpoint"],
            "gauc": result["metrics"]["gauc"],
            "ndcg5": result["metrics"]["ndcg5"],
            "primary": result["metrics"]["primary"],
            "curve": result["curve"],
        }
        member_records.append(member_record)
        append_progress(progress_path, {
            "stage": "final_member", "member": member, "seed": seed,
            "primary": round(result["metrics"]["primary"], 8),
            "best_checkpoint": result["best_checkpoint"],
            "config": winning_config,
        })

    final_scores = np.mean(np.stack(member_ranks, axis=0), axis=0)
    final_metrics = metric_dict(evaluator, val["user"], val["y"], final_scores)
    append_progress(progress_path, {
        "stage": "final_ensemble", "members": member_count,
        "primary": round(final_metrics["primary"], 8)
    })

    metrics_payload = {
        "gauc": final_metrics["gauc"],
        "ndcg5": final_metrics["ndcg5"],
        "primary": final_metrics["primary"],
        "history": history,
        "search_summaries": summaries,
        "coarse_winner": {"candidate": coarse_winner_id, "config": coarse_winner},
        "winning_config": winning_config,
        "parent_reference": parent_metrics,
        "members": member_records,
        "data_source": source,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics_payload, fh)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(final_scores):
            writer.writerow([i, val["user"][i], val["video_out"][i], format(float(score), ".8g")])


if __name__ == "__main__":
    main()
