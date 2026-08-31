import argparse
import datetime
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.official.evaluate import evaluate


class DCNLite(torch.nn.Module):
    def __init__(self, total_dim, fields=5, k=16, hidden=128, dropout=0.25):
        super().__init__()
        self.fields = fields
        self.k = k
        self.dropout = float(dropout)
        d = fields * k
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.cross_w = torch.nn.ParameterList([
            torch.nn.Parameter(torch.empty(d)) for _ in range(2)
        ])
        self.cross_b = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros(d)) for _ in range(2)
        ])
        self.cross_out = torch.nn.Linear(d, 1, bias=False)
        self.deep1 = torch.nn.Linear(d, hidden)
        self.deep2 = torch.nn.Linear(hidden, hidden // 2)
        self.deep_out = torch.nn.Linear(hidden // 2, 1, bias=False)
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)
        for w in self.cross_w:
            torch.nn.init.normal_(w, std=0.01)
        for layer in (self.cross_out, self.deep1, self.deep2, self.deep_out):
            if hasattr(layer, "weight"):
                torch.nn.init.xavier_uniform_(layer.weight)
            if getattr(layer, "bias", None) is not None:
                torch.nn.init.zeros_(layer.bias)

    def forward(self, x):
        e_raw = self.emb(x)
        e = F.dropout(e_raw, p=self.dropout, training=self.training)
        summed = e.sum(dim=1)
        fm_pair = 0.5 * (summed.square() - e.square().sum(dim=1)).sum(dim=1)
        linear = self.lin(x).sum(dim=(1, 2))
        x0 = e.flatten(1)
        cross = x0
        for w, b in zip(self.cross_w, self.cross_b):
            cross = x0 * (cross * w).sum(dim=1, keepdim=True) + b + cross
        deep = F.relu(self.deep1(x0))
        deep = F.dropout(deep, p=self.dropout, training=self.training)
        deep = F.relu(self.deep2(deep))
        deep = F.dropout(deep, p=self.dropout, training=self.training)
        return self.bias + linear + fm_pair + self.cross_out(cross).squeeze(1) + self.deep_out(deep).squeeze(1)


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def date_ordinals(values):
    arr = np.asarray(values)
    out = np.zeros(len(arr), dtype=np.float32)
    cache = {}
    for i, value in enumerate(arr):
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        text = str(value)
        if text.endswith(".0"):
            text = text[:-2]
        text = text.replace("-", "")
        if text not in cache:
            try:
                d = datetime.datetime.strptime(text[:8], "%Y%m%d").date()
                cache[text] = float(d.toordinal())
            except Exception:
                cache[text] = 0.0
        out[i] = cache[text]
    return out


def make_recency_weights(dates, half_life):
    ords = date_ordinals(dates)
    valid = ords > 0
    if not np.any(valid):
        return np.ones(len(ords), dtype=np.float32)
    newest = float(ords[valid].max())
    age = np.maximum(0.0, newest - ords)
    weights = np.power(0.5, age / float(half_life)).astype(np.float32)
    weights[~valid] = 1.0
    return weights


def build_pairs(users, labels, seed):
    users = np.asarray(users)
    labels = np.asarray(labels) > 0.5
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    rng = np.random.RandomState(seed)
    positives = []
    negatives = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        idx = order[left:right]
        pos = idx[labels[idx]]
        neg = idx[~labels[idx]]
        if len(pos) and len(neg):
            positives.append(pos)
            negatives.append(neg[rng.randint(0, len(neg), size=len(pos))])
    if not positives:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(positives).astype(np.int64), np.concatenate(negatives).astype(np.int64)


def metric_dict(users, labels, scores):
    m = evaluate(users, labels.astype(int), scores)
    return {
        "gauc": float(m.get("GAUC", m.get("gauc"))),
        "ndcg5": float(m.get("nDCG@5", m.get("ndcg5"))),
        "primary": float(m["primary"]),
    }


def predict(model, Xv, device, batch_size=65536):
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(Xv), batch_size):
            xb = Xv[start:start + batch_size].to(device, non_blocking=True)
            parts.append(model(xb).detach().cpu().numpy())
    return np.concatenate(parts).astype(np.float64)


def train_one(config, seed, epochs, total_dim, Xt, yt, Xv, val_users, val_y,
              recency_np, pair_pos, pair_neg, device, row_fraction=1.0,
              half_epoch_checkpoints=False):
    seed_all(seed)
    model = DCNLite(total_dim, dropout=config["dropout"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config["step_size"], gamma=config["step_gamma"]
    )
    n = len(yt)
    rng = np.random.RandomState(seed + 17)
    if row_fraction < 0.999:
        count = max(1, int(n * row_fraction))
        active = rng.choice(n, size=count, replace=False).astype(np.int64)
    else:
        active = np.arange(n, dtype=np.int64)
    if row_fraction < 0.999 and len(pair_pos):
        pair_count = max(1, int(len(pair_pos) * row_fraction))
        chosen = rng.choice(len(pair_pos), size=pair_count, replace=False)
        active_pos = pair_pos[chosen]
        active_neg = pair_neg[chosen]
    else:
        active_pos = pair_pos
        active_neg = pair_neg
    recency = torch.from_numpy(recency_np)
    bs = 8192 if device.type == "cuda" else 4096
    best_primary = -1.0
    best_scores = None
    checkpoints = []
    global_step = 0
    for epoch in range(epochs):
        model.train()
        perm = active[rng.permutation(len(active))]
        if len(active_pos):
            pair_order = rng.permutation(len(active_pos))
        else:
            pair_order = np.empty(0, dtype=np.int64)
        pair_cursor = 0
        split_point = (len(perm) + 1) // 2
        segments = [(0, len(perm))]
        if half_epoch_checkpoints and len(perm) > 1:
            segments = [(0, split_point), (split_point, len(perm))]
        last_loss = 0.0
        for segment_number, (segment_left, segment_right) in enumerate(segments, 1):
            for start in range(segment_left, segment_right, bs):
                idx_np = perm[start:min(start + bs, segment_right)]
                idx = torch.from_numpy(idx_np).to(device)
                xb = Xt[idx_np].to(device, non_blocking=True)
                yb = yt[idx_np].to(device, non_blocking=True)
                wb = recency[idx_np].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                point_losses = F.binary_cross_entropy_with_logits(logits, yb, reduction="none")
                point_loss = (point_losses * wb).sum() / wb.sum().clamp_min(1e-6)
                if len(pair_order):
                    wanted = max(1, len(idx_np) // 2)
                    if pair_cursor + wanted > len(pair_order):
                        pair_order = rng.permutation(len(active_pos))
                        pair_cursor = 0
                    take = pair_order[pair_cursor:pair_cursor + wanted]
                    pair_cursor += len(take)
                    p_np = active_pos[take]
                    n_np = active_neg[take]
                    px = Xt[p_np].to(device, non_blocking=True)
                    nx = Xt[n_np].to(device, non_blocking=True)
                    pw = recency[p_np].to(device, non_blocking=True)
                    difference = model(px) - model(nx)
                    rank_losses = F.softplus(-difference)
                    rank_loss = (rank_losses * pw).sum() / pw.sum().clamp_min(1e-6)
                    loss = 0.5 * point_loss + 0.5 * rank_loss
                else:
                    loss = point_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                global_step += 1
                last_loss = float(loss.detach().cpu())
            if half_epoch_checkpoints:
                scores = predict(model, Xv, device)
                metrics = metric_dict(val_users, val_y, scores)
                checkpoints.append({
                    "epoch": epoch + 0.5 * segment_number,
                    "train_loss": round(last_loss, 6),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "gauc": round(metrics["gauc"], 7),
                    "ndcg5": round(metrics["ndcg5"], 7),
                    "primary": round(metrics["primary"], 7),
                })
                if metrics["primary"] > best_primary:
                    best_primary = metrics["primary"]
                    best_scores = scores.copy()
            model.train()
        scheduler.step()
        if not half_epoch_checkpoints:
            scores = predict(model, Xv, device)
            metrics = metric_dict(val_users, val_y, scores)
            checkpoints.append({
                "epoch": epoch + 1,
                "train_loss": round(last_loss, 6),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "gauc": round(metrics["gauc"], 7),
                "ndcg5": round(metrics["ndcg5"], 7),
                "primary": round(metrics["primary"], 7),
            })
            if metrics["primary"] > best_primary:
                best_primary = metrics["primary"]
                best_scores = scores.copy()
    if best_scores is None:
        best_scores = predict(model, Xv, device)
        best_primary = metric_dict(val_users, val_y, best_scores)["primary"]
    del model, optimizer, scheduler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return float(best_primary), best_scores, checkpoints


def make_coarse_configs(seed):
    rng = np.random.RandomState(seed + 301)
    configs = []
    half_lives = [3.5, 5.0, 7.0, 10.0, 14.0, 21.0]
    for i in range(12):
        configs.append({
            "dropout": float(rng.uniform(0.14, 0.42)),
            "weight_decay": float(np.exp(rng.uniform(np.log(2e-5), np.log(4e-3)))),
            "lr": float(np.exp(rng.uniform(np.log(4e-4), np.log(2.2e-3)))),
            "step_size": int(rng.choice([1, 2, 3])),
            "step_gamma": float(rng.uniform(0.32, 0.78)),
            "half_life": float(half_lives[i % len(half_lives)]),
        })
    return configs


def make_refine_configs(winner, seed):
    rng = np.random.RandomState(seed + 907)
    configs = [dict(winner)]
    for _ in range(7):
        configs.append({
            "dropout": float(np.clip(winner["dropout"] + rng.uniform(-0.065, 0.065), 0.10, 0.48)),
            "weight_decay": float(np.clip(winner["weight_decay"] * np.exp(rng.uniform(-0.8, 0.8)), 1e-5, 8e-3)),
            "lr": float(np.clip(winner["lr"] * np.exp(rng.uniform(-0.38, 0.38)), 2.5e-4, 3e-3)),
            "step_size": int(np.clip(winner["step_size"] + rng.choice([-1, 0, 0, 1]), 1, 4)),
            "step_gamma": float(np.clip(winner["step_gamma"] + rng.uniform(-0.13, 0.13), 0.22, 0.88)),
            "half_life": float(np.clip(winner["half_life"] * np.exp(rng.uniform(-0.42, 0.42)), 2.5, 28.0)),
        })
    return configs


def append_progress(path, record):
    with open(path, "a", encoding="utf-8") as fh:
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
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        device = torch.device("cpu")
        torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

    train_path = os.path.join(args.data_dir, "train.npz")
    val_path = os.path.join(args.data_dir, "val.npz")
    tr = np.load(train_path, allow_pickle=False)
    va = np.load(val_path, allow_pickle=False)

    Xt = torch.from_numpy(tr["X"].astype(np.int64, copy=False))
    yt = torch.from_numpy(tr["y"].astype(np.float32, copy=False))
    Xv = torch.from_numpy(va["X"].astype(np.int64, copy=False))
    val_users = np.asarray(va["user"])
    val_y = np.asarray(va["y"], dtype=np.float32)
    total_dim = int(np.asarray(tr["field_dims"]).sum())
    pair_pos, pair_neg = build_pairs(tr["user"], tr["y"], args.seed + 71)

    smoke_value = os.environ.get("SMOKE_EPOCHS")
    smoke_cap = int(smoke_value) if smoke_value is not None else None
    coarse_epochs = 2
    refine_epochs = 5
    final_epochs = args.epochs
    if smoke_cap is not None:
        coarse_epochs = min(coarse_epochs, smoke_cap)
        refine_epochs = min(refine_epochs, smoke_cap)
        final_epochs = min(final_epochs, smoke_cap)
    coarse_configs = make_coarse_configs(args.seed)
    coarse_repeats = 3
    refine_repeats = 2
    if smoke_cap is not None:
        coarse_configs = coarse_configs[:2]
        coarse_repeats = 1
        refine_repeats = 1

    history = []
    recency_cache = {}

    def weights_for(half_life):
        key = round(float(half_life), 6)
        if key not in recency_cache:
            recency_cache[key] = make_recency_weights(tr["date"], half_life)
        return recency_cache[key]

    coarse_summaries = []
    for config_id, config in enumerate(coarse_configs):
        scores_for_config = []
        for repeat in range(coarse_repeats):
            probe_seed = args.seed + 1000 + config_id * 37 + repeat
            primary, _, checkpoints = train_one(
                config, probe_seed, coarse_epochs, total_dim, Xt, yt, Xv,
                val_users, val_y, weights_for(config["half_life"]),
                pair_pos, pair_neg, device, row_fraction=0.68,
                half_epoch_checkpoints=False
            )
            record = {
                "stage": "coarse", "config_id": config_id, "repeat": repeat,
                "seed": probe_seed, "config": config, "primary": primary,
                "checkpoints": checkpoints,
            }
            history.append(record)
            append_progress(progress_path, {k: record[k] for k in ("stage", "config_id", "repeat", "seed", "config", "primary")})
            scores_for_config.append(primary)
        coarse_summaries.append({
            "config": config,
            "mean_primary": float(np.mean(scores_for_config)),
            "std_primary": float(np.std(scores_for_config)),
        })

    coarse_winner = max(coarse_summaries, key=lambda x: x["mean_primary"])["config"]
    refine_configs = make_refine_configs(coarse_winner, args.seed)
    if smoke_cap is not None:
        refine_configs = refine_configs[:1]
    refine_summaries = []
    for config_id, config in enumerate(refine_configs):
        scores_for_config = []
        for repeat in range(refine_repeats):
            probe_seed = args.seed + 5000 + config_id * 41 + repeat
            primary, _, checkpoints = train_one(
                config, probe_seed, refine_epochs, total_dim, Xt, yt, Xv,
                val_users, val_y, weights_for(config["half_life"]),
                pair_pos, pair_neg, device, row_fraction=1.0,
                half_epoch_checkpoints=False
            )
            record = {
                "stage": "refine", "config_id": config_id, "repeat": repeat,
                "seed": probe_seed, "config": config, "primary": primary,
                "checkpoints": checkpoints,
            }
            history.append(record)
            append_progress(progress_path, {k: record[k] for k in ("stage", "config_id", "repeat", "seed", "config", "primary")})
            scores_for_config.append(primary)
        refine_summaries.append({
            "config": config,
            "mean_primary": float(np.mean(scores_for_config)),
            "std_primary": float(np.std(scores_for_config)),
        })

    winning = max(refine_summaries, key=lambda x: x["mean_primary"])["config"]
    final_seed = args.seed + 9001
    final_primary, final_scores, final_checkpoints = train_one(
        winning, final_seed, final_epochs, total_dim, Xt, yt, Xv,
        val_users, val_y, weights_for(winning["half_life"]),
        pair_pos, pair_neg, device, row_fraction=1.0,
        half_epoch_checkpoints=True
    )
    final_record = {
        "stage": "final", "seed": final_seed, "config": winning,
        "primary": final_primary, "checkpoints": final_checkpoints,
    }
    history.append(final_record)
    append_progress(progress_path, {
        "stage": "final", "seed": final_seed, "config": winning,
        "primary": final_primary,
    })

    final_metrics = metric_dict(val_users, val_y, final_scores)
    metrics_output = {
        "gauc": final_metrics["gauc"],
        "ndcg5": final_metrics["ndcg5"],
        "primary": final_metrics["primary"],
        "winning_config": winning,
        "coarse_summaries": coarse_summaries,
        "refine_summaries": refine_summaries,
        "history": history,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics_output, fh)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w", encoding="utf-8") as fh:
        fh.write("row_id,user_id,video_id,score\n")
        for row_id, score in enumerate(final_scores):
            fh.write(f"{row_id},{val_users[row_id]},0,{score:.8g}\n")


if __name__ == "__main__":
    main()
