"""Two-stage dial search for a regularized DCNv2-lite hybrid ranker.

Uses the official five offset-encoded fields from the NPZ fast path, hybrid
pointwise/pairwise training, recency weighting, rapid step decay, and
half-epoch checkpoint selection for the final full-length fit.
"""
import argparse
import datetime
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.official.evaluate import evaluate


class LowRankCross(torch.nn.Module):
    def __init__(self, dim, rank=16):
        super().__init__()
        self.down = torch.nn.Linear(dim, rank, bias=False)
        self.up = torch.nn.Linear(rank, dim, bias=True)
        torch.nn.init.xavier_uniform_(self.down.weight)
        torch.nn.init.xavier_uniform_(self.up.weight)
        torch.nn.init.zeros_(self.up.bias)

    def forward(self, x0, x):
        return x + x0 * self.up(self.down(x))


class DCNFM(torch.nn.Module):
    def __init__(self, total_dim, n_fields, k=16, dropout=0.25):
        super().__init__()
        self.dropout = float(dropout)
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        dim = n_fields * k
        self.crosses = torch.nn.ModuleList([
            LowRankCross(dim, rank=16), LowRankCross(dim, rank=16)
        ])
        self.deep1 = torch.nn.Linear(dim, 128)
        self.deep2 = torch.nn.Linear(128, 64)
        self.deep_out = torch.nn.Linear(64, 1)
        self.cross_out = torch.nn.Linear(dim, 1)
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)
        for layer in (self.deep1, self.deep2, self.deep_out, self.cross_out):
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)

    def forward(self, x):
        raw_e = self.emb(x)
        e = F.dropout(raw_e, p=self.dropout, training=self.training)
        summed = e.sum(1)
        fm = 0.5 * (summed * summed - (e * e).sum(1)).sum(1)
        flat = e.reshape(e.shape[0], -1)
        crossed = flat
        for layer in self.crosses:
            crossed = layer(flat, crossed)
        deep = F.relu(self.deep1(flat))
        deep = F.dropout(deep, p=self.dropout, training=self.training)
        deep = F.relu(self.deep2(deep))
        deep = F.dropout(deep, p=self.dropout, training=self.training)
        return (self.bias + self.lin(x).sum((1, 2)) + fm +
                self.cross_out(crossed).squeeze(1) + self.deep_out(deep).squeeze(1))


def metric_values(users, labels, scores):
    m = evaluate(users, labels, scores)
    return {
        "gauc": float(m["GAUC"] if "GAUC" in m else m["gauc"]),
        "ndcg5": float(m.get("nDCG@5", m.get("ndcg5"))),
        "primary": float(m["primary"]),
    }


def infer(model, x, device):
    model.eval()
    pieces = []
    with torch.no_grad():
        for start in range(0, len(x), 65536):
            xb = x[start:start + 65536].to(device, non_blocking=True)
            pieces.append(model(xb).detach().cpu().numpy())
    return np.concatenate(pieces)


def date_ages(raw_dates):
    values = np.asarray(raw_dates)
    unique = np.unique(values)
    mapping = {}
    for value in unique:
        ivalue = int(value)
        year = ivalue // 10000
        month = (ivalue // 100) % 100
        day = ivalue % 100
        mapping[ivalue] = datetime.date(year, month, day).toordinal()
    ordinals = np.fromiter((mapping[int(v)] for v in values), dtype=np.int64,
                           count=len(values))
    return (ordinals.max() - ordinals).astype(np.float32)


def make_pairs(users, labels, seed):
    rng = np.random.RandomState(seed)
    users = np.asarray(users)
    labels = np.asarray(labels) > 0.5
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    pos_parts = []
    neg_parts = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        idx = order[left:right]
        pos = idx[labels[idx]]
        neg = idx[~labels[idx]]
        if len(pos) == 0 or len(neg) == 0:
            continue
        count = max(len(pos), len(neg))
        pos_parts.append(rng.choice(pos, size=count, replace=len(pos) < count))
        neg_parts.append(rng.choice(neg, size=count, replace=len(neg) < count))
    if not pos_parts:
        raise RuntimeError("No users with both positive and negative training labels")
    return (torch.from_numpy(np.concatenate(pos_parts).astype(np.int64)),
            torch.from_numpy(np.concatenate(neg_parts).astype(np.int64)))


def rounded_config(config):
    result = {}
    for key, value in config.items():
        result[key] = round(float(value), 8)
    return result


def train_candidate(config, model_seed, epochs, row_fraction, eval_half_epoch,
                    xt, yt, xv, val_users, val_labels, ages, pair_pos, pair_neg,
                    total_dim, n_fields, device):
    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(model_seed + 7919)
    model = DCNFM(total_dim, n_fields, k=16,
                  dropout=config["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                  weight_decay=config["weight_decay"])
    recency = np.exp(-math.log(2.0) * ages / config["half_life"]).astype(np.float32)
    recency /= max(float(recency.mean()), 1e-8)
    weights = torch.from_numpy(recency)
    n = len(yt)
    max_rows = max(2, min(n, int(round(n * row_fraction))))
    batch_size = 8192
    pair_count = len(pair_pos)
    completed_halves = 0
    start_half = max(1, int(round(config["decay_start"] * 2.0)))
    every_half = max(1, int(round(config["decay_every"] * 2.0)))
    best_primary = -1.0
    best_scores = None
    checkpoints = []

    for epoch in range(epochs):
        permutation = torch.randperm(n, generator=generator)[:max_rows]
        split = (max_rows + 1) // 2
        halves = (permutation[:split], permutation[split:])
        for half_index, half_indices in enumerate(halves):
            if len(half_indices) == 0:
                continue
            model.train()
            loss_total = 0.0
            steps = 0
            for start in range(0, len(half_indices), batch_size):
                idx = half_indices[start:start + batch_size]
                pairs_needed = max(1, len(idx) // 2)
                selected_pairs = torch.randint(pair_count, (pairs_needed,),
                                                generator=generator)
                pidx = pair_pos[selected_pairs]
                nidx = pair_neg[selected_pairs]
                xb = xt[idx].to(device, non_blocking=True)
                xp = xt[pidx].to(device, non_blocking=True)
                xn = xt[nidx].to(device, non_blocking=True)
                all_logits = model(torch.cat((xb, xp, xn), dim=0))
                nb = len(idx)
                npair = len(pidx)
                logits = all_logits[:nb]
                pos_logits = all_logits[nb:nb + npair]
                neg_logits = all_logits[nb + npair:]
                yb = yt[idx].to(device, non_blocking=True)
                wb = weights[idx].to(device, non_blocking=True)
                point_losses = F.binary_cross_entropy_with_logits(logits, yb,
                                                                   reduction="none")
                point_loss = (point_losses * wb).sum() / wb.sum().clamp_min(1e-8)
                pair_weights = 0.5 * (weights[pidx] + weights[nidx])
                pair_weights = pair_weights.to(device, non_blocking=True)
                rank_losses = F.softplus(-(pos_logits - neg_logits))
                rank_loss = ((rank_losses * pair_weights).sum() /
                             pair_weights.sum().clamp_min(1e-8))
                loss = 0.5 * point_loss + 0.5 * rank_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                loss_total += float(loss.detach().cpu())
                steps += 1

            completed_halves += 1
            should_evaluate = eval_half_epoch or half_index == 1
            if should_evaluate:
                scores = infer(model, xv, device)
                metrics = metric_values(val_users, val_labels, scores)
                checkpoint = {
                    "epoch": round(epoch + (half_index + 1) / 2.0, 1),
                    "train_loss": round(loss_total / max(steps, 1), 6),
                    "lr": round(float(optimizer.param_groups[0]["lr"]), 9),
                    "val_gauc": round(metrics["gauc"], 6),
                    "val_primary": round(metrics["primary"], 6),
                }
                checkpoints.append(checkpoint)
                if metrics["primary"] > best_primary + 1e-8:
                    best_primary = metrics["primary"]
                    best_scores = scores.copy()
            if completed_halves >= start_half and (completed_halves - start_half) % every_half == 0:
                for group in optimizer.param_groups:
                    group["lr"] *= config["decay_gamma"]

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_primary, best_scores, checkpoints


def coarse_configs(seed):
    rng = np.random.RandomState(seed + 13007)
    configs = []
    for _ in range(10):
        configs.append({
            "dropout": float(rng.uniform(0.14, 0.42)),
            "weight_decay": float(np.exp(rng.uniform(np.log(2.5e-5), np.log(3.5e-3)))),
            "lr": float(np.exp(rng.uniform(np.log(4.5e-4), np.log(1.7e-3)))),
            "decay_gamma": float(rng.uniform(0.24, 0.63)),
            "decay_start": float(rng.choice([0.5, 1.0, 1.5, 2.0])),
            "decay_every": float(rng.choice([0.5, 1.0, 1.5, 2.0])),
            "half_life": float(rng.choice([3.0, 4.5, 6.5, 9.5, 13.5, 18.0])),
        })
    return configs


def refined_configs(winner, seed):
    rng = np.random.RandomState(seed + 29023)
    configs = [dict(winner)]
    for _ in range(3):
        config = dict(winner)
        config["dropout"] = float(np.clip(config["dropout"] + rng.normal(0.0, 0.025),
                                          0.10, 0.48))
        config["weight_decay"] = float(np.clip(config["weight_decay"] *
                                                np.exp(rng.normal(0.0, 0.28)),
                                                1e-5, 6e-3))
        config["lr"] = float(np.clip(config["lr"] * np.exp(rng.normal(0.0, 0.16)),
                                     2.5e-4, 2.2e-3))
        config["decay_gamma"] = float(np.clip(config["decay_gamma"] +
                                               rng.normal(0.0, 0.055), 0.15, 0.75))
        config["half_life"] = float(np.clip(config["half_life"] *
                                             np.exp(rng.normal(0.0, 0.18)), 2.0, 24.0))
        config["decay_start"] = float(np.clip(config["decay_start"] +
                                               rng.choice([-0.5, 0.0, 0.5]), 0.5, 3.0))
        config["decay_every"] = float(np.clip(config["decay_every"] +
                                               rng.choice([-0.5, 0.0, 0.5]), 0.5, 2.5))
        configs.append(config)
    return configs


def append_progress(path, record):
    with open(path, "a") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")
        fh.flush()


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

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train = np.load(os.path.join(args.data_dir, "train.npz"))
    val = np.load(os.path.join(args.data_dir, "val.npz"))
    xt = torch.from_numpy(train["X"].astype(np.int64))
    yt = torch.from_numpy(train["y"].astype(np.float32))
    xv = torch.from_numpy(val["X"].astype(np.int64))
    val_users = np.asarray(val["user"])
    val_labels = val["y"].astype(int)
    total_dim = int(train["field_dims"].sum())
    n_fields = int(train["X"].shape[1])
    ages = date_ages(train["date"])
    pair_pos, pair_neg = make_pairs(train["user"], train["y"], args.seed + 17)

    smoke_value = os.environ.get("SMOKE_EPOCHS")
    smoke_cap = int(smoke_value) if smoke_value is not None else None
    coarse_epochs = 2 if smoke_cap is None else max(1, min(2, smoke_cap))
    refine_epochs = 4 if smoke_cap is None else max(1, min(4, smoke_cap))
    final_epochs = max(1, args.epochs)
    if smoke_cap is not None:
        final_epochs = max(1, min(final_epochs, smoke_cap))

    history = []
    stage1_results = []
    for probe_index, config in enumerate(coarse_configs(args.seed)):
        primary, _, checkpoints = train_candidate(
            config, args.seed + 100 + probe_index, coarse_epochs, 0.45, False,
            xt, yt, xv, val_users, val_labels, ages, pair_pos, pair_neg,
            total_dim, n_fields, device)
        record = {
            "stage": "coarse",
            "probe": probe_index + 1,
            "config": rounded_config(config),
            "primary": round(float(primary), 6),
            "checkpoints": checkpoints,
        }
        history.append(record)
        stage1_results.append((primary, config))
        append_progress(progress_path, {k: v for k, v in record.items()
                                        if k != "checkpoints"})

    stage1_results.sort(key=lambda item: item[0], reverse=True)
    coarse_winner = stage1_results[0][1]
    stage2_results = []
    for probe_index, config in enumerate(refined_configs(coarse_winner, args.seed)):
        primary, _, checkpoints = train_candidate(
            config, args.seed + 1000 + probe_index, refine_epochs, 1.0, False,
            xt, yt, xv, val_users, val_labels, ages, pair_pos, pair_neg,
            total_dim, n_fields, device)
        record = {
            "stage": "refine",
            "probe": probe_index + 1,
            "config": rounded_config(config),
            "primary": round(float(primary), 6),
            "checkpoints": checkpoints,
        }
        history.append(record)
        stage2_results.append((primary, config))
        append_progress(progress_path, {k: v for k, v in record.items()
                                        if k != "checkpoints"})

    stage2_results.sort(key=lambda item: item[0], reverse=True)
    final_config = stage2_results[0][1]
    final_primary, best_scores, final_checkpoints = train_candidate(
        final_config, args.seed + 5000, final_epochs, 1.0, True,
        xt, yt, xv, val_users, val_labels, ages, pair_pos, pair_neg,
        total_dim, n_fields, device)
    final_record = {
        "stage": "final",
        "probe": 1,
        "config": rounded_config(final_config),
        "primary": round(float(final_primary), 6),
        "checkpoints": final_checkpoints,
    }
    history.append(final_record)

    final_metrics = metric_values(val_users, val_labels, best_scores)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump({
            "gauc": final_metrics["gauc"],
            "ndcg5": final_metrics["ndcg5"],
            "primary": final_metrics["primary"],
            "selected_config": rounded_config(final_config),
            "history": history,
        }, fh)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w") as fh:
        fh.write("row_id,user_id,video_id,score\n")
        for row_id, score in enumerate(best_scores):
            fh.write(f"{row_id},{val_users[row_id]},0,{score:.6g}\n")


if __name__ == "__main__":
    main()
