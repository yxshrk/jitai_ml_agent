"""Two-stage dial sweep for a regularized DCN-lite/BPR/recency package."""
import argparse
import datetime
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.official.evaluate import evaluate


class DCNFM(torch.nn.Module):
    def __init__(self, total_dim, fields, k, hidden, cross_layers, dropout):
        super().__init__()
        self.dropout = float(dropout)
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        d = fields * k
        self.cross_w = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.empty(d)) for _ in range(cross_layers)]
        )
        self.cross_b = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.zeros(d)) for _ in range(cross_layers)]
        )
        h2 = max(32, hidden // 2)
        self.deep = torch.nn.Sequential(
            torch.nn.Linear(d, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, h2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.cross_out = torch.nn.Linear(d, 1, bias=False)
        self.deep_out = torch.nn.Linear(h2, 1, bias=False)
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)
        for w in self.cross_w:
            torch.nn.init.normal_(w, std=0.015)
        for layer in self.deep:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
        torch.nn.init.normal_(self.cross_out.weight, std=0.01)
        torch.nn.init.normal_(self.deep_out.weight, std=0.01)

    def forward(self, x):
        e0 = self.emb(x)
        e = F.dropout(e0, p=self.dropout, training=self.training)
        summed = e.sum(1)
        fm = 0.5 * (summed.square() - e.square().sum(1)).sum(1)
        x0 = e.flatten(1)
        cross = x0
        for w, b in zip(self.cross_w, self.cross_b):
            cross = x0 * (cross * w).sum(1, keepdim=True) + b + cross
        deep = self.deep(x0)
        return (self.bias + self.lin(x).sum((1, 2)) + fm +
                self.cross_out(cross).squeeze(1) + self.deep_out(deep).squeeze(1))


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metric_values(m):
    return {
        "gauc": float(m["GAUC"] if "GAUC" in m else m["gauc"]),
        "ndcg5": float(m.get("nDCG@5", m.get("ndcg5"))),
        "primary": float(m["primary"]),
    }


def make_pair_indices(users, labels, seed):
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    rng = np.random.default_rng(seed)
    pos_parts = []
    neg_parts = []
    for j in range(len(boundaries) - 1):
        ids = order[boundaries[j]:boundaries[j + 1]]
        positive = ids[labels[ids] >= 0.5]
        negative = ids[labels[ids] < 0.5]
        if len(positive) and len(negative):
            pos_parts.append(positive.astype(np.int64, copy=False))
            neg_parts.append(rng.choice(negative, size=len(positive), replace=True).astype(np.int64))
    if not pos_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(pos_parts), np.concatenate(neg_parts)


def date_ages(values):
    raw = np.asarray(values)
    unique = np.unique(raw)
    parsed = []
    valid = True
    for value in unique:
        try:
            text = str(int(value))
            parsed.append(datetime.datetime.strptime(text, "%Y%m%d").date())
        except (TypeError, ValueError, OverflowError):
            valid = False
            break
    if valid:
        newest = max(parsed)
        age_lookup = np.asarray([(newest - d).days for d in parsed], dtype=np.float32)
    else:
        numeric = unique.astype(np.float64)
        age_lookup = (numeric.max() - numeric).astype(np.float32)
    return age_lookup[np.searchsorted(unique, raw)]


def recency_weights(ages, half_life):
    weights = np.exp2(-ages / float(half_life)).astype(np.float32)
    weights /= max(float(weights.mean()), 1e-8)
    return weights


def make_scheduler(opt, name):
    if name == "step_fast":
        return torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.58)
    if name == "step_two":
        return torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.39)
    if name == "step_three":
        return torch.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.27)
    return torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[1, 3, 6, 9], gamma=0.52)


def score_model(model, Xv, val_users, val_labels):
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(Xv), 65536):
            chunks.append(model(Xv[start:start + 65536]).detach().cpu().numpy())
    scores = np.concatenate(chunks)
    return scores, evaluate(val_users, val_labels, scores)


def train_model(config, epochs, seed, Xt, yt, Xv, pair_pos, pair_neg,
                sample_weights, val_users, val_labels, total_dim, fields,
                device, half_checkpoints=False, keep_state=False):
    set_seed(seed)
    model = DCNFM(
        total_dim=total_dim,
        fields=fields,
        k=16,
        hidden=int(config["hidden"]),
        cross_layers=int(config["cross_layers"]),
        dropout=float(config["dropout"]),
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"])
    )
    scheduler = make_scheduler(opt, config["schedule"])
    n = len(yt)
    bs = 16384 if device.type == "cuda" else 8192
    pair_bs = max(512, bs // 4)
    num_batches = int(math.ceil(n / bs))
    best_primary = -1.0
    best_scores = None
    best_state = None
    curve = []
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(n, device=device)
        if len(pair_pos):
            pair_permutation = torch.randperm(len(pair_pos), device=device)
        else:
            pair_permutation = None
        endpoints = [num_batches]
        if half_checkpoints:
            endpoints = [int(math.ceil(num_batches / 2)), num_batches]
        begin_batch = 0
        for endpoint_index, end_batch in enumerate(endpoints):
            loss_sum = 0.0
            loss_count = 0
            for batch_number in range(begin_batch, end_batch):
                start = batch_number * bs
                idx = permutation[start:min(start + bs, n)]
                b = len(idx)
                opt.zero_grad(set_to_none=True)
                if pair_permutation is not None:
                    q0 = (batch_number * pair_bs) % len(pair_pos)
                    if q0 + pair_bs <= len(pair_pos):
                        qidx = pair_permutation[q0:q0 + pair_bs]
                    else:
                        qidx = torch.cat((pair_permutation[q0:],
                                          pair_permutation[:q0 + pair_bs - len(pair_pos)]))
                    pidx = pair_pos[qidx]
                    nidx = pair_neg[qidx]
                    joined = torch.cat((idx, pidx, nidx))
                    logits = model(Xt[joined])
                    bce_each = F.binary_cross_entropy_with_logits(
                        logits[:b], yt[idx], reduction="none"
                    )
                    wb = sample_weights[idx]
                    bce_loss = (bce_each * wb).sum() / wb.sum().clamp_min(1e-8)
                    pcount = len(pidx)
                    pair_each = F.softplus(-(logits[b:b + pcount] - logits[b + pcount:]))
                    wp = 0.5 * (sample_weights[pidx] + sample_weights[nidx])
                    pair_loss = (pair_each * wp).sum() / wp.sum().clamp_min(1e-8)
                    mix = float(config["bpr_mix"])
                    loss = (1.0 - mix) * bce_loss + mix * pair_loss
                else:
                    logits = model(Xt[idx])
                    each = F.binary_cross_entropy_with_logits(logits, yt[idx], reduction="none")
                    wb = sample_weights[idx]
                    loss = (each * wb).sum() / wb.sum().clamp_min(1e-8)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                loss_sum += float(loss.detach().item())
                loss_count += 1
            begin_batch = end_batch
            scores, measured = score_model(model, Xv, val_users, val_labels)
            primary = float(measured["primary"])
            checkpoint = epoch + (endpoint_index + 1) / len(endpoints)
            curve.append({
                "checkpoint": round(float(checkpoint), 3),
                "train_loss": round(loss_sum / max(loss_count, 1), 6),
                "lr": float(opt.param_groups[0]["lr"]),
                "val_gauc": round(float(measured.get("GAUC", measured.get("gauc"))), 6),
                "val_primary": round(primary, 6),
            })
            if primary > best_primary + 1e-8:
                best_primary = primary
                best_scores = scores.copy()
                if keep_state:
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            model.train()
        scheduler.step()
    if keep_state and best_state is not None:
        model.load_state_dict(best_state)
        best_scores, measured = score_model(model, Xv, val_users, val_labels)
        best_primary = float(measured["primary"])
    del model, opt, scheduler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_primary, best_scores, curve


def coarse_configs(seed):
    rng = np.random.default_rng(seed + 1701)
    schedules = ["step_fast", "step_two", "step_three", "milestone"] * 3
    rng.shuffle(schedules)
    half_lives = np.asarray([3.5, 7.0, 14.0] * 4, dtype=np.float64)
    rng.shuffle(half_lives)
    configs = []
    for i in range(12):
        configs.append({
            "dropout": float(0.15 + 0.25 * rng.random()),
            "weight_decay": float(np.exp(rng.uniform(np.log(3e-5), np.log(3e-3)))),
            "lr": float(np.exp(rng.uniform(np.log(4.5e-4), np.log(1.5e-3)))),
            "schedule": schedules[i],
            "half_life": float(half_lives[i]),
            "hidden": int(rng.choice([64, 96, 128])),
            "cross_layers": int(rng.choice([1, 2])),
            "bpr_mix": float(rng.choice([0.4, 0.5, 0.6])),
        })
    return configs


def refine_configs(winner, seed):
    rng = np.random.default_rng(seed + 2909)
    schedules = ["step_fast", "step_two", "step_three", "milestone"]
    configs = [dict(winner)]
    for _ in range(5):
        c = dict(winner)
        c["dropout"] = float(np.clip(c["dropout"] + rng.normal(0.0, 0.032), 0.10, 0.46))
        c["weight_decay"] = float(np.clip(c["weight_decay"] * np.exp(rng.normal(0.0, 0.38)), 1e-5, 6e-3))
        c["lr"] = float(np.clip(c["lr"] * np.exp(rng.normal(0.0, 0.17)), 3e-4, 2e-3))
        c["half_life"] = float(np.clip(c["half_life"] * np.exp(rng.normal(0.0, 0.18)), 3.0, 18.0))
        c["bpr_mix"] = float(np.clip(c["bpr_mix"] + rng.normal(0.0, 0.055), 0.30, 0.68))
        if rng.random() < 0.45:
            c["schedule"] = schedules[int(rng.integers(0, len(schedules)))]
        if rng.random() < 0.35:
            c["hidden"] = int(rng.choice([64, 96, 128]))
        if rng.random() < 0.25:
            c["cross_layers"] = int(3 - c["cross_layers"])
        configs.append(c)
    return configs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    progress_path = os.path.join(args.out_dir, "progress.log")
    with open(progress_path, "w"):
        pass
    start_time = time.monotonic()
    set_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        device = torch.device("cpu")

    tr = np.load(os.path.join(args.data_dir, "train.npz"))
    va = np.load(os.path.join(args.data_dir, "val.npz"))
    train_x_np = np.asarray(tr["X"], dtype=np.int64)
    val_x_np = np.asarray(va["X"], dtype=np.int64)
    train_y_np = np.asarray(tr["y"], dtype=np.float32)
    val_y_np = np.asarray(va["y"], dtype=np.int64)
    train_users = np.asarray(tr["user"])
    val_users = np.asarray(va["user"])
    field_dims = np.asarray(tr["field_dims"], dtype=np.int64)
    total_dim = int(field_dims.sum())
    fields = int(train_x_np.shape[1])

    pair_pos_np, pair_neg_np = make_pair_indices(train_users, train_y_np, args.seed + 991)
    ages = date_ages(tr["date"])
    Xt = torch.from_numpy(train_x_np).to(device)
    yt = torch.from_numpy(train_y_np).to(device)
    Xv = torch.from_numpy(val_x_np).to(device)
    pair_pos = torch.from_numpy(pair_pos_np).to(device)
    pair_neg = torch.from_numpy(pair_neg_np).to(device)

    smoke_value = os.environ.get("SMOKE_EPOCHS")
    smoke_cap = int(smoke_value) if smoke_value is not None else None
    coarse_default = 3 if device.type == "cuda" else 2
    refine_default = 6 if device.type == "cuda" else 5
    coarse_epochs = coarse_default if smoke_cap is None else min(coarse_default, smoke_cap)
    refine_epochs = refine_default if smoke_cap is None else min(refine_default, smoke_cap)
    final_epochs = args.epochs if smoke_cap is None else min(args.epochs, smoke_cap)
    coarse_epochs = max(1, coarse_epochs)
    refine_epochs = max(1, refine_epochs)
    final_epochs = max(1, final_epochs)

    history = []
    weight_cache = {}

    def weights_for(half_life):
        key = round(float(half_life), 8)
        if key not in weight_cache:
            weight_cache[key] = torch.from_numpy(recency_weights(ages, half_life)).to(device)
        return weight_cache[key]

    stage1 = []
    for probe_id, config in enumerate(coarse_configs(args.seed)):
        probe_start = time.monotonic()
        primary, _, curve = train_model(
            config, coarse_epochs, args.seed, Xt, yt, Xv, pair_pos, pair_neg,
            weights_for(config["half_life"]), val_users, val_y_np,
            total_dim, fields, device
        )
        record = {
            "stage": "coarse", "probe": probe_id + 1, "config": config,
            "epochs": coarse_epochs, "primary": float(primary), "curve": curve,
            "seconds": round(time.monotonic() - probe_start, 3),
        }
        stage1.append((primary, config))
        history.append(record)
        with open(progress_path, "a") as fh:
            fh.write(json.dumps({"stage": "coarse", "probe": probe_id + 1,
                                 "config": config, "primary": float(primary)}) + "\n")

    stage1.sort(key=lambda z: z[0], reverse=True)
    stage2 = []
    for probe_id, config in enumerate(refine_configs(stage1[0][1], args.seed)):
        probe_start = time.monotonic()
        primary, _, curve = train_model(
            config, refine_epochs, args.seed, Xt, yt, Xv, pair_pos, pair_neg,
            weights_for(config["half_life"]), val_users, val_y_np,
            total_dim, fields, device
        )
        record = {
            "stage": "refine", "probe": probe_id + 1, "config": config,
            "epochs": refine_epochs, "primary": float(primary), "curve": curve,
            "seconds": round(time.monotonic() - probe_start, 3),
        }
        stage2.append((primary, config))
        history.append(record)
        with open(progress_path, "a") as fh:
            fh.write(json.dumps({"stage": "refine", "probe": probe_id + 1,
                                 "config": config, "primary": float(primary)}) + "\n")

    stage2.sort(key=lambda z: z[0], reverse=True)
    selected = stage2[0][1]
    final_start = time.monotonic()
    final_primary, final_scores, final_curve = train_model(
        selected, final_epochs, args.seed, Xt, yt, Xv, pair_pos, pair_neg,
        weights_for(selected["half_life"]), val_users, val_y_np,
        total_dim, fields, device, half_checkpoints=True, keep_state=True
    )
    history.append({
        "stage": "final", "probe": 1, "config": selected,
        "epochs": final_epochs, "primary": float(final_primary),
        "curve": final_curve, "seconds": round(time.monotonic() - final_start, 3),
    })
    with open(progress_path, "a") as fh:
        fh.write(json.dumps({"stage": "final", "config": selected,
                             "primary": float(final_primary)}) + "\n")

    final_metrics = metric_values(evaluate(val_users, val_y_np, final_scores))
    output_metrics = {
        "gauc": final_metrics["gauc"],
        "ndcg5": final_metrics["ndcg5"],
        "primary": final_metrics["primary"],
        "selected_config": selected,
        "history": history,
        "runtime_seconds": round(time.monotonic() - start_time, 3),
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(output_metrics, fh)

    video_offset = int(field_dims[0])
    video_ids = val_x_np[:, 1] - video_offset
    with open(os.path.join(args.out_dir, "predictions.csv"), "w") as fh:
        fh.write("row_id,user_id,video_id,score\n")
        for i, score in enumerate(final_scores):
            fh.write(f"{i},{val_users[i]},{video_ids[i]},{float(score):.8g}\n")


if __name__ == "__main__":
    main()
