import argparse
import copy
import datetime
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.official.evaluate import evaluate


class DCNLite(torch.nn.Module):
    def __init__(self, total_dim, fields=5, k=16, hidden=128, dropout=0.25):
        super().__init__()
        self.fields = fields
        self.k = k
        d = fields * k
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.emb_drop = torch.nn.Dropout(dropout)
        self.cross_w = torch.nn.Parameter(torch.empty(d))
        self.cross_b = torch.nn.Parameter(torch.zeros(d))
        self.deep = torch.nn.Sequential(
            torch.nn.Linear(d, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, hidden // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden // 2, 1),
        )
        self.cross_out = torch.nn.Linear(d, 1)
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)
        torch.nn.init.normal_(self.cross_w, std=0.01)
        torch.nn.init.xavier_uniform_(self.cross_out.weight)
        torch.nn.init.zeros_(self.cross_out.bias)
        for mod in self.deep:
            if isinstance(mod, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(mod.weight)
                torch.nn.init.zeros_(mod.bias)

    def forward(self, x):
        e0 = self.emb(x)
        e = self.emb_drop(e0)
        summed = e.sum(1)
        fm = 0.5 * (summed.square() - e.square().sum(1)).sum(1)
        linear = self.lin(x).sum((1, 2))
        flat = e.reshape(e.shape[0], -1)
        scalar = (flat * self.cross_w).sum(1, keepdim=True)
        cross = flat + flat * scalar + self.cross_b
        return self.bias + linear + fm + self.cross_out(cross).squeeze(1) + self.deep(flat).squeeze(1)


class ParentFM(torch.nn.Module):
    def __init__(self, total_dim, k=16):
        super().__init__()
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)

    def forward(self, x):
        e = self.emb(x)
        summed = e.sum(1)
        pair = 0.5 * (summed.square() - e.square().sum(1)).sum(1)
        return self.bias + self.lin(x).sum((1, 2)) + pair


def metric_values(metrics):
    return (
        float(metrics.get("GAUC", metrics.get("gauc"))),
        float(metrics.get("nDCG@5", metrics.get("ndcg5"))),
        float(metrics["primary"]),
    )


def date_ordinals(values):
    vals = np.asarray(values).reshape(-1)
    unique, inv = np.unique(vals, return_inverse=True)
    converted = np.empty(len(unique), dtype=np.float64)
    for i, value in enumerate(unique):
        text = str(value).replace("-", "")[:8]
        try:
            converted[i] = datetime.date(int(text[:4]), int(text[4:6]), int(text[6:8])).toordinal()
        except Exception:
            converted[i] = float(i)
    return converted[inv]


def recency_weights(dates, half_life):
    ords = date_ordinals(dates)
    age = ords.max() - ords
    weights = np.power(0.5, age / float(half_life)).astype(np.float32)
    weights /= max(float(weights.mean()), 1e-8)
    return weights


def make_pair_data(users, labels):
    users = np.asarray(users).reshape(-1)
    labels = np.asarray(labels).reshape(-1) > 0.5
    _, group = np.unique(users, return_inverse=True)
    groups = int(group.max()) + 1
    neg_idx = np.flatnonzero(~labels)
    order = np.argsort(group[neg_idx], kind="stable")
    neg_sorted = neg_idx[order].astype(np.int64)
    neg_groups = group[neg_sorted]
    counts = np.bincount(neg_groups, minlength=groups).astype(np.int64)
    starts = np.zeros(groups, dtype=np.int64)
    if groups > 1:
        starts[1:] = np.cumsum(counts[:-1])
    pos = np.flatnonzero(labels & (counts[group] > 0)).astype(np.int64)
    return pos, group.astype(np.int64), neg_sorted, starts, counts


def predict(model, X, device, batch=65536):
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(X), batch):
            xb = X[start:start + batch].to(device, non_blocking=True)
            chunks.append(model(xb).detach().cpu().numpy())
    return np.concatenate(chunks).astype(np.float64)


def lr_multiplier(schedule, progress):
    if schedule == "rapid_half":
        return 0.55 ** int(progress / 0.5)
    if schedule == "rapid_one":
        return 0.42 ** int(progress / 1.0)
    if schedule == "stair_1p5":
        return 0.35 ** int(progress / 1.5)
    if schedule == "two_drop":
        if progress >= 3.0:
            return 0.12
        if progress >= 1.5:
            return 0.38
        return 1.0
    return 1.0


def sample_pairs(rng, pair_data, pair_n, negative_count=1):
    pos_pool, group, neg_sorted, neg_starts, neg_counts = pair_data
    chosen = pos_pool[rng.randint(0, len(pos_pool), size=pair_n)]
    chosen_group = group[chosen]
    random_values = rng.random_sample((pair_n, negative_count))
    offsets = (random_values * neg_counts[chosen_group, None]).astype(np.int64)
    negatives = neg_sorted[neg_starts[chosen_group, None] + offsets]
    return chosen.astype(np.int64), negatives.astype(np.int64)


def bpr_loss(model, Xt, train_weights, pair_data, rng, pair_n, device):
    chosen, negatives = sample_pairs(rng, pair_data, pair_n, 1)
    negative_flat = negatives[:, 0]
    pair_indices = np.concatenate([chosen, negative_flat])
    pair_x = Xt[torch.from_numpy(pair_indices).long()].to(device, non_blocking=True)
    pair_scores = model(pair_x)
    pos_scores = pair_scores[:pair_n]
    neg_scores = pair_scores[pair_n:]
    pair_w = train_weights[torch.from_numpy(chosen).long()].to(device, non_blocking=True)
    losses = torch.nn.functional.softplus(-(pos_scores - neg_scores))
    return (losses * pair_w).sum() / pair_w.sum().clamp_min(1e-6)


def sampled_lambda_loss(model, Xt, train_weights, pair_data, rng, pair_n, device, negative_count=4):
    chosen, negatives = sample_pairs(rng, pair_data, pair_n, negative_count)
    all_indices = np.concatenate([chosen[:, None], negatives], axis=1)
    flat_indices = all_indices.reshape(-1)
    all_x = Xt[torch.from_numpy(flat_indices).long()].to(device, non_blocking=True)
    scores = model(all_x).reshape(pair_n, negative_count + 1)
    pos_scores = scores[:, :1]
    neg_scores = scores[:, 1:]
    with torch.no_grad():
        comparisons = scores.unsqueeze(1) > scores.unsqueeze(2)
        ranks = 1 + comparisons.sum(dim=1)
        ranks = ranks.to(torch.float32)
        discounts = torch.where(
            ranks <= 5.0,
            1.0 / torch.log2(ranks + 1.0),
            torch.zeros_like(ranks),
        )
        delta_ndcg = torch.abs(discounts[:, :1] - discounts[:, 1:])
        lambda_weights = delta_ndcg + 0.05
        lambda_weights = lambda_weights / lambda_weights.mean().clamp_min(1e-6)
        row_weights = train_weights[torch.from_numpy(chosen).long()].to(device, non_blocking=True)
        lambda_weights = lambda_weights * row_weights[:, None]
    losses = torch.nn.functional.softplus(-(pos_scores - neg_scores))
    return (losses * lambda_weights).sum() / lambda_weights.sum().clamp_min(1e-6)


def train_package(cfg, objective, seed, epochs, Xt, yt, Xv, val_user, val_y, train_weights, pair_data, total_dim, user_dim, device, half_checkpoints):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    rng = np.random.RandomState(seed)
    model = DCNLite(total_dim, dropout=float(cfg["dropout"])).to(device)
    gauge_hook = None
    if objective == "gauge_bce":
        with torch.no_grad():
            model.lin.weight[:user_dim].zero_()
        mask = torch.ones_like(model.lin.weight)
        mask[:user_dim] = 0.0
        gauge_hook = model.lin.weight.register_hook(lambda grad: grad * mask)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    n = len(yt)
    batch_size = 16384
    checks_per_epoch = 2 if half_checkpoints else 1
    best_primary = -1.0
    best_scores = None
    trace = []
    last_loss = 0.0
    for epoch in range(epochs):
        permutation = rng.permutation(n)
        segment_edges = [0, n // 2, n] if half_checkpoints else [0, n]
        for segment in range(checks_per_epoch):
            model.train()
            left, right = segment_edges[segment], segment_edges[segment + 1]
            progress = epoch + segment / float(checks_per_epoch)
            current_lr = float(cfg["lr"]) * lr_multiplier(cfg["schedule"], progress)
            for group in opt.param_groups:
                group["lr"] = current_lr
            for start in range(left, right, batch_size):
                idx_np = permutation[start:min(start + batch_size, right)]
                if len(idx_np) == 0:
                    continue
                idx = torch.from_numpy(idx_np).long()
                xb = Xt[idx].to(device, non_blocking=True)
                yb = yt[idx].to(device, non_blocking=True)
                wb = train_weights[idx].to(device, non_blocking=True)
                logits = model(xb)
                bce_each = torch.nn.functional.binary_cross_entropy_with_logits(logits, yb, reduction="none")
                bce = (bce_each * wb).sum() / wb.sum().clamp_min(1e-6)
                if objective == "hybrid_bpr":
                    pair = bpr_loss(model, Xt, train_weights, pair_data, rng, len(idx_np), device)
                    loss = 0.5 * bce + 0.5 * pair
                elif objective == "gauge_bce":
                    loss = bce
                elif objective == "lambda_rank":
                    pair_n = min(len(idx_np), 4096)
                    lam = sampled_lambda_loss(model, Xt, train_weights, pair_data, rng, pair_n, device, 4)
                    loss = 0.35 * bce + 0.65 * lam
                else:
                    raise ValueError("unknown objective")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                if objective == "gauge_bce":
                    with torch.no_grad():
                        model.lin.weight[:user_dim].zero_()
                last_loss = float(loss.detach().cpu())
            scores = predict(model, Xv, device)
            metrics = evaluate(val_user, val_y, scores)
            gauc, ndcg, primary = metric_values(metrics)
            checkpoint = epoch + (segment + 1) / float(checks_per_epoch)
            trace.append({
                "checkpoint": checkpoint,
                "objective": objective,
                "train_loss": round(last_loss, 6),
                "lr": current_lr,
                "val_gauc": round(gauc, 6),
                "val_ndcg5": round(ndcg, 6),
                "val_primary": round(primary, 6),
            })
            if primary > best_primary + 1e-8:
                best_primary = primary
                best_scores = scores.copy()
    if gauge_hook is not None:
        gauge_hook.remove()
    model.to("cpu")
    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_primary, best_scores, trace


def train_parent_reference(seed, epochs, Xt, yt, Xv, val_user, val_y, total_dim, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    rng = np.random.RandomState(seed)
    model = ParentFM(total_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n = len(yt)
    best = -1.0
    best_scores = None
    patience = 0
    for _ in range(epochs):
        model.train()
        permutation = rng.permutation(n)
        for start in range(0, n, 8192):
            idx = torch.from_numpy(permutation[start:start + 8192]).long()
            xb = Xt[idx].to(device, non_blocking=True)
            yb = yt[idx].to(device, non_blocking=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        scores = predict(model, Xv, device)
        primary = metric_values(evaluate(val_user, val_y, scores))[2]
        if primary > best + 1e-8:
            best = primary
            best_scores = scores.copy()
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break
    model.to("cpu")
    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_scores


def per_user_rank_transform(scores, users):
    scores = np.asarray(scores, dtype=np.float64)
    users = np.asarray(users).reshape(-1)
    _, inverse = np.unique(users, return_inverse=True)
    group_order = np.argsort(inverse, kind="stable")
    sorted_groups = inverse[group_order]
    boundaries = np.flatnonzero(np.r_[True, sorted_groups[1:] != sorted_groups[:-1], True])
    ranks = np.empty(len(scores), dtype=np.float64)
    for boundary in range(len(boundaries) - 1):
        idx = group_order[boundaries[boundary]:boundaries[boundary + 1]]
        count = len(idx)
        if count == 1:
            ranks[idx[0]] = 0.5
        else:
            local_order = np.argsort(scores[idx], kind="mergesort")
            local_ranks = np.empty(count, dtype=np.float64)
            local_ranks[local_order] = np.arange(count, dtype=np.float64) / float(count - 1)
            ranks[idx] = local_ranks
    return ranks


def append_progress(path, record):
    with open(path, "a") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=18)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    progress_path = os.path.join(args.out_dir, "progress.log")
    if os.path.exists(progress_path):
        os.remove(progress_path)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    train_path = os.path.join(args.data_dir, "train.npz")
    val_path = os.path.join(args.data_dir, "val.npz")
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError("hetero-objective ensemble requires the npz fast path")

    tr = np.load(train_path)
    va = np.load(val_path)
    Xt = torch.from_numpy(tr["X"].astype(np.int64))
    yt = torch.from_numpy(tr["y"].astype(np.float32))
    Xv_array = va["X"].astype(np.int64)
    Xv = torch.from_numpy(Xv_array)
    val_user = np.asarray(va["user"])
    val_y = np.asarray(va["y"]).astype(int)
    field_dims = np.asarray(tr["field_dims"]).astype(np.int64)
    total_dim = int(field_dims.sum())
    user_dim = int(field_dims[0])
    pair_data = make_pair_data(tr["user"], tr["y"])
    if len(pair_data[0]) == 0:
        raise RuntimeError("no eligible within-user positive-negative pairs")

    smoke_value = os.environ.get("SMOKE_EPOCHS")
    smoke_cap = int(smoke_value) if smoke_value is not None else None
    coarse_epochs = 3 if smoke_cap is None else min(3, smoke_cap)
    refine_epochs = 6 if smoke_cap is None else min(6, smoke_cap)
    final_epochs = args.epochs if smoke_cap is None else min(args.epochs, smoke_cap)
    coarse_count = 12 if smoke_cap is None else 3
    refine_count = 10 if smoke_cap is None else 2

    rng = np.random.RandomState(args.seed + 7919)
    schedules = ["rapid_half", "rapid_one", "stair_1p5", "two_drop"]
    half_lives = [3.5, 7.0, 14.0]
    coarse_configs = []
    for i in range(coarse_count):
        frac = (i + rng.uniform(0.05, 0.95)) / float(coarse_count)
        coarse_configs.append({
            "dropout": float(rng.uniform(0.15, 0.40)),
            "weight_decay": float(10.0 ** (-4.522878745 + frac * 2.0)),
            "lr": float(10.0 ** rng.uniform(math.log10(4.5e-4), math.log10(1.4e-3))),
            "schedule": schedules[i % len(schedules)],
            "half_life": float(half_lives[(i * 2 + 1) % len(half_lives)]),
        })
    rng.shuffle(coarse_configs)

    history = []
    weight_cache = {}
    best_probe = None
    for i, cfg in enumerate(coarse_configs):
        half_life = cfg["half_life"]
        if half_life not in weight_cache:
            weight_cache[half_life] = torch.from_numpy(recency_weights(tr["date"], half_life))
        primary, _, trace = train_package(
            cfg, "hybrid_bpr", args.seed + 1000 + i, coarse_epochs,
            Xt, yt, Xv, val_user, val_y, weight_cache[half_life], pair_data,
            total_dim, user_dim, device, False,
        )
        record = {
            "stage": "coarse",
            "probe": i,
            "objective": "hybrid_bpr",
            "seed": args.seed + 1000 + i,
            "epochs": coarse_epochs,
            "config": cfg,
            "primary": round(primary, 7),
            "trace": trace,
        }
        history.append(record)
        append_progress(progress_path, {"stage": "coarse", "probe": i, "config": cfg, "primary": round(primary, 7)})
        if best_probe is None or primary > best_probe[0]:
            best_probe = (primary, copy.deepcopy(cfg))

    center = best_probe[1]
    neighboring_schedules = [center["schedule"]] + [schedule for schedule in schedules if schedule != center["schedule"]]
    half_options = sorted(set([3.5, 5.0, 7.0, 10.0, 14.0, center["half_life"]]))
    refine_configs = []
    for i in range(refine_count):
        cfg = {
            "dropout": float(np.clip(center["dropout"] + rng.normal(0.0, 0.045), 0.12, 0.44)),
            "weight_decay": float(np.clip(center["weight_decay"] * (10.0 ** rng.uniform(-0.38, 0.38)), 2e-5, 5e-3)),
            "lr": float(np.clip(center["lr"] * (10.0 ** rng.uniform(-0.20, 0.20)), 3.5e-4, 1.7e-3)),
            "schedule": neighboring_schedules[i % min(3, len(neighboring_schedules))],
            "half_life": float(min(half_options, key=lambda value: abs(math.log(value / center["half_life"]) - rng.normal(0.0, 0.24)))),
        }
        if i == 0:
            cfg = copy.deepcopy(center)
        refine_configs.append(cfg)

    best_refined = None
    for i, cfg in enumerate(refine_configs):
        half_life = cfg["half_life"]
        if half_life not in weight_cache:
            weight_cache[half_life] = torch.from_numpy(recency_weights(tr["date"], half_life))
        primary, _, trace = train_package(
            cfg, "hybrid_bpr", args.seed + 3000 + i, refine_epochs,
            Xt, yt, Xv, val_user, val_y, weight_cache[half_life], pair_data,
            total_dim, user_dim, device, False,
        )
        record = {
            "stage": "refine",
            "probe": i,
            "objective": "hybrid_bpr",
            "seed": args.seed + 3000 + i,
            "epochs": refine_epochs,
            "config": cfg,
            "primary": round(primary, 7),
            "trace": trace,
        }
        history.append(record)
        append_progress(progress_path, {"stage": "refine", "probe": i, "config": cfg, "primary": round(primary, 7)})
        if best_refined is None or primary > best_refined[0]:
            best_refined = (primary, copy.deepcopy(cfg))

    winning_cfg = best_refined[1]
    winning_half = winning_cfg["half_life"]
    if winning_half not in weight_cache:
        weight_cache[winning_half] = torch.from_numpy(recency_weights(tr["date"], winning_half))

    objectives = ["hybrid_bpr", "gauge_bce", "lambda_rank"]
    member_scores = []
    member_metrics = []
    for member, objective in enumerate(objectives):
        member_seed = args.seed + member
        primary, scores, trace = train_package(
            winning_cfg, objective, member_seed, final_epochs,
            Xt, yt, Xv, val_user, val_y, weight_cache[winning_half], pair_data,
            total_dim, user_dim, device, True,
        )
        metrics = evaluate(val_user, val_y, scores)
        gauc, ndcg, measured_primary = metric_values(metrics)
        member_scores.append(scores)
        member_metrics.append({
            "objective": objective,
            "seed": member_seed,
            "gauc": gauc,
            "ndcg5": ndcg,
            "primary": measured_primary,
        })
        record = {
            "stage": "objective_member",
            "member": member,
            "objective": objective,
            "seed": member_seed,
            "epochs": final_epochs,
            "config": winning_cfg,
            "primary": round(primary, 7),
            "trace": trace,
        }
        history.append(record)
        append_progress(progress_path, {
            "stage": "objective_member",
            "member": member,
            "objective": objective,
            "seed": member_seed,
            "primary": round(measured_primary, 7),
        })

    for i in range(len(member_scores)):
        for j in range(i):
            assert not np.allclose(member_scores[i], member_scores[j]), "objective members produced identical scores"

    parent_epochs = 12 if smoke_cap is None else min(12, smoke_cap)
    parent_scores = train_parent_reference(
        args.seed + 90000, parent_epochs, Xt, yt, Xv, val_user, val_y, total_dim, device
    )
    for scores in member_scores:
        assert not np.allclose(scores, parent_scores), "objective member is identical to parent predictions"

    ranked_members = [per_user_rank_transform(scores, val_user) for scores in member_scores]
    blend_scores = np.mean(np.stack(ranked_members, axis=0), axis=0)
    blend_metrics_raw = evaluate(val_user, val_y, blend_scores)
    blend_gauc, blend_ndcg, blend_primary = metric_values(blend_metrics_raw)

    best_member_index = int(np.argmax([metrics["primary"] for metrics in member_metrics]))
    best_member_primary = member_metrics[best_member_index]["primary"]
    if blend_primary >= best_member_primary:
        final_scores = blend_scores
        selected = "fixed_equal_objective_rank_blend"
    else:
        final_scores = member_scores[best_member_index]
        selected = "rescue_harm_gate_member_fallback"

    final_metrics_raw = evaluate(val_user, val_y, final_scores)
    gauc, ndcg, primary = metric_values(final_metrics_raw)
    history.append({
        "stage": "ensemble_gate",
        "fixed_weights": [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        "rank_scope": "within_user",
        "member_metrics": member_metrics,
        "blend_gauc": round(blend_gauc, 7),
        "blend_ndcg5": round(blend_ndcg, 7),
        "blend_primary": round(blend_primary, 7),
        "best_member_index": best_member_index,
        "best_member_primary": round(best_member_primary, 7),
        "selected": selected,
        "val_gauc": round(gauc, 7),
        "val_ndcg5": round(ndcg, 7),
        "val_primary": round(primary, 7),
    })
    append_progress(progress_path, {
        "stage": "ensemble_gate",
        "member_primaries": [round(metrics["primary"], 7) for metrics in member_metrics],
        "blend_primary": round(blend_primary, 7),
        "selected": selected,
        "final_primary": round(primary, 7),
    })

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump({
            "gauc": gauc,
            "ndcg5": ndcg,
            "primary": primary,
            "winning_config": winning_cfg,
            "selected": selected,
            "history": history,
        }, fh)

    video_offset = int(field_dims[0])
    val_video = Xv_array[:, 1] - video_offset
    with open(os.path.join(args.out_dir, "predictions.csv"), "w") as fh:
        fh.write("row_id,user_id,video_id,score\n")
        for i, score in enumerate(final_scores):
            fh.write(f"{i},{val_user[i]},{val_video[i]},{score:.9g}\n")


if __name__ == "__main__":
    main()
