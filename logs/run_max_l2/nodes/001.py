import argparse
import copy
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


class DCNLite(torch.nn.Module):
    def __init__(self, total_dim, fields=5, k=16, hidden=128, dropout=0.2, cross_layers=2):
        super().__init__()
        self.fields = fields
        self.k = k
        width = fields * k
        self.emb = torch.nn.Embedding(total_dim, k)
        self.linear = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.emb_drop = torch.nn.Dropout(dropout)
        self.cross_w = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.empty(width)) for _ in range(cross_layers)]
        )
        self.cross_b = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.zeros(width)) for _ in range(cross_layers)]
        )
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(width, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, hidden // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.out = torch.nn.Linear(width + hidden // 2, 1)
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.linear.weight)
        for w in self.cross_w:
            torch.nn.init.normal_(w, std=0.01)
        for module in self.mlp:
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                torch.nn.init.zeros_(module.bias)
        torch.nn.init.xavier_uniform_(self.out.weight)
        torch.nn.init.zeros_(self.out.bias)

    def forward(self, x):
        e = self.emb_drop(self.emb(x))
        x0 = e.reshape(e.shape[0], -1)
        cross = x0
        for w, b in zip(self.cross_w, self.cross_b):
            scale = (cross * w).sum(dim=1, keepdim=True)
            cross = cross + x0 * scale + b
        deep = self.mlp(x0)
        nonlinear = self.out(torch.cat((cross, deep), dim=1)).squeeze(1)
        return self.bias + self.linear(x).sum(dim=(1, 2)) + nonlinear


def metric_values(users, labels, scores):
    m = evaluate(users, labels, scores)
    return {
        "gauc": float(m.get("GAUC", m.get("gauc"))),
        "ndcg5": float(m.get("nDCG@5", m.get("ndcg5"))),
        "primary": float(m["primary"]),
    }


def date_ordinals(values):
    values = np.asarray(values)
    unique = np.unique(values)
    parsed = {}
    valid = True
    for value in unique:
        raw = value.item() if hasattr(value, "item") else value
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        text = str(raw)
        try:
            if "-" in text:
                day = datetime.date.fromisoformat(text[:10])
            else:
                text = str(int(float(text))).zfill(8)
                day = datetime.datetime.strptime(text, "%Y%m%d").date()
            parsed[value] = day.toordinal()
        except (ValueError, TypeError, OverflowError):
            valid = False
            break
    if not valid:
        sorted_unique = sorted(unique.tolist())
        parsed = {value: i for i, value in enumerate(sorted_unique)}
    return np.asarray([parsed[value] for value in values], dtype=np.float32)


def make_pairs(users, labels, seed):
    users = np.asarray(users)
    labels = np.asarray(labels) > 0.5
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    cuts = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
    starts = np.concatenate(([0], cuts))
    ends = np.concatenate((cuts, [len(order)]))
    rng = np.random.default_rng(seed)
    positives = []
    negatives = []
    for start, end in zip(starts, ends):
        rows = order[start:end]
        pos = rows[labels[rows]]
        neg = rows[~labels[rows]]
        if len(pos) and len(neg):
            positives.append(pos)
            negatives.append(rng.choice(neg, size=len(pos), replace=True))
    if not positives:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
    return (
        torch.from_numpy(np.concatenate(positives).astype(np.int64)),
        torch.from_numpy(np.concatenate(negatives).astype(np.int64)),
    )


def predict(model, X, device, batch_size=65536):
    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = X[start:start + batch_size].to(device, non_blocking=True)
            result.append(model(xb).detach().cpu().numpy())
    return np.concatenate(result).astype(np.float64, copy=False)


def rank_normalize(scores):
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(len(scores), dtype=np.float64)
    if len(scores) > 1:
        ranks /= float(len(scores) - 1)
    return ranks


def train_one(config, seed, epochs, checks_per_epoch, Xt, yt, recency_age,
              pair_pos, pair_neg, Xv, val_users, val_labels, total_dim, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = DCNLite(
        total_dim=total_dim,
        fields=Xt.shape[1],
        k=16,
        hidden=128,
        dropout=float(config["dropout"]),
        cross_layers=2,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"])
    )
    half_life = float(config["half_life"])
    weights_np = np.exp2(-recency_age / half_life).astype(np.float32)
    weights = torch.from_numpy(weights_np)
    n = len(yt)
    batch_size = 8192
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 104729)
    best_primary = -1.0
    best_scores = None
    best_metrics = None
    best_checkpoint = None
    curve = []
    global_step = 0
    parts = max(1, int(checks_per_epoch))
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(n, generator=generator)
        boundaries = np.linspace(0, n, parts + 1, dtype=np.int64)
        running_loss = 0.0
        running_batches = 0
        for part in range(parts):
            section = permutation[boundaries[part]:boundaries[part + 1]]
            for offset in range(0, len(section), batch_size):
                idx = section[offset:offset + batch_size]
                if len(idx) == 0:
                    continue
                xb = Xt[idx].to(device, non_blocking=True)
                yb = yt[idx].to(device, non_blocking=True)
                wb = weights[idx].to(device, non_blocking=True)
                pair_count = min(max(1, len(idx) // 2), len(pair_pos))
                if pair_count > 0:
                    chosen = torch.randint(len(pair_pos), (pair_count,), generator=generator)
                    pi = pair_pos[chosen]
                    ni = pair_neg[chosen]
                    xp = Xt[pi].to(device, non_blocking=True)
                    xn = Xt[ni].to(device, non_blocking=True)
                    all_x = torch.cat((xb, xp, xn), dim=0)
                    all_scores = model(all_x)
                    base_end = len(idx)
                    pos_end = base_end + pair_count
                    logits = all_scores[:base_end]
                    pos_scores = all_scores[base_end:pos_end]
                    neg_scores = all_scores[pos_end:]
                    point_loss = F.binary_cross_entropy_with_logits(logits, yb, reduction="none")
                    point_loss = (point_loss * wb).sum() / wb.sum().clamp_min(1e-6)
                    pair_weights = weights[pi].to(device, non_blocking=True)
                    pair_loss = F.softplus(-(pos_scores - neg_scores))
                    pair_loss = (pair_loss * pair_weights).sum() / pair_weights.sum().clamp_min(1e-6)
                    loss = 0.5 * point_loss + 0.5 * pair_loss
                else:
                    logits = model(xb)
                    point_loss = F.binary_cross_entropy_with_logits(logits, yb, reduction="none")
                    loss = (point_loss * wb).sum() / wb.sum().clamp_min(1e-6)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                running_loss += float(loss.detach().cpu())
                running_batches += 1
                global_step += 1
            scores = predict(model, Xv, device)
            metrics = metric_values(val_users, val_labels, scores)
            record = {
                "epoch": epoch + (part + 1) / parts,
                "train_loss": round(running_loss / max(1, running_batches), 6),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "val_gauc": round(metrics["gauc"], 6),
                "val_primary": round(metrics["primary"], 6),
            }
            curve.append(record)
            if metrics["primary"] > best_primary + 1e-8:
                best_primary = metrics["primary"]
                best_scores = scores.copy()
                best_metrics = metrics
                best_checkpoint = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
            model.train()
        if (epoch + 1) % int(config["step_size"]) == 0:
            new_lr = optimizer.param_groups[0]["lr"] * float(config["gamma"])
            for group in optimizer.param_groups:
                group["lr"] = new_lr
    del best_checkpoint
    del optimizer
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_scores, best_metrics, curve


def coarse_configs(seed, count):
    rng = np.random.default_rng(seed + 7717)
    configs = []
    half_lives = [3.5, 7.0, 14.0]
    gammas = [0.35, 0.5, 0.68, 0.82]
    steps = [1, 2, 3]
    for i in range(count):
        fraction = (i + 0.5) / count
        dropout = 0.15 + 0.25 * ((fraction + rng.random()) % 1.0)
        log_wd = math.log10(3e-5) + rng.random() * (math.log10(3e-3) - math.log10(3e-5))
        log_lr = math.log10(3.5e-4) + rng.random() * (math.log10(1.6e-3) - math.log10(3.5e-4))
        configs.append({
            "dropout": float(dropout),
            "weight_decay": float(10 ** log_wd),
            "lr": float(10 ** log_lr),
            "step_size": int(steps[i % len(steps)]),
            "gamma": float(gammas[(i // len(steps)) % len(gammas)]),
            "half_life": float(half_lives[(i * 2 + i // 3) % len(half_lives)]),
        })
    return configs


def refinement_configs(winner, seed, count):
    rng = np.random.default_rng(seed + 19001)
    half_grid = np.asarray([3.5, 5.0, 7.0, 10.0, 14.0])
    center_half = int(np.argmin(np.abs(half_grid - float(winner["half_life"]))))
    configs = [copy.deepcopy(winner)]
    drop_offsets = [-0.055, -0.03, -0.012, 0.012, 0.03, 0.055]
    wd_factors = [0.45, 0.65, 0.82, 1.2, 1.55, 2.2]
    lr_factors = [0.72, 0.86, 0.94, 1.06, 1.16, 1.32]
    gamma_offsets = [-0.12, -0.07, -0.03, 0.03, 0.07, 0.12]
    for i in range(1, count):
        j = (i - 1) % 6
        half_shift = [-1, 0, 1][(i - 1) % 3]
        half_index = min(len(half_grid) - 1, max(0, center_half + half_shift))
        step_shift = [-1, 0, 1][(i + 1) % 3]
        config = {
            "dropout": float(np.clip(winner["dropout"] + drop_offsets[j], 0.1, 0.48)),
            "weight_decay": float(np.clip(winner["weight_decay"] * wd_factors[(j + i // 6) % 6], 1e-5, 6e-3)),
            "lr": float(np.clip(winner["lr"] * lr_factors[(j * 5 + i // 6) % 6], 2e-4, 2.2e-3)),
            "step_size": int(np.clip(winner["step_size"] + step_shift, 1, 4)),
            "gamma": float(np.clip(winner["gamma"] + gamma_offsets[(j * 3 + i // 6) % 6], 0.25, 0.9)),
            "half_life": float(half_grid[half_index]),
        }
        if rng.random() < 0.25:
            config["dropout"] = float(np.clip(config["dropout"] + rng.normal(0, 0.008), 0.1, 0.48))
        configs.append(config)
    return configs


def append_progress(path, payload):
    with open(path, "a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


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

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_path = os.path.join(args.data_dir, "train.npz")
    val_path = os.path.join(args.data_dir, "val.npz")
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError("This package-dial sweep requires train.npz and val.npz")

    tr = np.load(train_path)
    va = np.load(val_path)
    Xt = torch.from_numpy(tr["X"].astype(np.int64, copy=False))
    yt_np = tr["y"].astype(np.float32, copy=False)
    yt = torch.from_numpy(yt_np)
    Xv = torch.from_numpy(va["X"].astype(np.int64, copy=False))
    val_users = va["user"]
    val_labels = va["y"].astype(np.int64, copy=False)
    total_dim = int(tr["field_dims"].sum())
    train_days = date_ordinals(tr["date"])
    recency_age = train_days.max() - train_days
    pair_pos, pair_neg = make_pairs(tr["user"], yt_np, args.seed + 31)

    smoke_value = os.environ.get("SMOKE_EPOCHS")
    smoke_cap = int(smoke_value) if smoke_value is not None else None
    coarse_epochs = 3 if smoke_cap is None else min(3, smoke_cap)
    refine_epochs = 5 if smoke_cap is None else min(5, smoke_cap)
    final_epochs = args.epochs if smoke_cap is None else min(args.epochs, smoke_cap)
    coarse_count = 24 if smoke_cap is None else 2
    refine_count = 12 if smoke_cap is None else 1
    final_seed_count = 5 if smoke_cap is None else 1

    history = []
    best_config = None
    best_probe_primary = -1.0
    coarse = coarse_configs(args.seed, coarse_count)
    for probe_id, config in enumerate(coarse):
        scores, metrics, curve = train_one(
            config, args.seed + 1000 + probe_id, coarse_epochs, 1,
            Xt, yt, recency_age, pair_pos, pair_neg, Xv,
            val_users, val_labels, total_dim, device,
        )
        entry = {
            "stage": "coarse",
            "probe": probe_id,
            "config": config,
            "metrics": metrics,
            "best_checkpoint": curve[-1] if len(curve) == 1 else max(curve, key=lambda x: x["val_primary"]),
        }
        history.append(entry)
        append_progress(progress_path, entry)
        if metrics["primary"] > best_probe_primary:
            best_probe_primary = metrics["primary"]
            best_config = copy.deepcopy(config)
        del scores

    refined = refinement_configs(best_config, args.seed, refine_count)
    refine_best_config = copy.deepcopy(best_config)
    refine_best_primary = -1.0
    for probe_id, config in enumerate(refined):
        scores, metrics, curve = train_one(
            config, args.seed + 3000 + probe_id, refine_epochs, 1,
            Xt, yt, recency_age, pair_pos, pair_neg, Xv,
            val_users, val_labels, total_dim, device,
        )
        entry = {
            "stage": "refine",
            "probe": probe_id,
            "config": config,
            "metrics": metrics,
            "best_checkpoint": max(curve, key=lambda x: x["val_primary"]),
        }
        history.append(entry)
        append_progress(progress_path, entry)
        if metrics["primary"] > refine_best_primary:
            refine_best_primary = metrics["primary"]
            refine_best_config = copy.deepcopy(config)
        del scores

    final_members = []
    final_seed_records = []
    for member in range(final_seed_count):
        member_seed = args.seed + member
        scores, metrics, curve = train_one(
            refine_best_config, member_seed, final_epochs, 2,
            Xt, yt, recency_age, pair_pos, pair_neg, Xv,
            val_users, val_labels, total_dim, device,
        )
        final_members.append(scores)
        record = {
            "stage": "final",
            "member": member,
            "seed": member_seed,
            "config": refine_best_config,
            "metrics": metrics,
            "curve": curve,
        }
        final_seed_records.append(record)
        append_progress(progress_path, {
            "stage": "final",
            "member": member,
            "seed": member_seed,
            "primary": metrics["primary"],
        })

    normalized = [rank_normalize(scores) for scores in final_members]
    prefix_history = []
    running = np.zeros(len(Xv), dtype=np.float64)
    selected_scores = None
    selected_metrics = None
    selected_count = 0
    for i, member_scores in enumerate(normalized):
        running += member_scores
        ensemble_scores = running / float(i + 1)
        metrics = metric_values(val_users, val_labels, ensemble_scores)
        prefix_history.append({"members": i + 1, "metrics": metrics})
        if selected_metrics is None or metrics["primary"] > selected_metrics["primary"]:
            selected_metrics = metrics
            selected_scores = ensemble_scores.copy()
            selected_count = i + 1

    output_metrics = {
        "gauc": selected_metrics["gauc"],
        "ndcg5": selected_metrics["ndcg5"],
        "primary": selected_metrics["primary"],
        "history": history,
        "winning_config": refine_best_config,
        "final_seeds": final_seed_records,
        "ensemble_prefixes": prefix_history,
        "selected_ensemble_members": selected_count,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as handle:
        json.dump(output_metrics, handle)

    field_dims = tr["field_dims"].astype(np.int64)
    video_offset = int(field_dims[0])
    video_ids = va["X"][:, 1].astype(np.int64) - video_offset
    with open(os.path.join(args.out_dir, "predictions.csv"), "w") as handle:
        handle.write("row_id,user_id,video_id,score\n")
        for row_id, score in enumerate(selected_scores):
            handle.write(f"{row_id},{val_users[row_id]},{video_ids[row_id]},{score:.8g}\n")


if __name__ == "__main__":
    main()
