"""Joint DCN-lite, hybrid-BPR, regularization, schedule, and recency dial search."""
import argparse
import copy
import datetime
import json
import math
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.official.evaluate import evaluate


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def metric_values(m):
    return {
        "gauc": float(m.get("GAUC", m.get("gauc", 0.0))),
        "ndcg5": float(m.get("nDCG@5", m.get("ndcg5", 0.0))),
        "primary": float(m["primary"]),
    }


def json_config(cfg):
    out = {}
    for key, value in cfg.items():
        if isinstance(value, (np.integer,)):
            value = int(value)
        elif isinstance(value, (np.floating,)):
            value = float(value)
        out[key] = value
    return out


def append_progress(path, record):
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")
        fh.flush()


class DCNHybrid(torch.nn.Module):
    def __init__(self, total_dim, n_fields, k, hidden, cross_layers, dropout):
        super().__init__()
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.emb_dropout = torch.nn.Dropout(dropout)
        flat_dim = n_fields * k
        self.cross_w = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.empty(flat_dim)) for _ in range(cross_layers)]
        )
        self.cross_b = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.zeros(flat_dim)) for _ in range(cross_layers)]
        )
        self.cross_out = torch.nn.Linear(flat_dim, 1, bias=False)
        hidden2 = max(32, hidden // 2)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(flat_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, hidden2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden2, 1),
        )
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)
        for w in self.cross_w:
            torch.nn.init.normal_(w, std=0.01)
        torch.nn.init.normal_(self.cross_out.weight, std=0.01)
        for layer in self.mlp:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)
        torch.nn.init.normal_(self.mlp[-1].weight, std=0.01)

    def forward(self, x):
        raw_e = self.emb(x)
        e = self.emb_dropout(raw_e)
        summed = e.sum(dim=1)
        fm = 0.5 * (summed.square() - e.square().sum(dim=1)).sum(dim=1)
        linear = self.lin(x).sum(dim=(1, 2))
        x0 = e.reshape(e.shape[0], -1)
        cross = x0
        for w, b in zip(self.cross_w, self.cross_b):
            scale = (cross * w).sum(dim=1, keepdim=True)
            cross = cross + x0 * scale + b
        cross_score = self.cross_out(cross).squeeze(1)
        mlp_score = self.mlp(x0).squeeze(1)
        return self.bias + linear + fm + cross_score + mlp_score


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
        s = e.sum(dim=1)
        pair = 0.5 * (s.square() - e.square().sum(dim=1)).sum(dim=1)
        return self.bias + self.lin(x).sum(dim=(1, 2)) + pair


def date_ages(date_values):
    values = np.asarray(date_values)
    unique = np.unique(values)
    ordinals = {}
    valid = True
    for value in unique:
        try:
            text = str(int(value)).zfill(8)
            ordinals[value] = datetime.datetime.strptime(text, "%Y%m%d").date().toordinal()
        except (ValueError, TypeError, OverflowError):
            valid = False
            break
    if not valid:
        ordered = sorted(unique.tolist())
        ordinals = {value: rank for rank, value in enumerate(ordered)}
    mapped = np.fromiter((ordinals[v] for v in values), dtype=np.float64, count=len(values))
    return mapped.max() - mapped


def build_pair_pool(users, labels, choices, seed):
    users = np.asarray(users)
    labels = np.asarray(labels)
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    boundaries = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(order)]))
    rng = np.random.RandomState(seed)
    pos_parts = []
    neg_parts = []
    for start, end in zip(starts, ends):
        group = order[start:end]
        pos = group[labels[group] >= 0.5]
        neg = group[labels[group] < 0.5]
        if len(pos) == 0 or len(neg) == 0:
            continue
        sampled = neg[rng.randint(0, len(neg), size=(len(pos), choices))]
        pos_parts.append(pos.astype(np.int64, copy=False))
        neg_parts.append(sampled.astype(np.int64, copy=False))
    if not pos_parts:
        raise RuntimeError("No within-user positive/negative pairs are available")
    return (
        torch.from_numpy(np.concatenate(pos_parts)),
        torch.from_numpy(np.concatenate(neg_parts, axis=0)),
    )


def predict(model, X, device, batch_size=32768):
    model.eval()
    pieces = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = X[start:start + batch_size].to(device, non_blocking=True)
            pieces.append(model(xb).detach().cpu().numpy())
    scores = np.concatenate(pieces).astype(np.float64, copy=False)
    if not np.all(np.isfinite(scores)):
        scores = np.nan_to_num(scores, nan=0.0, posinf=30.0, neginf=-30.0)
    return scores


def train_candidate(cfg, train_data, val_data, pair_pool, ages, device, epochs, seed,
                    keep_state=False):
    seed_all(seed)
    Xt, yt = train_data
    Xv, val_users, val_y = val_data
    pair_pos, pair_neg = pair_pool
    model = DCNHybrid(
        int(cfg["total_dim"]), int(cfg["n_fields"]), 16,
        int(cfg["hidden"]), int(cfg["cross_layers"]), float(cfg["dropout"])
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"])
    )
    recency = np.exp(-math.log(2.0) * ages / float(cfg["half_life"])).astype(np.float32)
    recency /= max(float(recency.mean()), 1e-8)
    recency_t = torch.from_numpy(recency)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 104729)
    n = len(yt)
    batch_size = 8192
    pair_batch = batch_size // 2
    best_primary = -1.0
    best_scores = None
    best_state = None
    best_metrics = None
    best_checkpoint = 0.0
    curve = []
    half_step = 0
    for epoch in range(epochs):
        permutation = torch.randperm(n, generator=generator)
        split = n // 2
        segments = ((0, split), (split, n))
        for half, (seg_start, seg_end) in enumerate(segments):
            model.train()
            loss_sum = 0.0
            examples = 0
            for start in range(seg_start, seg_end, batch_size):
                idx = permutation[start:min(start + batch_size, seg_end)]
                bsz = len(idx)
                pb = min(pair_batch, max(1, bsz // 2))
                rows = torch.randint(0, len(pair_pos), (pb,), generator=generator)
                choice = half_step % pair_neg.shape[1]
                pidx = pair_pos[rows]
                nidx = pair_neg[rows, choice]
                all_idx = torch.cat((idx, pidx, nidx), dim=0)
                xb = Xt[all_idx].to(device, non_blocking=True)
                logits = model(xb)
                main_logits = logits[:bsz]
                pos_logits = logits[bsz:bsz + pb]
                neg_logits = logits[bsz + pb:]
                target = yt[idx].to(device, non_blocking=True)
                weight = recency_t[idx].to(device, non_blocking=True)
                bce = torch.nn.functional.binary_cross_entropy_with_logits(
                    main_logits, target, reduction="none"
                )
                bce = (bce * weight).sum() / weight.sum().clamp_min(1e-8)
                pair_weight = 0.5 * (
                    recency_t[pidx].to(device, non_blocking=True)
                    + recency_t[nidx].to(device, non_blocking=True)
                )
                bpr_each = torch.nn.functional.softplus(-(pos_logits - neg_logits))
                bpr = (bpr_each * pair_weight).sum() / pair_weight.sum().clamp_min(1e-8)
                mix = float(cfg["bpr_mix"])
                loss = (1.0 - mix) * bce + mix * bpr
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                loss_sum += float(loss.detach().cpu()) * bsz
                examples += bsz
            half_step += 1
            scores = predict(model, Xv, device)
            metrics = metric_values(evaluate(val_users, val_y, scores))
            checkpoint = epoch + 0.5 * (half + 1)
            curve.append({
                "checkpoint": checkpoint,
                "train_loss": round(loss_sum / max(examples, 1), 6),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "val_gauc": round(metrics["gauc"], 6),
                "val_primary": round(metrics["primary"], 6),
            })
            if metrics["primary"] > best_primary + 1e-9:
                best_primary = metrics["primary"]
                best_scores = scores.copy()
                best_metrics = metrics
                best_checkpoint = checkpoint
                if keep_state:
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
            if half_step % int(cfg["step_halves"]) == 0:
                for group in optimizer.param_groups:
                    group["lr"] = max(1e-6, group["lr"] * float(cfg["lr_gamma"]))
    if keep_state and best_state is not None:
        model.load_state_dict(best_state)
    return {
        "primary": float(best_primary),
        "metrics": best_metrics,
        "scores": best_scores,
        "best_checkpoint": float(best_checkpoint),
        "curve": curve,
    }


def train_parent_reference(total_dim, Xt, yt, Xv, val_users, val_y, device, epochs, seed):
    seed_all(seed)
    model = ParentFM(total_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 17)
    best_primary = -1.0
    best_scores = None
    patience = 0
    n = len(yt)
    for _ in range(epochs):
        model.train()
        permutation = torch.randperm(n, generator=generator)
        for start in range(0, n, 8192):
            idx = permutation[start:start + 8192]
            xb = Xt[idx].to(device, non_blocking=True)
            target = yt[idx].to(device, non_blocking=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(xb), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scores = predict(model, Xv, device)
        primary = metric_values(evaluate(val_users, val_y, scores))["primary"]
        if primary > best_primary + 1e-6:
            best_primary = primary
            best_scores = scores.copy()
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break
    return best_scores, best_primary


def coarse_configs(count, total_dim, n_fields, seed):
    rng = np.random.RandomState(seed + 3109)
    configs = []
    half_lives = [3.5, 7.0, 14.0]
    hidden_values = [64, 96, 128, 160]
    for idx in range(count):
        configs.append({
            "id": "coarse_%02d" % idx,
            "total_dim": total_dim,
            "n_fields": n_fields,
            "dropout": float(rng.uniform(0.15, 0.40)),
            "weight_decay": float(np.exp(rng.uniform(np.log(3e-5), np.log(3e-3)))),
            "lr": float(np.exp(rng.uniform(np.log(4e-4), np.log(1.8e-3)))),
            "step_halves": int(rng.choice([2, 3, 4, 5, 6])),
            "lr_gamma": float(rng.uniform(0.25, 0.72)),
            "half_life": float(rng.choice(half_lives)),
            "bpr_mix": float(rng.uniform(0.38, 0.62)),
            "hidden": int(rng.choice(hidden_values)),
            "cross_layers": int(rng.choice([1, 2])),
        })
    return configs


def refined_configs(winner, count, seed):
    rng = np.random.RandomState(seed + 7919)
    configs = []
    base = copy.deepcopy(winner)
    base["id"] = "refine_00"
    configs.append(base)
    hidden_grid = np.arange(64, 193, 16)
    for idx in range(1, count):
        cfg = copy.deepcopy(winner)
        cfg["id"] = "refine_%02d" % idx
        cfg["dropout"] = float(np.clip(
            float(winner["dropout"]) + rng.normal(0.0, 0.035), 0.12, 0.44
        ))
        cfg["weight_decay"] = float(np.clip(
            float(winner["weight_decay"]) * np.exp(rng.normal(0.0, 0.32)), 2e-5, 5e-3
        ))
        cfg["lr"] = float(np.clip(
            float(winner["lr"]) * np.exp(rng.normal(0.0, 0.18)), 3e-4, 2.2e-3
        ))
        cfg["step_halves"] = int(np.clip(
            int(winner["step_halves"]) + rng.choice([-1, 0, 0, 1]), 2, 7
        ))
        cfg["lr_gamma"] = float(np.clip(
            float(winner["lr_gamma"]) + rng.normal(0.0, 0.065), 0.18, 0.82
        ))
        cfg["half_life"] = float(np.clip(
            float(winner["half_life"]) * np.exp(rng.normal(0.0, 0.20)), 2.75, 18.0
        ))
        cfg["bpr_mix"] = float(np.clip(
            float(winner["bpr_mix"]) + rng.normal(0.0, 0.045), 0.30, 0.70
        ))
        target_hidden = int(winner["hidden"]) + int(rng.choice([-32, -16, 0, 0, 16, 32]))
        cfg["hidden"] = int(hidden_grid[np.argmin(np.abs(hidden_grid - target_hidden))])
        cfg["cross_layers"] = int(
            winner["cross_layers"] if rng.rand() < 0.75 else 3 - int(winner["cross_layers"])
        )
        configs.append(cfg)
    return configs


def group_rank_average(users, score_vectors):
    users = np.asarray(users)
    n = len(users)
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    boundaries = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [n]))
    result = np.zeros(n, dtype=np.float64)
    for scores in score_vectors:
        ranks = np.empty(n, dtype=np.float64)
        for start, end in zip(starts, ends):
            idx = order[start:end]
            size = end - start
            if size == 1:
                ranks[idx[0]] = 0.5
            else:
                local_order = np.argsort(scores[idx], kind="mergesort")
                local_ranks = np.empty(size, dtype=np.float64)
                local_ranks[local_order] = np.arange(size, dtype=np.float64) / (size - 1)
                ranks[idx] = local_ranks
        result += ranks
    return result / len(score_vectors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    progress_path = os.path.join(args.out_dir, "progress.log")
    with open(progress_path, "w", encoding="utf-8"):
        pass

    seed_all(args.seed)
    try:
        torch.use_deterministic_algorithms(True)
    except (AttributeError, RuntimeError):
        pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_path = os.path.join(args.data_dir, "train.npz")
    val_path = os.path.join(args.data_dir, "val.npz")
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError("This search package requires the train.npz and val.npz fast path")
    tr = np.load(train_path)
    va = np.load(val_path)

    Xt = torch.from_numpy(tr["X"].astype(np.int64, copy=False))
    yt = torch.from_numpy(tr["y"].astype(np.float32, copy=False))
    Xv = torch.from_numpy(va["X"].astype(np.int64, copy=False))
    val_users = np.asarray(va["user"])
    val_y = va["y"].astype(int, copy=False)
    total_dim = int(np.asarray(tr["field_dims"]).sum())
    n_fields = int(Xt.shape[1])
    ages = date_ages(tr["date"])
    pair_pool = build_pair_pool(tr["user"], tr["y"], 4, args.seed + 53)

    smoke_value = os.environ.get("SMOKE_EPOCHS")
    smoke_cap = None
    if smoke_value is not None:
        smoke_cap = max(1, int(smoke_value))

    def cap_epochs(default):
        if smoke_cap is None:
            return default
        return min(default, smoke_cap)

    if smoke_cap is None:
        coarse_count = 44
        promote_count = 12
        refine_count = 10
        final_seed_count = 5
    else:
        coarse_count = 2
        promote_count = 1
        refine_count = 1
        final_seed_count = 2

    history = []
    initial_results = []
    configs = coarse_configs(coarse_count, total_dim, n_fields, args.seed)
    for cfg in configs:
        result = train_candidate(
            cfg, (Xt, yt), (Xv, val_users, val_y), pair_pool, ages, device,
            cap_epochs(2), args.seed + 100003, keep_state=False
        )
        entry = {
            "stage": "coarse",
            "config": json_config(cfg),
            "epochs": cap_epochs(2),
            "best_checkpoint": result["best_checkpoint"],
            **result["metrics"],
        }
        history.append(entry)
        append_progress(progress_path, entry)
        initial_results.append((result["primary"], cfg))

    initial_results.sort(key=lambda item: item[0], reverse=True)
    promoted_results = []
    for rank, (_, original_cfg) in enumerate(initial_results[:promote_count]):
        cfg = copy.deepcopy(original_cfg)
        cfg["id"] = "promoted_%02d_from_%s" % (rank, original_cfg["id"])
        result = train_candidate(
            cfg, (Xt, yt), (Xv, val_users, val_y), pair_pool, ages, device,
            cap_epochs(3), args.seed + 100003, keep_state=False
        )
        entry = {
            "stage": "coarse_successive_halving",
            "config": json_config(cfg),
            "epochs": cap_epochs(3),
            "best_checkpoint": result["best_checkpoint"],
            **result["metrics"],
        }
        history.append(entry)
        append_progress(progress_path, entry)
        promoted_results.append((result["primary"], cfg))

    promoted_results.sort(key=lambda item: item[0], reverse=True)
    stage1_winner = promoted_results[0][1]
    refinement = refined_configs(stage1_winner, refine_count, args.seed)
    refined_results = []
    for cfg in refinement:
        result = train_candidate(
            cfg, (Xt, yt), (Xv, val_users, val_y), pair_pool, ages, device,
            cap_epochs(5), args.seed + 200003, keep_state=False
        )
        entry = {
            "stage": "refine",
            "config": json_config(cfg),
            "epochs": cap_epochs(5),
            "best_checkpoint": result["best_checkpoint"],
            **result["metrics"],
        }
        history.append(entry)
        append_progress(progress_path, entry)
        refined_results.append((result["primary"], cfg))

    refined_results.sort(key=lambda item: item[0], reverse=True)
    final_cfg = copy.deepcopy(refined_results[0][1])
    final_cfg["id"] = "final_winner"
    final_epochs = cap_epochs(max(1, args.epochs))

    member_scores = []
    member_records = []
    for member_index in range(final_seed_count):
        member_seed = args.seed + member_index
        result = train_candidate(
            final_cfg, (Xt, yt), (Xv, val_users, val_y), pair_pool, ages, device,
            final_epochs, member_seed, keep_state=True
        )
        record = {
            "stage": "final_member",
            "member": member_index,
            "seed": member_seed,
            "config": json_config(final_cfg),
            "epochs": final_epochs,
            "best_checkpoint": result["best_checkpoint"],
            **result["metrics"],
            "curve": result["curve"],
        }
        member_records.append(record)
        history.append(record)
        append_progress(progress_path, record)
        member_scores.append(result["scores"])

    for left in range(len(member_scores)):
        for right in range(left + 1, len(member_scores)):
            assert not np.allclose(member_scores[left], member_scores[right], rtol=1e-7, atol=1e-8)

    parent_scores, parent_primary = train_parent_reference(
        total_dim, Xt, yt, Xv, val_users, val_y, device, cap_epochs(12), args.seed
    )
    for scores in member_scores:
        assert not np.allclose(scores, parent_scores, rtol=1e-7, atol=1e-8)
    parent_record = {
        "stage": "parent_noop_check",
        "seed": args.seed,
        "primary": float(parent_primary),
    }
    history.append(parent_record)
    append_progress(progress_path, parent_record)

    final_scores = group_rank_average(val_users, member_scores)
    final_metrics = metric_values(evaluate(val_users, val_y, final_scores))

    metrics_payload = {
        "gauc": final_metrics["gauc"],
        "ndcg5": final_metrics["ndcg5"],
        "primary": final_metrics["primary"],
        "selected_config": json_config(final_cfg),
        "member_primaries": [float(record["primary"]) for record in member_records],
        "parent_reference_primary": float(parent_primary),
        "history": history,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics_payload, fh)

    field_dims = np.asarray(tr["field_dims"], dtype=np.int64)
    video_ids = va["X"][:, 1].astype(np.int64, copy=False) - int(field_dims[0])
    with open(os.path.join(args.out_dir, "predictions.csv"), "w", encoding="utf-8") as fh:
        fh.write("row_id,user_id,video_id,score\n")
        for row_id, score in enumerate(final_scores):
            fh.write(f"{row_id},{val_users[row_id]},{video_ids[row_id]},{score:.8g}\n")


if __name__ == "__main__":
    main()
