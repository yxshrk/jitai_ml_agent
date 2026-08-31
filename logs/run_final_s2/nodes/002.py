"""Two-stage dial search for a regularized DCN-lite hybrid BCE/BPR package."""
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
    def __init__(self, total_dim, fields=5, k=16, dropout=0.2):
        super().__init__()
        width = fields * k
        self.emb = torch.nn.Embedding(total_dim, k)
        self.emb_dropout = torch.nn.Dropout(dropout)
        self.cross_w = torch.nn.Parameter(torch.empty(2, width))
        self.cross_b = torch.nn.Parameter(torch.zeros(2, width))
        self.deep = torch.nn.Sequential(
            torch.nn.Linear(width, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.cross_out = torch.nn.Linear(width, 1, bias=False)
        self.deep_out = torch.nn.Linear(64, 1, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.normal_(self.cross_w, std=0.01)
        for module in self.deep:
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                torch.nn.init.zeros_(module.bias)
        torch.nn.init.xavier_uniform_(self.cross_out.weight)
        torch.nn.init.xavier_uniform_(self.deep_out.weight)

    def forward(self, x):
        x0 = self.emb_dropout(self.emb(x)).flatten(1)
        cross = x0
        for layer in range(2):
            scale = (cross * self.cross_w[layer]).sum(1, keepdim=True)
            cross = cross + x0 * scale + self.cross_b[layer]
        deep = self.deep(x0)
        return self.cross_out(cross).squeeze(1) + self.deep_out(deep).squeeze(1) + self.bias


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def date_to_day(value):
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) >= 8:
        try:
            day = datetime.datetime.strptime(digits[:8], "%Y%m%d").date()
            return day.toordinal()
        except ValueError:
            pass
    try:
        return int(float(text))
    except ValueError:
        return 0


def load_csv_data(data_dir):
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    train_rows = []
    with open(train_path, newline="") as fh:
        reader = csv.DictReader(fh)
        has_author = "author_id" in (reader.fieldnames or [])
        for row in reader:
            train_rows.append({
                "user": row["user_id"],
                "video": row["video_id"],
                "author": row["author_id"] if has_author else "__NO_AUTHOR__",
                "tab": row["tab"],
                "duration": float(row["duration_ms"]),
                "date": date_to_day(row["date"]),
                "y": float(row["long_view"]),
            })
    val_rows = []
    with open(val_path, newline="") as fh:
        reader = csv.DictReader(fh)
        has_author = "author_id" in (reader.fieldnames or [])
        for row in reader:
            val_rows.append({
                "user": row["user_id"],
                "video": row["video_id"],
                "author": row["author_id"] if has_author else "__NO_AUTHOR__",
                "tab": row["tab"],
                "duration": float(row["duration_ms"]),
                "y": float(row["long_view"]),
            })
    durations = np.asarray([r["duration"] for r in train_rows], dtype=np.float64)
    quantiles = np.quantile(durations, np.arange(1, 10) / 10.0)
    fields = ("user", "video", "author", "tab")
    mappings = {}
    dimensions = []
    for field in fields:
        values = sorted({r[field] for r in train_rows})
        mappings[field] = {value: i + 1 for i, value in enumerate(values)}
        dimensions.append(len(values) + 1)
    dimensions.append(10)
    field_dims = np.asarray(dimensions, dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1]))).astype(np.int64)

    def encode(rows):
        x = np.zeros((len(rows), 5), dtype=np.int64)
        for i, row in enumerate(rows):
            for j, field in enumerate(fields):
                x[i, j] = mappings[field].get(row[field], 0) + offsets[j]
            bucket = int(np.searchsorted(quantiles, row["duration"], side="right"))
            x[i, 4] = min(bucket, 9) + offsets[4]
        return x

    xt = encode(train_rows)
    xv = encode(val_rows)
    return {
        "Xt": xt,
        "yt": np.asarray([r["y"] for r in train_rows], dtype=np.float32),
        "train_user": xt[:, 0].copy(),
        "train_day": np.asarray([r["date"] for r in train_rows], dtype=np.int64),
        "Xv": xv,
        "yv": np.asarray([r["y"] for r in val_rows], dtype=np.int64),
        "val_user": np.asarray([r["user"] for r in val_rows]),
        "val_video": np.asarray([r["video"] for r in val_rows]),
        "field_dims": field_dims,
        "fast": False,
    }


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        tr = np.load(train_npz)
        va = np.load(val_npz)
        field_dims = tr["field_dims"].astype(np.int64)
        xt = tr["X"].astype(np.int64)
        xv = va["X"].astype(np.int64)
        raw_dates = np.asarray(tr["date"])
        train_day = np.asarray([date_to_day(v) for v in raw_dates], dtype=np.int64)
        return {
            "Xt": xt,
            "yt": tr["y"].astype(np.float32),
            "train_user": xt[:, 0].astype(np.int64),
            "train_day": train_day,
            "Xv": xv,
            "yv": va["y"].astype(np.int64),
            "val_user": np.asarray(va["user"]),
            "val_video": xv[:, 1] - int(field_dims[0]),
            "field_dims": field_dims,
            "fast": True,
        }
    return load_csv_data(data_dir)


def get_evaluator(fast):
    if fast:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    return evaluate


def metric_values(metrics):
    return (
        float(metrics.get("GAUC", metrics.get("gauc"))),
        float(metrics.get("nDCG@5", metrics.get("ndcg5"))),
        float(metrics["primary"]),
    )


def make_user_groups(users):
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    boundaries = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(order)]))
    return order, starts, ends


def make_pairs(labels, order, starts, ends, rng):
    positives = []
    negatives = []
    for start, end in zip(starts, ends):
        indices = order[start:end]
        group_y = labels[indices]
        pos = indices[group_y > 0.5]
        neg = indices[group_y <= 0.5]
        if len(pos) and len(neg):
            positives.append(pos)
            negatives.append(rng.choice(neg, size=len(pos), replace=True))
    if not positives:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    pos = np.concatenate(positives).astype(np.int64, copy=False)
    neg = np.concatenate(negatives).astype(np.int64, copy=False)
    permutation = rng.permutation(len(pos))
    return pos[permutation], neg[permutation]


def recency_weights(days, half_life):
    days = np.asarray(days, dtype=np.float64)
    maximum = float(days.max()) if len(days) else 0.0
    weights = np.exp2(-(maximum - days) / float(half_life))
    mean = float(weights.mean())
    if not np.isfinite(mean) or mean <= 0.0:
        return np.ones(len(days), dtype=np.float32)
    return (weights / mean).astype(np.float32)


def predict(model, xv, device):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(xv), 65536):
            xb = xv[start:start + 65536].to(device, non_blocking=True)
            outputs.append(model(xb).detach().cpu().numpy())
    return np.concatenate(outputs).astype(np.float64)


def train_one(data, evaluate, device, config, seed, epochs, half_checkpoints=False):
    seed_everything(seed)
    xt = torch.from_numpy(data["Xt"])
    yt = torch.from_numpy(data["yt"])
    xv = torch.from_numpy(data["Xv"])
    weights = torch.from_numpy(recency_weights(data["train_day"], config["half_life"]))
    model = DCNLite(
        int(data["field_dims"].sum()), fields=data["Xt"].shape[1],
        k=16, dropout=config["dropout"]
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config["decay_step"], gamma=config["decay_gamma"]
    )
    order, starts, ends = make_user_groups(data["train_user"])
    rng = np.random.RandomState(seed + 7919)
    batch_size = 8192 if device.type == "cuda" else 4096
    best_primary = -1.0
    best_scores = None
    best_checkpoint = 0.0
    checkpoint_history = []
    n = len(xt)

    for epoch in range(epochs):
        model.train()
        permutation = rng.permutation(n)
        pos_indices, neg_indices = make_pairs(data["yt"], order, starts, ends, rng)
        pair_pointer = 0
        phases = 2 if half_checkpoints else 1
        phase_edges = np.linspace(0, n, phases + 1, dtype=np.int64)
        last_loss = 0.0
        for phase in range(phases):
            left = int(phase_edges[phase])
            right = int(phase_edges[phase + 1])
            for batch_start in range(left, right, batch_size):
                idx_np = permutation[batch_start:min(batch_start + batch_size, right)]
                idx = torch.from_numpy(idx_np)
                xb = xt[idx].to(device, non_blocking=True)
                yb = yt[idx].to(device, non_blocking=True)
                wb = weights[idx].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                bce_each = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, yb, reduction="none"
                )
                bce_loss = (bce_each * wb).sum() / wb.sum().clamp_min(1.0)
                if len(pos_indices):
                    pair_count = max(1, len(idx_np) // 2)
                    if pair_pointer + pair_count > len(pos_indices):
                        pair_perm = rng.permutation(len(pos_indices))
                        pos_indices = pos_indices[pair_perm]
                        neg_indices = neg_indices[pair_perm]
                        pair_pointer = 0
                    pair_count = min(pair_count, len(pos_indices) - pair_pointer)
                    p_np = pos_indices[pair_pointer:pair_pointer + pair_count]
                    q_np = neg_indices[pair_pointer:pair_pointer + pair_count]
                    pair_pointer += pair_count
                    p = torch.from_numpy(p_np)
                    q = torch.from_numpy(q_np)
                    pos_logits = model(xt[p].to(device, non_blocking=True))
                    neg_logits = model(xt[q].to(device, non_blocking=True))
                    pair_w = weights[p].to(device, non_blocking=True)
                    bpr_each = torch.nn.functional.softplus(-(pos_logits - neg_logits))
                    bpr_loss = (bpr_each * pair_w).sum() / pair_w.sum().clamp_min(1.0)
                else:
                    bpr_loss = bce_loss.new_zeros(())
                loss = 0.5 * bce_loss + 0.5 * bpr_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                last_loss = float(loss.detach().cpu())
            scores = predict(model, xv, device)
            metrics = evaluate(data["val_user"], data["yv"], scores)
            gauc, ndcg5, primary = metric_values(metrics)
            checkpoint = epoch + float(phase + 1) / phases
            checkpoint_history.append({
                "checkpoint": checkpoint,
                "train_loss": round(last_loss, 7),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "val_gauc": round(gauc, 9),
                "val_ndcg5": round(ndcg5, 9),
                "val_primary": round(primary, 9),
            })
            if primary > best_primary + 1e-9:
                best_primary = primary
                best_scores = scores.copy()
                best_checkpoint = checkpoint
            model.train()
        scheduler.step()

    final_metrics = evaluate(data["val_user"], data["yv"], best_scores)
    gauc, ndcg5, primary = metric_values(final_metrics)
    result = {
        "seed": int(seed),
        "config": dict(config),
        "gauc": gauc,
        "ndcg5": ndcg5,
        "primary": primary,
        "best_checkpoint": best_checkpoint,
        "checkpoints": checkpoint_history,
    }
    return result, best_scores


def coarse_configs(seed):
    rng = np.random.RandomState(seed + 17011)
    configs = []
    dropout_levels = np.linspace(0.15, 0.40, 12)
    rng.shuffle(dropout_levels)
    half_lives = np.asarray([3.5, 5.0, 7.0, 9.5, 14.0, 3.5, 5.0, 7.0, 9.5, 14.0, 7.0, 14.0])
    rng.shuffle(half_lives)
    gammas = np.asarray([0.35, 0.45, 0.55, 0.65] * 3, dtype=np.float64)
    rng.shuffle(gammas)
    steps = np.asarray([1, 1, 2, 2] * 3, dtype=np.int64)
    rng.shuffle(steps)
    for i in range(12):
        configs.append({
            "dropout": float(dropout_levels[i]),
            "weight_decay": float(10.0 ** rng.uniform(math.log10(3e-5), math.log10(3e-3))),
            "lr": float(10.0 ** rng.uniform(math.log10(4e-4), math.log10(1.6e-3))),
            "decay_gamma": float(gammas[i]),
            "decay_step": int(steps[i]),
            "half_life": float(half_lives[i]),
        })
    return configs


def refine_configs(winner):
    patterns = [
        (0.82, 0.45, 0.85, 0.85, 0.80),
        (0.90, 0.70, 0.92, 1.00, 1.00),
        (0.96, 1.00, 1.00, 0.90, 1.20),
        (1.00, 0.72, 1.08, 1.10, 0.80),
        (1.04, 1.35, 0.92, 1.00, 1.00),
        (1.10, 1.80, 1.00, 0.90, 1.20),
        (1.16, 1.00, 1.08, 1.10, 1.00),
        (1.00, 2.20, 0.85, 0.82, 1.35),
    ]
    configs = []
    for drop_m, wd_m, lr_m, gamma_m, half_m in patterns:
        configs.append({
            "dropout": float(np.clip(winner["dropout"] * drop_m, 0.12, 0.45)),
            "weight_decay": float(np.clip(winner["weight_decay"] * wd_m, 2e-5, 5e-3)),
            "lr": float(np.clip(winner["lr"] * lr_m, 3e-4, 2e-3)),
            "decay_gamma": float(np.clip(winner["decay_gamma"] * gamma_m, 0.28, 0.78)),
            "decay_step": int(winner["decay_step"]),
            "half_life": float(np.clip(winner["half_life"] * half_m, 2.5, 18.0)),
        })
    return configs


def within_user_rank_scores(users, scores):
    users = np.asarray(users)
    scores = np.asarray(scores)
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    boundaries = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(order)]))
    ranked = np.zeros(len(scores), dtype=np.float64)
    for start, end in zip(starts, ends):
        indices = order[start:end]
        local_order = np.argsort(scores[indices], kind="stable")
        if len(indices) == 1:
            ranked[indices] = 0.5
        else:
            values = np.empty(len(indices), dtype=np.float64)
            values[local_order] = np.arange(len(indices), dtype=np.float64) / (len(indices) - 1)
            ranked[indices] = values
    return ranked


def summarize_config(config, results):
    values = np.asarray([result["primary"] for result in results], dtype=np.float64)
    return {
        "config": dict(config),
        "mean_primary": float(values.mean()),
        "std_primary": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "runs": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=14)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    smoke_text = os.environ.get("SMOKE_EPOCHS")
    smoke = smoke_text is not None
    smoke_cap = max(1, int(smoke_text)) if smoke else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_data(args.data_dir)
    evaluate = get_evaluator(data["fast"])

    coarse_epoch_count = 3
    refine_epoch_count = 6
    final_epoch_count = args.epochs
    if smoke:
        coarse_epoch_count = min(coarse_epoch_count, smoke_cap)
        refine_epoch_count = min(refine_epoch_count, smoke_cap)
        final_epoch_count = min(final_epoch_count, smoke_cap)

    coarse = coarse_configs(args.seed)
    if smoke:
        coarse = coarse[:1]
    coarse_repeats = 1 if smoke else 3
    refine_repeats = 1 if smoke else 2
    final_seed_count = 1 if smoke else 5
    progress_path = os.path.join(args.out_dir, "progress.log")
    history = {"coarse": [], "refine": [], "final": []}

    with open(progress_path, "w") as progress:
        for config_index, config in enumerate(coarse):
            runs = []
            for repeat in range(coarse_repeats):
                run_seed = args.seed + 1009 * repeat
                result, _ = train_one(
                    data, evaluate, device, config, run_seed,
                    coarse_epoch_count, half_checkpoints=False
                )
                result["stage"] = "coarse"
                result["config_index"] = config_index
                runs.append(result)
                progress.write(json.dumps({
                    "stage": "coarse", "config_index": config_index,
                    "seed": run_seed, "config": config,
                    "primary": result["primary"]
                }, sort_keys=True) + "\n")
                progress.flush()
            history["coarse"].append(summarize_config(config, runs))

        coarse_winner = max(history["coarse"], key=lambda item: item["mean_primary"])
        refined = refine_configs(coarse_winner["config"])
        if smoke:
            refined = refined[:1]
        for config_index, config in enumerate(refined):
            runs = []
            for repeat in range(refine_repeats):
                run_seed = args.seed + 2003 * repeat
                result, _ = train_one(
                    data, evaluate, device, config, run_seed,
                    refine_epoch_count, half_checkpoints=False
                )
                result["stage"] = "refine"
                result["config_index"] = config_index
                runs.append(result)
                progress.write(json.dumps({
                    "stage": "refine", "config_index": config_index,
                    "seed": run_seed, "config": config,
                    "primary": result["primary"]
                }, sort_keys=True) + "\n")
                progress.flush()
            history["refine"].append(summarize_config(config, runs))

        refine_winner = max(history["refine"], key=lambda item: item["mean_primary"])
        winning_config = refine_winner["config"]
        final_score_sets = []
        for repeat in range(final_seed_count):
            run_seed = args.seed + repeat
            result, scores = train_one(
                data, evaluate, device, winning_config, run_seed,
                final_epoch_count, half_checkpoints=True
            )
            result["stage"] = "final"
            history["final"].append(result)
            final_score_sets.append(scores)
            progress.write(json.dumps({
                "stage": "final", "seed": run_seed,
                "config": winning_config, "primary": result["primary"],
                "best_checkpoint": result["best_checkpoint"]
            }, sort_keys=True) + "\n")
            progress.flush()

    candidates = []
    for index, scores in enumerate(final_score_sets):
        metrics = evaluate(data["val_user"], data["yv"], scores)
        gauc, ndcg5, primary = metric_values(metrics)
        candidates.append((primary, gauc, ndcg5, scores, "single_seed_%d" % index))
    if len(final_score_sets) > 1:
        rank_sets = [within_user_rank_scores(data["val_user"], scores) for scores in final_score_sets]
        ensemble_scores = np.mean(np.stack(rank_sets, axis=0), axis=0)
        metrics = evaluate(data["val_user"], data["yv"], ensemble_scores)
        gauc, ndcg5, primary = metric_values(metrics)
        candidates.append((primary, gauc, ndcg5, ensemble_scores, "five_seed_rank_average"))

    primary, gauc, ndcg5, final_scores, selection = max(candidates, key=lambda item: item[0])
    candidate_metrics = [
        {"name": item[4], "primary": item[0], "gauc": item[1], "ndcg5": item[2]}
        for item in candidates
    ]
    metrics_output = {
        "gauc": float(gauc),
        "ndcg5": float(ndcg5),
        "primary": float(primary),
        "selected_prediction": selection,
        "winning_config": winning_config,
        "coarse_winner_mean_primary": coarse_winner["mean_primary"],
        "refine_winner_mean_primary": refine_winner["mean_primary"],
        "candidate_metrics": candidate_metrics,
        "history": history,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics_output, fh)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w") as fh:
        fh.write("row_id,user_id,video_id,score\n")
        for i, score in enumerate(final_scores):
            fh.write(f"{i},{data['val_user'][i]},{data['val_video'][i]},{score:.9g}\n")


if __name__ == "__main__":
    main()
