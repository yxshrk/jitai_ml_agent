import argparse
import csv
import json
import math
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DCNLite(torch.nn.Module):
    def __init__(self, total_dim, fields=5, k=16, hidden=128, dropout=0.25):
        super().__init__()
        width = fields * k
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.emb_dropout = torch.nn.Dropout(dropout)
        self.cross_w = torch.nn.Parameter(torch.empty(width))
        self.cross_b = torch.nn.Parameter(torch.zeros(width))
        self.cross_out = torch.nn.Linear(width, 1)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(width, hidden),
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
        torch.nn.init.zeros_(self.cross_out.weight)
        torch.nn.init.zeros_(self.cross_out.bias)
        torch.nn.init.zeros_(self.mlp[-1].weight)
        torch.nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x):
        e = self.emb_dropout(self.emb(x))
        summed = e.sum(dim=1)
        fm = 0.5 * (summed.square() - e.square().sum(dim=1)).sum(dim=1)
        linear = self.lin(x).sum(dim=(1, 2))
        x0 = e.flatten(start_dim=1)
        cross = x0 * (x0 * self.cross_w).sum(dim=1, keepdim=True) + self.cross_b + x0
        return self.bias + linear + fm + self.cross_out(cross).squeeze(1) + self.mlp(x0).squeeze(1)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def metric_dict(evaluator, users, labels, scores):
    raw = evaluator(users, labels, scores)
    return {
        "gauc": float(raw.get("GAUC", raw.get("gauc"))),
        "ndcg5": float(raw.get("nDCG@5", raw.get("ndcg5"))),
        "primary": float(raw["primary"]),
    }


def parse_day(value):
    text = str(value)
    try:
        number = int(float(text))
    except ValueError:
        return 0
    text = str(number)
    if len(text) == 8:
        year = number // 10000
        month = (number // 100) % 100
        day = number % 100
        return year * 372 + month * 31 + day
    return number


def recency_weights(values, half_life):
    days = np.asarray([parse_day(v) for v in values], dtype=np.float32)
    latest = float(days.max()) if len(days) else 0.0
    weights = np.exp2(-(latest - days) / float(half_life)).astype(np.float32)
    weights /= max(float(weights.mean()), 1e-8)
    return weights


def build_user_pairs(users, labels, seed):
    users = np.asarray(users)
    labels = np.asarray(labels) > 0.5
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    rng = np.random.RandomState(seed)
    positives = []
    negatives = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        group = order[left:right]
        pos = group[labels[group]]
        neg = group[~labels[group]]
        if len(pos) and len(neg):
            positives.append(pos.astype(np.int64, copy=False))
            negatives.append(rng.choice(neg, size=len(pos), replace=True).astype(np.int64, copy=False))
    if not positives:
        raise RuntimeError("No users have both positive and negative impressions")
    return np.concatenate(positives), np.concatenate(negatives)


def predict(model, x, device, batch_size):
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start:start + batch_size]).to(device)
            outputs.append(model(xb).cpu().numpy())
    return np.concatenate(outputs).astype(np.float64, copy=False)


def clean_config(config):
    return {
        "dropout": round(float(config["dropout"]), 6),
        "weight_decay": float(config["weight_decay"]),
        "lr": float(config["lr"]),
        "half_life": round(float(config["half_life"]), 6),
        "step_interval": int(config["step_interval"]),
        "step_gamma": round(float(config["step_gamma"]), 6),
    }


def append_progress(path, stage, index, config, primary):
    record = {
        "stage": stage,
        "probe": int(index),
        "config": clean_config(config),
        "primary": round(float(primary), 8),
    }
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def stratified_probe_indices(users, limit, seed):
    n = len(users)
    if n <= limit:
        return np.arange(n, dtype=np.int64)
    rng = np.random.RandomState(seed)
    unique = np.unique(users)
    rng.shuffle(unique)
    selected = []
    count = 0
    for user in unique:
        rows = np.flatnonzero(users == user)
        selected.append(rows)
        count += len(rows)
        if count >= limit:
            break
    result = np.concatenate(selected).astype(np.int64, copy=False)
    if len(result) > limit * 2:
        result = np.sort(rng.choice(result, size=limit * 2, replace=False))
    return result


def train_one(config, seed, epochs, data, evaluator, device, train_indices, eval_indices, checkpoint_each_half):
    seed_everything(seed)
    rng = np.random.RandomState(seed)
    model = DCNLite(
        data["total_dim"], fields=data["fields"], k=16, hidden=128,
        dropout=float(config["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"])
    )
    weights = recency_weights(data["date"], float(config["half_life"]))
    pair_pos = data["pair_pos"]
    pair_neg = data["pair_neg"]
    batch_size = 32768 if device.type == "cuda" else 8192
    pred_batch = 131072 if device.type == "cuda" else 32768
    best_primary = -float("inf")
    best_scores = None
    checkpoints = []
    half_step = 0
    last_loss = 0.0

    for epoch in range(epochs):
        permutation = train_indices[rng.permutation(len(train_indices))]
        split = (len(permutation) + 1) // 2
        halves = (permutation[:split], permutation[split:])
        for half_number, half_indices in enumerate(halves):
            model.train()
            loss_total = 0.0
            steps = 0
            for start in range(0, len(half_indices), batch_size):
                batch_np = half_indices[start:start + batch_size]
                if len(batch_np) == 0:
                    continue
                pair_count = max(1, len(batch_np) // 4)
                choices = rng.randint(0, len(pair_pos), size=pair_count)
                pos_np = pair_pos[choices]
                neg_np = pair_neg[choices]
                combined = np.concatenate((batch_np, pos_np, neg_np)).astype(np.int64, copy=False)
                xb = torch.from_numpy(data["X"][combined]).to(device)
                logits_all = model(xb)
                b = len(batch_np)
                p = len(pos_np)
                logits = logits_all[:b]
                pos_scores = logits_all[b:b + p]
                neg_scores = logits_all[b + p:]
                target = torch.from_numpy(data["y"][batch_np]).to(device)
                bw = torch.from_numpy(weights[batch_np]).to(device)
                pw = torch.from_numpy(0.5 * (weights[pos_np] + weights[neg_np])).to(device)
                bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
                bce_loss = (bce * bw).sum() / bw.sum().clamp_min(1e-8)
                pair = torch.nn.functional.softplus(-(pos_scores - neg_scores))
                pair_loss = (pair * pw).sum() / pw.sum().clamp_min(1e-8)
                loss = 0.5 * bce_loss + 0.5 * pair_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                loss_total += float(loss.detach().cpu())
                steps += 1
            last_loss = loss_total / max(steps, 1)
            half_step += 1
            if half_step % int(config["step_interval"]) == 0:
                for group in optimizer.param_groups:
                    group["lr"] *= float(config["step_gamma"])
            if checkpoint_each_half:
                scores = predict(model, data["val_X"][eval_indices], device, pred_batch)
                metrics = metric_dict(evaluator, data["val_user"][eval_indices], data["val_y"][eval_indices], scores)
                checkpoints.append({
                    "epoch": float(epoch) + 0.5 * float(half_number + 1),
                    "train_loss": round(last_loss, 6),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "gauc": round(metrics["gauc"], 6),
                    "ndcg5": round(metrics["ndcg5"], 6),
                    "primary": round(metrics["primary"], 6),
                })
                if metrics["primary"] > best_primary:
                    best_primary = metrics["primary"]
                    best_scores = scores.copy()

    if not checkpoint_each_half:
        scores = predict(model, data["val_X"][eval_indices], device, pred_batch)
        metrics = metric_dict(evaluator, data["val_user"][eval_indices], data["val_y"][eval_indices], scores)
        best_primary = metrics["primary"]
        best_scores = scores
        checkpoints.append({
            "epoch": float(epochs),
            "train_loss": round(last_loss, 6),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "gauc": round(metrics["gauc"], 6),
            "ndcg5": round(metrics["ndcg5"], 6),
            "primary": round(metrics["primary"], 6),
        })

    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return float(best_primary), best_scores, checkpoints


def encode_column(train_values, val_values):
    mapping = {value: i + 1 for i, value in enumerate(sorted(set(train_values)))}
    train_encoded = np.asarray([mapping[v] for v in train_values], dtype=np.int64)
    val_encoded = np.asarray([mapping.get(v, 0) for v in val_values], dtype=np.int64)
    return train_encoded, val_encoded, len(mapping) + 1


def load_csv_data(data_dir):
    def read(path, training):
        rows = []
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                item = {
                    "user": row["user_id"],
                    "video": row["video_id"],
                    "tab": row["tab"],
                    "duration": float(row["duration_ms"]),
                    "date": row["date"],
                    "y": float(row["long_view"]),
                }
                rows.append(item)
        return rows

    train_rows = read(os.path.join(data_dir, "train.csv"), True)
    val_rows = read(os.path.join(data_dir, "val.csv"), False)
    tr_user_raw = [r["user"] for r in train_rows]
    va_user_raw = [r["user"] for r in val_rows]
    tr_video_raw = [r["video"] for r in train_rows]
    va_video_raw = [r["video"] for r in val_rows]
    tr_tab_raw = [r["tab"] for r in train_rows]
    va_tab_raw = [r["tab"] for r in val_rows]
    tr_user, va_user_enc, du = encode_column(tr_user_raw, va_user_raw)
    tr_video, va_video, dv = encode_column(tr_video_raw, va_video_raw)
    tr_tab, va_tab, dt = encode_column(tr_tab_raw, va_tab_raw)
    durations = np.asarray([r["duration"] for r in train_rows], dtype=np.float64)
    edges = np.unique(np.quantile(durations, np.linspace(0.1, 0.9, 9)))
    tr_dur = np.searchsorted(edges, durations, side="right").astype(np.int64)
    va_dur = np.searchsorted(edges, np.asarray([r["duration"] for r in val_rows]), side="right").astype(np.int64)
    dims = np.asarray([du, dv, 1, dt, len(edges) + 1], dtype=np.int64)
    offsets = np.r_[0, np.cumsum(dims[:-1])]
    train_x = np.column_stack((tr_user, tr_video, np.zeros(len(train_rows), dtype=np.int64), tr_tab, tr_dur)) + offsets
    val_x = np.column_stack((va_user_enc, va_video, np.zeros(len(val_rows), dtype=np.int64), va_tab, va_dur)) + offsets
    return {
        "X": train_x.astype(np.int64),
        "y": np.asarray([r["y"] for r in train_rows], dtype=np.float32),
        "user": np.asarray(tr_user_raw),
        "date": np.asarray([r["date"] for r in train_rows]),
        "val_X": val_x.astype(np.int64),
        "val_y": np.asarray([r["y"] for r in val_rows], dtype=np.int64),
        "val_user": np.asarray(va_user_raw),
        "val_video_output": np.asarray(va_video_raw),
        "field_dims": dims,
        "evaluator_kind": "csv",
    }


def load_data(data_dir):
    train_path = os.path.join(data_dir, "train.npz")
    val_path = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_path) and os.path.exists(val_path):
        with np.load(train_path) as train, np.load(val_path) as val:
            dims = train["field_dims"].astype(np.int64, copy=True)
            val_x = val["X"].astype(np.int64, copy=True)
            video_offset = int(dims[0])
            return {
                "X": train["X"].astype(np.int64, copy=True),
                "y": train["y"].astype(np.float32, copy=True),
                "user": train["user"].copy(),
                "date": train["date"].copy(),
                "val_X": val_x,
                "val_y": val["y"].astype(np.int64, copy=True),
                "val_user": val["user"].copy(),
                "val_video_output": (val_x[:, 1] - video_offset).copy(),
                "field_dims": dims,
                "evaluator_kind": "npz",
            }
    return load_csv_data(data_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    progress_path = os.path.join(args.out_dir, "progress.log")
    if os.path.exists(progress_path):
        os.remove(progress_path)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_data(args.data_dir)
    if data["evaluator_kind"] == "npz":
        from data.official.evaluate import evaluate as evaluator
    else:
        from harness.evaluate_provisional import evaluate as evaluator

    data["total_dim"] = int(data["field_dims"].sum())
    data["fields"] = int(data["X"].shape[1])
    pair_pos, pair_neg = build_user_pairs(data["user"], data["y"], args.seed + 7919)
    data["pair_pos"] = pair_pos
    data["pair_neg"] = pair_neg

    smoke_text = os.environ.get("SMOKE_EPOCHS")
    smoke_cap = max(1, int(smoke_text)) if smoke_text is not None else None
    coarse_epochs = min(1, smoke_cap) if smoke_cap is not None else 1
    refine_epochs = min(2, smoke_cap) if smoke_cap is not None else 2
    final_epochs = min(max(1, args.epochs), smoke_cap) if smoke_cap is not None else max(1, args.epochs)

    rng = np.random.RandomState(args.seed + 104729)
    all_train = np.arange(len(data["X"]), dtype=np.int64)
    coarse_n = min(len(all_train), 12000 if device.type == "cpu" else 30000)
    refine_n = min(len(all_train), 24000 if device.type == "cpu" else 60000)
    coarse_train = np.sort(rng.choice(all_train, size=coarse_n, replace=False))
    refine_train = np.sort(rng.choice(all_train, size=refine_n, replace=False))
    probe_limit = 12000 if device.type == "cpu" else 30000
    probe_val = stratified_probe_indices(data["val_user"], probe_limit, args.seed + 31337)
    full_val = np.arange(len(data["val_X"]), dtype=np.int64)

    coarse_configs = []
    lr_values = np.asarray([3.5e-4, 5.0e-4, 7.0e-4, 9.5e-4, 1.3e-3], dtype=np.float64)
    half_lives = np.asarray([3.5, 5.0, 7.0, 10.0, 14.0], dtype=np.float64)
    for _ in range(8):
        coarse_configs.append({
            "dropout": float(rng.uniform(0.15, 0.40)),
            "weight_decay": float(10.0 ** rng.uniform(math.log10(3e-5), math.log10(3e-3))),
            "lr": float(rng.choice(lr_values)),
            "half_life": float(rng.choice(half_lives)),
            "step_interval": int(rng.choice([1, 2, 3, 4])),
            "step_gamma": float(rng.uniform(0.25, 0.70)),
        })

    history = []
    coarse_results = []
    for index, config in enumerate(coarse_configs):
        primary, _, checkpoints = train_one(
            config, args.seed + 1000 + index, coarse_epochs, data, evaluator, device,
            coarse_train, probe_val, False,
        )
        coarse_results.append((primary, config))
        history.append({
            "stage": "coarse", "probe": index, "seed": args.seed + 1000 + index,
            "epochs": coarse_epochs, "rows": int(len(coarse_train)),
            "config": clean_config(config), "best_primary": round(primary, 8),
            "checkpoints": checkpoints,
        })
        append_progress(progress_path, "coarse", index, config, primary)

    coarse_results.sort(key=lambda item: item[0], reverse=True)
    center = coarse_results[0][1]
    refine_configs = [dict(center)]
    for _ in range(5):
        refine_configs.append({
            "dropout": float(np.clip(center["dropout"] + rng.normal(0.0, 0.035), 0.10, 0.45)),
            "weight_decay": float(np.clip(center["weight_decay"] * math.exp(rng.normal(0.0, 0.55)), 1e-5, 6e-3)),
            "lr": float(np.clip(center["lr"] * math.exp(rng.normal(0.0, 0.25)), 2.5e-4, 1.8e-3)),
            "half_life": float(np.clip(center["half_life"] * math.exp(rng.normal(0.0, 0.25)), 2.5, 18.0)),
            "step_interval": int(np.clip(center["step_interval"] + rng.choice([-1, 0, 1]), 1, 5)),
            "step_gamma": float(np.clip(center["step_gamma"] + rng.normal(0.0, 0.08), 0.18, 0.80)),
        })

    refine_results = []
    for index, config in enumerate(refine_configs):
        primary, _, checkpoints = train_one(
            config, args.seed + 2000 + index, refine_epochs, data, evaluator, device,
            refine_train, probe_val, False,
        )
        refine_results.append((primary, config))
        history.append({
            "stage": "refine", "probe": index, "seed": args.seed + 2000 + index,
            "epochs": refine_epochs, "rows": int(len(refine_train)),
            "config": clean_config(config), "best_primary": round(primary, 8),
            "checkpoints": checkpoints,
        })
        append_progress(progress_path, "refine", index, config, primary)

    refine_results.sort(key=lambda item: item[0], reverse=True)
    winning_config = refine_results[0][1]
    final_primary, final_scores, final_checkpoints = train_one(
        winning_config, args.seed, final_epochs, data, evaluator, device,
        all_train, full_val, True,
    )
    final_metrics = metric_dict(evaluator, data["val_user"], data["val_y"], final_scores)
    history.append({
        "stage": "final", "probe": 0, "seed": args.seed,
        "epochs": final_epochs, "rows": int(len(all_train)),
        "config": clean_config(winning_config), "best_primary": round(final_primary, 8),
        "checkpoints": final_checkpoints,
    })
    append_progress(progress_path, "final", 0, winning_config, final_primary)

    payload = {
        "gauc": final_metrics["gauc"],
        "ndcg5": final_metrics["ndcg5"],
        "primary": final_metrics["primary"],
        "selected_output": "best_half_epoch_checkpoint",
        "winning_config": clean_config(winning_config),
        "history": history,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for row_id, score in enumerate(final_scores):
            writer.writerow([row_id, data["val_user"][row_id], data["val_video_output"][row_id], format(float(score), ".9g")])


if __name__ == "__main__":
    main()
