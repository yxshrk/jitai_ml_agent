"""Two-stage search for a regularized DCN-lite and context-stratified BPR package."""
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
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.emb_dropout = torch.nn.Dropout(dropout)
        width = fields * k
        self.cross_w = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.empty(width)) for _ in range(cross_layers)]
        )
        self.cross_b = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.zeros(width)) for _ in range(cross_layers)]
        )
        self.deep = torch.nn.Sequential(
            torch.nn.Linear(width, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.cross_out = torch.nn.Linear(width, 1, bias=False)
        self.deep_out = torch.nn.Linear(64, 1, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)
        for w in self.cross_w:
            torch.nn.init.normal_(w, std=0.01)
        for module in self.deep:
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                torch.nn.init.zeros_(module.bias)
        torch.nn.init.xavier_uniform_(self.cross_out.weight)
        torch.nn.init.xavier_uniform_(self.deep_out.weight)

    def forward(self, x):
        e = self.emb_dropout(self.emb(x))
        x0 = e.flatten(1)
        xl = x0
        for w, b in zip(self.cross_w, self.cross_b):
            xl = x0 * (xl * w).sum(1, keepdim=True) + b + xl
        # Gauge fixing: omit the user-constant first-field linear bias while retaining
        # user/item interactions in the cross and deep branches.
        linear = self.lin(x[:, 1:]).sum((1, 2))
        return self.bias + linear + self.cross_out(xl).squeeze(1) + self.deep_out(self.deep(x0)).squeeze(1)


def read_csv_data(data_dir):
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    with open(train_path, newline="") as fh:
        train_rows = list(csv.DictReader(fh))
    with open(val_path, newline="") as fh:
        val_rows = list(csv.DictReader(fh))

    durations = np.asarray([float(r.get("duration_ms", 0) or 0) for r in train_rows], dtype=np.float64)
    cuts = np.unique(np.quantile(durations, np.linspace(0.1, 0.9, 9)))
    field_names = ["user_id", "video_id", "author_id", "tab"]
    maps = []
    for name in field_names:
        values = []
        seen = set()
        for row in train_rows:
            value = row.get(name, "0")
            if value not in seen:
                seen.add(value)
                values.append(value)
        maps.append({v: i + 1 for i, v in enumerate(values)})
    dims = [len(m) + 1 for m in maps] + [len(cuts) + 1]
    offsets = np.cumsum([0] + dims[:-1]).astype(np.int64)

    def encode(rows):
        x = np.zeros((len(rows), 5), dtype=np.int64)
        for j, name in enumerate(field_names):
            x[:, j] = np.asarray([maps[j].get(r.get(name, "0"), 0) for r in rows]) + offsets[j]
        d = np.asarray([float(r.get("duration_ms", 0) or 0) for r in rows], dtype=np.float64)
        x[:, 4] = np.searchsorted(cuts, d, side="right") + offsets[4]
        return x

    xt = encode(train_rows)
    xv = encode(val_rows)
    yt = np.asarray([float(r["long_view"]) for r in train_rows], dtype=np.float32)
    yv = np.asarray([float(r["long_view"]) for r in val_rows], dtype=np.float32)
    users_t = np.asarray([r["user_id"] for r in train_rows])
    users_v = np.asarray([r["user_id"] for r in val_rows])
    dates_t = np.asarray([r.get("date", "0") for r in train_rows])
    videos_v = np.asarray([r["video_id"] for r in val_rows])
    return {
        "Xt": xt, "yt": yt, "ut": users_t, "dt": dates_t,
        "Xv": xv, "yv": yv, "uv": users_v, "vv": videos_v,
        "field_dims": np.asarray(dims, dtype=np.int64), "fast": False,
    }


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        tr = np.load(train_npz)
        va = np.load(val_npz)
        video_values = va["X"][:, 1].astype(np.int64)
        return {
            "Xt": tr["X"].astype(np.int64),
            "yt": tr["y"].astype(np.float32),
            "ut": tr["user"],
            "dt": tr["date"],
            "Xv": va["X"].astype(np.int64),
            "yv": va["y"].astype(np.float32),
            "uv": va["user"],
            "vv": video_values,
            "field_dims": tr["field_dims"].astype(np.int64),
            "fast": True,
        }
    return read_csv_data(data_dir)


def date_ordinals(values):
    out = np.zeros(len(values), dtype=np.int64)
    for i, value in enumerate(values):
        text = str(value)
        if text.endswith(".0"):
            text = text[:-2]
        try:
            out[i] = datetime.datetime.strptime(text, "%Y%m%d").date().toordinal()
        except ValueError:
            try:
                out[i] = int(float(text))
            except ValueError:
                out[i] = 0
    return out


def recency_weights(dates, half_life):
    ordinal = date_ordinals(dates)
    valid = ordinal > 0
    newest = int(ordinal[valid].max()) if valid.any() else 0
    ages = np.maximum(newest - ordinal, 0).astype(np.float32)
    weights = np.power(0.5, ages / float(half_life)).astype(np.float32)
    weights /= max(float(weights.mean()), 1e-6)
    return weights


def build_context_pairs(users, labels, x, seed):
    rng = np.random.default_rng(seed)
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    pos_parts = []
    neg_parts = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        group = order[left:right]
        pos = group[labels[group] > 0.5]
        neg = group[labels[group] <= 0.5]
        if len(pos) == 0 or len(neg) == 0:
            continue
        candidates = neg[rng.integers(0, len(neg), size=(len(pos), min(8, max(2, len(neg)))))]
        ptab = x[pos, 3][:, None]
        pdur = x[pos, 4][:, None]
        penalty = 3.0 * (x[candidates, 3] != ptab) + np.abs(x[candidates, 4] - pdur)
        chosen = candidates[np.arange(len(pos)), np.argmin(penalty, axis=1)]
        pos_parts.append(pos.astype(np.int64))
        neg_parts.append(chosen.astype(np.int64))
    if not pos_parts:
        positives = np.flatnonzero(labels > 0.5).astype(np.int64)
        negatives = np.flatnonzero(labels <= 0.5).astype(np.int64)
        count = min(len(positives), len(negatives))
        return positives[:count], negatives[:count]
    return np.concatenate(pos_parts), np.concatenate(neg_parts)


def metric_values(result):
    return {
        "gauc": float(result.get("GAUC", result.get("gauc"))),
        "ndcg5": float(result.get("nDCG@5", result.get("ndcg5"))),
        "primary": float(result["primary"]),
    }


def predict(model, xv, device):
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(xv), 65536):
            xb = xv[start:start + 65536].to(device, non_blocking=True)
            chunks.append(model(xb).detach().cpu().numpy())
    return np.concatenate(chunks).astype(np.float64)


def train_one(config, seed, epochs, data_tensors, arrays, pairs, device, evaluator):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    xt, yt, xv = data_tensors
    labels = arrays["yt"]
    total_dim = int(arrays["field_dims"].sum())
    model = DCNLite(total_dim, dropout=float(config["dropout"]), cross_layers=2).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"])
    )
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    weights_np = recency_weights(arrays["dt"], float(config["half_life"]))
    weights = torch.from_numpy(weights_np)
    pos_idx, neg_idx = pairs
    n = len(labels)
    batch_size = 8192 if device.type == "cuda" else 4096
    rng = np.random.default_rng(seed + 17011)
    best_primary = -1.0
    best_scores = None
    best_metrics = None
    best_step = 0.0
    curve = []
    global_check = 0

    for epoch in range(epochs):
        model.train()
        permutation = rng.permutation(n)
        pair_permutation = rng.permutation(len(pos_idx)) if len(pos_idx) else np.empty(0, dtype=np.int64)
        pair_pointer = 0
        batch_losses = []
        batches = int(math.ceil(n / batch_size))
        half_batch = max(1, int(math.ceil(batches / 2)))
        for batch_number, start in enumerate(range(0, n, batch_size), 1):
            ids_np = permutation[start:start + batch_size]
            ids = torch.from_numpy(ids_np)
            xb = xt[ids].to(device, non_blocking=True)
            yb = yt[ids].to(device, non_blocking=True)
            wb = weights[ids].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            point_loss = (bce(logits, yb) * wb).mean()

            need = len(ids_np)
            if len(pos_idx):
                if pair_pointer + need > len(pair_permutation):
                    pair_permutation = rng.permutation(len(pos_idx))
                    pair_pointer = 0
                chosen = pair_permutation[pair_pointer:pair_pointer + min(need, len(pair_permutation))]
                pair_pointer += len(chosen)
                pi = torch.from_numpy(pos_idx[chosen])
                ni = torch.from_numpy(neg_idx[chosen])
                xp = xt[pi].to(device, non_blocking=True)
                xn = xt[ni].to(device, non_blocking=True)
                pair_weight = (weights[pi] + weights[ni]).mul(0.5).to(device, non_blocking=True)
                pair_loss = (torch.nn.functional.softplus(-(model(xp) - model(xn))) * pair_weight).mean()
                loss = 0.5 * point_loss + 0.5 * pair_loss
            else:
                loss = point_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))

            if batch_number == half_batch or batch_number == batches:
                global_check += 1
                scores = predict(model, xv, device)
                measured = metric_values(evaluator(arrays["uv"], arrays["yv"].astype(int), scores))
                step = epoch + (0.5 if batch_number == half_batch and batch_number != batches else 1.0)
                curve.append({
                    "epoch": float(step),
                    "train_loss": round(float(np.mean(batch_losses)), 6),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "val_gauc": round(measured["gauc"], 7),
                    "val_ndcg5": round(measured["ndcg5"], 7),
                    "val_primary": round(measured["primary"], 7),
                })
                if measured["primary"] > best_primary + 1e-9:
                    best_primary = measured["primary"]
                    best_scores = scores.copy()
                    best_metrics = measured
                    best_step = step
                model.train()
        if (epoch + 1) % int(config["step_every"]) == 0:
            for group in optimizer.param_groups:
                group["lr"] *= float(config["gamma"])

    return {
        "scores": best_scores,
        "metrics": best_metrics,
        "best_step": float(best_step),
        "curve": curve,
    }


def make_coarse_configs(rng, count):
    half_lives = [3.5, 5.0, 7.0, 10.0, 14.0]
    gammas = [0.35, 0.48, 0.62, 0.76, 0.88]
    intervals = [1, 2, 3]
    configs = []
    anchors = [
        (0.15, 3e-5, 4.0e-4, 0.48, 1, 3.5),
        (0.22, 1e-4, 7.0e-4, 0.62, 1, 7.0),
        (0.30, 4e-4, 1.0e-3, 0.76, 2, 10.0),
        (0.40, 3e-3, 1.6e-3, 0.88, 3, 14.0),
    ]
    for drop, wd, lr, gamma, every, half in anchors[:count]:
        configs.append({"dropout": drop, "weight_decay": wd, "lr": lr,
                        "gamma": gamma, "step_every": every, "half_life": half})
    while len(configs) < count:
        configs.append({
            "dropout": float(rng.uniform(0.13, 0.42)),
            "weight_decay": float(10 ** rng.uniform(math.log10(3e-5), math.log10(3e-3))),
            "lr": float(10 ** rng.uniform(math.log10(3.5e-4), math.log10(1.8e-3))),
            "gamma": float(gammas[int(rng.integers(len(gammas)))]),
            "step_every": int(intervals[int(rng.integers(len(intervals)))]),
            "half_life": float(half_lives[int(rng.integers(len(half_lives)))]),
        })
    return configs


def make_refine_configs(rng, winner, count):
    configs = [dict(winner)]
    half_options = np.asarray([3.5, 5.0, 7.0, 10.0, 14.0])
    while len(configs) < count:
        nearest = int(np.argmin(np.abs(half_options - float(winner["half_life"]))))
        half_index = int(np.clip(nearest + rng.integers(-1, 2), 0, len(half_options) - 1))
        configs.append({
            "dropout": float(np.clip(float(winner["dropout"]) + rng.normal(0, 0.035), 0.10, 0.46)),
            "weight_decay": float(np.clip(float(winner["weight_decay"]) * math.exp(rng.normal(0, 0.55)), 1e-5, 5e-3)),
            "lr": float(np.clip(float(winner["lr"]) * math.exp(rng.normal(0, 0.28)), 2.5e-4, 2.2e-3)),
            "gamma": float(np.clip(float(winner["gamma"]) + rng.normal(0, 0.07), 0.25, 0.94)),
            "step_every": int(np.clip(int(winner["step_every"]) + rng.integers(-1, 2), 1, 4)),
            "half_life": float(half_options[half_index]),
        })
    return configs


def rank_transform(scores):
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(len(scores), dtype=np.float64)
    if len(scores) > 1:
        ranks /= float(len(scores) - 1)
    return ranks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=18)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    progress_path = os.path.join(args.out_dir, "progress.log")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        device = torch.device("cpu")
    torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))

    arrays = load_data(args.data_dir)
    if arrays["fast"]:
        from data.official.evaluate import evaluate as evaluator
    else:
        from harness.evaluate_provisional import evaluate as evaluator

    xt = torch.from_numpy(arrays["Xt"])
    yt = torch.from_numpy(arrays["yt"])
    xv = torch.from_numpy(arrays["Xv"])
    data_tensors = (xt, yt, xv)
    pairs = build_context_pairs(arrays["ut"], arrays["yt"], arrays["Xt"], args.seed + 991)

    smoke_value = os.environ.get("SMOKE_EPOCHS")
    smoke = int(smoke_value) if smoke_value is not None else None
    if smoke is not None:
        coarse_count, refine_count, final_seeds = 2, 1, 1
        coarse_epochs = min(4, smoke)
        refine_epochs = min(6, smoke)
        final_epochs = min(args.epochs, smoke)
    else:
        if device.type == "cuda":
            coarse_count, refine_count = 120, 48
        else:
            coarse_count, refine_count = 72, 32
        final_seeds = 5
        coarse_epochs, refine_epochs, final_epochs = 4, 6, args.epochs

    rng = np.random.default_rng(args.seed + 4409)
    search_history = []
    coarse_configs = make_coarse_configs(rng, coarse_count)
    best_probe = None
    with open(progress_path, "a") as progress:
        for probe_id, config in enumerate(coarse_configs):
            result = train_one(config, args.seed, coarse_epochs, data_tensors, arrays, pairs, device, evaluator)
            record = {
                "stage": "coarse", "probe": probe_id + 1, "config": config,
                "best_step": result["best_step"], "gauc": result["metrics"]["gauc"],
                "ndcg5": result["metrics"]["ndcg5"], "primary": result["metrics"]["primary"],
                "curve": result["curve"],
            }
            search_history.append(record)
            progress.write(json.dumps({k: v for k, v in record.items() if k != "curve"}) + "\n")
            progress.flush()
            if best_probe is None or record["primary"] > best_probe["primary"]:
                best_probe = record

        refine_configs = make_refine_configs(rng, best_probe["config"], refine_count)
        best_refined = None
        for probe_id, config in enumerate(refine_configs):
            result = train_one(config, args.seed, refine_epochs, data_tensors, arrays, pairs, device, evaluator)
            record = {
                "stage": "refine", "probe": probe_id + 1, "config": config,
                "best_step": result["best_step"], "gauc": result["metrics"]["gauc"],
                "ndcg5": result["metrics"]["ndcg5"], "primary": result["metrics"]["primary"],
                "curve": result["curve"],
            }
            search_history.append(record)
            progress.write(json.dumps({k: v for k, v in record.items() if k != "curve"}) + "\n")
            progress.flush()
            if best_refined is None or record["primary"] > best_refined["primary"]:
                best_refined = record

        winning_config = best_refined["config"]
        final_history = []
        final_scores = []
        for seed_offset in range(final_seeds):
            final_seed = args.seed + seed_offset
            result = train_one(
                winning_config, final_seed, final_epochs, data_tensors, arrays, pairs, device, evaluator
            )
            final_scores.append(result["scores"])
            final_record = {
                "seed": final_seed, "best_step": result["best_step"],
                "gauc": result["metrics"]["gauc"], "ndcg5": result["metrics"]["ndcg5"],
                "primary": result["metrics"]["primary"], "curve": result["curve"],
            }
            final_history.append(final_record)
            progress.write(json.dumps({k: v for k, v in final_record.items() if k != "curve"}) + "\n")
            progress.flush()

    if len(final_scores) == 1:
        ensemble_scores = final_scores[0]
    else:
        ensemble_scores = np.mean(np.stack([rank_transform(s) for s in final_scores]), axis=0)
    final_metrics = metric_values(evaluator(arrays["uv"], arrays["yv"].astype(int), ensemble_scores))

    metrics_payload = {
        "gauc": final_metrics["gauc"],
        "ndcg5": final_metrics["ndcg5"],
        "primary": final_metrics["primary"],
        "winning_config": winning_config,
        "search_history": search_history,
        "final_history": final_history,
        "ensemble_seeds": [args.seed + i for i in range(final_seeds)],
        "context_pairs": int(len(pairs[0])),
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics_payload, fh)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for row_id, (user_id, video_id, score) in enumerate(zip(arrays["uv"], arrays["vv"], ensemble_scores)):
            writer.writerow([row_id, user_id, video_id, format(float(score), ".9g")])


if __name__ == "__main__":
    main()
