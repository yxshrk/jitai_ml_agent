import argparse
import csv
import datetime
import json
import math
import os
import sys
import gc

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_npz(data_dir):
    tr = np.load(os.path.join(data_dir, "train.npz"))
    va = np.load(os.path.join(data_dir, "val.npz"))
    field_dims = tr["field_dims"].astype(np.int64)
    xtr = tr["X"].astype(np.int64)
    xva = va["X"].astype(np.int64)
    ytr = tr["y"].astype(np.float32)
    yva = va["y"].astype(np.float32)
    users_tr = tr["user"]
    users_va = va["user"]
    dates = tr["date"] if "date" in tr.files else np.zeros(len(ytr), dtype=np.int64)
    video_offset = int(field_dims[0])
    videos_va = xva[:, 1] - video_offset
    return {
        "xtr": xtr, "ytr": ytr, "utr": users_tr, "dates": dates,
        "xva": xva, "yva": yva, "uva": users_va, "vva": videos_va,
        "field_dims": field_dims, "fast": True
    }


def read_csv_rows(path, training):
    rows = []
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            item = {
                "user": row["user_id"],
                "video": row["video_id"],
                "tab": row.get("tab", ""),
                "duration": float(row.get("duration_ms", 0) or 0),
                "date": row.get("date", ""),
                "y": float(row["long_view"])
            }
            rows.append(item)
    return rows


def make_mapping(values):
    unique = sorted(set(values))
    return {v: i for i, v in enumerate(unique)}


def load_csv(data_dir):
    tr_rows = read_csv_rows(os.path.join(data_dir, "train.csv"), True)
    va_rows = read_csv_rows(os.path.join(data_dir, "val.csv"), False)
    user_map = make_mapping([r["user"] for r in tr_rows])
    video_map = make_mapping([r["video"] for r in tr_rows])
    tab_map = make_mapping([r["tab"] for r in tr_rows])
    durations = np.asarray([r["duration"] for r in tr_rows], dtype=np.float64)
    cuts = np.unique(np.quantile(durations, np.linspace(0.1, 0.9, 9))) if len(durations) else np.array([])
    field_dims = np.asarray([len(user_map) + 1, len(video_map) + 1, 1,
                             len(tab_map) + 1, len(cuts) + 1], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1])).astype(np.int64)

    def encode(rows):
        x = np.empty((len(rows), 5), dtype=np.int64)
        y = np.empty(len(rows), dtype=np.float32)
        users = np.empty(len(rows), dtype=object)
        videos = np.empty(len(rows), dtype=object)
        dates = np.empty(len(rows), dtype=object)
        for i, r in enumerate(rows):
            vals = [
                user_map.get(r["user"], len(user_map)),
                video_map.get(r["video"], len(video_map)),
                0,
                tab_map.get(r["tab"], len(tab_map)),
                int(np.searchsorted(cuts, r["duration"], side="right"))
            ]
            x[i] = np.asarray(vals, dtype=np.int64) + offsets
            y[i] = r["y"]
            users[i] = r["user"]
            videos[i] = r["video"]
            dates[i] = r["date"]
        return x, y, users, videos, dates

    xtr, ytr, utr, _, dates = encode(tr_rows)
    xva, yva, uva, vva, _ = encode(va_rows)
    return {
        "xtr": xtr, "ytr": ytr, "utr": utr, "dates": dates,
        "xva": xva, "yva": yva, "uva": uva, "vva": vva,
        "field_dims": field_dims, "fast": False
    }


def date_ordinal(value):
    s = str(value)
    if s.endswith(".0"):
        s = s[:-2]
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) >= 8:
        try:
            return datetime.date(int(digits[:4]), int(digits[4:6]), int(digits[6:8])).toordinal()
        except ValueError:
            pass
    return datetime.date(2022, 4, 21).toordinal()


def recency_weights(dates, half_life):
    anchor = datetime.date(2022, 4, 21).toordinal()
    ordinals = np.fromiter((date_ordinal(v) for v in dates), dtype=np.int64, count=len(dates))
    age = np.maximum(0, anchor - ordinals).astype(np.float32)
    weights = np.exp2(-age / float(half_life)).astype(np.float32)
    weights /= max(float(weights.mean()), 1e-6)
    return weights


def make_pairs(users, labels, seed):
    rng = np.random.RandomState(seed)
    order = np.argsort(users, kind="mergesort")
    pos_parts = []
    neg_parts = []
    start = 0
    while start < len(order):
        end = start + 1
        u = users[order[start]]
        while end < len(order) and users[order[end]] == u:
            end += 1
        group = order[start:end]
        pos = group[labels[group] > 0.5]
        neg = group[labels[group] <= 0.5]
        if len(pos) and len(neg):
            shuffled_neg = neg.copy()
            rng.shuffle(shuffled_neg)
            chosen = shuffled_neg[np.arange(len(pos)) % len(shuffled_neg)]
            pos_parts.append(pos.astype(np.int64))
            neg_parts.append(chosen.astype(np.int64))
        start = end
    if not pos_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(pos_parts), np.concatenate(neg_parts)


class DCNLite(torch.nn.Module):
    def __init__(self, total_dim, fields, k, dropout):
        super().__init__()
        self.fields = fields
        self.k = k
        width = fields * k
        self.emb = torch.nn.Embedding(total_dim, k)
        self.linear = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.emb_dropout = torch.nn.Dropout(dropout)
        self.cross_w = torch.nn.Parameter(torch.empty(width))
        self.cross_b = torch.nn.Parameter(torch.zeros(width))
        self.cross_out = torch.nn.Linear(width, 1, bias=False)
        self.deep = torch.nn.Sequential(
            torch.nn.Linear(width, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(64, 1, bias=False)
        )
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.normal_(self.cross_w, std=0.01)
        torch.nn.init.xavier_uniform_(self.cross_out.weight)

    def forward(self, x):
        e0 = self.emb(x)
        e = self.emb_dropout(e0)
        summed = e.sum(dim=1)
        fm = 0.5 * (summed.square() - e.square().sum(dim=1)).sum(dim=1)
        linear = self.linear(x).sum(dim=(1, 2))
        flat = e.reshape(e.shape[0], -1)
        cross = flat * torch.sum(flat * self.cross_w, dim=1, keepdim=True) + self.cross_b + flat
        return self.bias + linear + fm + self.cross_out(cross).squeeze(1) + self.deep(flat).squeeze(1)


def get_metrics(evaluate_fn, users, labels, scores):
    m = evaluate_fn(users, labels.astype(int), scores)
    return {
        "gauc": float(m["GAUC"] if "GAUC" in m else m["gauc"]),
        "ndcg5": float(m["nDCG@5"] if "nDCG@5" in m else m["ndcg5"]),
        "primary": float(m["primary"])
    }


def predict(model, x, device, batch_size=65536):
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start:start + batch_size]).to(device)
            parts.append(model(xb).detach().cpu().numpy())
    return np.concatenate(parts).astype(np.float32)


def train_one(data, config, epochs, seed, device, evaluate_fn, keep_trace):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = DCNLite(int(data["field_dims"].sum()), data["xtr"].shape[1], 16,
                    float(config["dropout"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["lr"]),
                                  weight_decay=float(config["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(config["step_every"]),
                                                gamma=float(config["step_gamma"]))
    sample_w = recency_weights(data["dates"], float(config["half_life"]))
    pair_pos, pair_neg = make_pairs(data["utr"], data["ytr"], seed + 991)
    rng = np.random.RandomState(seed + 17)
    n = len(data["ytr"])
    batch_size = 8192
    best_primary = -1.0
    best_scores = None
    best_metrics = None
    trace = []
    pair_order = np.arange(len(pair_pos), dtype=np.int64)
    for epoch in range(epochs):
        permutation = rng.permutation(n)
        if len(pair_order):
            pair_order = rng.permutation(len(pair_pos))
        halves = np.array_split(permutation, 2)
        pair_halves = np.array_split(pair_order, 2) if len(pair_order) else [pair_order, pair_order]
        for half_idx, indices in enumerate(halves):
            model.train()
            pair_indices = pair_halves[half_idx]
            pair_cursor = 0
            losses = []
            for start in range(0, len(indices), batch_size):
                idx = indices[start:start + batch_size]
                xb = torch.from_numpy(data["xtr"][idx]).to(device)
                yb = torch.from_numpy(data["ytr"][idx]).to(device)
                wb = torch.from_numpy(sample_w[idx]).to(device)
                logits = model(xb)
                bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, yb, reduction="none")
                bce_loss = torch.sum(bce * wb) / torch.sum(wb).clamp_min(1e-6)
                if len(pair_indices):
                    count = min(len(idx), len(pair_indices))
                    if pair_cursor + count <= len(pair_indices):
                        chosen = pair_indices[pair_cursor:pair_cursor + count]
                    else:
                        chosen = np.resize(pair_indices, count)
                    pair_cursor += count
                    pi = pair_pos[chosen]
                    ni = pair_neg[chosen]
                    xp = torch.from_numpy(data["xtr"][pi]).to(device)
                    xn = torch.from_numpy(data["xtr"][ni]).to(device)
                    pw = torch.from_numpy(0.5 * (sample_w[pi] + sample_w[ni])).to(device)
                    pair_loss_each = torch.nn.functional.softplus(-(model(xp) - model(xn)))
                    pair_loss = torch.sum(pair_loss_each * pw) / torch.sum(pw).clamp_min(1e-6)
                    loss = 0.5 * bce_loss + 0.5 * pair_loss
                else:
                    loss = bce_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            scores = predict(model, data["xva"], device)
            metrics = get_metrics(evaluate_fn, data["uva"], data["yva"], scores)
            if metrics["primary"] > best_primary:
                best_primary = metrics["primary"]
                best_scores = scores.copy()
                best_metrics = metrics
            if keep_trace:
                trace.append({
                    "checkpoint": epoch + 0.5 * (half_idx + 1),
                    "loss": round(float(np.mean(losses)) if losses else 0.0, 6),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "gauc": metrics["gauc"],
                    "ndcg5": metrics["ndcg5"],
                    "primary": metrics["primary"]
                })
        scheduler.step()
    del optimizer, scheduler, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_metrics, best_scores, trace


def rank_within_user(users, scores):
    result = np.empty(len(scores), dtype=np.float64)
    order = np.argsort(users, kind="mergesort")
    start = 0
    while start < len(order):
        end = start + 1
        u = users[order[start]]
        while end < len(order) and users[order[end]] == u:
            end += 1
        idx = order[start:end]
        if len(idx) == 1:
            result[idx] = 0.5
        else:
            local_order = np.argsort(scores[idx], kind="mergesort")
            ranks = np.empty(len(idx), dtype=np.float64)
            ranks[local_order] = np.arange(len(idx), dtype=np.float64) / float(len(idx) - 1)
            result[idx] = ranks
        start = end
    return result


def config_for_json(config):
    return {
        "dropout": round(float(config["dropout"]), 7),
        "weight_decay": float(config["weight_decay"]),
        "lr": float(config["lr"]),
        "step_every": int(config["step_every"]),
        "step_gamma": round(float(config["step_gamma"]), 7),
        "half_life": round(float(config["half_life"]), 7)
    }


def append_progress(path, record):
    with open(path, "a") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


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
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        device = torch.device("cpu")
    fast = os.path.exists(os.path.join(args.data_dir, "train.npz")) and os.path.exists(os.path.join(args.data_dir, "val.npz"))
    if fast:
        from data.official.evaluate import evaluate as evaluate_fn
        data = load_npz(args.data_dir)
    else:
        from harness.evaluate_provisional import evaluate as evaluate_fn
        data = load_csv(args.data_dir)

    smoke_value = os.environ.get("SMOKE_EPOCHS")
    smoke_cap = int(smoke_value) if smoke_value is not None else None
    coarse_epochs = min(3, smoke_cap) if smoke_cap is not None else 3
    refine_epochs = min(6, smoke_cap) if smoke_cap is not None else 6
    final_epochs = min(args.epochs, smoke_cap) if smoke_cap is not None else args.epochs
    coarse_count = 1 if smoke_cap is not None else 96
    refine_count = 1 if smoke_cap is not None else 44
    final_seed_count = 1 if smoke_cap is not None else 5

    rng = np.random.RandomState(args.seed + 3001)
    coarse = []
    for _ in range(coarse_count):
        coarse.append({
            "dropout": rng.uniform(0.12, 0.43),
            "weight_decay": 10.0 ** rng.uniform(math.log10(2e-5), math.log10(4e-3)),
            "lr": 10.0 ** rng.uniform(math.log10(3.5e-4), math.log10(1.8e-3)),
            "step_every": int(rng.choice([1, 2, 3])),
            "step_gamma": rng.uniform(0.25, 0.72),
            "half_life": float(rng.choice([3.0, 4.5, 6.0, 8.5, 11.0, 15.0, 20.0]))
        })

    history = []
    best_config = None
    best_probe_primary = -1.0
    for probe_id, config in enumerate(coarse):
        metrics, _, _ = train_one(data, config, coarse_epochs, args.seed, device, evaluate_fn, False)
        record = {"stage": "coarse", "probe": probe_id, "epochs": coarse_epochs,
                  "config": config_for_json(config), **metrics}
        history.append(record)
        append_progress(progress_path, record)
        if metrics["primary"] > best_probe_primary:
            best_probe_primary = metrics["primary"]
            best_config = dict(config)

    refine = []
    for _ in range(refine_count):
        refine.append({
            "dropout": float(np.clip(rng.normal(best_config["dropout"], 0.045), 0.08, 0.5)),
            "weight_decay": float(np.clip(best_config["weight_decay"] * math.exp(rng.normal(0.0, 0.55)), 8e-6, 8e-3)),
            "lr": float(np.clip(best_config["lr"] * math.exp(rng.normal(0.0, 0.28)), 2e-4, 2.5e-3)),
            "step_every": int(rng.choice(sorted(set([max(1, best_config["step_every"] - 1), best_config["step_every"], min(4, best_config["step_every"] + 1)])))),
            "step_gamma": float(np.clip(rng.normal(best_config["step_gamma"], 0.09), 0.18, 0.82)),
            "half_life": float(np.clip(best_config["half_life"] * math.exp(rng.normal(0.0, 0.28)), 2.0, 24.0))
        })

    refined_best = best_config
    refined_primary = -1.0
    for probe_id, config in enumerate(refine):
        metrics, _, _ = train_one(data, config, refine_epochs, args.seed, device, evaluate_fn, False)
        record = {"stage": "refine", "probe": probe_id, "epochs": refine_epochs,
                  "config": config_for_json(config), **metrics}
        history.append(record)
        append_progress(progress_path, record)
        if metrics["primary"] > refined_primary:
            refined_primary = metrics["primary"]
            refined_best = dict(config)

    final_scores = []
    final_candidates = []
    for seed_offset in range(final_seed_count):
        final_seed = args.seed + seed_offset
        metrics, scores, trace = train_one(data, refined_best, final_epochs, final_seed,
                                           device, evaluate_fn, True)
        final_scores.append(scores)
        record = {"stage": "final", "seed": final_seed, "epochs": final_epochs,
                  "config": config_for_json(refined_best), **metrics,
                  "checkpoints": trace}
        history.append(record)
        append_progress(progress_path, {k: v for k, v in record.items() if k != "checkpoints"})
        final_candidates.append((metrics["primary"], scores, "single_seed_%d" % final_seed, metrics))

    ranked = np.stack([rank_within_user(data["uva"], s) for s in final_scores], axis=0)
    ensemble_scores = ranked.mean(axis=0).astype(np.float32)
    ensemble_metrics = get_metrics(evaluate_fn, data["uva"], data["yva"], ensemble_scores)
    ensemble_record = {"stage": "ensemble", "kind": "within_user_rank_average",
                       "seeds": [args.seed + i for i in range(final_seed_count)], **ensemble_metrics}
    history.append(ensemble_record)
    append_progress(progress_path, ensemble_record)
    final_candidates.append((ensemble_metrics["primary"], ensemble_scores,
                             "within_user_rank_average", ensemble_metrics))
    final_candidates.sort(key=lambda x: x[0], reverse=True)
    _, best_scores, selected_kind, metrics = final_candidates[0]

    output_metrics = {
        "gauc": metrics["gauc"],
        "ndcg5": metrics["ndcg5"],
        "primary": metrics["primary"],
        "selected": selected_kind,
        "winning_config": config_for_json(refined_best),
        "history": history
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(output_metrics, fh)
    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(best_scores):
            writer.writerow([i, data["uva"][i], data["vva"][i], format(float(score), ".9g")])


if __name__ == "__main__":
    main()
