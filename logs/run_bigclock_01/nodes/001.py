import argparse
import csv
import datetime
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_npz(data_dir):
    from data.official.evaluate import evaluate
    tr = np.load(os.path.join(data_dir, "train.npz"))
    va = np.load(os.path.join(data_dir, "val.npz"))
    data = {
        "Xt": tr["X"].astype(np.int64),
        "yt": tr["y"].astype(np.float32),
        "ut": tr["user"],
        "dates": tr["date"],
        "Xv": va["X"].astype(np.int64),
        "yv": va["y"].astype(np.int64),
        "uv": va["user"],
        "video_v": np.zeros(len(va["y"]), dtype=np.int64),
        "field_dims": tr["field_dims"].astype(np.int64),
    }
    return data, evaluate


def scalar_value(s):
    try:
        return int(s)
    except ValueError:
        return s


def load_csv_data(data_dir):
    from harness.evaluate_provisional import evaluate
    train_rows = []
    with open(os.path.join(data_dir, "train.csv"), newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            train_rows.append({
                "user": row["user_id"],
                "video": row["video_id"],
                "tab": row["tab"],
                "duration": float(row["duration_ms"]),
                "date": row["date"],
                "label": float(row["long_view"]),
            })
    val_rows = []
    with open(os.path.join(data_dir, "val.csv"), newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            val_rows.append({
                "user": row["user_id"],
                "video": row["video_id"],
                "tab": row["tab"],
                "duration": float(row["duration_ms"]),
                "label": int(float(row["long_view"])),
            })

    def make_map(values):
        return {v: i + 1 for i, v in enumerate(sorted(set(values)))}

    user_map = make_map([r["user"] for r in train_rows])
    video_map = make_map([r["video"] for r in train_rows])
    tab_map = make_map([r["tab"] for r in train_rows])
    durations = np.asarray([r["duration"] for r in train_rows], dtype=np.float64)
    edges = np.unique(np.quantile(durations, np.linspace(0.1, 0.9, 9)))
    field_dims = np.asarray([
        len(user_map) + 1,
        len(video_map) + 1,
        1,
        len(tab_map) + 1,
        len(edges) + 2,
    ], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1]))

    def encode(rows):
        x = np.zeros((len(rows), 5), dtype=np.int64)
        for i, r in enumerate(rows):
            x[i, 0] = user_map.get(r["user"], 0)
            x[i, 1] = video_map.get(r["video"], 0)
            x[i, 2] = 0
            x[i, 3] = tab_map.get(r["tab"], 0)
            x[i, 4] = int(np.searchsorted(edges, r["duration"], side="right")) + 1
        return x + offsets[None, :]

    data = {
        "Xt": encode(train_rows),
        "yt": np.asarray([r["label"] for r in train_rows], dtype=np.float32),
        "ut": np.asarray([scalar_value(r["user"]) for r in train_rows]),
        "dates": np.asarray([r["date"] for r in train_rows]),
        "Xv": encode(val_rows),
        "yv": np.asarray([r["label"] for r in val_rows], dtype=np.int64),
        "uv": np.asarray([scalar_value(r["user"]) for r in val_rows]),
        "video_v": np.asarray([scalar_value(r["video"]) for r in val_rows]),
        "field_dims": field_dims,
    }
    return data, evaluate


def date_ordinal(v):
    s = str(v.decode() if isinstance(v, bytes) else v)
    if s.endswith(".0"):
        s = s[:-2]
    digits = "".join(c for c in s if c.isdigit())
    if len(digits) >= 8:
        try:
            return datetime.datetime.strptime(digits[:8], "%Y%m%d").date().toordinal()
        except ValueError:
            pass
    try:
        return int(float(s))
    except ValueError:
        return 0


def recency_weights(dates, half_life):
    unique = np.unique(dates)
    mapping = {v: date_ordinal(v) for v in unique}
    ordinals = np.asarray([mapping[v] for v in dates], dtype=np.float32)
    ages = ordinals.max() - ordinals
    weights = np.exp(-math.log(2.0) * ages / float(half_life)).astype(np.float32)
    return weights / max(float(weights.mean()), 1e-6)


def make_pairs(users, labels, seed):
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    rng = np.random.RandomState(seed)
    pos_parts = []
    neg_parts = []
    for j in range(len(boundaries) - 1):
        rows = order[boundaries[j]:boundaries[j + 1]]
        pos = rows[labels[rows] > 0.5]
        neg = rows[labels[rows] <= 0.5]
        if len(pos) and len(neg):
            pos_parts.append(pos)
            neg_parts.append(rng.choice(neg, size=len(pos), replace=True))
    if not pos_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(pos_parts).astype(np.int64), np.concatenate(neg_parts).astype(np.int64)


class DCNLite(torch.nn.Module):
    def __init__(self, total_dim, fields=5, k=16, hidden=128, dropout=0.2):
        super().__init__()
        width = fields * k
        self.emb = torch.nn.Embedding(total_dim, k)
        self.linear = torch.nn.Embedding(total_dim, 1)
        self.emb_drop = torch.nn.Dropout(dropout)
        self.cross_w = torch.nn.Parameter(torch.empty(width))
        self.cross_b = torch.nn.Parameter(torch.zeros(width))
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(width, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, hidden // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.head = torch.nn.Linear(width + hidden // 2, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.normal_(self.cross_w, std=0.01)
        torch.nn.init.xavier_uniform_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)

    def forward(self, x):
        e = self.emb_drop(self.emb(x)).flatten(1)
        cross = e * torch.sum(e * self.cross_w, dim=1, keepdim=True) + self.cross_b + e
        deep = self.mlp(e)
        linear = self.linear(x).sum((1, 2))
        return self.bias + linear + self.head(torch.cat((cross, deep), dim=1)).squeeze(1)


def metric_values(evaluator, users, labels, scores):
    m = evaluator(users, labels, scores)
    return {
        "gauc": float(m.get("GAUC", m.get("gauc"))),
        "ndcg5": float(m.get("nDCG@5", m.get("ndcg5"))),
        "primary": float(m["primary"]),
    }


def predict(model, x, batch_size=65536):
    model.eval()
    pieces = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            pieces.append(model(x[start:start + batch_size]).cpu().numpy())
    return np.concatenate(pieces)


def train_variant(config, epochs, seed, tensors, evaluator, train_indices, pair_indices,
                  checkpoint_half, history, tag):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    Xt, yt, Xv = tensors["Xt"], tensors["yt"], tensors["Xv"]
    pos_all, neg_all = tensors["pos"], tensors["neg"]
    weights_np = recency_weights(tensors["dates_np"], config["half_life"])
    weights = torch.from_numpy(weights_np)
    model = DCNLite(tensors["total_dim"], dropout=config["dropout"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                  weight_decay=config["weight_decay"])
    bs = 8192
    best_primary = -1.0
    best_scores = None
    global_step = 0
    train_indices = np.asarray(train_indices, dtype=np.int64)
    pair_indices = np.asarray(pair_indices, dtype=np.int64)

    for epoch in range(epochs):
        b_order = train_indices[rng.permutation(len(train_indices))]
        p_order = pair_indices[rng.permutation(len(pair_indices))] if len(pair_indices) else pair_indices
        b_splits = np.array_split(b_order, 2)
        p_splits = np.array_split(p_order, 2)
        for half in range(2):
            model.train()
            bpart = b_splits[half]
            ppart = p_splits[half]
            steps = max(1, int(math.ceil(max(len(bpart), len(ppart), 1) / float(bs))))
            last_loss = 0.0
            for step in range(steps):
                optimizer.zero_grad()
                losses = []
                bi = bpart[step * bs:(step + 1) * bs]
                if len(bi):
                    bt = torch.from_numpy(bi)
                    logits = model(Xt[bt])
                    raw = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits, yt[bt], reduction="none")
                    losses.append(0.5 * (raw * weights[bt]).mean())
                pi = ppart[step * bs:(step + 1) * bs]
                if len(pi):
                    pt = torch.from_numpy(pi)
                    pos_idx = pos_all[pt]
                    neg_idx = neg_all[pt]
                    diff = model(Xt[pos_idx]) - model(Xt[neg_idx])
                    pair_w = 0.5 * (weights[pos_idx] + weights[neg_idx])
                    bpr = (torch.nn.functional.softplus(-diff) * pair_w).mean()
                    losses.append(0.5 * bpr)
                if losses:
                    loss = sum(losses)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    last_loss = float(loss.detach())
                    global_step += 1
            for group in optimizer.param_groups:
                group["lr"] *= config["decay"]
            should_eval = checkpoint_half or half == 1
            if should_eval:
                scores = predict(model, Xv)
                metrics = metric_values(evaluator, tensors["uv_np"], tensors["yv_np"], scores)
                history.append({
                    "stage": tag,
                    "epoch": epoch + 1,
                    "half": half + 1,
                    "step": global_step,
                    "train_loss": round(last_loss, 6),
                    "lr": optimizer.param_groups[0]["lr"],
                    "val_gauc": round(metrics["gauc"], 6),
                    "val_primary": round(metrics["primary"], 6),
                })
                if metrics["primary"] > best_primary + 1e-9:
                    best_primary = metrics["primary"]
                    best_scores = scores.copy()
    return best_primary, best_scores


def rank_transform(users, scores):
    result = np.empty(len(scores), dtype=np.float64)
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    for j in range(len(boundaries) - 1):
        rows = order[boundaries[j]:boundaries[j + 1]]
        local = np.argsort(scores[rows], kind="mergesort")
        ranks = np.empty(len(rows), dtype=np.float64)
        ranks[local] = np.arange(len(rows), dtype=np.float64)
        result[rows] = ranks / max(len(rows) - 1, 1)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    started = time.monotonic()

    fast = os.path.exists(os.path.join(args.data_dir, "train.npz")) and os.path.exists(
        os.path.join(args.data_dir, "val.npz"))
    if fast:
        data, evaluator = load_npz(args.data_dir)
    else:
        data, evaluator = load_csv_data(args.data_dir)

    Xt = torch.from_numpy(data["Xt"])
    yt = torch.from_numpy(data["yt"])
    Xv = torch.from_numpy(data["Xv"])
    pos_np, neg_np = make_pairs(data["ut"], data["yt"], args.seed + 91)
    tensors = {
        "Xt": Xt,
        "yt": yt,
        "Xv": Xv,
        "pos": torch.from_numpy(pos_np),
        "neg": torch.from_numpy(neg_np),
        "dates_np": data["dates"],
        "uv_np": data["uv"],
        "yv_np": data["yv"],
        "total_dim": int(data["field_dims"].sum()),
    }

    smoke = os.environ.get("SMOKE_EPOCHS")
    smoke_cap = int(smoke) if smoke is not None else None
    coarse_epochs = min(2, smoke_cap) if smoke_cap is not None else 2
    refine_epochs = min(4, smoke_cap) if smoke_cap is not None else 4
    final_epochs = min(args.epochs, smoke_cap) if smoke_cap is not None else args.epochs

    coarse_configs = [
        {"dropout": 0.15, "weight_decay": 0.00003, "lr": 0.0010, "decay": 0.78, "half_life": 14.0},
        {"dropout": 0.19, "weight_decay": 0.00009, "lr": 0.0010, "decay": 0.66, "half_life": 7.0},
        {"dropout": 0.23, "weight_decay": 0.00030, "lr": 0.0009, "decay": 0.57, "half_life": 3.5},
        {"dropout": 0.27, "weight_decay": 0.00070, "lr": 0.0012, "decay": 0.49, "half_life": 7.0},
        {"dropout": 0.31, "weight_decay": 0.00140, "lr": 0.0011, "decay": 0.61, "half_life": 14.0},
        {"dropout": 0.35, "weight_decay": 0.00300, "lr": 0.0008, "decay": 0.72, "half_life": 3.5},
        {"dropout": 0.40, "weight_decay": 0.00030, "lr": 0.0013, "decay": 0.43, "half_life": 7.0},
        {"dropout": 0.24, "weight_decay": 0.00300, "lr": 0.0007, "decay": 0.82, "half_life": 14.0},
    ]

    rng = np.random.RandomState(args.seed + 17)
    coarse_n = max(1, int(round(0.28 * len(data["yt"]))))
    coarse_train = rng.choice(len(data["yt"]), size=coarse_n, replace=False)
    coarse_pair_n = max(1, int(round(0.28 * len(pos_np)))) if len(pos_np) else 0
    coarse_pairs = rng.choice(len(pos_np), size=coarse_pair_n, replace=False) if coarse_pair_n else np.empty(0, dtype=np.int64)
    full_train = np.arange(len(data["yt"]), dtype=np.int64)
    full_pairs = np.arange(len(pos_np), dtype=np.int64)
    history = []
    probe_records = []

    coarse_results = []
    for i, config in enumerate(coarse_configs):
        score, _ = train_variant(config, coarse_epochs, args.seed + 100 + i, tensors,
                                 evaluator, coarse_train, coarse_pairs, False, history,
                                 "coarse_%02d" % i)
        coarse_results.append(score)
        probe_records.append({"stage": "coarse", "probe": i, "config": config,
                              "epochs": coarse_epochs, "row_fraction": 0.28,
                              "best_primary": score})
    coarse_winner = dict(coarse_configs[int(np.argmax(coarse_results))])

    d0 = coarse_winner["dropout"]
    w0 = coarse_winner["weight_decay"]
    l0 = coarse_winner["lr"]
    g0 = coarse_winner["decay"]
    h0 = coarse_winner["half_life"]
    refine_specs = [
        (0.00, 1.00, 1.00, 0.00, 1.00),
        (-0.035, 0.58, 0.92, 0.05, 1.00),
        (0.035, 1.70, 1.08, -0.05, 1.00),
        (-0.018, 1.28, 1.00, -0.025, 0.72),
        (0.018, 0.78, 0.96, 0.025, 1.45),
        (0.000, 2.15, 1.05, -0.075, 0.72),
    ]
    refine_configs = []
    for dd, wm, lm, dg, hm in refine_specs:
        refine_configs.append({
            "dropout": float(np.clip(d0 + dd, 0.12, 0.45)),
            "weight_decay": float(np.clip(w0 * wm, 0.00002, 0.004)),
            "lr": float(np.clip(l0 * lm, 0.00055, 0.0015)),
            "decay": float(np.clip(g0 + dg, 0.38, 0.86)),
            "half_life": float(np.clip(h0 * hm, 3.0, 18.0)),
        })

    refine_results = []
    for i, config in enumerate(refine_configs):
        score, _ = train_variant(config, refine_epochs, args.seed + 300 + i, tensors,
                                 evaluator, full_train, full_pairs, False, history,
                                 "refine_%02d" % i)
        refine_results.append(score)
        probe_records.append({"stage": "refine", "probe": i, "config": config,
                              "epochs": refine_epochs, "row_fraction": 1.0,
                              "best_primary": score})
    winning_config = dict(refine_configs[int(np.argmax(refine_results))])

    final_start = time.monotonic()
    _, first_scores = train_variant(winning_config, final_epochs, args.seed, tensors,
                                    evaluator, full_train, full_pairs, True, history,
                                    "final_seed_%d" % args.seed)
    final_duration = time.monotonic() - final_start
    member_scores = [first_scores]
    ensemble_seeds = [args.seed]

    estimated_finish = (time.monotonic() - started) + 4.0 * final_duration
    if fast and estimated_finish < 540.0:
        for offset in range(1, 5):
            member_seed = args.seed + offset
            _, scores = train_variant(winning_config, final_epochs, member_seed, tensors,
                                      evaluator, full_train, full_pairs, True, history,
                                      "final_seed_%d" % member_seed)
            member_scores.append(scores)
            ensemble_seeds.append(member_seed)

    if len(member_scores) == 1:
        best_scores = member_scores[0]
    else:
        ranked = [rank_transform(data["uv"], s) for s in member_scores]
        best_scores = np.mean(np.stack(ranked, axis=0), axis=0)

    final_metrics = metric_values(evaluator, data["uv"], data["yv"], best_scores)
    os.makedirs(args.out_dir, exist_ok=True)
    result = {
        "gauc": final_metrics["gauc"],
        "ndcg5": final_metrics["ndcg5"],
        "primary": final_metrics["primary"],
        "winning_config": winning_config,
        "ensemble_seeds": ensemble_seeds,
        "probes": probe_records,
        "history": history,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(result, fh)
    with open(os.path.join(args.out_dir, "predictions.csv"), "w") as fh:
        fh.write("row_id,user_id,video_id,score\n")
        for i, score in enumerate(best_scores):
            fh.write("%d,%s,%s,%.8g\n" % (i, data["uv"][i], data["video_v"][i], score))


if __name__ == "__main__":
    main()
