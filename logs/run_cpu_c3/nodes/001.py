"""Two-stage dial search for a regularized DCN-lite and hybrid BCE/BPR package."""
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
from data.official.evaluate import evaluate as official_evaluate


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DCNHybrid(torch.nn.Module):
    def __init__(self, total_dim, fields=5, k=16, hidden=128, cross_layers=1, dropout=0.25):
        super().__init__()
        width = fields * k
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.emb_drop = torch.nn.Dropout(dropout)
        self.cross_w = torch.nn.ModuleList(
            [torch.nn.Linear(width, 1, bias=False) for _ in range(cross_layers)]
        )
        self.cross_b = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.zeros(width)) for _ in range(cross_layers)]
        )
        second = max(32, hidden // 2)
        self.deep = torch.nn.Sequential(
            torch.nn.Linear(width, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, second),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(second, 1),
        )
        self.cross_out = torch.nn.Linear(width, 1)
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)
        torch.nn.init.zeros_(self.cross_out.weight)
        torch.nn.init.zeros_(self.cross_out.bias)
        torch.nn.init.normal_(self.deep[-1].weight, std=0.01)
        torch.nn.init.zeros_(self.deep[-1].bias)

    def forward(self, x):
        raw = self.emb(x)
        e = self.emb_drop(raw)
        summed = e.sum(1)
        fm = 0.5 * (summed.square() - e.square().sum(1)).sum(1)
        linear = self.bias + self.lin(x).sum((1, 2))
        x0 = e.flatten(1)
        xc = x0
        for w, b in zip(self.cross_w, self.cross_b):
            xc = x0 * w(xc) + b + xc
        return linear + fm + self.cross_out(xc).squeeze(1) + self.deep(x0).squeeze(1)


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def metric_values(m):
    return {
        "gauc": float(m.get("GAUC", m.get("gauc", 0.0))),
        "ndcg5": float(m.get("nDCG@5", m.get("ndcg5", 0.0))),
        "primary": float(m["primary"]),
    }


def load_csv_split(data_dir):
    def rows(path, validation=False):
        out = []
        with open(path, "r", newline="") as fh:
            for r in csv.DictReader(fh):
                out.append({
                    "user": r["user_id"],
                    "video": r["video_id"],
                    "tab": r["tab"],
                    "duration": float(r["duration_ms"]),
                    "date": r["date"],
                    "y": float(r["long_view"]),
                })
        return out

    train_rows = rows(os.path.join(data_dir, "train.csv"))
    val_rows = rows(os.path.join(data_dir, "val.csv"), True)
    user_map = {v: i + 1 for i, v in enumerate(sorted({r["user"] for r in train_rows}))}
    video_map = {v: i + 1 for i, v in enumerate(sorted({r["video"] for r in train_rows}))}
    tab_map = {v: i + 1 for i, v in enumerate(sorted({r["tab"] for r in train_rows}))}
    durations = np.asarray([r["duration"] for r in train_rows], dtype=np.float64)
    cuts = np.unique(np.quantile(durations, np.linspace(0.1, 0.9, 9)))
    dims = np.asarray([len(user_map) + 1, len(video_map) + 1, 1,
                       len(tab_map) + 1, len(cuts) + 1], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(dims)[:-1]))

    def encode(rs):
        x = np.zeros((len(rs), 5), dtype=np.int64)
        x[:, 0] = [user_map.get(r["user"], 0) for r in rs]
        x[:, 1] = [video_map.get(r["video"], 0) for r in rs]
        x[:, 2] = 0
        x[:, 3] = [tab_map.get(r["tab"], 0) for r in rs]
        x[:, 4] = np.searchsorted(cuts, [r["duration"] for r in rs], side="right")
        x += offsets
        return {
            "X": x.astype(np.int32),
            "y": np.asarray([r["y"] for r in rs], dtype=np.float32),
            "user": np.asarray([r["user"] for r in rs]),
            "video": np.asarray([r["video"] for r in rs]),
            "date": np.asarray([r["date"] for r in rs]),
            "field_dims": dims,
        }

    return encode(train_rows), encode(val_rows), False


def load_data(data_dir):
    train_path = os.path.join(data_dir, "train.npz")
    val_path = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_path) and os.path.exists(val_path):
        tr_npz = np.load(train_path)
        va_npz = np.load(val_path)
        tr = {k: tr_npz[k] for k in tr_npz.files}
        va = {k: va_npz[k] for k in va_npz.files}
        tr_npz.close()
        va_npz.close()
        offset = int(np.asarray(tr["field_dims"])[0])
        va["video"] = va["X"][:, 1].astype(np.int64) - offset
        return tr, va, True
    return load_csv_split(data_dir)


def date_ordinals(values):
    values = np.asarray(values)
    unique = np.unique(values.astype(str))
    parsed = {}
    ok = True
    for value in unique:
        text = str(value).strip()
        try:
            if len(text) >= 8 and text[:8].isdigit():
                d = datetime.datetime.strptime(text[:8], "%Y%m%d").date()
            else:
                d = datetime.date.fromisoformat(text[:10])
            parsed[value] = d.toordinal()
        except Exception:
            ok = False
            break
    if not ok:
        parsed = {value: i for i, value in enumerate(sorted(unique))}
    return np.asarray([parsed[str(v)] for v in values.astype(str)], dtype=np.float32)


def build_pair_index(users, labels):
    users = np.asarray(users)
    labels = np.asarray(labels) > 0.5
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    positives = []
    lows = []
    lengths = []
    negatives = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        idx = order[left:right]
        pos = idx[labels[idx]]
        neg = idx[~labels[idx]]
        if len(pos) == 0 or len(neg) == 0:
            continue
        low = len(negatives)
        negatives.extend(neg.tolist())
        positives.extend(pos.tolist())
        lows.extend([low] * len(pos))
        lengths.extend([len(neg)] * len(pos))
    return (torch.as_tensor(positives, dtype=torch.long),
            torch.as_tensor(lows, dtype=torch.long),
            torch.as_tensor(lengths, dtype=torch.long),
            torch.as_tensor(negatives, dtype=torch.long))


def predict(model, xv, batch_size):
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(xv), batch_size):
            xb = xv[start:start + batch_size].to(DEVICE, non_blocking=True)
            chunks.append(model(xb).detach().cpu().numpy())
    return np.concatenate(chunks).astype(np.float64)


def train_model(Xt, yt, Xv, val_users, val_y, date_age, pair_data, total_dim,
                config, epochs, seed, half_epoch_checks=False):
    seed_all(seed)
    model = DCNHybrid(total_dim, hidden=int(config["hidden"]),
                      cross_layers=int(config["cross_layers"]),
                      dropout=float(config["dropout"])).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=float(config["lr"]),
                            weight_decay=float(config["weight_decay"]))
    bs = 16384 if DEVICE.type == "cuda" else 8192
    eval_bs = 65536 if DEVICE.type == "cuda" else 32768
    n = len(yt)
    ages = torch.from_numpy(date_age.astype(np.float32))
    recency = torch.exp(-math.log(2.0) * ages / float(config["half_life"]))
    pair_pos, pair_low, pair_len, neg_flat = pair_data
    pair_count = len(pair_pos)
    best_primary = -1.0
    best_scores = None
    best_metrics = None
    curve = []
    last_loss = 0.0
    step_size = max(1, int(config["step_size"]))
    for epoch in range(epochs):
        lr_now = float(config["lr"]) * float(config["gamma"]) ** (epoch // step_size)
        for group in opt.param_groups:
            group["lr"] = lr_now
        model.train()
        perm = torch.randperm(n)
        nb = int(math.ceil(n / bs))
        checkpoints = {nb - 1}
        if half_epoch_checks:
            checkpoints.add(max(0, nb // 2 - 1))
        for batch_no, start in enumerate(range(0, n, bs)):
            idx = perm[start:start + bs]
            xb = Xt[idx].to(DEVICE, non_blocking=True)
            yb = yt[idx].to(DEVICE, non_blocking=True)
            wb = recency[idx].to(DEVICE, non_blocking=True)
            logits = model(xb)
            point = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, yb, reduction="none")
            point = (point * wb).sum() / wb.sum().clamp_min(1e-8)
            if pair_count:
                chosen = torch.randint(0, pair_count, (len(idx),))
                pos_cpu = pair_pos[chosen]
                lens = pair_len[chosen]
                offsets = torch.floor(torch.rand(len(idx)) * lens.float()).long()
                neg_cpu = neg_flat[pair_low[chosen] + offsets]
                pos_x = Xt[pos_cpu].to(DEVICE, non_blocking=True)
                neg_x = Xt[neg_cpu].to(DEVICE, non_blocking=True)
                pair_w = recency[pos_cpu].to(DEVICE, non_blocking=True)
                rank_loss = torch.nn.functional.softplus(-(model(pos_x) - model(neg_x)))
                rank_loss = (rank_loss * pair_w).sum() / pair_w.sum().clamp_min(1e-8)
                loss = 0.5 * point + 0.5 * rank_loss
            else:
                loss = point
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            last_loss = float(loss.detach().cpu())
            if batch_no in checkpoints:
                scores = predict(model, Xv, eval_bs)
                met = metric_values(official_evaluate(val_users, val_y, scores))
                fraction = epoch + float(batch_no + 1) / nb
                curve.append({"epoch": round(fraction, 3), "train_loss": round(last_loss, 6),
                              "lr": lr_now, "val_gauc": round(met["gauc"], 6),
                              "val_primary": round(met["primary"], 6)})
                if met["primary"] > best_primary + 1e-9:
                    best_primary = met["primary"]
                    best_scores = scores.copy()
                    best_metrics = met
                model.train()
    del model, opt
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return best_metrics, best_scores, curve


def coarse_configs(seed, count):
    rng = np.random.default_rng(seed + 1701)
    configs = []
    half_lives = np.asarray([3.5, 5.0, 7.0, 10.0, 14.0])
    gammas = np.asarray([0.28, 0.38, 0.50, 0.62, 0.74])
    for i in range(count):
        q = (i + rng.random()) / count
        configs.append({
            "dropout": float(rng.uniform(0.13, 0.42)),
            "weight_decay": float(10.0 ** (-4.52 + 2.0 * q)),
            "lr": float(10.0 ** rng.uniform(-3.35, -2.68)),
            "gamma": float(rng.choice(gammas)),
            "step_size": int(rng.choice([1, 2, 3])),
            "half_life": float(rng.choice(half_lives)),
            "hidden": int(rng.choice([64, 96, 128, 160])),
            "cross_layers": int(rng.choice([1, 2])),
        })
    rng.shuffle(configs)
    return configs


def refined_configs(winner, seed, count):
    rng = np.random.default_rng(seed + 2903)
    configs = [dict(winner)]
    hidden_choices = np.asarray([64, 96, 128, 160])
    for _ in range(count - 1):
        hidden_pos = int(np.argmin(np.abs(hidden_choices - int(winner["hidden"]))))
        hidden_pos = int(np.clip(hidden_pos + rng.choice([-1, 0, 0, 1]), 0,
                                 len(hidden_choices) - 1))
        configs.append({
            "dropout": float(np.clip(float(winner["dropout"]) + rng.normal(0, 0.035),
                                     0.10, 0.46)),
            "weight_decay": float(np.clip(float(winner["weight_decay"]) *
                                          math.exp(rng.normal(0, 0.48)), 2e-5, 5e-3)),
            "lr": float(np.clip(float(winner["lr"]) * math.exp(rng.normal(0, 0.22)),
                                3e-4, 2.5e-3)),
            "gamma": float(np.clip(float(winner["gamma"]) + rng.normal(0, 0.065),
                                   0.20, 0.82)),
            "step_size": int(np.clip(int(winner["step_size"]) +
                                     rng.choice([-1, 0, 0, 1]), 1, 4)),
            "half_life": float(np.clip(float(winner["half_life"]) *
                                         math.exp(rng.normal(0, 0.20)), 3.0, 16.0)),
            "hidden": int(hidden_choices[hidden_pos]),
            "cross_layers": int(np.clip(int(winner["cross_layers"]) +
                                         rng.choice([-1, 0, 0, 1]), 1, 2)),
        })
    return configs


def within_user_ranks(users, scores):
    users = np.asarray(users)
    scores = np.asarray(scores)
    result = np.zeros(len(scores), dtype=np.float64)
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    bounds = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    for left, right in zip(bounds[:-1], bounds[1:]):
        idx = order[left:right]
        local_order = np.argsort(scores[idx], kind="mergesort")
        ranks = np.empty(len(idx), dtype=np.float64)
        ranks[local_order] = np.arange(len(idx), dtype=np.float64)
        if len(idx) > 1:
            ranks /= len(idx) - 1
        result[idx] = ranks
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=14)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    seed_all(args.seed)

    tr, va, fast_path = load_data(args.data_dir)
    evaluator = official_evaluate
    if not fast_path:
        from harness.evaluate_provisional import evaluate as provisional_evaluate
        evaluator = provisional_evaluate

    Xt = torch.from_numpy(np.asarray(tr["X"], dtype=np.int64))
    yt = torch.from_numpy(np.asarray(tr["y"], dtype=np.float32))
    Xv = torch.from_numpy(np.asarray(va["X"], dtype=np.int64))
    val_y = np.asarray(va["y"], dtype=np.int64)
    val_users = np.asarray(va["user"])
    total_dim = int(np.asarray(tr["field_dims"]).sum())
    ordinals = date_ordinals(tr["date"])
    date_age = ordinals.max() - ordinals
    pair_data = build_pair_index(tr["user"], tr["y"])

    smoke_text = os.environ.get("SMOKE_EPOCHS")
    smoke = int(smoke_text) if smoke_text is not None else None
    coarse_n, refine_n, ensemble_n = (40, 20, 5) if smoke is None else (2, 1, 1)
    coarse_epochs = 4 if smoke is None else min(4, smoke)
    refine_epochs = 6 if smoke is None else min(6, smoke)
    final_epochs = max(1, args.epochs)
    if smoke is not None:
        final_epochs = min(final_epochs, smoke)

    history = []
    progress_path = os.path.join(args.out_dir, "progress.log")
    coarse_results = []
    for i, config in enumerate(coarse_configs(args.seed, coarse_n)):
        met, scores, curve = train_model(
            Xt, yt, Xv, val_users, val_y, date_age, pair_data, total_dim,
            config, coarse_epochs, args.seed + 500, False)
        entry = {"stage": "coarse", "probe": i + 1, "config": config,
                 "gauc": met["gauc"], "ndcg5": met["ndcg5"],
                 "primary": met["primary"], "curve": curve}
        history.append(entry)
        coarse_results.append((met["primary"], config))
        with open(progress_path, "a") as fh:
            fh.write(json.dumps({"stage": "coarse", "probe": i + 1,
                                 "config": config, "primary": met["primary"]}) + "\n")
        del scores

    coarse_results.sort(key=lambda x: x[0], reverse=True)
    coarse_winner = coarse_results[0][1]
    refine_results = []
    for i, config in enumerate(refined_configs(coarse_winner, args.seed, refine_n)):
        met, scores, curve = train_model(
            Xt, yt, Xv, val_users, val_y, date_age, pair_data, total_dim,
            config, refine_epochs, args.seed + 900, False)
        entry = {"stage": "refine", "probe": i + 1, "config": config,
                 "gauc": met["gauc"], "ndcg5": met["ndcg5"],
                 "primary": met["primary"], "curve": curve}
        history.append(entry)
        refine_results.append((met["primary"], config))
        with open(progress_path, "a") as fh:
            fh.write(json.dumps({"stage": "refine", "probe": i + 1,
                                 "config": config, "primary": met["primary"]}) + "\n")
        del scores

    refine_results.sort(key=lambda x: x[0], reverse=True)
    winner = refine_results[0][1]
    final_scores = []
    final_metrics = []
    for j in range(ensemble_n):
        run_seed = args.seed + j
        met, scores, curve = train_model(
            Xt, yt, Xv, val_users, val_y, date_age, pair_data, total_dim,
            winner, final_epochs, run_seed, True)
        final_scores.append(scores)
        final_metrics.append(met)
        entry = {"stage": "final", "seed": run_seed, "config": winner,
                 "gauc": met["gauc"], "ndcg5": met["ndcg5"],
                 "primary": met["primary"], "curve": curve}
        history.append(entry)
        with open(progress_path, "a") as fh:
            fh.write(json.dumps({"stage": "final", "seed": run_seed,
                                 "config": winner, "primary": met["primary"]}) + "\n")

    best_single = int(np.argmax([m["primary"] for m in final_metrics]))
    chosen_scores = final_scores[best_single]
    chosen_metric = final_metrics[best_single]
    selected_kind = "single"
    if len(final_scores) > 1:
        rank_scores = np.mean(
            np.stack([within_user_ranks(val_users, s) for s in final_scores], axis=0), axis=0)
        ensemble_metric = metric_values(evaluator(val_users, val_y, rank_scores))
        history.append({"stage": "ensemble", "seeds": [args.seed + j for j in range(ensemble_n)],
                        "method": "within_user_rank_average", **ensemble_metric})
        if ensemble_metric["primary"] >= chosen_metric["primary"]:
            chosen_scores = rank_scores
            chosen_metric = ensemble_metric
            selected_kind = "rank_ensemble"

    final_eval = metric_values(evaluator(val_users, val_y, chosen_scores))
    metrics_payload = {
        "gauc": final_eval["gauc"],
        "ndcg5": final_eval["ndcg5"],
        "primary": final_eval["primary"],
        "selected": selected_kind,
        "winning_config": winner,
        "history": history,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics_payload, fh)

    videos = np.asarray(va.get("video", np.zeros(len(chosen_scores), dtype=np.int64)))
    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(chosen_scores):
            writer.writerow([i, val_users[i], videos[i], format(float(score), ".8g")])


if __name__ == "__main__":
    main()
