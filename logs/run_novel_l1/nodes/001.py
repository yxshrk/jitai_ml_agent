import argparse
import csv
import datetime as dt
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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


def load_npz(data_dir):
    from data.official.evaluate import evaluate

    tr = np.load(os.path.join(data_dir, "train.npz"), allow_pickle=False)
    va = np.load(os.path.join(data_dir, "val.npz"), allow_pickle=False)
    field_dims = tr["field_dims"].astype(np.int64)
    xtr = tr["X"].astype(np.int64)
    xva = va["X"].astype(np.int64)
    users_tr = tr["user"]
    users_va = va["user"]
    if "video" in va.files:
        videos_va = va["video"]
    else:
        videos_va = xva[:, 1] - int(field_dims[0])
    dates = tr["date"] if "date" in tr.files else np.zeros(len(xtr), dtype=np.int64)
    return {
        "Xtr": xtr,
        "ytr": tr["y"].astype(np.float32),
        "utr": users_tr,
        "date": np.asarray(dates),
        "Xva": xva,
        "yva": va["y"].astype(np.int64),
        "uva": users_va,
        "vva": videos_va,
        "field_dims": field_dims,
        "evaluate": evaluate,
    }


def read_csv_rows(path, train):
    rows = []
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            row = {
                "user_id": r["user_id"],
                "video_id": r["video_id"],
                "tab": r["tab"],
                "duration_ms": float(r["duration_ms"]),
                "date": r.get("date", "0"),
                "long_view": float(r["long_view"]),
            }
            rows.append(row)
    return rows


def make_map(values):
    d = {}
    for v in values:
        if v not in d:
            d[v] = len(d) + 1
    return d


def load_csv(data_dir):
    from harness.evaluate_provisional import evaluate

    tr = read_csv_rows(os.path.join(data_dir, "train.csv"), True)
    va = read_csv_rows(os.path.join(data_dir, "val.csv"), False)
    um = make_map([r["user_id"] for r in tr])
    vm = make_map([r["video_id"] for r in tr])
    tm = make_map([r["tab"] for r in tr])
    durations = np.asarray([r["duration_ms"] for r in tr], dtype=np.float64)
    cuts = np.unique(np.quantile(durations, np.linspace(0.1, 0.9, 9)))
    field_dims = np.asarray([len(um) + 1, len(vm) + 1, 1, len(tm) + 1, 10], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1]))

    def encode(rows):
        x = np.empty((len(rows), 5), dtype=np.int64)
        for i, r in enumerate(rows):
            x[i, 0] = um.get(r["user_id"], 0) + offsets[0]
            x[i, 1] = vm.get(r["video_id"], 0) + offsets[1]
            x[i, 2] = offsets[2]
            x[i, 3] = tm.get(r["tab"], 0) + offsets[3]
            x[i, 4] = int(np.searchsorted(cuts, r["duration_ms"], side="right")) + offsets[4]
        return x

    return {
        "Xtr": encode(tr),
        "ytr": np.asarray([r["long_view"] for r in tr], dtype=np.float32),
        "utr": np.asarray([r["user_id"] for r in tr]),
        "date": np.asarray([r["date"] for r in tr]),
        "Xva": encode(va),
        "yva": np.asarray([r["long_view"] for r in va], dtype=np.int64),
        "uva": np.asarray([r["user_id"] for r in va]),
        "vva": np.asarray([r["video_id"] for r in va]),
        "field_dims": field_dims,
        "evaluate": evaluate,
    }


def day_numbers(values):
    values = np.asarray(values)
    unique = np.unique(values)
    parsed = {}
    for value in unique:
        text = str(value)
        if text.endswith(".0"):
            text = text[:-2]
        try:
            parsed[value] = dt.datetime.strptime(text, "%Y%m%d").date().toordinal()
        except Exception:
            try:
                parsed[value] = int(float(text))
            except Exception:
                parsed[value] = 0
    out = np.asarray([parsed[v] for v in values], dtype=np.float32)
    return out


def build_pairs(users, labels, seed):
    users = np.asarray(users)
    labels = np.asarray(labels)
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    rng = np.random.RandomState(seed)
    pos_parts = []
    neg_parts = []
    for j in range(len(boundaries) - 1):
        group = order[boundaries[j] : boundaries[j + 1]]
        pos = group[labels[group] > 0.5]
        neg = group[labels[group] <= 0.5]
        if len(pos) and len(neg):
            pos_parts.append(pos)
            neg_parts.append(neg[rng.randint(0, len(neg), size=len(pos))])
    if not pos_parts:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.concatenate(pos_parts).astype(np.int64), np.concatenate(neg_parts).astype(np.int64)


class DCNLite(torch.nn.Module):
    def __init__(self, total_dim, fields=5, k=8, hidden=64, dropout=0.2, cross_layers=1):
        super().__init__()
        width = fields * k
        self.emb = torch.nn.Embedding(total_dim, k)
        self.linear = torch.nn.Embedding(total_dim, 1)
        self.emb_drop = torch.nn.Dropout(dropout)
        self.cross_w = torch.nn.ParameterList([torch.nn.Parameter(torch.empty(width)) for _ in range(cross_layers)])
        self.cross_b = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(width)) for _ in range(cross_layers)])
        self.deep = torch.nn.Sequential(
            torch.nn.Linear(width, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, hidden // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.out = torch.nn.Linear(width + hidden // 2, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.linear.weight)
        for w in self.cross_w:
            torch.nn.init.normal_(w, std=0.01)

    def forward(self, x):
        x0 = self.emb_drop(self.emb(x)).flatten(1)
        cross = x0
        for w, b in zip(self.cross_w, self.cross_b):
            cross = x0 * (cross * w).sum(1, keepdim=True) + b + cross
        deep = self.deep(x0)
        first = self.linear(x).sum((1, 2))
        return self.bias + first + self.out(torch.cat([cross, deep], dim=1)).squeeze(1)


def predict(model, X, device, batch_size=65536):
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[start : start + batch_size]).to(device)
            parts.append(model(xb).detach().cpu().numpy())
    return np.concatenate(parts).astype(np.float64)


def train_run(data, pair_pos, pair_neg, cfg, seed, epochs, fraction, device, checkpoints_per_epoch=1, keep_scores=False):
    seed_all(seed)
    X = data["Xtr"]
    y = data["ytr"]
    n = len(y)
    total_dim = int(data["field_dims"].sum())
    model = DCNLite(total_dim, dropout=cfg["dropout"], cross_layers=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    rng = np.random.RandomState(seed + 17011)
    dates = day_numbers(data["date"])
    ages = float(dates.max()) - dates
    recency = np.power(0.5, ages / cfg["half_life"]).astype(np.float32)
    recency /= max(float(recency.mean()), 1e-8)
    bs = 16384 if device.type == "cuda" else 8192
    epoch_n = max(bs, min(n, int(round(n * fraction))))
    best_primary = -1.0
    best_gauc = -1.0
    best_metrics = None
    best_scores = None
    curve = []
    steps_total = int(math.ceil(epoch_n / bs))
    checkpoints = set()
    for q in range(1, checkpoints_per_epoch + 1):
        checkpoints.add(max(1, int(math.ceil(steps_total * q / checkpoints_per_epoch))))
    for epoch in range(epochs):
        model.train()
        if epoch_n >= n:
            indices = rng.permutation(n)
        else:
            indices = rng.choice(n, size=epoch_n, replace=False)
        running = 0.0
        seen = 0
        for step, start in enumerate(range(0, epoch_n, bs), 1):
            idx = indices[start : start + bs]
            b = len(idx)
            if len(pair_pos):
                chosen = rng.randint(0, len(pair_pos), size=b)
                pi = pair_pos[chosen]
                ni = pair_neg[chosen]
                joined = np.concatenate([idx, pi, ni])
            else:
                pi = idx
                ni = idx
                joined = np.concatenate([idx, pi, ni])
            xb = torch.from_numpy(X[joined]).to(device)
            logits = model(xb)
            main_logits = logits[:b]
            pos_logits = logits[b : 2 * b]
            neg_logits = logits[2 * b :]
            target = torch.from_numpy(y[idx]).to(device)
            main_w = torch.from_numpy(recency[idx]).to(device)
            pair_w_np = 0.5 * (recency[pi] + recency[ni])
            pair_w = torch.from_numpy(pair_w_np).to(device)
            point_loss = (bce(main_logits, target) * main_w).mean()
            pair_loss = (torch.nn.functional.softplus(-(pos_logits - neg_logits)) * pair_w).mean()
            loss = 0.5 * point_loss + 0.5 * pair_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            running += float(loss.detach().cpu()) * b
            seen += b
            if step in checkpoints:
                scores = predict(model, data["Xva"], device)
                met = metric_values(data["evaluate"](data["uva"], data["yva"], scores))
                curve.append(
                    {
                        "epoch": round(epoch + step / steps_total, 3),
                        "train_loss": round(running / max(seen, 1), 6),
                        "val_gauc": round(met["gauc"], 6),
                        "val_primary": round(met["primary"], 6),
                    }
                )
                if met["primary"] > best_primary + 1e-9:
                    best_primary = met["primary"]
                    best_gauc = met["gauc"]
                    best_metrics = met
                    if keep_scores:
                        best_scores = scores.copy()
                model.train()
        if (epoch + 1) % int(cfg["step_size"]) == 0:
            for group in opt.param_groups:
                group["lr"] *= cfg["gamma"]
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "primary": best_primary,
        "gauc": best_gauc,
        "metrics": best_metrics,
        "scores": best_scores,
        "curve": curve,
    }


def coarse_configs(seed, count):
    rng = np.random.RandomState(seed + 913)
    configs = []
    half_choices = np.asarray([3.5, 5.0, 7.0, 10.0, 14.0], dtype=np.float64)
    gamma_choices = np.asarray([0.35, 0.48, 0.62, 0.76, 0.88], dtype=np.float64)
    step_choices = np.asarray([1, 2, 3, 4], dtype=np.int64)
    for i in range(count):
        u = (i + rng.rand()) / count
        configs.append(
            {
                "dropout": float(np.clip(0.12 + 0.32 * ((i * 17 % count) + rng.rand()) / count, 0.12, 0.44)),
                "weight_decay": float(10 ** (-4.55 + 2.1 * u)),
                "lr": float(10 ** rng.uniform(-3.35, -2.75)),
                "gamma": float(gamma_choices[rng.randint(len(gamma_choices))]),
                "step_size": int(step_choices[rng.randint(len(step_choices))]),
                "half_life": float(half_choices[rng.randint(len(half_choices))]),
            }
        )
    rng.shuffle(configs)
    return configs


def refined_configs(winner, seed, count):
    rng = np.random.RandomState(seed + 2719)
    configs = [dict(winner)]
    for _ in range(count - 1):
        configs.append(
            {
                "dropout": float(np.clip(winner["dropout"] + rng.normal(0.0, 0.035), 0.10, 0.48)),
                "weight_decay": float(np.clip(winner["weight_decay"] * math.exp(rng.normal(0.0, 0.45)), 2e-5, 5e-3)),
                "lr": float(np.clip(winner["lr"] * math.exp(rng.normal(0.0, 0.22)), 3e-4, 2.2e-3)),
                "gamma": float(np.clip(winner["gamma"] + rng.normal(0.0, 0.07), 0.28, 0.94)),
                "step_size": int(np.clip(winner["step_size"] + rng.choice([-1, 0, 0, 1]), 1, 4)),
                "half_life": float(np.clip(winner["half_life"] * math.exp(rng.normal(0.0, 0.22)), 3.0, 16.0)),
            }
        )
    return configs


def append_progress(path, record):
    with open(path, "a") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def rank_average(score_list, users):
    users = np.asarray(users)
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    result = np.zeros(len(users), dtype=np.float64)
    for scores in score_list:
        ranked = np.zeros(len(users), dtype=np.float64)
        for j in range(len(boundaries) - 1):
            idx = order[boundaries[j] : boundaries[j + 1]]
            local_order = np.argsort(scores[idx], kind="mergesort")
            local_rank = np.empty(len(idx), dtype=np.float64)
            local_rank[local_order] = np.arange(len(idx), dtype=np.float64)
            if len(idx) > 1:
                local_rank /= len(idx) - 1
            ranked[idx] = local_rank
        result += ranked
    return result / len(score_list)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=16)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    progress_path = os.path.join(args.out_dir, "progress.log")
    if os.path.exists(progress_path):
        os.remove(progress_path)
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fast = os.path.exists(os.path.join(args.data_dir, "train.npz")) and os.path.exists(os.path.join(args.data_dir, "val.npz"))
    data = load_npz(args.data_dir) if fast else load_csv(args.data_dir)
    pair_pos, pair_neg = build_pairs(data["utr"], data["ytr"], args.seed)
    smoke_text = os.environ.get("SMOKE_EPOCHS")
    smoke_cap = int(smoke_text) if smoke_text is not None else None

    def cap(value):
        return min(value, smoke_cap) if smoke_cap is not None else value

    coarse_count = 6 if smoke_cap is not None else 8
    refine_count = 3 if smoke_cap is not None else 4
    coarse = coarse_configs(args.seed, coarse_count)
    history = []

    rung_specs = [
        ("coarse_rung1", cap(2), 0.50, max(3, coarse_count // 2)),
        ("coarse_rung2", cap(3), 1.00, max(2, coarse_count // 4)),
    ]
    survivors = list(enumerate(coarse))
    for rung_index, (phase, epochs, fraction, keep_n) in enumerate(rung_specs):
        results = []
        for candidate_id, cfg in survivors:
            run_seed = args.seed + 1000 * (rung_index + 1) + candidate_id
            result = train_run(data, pair_pos, pair_neg, cfg, run_seed, epochs, fraction, device)
            rec = {
                "phase": phase,
                "candidate": candidate_id,
                "epochs": epochs,
                "fraction": fraction,
                "config": cfg,
                "gauc": round(result["gauc"], 6),
                "primary": round(result["primary"], 6),
            }
            history.append(rec)
            append_progress(progress_path, rec)
            results.append((result["primary"], candidate_id, cfg))
        results.sort(key=lambda z: (z[0], -z[1]), reverse=True)
        survivors = [(candidate_id, cfg) for _, candidate_id, cfg in results[: min(keep_n, len(results))]]

    coarse_winner = survivors[0][1]
    refined = refined_configs(coarse_winner, args.seed, refine_count)
    refine_results = []
    for candidate_id, cfg in enumerate(refined):
        result = train_run(data, pair_pos, pair_neg, cfg, args.seed + 5000 + candidate_id, cap(3), 1.0, device)
        rec = {
            "phase": "local_refinement",
            "candidate": candidate_id,
            "epochs": cap(3),
            "fraction": 1.0,
            "config": cfg,
            "gauc": round(result["gauc"], 6),
            "primary": round(result["primary"], 6),
        }
        history.append(rec)
        append_progress(progress_path, rec)
        refine_results.append((result["primary"], candidate_id, cfg))
    refine_results.sort(key=lambda z: (z[0], -z[1]), reverse=True)
    final_cfg = refine_results[0][2]

    final_epochs = cap(min(args.epochs, 6))
    final_seed_count = 1 if smoke_cap is not None else 2
    member_scores = []
    member_curves = []
    for member in range(final_seed_count):
        member_seed = args.seed + member
        result = train_run(
            data,
            pair_pos,
            pair_neg,
            final_cfg,
            member_seed,
            final_epochs,
            1.0,
            device,
            checkpoints_per_epoch=1,
            keep_scores=True,
        )
        member_scores.append(result["scores"])
        member_curves.append({"seed": member_seed, "curve": result["curve"]})
        rec = {
            "phase": "matched_seed_final",
            "candidate": member,
            "seed": member_seed,
            "epochs": final_epochs,
            "fraction": 1.0,
            "config": final_cfg,
            "gauc": round(result["gauc"], 6),
            "primary": round(result["primary"], 6),
        }
        history.append(rec)
        append_progress(progress_path, rec)

    ensemble_trials = []
    best_scores = member_scores[0]
    best_met = metric_values(data["evaluate"](data["uva"], data["yva"], best_scores))
    best_count = 1
    for count in range(1, len(member_scores) + 1):
        scores = member_scores[0] if count == 1 else rank_average(member_scores[:count], data["uva"])
        met = metric_values(data["evaluate"](data["uva"], data["yva"], scores))
        trial = {"members": count, "gauc": round(met["gauc"], 6), "primary": round(met["primary"], 6)}
        ensemble_trials.append(trial)
        if met["primary"] > best_met["primary"] + 1e-9:
            best_met = met
            best_scores = scores.copy()
            best_count = count

    final_eval = metric_values(data["evaluate"](data["uva"], data["yva"], best_scores))
    metrics = {
        "gauc": final_eval["gauc"],
        "ndcg5": final_eval["ndcg5"],
        "primary": final_eval["primary"],
        "history": history,
        "winning_config": final_cfg,
        "selected_ensemble_members": best_count,
        "ensemble_trials": ensemble_trials,
        "final_learning_curves": member_curves,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh)
    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(best_scores):
            writer.writerow([i, data["uva"][i], data["vva"][i], format(float(score), ".9g")])


if __name__ == "__main__":
    main()
