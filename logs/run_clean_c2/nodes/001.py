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
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(d, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, 1),
        )
        self.cross_out = torch.nn.Linear(d, 1)
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)
        torch.nn.init.normal_(self.cross_w, std=0.01)
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        torch.nn.init.xavier_uniform_(self.mlp[3].weight)
        torch.nn.init.xavier_uniform_(self.cross_out.weight)
        torch.nn.init.zeros_(self.mlp[0].bias)
        torch.nn.init.zeros_(self.mlp[3].bias)
        torch.nn.init.zeros_(self.cross_out.bias)

    def forward(self, x):
        e = self.emb_drop(self.emb(x))
        x0 = e.reshape(e.shape[0], -1)
        xl = x0 + x0 * torch.sum(x0 * self.cross_w, dim=1, keepdim=True) + self.cross_b
        return self.bias + self.lin(x).sum((1, 2)) + self.cross_out(xl).squeeze(1) + self.mlp(x0).squeeze(1)


class PairSampler:
    def __init__(self, users, labels, allowed=None):
        if allowed is None:
            allowed = np.arange(len(labels), dtype=np.int64)
        else:
            allowed = np.asarray(allowed, dtype=np.int64)
        u_all = np.asarray(users)
        unique, codes = np.unique(u_all, return_inverse=True)
        del unique
        self.n_users = int(codes.max()) + 1 if len(codes) else 0
        pos = allowed[np.asarray(labels)[allowed] > 0.5]
        neg = allowed[np.asarray(labels)[allowed] <= 0.5]
        pos_codes = codes[pos]
        neg_codes = codes[neg]
        po = np.argsort(pos_codes, kind="stable")
        no = np.argsort(neg_codes, kind="stable")
        self.pos = pos[po]
        self.neg = neg[no]
        pc = np.bincount(pos_codes, minlength=self.n_users).astype(np.int64)
        nc = np.bincount(neg_codes, minlength=self.n_users).astype(np.int64)
        self.pstart = np.concatenate(([0], np.cumsum(pc)[:-1])).astype(np.int64)
        self.nstart = np.concatenate(([0], np.cumsum(nc)[:-1])).astype(np.int64)
        self.pc = pc
        self.nc = nc
        self.eligible = np.flatnonzero((pc > 0) & (nc > 0)).astype(np.int64)

    def sample(self, size, rng):
        us = self.eligible[rng.integers(0, len(self.eligible), size=size)]
        pi = self.pstart[us] + (rng.random(size) * self.pc[us]).astype(np.int64)
        ni = self.nstart[us] + (rng.random(size) * self.nc[us]).astype(np.int64)
        return self.pos[pi], self.neg[ni]


def date_ordinals(values):
    vals = np.asarray(values)
    out = np.empty(len(vals), dtype=np.float32)
    mapping = {}
    for v in np.unique(vals):
        s = str(v.decode() if isinstance(v, bytes) else v).strip()
        try:
            if s.endswith(".0"):
                s = s[:-2]
            if len(s) >= 8 and s[:8].isdigit():
                d = datetime.datetime.strptime(s[:8], "%Y%m%d").date()
                mapping[v] = float(d.toordinal())
            else:
                mapping[v] = float(s)
        except Exception:
            mapping[v] = float(len(mapping))
    for v, z in mapping.items():
        out[vals == v] = z
    return out


def load_csv_data(data_dir):
    def read_one(name, need_label):
        path = os.path.join(data_dir, name)
        rows = []
        with open(path, "r", newline="") as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                row = {
                    "user_id": r["user_id"],
                    "video_id": r["video_id"],
                    "tab": r["tab"],
                    "duration_ms": float(r["duration_ms"]),
                    "date": r["date"],
                }
                if need_label:
                    row["long_view"] = float(r["long_view"])
                else:
                    row["long_view"] = float(r["long_view"])
                rows.append(row)
        return rows

    trr = read_one("train.csv", True)
    var = read_one("val.csv", True)
    durations = np.asarray([r["duration_ms"] for r in trr], dtype=np.float64)
    edges = np.unique(np.quantile(durations, np.linspace(0.1, 0.9, 9)))
    field_names = ["user_id", "video_id", "author_proxy", "tab", "dur_bucket"]
    transformed = []
    for rows in (trr, var):
        block = []
        for r in rows:
            bucket = str(int(np.searchsorted(edges, r["duration_ms"], side="right")))
            block.append([r["user_id"], r["video_id"], r["video_id"], r["tab"], bucket])
        transformed.append(block)
    dims = []
    offsets = []
    maps = []
    offset = 0
    for j, _ in enumerate(field_names):
        values = sorted(set(row[j] for row in transformed[0]))
        mp = {v: i for i, v in enumerate(values)}
        maps.append(mp)
        offsets.append(offset)
        dims.append(len(values) + 1)
        offset += len(values) + 1
    arrays = []
    for block in transformed:
        x = np.empty((len(block), 5), dtype=np.int64)
        for i, row in enumerate(block):
            for j, value in enumerate(row):
                x[i, j] = offsets[j] + maps[j].get(value, dims[j] - 1)
        arrays.append(x)
    tr = {
        "X": arrays[0],
        "y": np.asarray([r["long_view"] for r in trr], dtype=np.float32),
        "user": np.asarray([r["user_id"] for r in trr]),
        "date": np.asarray([r["date"] for r in trr]),
        "field_dims": np.asarray(dims, dtype=np.int64),
    }
    va = {
        "X": arrays[1],
        "y": np.asarray([r["long_view"] for r in var], dtype=np.float32),
        "user": np.asarray([r["user_id"] for r in var]),
        "video": np.asarray([r["video_id"] for r in var]),
    }
    return tr, va


def metric_values(m):
    return {
        "gauc": float(m["GAUC"] if "GAUC" in m else m["gauc"]),
        "ndcg5": float(m["nDCG@5"] if "nDCG@5" in m else m["ndcg5"]),
        "primary": float(m["primary"]),
    }


def predict(model, X, device, batch_size=65536):
    model.eval()
    pieces = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = X[start:start + batch_size].to(device, non_blocking=True)
            pieces.append(model(xb).detach().cpu().numpy())
    return np.concatenate(pieces).astype(np.float64)


def train_run(config, run_seed, epochs, train_indices, pair_sampler, Xt, yt, weights, Xv,
              val_users, val_y, evaluator, device, half_checks=False):
    torch.manual_seed(run_seed)
    np.random.seed(run_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(run_seed)
    rng = np.random.default_rng(run_seed)
    model = DCNLite(int(config["total_dim"]), dropout=float(config["dropout"])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(config["lr"]),
                            weight_decay=float(config["weight_decay"]))
    indices = np.asarray(train_indices, dtype=np.int64)
    bs = 8192
    best_primary = -1.0
    best_scores = None
    curve = []
    checks_per_epoch = 2 if half_checks else 1
    for epoch in range(epochs):
        perm = indices[rng.permutation(len(indices))]
        segments = np.array_split(perm, checks_per_epoch)
        epoch_loss = 0.0
        steps = 0
        for segment_id, segment in enumerate(segments):
            model.train()
            for start in range(0, len(segment), bs):
                idx = segment[start:start + bs]
                if len(idx) == 0:
                    continue
                pidx, nidx = pair_sampler.sample(len(idx), rng)
                xb = Xt[idx].to(device, non_blocking=True)
                yb = yt[idx].to(device, non_blocking=True)
                wb = weights[idx].to(device, non_blocking=True)
                xp = Xt[pidx].to(device, non_blocking=True)
                xn = Xt[nidx].to(device, non_blocking=True)
                wp = weights[pidx].to(device, non_blocking=True)
                wn = weights[nidx].to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                logits = model(xb)
                raw_bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, yb, reduction="none")
                bce = torch.sum(raw_bce * wb) / torch.clamp(torch.sum(wb), min=1e-6)
                diff = model(xp) - model(xn)
                pair_w = 0.5 * (wp + wn)
                raw_pair = torch.nn.functional.softplus(-diff)
                bpr = torch.sum(raw_pair * pair_w) / torch.clamp(torch.sum(pair_w), min=1e-6)
                loss = 0.5 * bce + 0.5 * bpr
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                epoch_loss += float(loss.detach().cpu())
                steps += 1
            scores = predict(model, Xv, device)
            mv = metric_values(evaluator(val_users, val_y, scores))
            curve.append({
                "epoch": epoch + (segment_id + 1) / checks_per_epoch,
                "train_loss": round(epoch_loss / max(steps, 1), 6),
                "val_gauc": round(mv["gauc"], 6),
                "val_primary": round(mv["primary"], 6),
            })
            if mv["primary"] > best_primary + 1e-8:
                best_primary = mv["primary"]
                best_scores = scores.copy()
        if (epoch + 1) % int(config["step_every"]) == 0:
            for group in opt.param_groups:
                group["lr"] *= float(config["gamma"])
    return best_primary, best_scores, curve


def config_public(c):
    return {
        "dropout": round(float(c["dropout"]), 6),
        "weight_decay": float(c["weight_decay"]),
        "lr": float(c["lr"]),
        "gamma": round(float(c["gamma"]), 6),
        "step_every": int(c["step_every"]),
        "half_life": round(float(c["half_life"]), 6),
    }


def append_progress(path, record):
    with open(path, "a") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")
        fh.flush()


def rank_average(score_list):
    acc = np.zeros(len(score_list[0]), dtype=np.float64)
    for scores in score_list:
        order = np.argsort(scores, kind="mergesort")
        ranks = np.empty(len(scores), dtype=np.float64)
        ranks[order] = np.arange(len(scores), dtype=np.float64)
        acc += ranks / max(len(scores) - 1, 1)
    return acc / len(score_list)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=14)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    progress_path = os.path.join(args.out_dir, "progress.log")
    if os.path.exists(progress_path):
        os.remove(progress_path)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fast = os.path.exists(os.path.join(args.data_dir, "train.npz")) and os.path.exists(os.path.join(args.data_dir, "val.npz"))
    if fast:
        from data.official.evaluate import evaluate as evaluator
        tr_npz = np.load(os.path.join(args.data_dir, "train.npz"), allow_pickle=False)
        va_npz = np.load(os.path.join(args.data_dir, "val.npz"), allow_pickle=False)
        tr = {k: tr_npz[k] for k in tr_npz.files}
        va = {k: va_npz[k] for k in va_npz.files}
        video_out = np.zeros(len(va["y"]), dtype=np.int64)
    else:
        from harness.evaluate_provisional import evaluate as evaluator
        tr, va = load_csv_data(args.data_dir)
        video_out = va["video"]

    Xt = torch.from_numpy(np.asarray(tr["X"], dtype=np.int64))
    yt_np = np.asarray(tr["y"], dtype=np.float32)
    yt = torch.from_numpy(yt_np)
    Xv = torch.from_numpy(np.asarray(va["X"], dtype=np.int64))
    val_y = np.asarray(va["y"], dtype=np.int64)
    val_users = np.asarray(va["user"])
    total_dim = int(np.asarray(tr["field_dims"]).sum())
    n = len(yt_np)

    ords = date_ordinals(tr["date"])
    age = float(np.max(ords)) - ords
    weight_cache = {}

    def recency_weights(half_life):
        key = round(float(half_life), 8)
        if key not in weight_cache:
            w = np.exp2(-age / max(float(half_life), 0.25)).astype(np.float32)
            w /= max(float(w.mean()), 1e-6)
            weight_cache[key] = torch.from_numpy(w)
        return weight_cache[key]

    smoke_raw = os.environ.get("SMOKE_EPOCHS")
    smoke_cap = int(smoke_raw) if smoke_raw is not None else None
    final_epochs = max(1, int(args.epochs))
    if smoke_cap is not None:
        final_epochs = min(final_epochs, max(1, smoke_cap))

    rng = np.random.default_rng(args.seed + 1701)
    coarse_fraction = 0.55
    coarse_size = max(1000, int(n * coarse_fraction)) if n >= 1000 else n
    coarse_indices = np.sort(rng.choice(n, size=min(n, coarse_size), replace=False)).astype(np.int64)
    full_indices = np.arange(n, dtype=np.int64)
    coarse_sampler = PairSampler(tr["user"], yt_np, coarse_indices)
    full_sampler = PairSampler(tr["user"], yt_np, full_indices)

    coarse_configs = []
    half_choices = np.asarray([3.5, 5.0, 7.0, 10.0, 14.0], dtype=np.float64)
    gamma_choices = np.asarray([0.38, 0.5, 0.62, 0.76], dtype=np.float64)
    for _ in range(12):
        coarse_configs.append({
            "total_dim": total_dim,
            "dropout": float(rng.uniform(0.14, 0.41)),
            "weight_decay": float(10.0 ** rng.uniform(math.log10(3e-5), math.log10(3e-3))),
            "lr": float(10.0 ** rng.uniform(math.log10(4.5e-4), math.log10(1.6e-3))),
            "gamma": float(rng.choice(gamma_choices)),
            "step_every": int(rng.choice([1, 1, 2])),
            "half_life": float(rng.choice(half_choices)),
        })
    coarse_configs[0].update({"dropout": 0.28, "weight_decay": 3e-4, "lr": 9e-4,
                              "gamma": 0.5, "step_every": 1, "half_life": 7.0})

    if smoke_cap is not None:
        coarse_configs = coarse_configs[:2]
        coarse_repeats = 1
    else:
        coarse_repeats = 3
    coarse_epochs = 3
    if smoke_cap is not None:
        coarse_epochs = min(coarse_epochs, max(1, smoke_cap))

    history = []
    coarse_summary = []
    for ci, config in enumerate(coarse_configs):
        vals = []
        for rep in range(coarse_repeats):
            run_seed = args.seed + 1000 + ci * 17 + rep
            primary, _, curve = train_run(config, run_seed, coarse_epochs, coarse_indices,
                                          coarse_sampler, Xt, yt, recency_weights(config["half_life"]),
                                          Xv, val_users, val_y, evaluator, device, False)
            record = {"stage": "coarse", "config_id": ci, "repeat": rep,
                      "seed": run_seed, "config": config_public(config),
                      "primary": round(float(primary), 7), "curve": curve}
            history.append(record)
            append_progress(progress_path, {k: v for k, v in record.items() if k != "curve"})
            vals.append(primary)
        coarse_summary.append((float(np.mean(vals)), float(np.std(vals)), config))
    coarse_summary.sort(key=lambda z: z[0], reverse=True)
    center = coarse_summary[0][2]

    refine_configs = [dict(center)]
    local_offsets = [
        (-0.055, -0.45, -0.12, -0.09, 0.75),
        (-0.025, -0.20, 0.08, 0.06, 0.9),
        (0.0, 0.0, -0.08, 0.0, 1.0),
        (0.025, 0.20, 0.06, -0.05, 1.1),
        (0.055, 0.45, -0.03, 0.08, 1.3),
        (-0.04, 0.3, 0.14, 0.03, 1.2),
        (0.04, -0.3, -0.16, -0.03, 0.82),
    ]
    for dd, dwd, dlr, dg, dh in local_offsets:
        c = dict(center)
        c["dropout"] = float(np.clip(center["dropout"] + dd, 0.10, 0.48))
        c["weight_decay"] = float(np.clip(center["weight_decay"] * (10.0 ** dwd), 1e-5, 7e-3))
        c["lr"] = float(np.clip(center["lr"] * (10.0 ** dlr), 2.5e-4, 2.2e-3))
        c["gamma"] = float(np.clip(center["gamma"] + dg, 0.28, 0.88))
        c["half_life"] = float(np.clip(center["half_life"] * dh, 2.5, 20.0))
        refine_configs.append(c)
    if smoke_cap is not None:
        refine_configs = refine_configs[:2]
        refine_repeats = 1
    else:
        refine_repeats = 2
    refine_epochs = 6
    if smoke_cap is not None:
        refine_epochs = min(refine_epochs, max(1, smoke_cap))

    refine_summary = []
    for ci, config in enumerate(refine_configs):
        vals = []
        for rep in range(refine_repeats):
            run_seed = args.seed + 5000 + ci * 19 + rep
            primary, _, curve = train_run(config, run_seed, refine_epochs, full_indices,
                                          full_sampler, Xt, yt, recency_weights(config["half_life"]),
                                          Xv, val_users, val_y, evaluator, device, False)
            record = {"stage": "refine", "config_id": ci, "repeat": rep,
                      "seed": run_seed, "config": config_public(config),
                      "primary": round(float(primary), 7), "curve": curve}
            history.append(record)
            append_progress(progress_path, {k: v for k, v in record.items() if k != "curve"})
            vals.append(primary)
        refine_summary.append((float(np.mean(vals)), float(np.std(vals)), config))
    refine_summary.sort(key=lambda z: z[0], reverse=True)
    winner = refine_summary[0][2]

    ensemble_count = 1 if smoke_cap is not None else 5
    final_scores = []
    final_primaries = []
    for rep in range(ensemble_count):
        run_seed = args.seed + rep
        primary, scores, curve = train_run(winner, run_seed, final_epochs, full_indices,
                                           full_sampler, Xt, yt, recency_weights(winner["half_life"]),
                                           Xv, val_users, val_y, evaluator, device, True)
        final_scores.append(scores)
        final_primaries.append(primary)
        record = {"stage": "final", "repeat": rep, "seed": run_seed,
                  "config": config_public(winner), "primary": round(float(primary), 7),
                  "curve": curve}
        history.append(record)
        append_progress(progress_path, {k: v for k, v in record.items() if k != "curve"})

    ensemble_scores = rank_average(final_scores)
    ensemble_metric = metric_values(evaluator(val_users, val_y, ensemble_scores))
    best_single_index = int(np.argmax(final_primaries))
    best_single_scores = final_scores[best_single_index]
    best_single_metric = metric_values(evaluator(val_users, val_y, best_single_scores))
    if ensemble_metric["primary"] >= best_single_metric["primary"]:
        best_scores = ensemble_scores
        selected = "rank_average"
        final_metric = ensemble_metric
    else:
        best_scores = best_single_scores
        selected = "best_single"
        final_metric = best_single_metric
    history.append({"stage": "ensemble_gate", "selected": selected,
                    "ensemble": ensemble_metric, "best_single": best_single_metric,
                    "best_single_seed": args.seed + best_single_index})

    metrics = {
        "gauc": final_metric["gauc"],
        "ndcg5": final_metric["ndcg5"],
        "primary": final_metric["primary"],
        "selected_output": selected,
        "winning_config": config_public(winner),
        "coarse_winner_mean": coarse_summary[0][0],
        "coarse_winner_std": coarse_summary[0][1],
        "refine_winner_mean": refine_summary[0][0],
        "refine_winner_std": refine_summary[0][1],
        "history": history,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w") as fh:
        fh.write("row_id,user_id,video_id,score\n")
        for i, score in enumerate(best_scores):
            fh.write(f"{i},{val_users[i]},{video_out[i]},{float(score):.9g}\n")


if __name__ == "__main__":
    main()
