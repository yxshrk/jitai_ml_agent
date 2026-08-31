"""Successive-halving search over normalized exponential recency weighting for the tuned FM family."""
import argparse
import csv
import datetime
import gc
import json
import math
import os
import sys

import numpy as np
import torch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FM(torch.nn.Module):
    def __init__(self, total_dim, k=16, dropout=0.0):
        super().__init__()
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.dropout = float(dropout)
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)

    def forward(self, x):
        e = self.emb(x)
        if self.dropout > 0.0:
            e = torch.nn.functional.dropout(e, p=self.dropout, training=self.training)
        s = e.sum(1)
        pair = 0.5 * (s * s - (e * e).sum(1)).sum(1)
        return self.bias + self.lin(x).sum((1, 2)) + pair


def load_fast(data_dir):
    tr = np.load(os.path.join(data_dir, "train.npz"))
    va = np.load(os.path.join(data_dir, "val.npz"))
    field_dims = tr["field_dims"].astype(np.int64)
    video_offset = int(field_dims[0])
    video_ids = va["X"][:, 1].astype(np.int64) - video_offset
    return {
        "Xt": tr["X"].astype(np.int64),
        "yt": tr["y"].astype(np.float32),
        "dt": np.asarray(tr["date"]),
        "Xv": va["X"].astype(np.int64),
        "yv": va["y"].astype(np.int64),
        "dv": np.asarray(va["date"]),
        "users": va["user"],
        "videos": video_ids,
        "field_dims": field_dims,
        "fast": True,
    }


def read_csv_rows(path, training):
    rows = []
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            item = {
                "user_id": r["user_id"],
                "video_id": r["video_id"],
                "tab": r["tab"],
                "duration_ms": float(r["duration_ms"]),
                "date": r["date"],
            }
            if training or "long_view" in r:
                item["long_view"] = float(r["long_view"])
            rows.append(item)
    return rows


def make_map(values):
    uniq = sorted(set(values))
    return {v: i + 1 for i, v in enumerate(uniq)}


def load_csv_data(data_dir):
    train_rows = read_csv_rows(os.path.join(data_dir, "train.csv"), True)
    val_rows = read_csv_rows(os.path.join(data_dir, "val.csv"), False)
    user_map = make_map([r["user_id"] for r in train_rows])
    video_map = make_map([r["video_id"] for r in train_rows])
    tab_map = make_map([r["tab"] for r in train_rows])
    durations = np.asarray([r["duration_ms"] for r in train_rows], dtype=np.float64)
    positive = durations[durations > 0]
    if len(positive) == 0:
        edges = np.asarray([], dtype=np.float64)
    else:
        edges = np.unique(np.quantile(np.log1p(positive), np.linspace(0.0, 1.0, 17)[1:-1]))
    dims = np.asarray([len(user_map) + 1, len(video_map) + 1, 1,
                       len(tab_map) + 1, len(edges) + 2], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(dims)[:-1])).astype(np.int64)

    def encode(rows):
        x = np.zeros((len(rows), 5), dtype=np.int64)
        for i, r in enumerate(rows):
            x[i, 0] = user_map.get(r["user_id"], 0) + offsets[0]
            x[i, 1] = video_map.get(r["video_id"], 0) + offsets[1]
            x[i, 2] = offsets[2]
            x[i, 3] = tab_map.get(r["tab"], 0) + offsets[3]
            bucket = int(np.searchsorted(edges, np.log1p(max(0.0, r["duration_ms"])), side="right")) + 1
            x[i, 4] = bucket + offsets[4]
        return x

    return {
        "Xt": encode(train_rows),
        "yt": np.asarray([r["long_view"] for r in train_rows], dtype=np.float32),
        "dt": np.asarray([r["date"] for r in train_rows]),
        "Xv": encode(val_rows),
        "yv": np.asarray([r["long_view"] for r in val_rows], dtype=np.int64),
        "dv": np.asarray([r["date"] for r in val_rows]),
        "users": np.asarray([int(r["user_id"]) for r in val_rows]),
        "videos": np.asarray([r["video_id"] for r in val_rows]),
        "field_dims": dims,
        "fast": False,
    }


def metric_values(m):
    return {
        "gauc": float(m["GAUC"] if "GAUC" in m else m["gauc"]),
        "ndcg5": float(m["nDCG@5"] if "nDCG@5" in m else m["ndcg5"]),
        "primary": float(m["primary"]),
    }


def parse_one_date(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    for fmt in ("%Y%m%d", "%Y-%m-%d", "%Y/%m/%d", "%Y%m%d%H%M%S"):
        try:
            return float(datetime.datetime.strptime(text, fmt).date().toordinal())
        except ValueError:
            pass
    try:
        number = float(text)
    except ValueError:
        return np.nan
    if 1000000000.0 <= number <= 3000000000.0:
        return float(datetime.datetime.utcfromtimestamp(number).date().toordinal())
    if np.isfinite(number):
        return number
    return np.nan


def dates_to_days(values):
    flat = np.asarray(values).reshape(-1)
    unique, inverse = np.unique(flat.astype(str), return_inverse=True)
    parsed = np.asarray([parse_one_date(v) for v in unique], dtype=np.float64)
    return parsed[inverse]


def make_recency_options(train_dates, val_dates):
    td = dates_to_days(train_dates)
    vd = dates_to_days(val_dates)
    valid_t = np.isfinite(td)
    valid_v = np.isfinite(vd)
    if not np.any(valid_t):
        td = np.zeros(len(td), dtype=np.float64)
        valid_t = np.ones(len(td), dtype=bool)
    replacement = float(np.nanmedian(td[valid_t]))
    td = np.where(valid_t, td, replacement)
    boundary = float(np.min(vd[valid_v])) if np.any(valid_v) else float(np.max(td))
    ages = np.maximum(0.0, boundary - td)
    span = max(1.0, float(np.max(td) - np.min(td)))
    requested = [None, span / 8.0, span / 4.0, span / 2.0, span, span * 2.0]
    options = []
    n = max(1, len(ages))
    for option_id, half_life in enumerate(requested):
        if half_life is None:
            weights = np.ones(len(ages), dtype=np.float64)
            actual = None
        else:
            actual = max(0.25, float(half_life))
            while True:
                weights = np.exp2(-ages / actual)
                total = float(weights.sum())
                square_total = float(np.square(weights).sum())
                ess_ratio = (total * total) / (n * square_total) if square_total > 0 else 0.0
                if ess_ratio >= 0.50 or actual >= span * 64.0:
                    break
                actual *= 1.35
            weights /= max(float(weights.mean()), 1e-12)
        total = float(weights.sum())
        square_total = float(np.square(weights).sum())
        ess_ratio = (total * total) / (n * square_total) if square_total > 0 else 0.0
        options.append({
            "id": option_id,
            "requested_half_life_days": half_life,
            "half_life_days": actual,
            "ess_ratio": ess_ratio,
            "min_weight": float(weights.min()),
            "max_weight": float(weights.max()),
            "weights": torch.from_numpy(weights.astype(np.float32)),
        })
    return options, boundary, span


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=12)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    progress_path = os.path.join(a.out_dir, "progress.log")

    smoke = os.environ.get("SMOKE_EPOCHS")
    smoke_cap = max(1, int(smoke)) if smoke is not None else None
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fast = (os.path.exists(os.path.join(a.data_dir, "train.npz")) and
            os.path.exists(os.path.join(a.data_dir, "val.npz")))
    data = load_fast(a.data_dir) if fast else load_csv_data(a.data_dir)
    if fast:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate

    Xt = torch.from_numpy(data["Xt"])
    yt = torch.from_numpy(data["yt"])
    Xv = torch.from_numpy(data["Xv"])
    total_dim = int(data["field_dims"].sum())
    n = len(yt)
    bs = 8192
    recency_options, validation_boundary, date_span = make_recency_options(data["dt"], data["dv"])

    rng = np.random.default_rng(a.seed + 7717)
    k_choices = np.asarray([8, 12, 16, 16, 24, 32, 48], dtype=np.int64)
    count = 4 if smoke_cap is not None else 72
    configs = []
    for i in range(count):
        option = recency_options[i % len(recency_options)]
        configs.append({
            "id": i,
            "seed": int(a.seed + 1009 * (i + 1)),
            "k": int(rng.choice(k_choices)),
            "lr": float(10.0 ** rng.uniform(math.log10(2e-4), math.log10(5e-3))),
            "weight_decay": float(10.0 ** rng.uniform(-8.0, math.log10(3e-3))),
            "dropout": float(rng.uniform(0.0, 0.5)),
            "lr_gamma": float(rng.uniform(0.72, 1.0)),
            "recency_option": int(option["id"]),
            "half_life_days": option["half_life_days"],
            "ess_ratio": float(option["ess_ratio"]),
        })

    probe_history = []

    def run_training(config, requested_epochs, stage, keep_scores):
        epochs = min(int(requested_epochs), int(a.epochs))
        if smoke_cap is not None:
            epochs = min(epochs, smoke_cap)
        epochs = max(1, epochs)
        seed = int(config["seed"])
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model = FM(total_dim, config["k"], config["dropout"]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=config["lr"],
                               weight_decay=config["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=config["lr_gamma"])
        train_weights = recency_options[int(config["recency_option"])]["weights"]
        best_primary = -1.0
        best_scores = None
        best_metrics = None
        best_epoch = 0
        curve = []
        for epoch in range(epochs):
            model.train()
            perm = torch.randperm(n)
            last_loss = 0.0
            for start in range(0, n, bs):
                idx = perm[start:start + bs]
                xb = Xt[idx].to(device, non_blocking=True)
                yb = yt[idx].to(device, non_blocking=True)
                wb = train_weights[idx].to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                logits = model(xb)
                row_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, yb, reduction="none")
                loss = (row_loss * wb).mean()
                loss.backward()
                opt.step()
                last_loss = float(loss.detach().cpu().item())
            scheduler.step()
            model.eval()
            chunks = []
            with torch.no_grad():
                for start in range(0, len(Xv), 65536):
                    xb = Xv[start:start + 65536].to(device, non_blocking=True)
                    chunks.append(model(xb).detach().cpu().numpy())
            scores = np.concatenate(chunks)
            m = metric_values(evaluate(data["users"], data["yv"], scores))
            curve.append({
                "epoch": epoch + 1,
                "train_loss": round(last_loss, 6),
                "gauc": round(m["gauc"], 7),
                "ndcg5": round(m["ndcg5"], 7),
                "primary": round(m["primary"], 7),
            })
            if m["primary"] > best_primary + 1e-12:
                best_primary = m["primary"]
                best_epoch = epoch + 1
                best_metrics = m
                if keep_scores:
                    best_scores = scores.copy()
        record = {
            "stage": stage,
            "max_epochs": epochs,
            "config": dict(config),
            "best_epoch": best_epoch,
            "gauc": best_metrics["gauc"],
            "ndcg5": best_metrics["ndcg5"],
            "primary": best_metrics["primary"],
            "curve": curve,
        }
        probe_history.append(record)
        with open(progress_path, "a") as fh:
            fh.write(json.dumps({k: v for k, v in record.items() if k != "curve"}, sort_keys=True) + "\n")
        del model, opt, scheduler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return best_primary, best_scores, best_metrics, best_epoch

    if smoke_cap is not None:
        stage_plan = [(1, len(configs))]
    else:
        stage_plan = [(3, 26), (7, 8), (12, 1)]

    active = list(configs)
    for stage_index, (stage_epochs, survivors) in enumerate(stage_plan):
        results = []
        for config in active:
            score, _, _, _ = run_training(config, stage_epochs,
                                           "probe_%d" % (stage_index + 1), False)
            results.append((score, config))
        results.sort(key=lambda z: (-z[0], z[1]["id"]))
        active = [z[1] for z in results[:min(survivors, len(results))]]

    winner = active[0]
    final_epochs = min(a.epochs, 12)
    if smoke_cap is not None:
        final_epochs = min(final_epochs, smoke_cap)
    _, best_scores, final_metrics, best_epoch = run_training(
        winner, final_epochs, "final_full_fidelity", True)

    recency_summary = []
    for option in recency_options:
        recency_summary.append({
            "id": option["id"],
            "requested_half_life_days": option["requested_half_life_days"],
            "half_life_days": option["half_life_days"],
            "ess_ratio": option["ess_ratio"],
            "min_weight": option["min_weight"],
            "max_weight": option["max_weight"],
        })
    output_metrics = {
        "gauc": final_metrics["gauc"],
        "ndcg5": final_metrics["ndcg5"],
        "primary": final_metrics["primary"],
        "selected_config": winner,
        "selected_checkpoint_epoch": best_epoch,
        "validation_boundary_day": validation_boundary,
        "training_date_span_days": date_span,
        "recency_options": recency_summary,
        "history": probe_history,
    }
    with open(os.path.join(a.out_dir, "metrics.json"), "w") as fh:
        json.dump(output_metrics, fh)

    with open(os.path.join(a.out_dir, "predictions.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(best_scores):
            writer.writerow([i, data["users"][i], data["videos"][i], format(float(score), ".9g")])


if __name__ == "__main__":
    main()
