import argparse
import csv
import datetime as dt
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def date_ordinal(value):
    s = str(value).strip()
    try:
        if "." in s:
            s = str(int(float(s)))
        s = s.replace("-", "")
        return dt.date(int(s[:4]), int(s[4:6]), int(s[6:8])).toordinal()
    except Exception:
        return 0


def parse_hourmin(value):
    try:
        v = int(float(str(value).strip()))
        hour, minute = v // 100, v % 100
        if 0 <= hour < 24 and 0 <= minute < 60:
            return hour, minute
    except Exception:
        pass
    return 24, 0


def safe_float(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else 0.0
    except Exception:
        return 0.0


def load_fast(data_dir):
    tr = np.load(Path(data_dir) / "train.npz", allow_pickle=False)
    va = np.load(Path(data_dir) / "val.npz", allow_pickle=False)
    Xtr = np.asarray(tr["X"], dtype=np.int64)
    Xva = np.asarray(va["X"], dtype=np.int64)
    ytr = np.asarray(tr["y"], dtype=np.float32)
    yva = np.asarray(va["y"], dtype=np.float32)
    dims = np.asarray(tr["field_dims"], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(dims[:-1]))).astype(np.int64)
    return {
        "Xtr": Xtr,
        "ytr": ytr,
        "utr": np.asarray(tr["user"]),
        "dates": np.asarray(tr["date"]),
        "hourmin_tr": np.asarray(tr["hourmin"]),
        "duration_tr": np.asarray(tr["duration_ms"], dtype=np.float32),
        "play_tr": np.asarray(tr["play_time_ms"], dtype=np.float32),
        "Xva": Xva,
        "yva": yva,
        "uva": np.asarray(va["user"]),
        "dates_va": np.asarray(va["date"]),
        "hourmin_va": np.asarray(va["hourmin"]),
        "duration_va": np.asarray(va["duration_ms"], dtype=np.float32),
        "video_out": Xva[:, 1] - offsets[1],
        "field_dims": dims,
        "fast": True,
    }


def read_csv_rows(path, training):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            item = {
                "user_id": row["user_id"],
                "video_id": row["video_id"],
                "author_id": row.get("author_id", "__missing_author__"),
                "tab": row.get("tab", "0"),
                "duration_ms": safe_float(row.get("duration_ms", 0)),
                "date": row.get("date", "0"),
                "hourmin": row.get("hourmin", "0"),
                "long_view": safe_float(row["long_view"]),
            }
            if training:
                item["play_time_ms"] = safe_float(row.get("play_time_ms", 0))
            rows.append(item)
    return rows


def load_csv(data_dir):
    train_rows = read_csv_rows(Path(data_dir) / "train.csv", True)
    val_rows = read_csv_rows(Path(data_dir) / "val.csv", False)
    durations = np.asarray([r["duration_ms"] for r in train_rows], dtype=np.float64)
    cuts = np.quantile(durations, np.linspace(0.1, 0.9, 9)) if len(durations) else np.zeros(9)

    def raw(rows):
        return [[r["user_id"], r["video_id"], r["author_id"], r["tab"],
                 str(int(np.searchsorted(cuts, r["duration_ms"], side="right")))] for r in rows]

    raw_tr, raw_va = raw(train_rows), raw(val_rows)
    maps, dims = [], []
    for j in range(5):
        values = sorted({r[j] for r in raw_tr})
        mapping = {v: i + 1 for i, v in enumerate(values)}
        maps.append(mapping)
        dims.append(len(mapping) + 1)
    offsets = np.concatenate(([0], np.cumsum(dims[:-1]))).astype(np.int64)

    def encode(rows):
        X = np.empty((len(rows), 5), dtype=np.int64)
        for i, row in enumerate(rows):
            for j in range(5):
                X[i, j] = maps[j].get(row[j], 0) + offsets[j]
        return X

    return {
        "Xtr": encode(raw_tr),
        "ytr": np.asarray([r["long_view"] for r in train_rows], dtype=np.float32),
        "utr": np.asarray([r["user_id"] for r in train_rows]),
        "dates": np.asarray([r["date"] for r in train_rows]),
        "hourmin_tr": np.asarray([r["hourmin"] for r in train_rows]),
        "duration_tr": np.asarray([r["duration_ms"] for r in train_rows], dtype=np.float32),
        "play_tr": np.asarray([r["play_time_ms"] for r in train_rows], dtype=np.float32),
        "Xva": encode(raw_va),
        "yva": np.asarray([r["long_view"] for r in val_rows], dtype=np.float32),
        "uva": np.asarray([r["user_id"] for r in val_rows]),
        "dates_va": np.asarray([r["date"] for r in val_rows]),
        "hourmin_va": np.asarray([r["hourmin"] for r in val_rows]),
        "duration_va": np.asarray([r["duration_ms"] for r in val_rows], dtype=np.float32),
        "video_out": np.asarray([r["video_id"] for r in val_rows]),
        "field_dims": np.asarray(dims, dtype=np.int64),
        "fast": False,
    }


def add_session_time_features(data):
    base_dims = np.asarray(data["field_dims"], dtype=np.int64)
    base_offsets = np.concatenate(([0], np.cumsum(base_dims[:-1]))).astype(np.int64)
    tab_dim = int(base_dims[3])
    gap_edges = np.asarray([1, 5, 15, 30, 60, 180, 720], dtype=np.float64)
    position_edges = np.asarray([1, 2, 3, 5, 8, 16], dtype=np.int64)
    extra_dims = np.asarray([9, 7, 25 * tab_dim, 8 * tab_dim], dtype=np.int64)
    cache = {}

    def ordinal(value):
        key = str(value)
        if key not in cache:
            cache[key] = date_ordinal(value)
        return cache[key]

    def derive(users, dates, hourmins, X, state):
        result = np.empty((len(users), 4), dtype=np.int64)
        tabs = np.clip(X[:, 3] - base_offsets[3], 0, tab_dim - 1).astype(np.int64)
        for i in range(len(users)):
            user = users[i].item() if isinstance(users[i], np.generic) else users[i]
            day = ordinal(dates[i])
            hour, minute = parse_hourmin(hourmins[i])
            timestamp = day * 1440 + hour * 60 + minute if day > 0 and hour < 24 else None
            previous = state.get(user)
            gap_code, position = 0, 1
            if previous is not None and timestamp is not None and previous[0] is not None:
                gap = timestamp - previous[0]
                if gap >= 0:
                    gap_code = 1 + int(np.searchsorted(gap_edges, float(gap), side="right"))
                    position = previous[1] + 1 if gap <= 30 else 1
            result[i, 0] = gap_code
            result[i, 1] = int(np.searchsorted(position_edges, position, side="right"))
            result[i, 2] = hour * tab_dim + tabs[i]
            weekday = (day - 1) % 7 if day > 0 else 7
            result[i, 3] = weekday * tab_dim + tabs[i]
            state[user] = (timestamp, position)
        return result

    state = {}
    tr = derive(data["utr"], data["dates"], data["hourmin_tr"], data["Xtr"], state)
    va = derive(data["uva"], data["dates_va"], data["hourmin_va"], data["Xva"], state)
    offsets = int(np.sum(base_dims)) + np.concatenate(([0], np.cumsum(extra_dims[:-1]))).astype(np.int64)
    data["Xtr"] = np.concatenate([data["Xtr"], tr + offsets], axis=1)
    data["Xva"] = np.concatenate([data["Xva"], va + offsets], axis=1)
    data["field_dims"] = np.concatenate([base_dims, extra_dims])


def make_recency_weights(dates, half_life):
    if half_life <= 0:
        return np.ones(len(dates), dtype=np.float32)
    ords = np.asarray([date_ordinal(x) for x in dates], dtype=np.int64)
    valid = ords > 0
    if not np.any(valid):
        return np.ones(len(dates), dtype=np.float32)
    ages = np.maximum(0, int(np.max(ords[valid])) - ords)
    weights = np.exp(-math.log(2.0) * ages / half_life).astype(np.float32)
    weights[~valid] = 1.0
    weights /= max(float(weights.mean()), 1e-8)
    return weights


def make_aux_targets(data):
    duration = np.maximum(np.nan_to_num(data["duration_tr"], nan=0.0), 1.0).astype(np.float32)
    play = np.maximum(np.nan_to_num(data["play_tr"], nan=0.0), 0.0).astype(np.float32)
    effective = np.minimum(duration, 18000.0)
    ratio = np.clip(play / np.maximum(effective, 1.0), 0.0, 1.5).astype(np.float32)
    thresholds = np.asarray([0.10, 0.25, 0.50, 0.75, 1.00], dtype=np.float32)
    data["ordinal_targets"] = (ratio[:, None] >= thresholds[None, :]).astype(np.float32)
    clipped_play = np.minimum(play, duration)
    data["cwm_target"] = (np.log1p(clipped_play) / np.maximum(np.log1p(duration), 1e-6)).astype(np.float32)
    data["cwm_completed"] = ((play >= duration) & (duration > 0)).astype(np.float32)
    data["duration_regime"] = np.where(duration <= 18000.0, 0, np.where(duration <= 60000.0, 1, 2)).astype(np.int64)


def make_pairs(users, labels, seed):
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    rng = np.random.default_rng(seed)
    pos_parts, neg_parts = [], []
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_users[end] == sorted_users[start]:
            end += 1
        idx = order[start:end]
        pos = idx[labels[idx] > 0.5]
        neg = idx[labels[idx] <= 0.5]
        if len(pos) and len(neg):
            pos_parts.append(pos.astype(np.int64, copy=False))
            neg_parts.append(neg[rng.integers(0, len(neg), size=len(pos))].astype(np.int64, copy=False))
        start = end
    if not pos_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(pos_parts), np.concatenate(neg_parts)


class RankModel(nn.Module):
    def __init__(self, n_vocab, n_fields, dropout):
        super().__init__()
        k = 16
        d = n_fields * k
        self.embedding = nn.Embedding(n_vocab, k)
        self.linear = nn.Embedding(n_vocab, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        self.embed_dropout = nn.Dropout(dropout)
        self.cross_scalar = nn.Linear(d, 1, bias=False)
        self.cross_bias = nn.Parameter(torch.zeros(d))
        self.deep = nn.Sequential(
            nn.Linear(d, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout),
        )
        self.head = nn.Linear(d + 32, 1)
        self.ordinal_head = nn.Linear(d + 32, 5)
        self.regime_head = nn.Linear(d + 32, 3)
        self.cwm_head = nn.Linear(d + 32, 1)
        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x, return_aux=False):
        emb = self.embed_dropout(self.embedding(x))
        linear = self.linear(x).sum(dim=1).squeeze(-1) + self.bias
        x0 = emb.reshape(emb.shape[0], -1)
        cross = x0 * self.cross_scalar(x0) + x0 + self.cross_bias
        deep = self.deep(x0)
        representation = torch.cat([cross, deep], dim=1)
        logits = linear + self.head(representation).squeeze(-1)
        if not return_aux:
            return logits
        return logits, self.ordinal_head(representation), self.regime_head(representation), self.cwm_head(representation).squeeze(-1)


def metric_function(fast):
    if fast:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    return evaluate


def normalize_metrics(result):
    return {
        "gauc": float(result.get("GAUC", result.get("gauc"))),
        "ndcg5": float(result.get("nDCG@5", result.get("ndcg5"))),
        "primary": float(result["primary"]),
    }


def score_model(model, X, batch_size, device):
    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.as_tensor(X[start:start + batch_size], dtype=torch.long, device=device)
            result.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(result).astype(np.float64)


def auxiliary_loss(outputs, idx, data, cfg, device):
    _, ordinal_logits, regime_logits, cwm_raw = outputs
    total = torch.zeros((), device=device)
    addons = set(cfg["addons"])
    if "ordinal" in addons:
        targets = torch.as_tensor(data["ordinal_targets"][idx], dtype=torch.float32, device=device)
        total = total + cfg["ordinal_weight"] * F.binary_cross_entropy_with_logits(ordinal_logits, targets)
    if "duration-regime" in addons:
        y = torch.as_tensor(data["ytr"][idx], dtype=torch.float32, device=device)
        regime = torch.as_tensor(data["duration_regime"][idx], dtype=torch.long, device=device)
        selected = regime_logits.gather(1, regime[:, None]).squeeze(1)
        total = total + cfg["regime_weight"] * F.binary_cross_entropy_with_logits(selected, y)
    if "cwm" in addons:
        target = torch.as_tensor(data["cwm_target"][idx], dtype=torch.float32, device=device)
        completed = torch.as_tensor(data["cwm_completed"][idx], dtype=torch.float32, device=device)
        prediction = torch.sigmoid(cwm_raw)
        uncensored = F.smooth_l1_loss(prediction, target, reduction="none")
        censored = F.relu(target - prediction).square()
        cwm = ((1.0 - completed) * uncensored + completed * censored).mean()
        total = total + cfg["cwm_weight"] * cwm
    return total


def train_one(data, cfg, seed, epochs, device, evaluator, pair_pos, pair_neg, sample_fraction, keep_predictions):
    seed_all(seed)
    Xtr, ytr = data["Xtr"], data["ytr"]
    n_vocab = max(int(np.sum(data["field_dims"])), int(Xtr.max()) + 1, int(data["Xva"].max()) + 1)
    model = RankModel(n_vocab, Xtr.shape[1], cfg["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    batch_size = 16384 if device.type == "cuda" else 8192
    recency = data["recency"][str(cfg["half_life"])]
    rng = np.random.default_rng(seed + 991)
    best_primary = -float("inf")
    best_state, best_metrics, best_predictions = None, None, None
    best_checkpoint = 0.0
    trajectory = []
    pair_order = np.arange(len(pair_pos), dtype=np.int64)
    pair_cursor = 0
    half_steps = max(1, int(math.ceil(len(Xtr) * sample_fraction / batch_size)))

    for epoch in range(epochs):
        for half in range(2):
            model.train()
            point_order = rng.permutation(len(Xtr))[:min(len(Xtr), half_steps * batch_size)]
            if len(pair_order):
                pair_order = rng.permutation(len(pair_pos))
                pair_cursor = 0
            for step in range(half_steps):
                idx = point_order[step * batch_size:(step + 1) * batch_size]
                if len(idx) == 0:
                    continue
                xb = torch.as_tensor(Xtr[idx], dtype=torch.long, device=device)
                yb = torch.as_tensor(ytr[idx], dtype=torch.float32, device=device)
                wb = torch.as_tensor(recency[idx], dtype=torch.float32, device=device)
                outputs = model(xb, True)
                logits = outputs[0]
                point = (F.binary_cross_entropy_with_logits(logits, yb, reduction="none") * wb).mean()
                aux = auxiliary_loss(outputs, idx, data, cfg, device)
                pair = torch.zeros((), device=device)
                if len(pair_order):
                    need = len(idx)
                    if pair_cursor + need > len(pair_order):
                        pair_order = rng.permutation(len(pair_pos))
                        pair_cursor = 0
                    selected = pair_order[pair_cursor:pair_cursor + need]
                    pair_cursor += len(selected)
                    pi, ni = pair_pos[selected], pair_neg[selected]
                    xp = torch.as_tensor(Xtr[pi], dtype=torch.long, device=device)
                    xn = torch.as_tensor(Xtr[ni], dtype=torch.long, device=device)
                    pw = torch.as_tensor(0.5 * (recency[pi] + recency[ni]), dtype=torch.float32, device=device)
                    pair = (F.softplus(-(model(xp) - model(xn))) * pw).mean()
                loss = 0.5 * point + 0.5 * pair + aux
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            predictions = score_model(model, data["Xva"], batch_size, device)
            metrics = normalize_metrics(evaluator(data["uva"], data["yva"], predictions))
            checkpoint = epoch + 0.5 * (half + 1)
            trajectory.append({"checkpoint": float(checkpoint), **metrics})
            if metrics["primary"] > best_primary:
                best_primary = metrics["primary"]
                best_metrics = metrics
                best_checkpoint = float(checkpoint)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if keep_predictions:
                    best_predictions = predictions.copy()
            for group in optimizer.param_groups:
                group["lr"] *= math.sqrt(cfg["lr_decay"])

    if best_state is not None:
        model.load_state_dict(best_state)
    if keep_predictions and best_predictions is None:
        best_predictions = score_model(model, data["Xva"], batch_size, device)
    result = {"metrics": best_metrics, "best_checkpoint": best_checkpoint, "trajectory": trajectory}
    if keep_predictions:
        result["predictions"] = best_predictions
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def append_progress(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def config_key(cfg):
    return json.dumps(cfg, sort_keys=True, separators=(",", ":"))


def base_config():
    return {
        "addons": [],
        "ordinal_weight": 0.0,
        "regime_weight": 0.0,
        "cwm_weight": 0.0,
        "half_life": 7.0,
        "dropout": 0.21,
        "weight_decay": 3.7e-5,
        "lr_decay": 0.72,
        "lr": 0.00168,
        "architecture": "dcn-lite",
        "loss": "bpr-hybrid",
        "session_time_features": True,
    }


def make_single_configs():
    configs = [base_config()]
    for weight in [0.08, 0.15, 0.25, 0.40]:
        cfg = base_config(); cfg["addons"] = ["ordinal"]; cfg["ordinal_weight"] = weight; configs.append(cfg)
    for weight in [0.08, 0.16, 0.28]:
        cfg = base_config(); cfg["addons"] = ["duration-regime"]; cfg["regime_weight"] = weight; configs.append(cfg)
    for weight in [0.05, 0.10, 0.18, 0.30, 0.45]:
        cfg = base_config(); cfg["addons"] = ["cwm"]; cfg["cwm_weight"] = weight; configs.append(cfg)
    for half_life in [3.5, 5.0, 10.0]:
        cfg = base_config(); cfg["addons"] = ["recency-variant"]; cfg["half_life"] = half_life; configs.append(cfg)
    return configs


def combine_configs(a, b):
    cfg = base_config()
    addons = []
    for name in a["addons"] + b["addons"]:
        if name not in addons:
            addons.append(name)
    cfg["addons"] = addons
    for key in ["ordinal_weight", "regime_weight", "cwm_weight"]:
        cfg[key] = max(a[key], b[key])
    cfg["half_life"] = b["half_life"] if "recency-variant" in b["addons"] else a["half_life"]
    return cfg


def main():
    args = parse_args()
    seed_all(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.log"
    if progress_path.exists():
        progress_path.unlink()
    data_dir = Path(args.data_dir)
    fast = (data_dir / "train.npz").exists() and (data_dir / "val.npz").exists()
    data = load_fast(data_dir) if fast else load_csv(data_dir)
    add_session_time_features(data)
    make_aux_targets(data)
    data["recency"] = {str(x): make_recency_weights(data["dates"], x) for x in [3.5, 5.0, 7.0, 10.0]}
    evaluator = metric_function(fast)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pair_pos, pair_neg = make_pairs(data["utr"], data["ytr"], args.seed + 17)
    smoke_value = os.environ.get("SMOKE_EPOCHS")
    smoke_cap = int(smoke_value) if smoke_value is not None else None
    probe_epochs = 3 if device.type == "cuda" else 2
    final_epochs = 8
    if smoke_cap is not None:
        probe_epochs = max(1, min(probe_epochs, smoke_cap))
        final_epochs = max(1, min(final_epochs, smoke_cap))
    probe_fraction = 0.7 if device.type == "cuda" else 0.55
    probe_seeds = [args.seed, args.seed + 1, args.seed + 2]
    single_configs = make_single_configs()
    if smoke_cap is not None:
        single_configs = single_configs[:2]
        probe_seeds = [args.seed]
        probe_fraction = 1.0

    history = []
    summaries = []

    def evaluate_configs(configs, phase):
        phase_summaries = []
        for cell, cfg in enumerate(configs):
            scores = []
            for seed in probe_seeds:
                result = train_one(data, cfg, seed, probe_epochs, device, evaluator, pair_pos, pair_neg, probe_fraction, False)
                record = {
                    "phase": phase,
                    "cell": cell,
                    "seed": seed,
                    "config": cfg,
                    "best_checkpoint": result["best_checkpoint"],
                    "metrics": result["metrics"],
                    "trajectory": result["trajectory"],
                }
                history.append(record)
                scores.append(result["metrics"]["primary"])
                append_progress(progress_path, {"phase": phase, "cell": cell, "seed": seed,
                                                "primary": scores[-1], "config": cfg})
            summary = {"phase": phase, "cell": cell, "config": cfg,
                       "mean_primary": float(np.mean(scores)), "std_primary": float(np.std(scores))}
            phase_summaries.append(summary)
            summaries.append(summary)
        return phase_summaries

    single_summary = evaluate_configs(single_configs, "single-addon")
    best_by_mechanism = {}
    for summary in single_summary:
        addons = summary["config"]["addons"]
        mechanism = addons[0] if addons else "control"
        previous = best_by_mechanism.get(mechanism)
        if previous is None or summary["mean_primary"] > previous["mean_primary"]:
            best_by_mechanism[mechanism] = summary

    ranked_mechanisms = [v for k, v in best_by_mechanism.items() if k != "control"]
    ranked_mechanisms.sort(key=lambda x: (-x["mean_primary"], x["std_primary"], x["cell"]))
    pair_configs = []
    top = ranked_mechanisms[:3]
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            pair_configs.append(combine_configs(top[i]["config"], top[j]["config"]))
    unique_pairs = []
    seen = set()
    for cfg in pair_configs:
        key = config_key(cfg)
        if key not in seen:
            seen.add(key)
            unique_pairs.append(cfg)
    if smoke_cap is not None:
        unique_pairs = []
    if unique_pairs:
        evaluate_configs(unique_pairs, "promising-pairs")

    summaries.sort(key=lambda x: (-x["mean_primary"], x["std_primary"], x["phase"], x["cell"]))
    winning_cfg = dict(summaries[0]["config"])
    final_result = train_one(data, winning_cfg, args.seed, final_epochs, device, evaluator,
                             pair_pos, pair_neg, 1.0, True)
    predictions = final_result["predictions"]
    final_metrics = normalize_metrics(evaluator(data["uva"], data["yva"], predictions))
    final_record = {
        "phase": "final",
        "seed": args.seed,
        "config": winning_cfg,
        "best_checkpoint": final_result["best_checkpoint"],
        "metrics": final_result["metrics"],
        "trajectory": final_result["trajectory"],
    }
    history.append(final_record)
    append_progress(progress_path, {"phase": "final", "seed": args.seed,
                                    "primary": final_result["metrics"]["primary"], "config": winning_cfg})

    with open(out_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, (user, video, score) in enumerate(zip(data["uva"], data["video_out"], predictions)):
            writer.writerow([i, user, video, format(float(score), ".10g")])

    payload = {
        "gauc": final_metrics["gauc"],
        "ndcg5": final_metrics["ndcg5"],
        "primary": final_metrics["primary"],
        "selected_config": winning_cfg,
        "selection_rule": "highest mean validation primary across equal-seed short probes",
        "objective_notes": {
            "cwm": "completed plays are right-censored at duration and incur only underprediction penalty",
            "ordinal": "cumulative watch-ratio thresholds at 0.10,0.25,0.50,0.75,1.00",
            "duration_regime": "auxiliary long-view heads for <=18s, 18-60s, and >60s",
        },
        "probe_summary": summaries,
        "history": history,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, sort_keys=True)


if __name__ == "__main__":
    main()
