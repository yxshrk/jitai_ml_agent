import argparse
import csv
import datetime
import json
import math
import os
import sys

import numpy as np
import torch


class DCNLite(torch.nn.Module):
    def __init__(self, total_dim, fields=5, k=16, hidden=128, dropout=0.30):
        super().__init__()
        self.fields = fields
        self.k = k
        d = fields * k
        self.emb = torch.nn.Embedding(total_dim, k)
        self.cross_w = torch.nn.ParameterList([
            torch.nn.Parameter(torch.empty(d)) for _ in range(2)
        ])
        self.cross_b = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros(d)) for _ in range(2)
        ])
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(d, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.main = torch.nn.Linear(d + 64, 1)
        self.ordinal = torch.nn.Linear(64, 4)
        self.cwm = torch.nn.Linear(64, 1)
        self.regime = torch.nn.Linear(64, 2)
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        for w in self.cross_w:
            torch.nn.init.normal_(w, std=0.01)

    def forward(self, x, short_regime, duration_strength=0.0):
        x0 = self.emb(x).reshape(x.shape[0], -1)
        xc = x0
        for w, b in zip(self.cross_w, self.cross_b):
            xc = x0 * torch.sum(xc * w, dim=1, keepdim=True) + b + xc
        h = self.mlp(x0)
        score = self.main(torch.cat([xc, h], dim=1)).squeeze(1)
        if duration_strength > 0.0:
            r = self.regime(h)
            selected = torch.where(short_regime, r[:, 0], r[:, 1])
            score = score + duration_strength * selected
        return score, self.ordinal(h), torch.sigmoid(self.cwm(h).squeeze(1))


def date_ord(value):
    s = str(int(value)) if not isinstance(value, str) else value.strip()
    try:
        return datetime.datetime.strptime(s[:8], "%Y%m%d").date().toordinal()
    except Exception:
        try:
            return int(float(s))
        except Exception:
            return 0


def load_csv_data(data_dir):
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    with open(train_path, newline="") as fh:
        train_rows = list(csv.DictReader(fh))
    with open(val_path, newline="") as fh:
        val_rows = list(csv.DictReader(fh))

    durations = np.asarray([float(r["duration_ms"]) for r in train_rows], dtype=np.float64)
    quantiles = np.quantile(durations, np.linspace(0.1, 0.9, 9))

    def make_map(rows, column):
        values = sorted({r[column] for r in rows})
        return {v: i + 1 for i, v in enumerate(values)}

    user_map = make_map(train_rows, "user_id")
    video_map = make_map(train_rows, "video_id")
    tab_map = make_map(train_rows, "tab")
    field_dims = np.asarray([
        len(user_map) + 1,
        len(video_map) + 1,
        1,
        len(tab_map) + 1,
        10,
    ], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1]))

    def encode(rows):
        x = np.zeros((len(rows), 5), dtype=np.int64)
        for i, r in enumerate(rows):
            dur = float(r["duration_ms"])
            raw = [
                user_map.get(r["user_id"], 0),
                video_map.get(r["video_id"], 0),
                0,
                tab_map.get(r["tab"], 0),
                int(np.searchsorted(quantiles, dur, side="right")),
            ]
            x[i] = np.asarray(raw, dtype=np.int64) + offsets
        return x

    train = {
        "X": encode(train_rows),
        "y": np.asarray([float(r["long_view"]) for r in train_rows], dtype=np.float32),
        "user": np.asarray([r["user_id"] for r in train_rows]),
        "play_time_ms": np.asarray([float(r["play_time_ms"]) for r in train_rows], dtype=np.float32),
        "duration_ms": np.asarray([float(r["duration_ms"]) for r in train_rows], dtype=np.float32),
        "date": np.asarray([date_ord(r["date"]) for r in train_rows], dtype=np.int64),
        "field_dims": field_dims,
    }
    val = {
        "X": encode(val_rows),
        "y": np.asarray([float(r["long_view"]) for r in val_rows], dtype=np.float32),
        "user": np.asarray([r["user_id"] for r in val_rows]),
        "duration_ms": np.asarray([float(r["duration_ms"]) for r in val_rows], dtype=np.float32),
        "video_out": np.asarray([r["video_id"] for r in val_rows]),
    }
    return train, val, False


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        tr_npz = np.load(train_npz)
        va_npz = np.load(val_npz)
        train = {k: tr_npz[k] for k in tr_npz.files}
        val = {k: va_npz[k] for k in va_npz.files}
        offset = int(train["field_dims"][0])
        val["video_out"] = val["X"][:, 1].astype(np.int64) - offset
        return train, val, True
    return load_csv_data(data_dir)


def metric_values(result):
    return {
        "gauc": float(result.get("GAUC", result.get("gauc"))),
        "ndcg5": float(result.get("nDCG@5", result.get("ndcg5"))),
        "primary": float(result["primary"]),
    }


def make_pairs(users, labels, seed):
    rng = np.random.default_rng(seed)
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    positives = []
    negatives = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        idx = order[left:right]
        pos = idx[labels[idx] > 0.5]
        neg = idx[labels[idx] <= 0.5]
        if len(pos) == 0 or len(neg) == 0:
            continue
        count = max(len(pos), len(neg))
        positives.append(rng.choice(pos, size=count, replace=len(pos) < count))
        negatives.append(rng.choice(neg, size=count, replace=len(neg) < count))
    if not positives:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(positives), np.concatenate(negatives)


def recency_weights(date_values, half_life):
    if half_life <= 0:
        return np.ones(len(date_values), dtype=np.float32)
    raw = np.asarray(date_values)
    if np.issubdtype(raw.dtype, np.number):
        unique = np.unique(raw)
        mapped = {v: date_ord(v) for v in unique}
        days = np.asarray([mapped[v] for v in raw], dtype=np.float32)
    else:
        days = np.asarray([date_ord(v) for v in raw], dtype=np.float32)
    age = np.max(days) - days
    weights = np.exp(-math.log(2.0) * age / half_life)
    weights /= max(float(weights.mean()), 1e-8)
    return weights.astype(np.float32)


def predict(model, x_cpu, duration_cpu, device, duration_strength):
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(x_cpu), 65536):
            stop = min(start + 65536, len(x_cpu))
            xb = x_cpu[start:stop].to(device)
            short = duration_cpu[start:stop].to(device) <= 18000.0
            score, _, _ = model(xb, short, duration_strength)
            chunks.append(score.detach().cpu().numpy())
    return np.concatenate(chunks)


def train_one(config, train, val, pair_pos, pair_neg, evaluator, device, seed,
              epochs, probe):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    x = torch.from_numpy(np.asarray(train["X"], dtype=np.int64))
    y = torch.from_numpy(np.asarray(train["y"], dtype=np.float32))
    duration = torch.from_numpy(np.asarray(train["duration_ms"], dtype=np.float32))
    play = torch.from_numpy(np.asarray(train["play_time_ms"], dtype=np.float32))
    xv = torch.from_numpy(np.asarray(val["X"], dtype=np.int64))
    duration_v = torch.from_numpy(np.asarray(val["duration_ms"], dtype=np.float32))
    weights = torch.from_numpy(recency_weights(train["date"], config["half_life"]))

    total_dim = int(np.asarray(train["field_dims"]).sum())
    model = DCNLite(total_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-3)
    bce_none = torch.nn.BCEWithLogitsLoss(reduction="none")
    rng = np.random.default_rng(seed)
    n = len(y)
    bs = 16384 if device.type == "cuda" else 8192
    best_primary = -1.0
    best_scores = None
    best_metrics = None
    checkpoint_history = []
    stale = 0

    ratio_np = np.clip(
        np.asarray(train["play_time_ms"], dtype=np.float32) /
        np.maximum(np.minimum(np.asarray(train["duration_ms"], dtype=np.float32), 18000.0), 1.0),
        0.0, 2.0,
    )
    thresholds = np.asarray([0.25, 0.50, 0.75, 1.00], dtype=np.float32)
    ordinal_targets = torch.from_numpy((ratio_np[:, None] >= thresholds[None, :]).astype(np.float32))
    watch_fraction = torch.from_numpy(np.clip(
        np.asarray(train["play_time_ms"], dtype=np.float32) /
        np.maximum(np.asarray(train["duration_ms"], dtype=np.float32), 1.0), 0.0, 1.0
    ).astype(np.float32))
    completed = torch.from_numpy((
        np.asarray(train["play_time_ms"], dtype=np.float32) >=
        np.asarray(train["duration_ms"], dtype=np.float32)
    ))

    pair_pos_t = torch.from_numpy(pair_pos)
    pair_neg_t = torch.from_numpy(pair_neg)
    last_loss = 0.0

    for epoch in range(epochs):
        point_perm = torch.from_numpy(rng.permutation(n).astype(np.int64))
        if len(pair_pos):
            pair_perm = torch.from_numpy(rng.permutation(len(pair_pos)).astype(np.int64))
        else:
            pair_perm = torch.empty(0, dtype=torch.int64)
        split = (n + 1) // 2
        phase_ranges = [(0, split), (split, n)]

        for phase, (phase_start, phase_stop) in enumerate(phase_ranges):
            model.train()
            pair_cursor = int(len(pair_perm) * phase / 2)
            pair_limit = int(len(pair_perm) * (phase + 1) / 2)
            for start in range(phase_start, phase_stop, bs):
                idx = point_perm[start:min(start + bs, phase_stop)]
                xb = x[idx].to(device)
                yb = y[idx].to(device)
                db = duration[idx].to(device)
                wb = weights[idx].to(device)
                short = db <= 18000.0
                score, ordinal_logits, watch_pred = model(
                    xb, short, config["duration_strength"]
                )
                point_loss = (bce_none(score, yb) * wb).mean()

                if pair_cursor < pair_limit:
                    take = min(len(idx), pair_limit - pair_cursor)
                    psel = pair_perm[pair_cursor:pair_cursor + take]
                    pair_cursor += take
                    pi = pair_pos_t[psel]
                    ni = pair_neg_t[psel]
                    xp = x[pi].to(device)
                    xn = x[ni].to(device)
                    dp = duration[pi].to(device)
                    dn = duration[ni].to(device)
                    sp, _, _ = model(xp, dp <= 18000.0, config["duration_strength"])
                    sn, _, _ = model(xn, dn <= 18000.0, config["duration_strength"])
                    pair_loss = torch.nn.functional.softplus(-(sp - sn)).mean()
                else:
                    pair_loss = point_loss.new_zeros(())

                loss = 0.5 * point_loss + 0.5 * pair_loss
                if config["ordinal_weight"] > 0:
                    ot = ordinal_targets[idx].to(device)
                    ordinal_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        ordinal_logits, ot, reduction="none"
                    ).mean(dim=1)
                    loss = loss + config["ordinal_weight"] * (ordinal_loss * wb).mean()
                if config["cwm_weight"] > 0:
                    target = watch_fraction[idx].to(device)
                    censored = completed[idx].to(device)
                    exact_loss = torch.nn.functional.smooth_l1_loss(
                        watch_pred, target, reduction="none"
                    )
                    lower_loss = torch.relu(target - watch_pred).square()
                    cwm_loss = torch.where(censored, lower_loss, exact_loss)
                    loss = loss + config["cwm_weight"] * (cwm_loss * wb).mean()

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                last_loss = float(loss.detach().cpu())

            scores = predict(model, xv, duration_v, device, config["duration_strength"])
            metrics = metric_values(evaluator(val["user"], val["y"].astype(int), scores))
            checkpoint_history.append({
                "epoch": epoch + 0.5 * (phase + 1),
                "train_loss": round(last_loss, 6),
                "val_gauc": round(metrics["gauc"], 6),
                "val_primary": round(metrics["primary"], 6),
            })
            if metrics["primary"] > best_primary + 1e-7:
                best_primary = metrics["primary"]
                best_scores = scores.copy()
                best_metrics = metrics
                stale = 0
            else:
                stale += 1

        for group in optimizer.param_groups:
            group["lr"] *= 0.5
        if not probe and stale >= 6:
            break

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best_metrics, best_scores, checkpoint_history


def base_config(name="champion"):
    return {
        "name": name,
        "ordinal_weight": 0.0,
        "cwm_weight": 0.0,
        "duration_strength": 0.0,
        "half_life": 7.0,
    }


def merge_pair(first, second, scale):
    cfg = base_config("pair")
    for source in (first, second):
        if source["ordinal_weight"] > 0:
            cfg["ordinal_weight"] = source["ordinal_weight"] * scale
        if source["cwm_weight"] > 0:
            cfg["cwm_weight"] = source["cwm_weight"] * scale
        if source["duration_strength"] > 0:
            cfg["duration_strength"] = source["duration_strength"] * scale
        if source["half_life"] != 7.0:
            cfg["half_life"] = source["half_life"]
    cfg["name"] = "pair_%s__%s_x%.1f" % (first["name"], second["name"], scale)
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=18)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train, val, fast_path = load_data(args.data_dir)
    if fast_path:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, root)
        from data.official.evaluate import evaluate as evaluator
    else:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, root)
        from harness.evaluate_provisional import evaluate as evaluator

    smoke = os.environ.get("SMOKE_EPOCHS")
    smoke_cap = int(smoke) if smoke is not None else None
    probe_epochs = 4 if device.type == "cpu" else 5
    final_epochs = args.epochs
    if smoke_cap is not None:
        probe_epochs = min(probe_epochs, smoke_cap)
        final_epochs = min(final_epochs, smoke_cap)

    pair_pos, pair_neg = make_pairs(
        np.asarray(train["user"]), np.asarray(train["y"]), args.seed
    )

    initial = [base_config()]
    for weight in (0.08, 0.15, 0.25, 0.40):
        cfg = base_config("ordinal_%.2f" % weight)
        cfg["ordinal_weight"] = weight
        initial.append(cfg)
    for weight in (0.08, 0.15, 0.25, 0.40):
        cfg = base_config("cwm_%.2f" % weight)
        cfg["cwm_weight"] = weight
        initial.append(cfg)
    for strength in (0.25, 0.50):
        cfg = base_config("duration_%.2f" % strength)
        cfg["duration_strength"] = strength
        initial.append(cfg)
    for half_life in (0.0, 3.5, 14.0):
        cfg = base_config("recency_%s" % ("uniform" if half_life == 0 else str(half_life)))
        cfg["half_life"] = half_life
        initial.append(cfg)

    if smoke_cap is not None:
        initial = initial[:5]

    history = []
    scored_initial = []
    for probe_index, cfg in enumerate(initial):
        metrics, _, checkpoints = train_one(
            cfg, train, val, pair_pos, pair_neg, evaluator, device,
            args.seed, probe_epochs, True
        )
        record = {
            "stage": "probe",
            "probe_index": probe_index,
            "config": cfg,
            "gauc": metrics["gauc"],
            "ndcg5": metrics["ndcg5"],
            "primary": metrics["primary"],
            "checkpoints": checkpoints,
        }
        history.append(record)
        scored_initial.append((metrics["primary"], cfg))

    if smoke_cap is None:
        families = {
            "ordinal": [z for z in scored_initial if z[1]["ordinal_weight"] > 0],
            "cwm": [z for z in scored_initial if z[1]["cwm_weight"] > 0],
            "duration": [z for z in scored_initial if z[1]["duration_strength"] > 0],
            "recency": [z for z in scored_initial if z[1]["half_life"] != 7.0],
        }
        representatives = [max(values, key=lambda z: z[0])[1] for values in families.values()]
        pair_configs = []
        scales = (0.6, 0.8, 1.0, 1.2, 1.4)
        for i in range(len(representatives)):
            for j in range(i + 1, len(representatives)):
                for scale in scales:
                    pair_configs.append(merge_pair(representatives[i], representatives[j], scale))

        for cfg in pair_configs:
            metrics, _, checkpoints = train_one(
                cfg, train, val, pair_pos, pair_neg, evaluator, device,
                args.seed, probe_epochs, True
            )
            record = {
                "stage": "probe",
                "probe_index": len(history),
                "config": cfg,
                "gauc": metrics["gauc"],
                "ndcg5": metrics["ndcg5"],
                "primary": metrics["primary"],
                "checkpoints": checkpoints,
            }
            history.append(record)

    winner_record = max(history, key=lambda r: r["primary"])
    winner = dict(winner_record["config"])
    final_metrics, best_scores, final_checkpoints = train_one(
        winner, train, val, pair_pos, pair_neg, evaluator, device,
        args.seed + 1000, final_epochs, False
    )
    history.append({
        "stage": "final",
        "config": winner,
        "gauc": final_metrics["gauc"],
        "ndcg5": final_metrics["ndcg5"],
        "primary": final_metrics["primary"],
        "checkpoints": final_checkpoints,
    })

    metrics_output = {
        "gauc": final_metrics["gauc"],
        "ndcg5": final_metrics["ndcg5"],
        "primary": final_metrics["primary"],
        "selected_config": winner,
        "probe_winner_primary": winner_record["primary"],
        "history": history,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics_output, fh)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(best_scores):
            writer.writerow([i, val["user"][i], val["video_out"][i], "%.8g" % float(score)])


if __name__ == "__main__":
    main()
