import argparse
import csv
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def load_npz(data_dir):
    tr = np.load(Path(data_dir) / "train.npz", allow_pickle=False)
    va = np.load(Path(data_dir) / "val.npz", allow_pickle=False)
    field_dims = np.asarray(tr["field_dims"], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1])).astype(np.int64)
    xtr = np.asarray(tr["X"], dtype=np.int64)
    xva = np.asarray(va["X"], dtype=np.int64)
    ytr = np.asarray(tr["y"], dtype=np.float32)
    yva = np.asarray(va["y"], dtype=np.float32)
    utr = np.asarray(tr["user"])
    uva = np.asarray(va["user"])
    video_local = xva[:, 1] - offsets[1]
    return {
        "xtr": xtr,
        "ytr": ytr,
        "utr": utr,
        "xva": xva,
        "yva": yva,
        "uva": uva,
        "val_video": video_local,
        "field_dims": field_dims,
        "fast": True,
    }


def read_csv_rows(path, train):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            row = {
                "user_id": r["user_id"],
                "video_id": r["video_id"],
                "author_id": r.get("author_id", "__unknown_author__"),
                "tab": r.get("tab", "0"),
                "duration_ms": float(r.get("duration_ms", 0.0) or 0.0),
                "long_view": float(r["long_view"]),
            }
            rows.append(row)
    return rows


def make_mapping(values):
    unique = sorted(set(values))
    return {v: i + 1 for i, v in enumerate(unique)}


def load_csv(data_dir):
    train_rows = read_csv_rows(Path(data_dir) / "train.csv", True)
    val_rows = read_csv_rows(Path(data_dir) / "val.csv", False)
    user_map = make_mapping([r["user_id"] for r in train_rows])
    video_map = make_mapping([r["video_id"] for r in train_rows])
    author_map = make_mapping([r["author_id"] for r in train_rows])
    tab_map = make_mapping([r["tab"] for r in train_rows])
    durations = np.asarray([r["duration_ms"] for r in train_rows], dtype=np.float64)
    quantiles = np.quantile(durations, np.linspace(0.1, 0.9, 9)) if len(durations) else np.zeros(9)
    quantiles = np.unique(quantiles)
    field_dims = np.asarray([
        len(user_map) + 1,
        len(video_map) + 1,
        len(author_map) + 1,
        len(tab_map) + 1,
        len(quantiles) + 2,
    ], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1])).astype(np.int64)

    def encode(rows):
        x = np.zeros((len(rows), 5), dtype=np.int64)
        y = np.zeros(len(rows), dtype=np.float32)
        users = []
        videos = []
        for i, r in enumerate(rows):
            local = [
                user_map.get(r["user_id"], 0),
                video_map.get(r["video_id"], 0),
                author_map.get(r["author_id"], 0),
                tab_map.get(r["tab"], 0),
                int(np.searchsorted(quantiles, r["duration_ms"], side="right")) + 1,
            ]
            x[i] = np.asarray(local, dtype=np.int64) + offsets
            y[i] = r["long_view"]
            users.append(r["user_id"])
            videos.append(r["video_id"])
        return x, y, np.asarray(users, dtype=object), np.asarray(videos, dtype=object)

    xtr, ytr, utr, _ = encode(train_rows)
    xva, yva, uva, val_video = encode(val_rows)
    return {
        "xtr": xtr,
        "ytr": ytr,
        "utr": utr,
        "xva": xva,
        "yva": yva,
        "uva": uva,
        "val_video": val_video,
        "field_dims": field_dims,
        "fast": False,
    }


class CrossLayer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(width))
        self.bias = nn.Parameter(torch.zeros(width))
        nn.init.normal_(self.weight, std=0.01)

    def forward(self, x0, x):
        scalar = torch.sum(x * self.weight, dim=1, keepdim=True)
        return x0 * scalar + self.bias + x


class DCNLite(nn.Module):
    def __init__(self, total_ids, embedding_dim=16, hidden=128, dropout=0.30):
        super().__init__()
        self.embedding = nn.Embedding(total_ids, embedding_dim)
        nn.init.normal_(self.embedding.weight, std=0.01)
        width = 5 * embedding_dim
        self.cross1 = CrossLayer(width)
        self.cross2 = CrossLayer(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.output = nn.Linear(width + hidden // 2, 1)

    def forward(self, x):
        x0 = self.embedding(x).flatten(1)
        cross = self.cross1(x0, x0)
        cross = self.cross2(x0, cross)
        deep = self.mlp(x0)
        return self.output(torch.cat((cross, deep), dim=1)).squeeze(1)


def adaptive_factors(x, field_dims):
    total = int(np.sum(field_dims))
    factors = np.zeros(total, dtype=np.float32)
    for field in (0, 2):
        ids = x[:, field]
        counts = np.bincount(ids, minlength=total).astype(np.float64)
        observed = counts[counts > 0]
        median = float(np.median(observed)) if len(observed) else 1.0
        raw = np.zeros(total, dtype=np.float64)
        mask = counts > 0
        raw[mask] = np.clip(median / counts[mask], 0.25, 12.0)
        normalizer = float(np.mean(raw[ids]))
        if normalizer > 0:
            raw /= normalizer
        factors += raw.astype(np.float32)
    return factors


def make_pairs(users, labels, seed, max_pairs=None):
    rng = np.random.default_rng(seed)
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    pos_parts = []
    neg_parts = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        idx = order[a:b]
        pos = idx[labels[idx] > 0.5]
        neg = idx[labels[idx] <= 0.5]
        if len(pos) and len(neg):
            pos_parts.append(pos)
            neg_parts.append(rng.choice(neg, size=len(pos), replace=True))
    if not pos_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    pos = np.concatenate(pos_parts).astype(np.int64)
    neg = np.concatenate(neg_parts).astype(np.int64)
    if max_pairs is not None and len(pos) > max_pairs:
        keep = rng.choice(len(pos), size=max_pairs, replace=False)
        pos, neg = pos[keep], neg[keep]
    return pos, neg


def predict(model, x, device, batch_size=16384):
    model.eval()
    out = np.empty(len(x), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            xb = torch.as_tensor(x[start:end], dtype=torch.long, device=device)
            out[start:end] = torch.sigmoid(model(xb)).detach().cpu().numpy()
    return out


def get_evaluator(fast):
    if fast:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    return evaluate


def metric_dict(evaluator, users, labels, scores):
    result = evaluator(users, labels, scores)
    return {
        "gauc": float(result["GAUC"]),
        "ndcg5": float(result["nDCG@5"]),
        "primary": float(result["primary"]),
    }


def clone_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def train_model(data, adaptive_lambda, run_seed, epochs, device, evaluator,
                sample_fraction=1.0, checkpoint_half_epochs=False):
    seed_all(run_seed)
    x_all = data["xtr"]
    y_all = data["ytr"]
    u_all = data["utr"]
    rng = np.random.default_rng(run_seed + 991)
    if sample_fraction < 0.999:
        count = max(10000, int(len(x_all) * sample_fraction))
        selected = np.sort(rng.choice(len(x_all), size=min(count, len(x_all)), replace=False))
        x = x_all[selected]
        y = y_all[selected]
        users = u_all[selected]
    else:
        x, y, users = x_all, y_all, u_all

    total_ids = int(np.sum(data["field_dims"]))
    model = DCNLite(total_ids).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0025, weight_decay=1e-4)
    factors_np = adaptive_factors(x_all, data["field_dims"])
    factors = torch.as_tensor(factors_np, dtype=torch.float32, device=device)
    pair_pos, pair_neg = make_pairs(users, y, run_seed + 177, max_pairs=len(x))
    batch_size = 4096 if device.type == "cuda" else 2048
    steps_per_epoch = max(1, math.ceil(len(x) / batch_size))
    total_steps = max(1, int(math.ceil(float(epochs) * steps_per_epoch)))
    half_steps = max(1, steps_per_epoch // 2)
    point_order = np.arange(len(x), dtype=np.int64)
    pair_order = np.arange(len(pair_pos), dtype=np.int64)
    rng.shuffle(point_order)
    if len(pair_order):
        rng.shuffle(pair_order)
    point_cursor = 0
    pair_cursor = 0
    best_state = None
    best_gauc = -1.0
    best_metrics = None
    curve = []

    for step in range(total_steps):
        epoch_index = step // steps_per_epoch
        lr = 0.0025 * (0.5 ** epoch_index)
        for group in optimizer.param_groups:
            group["lr"] = lr
        if point_cursor + batch_size > len(point_order):
            rng.shuffle(point_order)
            point_cursor = 0
        point_idx = point_order[point_cursor:point_cursor + batch_size]
        point_cursor += len(point_idx)
        xb = torch.as_tensor(x[point_idx], dtype=torch.long, device=device)
        yb = torch.as_tensor(y[point_idx], dtype=torch.float32, device=device)
        logits = model(xb)
        point_loss = F.binary_cross_entropy_with_logits(logits, yb)

        if len(pair_order):
            if pair_cursor + len(point_idx) > len(pair_order):
                rng.shuffle(pair_order)
                pair_cursor = 0
            take = pair_order[pair_cursor:pair_cursor + len(point_idx)]
            pair_cursor += len(take)
            pos_x = torch.as_tensor(x[pair_pos[take]], dtype=torch.long, device=device)
            neg_x = torch.as_tensor(x[pair_neg[take]], dtype=torch.long, device=device)
            pair_loss = F.softplus(-(model(pos_x) - model(neg_x))).mean()
        else:
            pair_loss = point_loss * 0.0

        target_ids = torch.cat((xb[:, 0], xb[:, 2]), dim=0)
        row_norm = model.embedding.weight[target_ids].pow(2).sum(dim=1)
        adaptive_penalty = (factors[target_ids] * row_norm).mean()
        loss = 0.5 * point_loss + 0.5 * pair_loss + adaptive_lambda * adaptive_penalty
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        should_check = checkpoint_half_epochs and (((step + 1) % half_steps == 0) or step + 1 == total_steps)
        if should_check:
            scores = predict(model, data["xva"], device)
            metrics = metric_dict(evaluator, data["uva"], data["yva"], scores)
            progress = float(step + 1) / float(steps_per_epoch)
            curve.append({"epoch": progress, **metrics})
            if metrics["gauc"] > best_gauc:
                best_gauc = metrics["gauc"]
                best_metrics = metrics
                best_state = clone_state(model)

    if not checkpoint_half_epochs:
        scores = predict(model, data["xva"], device)
        best_metrics = metric_dict(evaluator, data["uva"], data["yva"], scores)
        best_gauc = best_metrics["gauc"]
        best_state = clone_state(model)
        curve.append({"epoch": float(epochs), **best_metrics})

    model.load_state_dict(best_state)
    final_scores = predict(model, data["xva"], device)
    final_metrics = metric_dict(evaluator, data["uva"], data["yva"], final_scores)
    return model, final_scores, final_metrics, curve


def append_progress(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def write_predictions(path, users, videos, scores):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, (u, v, s) in enumerate(zip(users, videos, scores)):
            writer.writerow([i, u.item() if isinstance(u, np.generic) else u,
                             v.item() if isinstance(v, np.generic) else v, float(s)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.log"
    if progress_path.exists():
        progress_path.unlink()

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fast = (Path(args.data_dir) / "train.npz").exists() and (Path(args.data_dir) / "val.npz").exists()
    data = load_npz(args.data_dir) if fast else load_csv(args.data_dir)
    evaluator = get_evaluator(data["fast"])
    smoke_value = os.environ.get("SMOKE_EPOCHS")
    smoke_cap = int(smoke_value) if smoke_value is not None else None
    history = []

    if smoke_cap is not None:
        strengths = [0.003]
        probe_seeds = [args.seed]
        probe_epochs = max(1, min(3, smoke_cap))
        sample_fraction = 0.20
    else:
        strengths = [0.0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]
        seed_count = 7 if device.type == "cuda" else 5
        probe_seeds = [args.seed + 1009 * i for i in range(seed_count)]
        probe_epochs = 3
        sample_fraction = 0.35

    grouped = {value: [] for value in strengths}
    for strength in strengths:
        for probe_seed in probe_seeds:
            _, _, metrics, curve = train_model(
                data, strength, probe_seed, probe_epochs, device, evaluator,
                sample_fraction=sample_fraction, checkpoint_half_epochs=False
            )
            record = {
                "phase": "screen",
                "adaptive_lambda": strength,
                "seed": probe_seed,
                "epochs": probe_epochs,
                "sample_fraction": sample_fraction,
                **metrics,
            }
            history.append(record)
            grouped[strength].append(metrics["primary"])
            append_progress(progress_path, record)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    ranked = sorted(strengths, key=lambda z: (-float(np.mean(grouped[z])), z))
    if smoke_cap is None:
        refine_strengths = ranked[:3]
        refine_seeds = [args.seed + 7919 * i for i in range(5)]
        refine_grouped = {value: [] for value in refine_strengths}
        for strength in refine_strengths:
            for refine_seed in refine_seeds:
                _, _, metrics, curve = train_model(
                    data, strength, refine_seed, 5, device, evaluator,
                    sample_fraction=1.0, checkpoint_half_epochs=False
                )
                record = {
                    "phase": "refine",
                    "adaptive_lambda": strength,
                    "seed": refine_seed,
                    "epochs": 5,
                    "sample_fraction": 1.0,
                    **metrics,
                }
                history.append(record)
                refine_grouped[strength].append(metrics["primary"])
                append_progress(progress_path, record)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        winner = sorted(refine_strengths, key=lambda z: (-float(np.mean(refine_grouped[z])), z))[0]
        final_epochs = 7
    else:
        winner = ranked[0]
        final_epochs = max(1, min(7, smoke_cap))

    model, scores, final_metrics, final_curve = train_model(
        data, winner, args.seed, final_epochs, device, evaluator,
        sample_fraction=1.0, checkpoint_half_epochs=True
    )
    final_record = {
        "phase": "final",
        "adaptive_lambda": winner,
        "seed": args.seed,
        "epochs": final_epochs,
        "sample_fraction": 1.0,
        **final_metrics,
        "curve": final_curve,
    }
    history.append(final_record)
    append_progress(progress_path, final_record)

    write_predictions(out_dir / "predictions.csv", data["uva"], data["val_video"], scores)
    payload = {
        "gauc": final_metrics["gauc"],
        "ndcg5": final_metrics["ndcg5"],
        "primary": final_metrics["primary"],
        "selected_adaptive_lambda": winner,
        "history": history,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, sort_keys=True)


if __name__ == "__main__":
    main()
