import argparse
import contextlib
import csv
import io
import json
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
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_npz(data_dir):
    tr = np.load(Path(data_dir) / "train.npz", allow_pickle=False)
    va = np.load(Path(data_dir) / "val.npz", allow_pickle=False)
    field_dims = np.asarray(tr["field_dims"], dtype=np.int64).reshape(-1)[:5]
    xtr = np.asarray(tr["X"], dtype=np.int64)[:, :5]
    xva = np.asarray(va["X"], dtype=np.int64)[:, :5]
    ytr = np.asarray(tr["y"], dtype=np.float32).reshape(-1)
    yva = np.asarray(va["y"], dtype=np.float32).reshape(-1)
    utr = np.asarray(tr["user"]).reshape(-1)
    uva = np.asarray(va["user"]).reshape(-1)
    dtr = np.asarray(tr["duration_ms"], dtype=np.float32).reshape(-1)
    dva = np.asarray(va["duration_ms"], dtype=np.float32).reshape(-1)
    video_offset = int(field_dims[0])
    vva = xva[:, 1] - video_offset
    return xtr, ytr, utr, dtr, xva, yva, uva, dva, vva, field_dims, True


def read_csv_rows(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_float(value, default=0.0):
    try:
        x = float(value)
        return x if np.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def load_csv(data_dir):
    train_rows = read_csv_rows(Path(data_dir) / "train.csv")
    val_rows = read_csv_rows(Path(data_dir) / "val.csv")
    durations = np.asarray([safe_float(r.get("duration_ms")) for r in train_rows], dtype=np.float32)
    quantiles = np.quantile(durations, np.linspace(0.1, 0.9, 9)) if len(durations) else np.zeros(9)

    def raw_fields(row):
        video = row.get("video_id", "")
        author = row.get("author_id", video)
        dur = safe_float(row.get("duration_ms"))
        bucket = int(np.searchsorted(quantiles, dur, side="right"))
        return [row.get("user_id", ""), video, author, row.get("tab", ""), str(bucket)]

    mappings = []
    for j in range(5):
        vals = sorted({raw_fields(r)[j] for r in train_rows})
        mappings.append({v: i for i, v in enumerate(vals)})
    field_dims = np.asarray([len(m) + 1 for m in mappings], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1])))

    def encode(rows):
        x = np.empty((len(rows), 5), dtype=np.int64)
        for i, row in enumerate(rows):
            vals = raw_fields(row)
            for j, val in enumerate(vals):
                x[i, j] = offsets[j] + mappings[j].get(val, len(mappings[j]))
        return x

    xtr = encode(train_rows)
    xva = encode(val_rows)
    ytr = np.asarray([safe_float(r.get("long_view")) for r in train_rows], dtype=np.float32)
    yva = np.asarray([safe_float(r.get("long_view")) for r in val_rows], dtype=np.float32)
    utr = np.asarray([r.get("user_id", "") for r in train_rows])
    uva = np.asarray([r.get("user_id", "") for r in val_rows])
    dtr = durations
    dva = np.asarray([safe_float(r.get("duration_ms")) for r in val_rows], dtype=np.float32)
    vva = np.asarray([r.get("video_id", "") for r in val_rows])
    return xtr, ytr, utr, dtr, xva, yva, uva, dva, vva, field_dims, False


def make_pairs(users, labels, seed):
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    rng = np.random.default_rng(seed)
    pos_parts = []
    neg_parts = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        idx = order[a:b]
        pos = idx[labels[idx] > 0.5]
        neg = idx[labels[idx] <= 0.5]
        if len(pos) and len(neg):
            pos_parts.append(pos)
            neg_parts.append(neg[rng.integers(0, len(neg), size=len(pos))])
    if not pos_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(pos_parts), np.concatenate(neg_parts)


class CrossLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.weight, std=0.01)

    def forward(self, x0, x):
        return x0 * torch.sum(x * self.weight, dim=1, keepdim=True) + self.bias + x


class DurationRegimeDCN(nn.Module):
    def __init__(self, field_dims, embed_dim=16):
        super().__init__()
        total = int(np.sum(field_dims))
        input_dim = len(field_dims) * embed_dim
        self.embedding = nn.Embedding(total, embed_dim)
        nn.init.normal_(self.embedding.weight, std=0.01)
        self.cross1 = CrossLayer(input_dim)
        self.cross2 = CrossLayer(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
        )
        rep_dim = input_dim + 128
        self.shared_head = nn.Linear(rep_dim, 1)
        self.short_residual = nn.Linear(rep_dim, 1, bias=True)
        self.long_residual = nn.Linear(rep_dim, 1, bias=True)
        nn.init.zeros_(self.short_residual.weight)
        nn.init.zeros_(self.short_residual.bias)
        nn.init.zeros_(self.long_residual.weight)
        nn.init.zeros_(self.long_residual.bias)

    def forward(self, x, duration_ms):
        x0 = self.embedding(x).flatten(1)
        cross = self.cross1(x0, x0)
        cross = self.cross2(x0, cross)
        deep = self.mlp(x0)
        rep = torch.cat([cross, deep], dim=1)
        shared = self.shared_head(rep).squeeze(1)
        short_delta = self.short_residual(rep).squeeze(1)
        long_delta = self.long_residual(rep).squeeze(1)
        is_short = duration_ms <= 18000.0
        delta = torch.where(is_short, short_delta, long_delta)
        return shared + delta, delta


def predict(model, x, duration, device, batch_size=32768):
    model.eval()
    out = np.empty(len(x), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            xb = torch.as_tensor(x[start:end], dtype=torch.long, device=device)
            db = torch.as_tensor(duration[start:end], dtype=torch.float32, device=device)
            logits, _ = model(xb, db)
            out[start:end] = torch.sigmoid(logits).cpu().numpy()
    return out


def official_metrics(npz_mode, users, labels, scores):
    if npz_mode:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        result = evaluate(users, labels, scores)
    return {
        "gauc": float(result["GAUC"]),
        "ndcg5": float(result["nDCG@5"]),
        "primary": float(result["primary"]),
    }


def clone_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def main():
    args = parse_args()
    seed_all(args.seed)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (data_dir / "train.npz").exists() and (data_dir / "val.npz").exists():
        data = load_npz(data_dir)
    else:
        data = load_csv(data_dir)
    xtr, ytr, utr, dtr, xva, yva, uva, dva, vva, field_dims, npz_mode = data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DurationRegimeDCN(field_dims).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.70)
    pair_pos, pair_neg = make_pairs(utr, ytr, args.seed)
    rng = np.random.default_rng(args.seed)
    epochs = 7
    smoke = os.environ.get("SMOKE_EPOCHS")
    if smoke is not None:
        epochs = min(epochs, max(1, int(smoke)))
    batch_size = 8192
    best_gauc = -np.inf
    best_state = clone_state(model)
    stale_checks = 0
    stop = False

    for epoch in range(epochs):
        impression_order = rng.permutation(len(xtr))
        pair_order = rng.permutation(len(pair_pos)) if len(pair_pos) else np.empty(0, dtype=np.int64)
        imp_halves = np.array_split(impression_order, 2)
        pair_halves = np.array_split(pair_order, 2)
        for half in range(2):
            model.train()
            imp_idx = imp_halves[half]
            p_idx = pair_halves[half]
            steps = max((len(imp_idx) + batch_size - 1) // batch_size,
                        (len(p_idx) + batch_size - 1) // batch_size, 1)
            for step in range(steps):
                ib = imp_idx[step * batch_size:(step + 1) * batch_size]
                pb = p_idx[step * batch_size:(step + 1) * batch_size]
                optimizer.zero_grad(set_to_none=True)
                losses = []
                shrink_terms = []
                if len(ib):
                    xb = torch.as_tensor(xtr[ib], dtype=torch.long, device=device)
                    yb = torch.as_tensor(ytr[ib], dtype=torch.float32, device=device)
                    db = torch.as_tensor(dtr[ib], dtype=torch.float32, device=device)
                    logits, delta = model(xb, db)
                    losses.append(0.5 * F.binary_cross_entropy_with_logits(logits, yb))
                    shrink_terms.append(delta.square().mean())
                if len(pb):
                    pi = pair_pos[pb]
                    ni = pair_neg[pb]
                    xp = torch.as_tensor(xtr[pi], dtype=torch.long, device=device)
                    xn = torch.as_tensor(xtr[ni], dtype=torch.long, device=device)
                    dp = torch.as_tensor(dtr[pi], dtype=torch.float32, device=device)
                    dn = torch.as_tensor(dtr[ni], dtype=torch.float32, device=device)
                    sp, delta_p = model(xp, dp)
                    sn, delta_n = model(xn, dn)
                    losses.append(0.5 * F.softplus(-(sp - sn)).mean())
                    shrink_terms.extend([delta_p.square().mean(), delta_n.square().mean()])
                if not losses:
                    continue
                loss = sum(losses)
                if shrink_terms:
                    loss = loss + 0.02 * torch.stack(shrink_terms).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            val_scores = predict(model, xva, dva, device)
            metrics = official_metrics(npz_mode, uva, yva, val_scores)
            if metrics["gauc"] > best_gauc + 1.0e-7:
                best_gauc = metrics["gauc"]
                best_state = clone_state(model)
                stale_checks = 0
            else:
                stale_checks += 1
            if stale_checks >= 5:
                stop = True
                break
        scheduler.step()
        if stop:
            break

    model.load_state_dict(best_state)
    scores = predict(model, xva, dva, device)
    metrics = official_metrics(npz_mode, uva, yva, scores)
    with open(out_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, (user_id, video_id, score) in enumerate(zip(uva, vva, scores)):
            writer.writerow([i, user_id.item() if isinstance(user_id, np.generic) else user_id,
                             video_id.item() if isinstance(video_id, np.generic) else video_id,
                             float(score)])
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
