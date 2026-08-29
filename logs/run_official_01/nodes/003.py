import argparse
import contextlib
import csv
import io
import json
import os
import random
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def scalar_text(x):
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


def load_npz(data_dir):
    tr = np.load(Path(data_dir) / "train.npz", allow_pickle=False)
    va = np.load(Path(data_dir) / "val.npz", allow_pickle=False)
    xtr = np.asarray(tr["X"], dtype=np.int64)
    xva = np.asarray(va["X"], dtype=np.int64)
    ytr = np.asarray(tr["y"], dtype=np.float32)
    yva = np.asarray(va["y"], dtype=np.float32)
    utr = np.asarray(tr["user"])
    uva = np.asarray(va["user"])
    dtr = np.asarray(tr["duration_ms"], dtype=np.float32)
    dva = np.asarray(va["duration_ms"], dtype=np.float32)
    field_dims = np.asarray(tr["field_dims"], dtype=np.int64).reshape(-1)
    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1], dtype=np.int64)))
    video_out = xva[:, 1] - offsets[1]
    n_tokens = int(max(xtr.max(initial=0), xva.max(initial=0)) + 1)
    return {
        "xtr": xtr, "xva": xva, "ytr": ytr, "yva": yva,
        "utr": utr, "uva": uva, "dtr": dtr, "dva": dva,
        "video_out": video_out, "n_tokens": n_tokens, "npz": True
    }


def read_csv_rows(path, training):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            item = {
                "user_id": r["user_id"],
                "video_id": r["video_id"],
                "author_id": r.get("author_id", "__missing_author__"),
                "tab": r["tab"],
                "duration_ms": float(r["duration_ms"] or 0.0),
                "long_view": float(r["long_view"]),
            }
            rows.append(item)
    return rows


def load_csv(data_dir):
    train_rows = read_csv_rows(Path(data_dir) / "train.csv", True)
    val_rows = read_csv_rows(Path(data_dir) / "val.csv", False)
    durations = np.asarray([r["duration_ms"] for r in train_rows], dtype=np.float64)
    edges = np.unique(np.quantile(durations, np.linspace(0.1, 0.9, 9)))
    names = ["user_id", "video_id", "author_id", "tab"]
    maps = []
    dims = []
    for name in names:
        values = sorted({r[name] for r in train_rows})
        mapping = {v: i + 1 for i, v in enumerate(values)}
        maps.append(mapping)
        dims.append(len(mapping) + 1)
    dims.append(len(edges) + 1)
    offsets = np.concatenate(([0], np.cumsum(np.asarray(dims[:-1], dtype=np.int64))))

    def encode(rows):
        x = np.empty((len(rows), 5), dtype=np.int64)
        for j, name in enumerate(names):
            x[:, j] = np.fromiter((maps[j].get(r[name], 0) for r in rows), dtype=np.int64, count=len(rows)) + offsets[j]
        db = np.searchsorted(edges, np.fromiter((r["duration_ms"] for r in rows), dtype=np.float64, count=len(rows)), side="right")
        x[:, 4] = db + offsets[4]
        return x

    xtr = encode(train_rows)
    xva = encode(val_rows)
    return {
        "xtr": xtr,
        "xva": xva,
        "ytr": np.asarray([r["long_view"] for r in train_rows], dtype=np.float32),
        "yva": np.asarray([r["long_view"] for r in val_rows], dtype=np.float32),
        "utr": np.asarray([r["user_id"] for r in train_rows], dtype=object),
        "uva": np.asarray([r["user_id"] for r in val_rows], dtype=object),
        "dtr": np.asarray([r["duration_ms"] for r in train_rows], dtype=np.float32),
        "dva": np.asarray([r["duration_ms"] for r in val_rows], dtype=np.float32),
        "video_out": np.asarray([r["video_id"] for r in val_rows], dtype=object),
        "n_tokens": int(np.sum(dims)),
        "npz": False,
    }


class CrossLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.weight, std=0.01)

    def forward(self, x0, x):
        scale = torch.sum(x * self.weight, dim=1, keepdim=True)
        return x0 * scale + self.bias + x


class DurationRegimeDCN(nn.Module):
    def __init__(self, n_tokens, fields=5, emb_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(n_tokens, emb_dim)
        nn.init.normal_(self.embedding.weight, std=0.01)
        dim = fields * emb_dim
        self.cross1 = CrossLayer(dim)
        self.cross2 = CrossLayer(dim)
        self.deep = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
        )
        rep_dim = dim + 128
        self.shared_head = nn.Linear(rep_dim, 1)
        self.short_residual = nn.Linear(rep_dim, 1)
        self.long_residual = nn.Linear(rep_dim, 1)
        nn.init.zeros_(self.short_residual.weight)
        nn.init.zeros_(self.short_residual.bias)
        nn.init.zeros_(self.long_residual.weight)
        nn.init.zeros_(self.long_residual.bias)

    def forward(self, x, duration_ms):
        e = self.embedding(x).flatten(1)
        cross = self.cross1(e, e)
        cross = self.cross2(e, cross)
        rep = torch.cat((cross, self.deep(e)), dim=1)
        shared = self.shared_head(rep).squeeze(1)
        short_delta = self.short_residual(rep).squeeze(1)
        long_delta = self.long_residual(rep).squeeze(1)
        is_short = duration_ms <= 18000.0
        delta = torch.where(is_short, short_delta, long_delta)
        return shared + delta

    def regime_penalty(self):
        return (self.short_residual.weight.square().mean() +
                self.long_residual.weight.square().mean() +
                self.short_residual.bias.square().mean() +
                self.long_residual.bias.square().mean())


def make_pairs(users, labels, seed):
    rng = np.random.default_rng(seed)
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    positives = []
    negatives = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        idx = order[a:b]
        pos = idx[labels[idx] > 0.5]
        neg = idx[labels[idx] <= 0.5]
        if len(pos) == 0 or len(neg) == 0:
            continue
        m = min(len(pos), len(neg))
        positives.append(rng.choice(pos, size=m, replace=False))
        negatives.append(rng.choice(neg, size=m, replace=False))
    if not positives:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(positives).astype(np.int64), np.concatenate(negatives).astype(np.int64)


def predict(model, x, duration, device, batch_size=32768):
    model.eval()
    result = np.empty(len(x), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            xb = torch.as_tensor(x[start:end], dtype=torch.long, device=device)
            db = torch.as_tensor(duration[start:end], dtype=torch.float32, device=device)
            result[start:end] = torch.sigmoid(model(xb, db)).cpu().numpy()
    return result


def quiet_evaluate(evaluator, users, labels, scores):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return evaluator(users, labels, scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if (data_dir / "train.npz").is_file() and (data_dir / "val.npz").is_file():
        data = load_npz(data_dir)
        from data.official.evaluate import evaluate as evaluator
    else:
        data = load_csv(data_dir)
        from harness.evaluate_provisional import evaluate as evaluator

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DurationRegimeDCN(data["n_tokens"]).to(device)
    embedding_params = list(model.embedding.parameters())
    embedding_ids = {id(p) for p in embedding_params}
    dense_params = [p for p in model.parameters() if id(p) not in embedding_ids]
    optimizer = torch.optim.AdamW([
        {"params": embedding_params, "weight_decay": 0.0},
        {"params": dense_params, "weight_decay": 1e-4},
    ], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.65)

    max_epochs = 7
    smoke = os.environ.get("SMOKE_EPOCHS")
    if smoke is not None:
        max_epochs = min(max_epochs, max(1, int(smoke)))
    batch_size = 8192
    rng = np.random.default_rng(args.seed)
    pair_pos, pair_neg = make_pairs(data["utr"], data["ytr"], args.seed + 17)
    best_gauc = -float("inf")
    best_state = None
    stale = 0

    for epoch in range(max_epochs):
        model.train()
        permutation = rng.permutation(len(data["xtr"]))
        for start in range(0, len(permutation), batch_size):
            batch_idx = permutation[start:start + batch_size]
            xb = torch.as_tensor(data["xtr"][batch_idx], dtype=torch.long, device=device)
            yb = torch.as_tensor(data["ytr"][batch_idx], dtype=torch.float32, device=device)
            db = torch.as_tensor(data["dtr"][batch_idx], dtype=torch.float32, device=device)
            logits = model(xb, db)
            point_loss = F.binary_cross_entropy_with_logits(logits, yb)

            if len(pair_pos):
                take = rng.integers(0, len(pair_pos), size=len(batch_idx))
                pi = pair_pos[take]
                ni = pair_neg[take]
                px = torch.as_tensor(data["xtr"][pi], dtype=torch.long, device=device)
                nx = torch.as_tensor(data["xtr"][ni], dtype=torch.long, device=device)
                pd = torch.as_tensor(data["dtr"][pi], dtype=torch.float32, device=device)
                nd = torch.as_tensor(data["dtr"][ni], dtype=torch.float32, device=device)
                pair_loss = F.softplus(-(model(px, pd) - model(nx, nd))).mean()
            else:
                pair_loss = point_loss

            embedding_penalty = model.embedding(xb).square().mean()
            loss = 0.5 * point_loss + 0.5 * pair_loss
            loss = loss + 1e-6 * embedding_penalty + 1e-3 * model.regime_penalty()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        scheduler.step()
        val_scores = predict(model, data["xva"], data["dva"], device)
        metrics = quiet_evaluate(evaluator, data["uva"], data["yva"], val_scores)
        gauc = float(metrics["GAUC"])
        if gauc > best_gauc + 1e-7:
            best_gauc = gauc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 2 and epoch >= 3:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    scores = predict(model, data["xva"], data["dva"], device)
    metrics = quiet_evaluate(evaluator, data["uva"], data["yva"], scores)

    with open(out_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, (u, v, score) in enumerate(zip(data["uva"], data["video_out"], scores)):
            writer.writerow([i, scalar_text(u), scalar_text(v), format(float(score), ".9g")])

    output_metrics = {
        "gauc": float(metrics["GAUC"]),
        "ndcg5": float(metrics["nDCG@5"]),
        "primary": float(metrics["primary"]),
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(output_metrics, f, separators=(",", ":"), sort_keys=True)


if __name__ == "__main__":
    main()
