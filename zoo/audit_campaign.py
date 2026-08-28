"""Validation-only fresh-eyes audit campaign for KuaiRand-Pure.

The runner reads only ``train.npz`` and ``val.npz`` from the exported workspace.
It implements the five-field DCN-lite control and the four audit variants behind
flags, always selecting checkpoints by the official primary metric at half-epoch
intervals.  No shared zoo or data-loader code is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.official.evaluate import evaluate as official_evaluate


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def parser(description: str = __doc__) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--data-dir", required=True,
                    help="directory containing train/val .npz and .csv; 'real' aliases data/real_ws")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--patience-halves", type=int, default=6)
    ap.add_argument("--subsample", type=int, default=None,
                    help="smoke-test cap per split; never changes the date window")
    ap.add_argument("--lambda-weight", type=float, choices=(0.0, 0.3, 0.5), default=0.0)
    ap.add_argument("--duration-heads", action="store_true")
    ap.add_argument("--tab-bias", action="store_true",
                    help="add a separate tab-conditioned bias to each duration head")
    ap.add_argument("--metadata-crosses", action="store_true")
    ap.add_argument("--session-features", action="store_true")
    ap.add_argument("--raw-data-dir", default=None,
                    help="KuaiRand-Pure data directory (needed by side-table/session variants)")
    return ap


def _resolve_data_dir(value: str) -> Path:
    return ROOT / "data" / "real_ws" if value == "real" else Path(value)


def _read_video_ids(path: Path, expected: int) -> np.ndarray:
    values = np.empty(expected, dtype=np.int64)
    with path.open(newline="", encoding="utf-8") as fh:
        rows = csv.DictReader(fh)
        for i, row in enumerate(rows):
            if i >= expected:
                raise ValueError(f"{path} has more rows than its npz")
            values[i] = int(row["video_id"])
        if i + 1 != expected:
            raise ValueError(f"{path} row mismatch: {i + 1} != {expected}")
    return values


def load_validation_only(data_dir: str, subsample: int | None = None) -> dict:
    """Load exactly the exported train and validation windows, never a test file."""
    base = _resolve_data_dir(data_dir)
    out: dict = {}
    field_dims = None
    for name, stem, max_date in (("train", "train", 20220421),
                                  ("valid", "val", 20220428)):
        npz_path = base / f"{stem}.npz"
        csv_path = base / f"{stem}.csv"
        with np.load(npz_path, allow_pickle=False) as z:
            split = {k: np.asarray(z[k]) for k in z.files if k != "field_dims"}
            dims = np.asarray(z["field_dims"], dtype=np.int64)
        if field_dims is None:
            field_dims = dims
        elif not np.array_equal(field_dims, dims):
            raise ValueError("train/validation field dimensions differ")
        if split["date"].size and int(split["date"].max()) > max_date:
            raise ValueError(f"forbidden date in {name}: {int(split['date'].max())}")
        split["users"] = split.pop("user")
        split["videos"] = _read_video_ids(csv_path, len(split["y"]))
        if subsample is not None:
            split = {k: v[:subsample] for k, v in split.items()}
        out[name] = split
    out["field_dims_total"] = int(field_dims.sum())
    return out


def _append_categorical(ds: dict, train_values: np.ndarray,
                        valid_values: np.ndarray, cardinality: int) -> None:
    offset = ds["field_dims_total"]
    for name, values in (("train", train_values), ("valid", valid_values)):
        col = np.asarray(values, dtype=np.int64) + offset
        ds[name]["X"] = np.column_stack((ds[name]["X"], col))
    ds["field_dims_total"] += int(cardinality)


def _raw_data_dir(args) -> Path:
    if args.raw_data_dir:
        return Path(args.raw_data_dir)
    return ROOT.parent / "KuaiRand-Pure" / "data"


def add_metadata_crosses(ds: dict, raw_dir: Path) -> None:
    """Add only coarse user metadata crossed with three varying contexts.

    The author bucket is the floor(log2(train exposure count)), clipped at 15.
    This deliberately avoids a raw/high-cardinality author cross.
    """
    fields = ("user_active_degree", "follow_user_num_range", "fans_user_num_range",
              "register_days_range", "is_video_author")
    raw_by_user: dict[int, tuple[str, ...]] = {}
    values = [set(["__MISSING__"]) for _ in fields]
    with (raw_dir / "user_features_pure.csv").open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            vals = tuple((row.get(f) or "__MISSING__") for f in fields)
            raw_by_user[int(row["user_id"])] = vals
            for j, value in enumerate(vals):
                values[j].add(value)
    vocabs = [{v: i for i, v in enumerate(sorted(vs))} for vs in values]

    author = ds["train"]["X"][:, 2]
    counts = np.bincount(author, minlength=ds["field_dims_total"])
    contexts: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
    regimes = tuple((ds[n]["duration_ms"] > 18_000).astype(np.int64)
                    for n in ("train", "valid"))
    contexts["duration_regime"] = (regimes[0], regimes[1], 2)
    tabs = tuple((ds[n]["X"][:, 3] - ds[n]["X"][:, 3].min()).astype(np.int64)
                 for n in ("train", "valid"))
    contexts["tab"] = (tabs[0], tabs[1], int(max(t.max(initial=0) for t in tabs)) + 1)
    author_buckets = []
    for name in ("train", "valid"):
        ids = ds[name]["X"][:, 2]
        c = np.where(ids < len(counts), counts[ids], 0)
        author_buckets.append(np.minimum(15, np.floor(np.log2(c + 1))).astype(np.int64))
    contexts["author_bucket"] = (author_buckets[0], author_buckets[1], 16)

    metadata_codes: list[tuple[np.ndarray, np.ndarray, int]] = []
    for j, vocab in enumerate(vocabs):
        miss = vocab["__MISSING__"]
        encoded = []
        for name in ("train", "valid"):
            encoded.append(np.fromiter(
                (vocab.get(raw_by_user.get(int(u), ("__MISSING__",) * len(fields))[j], miss)
                 for u in ds[name]["users"]), dtype=np.int64, count=len(ds[name]["users"])))
        metadata_codes.append((encoded[0], encoded[1], len(vocab)))

    for meta_tr, meta_va, meta_card in metadata_codes:
        for ctx_tr, ctx_va, ctx_card in contexts.values():
            _append_categorical(ds, meta_tr * ctx_card + ctx_tr,
                                meta_va * ctx_card + ctx_va, meta_card * ctx_card)


def _aligned_times(export_csv: Path, raw_csv: Path, expected: int,
                   max_date: int) -> np.ndarray:
    """Align by (user, video, date, hour-minute), occurrence order breaking ties.

    The raw exporter preserves source row order.  Matching sequentially therefore
    implements the requested row-order fallback for duplicate user/video/time keys.
    Only rows in the requested date window are yielded by the raw-side iterator;
    later rows are discarded immediately and never enter a join or feature state.
    """
    result = np.empty(expected, dtype=np.int64)
    keys = ("user_id", "video_id", "date", "hourmin")
    with export_csv.open(newline="", encoding="utf-8") as efh, \
            raw_csv.open(newline="", encoding="utf-8") as rfh:
        exported, raw = csv.DictReader(efh), csv.DictReader(rfh)
        allowed_raw = (row for row in raw if int(row["date"]) <= max_date)
        for i, erow in enumerate(exported):
            if i >= expected:
                raise ValueError(f"{export_csv} has more rows than expected")
            rrow = next(allowed_raw)
            ekey = tuple(erow[k] for k in keys)
            rkey = tuple(rrow[k] for k in keys)
            if ekey != rkey:
                raise ValueError(f"raw/export join mismatch at row {i}: {ekey} != {rkey}")
            if int(rrow["date"]) > max_date:
                raise ValueError(f"forbidden raw date at row {i}: {rrow['date']}")
            result[i] = int(rrow["time_ms"])
        if i + 1 != expected:
            raise ValueError(f"{export_csv} row mismatch: {i + 1} != {expected}")
    return result


def add_session_features(ds: dict, data_dir: Path, raw_dir: Path) -> None:
    if any(len(ds[n]["y"]) != (1_141_112 if n == "train" else 124_909)
           for n in ("train", "valid")):
        raise ValueError("session joins require the full, unsampled real export")
    train_times = _aligned_times(data_dir / "train.csv",
                                 raw_dir / "log_standard_4_08_to_4_21_pure.csv",
                                 len(ds["train"]["y"]), 20220421)
    valid_times = _aligned_times(data_dir / "val.csv",
                                 raw_dir / "log_standard_4_22_to_5_08_pure.csv",
                                 len(ds["valid"]["y"]), 20220428)
    users = np.concatenate((ds["train"]["users"], ds["valid"]["users"]))
    times = np.concatenate((train_times, valid_times))
    split_at = len(train_times)
    order = np.lexsort((np.arange(len(times)), times, users))
    gap_bucket = np.empty(len(times), dtype=np.int64)
    session_index = np.empty(len(times), dtype=np.int64)
    session_start = np.empty(len(times), dtype=np.int64)
    last_user = last_time = -1
    current_index = 0
    gap_edges = np.asarray([1_000, 5_000, 30_000, 120_000, 600_000, 1_800_000], dtype=np.int64)
    for idx in order:
        user, now = int(users[idx]), int(times[idx])
        new = user != last_user or now - last_time > 1_800_000
        gap = np.iinfo(np.int64).max if user != last_user else max(0, now - last_time)
        current_index = 0 if new else current_index + 1
        gap_bucket[idx] = int(np.searchsorted(gap_edges, gap, side="right"))
        session_index[idx] = min(current_index, 31)
        session_start[idx] = int(new)
        last_user, last_time = user, now
    for values, cardinality in ((gap_bucket, 7), (session_index, 32), (session_start, 2)):
        _append_categorical(ds, values[:split_at], values[split_at:], cardinality)


class PairSampler:
    def __init__(self, users: np.ndarray, labels: np.ndarray):
        self.rng_groups: list[np.ndarray] = []
        pos_parts, neg_parts, starts, counts = [], [], [], []
        offset = 0
        order = np.argsort(users, kind="stable")
        boundaries = np.flatnonzero(np.r_[True, users[order][1:] != users[order][:-1], True])
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            idx = order[start:end]
            if len(idx) >= 3:
                self.rng_groups.append(idx)
            pos, neg = idx[labels[idx] == 1], idx[labels[idx] == 0]
            if len(pos) and len(neg):
                pos_parts.append(pos)
                neg_parts.append(neg)
                starts.append(np.full(len(pos), offset, dtype=np.int64))
                counts.append(np.full(len(pos), len(neg), dtype=np.int64))
                offset += len(neg)
        self.pos = np.concatenate(pos_parts) if pos_parts else np.empty(0, dtype=np.int64)
        self.neg = np.concatenate(neg_parts) if neg_parts else np.empty(0, dtype=np.int64)
        self.starts = np.concatenate(starts) if starts else np.empty(0, dtype=np.int64)
        self.counts = np.concatenate(counts) if counts else np.empty(0, dtype=np.int64)

    def pairs(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        neg = self.neg[self.starts + rng.integers(0, self.counts)]
        perm = rng.permutation(len(self.pos))
        return self.pos[perm], neg[perm]

    def lambda_groups(self, rng: np.random.Generator, count: int = 64) -> list[np.ndarray]:
        chosen = rng.integers(0, len(self.rng_groups), size=count)
        result = []
        for group_id in chosen:
            rows = self.rng_groups[int(group_id)]
            size = int(rng.integers(3, min(8, len(rows)) + 1))
            result.append(rng.choice(rows, size=size, replace=False))
        return result


class DCNAudit(nn.Module):
    def __init__(self, dim: int, n_fields: int, k: int = 16,
                 duration_heads: bool = False, tab_bias: bool = False):
        super().__init__()
        self.duration_heads = duration_heads
        self.use_tab_bias = tab_bias
        self.emb = nn.Embedding(dim, k)
        nn.init.normal_(self.emb.weight, std=0.01)
        width = n_fields * k
        self.cross = nn.Linear(width, width)
        self.mlp = nn.Sequential(nn.Linear(width, 128), nn.ReLU(), nn.Dropout(0.1))
        self.heads = nn.ModuleList([nn.Linear(128, 1) for _ in range(2 if duration_heads else 1)])
        self.tab_bias = nn.Embedding(dim, 2) if tab_bias else None
        if self.tab_bias is not None:
            nn.init.zeros_(self.tab_bias.weight)

    def forward(self, x: torch.Tensor, regime: torch.Tensor | None = None) -> torch.Tensor:
        x0 = self.emb(x).flatten(1)
        hidden = self.mlp(x0 * self.cross(x0) + x0)
        if not self.duration_heads:
            return self.heads[0](hidden).squeeze(1)
        if regime is None:
            raise ValueError("duration regime required by two-head model")
        short = self.heads[0](hidden).squeeze(1)
        long = self.heads[1](hidden).squeeze(1)
        logits = torch.where(regime.bool(), long, short)
        if self.tab_bias is not None:
            biases = self.tab_bias(x[:, 3])
            logits = logits + biases.gather(1, regime.long().unsqueeze(1)).squeeze(1)
        return logits


def _metrics(users: np.ndarray, labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    result = official_evaluate(users.tolist(), labels.tolist(), scores.tolist())
    return {"gauc": float(result["GAUC"]), "ndcg5": float(result["nDCG@5"]),
            "primary": float(result["primary"])}


def _lambda_ndcg_loss(model: DCNAudit, X: torch.Tensor, y: torch.Tensor,
                      regime: torch.Tensor, groups: list[np.ndarray]) -> torch.Tensor:
    group_losses = []
    discounts = torch.tensor([1 / math.log2(i + 2) if i < 5 else 0.0 for i in range(8)])
    for idx_np in groups:
        idx = torch.as_tensor(idx_np, dtype=torch.long)
        scores = model(X[idx], regime[idx])
        labels = y[idx]
        positives = int(labels.sum().item())
        if positives == 0 or positives == len(labels):
            continue
        ideal = sum(1 / math.log2(i + 2) for i in range(min(5, positives)))
        rank_order = torch.argsort(scores.detach(), descending=True)
        ranks = torch.empty_like(rank_order)
        ranks[rank_order] = torch.arange(len(labels))
        pair_losses, weights = [], []
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                if labels[i] == labels[j]:
                    continue
                pos, neg = (i, j) if labels[i] > labels[j] else (j, i)
                weight = abs(float(discounts[ranks[pos]] - discounts[ranks[neg]])) / ideal
                if weight > 0:
                    pair_losses.append(nn.functional.softplus(scores[neg] - scores[pos]) * weight)
                    weights.append(weight)
        if pair_losses:
            group_losses.append(torch.stack(pair_losses).sum() / sum(weights))
    if not group_losses:
        return model.emb.weight.sum() * 0.0
    return torch.stack(group_losses).mean()


def train_and_report(model: DCNAudit, ds: dict, args) -> dict:
    started = time.time()
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    tr, va = ds["train"], ds["valid"]
    Xtr = torch.as_tensor(np.ascontiguousarray(tr["X"]), dtype=torch.long)
    ytr = torch.as_tensor(tr["y"], dtype=torch.float32)
    Xva = torch.as_tensor(np.ascontiguousarray(va["X"]), dtype=torch.long)
    rtr = torch.as_tensor(tr["duration_ms"] > 18_000, dtype=torch.bool)
    rva = torch.as_tensor(va["duration_ms"] > 18_000, dtype=torch.bool)
    sampler = PairSampler(tr["users"], tr["y"])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()

    def predict() -> np.ndarray:
        model.eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, len(Xva), 200_000):
                chunks.append(model(Xva[start:start + 200_000], rva[start:start + 200_000]))
        model.train()
        return torch.cat(chunks).numpy()

    point_order = np.arange(len(ytr))
    half_size = math.ceil(len(ytr) / 2)
    best_primary, best_state, bad = -1.0, None, 0
    history: list[dict] = []
    stop = False
    for epoch in range(1, args.epochs + 1):
        point_order = rng.permutation(point_order)
        pair_pos, pair_neg = sampler.pairs(rng)
        for half in range(2):
            lo, hi = half * half_size, min(len(ytr), (half + 1) * half_size)
            if lo >= hi:
                continue
            half_rows = point_order[lo:hi]
            n_batches = math.ceil(len(half_rows) / args.batch_size)
            losses = []
            for batch in range(n_batches):
                idx_np = half_rows[batch * args.batch_size:(batch + 1) * args.batch_size]
                idx = torch.as_tensor(idx_np, dtype=torch.long)
                logits = model(Xtr[idx], rtr[idx])
                point_loss = bce(logits, ytr[idx])
                pair_lo = ((half * n_batches + batch) * len(pair_pos)) // (2 * n_batches)
                pair_hi = ((half * n_batches + batch + 1) * len(pair_pos)) // (2 * n_batches)
                if pair_hi > pair_lo:
                    p = torch.as_tensor(pair_pos[pair_lo:pair_hi], dtype=torch.long)
                    n = torch.as_tensor(pair_neg[pair_lo:pair_hi], dtype=torch.long)
                    bpr = nn.functional.softplus(
                        model(Xtr[n], rtr[n]) - model(Xtr[p], rtr[p])).mean()
                else:
                    bpr = point_loss * 0.0
                hybrid = 0.5 * point_loss + 0.5 * bpr
                if args.lambda_weight:
                    lambda_loss = _lambda_ndcg_loss(
                        model, Xtr, ytr, rtr, sampler.lambda_groups(rng))
                    loss = (1 - args.lambda_weight) * hybrid + args.lambda_weight * lambda_loss
                else:
                    loss = hybrid
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            scores = predict()
            metrics = _metrics(va["users"], va["y"], scores)
            entry = {"epoch": epoch - 0.5 + 0.5 * half,
                     "train_loss": float(np.mean(losses)),
                     "val_gauc": metrics["gauc"], "val_primary": metrics["primary"]}
            history.append(entry)
            print(f"epoch {entry['epoch']:.1f} | loss {entry['train_loss']:.4f} | "
                  f"valid gauc {metrics['gauc']:.4f} primary {metrics['primary']:.4f}", flush=True)
            if metrics["primary"] > best_primary + 1e-5:
                best_primary, bad = metrics["primary"], 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= args.patience_halves:
                    stop = True
                    break
        if stop:
            print(f"early stop after epoch {history[-1]['epoch']:.1f}", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    scores = predict()
    metrics = _metrics(va["users"], va["y"], scores)
    metrics["history"] = history
    metrics["runtime_s"] = round(time.time() - started, 1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(("row_id", "user_id", "video_id", "score"))
        for row_id, (user, video, score) in enumerate(zip(va["users"], va["videos"], scores)):
            writer.writerow((row_id, int(user), int(video), f"{float(score):.10f}"))
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, sort_keys=True)
        fh.write("\n")
    print("final:", {k: v for k, v in metrics.items() if k != "history"}, flush=True)
    return metrics


def run(args) -> dict:
    ds = load_validation_only(args.data_dir, args.subsample)
    data_dir = _resolve_data_dir(args.data_dir)
    raw_dir = _raw_data_dir(args)
    if args.metadata_crosses:
        add_metadata_crosses(ds, raw_dir)
    if args.session_features:
        add_session_features(ds, data_dir, raw_dir)
    set_seed(args.seed)
    model = DCNAudit(ds["field_dims_total"], ds["train"]["X"].shape[1], args.k,
                     duration_heads=args.duration_heads, tab_bias=args.tab_bias)
    return train_and_report(model, ds, args)


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
