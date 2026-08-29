"""Parameterized validation-only runner for the frozen five-field DCN-lite stack.

The defaults exactly describe ``ablate_fields.py --field-level 0 --regularized``.
Only train/validation exports are opened, and every checkpoint is selected with
the official scorer's primary metric.
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import json
import math
import random
import signal
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.official.evaluate import evaluate as official_evaluate

FROZEN_BASELINE = 0.6047
TRAIN_MIN_DATE, TRAIN_MAX_DATE = 20220408, 20220421
VALID_MIN_DATE, VALID_MAX_DATE = 20220422, 20220428
FROZEN_FIELDS = ("user", "video", "author", "tab", "dur_bucket10")


def parser(description: str = __doc__) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--data-dir", default="data/real_ws")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, choices=(4096, 8192, 16384), default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--step-decay-factor", type=float, default=0.5)
    ap.add_argument("--decay-every", type=float, choices=(0.5, 1.0, 1.5), default=1.0,
                    help="step-decay interval in epochs")
    ap.add_argument("--decay-start-epoch", type=float, default=0.0,
                    help="delay decay until this many epochs have completed")
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--embedding-dropout", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--k", type=int, choices=(8, 12, 16, 24, 32, 48, 64, 96), default=16)
    ap.add_argument("--recency-half-life", type=float, default=7.0)
    ap.add_argument("--bpr-weight", type=float, default=0.5)
    ap.add_argument("--patience-halves", type=int, default=6)
    ap.add_argument("--max-runtime", type=int, default=350)
    ap.add_argument("--subsample", type=int, default=None,
                    help="smoke-test row cap per split; date guards remain active")
    return ap


def baseline_args(out_dir: str = "/tmp/polish-baseline") -> argparse.Namespace:
    """Return the canonical defaults without consulting process argv."""
    return parser().parse_args(["--out-dir", out_dir])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _resolve_data_dir(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _read_ids(path: Path, expected: int) -> tuple[np.ndarray, np.ndarray]:
    videos = np.empty(expected, dtype=np.int64)
    authors = np.empty(expected, dtype=np.int64)
    count = 0
    with path.open(newline="", encoding="utf-8") as fh:
        for count, row in enumerate(csv.DictReader(fh), 1):
            if count > expected:
                raise ValueError(f"{path} contains more rows than its npz")
            videos[count - 1], authors[count - 1] = int(row["video_id"]), int(row["author_id"])
    if count != expected:
        raise ValueError(f"{path} row count {count} != {expected}")
    return videos, authors


def load_validation_only(data_dir: str, subsample: int | None = None) -> dict[str, Any]:
    base = _resolve_data_dir(data_dir)
    result: dict[str, Any] = {"field_names": list(FROZEN_FIELDS)}
    field_dims = None
    for name, stem, low, high in (
        ("train", "train", TRAIN_MIN_DATE, TRAIN_MAX_DATE),
        ("valid", "val", VALID_MIN_DATE, VALID_MAX_DATE),
    ):
        with np.load(base / f"{stem}.npz", allow_pickle=False) as archive:
            split = {key: np.asarray(archive[key]).copy() for key in archive.files
                     if key != "field_dims"}
            dims = np.asarray(archive["field_dims"], dtype=np.int64)
        if field_dims is None:
            field_dims = dims
        elif not np.array_equal(field_dims, dims):
            raise ValueError("train and validation field dimensions differ")
        dates = split["date"]
        if dates.size and (int(dates.min()) < low or int(dates.max()) > high):
            raise ValueError(f"forbidden date in {name}: {dates.min()}..{dates.max()}")
        videos, authors = _read_ids(base / f"{stem}.csv", len(split["y"]))
        split["users"] = split.pop("user").astype(np.int64)
        split["videos"], split["authors"] = videos, authors
        if subsample is not None:
            split = {key: values[:subsample] for key, values in split.items()}
        split["X"] = split["X"].astype(np.int64)
        split["y"] = split["y"].astype(np.float32)
        if split["X"].shape[1] != len(FROZEN_FIELDS):
            raise ValueError(f"expected five frozen fields, got {split['X'].shape[1]}")
        result[name] = split
    assert field_dims is not None
    result["field_dims_total"] = int(field_dims.sum())
    return result


def recency_weights(dates: np.ndarray, half_life_days: float) -> np.ndarray:
    if half_life_days <= 0:
        raise ValueError("recency half-life must be positive")
    end = dt.date(2022, 4, 21)
    ages = np.asarray([(end - dt.date(int(v) // 10000, int(v) // 100 % 100,
                                    int(v) % 100)).days for v in dates])
    if np.any(ages < 0):
        raise ValueError("recency weighting saw a post-training date")
    weights = np.exp2(-ages / half_life_days).astype(np.float32)
    return weights / weights.mean()


class PairSampler:
    def __init__(self, users: np.ndarray, labels: np.ndarray):
        order = np.argsort(users, kind="stable")
        sorted_users, sorted_labels = users[order], labels[order]
        starts = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1]])
        ends = np.r_[starts[1:], len(order)]
        positives, negatives, offsets, counts = [], [], [], []
        offset = 0
        for start, end in zip(starts, ends):
            indices = order[start:end]
            pos, neg = indices[sorted_labels[start:end] == 1], indices[sorted_labels[start:end] == 0]
            if len(pos) and len(neg):
                positives.append(pos)
                negatives.append(neg)
                offsets.append(np.full(len(pos), offset, dtype=np.int64))
                counts.append(np.full(len(pos), len(neg), dtype=np.int64))
                offset += len(neg)
        self.positives = np.concatenate(positives) if positives else np.empty(0, dtype=np.int64)
        self.negatives = np.concatenate(negatives) if negatives else np.empty(0, dtype=np.int64)
        self.offsets = np.concatenate(offsets) if offsets else np.empty(0, dtype=np.int64)
        self.counts = np.concatenate(counts) if counts else np.empty(0, dtype=np.int64)

    def sample(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        if not len(self.positives):
            return self.positives, self.positives
        negatives = self.negatives[self.offsets + rng.integers(0, self.counts)]
        permutation = rng.permutation(len(self.positives))
        return self.positives[permutation], negatives[permutation]


class DCNLite(nn.Module):
    def __init__(self, total_dim: int, n_fields: int, k: int, dropout: float,
                 embedding_dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(total_dim, k)
        nn.init.normal_(self.embedding.weight, std=0.01)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        width = n_fields * k
        self.cross = nn.Linear(width, width)
        self.mlp = nn.Sequential(nn.Linear(width, 128), nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.embedding_dropout(self.embedding(x)).flatten(1)
        return self.head(self.mlp(x0 * self.cross(x0) + x0)).squeeze(1)


def official_metrics(users: np.ndarray, labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    measured = official_evaluate(users.tolist(), labels.astype(int).tolist(), scores.tolist())
    return {"gauc": float(measured["GAUC"]), "ndcg5": float(measured["nDCG@5"]),
            "primary": float(measured["primary"])}


def _config(args: argparse.Namespace) -> dict[str, Any]:
    keys = ("seed", "epochs", "batch_size", "lr", "step_decay_factor", "decay_every",
            "decay_start_epoch", "dropout", "embedding_dropout", "weight_decay", "k",
            "recency_half_life", "bpr_weight", "patience_halves", "max_runtime", "subsample")
    return {key: getattr(args, key) for key in keys}


def train_and_report(ds: dict[str, Any], args: argparse.Namespace,
                     checkpoint_callback: Callable[[int, float], None] | None = None) -> dict[str, Any]:
    started = time.time()
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    tr, va = ds["train"], ds["valid"]
    train_x = torch.as_tensor(np.ascontiguousarray(tr["X"]), dtype=torch.long)
    train_y = torch.as_tensor(tr["y"], dtype=torch.float32)
    train_weights = torch.as_tensor(recency_weights(tr["date"], args.recency_half_life))
    valid_x = torch.as_tensor(np.ascontiguousarray(va["X"]), dtype=torch.long)
    model = DCNLite(ds["field_dims_total"], train_x.shape[1], args.k, args.dropout,
                    args.embedding_dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sampler = PairSampler(tr["users"], tr["y"])

    def predict() -> np.ndarray:
        model.eval()
        with torch.no_grad():
            chunks = [model(valid_x[start:start + 200_000])
                      for start in range(0, len(valid_x), 200_000)]
        model.train()
        return torch.cat(chunks).numpy()

    point_order = np.arange(len(train_y))
    half_size = math.ceil(len(train_y) / 2)
    interval_halves = int(round(args.decay_every * 2))
    start_halves = int(round(args.decay_start_epoch * 2))
    best_primary, best_state, best_epoch = -math.inf, None, 0.0
    bad, completed_halves, timed_out, stopped = 0, 0, False, False
    history: list[dict[str, float]] = []

    class RunTimeout(Exception):
        pass

    def timeout_handler(_signum: int, _frame: Any) -> None:
        raise RunTimeout

    previous_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(args.max_runtime)
    try:
        for epoch in range(1, args.epochs + 1):
            point_order = rng.permutation(point_order)
            pair_pos, pair_neg = sampler.sample(rng)
            for half in range(2):
                begin, end = half * half_size, min(len(train_y), (half + 1) * half_size)
                half_rows = point_order[begin:end]
                batches = math.ceil(len(half_rows) / args.batch_size)
                losses = []
                for batch in range(batches):
                    idx_np = half_rows[batch * args.batch_size:(batch + 1) * args.batch_size]
                    idx = torch.as_tensor(idx_np, dtype=torch.long)
                    logits, weights = model(train_x[idx]), train_weights[idx]
                    point = nn.functional.binary_cross_entropy_with_logits(
                        logits, train_y[idx], reduction="none")
                    point_loss = (point * weights).sum() / weights.sum()
                    pair_begin = ((half * batches + batch) * len(pair_pos)) // (2 * batches)
                    pair_end = ((half * batches + batch + 1) * len(pair_pos)) // (2 * batches)
                    if pair_end > pair_begin:
                        positive = torch.as_tensor(pair_pos[pair_begin:pair_end], dtype=torch.long)
                        negative = torch.as_tensor(pair_neg[pair_begin:pair_end], dtype=torch.long)
                        pair = nn.functional.softplus(model(train_x[negative]) - model(train_x[positive]))
                        pair_weights = 0.5 * (train_weights[positive] + train_weights[negative])
                        pair_loss = (pair * pair_weights).sum() / pair_weights.sum()
                    else:
                        pair_loss = point_loss * 0.0
                    loss = (1.0 - args.bpr_weight) * point_loss + args.bpr_weight * pair_loss
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.detach()))
                completed_halves += 1
                scores = predict()
                metrics = official_metrics(va["users"], va["y"], scores)
                epoch_value = completed_halves / 2
                entry = {"epoch": epoch_value, "train_loss": float(np.mean(losses)),
                         "lr": float(optimizer.param_groups[0]["lr"]),
                         "val_gauc": metrics["gauc"], "val_ndcg5": metrics["ndcg5"],
                         "val_primary": metrics["primary"]}
                history.append(entry)
                print(f"epoch {epoch_value:.1f} | loss {entry['train_loss']:.4f} | "
                      f"lr {entry['lr']:.7g} | valid primary {metrics['primary']:.6f}", flush=True)
                if metrics["primary"] > best_primary + 1e-5:
                    best_primary, best_state, best_epoch, bad = (
                        metrics["primary"], copy.deepcopy(model.state_dict()), epoch_value, 0)
                else:
                    bad += 1
                if checkpoint_callback is not None:
                    checkpoint_callback(completed_halves - 1, metrics["primary"])
                if completed_halves >= start_halves and (completed_halves - start_halves) % interval_halves == 0:
                    for group in optimizer.param_groups:
                        group["lr"] *= args.step_decay_factor
                if bad >= args.patience_halves:
                    stopped = True
                    break
            if stopped:
                break
    except RunTimeout:
        timed_out = True
        print(f"runtime alarm reached at {args.max_runtime}s; preserving best checkpoint", flush=True)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
    if best_state is None:
        raise RuntimeError("runtime cap expired before a checkpoint completed")
    model.load_state_dict(best_state)
    scores = predict()
    metrics: dict[str, Any] = official_metrics(va["users"], va["y"], scores)
    metrics.update({"history": history, "runtime_s": round(time.time() - started, 1),
                    "best_epoch": best_epoch, "config": _config(args), "timed_out": timed_out,
                    "delta_vs_frozen_baseline": metrics["primary"] - FROZEN_BASELINE,
                    "field_names": list(FROZEN_FIELDS)})
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "scores.npy", scores)
    with (out / "predictions.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(("row_id", "user_id", "video_id", "score"))
        for row_id, (user, video, score) in enumerate(zip(va["users"], va["videos"], scores)):
            writer.writerow((row_id, int(user), int(video), f"{float(score):.10f}"))
    with (out / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, sort_keys=True)
        fh.write("\n")
    print("final:", json.dumps({k: v for k, v in metrics.items() if k != "history"},
                                sort_keys=True), flush=True)
    return metrics


def run(args: argparse.Namespace,
        checkpoint_callback: Callable[[int, float], None] | None = None) -> dict[str, Any]:
    if not 0 <= args.bpr_weight <= 1:
        raise ValueError("BPR weight must be in [0, 1]")
    if not 0 < args.step_decay_factor <= 1:
        raise ValueError("step-decay factor must be in (0, 1]")
    return train_and_report(load_validation_only(args.data_dir, args.subsample), args,
                            checkpoint_callback)


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
