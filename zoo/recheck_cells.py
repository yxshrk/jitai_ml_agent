"""Final validation-only rechecks on top of the frozen ``polish_stack``.

The three isolated cells are ``session``, ``aux-click``, and ``temporal``.
Comma-separated cells are accepted solely for the protocol's best-pair follow-up.
All optimization, regularization, sampling, scheduling, and checkpoint-selection
defaults are inherited from :mod:`zoo.polish_stack`.
"""

from __future__ import annotations

import copy
import csv
import datetime as dt
import json
import math
import signal
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zoo import polish_stack as frozen  # noqa: E402


VALID_CELLS = frozenset(("session", "aux-click", "temporal"))


def parser():
    ap = frozen.parser(__doc__)
    ap.add_argument("--cell", required=True,
                    help="session, aux-click, temporal, or a comma-separated pair")
    ap.add_argument("--raw-data-dir", default=None,
                    help="directory with the two KuaiRand-Pure standard log CSVs")
    return ap


def _cells(spec: str) -> tuple[str, ...]:
    cells = tuple(part.strip() for part in spec.split(",") if part.strip())
    if not cells or len(cells) > 2 or len(set(cells)) != len(cells):
        raise ValueError("--cell must contain one cell or two distinct cells")
    unknown = set(cells) - VALID_CELLS
    if unknown:
        raise ValueError(f"unknown cell(s): {', '.join(sorted(unknown))}")
    return cells


def _append_categorical(ds: dict[str, Any], train_values: np.ndarray,
                        valid_values: np.ndarray, name: str) -> None:
    """Fit a train-only vocabulary and append an UNK-aware categorical field."""
    train_values = np.asarray(train_values)
    valid_values = np.asarray(valid_values)
    vocab = {value: i for i, value in enumerate(dict.fromkeys(train_values.tolist()))}
    unknown = len(vocab)
    offset = int(ds["field_dims_total"])
    tr = np.fromiter((vocab[value] + offset for value in train_values.tolist()),
                     dtype=np.int64, count=len(train_values))
    va = np.fromiter((vocab.get(value, unknown) + offset for value in valid_values.tolist()),
                     dtype=np.int64, count=len(valid_values))
    ds["train"]["X"] = np.column_stack((ds["train"]["X"], tr))
    ds["valid"]["X"] = np.column_stack((ds["valid"]["X"], va))
    ds["field_dims_total"] = offset + unknown + 1
    ds["field_names"].append(name)


def _weekday(values: np.ndarray) -> np.ndarray:
    lookup = {int(value): dt.date(int(value) // 10000, int(value) // 100 % 100,
                                  int(value) % 100).weekday()
              for value in np.unique(values)}
    return np.fromiter((lookup[int(value)] for value in values),
                       dtype=np.int64, count=len(values))


def add_temporal(ds: dict[str, Any]) -> None:
    tr, va = ds["train"], ds["valid"]
    _append_categorical(ds, tr["hourmin"] // 100, va["hourmin"] // 100, "hour")
    _append_categorical(ds, _weekday(tr["date"]), _weekday(va["date"]), "day_of_week")


def _aligned_times(export_csv: Path, raw_csv: Path, expected: int,
                   max_date: int) -> np.ndarray:
    """Align raw times in source occurrence order, matching the audited method."""
    result = np.empty(expected, dtype=np.int64)
    keys = ("user_id", "video_id", "date", "hourmin")
    count = 0
    with export_csv.open(newline="", encoding="utf-8") as efh, \
            raw_csv.open(newline="", encoding="utf-8") as rfh:
        exported, raw = csv.DictReader(efh), csv.DictReader(rfh)
        allowed_raw = (row for row in raw if int(row["date"]) <= max_date)
        for count, erow in enumerate(exported, start=1):
            if count > expected:
                raise ValueError(f"{export_csv} has more than {expected} rows")
            rrow = next(allowed_raw)
            if tuple(erow[key] for key in keys) != tuple(rrow[key] for key in keys):
                raise ValueError(f"raw/export join mismatch at row {count - 1}")
            result[count - 1] = int(rrow["time_ms"])
    if count != expected:
        raise ValueError(f"{export_csv} row mismatch: {count} != {expected}")
    return result


def add_session(ds: dict[str, Any], data_dir: Path, raw_dir: Path) -> None:
    if any(len(ds[name]["y"]) != expected for name, expected in
           (("train", 1_141_112), ("valid", 124_909))):
        raise ValueError("session fields require the full real export")
    train_times = _aligned_times(data_dir / "train.csv",
                                 raw_dir / "log_standard_4_08_to_4_21_pure.csv",
                                 len(ds["train"]["y"]), frozen.TRAIN_MAX_DATE)
    valid_times = _aligned_times(data_dir / "val.csv",
                                 raw_dir / "log_standard_4_22_to_5_08_pure.csv",
                                 len(ds["valid"]["y"]), frozen.VALID_MAX_DATE)
    users = np.concatenate((ds["train"]["users"], ds["valid"]["users"]))
    times = np.concatenate((train_times, valid_times))
    split_at = len(train_times)
    order = np.lexsort((np.arange(len(times)), times, users))
    gap_bucket = np.empty(len(times), dtype=np.int64)
    session_index = np.empty(len(times), dtype=np.int64)
    session_start = np.empty(len(times), dtype=np.int64)
    edges = np.asarray((1_000, 5_000, 30_000, 120_000, 600_000, 1_800_000),
                       dtype=np.int64)
    last_user = last_time = -1
    current_index = 0
    for idx in order:
        user, now = int(users[idx]), int(times[idx])
        new = user != last_user or now - last_time > 1_800_000
        gap = np.iinfo(np.int64).max if user != last_user else max(0, now - last_time)
        current_index = 0 if new else current_index + 1
        gap_bucket[idx] = int(np.searchsorted(edges, gap, side="right"))
        session_index[idx] = min(current_index, 31)
        session_start[idx] = int(new)
        last_user, last_time = user, now
    for values, name in ((gap_bucket, "previous_exposure_gap_bucket"),
                         (session_index, "within_session_index"),
                         (session_start, "session_start")):
        _append_categorical(ds, values[:split_at], values[split_at:], name)


class RecheckDCN(nn.Module):
    """Frozen shared bottom with an optional click head."""

    def __init__(self, total_dim: int, n_fields: int, args, click_head: bool):
        super().__init__()
        self.embedding = nn.Embedding(total_dim, args.k)
        nn.init.normal_(self.embedding.weight, std=0.01)
        self.embedding_dropout = nn.Dropout(args.embedding_dropout)
        width = n_fields * args.k
        self.cross = nn.Linear(width, width)
        self.mlp = nn.Sequential(nn.Linear(width, 128), nn.ReLU(), nn.Dropout(args.dropout))
        self.head = nn.Linear(128, 1)
        self.click_head = nn.Linear(128, 1) if click_head else None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        x0 = self.embedding_dropout(self.embedding(x)).flatten(1)
        hidden = self.mlp(x0 * self.cross(x0) + x0)
        main = self.head(hidden).squeeze(1)
        click = self.click_head(hidden).squeeze(1) if self.click_head is not None else None
        return main, click


def train_and_report(ds: dict[str, Any], args, cells: tuple[str, ...]) -> dict[str, Any]:
    started = time.time()
    frozen.set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    tr, va = ds["train"], ds["valid"]
    train_x = torch.as_tensor(np.ascontiguousarray(tr["X"]), dtype=torch.long)
    train_y = torch.as_tensor(tr["y"], dtype=torch.float32)
    click_y = torch.as_tensor(tr["click"], dtype=torch.float32)
    weights = torch.as_tensor(frozen.recency_weights(tr["date"], args.recency_half_life))
    valid_x = torch.as_tensor(np.ascontiguousarray(va["X"]), dtype=torch.long)
    use_click = "aux-click" in cells
    model = RecheckDCN(ds["field_dims_total"], train_x.shape[1], args, use_click)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    sampler = frozen.PairSampler(tr["users"], tr["y"])

    def predict() -> np.ndarray:
        model.eval()
        with torch.no_grad():
            chunks = [model(valid_x[start:start + 200_000])[0]
                      for start in range(0, len(valid_x), 200_000)]
        model.train()
        return torch.cat(chunks).numpy()

    point_order = np.arange(len(train_y))
    half_size = math.ceil(len(train_y) / 2)
    interval_halves = int(round(args.decay_every * 2))
    start_halves = int(round(args.decay_start_epoch * 2))
    best_primary, best_state, best_epoch = -math.inf, None, 0.0
    bad = completed_halves = 0
    timed_out = stopped = False
    history: list[dict[str, float]] = []

    class RunTimeout(Exception):
        pass

    def timeout_handler(_signum: int, _frame: Any) -> None:
        raise RunTimeout

    previous_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(args.max_runtime)
    try:
        for _epoch in range(1, args.epochs + 1):
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
                    logits, click_logits = model(train_x[idx])
                    point = nn.functional.binary_cross_entropy_with_logits(
                        logits, train_y[idx], reduction="none")
                    point_loss = (point * weights[idx]).sum() / weights[idx].sum()
                    pair_begin = ((half * batches + batch) * len(pair_pos)) // (2 * batches)
                    pair_end = ((half * batches + batch + 1) * len(pair_pos)) // (2 * batches)
                    if pair_end > pair_begin:
                        positive = torch.as_tensor(pair_pos[pair_begin:pair_end], dtype=torch.long)
                        negative = torch.as_tensor(pair_neg[pair_begin:pair_end], dtype=torch.long)
                        neg_logits = model(train_x[negative])[0]
                        pos_logits = model(train_x[positive])[0]
                        pair = nn.functional.softplus(neg_logits - pos_logits)
                        pair_weights = 0.5 * (weights[positive] + weights[negative])
                        pair_loss = (pair * pair_weights).sum() / pair_weights.sum()
                    else:
                        pair_loss = point_loss * 0.0
                    loss = ((1.0 - args.bpr_weight) * point_loss +
                            args.bpr_weight * pair_loss)
                    if use_click:
                        assert click_logits is not None
                        aux = nn.functional.binary_cross_entropy_with_logits(
                            click_logits, click_y[idx], reduction="none")
                        loss = loss + 0.1 * (aux * weights[idx]).sum() / weights[idx].sum()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.detach()))
                completed_halves += 1
                scores = predict()
                metrics = frozen.official_metrics(va["users"], va["y"], scores)
                epoch_value = completed_halves / 2
                entry = {"epoch": epoch_value, "train_loss": float(np.mean(losses)),
                         "lr": float(optimizer.param_groups[0]["lr"]),
                         "val_gauc": metrics["gauc"], "val_ndcg5": metrics["ndcg5"],
                         "val_primary": metrics["primary"]}
                history.append(entry)
                print(f"epoch {epoch_value:.1f} | loss {entry['train_loss']:.4f} | "
                      f"lr {entry['lr']:.7g} | valid primary {metrics['primary']:.6f}",
                      flush=True)
                if metrics["primary"] > best_primary + 1e-5:
                    best_primary, best_state, best_epoch, bad = (
                        metrics["primary"], copy.deepcopy(model.state_dict()), epoch_value, 0)
                else:
                    bad += 1
                if completed_halves >= start_halves and \
                        (completed_halves - start_halves) % interval_halves == 0:
                    for group in optimizer.param_groups:
                        group["lr"] *= args.step_decay_factor
                if bad >= args.patience_halves:
                    stopped = True
                    break
            if stopped:
                break
    except RunTimeout:
        timed_out = True
        print(f"runtime alarm reached at {args.max_runtime}s; preserving best checkpoint",
              flush=True)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
    if best_state is None:
        raise RuntimeError("runtime cap expired before a checkpoint completed")
    model.load_state_dict(best_state)
    scores = predict()
    metrics: dict[str, Any] = frozen.official_metrics(va["users"], va["y"], scores)
    metrics.update({"history": history, "runtime_s": round(time.time() - started, 1),
                    "best_epoch": best_epoch, "timed_out": timed_out,
                    "cells": list(cells), "aux_click_weight": 0.1 if use_click else 0.0,
                    "field_names": ds["field_names"],
                    "config": frozen._config(args)})
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "scores.npy", scores)
    with (out / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, sort_keys=True)
        fh.write("\n")
    print("final:", json.dumps({k: v for k, v in metrics.items() if k != "history"},
                                sort_keys=True), flush=True)
    return metrics


def run(args) -> dict[str, Any]:
    cells = _cells(args.cell)
    ds = frozen.load_validation_only(args.data_dir, args.subsample)
    ds["field_names"] = list(frozen.FROZEN_FIELDS)
    data_dir = frozen._resolve_data_dir(args.data_dir)
    if "session" in cells:
        raw_dir = (Path(args.raw_data_dir) if args.raw_data_dir else
                   frozen.ROOT.parent / "KuaiRand-Pure" / "data")
        add_session(ds, data_dir, raw_dir)
    if "temporal" in cells:
        add_temporal(ds)
    return train_and_report(ds, args, cells)


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
