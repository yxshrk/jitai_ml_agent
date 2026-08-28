"""Self-contained DCN-lite trainer for the regularization/schedule sweep.

This file deliberately duplicates the small amount of required infrastructure instead
of importing zoo/common.py or zoo/best.py, which are owned by another campaign.
It reads only train.npz and val.npz and scores only validation predictions with the
vendored official evaluator.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import random
import signal
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from data.official.evaluate import evaluate as official_evaluate

BASELINE_PRIMARY = 0.6016


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--schedule", choices=("constant", "cosine", "step"), default="constant")
    ap.add_argument("--min-lr", type=float, default=1e-4)
    ap.add_argument("--step-gamma", type=float, default=0.5)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--cross-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1, help="MLP dropout")
    ap.add_argument("--embedding-dropout", type=float, default=0.0)
    ap.add_argument("--weight-decay", type=float, default=0.0, help="non-embedding AdamW L2")
    ap.add_argument("--embedding-weight-decay", type=float, default=None)
    ap.add_argument("--aux-weight", type=float, default=0.1)
    ap.add_argument("--bpr-weight", type=float, default=0.5)
    ap.add_argument("--average", choices=("none", "ema"), default="none")
    ap.add_argument("--ema-decay", type=float, default=0.9)
    ap.add_argument("--ema-start", type=int, default=2)
    ap.add_argument("--subsample", type=int, default=None, help="train row cap for smoke tests")
    ap.add_argument("--max-runtime", type=int, default=330,
                    help="hard training alarm in seconds; preserves the best completed epoch")
    return ap


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def load_npz(data_dir: str | Path, subsample: int | None = None) -> dict[str, dict[str, np.ndarray]]:
    root = Path(data_dir)
    if str(data_dir) == "real":
        root = ROOT / "data" / "real_ws"
    result: dict[str, dict[str, np.ndarray]] = {}
    for name, filename in (("train", "train.npz"), ("valid", "val.npz")):
        with np.load(root / filename) as data:
            split = {key: data[key].copy() for key in data.files}
        if name == "train" and subsample is not None:
            split = {key: value[:subsample] if value.ndim and len(value) == len(split["y"]) else value
                     for key, value in split.items()}
        result[name] = split
    return add_features(result)


def _encode(train: np.ndarray, values: Mapping[str, np.ndarray], offset: int) -> tuple[dict[str, np.ndarray], int]:
    vocab = {value: index for index, value in enumerate(dict.fromkeys(train.tolist()))}
    unknown = len(vocab)
    encoded = {name: np.fromiter((vocab.get(value, unknown) + offset for value in column.tolist()),
                                 dtype=np.int64, count=len(column))
               for name, column in values.items()}
    return encoded, offset + unknown + 1


def add_features(ds: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, np.ndarray]]:
    """Reimplement the known-best train-only duration and temporal feature stack."""
    splits = ("train", "valid")
    train = ds["train"]
    edges = np.quantile(train["duration_ms"], np.linspace(0, 1, 51)[1:-1])
    b50 = {name: np.searchsorted(edges, ds[name]["duration_ms"]).astype(np.int64) for name in splits}
    short = {name: (ds[name]["duration_ms"] <= 18_000).astype(np.int64) for name in splits}
    cross = {name: b50[name] * 100 + ds[name]["X"][:, 3].astype(np.int64) for name in splits}
    hour = {name: (ds[name]["hourmin"] // 100).astype(np.int64) for name in splits}

    def weekdays(dates: np.ndarray) -> np.ndarray:
        lookup = {int(value): dt.date(int(value) // 10000, int(value) // 100 % 100,
                                      int(value) % 100).weekday() for value in np.unique(dates)}
        return np.fromiter((lookup[int(value)] for value in dates), dtype=np.int64, count=len(dates))

    dow = {name: weekdays(ds[name]["date"]) for name in splits}
    offset = int(train["field_dims"].sum())
    columns: list[dict[str, np.ndarray]] = []
    for raw in (b50, short, cross, hour, dow):
        encoded, offset = _encode(raw["train"], raw, offset)
        columns.append(encoded)
    for name in splits:
        extra = np.column_stack([column[name] for column in columns])
        ds[name]["X"] = np.hstack((ds[name]["X"].astype(np.int64), extra))
    ds["field_dims_total"] = offset  # type: ignore[assignment]
    return ds


class PairSampler:
    def __init__(self, users: np.ndarray, labels: np.ndarray):
        order = np.argsort(users, kind="stable")
        sorted_users, sorted_labels = users[order], labels[order]
        starts = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1]])
        ends = np.r_[starts[1:], len(order)]
        positives, negatives, negative_starts, negative_counts = [], [], [], []
        offset = 0
        for start, end in zip(starts, ends):
            indices = order[start:end]
            pos = indices[sorted_labels[start:end] == 1]
            neg = indices[sorted_labels[start:end] == 0]
            if len(pos) and len(neg):
                positives.append(pos)
                negatives.append(neg)
                negative_starts.append(np.full(len(pos), offset, dtype=np.int64))
                negative_counts.append(np.full(len(pos), len(neg), dtype=np.int64))
                offset += len(neg)
        self.positives = np.concatenate(positives)
        self.negatives = np.concatenate(negatives)
        self.negative_starts = np.concatenate(negative_starts)
        self.negative_counts = np.concatenate(negative_counts)

    def sample(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        neg = self.negatives[self.negative_starts + rng.integers(0, self.negative_counts)]
        order = rng.permutation(len(self.positives))
        return self.positives[order], neg[order]


class DCNLite(nn.Module):
    def __init__(self, total_dim: int, fields: int, k: int = 16, hidden: int = 128,
                 cross_layers: int = 2, dropout: float = 0.1,
                 embedding_dropout: float = 0.0, auxiliary: bool = True):
        super().__init__()
        self.embedding = nn.Embedding(total_dim, k)
        nn.init.normal_(self.embedding.weight, std=0.01)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        width = fields * k
        self.cross = nn.ModuleList(nn.Linear(width, width) for _ in range(cross_layers))
        self.mlp = nn.Sequential(nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(dropout))
        names = ("main", "click", "effective_view") if auxiliary else ("main",)
        self.heads = nn.ModuleDict({name: nn.Linear(hidden, 1) for name in names})

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x0 = self.embedding_dropout(self.embedding(x)).flatten(1)
        crossed = x0
        for layer in self.cross:
            crossed = x0 * layer(crossed) + crossed
        hidden = self.mlp(crossed)
        return {name: head(hidden).squeeze(1) for name, head in self.heads.items()}


def official_metrics(users: np.ndarray, labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    result = official_evaluate(users.tolist(), labels.astype(int).tolist(), scores.tolist())
    return {"gauc": float(result["GAUC"]), "ndcg5": float(result["nDCG@5"]),
            "primary": float(result["primary"])}


def _state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _ema_update(average: dict[str, torch.Tensor], model: nn.Module, decay: float) -> None:
    for name, value in model.state_dict().items():
        if torch.is_floating_point(value):
            average[name].mul_(decay).add_(value.detach(), alpha=1.0 - decay)
        else:
            average[name].copy_(value)


def train(args: argparse.Namespace) -> dict[str, float | int | str]:
    started = time.time()
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    ds = load_npz(args.data_dir, args.subsample)
    train_split, valid = ds["train"], ds["valid"]
    train_x = torch.as_tensor(np.ascontiguousarray(train_split["X"]), dtype=torch.long)
    train_y = torch.as_tensor(train_split["y"], dtype=torch.float32)
    valid_x = torch.as_tensor(np.ascontiguousarray(valid["X"]), dtype=torch.long)
    click = torch.as_tensor(train_split["click"], dtype=torch.float32)
    effective = torch.as_tensor(train_split["play_time_ms"] >= np.minimum(
        train_split["duration_ms"], 18_000), dtype=torch.float32)
    model = DCNLite(int(ds["field_dims_total"]), train_x.shape[1], args.k, args.hidden,
                    args.cross_layers, args.dropout, args.embedding_dropout,
                    auxiliary=args.aux_weight > 0)

    embedding_decay = args.weight_decay if args.embedding_weight_decay is None else args.embedding_weight_decay
    embedding_parameters = list(model.embedding.parameters())
    other_parameters = [parameter for name, parameter in model.named_parameters()
                        if not name.startswith("embedding.")]
    optimizer = torch.optim.AdamW((
        {"params": embedding_parameters, "weight_decay": embedding_decay},
        {"params": other_parameters, "weight_decay": args.weight_decay},
    ), lr=args.lr)
    if args.schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_lr)
    elif args.schedule == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.step_gamma)
    else:
        scheduler = None
    sampler = PairSampler(train_split["user"], train_split["y"])
    bce = nn.BCEWithLogitsLoss()

    def predict() -> np.ndarray:
        model.eval()
        with torch.no_grad():
            result = torch.cat([model(valid_x[start:start + 200_000])["main"]
                                for start in range(0, len(valid_x), 200_000)]).numpy()
        model.train()
        return result

    best_gauc, best_state, best_epoch, bad = -math.inf, None, 0, 0
    ema_state: dict[str, torch.Tensor] | None = None
    point_count, batch_size = len(train_y), args.batch_size
    class RunTimeout(Exception):
        pass

    def timeout_handler(_signum, _frame):
        raise RunTimeout

    previous_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(args.max_runtime)
    timed_out = False
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_started = time.time()
            positives, negatives = sampler.sample(rng)
            point_order = rng.permutation(point_count)
            batch_count = math.ceil(point_count / batch_size)
            losses = []
            for batch in range(batch_count):
                indices = point_order[batch * batch_size:(batch + 1) * batch_size]
                output = model(train_x[indices])
                loss = (1.0 - args.bpr_weight) * bce(output["main"], train_y[indices])
                lo = batch * len(positives) // batch_count
                hi = (batch + 1) * len(positives) // batch_count
                if args.bpr_weight > 0 and hi > lo:
                    pos_scores = model(train_x[positives[lo:hi]])["main"]
                    neg_scores = model(train_x[negatives[lo:hi]])["main"]
                    loss = loss + args.bpr_weight * nn.functional.softplus(neg_scores - pos_scores).mean()
                if args.aux_weight > 0:
                    loss = loss + args.aux_weight * (bce(output["click"], click[indices]) +
                                                      bce(output["effective_view"], effective[indices]))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            if args.average == "ema" and epoch >= args.ema_start:
                if ema_state is None:
                    ema_state = _state(model)
                else:
                    _ema_update(ema_state, model, args.ema_decay)
            scores = predict()
            metrics = official_metrics(valid["user"], valid["y"], scores)
            selection_state, selection_gauc, selected = _state(model), metrics["gauc"], "raw"
            if ema_state is not None:
                raw_state = _state(model)
                model.load_state_dict(ema_state)
                ema_metrics = official_metrics(valid["user"], valid["y"], predict())
                model.load_state_dict(raw_state)
                if ema_metrics["gauc"] > selection_gauc:
                    selection_state, selection_gauc, selected = {k: v.clone() for k, v in ema_state.items()}, ema_metrics["gauc"], "ema"
            lr = optimizer.param_groups[0]["lr"]
            print(f"epoch {epoch:2d} loss={np.mean(losses):.4f} lr={lr:.6g} "
                  f"gauc={metrics['gauc']:.6f} primary={metrics['primary']:.6f} select={selected} "
                  f"seconds={time.time() - epoch_started:.1f}", flush=True)
            if selection_gauc > best_gauc + 1e-5:
                best_gauc, best_state, best_epoch, bad = selection_gauc, selection_state, epoch, 0
            else:
                bad += 1
            if scheduler is not None:
                scheduler.step()
            if bad >= args.patience:
                print(f"early stop at epoch {epoch}", flush=True)
                break
    except RunTimeout:
        timed_out = True
        print(f"hard runtime cap reached after {args.max_runtime}s; preserving best completed epoch", flush=True)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
    assert best_state is not None
    model.load_state_dict(best_state)
    scores = predict()
    final = official_metrics(valid["user"], valid["y"], scores)
    result: dict[str, float | int | str] = {
        **final, "runtime_s": round(time.time() - started, 1), "best_epoch": best_epoch,
        "seed": args.seed, "delta": final["primary"] - BASELINE_PRIMARY,
        "schedule": args.schedule, "timed_out": timed_out,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("row_id", "user_id", "video_id", "score"))
        for row_id, (user, video, score) in enumerate(zip(valid["user"], valid["X"][:, 1], scores)):
            writer.writerow((row_id, int(user), int(video), f"{float(score):.10f}"))
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, sort_keys=True)
        handle.write("\n")
    with (out_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, sort_keys=True)
        handle.write("\n")
    print("final:", json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> None:
    train(parser().parse_args())


if __name__ == "__main__":
    main()
