"""Validation-only runner for the final marginal-method campaign.

This module imports the frozen stack and changes only the requested training
mechanism.  It never opens a test export and never fits anything on validation.
All training modes use the frozen ``polish_stack.py`` defaults.
"""

from __future__ import annotations

import argparse
import copy
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

from zoo import ensemble, polish_stack  # noqa: E402


def parser() -> argparse.ArgumentParser:
    ap = polish_stack.parser(__doc__)
    ap.add_argument("--mode", choices=("density", "sam", "window", "cyclic", "combine"),
                    required=True)
    ap.add_argument("--density-cap", type=float, choices=(3.0, 5.0))
    ap.add_argument("--combine-recency", action="store_true")
    ap.add_argument("--rho", type=float, choices=(0.02, 0.05))
    ap.add_argument("--window-days", type=int, choices=(7, 10, 14))
    ap.add_argument("--inputs", nargs="+", type=Path)
    ap.add_argument("--weights", nargs="+", type=float)
    return ap


def _density_ratio_weights(x: np.ndarray, dates: np.ndarray, cap: float,
                           seed: int) -> tuple[np.ndarray, dict[str, float]]:
    """Fit a tiny hashed additive logistic discriminator on train rows only."""
    polish_stack.set_seed(seed)
    buckets = 4096
    hashed = torch.as_tensor(
        np.ascontiguousarray(x % buckets + np.arange(x.shape[1]) * buckets),
        dtype=torch.long,
    )
    target = torch.as_tensor((dates >= 20220419).astype(np.float32))
    classifier = nn.Embedding(x.shape[1] * buckets, 1)
    nn.init.zeros_(classifier.weight)
    bias = nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.Adam([classifier.weight, bias], lr=0.03, weight_decay=1e-5)
    rng = np.random.default_rng(seed)
    batch_size = 65536
    for _epoch in range(4):
        order = rng.permutation(len(x))
        for begin in range(0, len(x), batch_size):
            stop = min(len(x), begin + batch_size)
            idx = torch.as_tensor(order[begin:stop], dtype=torch.long)
            logits = classifier(hashed[idx]).squeeze(-1).sum(1) + bias
            loss = nn.functional.binary_cross_entropy_with_logits(logits, target[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        logits = classifier(hashed).squeeze(-1).sum(1) + bias
        posterior = torch.sigmoid(logits).numpy().astype(np.float64)
    late_prior = float(target.mean())
    ratio = posterior / np.maximum(1.0 - posterior, 1e-6)
    ratio *= (1.0 - late_prior) / late_prior
    ratio = np.minimum(ratio, cap)
    ratio = np.maximum(ratio, 1e-3)
    ratio = (ratio / ratio.mean()).astype(np.float32)
    stats = {"late_prior": late_prior, "ratio_min": float(ratio.min()),
             "ratio_mean": float(ratio.mean()), "ratio_max": float(ratio.max())}
    return ratio, stats


def _loss(model: nn.Module, train_x: torch.Tensor, train_y: torch.Tensor,
          weights: torch.Tensor, idx_np: np.ndarray, pair_pos: np.ndarray,
          pair_neg: np.ndarray, pair_begin: int, pair_end: int,
          bpr_weight: float) -> torch.Tensor:
    idx = torch.as_tensor(idx_np, dtype=torch.long)
    point_weights = weights[idx]
    point = nn.functional.binary_cross_entropy_with_logits(
        model(train_x[idx]), train_y[idx], reduction="none")
    point_loss = (point * point_weights).sum() / point_weights.sum()
    if pair_end <= pair_begin:
        pair_loss = point_loss * 0.0
    else:
        positive = torch.as_tensor(pair_pos[pair_begin:pair_end], dtype=torch.long)
        negative = torch.as_tensor(pair_neg[pair_begin:pair_end], dtype=torch.long)
        pair = nn.functional.softplus(model(train_x[negative]) - model(train_x[positive]))
        pair_weights = 0.5 * (weights[positive] + weights[negative])
        pair_loss = (pair * pair_weights).sum() / pair_weights.sum()
    return (1.0 - bpr_weight) * point_loss + bpr_weight * pair_loss


def _sam_step(model: nn.Module, optimizer: torch.optim.Optimizer,
              closure: Any, rho: float) -> float:
    optimizer.zero_grad(set_to_none=True)
    first = closure()
    first.backward()
    parameters = [p for p in model.parameters() if p.grad is not None]
    norm = torch.linalg.vector_norm(torch.stack([p.grad.norm(2) for p in parameters]))
    scale = rho / (float(norm) + 1e-12)
    perturbations = []
    with torch.no_grad():
        for parameter in parameters:
            perturbation = parameter.grad * scale
            parameter.add_(perturbation)
            perturbations.append(perturbation)
    optimizer.zero_grad(set_to_none=True)
    second = closure()
    second.backward()
    with torch.no_grad():
        for parameter, perturbation in zip(parameters, perturbations):
            parameter.sub_(perturbation)
    optimizer.step()
    return float(first.detach())


def _save(out: Path, scores: np.ndarray, metrics: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "scores.npy", scores)
    with (out / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, sort_keys=True)
        handle.write("\n")
    print("final:", json.dumps(metrics, sort_keys=True), flush=True)


def train_variant(ds: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    polish_stack.set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    source, va = ds["train"], ds["valid"]
    mask = np.ones(len(source["y"]), dtype=bool)
    if args.mode == "window" and args.window_days != 14:
        first_date = 20220415 if args.window_days == 7 else 20220412
        mask = source["date"] >= first_date
    tr = {key: value[mask] for key, value in source.items() if isinstance(value, np.ndarray)}
    train_x = torch.as_tensor(np.ascontiguousarray(tr["X"]), dtype=torch.long)
    train_y = torch.as_tensor(tr["y"], dtype=torch.float32)
    valid_x = torch.as_tensor(np.ascontiguousarray(va["X"]), dtype=torch.long)
    weight_stats: dict[str, float] = {}
    if args.mode == "density":
        if args.density_cap is None:
            raise ValueError("density mode requires --density-cap")
        weights_np, weight_stats = _density_ratio_weights(
            tr["X"], tr["date"], args.density_cap, args.seed)
        if args.combine_recency:
            weights_np *= polish_stack.recency_weights(tr["date"], 7.0)
            weights_np /= weights_np.mean()
    else:
        weights_np = polish_stack.recency_weights(tr["date"], 7.0)
    train_weights = torch.as_tensor(weights_np, dtype=torch.float32)
    model = polish_stack.DCNLite(ds["field_dims_total"], train_x.shape[1], args.k,
                                 args.dropout, args.embedding_dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    sampler = polish_stack.PairSampler(tr["users"], tr["y"])

    def predict() -> np.ndarray:
        model.eval()
        with torch.no_grad():
            chunks = [model(valid_x[start:start + 200_000])
                      for start in range(0, len(valid_x), 200_000)]
        model.train()
        return torch.cat(chunks).numpy()

    point_order = np.arange(len(train_y))
    half_size = math.ceil(len(train_y) / 2)
    best_primary, best_state, best_epoch = -math.inf, None, 0.0
    bad = 0
    history: list[dict[str, float]] = []
    timed_out = False

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
                    pair_begin = ((half * batches + batch) * len(pair_pos)) // (2 * batches)
                    pair_end = ((half * batches + batch + 1) * len(pair_pos)) // (2 * batches)
                    def closure() -> torch.Tensor:
                        return _loss(model, train_x, train_y, train_weights, idx_np,
                                     pair_pos, pair_neg, pair_begin, pair_end,
                                     args.bpr_weight)
                    if args.mode == "sam":
                        if args.rho is None:
                            raise ValueError("sam mode requires --rho")
                        losses.append(_sam_step(model, optimizer, closure, args.rho))
                    else:
                        optimizer.zero_grad(set_to_none=True)
                        loss = closure()
                        loss.backward()
                        optimizer.step()
                        losses.append(float(loss.detach()))
                completed_halves = (epoch - 1) * 2 + half + 1
                scores = predict()
                metrics = polish_stack.official_metrics(va["users"], va["y"], scores)
                epoch_value = completed_halves / 2
                history.append({"epoch": epoch_value, "train_loss": float(np.mean(losses)),
                                "lr": float(optimizer.param_groups[0]["lr"]),
                                "val_primary": metrics["primary"]})
                print(f"epoch {epoch_value:.1f} | valid primary {metrics['primary']:.6f}",
                      flush=True)
                if metrics["primary"] > best_primary + 1e-5:
                    best_primary, best_state, best_epoch, bad = (
                        metrics["primary"], copy.deepcopy(model.state_dict()), epoch_value, 0)
                else:
                    bad += 1
                if completed_halves % 2 == 0:
                    for group in optimizer.param_groups:
                        group["lr"] *= args.step_decay_factor
                if bad >= args.patience_halves:
                    break
            if bad >= args.patience_halves:
                break
    except RunTimeout:
        timed_out = True
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
    if best_state is None:
        raise RuntimeError("runtime cap expired before a checkpoint completed")
    model.load_state_dict(best_state)
    scores = predict()
    metrics = polish_stack.official_metrics(va["users"], va["y"], scores)
    metrics.update({"mode": args.mode, "seed": args.seed, "best_epoch": best_epoch,
                    "runtime_s": round(time.time() - started, 1), "timed_out": timed_out,
                    "delta_vs_0.6047": metrics["primary"] - 0.6047,
                    "train_rows": len(train_y), "density_cap": args.density_cap,
                    "combine_recency": args.combine_recency, "rho": args.rho,
                    "window_days": args.window_days, "weight_stats": weight_stats,
                    "history": history})
    _save(Path(args.out_dir), scores, metrics)
    return metrics


def cyclic(ds: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Train one full-data run and capture fixed epoch-2/4/6 cosine-cycle minima."""
    started = time.time()
    polish_stack.set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    tr, va = ds["train"], ds["valid"]
    train_x = torch.as_tensor(np.ascontiguousarray(tr["X"]), dtype=torch.long)
    train_y = torch.as_tensor(tr["y"], dtype=torch.float32)
    weights = torch.as_tensor(polish_stack.recency_weights(tr["date"], 7.0))
    valid_x = torch.as_tensor(np.ascontiguousarray(va["X"]), dtype=torch.long)
    model = polish_stack.DCNLite(ds["field_dims_total"], 5, args.k, args.dropout,
                                 args.embedding_dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    sampler = polish_stack.PairSampler(tr["users"], tr["y"])
    order = np.arange(len(train_y))
    snapshots: list[np.ndarray] = []
    member_metrics: list[dict[str, float]] = []
    total_batches = math.ceil(len(order) / args.batch_size)
    for epoch in range(1, 7):
        order = rng.permutation(order)
        pos, neg = sampler.sample(rng)
        for batch in range(total_batches):
            phase = ((epoch - 1) * total_batches + batch) % (2 * total_batches)
            cosine = 0.5 * (1.0 + math.cos(math.pi * phase / (2 * total_batches - 1)))
            for group in optimizer.param_groups:
                group["lr"] = 1e-4 + (args.lr - 1e-4) * cosine
            begin, end = batch * args.batch_size, min(len(order), (batch + 1) * args.batch_size)
            pair_begin = batch * len(pos) // total_batches
            pair_end = (batch + 1) * len(pos) // total_batches
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model, train_x, train_y, weights, order[begin:end], pos, neg,
                         pair_begin, pair_end, args.bpr_weight)
            loss.backward()
            optimizer.step()
        if epoch in (2, 4, 6):
            model.eval()
            with torch.no_grad():
                score = model(valid_x).numpy()
            model.train()
            snapshots.append(score)
            member_metrics.append(polish_stack.official_metrics(va["users"], va["y"], score))
            print(f"snapshot epoch {epoch} | {member_metrics[-1]['primary']:.6f}", flush=True)
        if time.time() - started > args.max_runtime:
            raise TimeoutError("cyclic run exceeded runtime cap")
    scores = ensemble.rank_average(va["users"], snapshots)
    metrics = polish_stack.official_metrics(va["users"], va["y"], scores)
    metrics.update({"mode": "cyclic", "seed": args.seed,
                    "snapshot_epochs": [2, 4, 6], "member_metrics": member_metrics,
                    "runtime_s": round(time.time() - started, 1), "timed_out": False,
                    "delta_vs_0.6047": metrics["primary"] - 0.6047})
    _save(Path(args.out_dir), scores, metrics)
    return metrics


def combine(ds: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if not args.inputs or len(args.inputs) < 2:
        raise ValueError("combine mode requires at least two --inputs")
    started = time.time()
    arrays = [np.load(path) for path in args.inputs]
    weights = None if args.weights is None else np.asarray(args.weights, dtype=np.float64)
    scores = ensemble.rank_average(ds["valid"]["users"], arrays, weights)
    metrics = polish_stack.official_metrics(ds["valid"]["users"], ds["valid"]["y"], scores)
    metrics.update({"mode": "combine", "seed": args.seed,
                    "members": [str(path) for path in args.inputs],
                    "weights": None if weights is None else weights.tolist(),
                    "runtime_s": round(time.time() - started, 1), "timed_out": False,
                    "delta_vs_0.6047": metrics["primary"] - 0.6047})
    _save(Path(args.out_dir), scores, metrics)
    return metrics


def main() -> None:
    args = parser().parse_args()
    ds = polish_stack.load_validation_only(args.data_dir, args.subsample)
    if args.mode == "cyclic":
        cyclic(ds, args)
    elif args.mode == "combine":
        combine(ds, args)
    else:
        train_variant(ds, args)


if __name__ == "__main__":
    main()
