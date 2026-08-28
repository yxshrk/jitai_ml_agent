"""Validation-safe runners for the final long-shot campaign.

Only the frozen five-field strong-L0 inputs are used at inference.  Methods that
need play time use it solely while constructing training targets.  Rolling-day
and empirical-Bayes hyperparameters are selected on chronological train-only
folds before one official-validation read per finalized run.
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
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.official.evaluate import evaluate as official_evaluate
from zoo.dims_campaign import DimsDCN, PairSampler, load_data


BASELINE_PRIMARY = 0.6016
CONTROL_PRIMARY = 0.6047559520537656


def masking_probability(counts: np.ndarray, p0: float, alpha: float) -> np.ndarray:
    """Frequency-dependent masking probability from the campaign specification."""
    counts = np.asarray(counts, dtype=np.float64)
    if p0 < 0 or alpha < 0 or np.any(counts < 0):
        raise ValueError("p0, alpha, and counts must be nonnegative")
    return np.clip(p0 * np.power(1.0 + counts, -alpha), 0.0, 1.0)


def capped_day_weights(dates: np.ndarray, losses: np.ndarray, *, last_days: int = 4,
                       worst_k: int = 1, cap: float = 3.0) -> tuple[np.ndarray, tuple[int, ...]]:
    """Upweight the worst-loss days among the final ``last_days`` observed days."""
    dates = np.asarray(dates, dtype=np.int64)
    losses = np.asarray(losses, dtype=np.float64)
    if len(dates) != len(losses) or not len(dates):
        raise ValueError("dates and losses must be nonempty and aligned")
    if last_days < 1 or worst_k < 1 or cap < 1:
        raise ValueError("last_days, worst_k, and cap must be positive (cap >= 1)")
    candidates = np.unique(dates)[-last_days:]
    day_loss = [(float(losses[dates == day].mean()), int(day)) for day in candidates]
    worst = tuple(day for _, day in sorted(day_loss, reverse=True)[:min(worst_k, len(day_loss))])
    weights = np.ones(len(dates), dtype=np.float32)
    weights[np.isin(dates, worst)] = np.float32(cap)
    return weights, worst


def asymmetric_smoothed_targets(
    labels: np.ndarray,
    play_time_ms: np.ndarray,
    duration_ms: np.ndarray,
    *,
    long_near: tuple[float, float] = (0.10, 0.03),
    short_near: tuple[float, float] = (0.05, 0.015),
    far: tuple[float, float] = (0.01, 0.003),
    width: float = 0.20,
) -> np.ndarray:
    """Return asymmetric soft targets around min(duration, 18 seconds).

    Each pair is ``(positive smoothing, negative smoothing)``.  Short videos
    have their own near-boundary schedule because completion is a distinct
    regime from reaching the 18-second long-view threshold.
    """
    labels = np.asarray(labels, dtype=np.float32)
    play = np.asarray(play_time_ms, dtype=np.float64)
    duration = np.asarray(duration_ms, dtype=np.float64)
    if not (len(labels) == len(play) == len(duration)):
        raise ValueError("label/play/duration arrays must align")
    boundary = np.minimum(duration, 18_000.0)
    relative_distance = np.divide(np.abs(play - boundary), np.maximum(boundary, 1.0))
    near = relative_distance <= width
    short = duration < 18_000.0
    positive_eps = np.full(len(labels), far[0], dtype=np.float32)
    negative_eps = np.full(len(labels), far[1], dtype=np.float32)
    positive_eps[near & ~short], negative_eps[near & ~short] = long_near
    positive_eps[near & short], negative_eps[near & short] = short_near
    if np.any((positive_eps < 0) | (positive_eps >= 0.5) |
              (negative_eps < 0) | (negative_eps >= 0.5)):
        raise ValueError("smoothing rates must be in [0, 0.5)")
    return np.where(labels > 0.5, 1.0 - positive_eps, negative_eps).astype(np.float32)


def empirical_bayes_lambda(history_count: np.ndarray | float, tau: float) -> np.ndarray:
    """Canonical empirical-Bayes shrinkage n / (n + tau)."""
    count = np.asarray(history_count, dtype=np.float64)
    if tau <= 0 or np.any(count < 0):
        raise ValueError("tau must be positive and history counts nonnegative")
    return count / (count + tau)


def discrete_survival_targets(play_time_ms: np.ndarray, duration_ms: np.ndarray,
                              bins: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Event targets and at-risk mask up to min(duration, 18 seconds).

    Stops before the boundary are observed events. Reaching the boundary is
    right-censoring after surviving all intervals, so all event targets are zero.
    Intervals after an observed stop are excluded from the likelihood.
    """
    play = np.asarray(play_time_ms, dtype=np.float64)
    boundary = np.minimum(np.asarray(duration_ms, dtype=np.float64), 18_000.0)
    if bins < 1 or len(play) != len(boundary):
        raise ValueError("bins must be positive and arrays aligned")
    completed = play >= boundary
    fraction = np.divide(play, np.maximum(boundary, 1.0))
    event_bin = np.clip((fraction * bins).astype(np.int64), 0, bins - 1)
    grid = np.arange(bins)[None, :]
    risk = (grid <= event_bin[:, None]).astype(np.float32)
    risk[completed] = 1.0
    event = np.zeros_like(risk)
    event[np.flatnonzero(~completed), event_bin[~completed]] = 1.0
    return event, risk


def _set_seed(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    return np.random.default_rng(seed)


def _date_age(dates: np.ndarray) -> np.ndarray:
    end_value = int(np.max(dates))
    end = dt.date(end_value // 10000, end_value // 100 % 100, end_value % 100)
    return np.asarray([(end - dt.date(int(x) // 10000, int(x) // 100 % 100,
                                     int(x) % 100)).days for x in dates], dtype=np.float32)


def _recency_weights(dates: np.ndarray) -> np.ndarray:
    weights = np.exp2(-_date_age(dates) / 7.0).astype(np.float32)
    return weights / weights.mean()


def _subset(split: dict, mask: np.ndarray) -> dict:
    return {key: value[mask] for key, value in split.items()}


def _predict(model: DimsDCN, x: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        scores = torch.cat([model(x[start:start + 200_000])["main"]
                            for start in range(0, len(x), 200_000)]).cpu().numpy()
    model.train()
    return scores


def _id_probabilities(train: dict, p0: float, alpha: float,
                      oldest_boost: float) -> tuple[np.ndarray, np.ndarray]:
    result = []
    age = _date_age(train["date"])
    age_scale = age / max(float(age.max()), 1.0)
    for column in (0, 1):
        ids = train["X"][:, column]
        _, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
        probability = masking_probability(counts[inverse], p0, alpha)
        probability *= 1.0 + oldest_boost * age_scale
        result.append(np.clip(probability, 0.0, 1.0).astype(np.float32))
    return result[0], result[1]


def _masked_x(x: torch.Tensor, row_ids: np.ndarray, probabilities: tuple[np.ndarray, np.ndarray],
              unk_ids: tuple[int, int], rng: np.random.Generator) -> torch.Tensor:
    value = x[torch.as_tensor(row_ids, dtype=torch.long)].clone()
    for column in (0, 1):
        selected = rng.random(len(row_ids)) < probabilities[column][row_ids]
        if np.any(selected):
            value[torch.as_tensor(selected), column] = unk_ids[column]
    return value


def train_fixed(
    train: dict,
    total_dim: int,
    *,
    seed: int,
    half_epochs: int = 5,
    batch_size: int = 8192,
    point_targets: np.ndarray | None = None,
    extra_weights: np.ndarray | None = None,
    mask_config: tuple[float, float, float] | None = None,
    freeze_after_halves: int | None = None,
    max_runtime: int = 420,
) -> tuple[DimsDCN, dict]:
    """Train strong-L0 for a fixed horizon without reading official validation."""
    started = time.time()
    rng = _set_seed(seed)
    x = torch.as_tensor(np.ascontiguousarray(train["X"]), dtype=torch.long)
    labels = torch.as_tensor(train["y"] if point_targets is None else point_targets,
                             dtype=torch.float32)
    weights_np = _recency_weights(train["date"])
    if extra_weights is not None:
        weights_np = weights_np * np.asarray(extra_weights, dtype=np.float32)
        weights_np /= weights_np.mean()
    weights = torch.as_tensor(weights_np, dtype=torch.float32)
    added = 2 if mask_config is not None else 0
    model = DimsDCN(total_dim + added, x.shape[1], ())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sampler = PairSampler(train["user"], train["y"], train["X"][:, 1])
    probabilities = None
    unk_ids = (total_dim, total_dim + 1)
    if mask_config is not None:
        probabilities = _id_probabilities(train, *mask_config)
    order = np.arange(len(labels))
    losses: list[float] = []

    class Timeout(Exception):
        pass

    def timeout_handler(_signum, _frame):
        raise Timeout

    previous = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(max_runtime)
    completed = 0
    try:
        pair_pos, pair_neg = sampler.sample(rng, 1, "uniform")
        while completed < half_epochs:
            if completed and completed % 2 == 0:
                for group in optimizer.param_groups:
                    group["lr"] *= 0.5
                pair_pos, pair_neg = sampler.sample(rng, 1, "uniform")
            if freeze_after_halves is not None and completed == freeze_after_halves:
                model.embedding.weight.requires_grad_(False)
                dense = [p for p in model.parameters() if p.requires_grad]
                current_lr = optimizer.param_groups[0]["lr"] * 0.5
                optimizer = torch.optim.AdamW(dense, lr=current_lr, weight_decay=1e-5)
            order = rng.permutation(order) if completed % 2 == 0 else order
            half_size = math.ceil(len(order) / 2)
            half = completed % 2
            rows = order[half * half_size:min(len(order), (half + 1) * half_size)]
            batches = math.ceil(len(rows) / batch_size)
            half_losses = []
            for batch in range(batches):
                idx_np = rows[batch * batch_size:(batch + 1) * batch_size]
                idx = torch.as_tensor(idx_np, dtype=torch.long)
                batch_x = (x[idx] if probabilities is None else
                           _masked_x(x, idx_np, probabilities, unk_ids, rng))
                logits = model(batch_x)["main"]
                point = nn.functional.binary_cross_entropy_with_logits(
                    logits, labels[idx], reduction="none")
                point_loss = (point * weights[idx]).sum() / weights[idx].sum()
                # Match the control: one complete sampled pair set per epoch,
                # divided evenly over its two half-epochs.
                lo = half * len(pair_pos) // 2
                hi = (half + 1) * len(pair_pos) // 2
                p_np, n_np = pair_pos[lo:hi], pair_neg[lo:hi]
                # Divide this half's pair slice over its point batches.
                plo = batch * len(p_np) // batches
                phi = (batch + 1) * len(p_np) // batches
                p_np, n_np = p_np[plo:phi], n_np[plo:phi]
                if len(p_np):
                    px = (x[torch.as_tensor(p_np)] if probabilities is None else
                          _masked_x(x, p_np, probabilities, unk_ids, rng))
                    nx = (x[torch.as_tensor(n_np)] if probabilities is None else
                          _masked_x(x, n_np, probabilities, unk_ids, rng))
                    pair = nn.functional.softplus(model(nx)["main"] - model(px)["main"])
                    p = torch.as_tensor(p_np, dtype=torch.long)
                    n = torch.as_tensor(n_np, dtype=torch.long)
                    pair_weight = 0.5 * (weights[p] + weights[n])
                    pair_loss = (pair * pair_weight).sum() / pair_weight.sum()
                else:
                    pair_loss = point_loss * 0.0
                loss = 0.5 * point_loss + 0.5 * pair_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                half_losses.append(float(loss.detach()))
            completed += 1
            losses.append(float(np.mean(half_losses)))
            print(f"half={completed} loss={losses[-1]:.6f}", flush=True)
    except Timeout as exc:
        raise RuntimeError("training exceeded its per-run runtime cap") from exc
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    return model, {"runtime_s": round(time.time() - started, 1), "half_losses": losses,
                   "half_epochs": half_epochs}


def _official_metrics(users: np.ndarray, labels: np.ndarray,
                      scores: np.ndarray) -> dict[str, float]:
    raw = official_evaluate(users.astype(int).tolist(), labels.astype(int).tolist(),
                            scores.astype(float).tolist())
    return {"gauc": float(raw["GAUC"]), "ndcg5": float(raw["nDCG@5"]),
            "primary": float(raw["primary"])}


def _low_history_metrics(train: dict, valid: dict, scores: np.ndarray) -> tuple[dict, dict]:
    users, counts = np.unique(train["user"], return_counts=True)
    order = np.lexsort((users, counts))
    low_users = users[order[:math.ceil(len(users) / 3)]]
    mask = np.isin(valid["user"], low_users)
    info = {"definition": "bottom third of train users sorted by (history_count,user_id)",
            "train_users": int(len(users)), "low_users": int(len(low_users)),
            "validation_rows": int(mask.sum()), "validation_users": int(len(np.unique(valid["user"][mask])))}
    return _official_metrics(valid["user"][mask], valid["y"][mask], scores[mask]), info


def _write_result(out_dir: str, train: dict, valid: dict, scores: np.ndarray,
                  config: dict, training: dict, pseudo: list[dict] | None = None) -> dict:
    # This is the sole official-validation read in a finalized long-shot run.
    observed = _official_metrics(valid["user"], valid["y"], scores)
    low_metrics, low_info = _low_history_metrics(train, valid, scores)
    result = {**observed, "delta_vs_control": observed["primary"] - CONTROL_PRIMARY,
              "delta_vs_baseline": observed["primary"] - BASELINE_PRIMARY,
              "low_history": {**low_metrics, **low_info}, "config": config,
              "training": training, "pseudo_validation": pseudo or []}
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    with (target / "predictions.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(("row_id", "user_id", "video_id", "score"))
        for row_id, (user, video, score) in enumerate(zip(valid["user"], valid["videos"], scores)):
            writer.writerow((row_id, int(user), int(video), f"{float(score):.10f}"))
    (target / "metrics.json").write_text(json.dumps(result, sort_keys=True) + "\n",
                                          encoding="utf-8")
    print("final:", json.dumps(result, sort_keys=True), flush=True)
    return result


def _train_loss(model: DimsDCN, split: dict) -> np.ndarray:
    x = torch.as_tensor(np.ascontiguousarray(split["X"]), dtype=torch.long)
    logits = _predict(model, x)
    return np.maximum(logits, 0) - logits * split["y"] + np.log1p(np.exp(-np.abs(logits)))


def run_mask(args, ds: dict) -> dict:
    train, valid = ds["train"], ds["valid"]
    model, training = train_fixed(
        train, ds["field_dims_total"], seed=args.seed,
        half_epochs=args.half_epochs, mask_config=(args.p0, args.alpha, args.oldest_boost),
        freeze_after_halves=args.freeze_after_halves, max_runtime=args.max_runtime)
    scores = _predict(model, torch.as_tensor(np.ascontiguousarray(valid["X"]), dtype=torch.long))
    config = {"idea": "frequency_id_masking", "seed": args.seed, "p0": args.p0,
              "alpha": args.alpha, "oldest_boost": args.oldest_boost,
              "freeze_after_halves": args.freeze_after_halves,
              "half_epochs": args.half_epochs}
    return _write_result(args.out_dir, train, valid, scores, config, training)


def _rolling_tune(train: dict, total_dim: int, seed: int, caps: tuple[float, ...],
                  max_runtime: int) -> tuple[float, tuple[int, ...], list[dict]]:
    pilot_train = _subset(train, train["date"] <= 20220418)
    loss_window = _subset(train, (train["date"] >= 20220416) & (train["date"] <= 20220419))
    pilot, _ = train_fixed(pilot_train, total_dim, seed=seed, half_epochs=3,
                           max_runtime=max_runtime)
    window_losses = _train_loss(pilot, loss_window)
    _, worst_days = capped_day_weights(loss_window["date"], window_losses, cap=max(caps))
    fit = _subset(train, train["date"] <= 20220419)
    holdout = _subset(train, train["date"] >= 20220420)
    pseudo: list[dict] = []
    best = (-math.inf, caps[0])
    for cap in caps:
        extra = np.ones(len(fit["y"]), dtype=np.float32)
        extra[np.isin(fit["date"], worst_days)] = cap
        model, details = train_fixed(fit, total_dim, seed=seed, half_epochs=4,
                                     extra_weights=extra, max_runtime=max_runtime)
        scores = _predict(model, torch.as_tensor(np.ascontiguousarray(holdout["X"]), dtype=torch.long))
        observed = _official_metrics(holdout["user"], holdout["y"], scores)
        row = {"cap": cap, "worst_days": list(worst_days), **observed,
               "runtime_s": details["runtime_s"]}
        pseudo.append(row)
        if observed["primary"] > best[0]:
            best = observed["primary"], cap
    return best[1], worst_days, pseudo


def run_reweight(args, ds: dict) -> dict:
    train, valid = ds["train"], ds["valid"]
    if args.day_cap is None:
        cap, worst_days, pseudo = _rolling_tune(
            train, ds["field_dims_total"], args.seed, (1.5, 2.0, 3.0), args.max_runtime)
    else:
        cap = args.day_cap
        pseudo = []
        if not args.worst_days:
            raise ValueError("--worst-days is required when --day-cap bypasses rolling tuning")
        worst_days = tuple(args.worst_days)
    extra = np.ones(len(train["y"]), dtype=np.float32)
    extra[np.isin(train["date"], worst_days)] = cap
    model, training = train_fixed(train, ds["field_dims_total"], seed=args.seed,
                                  half_epochs=args.half_epochs, extra_weights=extra,
                                  max_runtime=args.max_runtime)
    scores = _predict(model, torch.as_tensor(np.ascontiguousarray(valid["X"]), dtype=torch.long))
    config = {"idea": "rolling_day_reweighting", "seed": args.seed, "day_cap": cap,
              "worst_days": list(worst_days), "last_days": 4, "worst_k": 1,
              "half_epochs": args.half_epochs}
    return _write_result(args.out_dir, train, valid, scores, config, training, pseudo)


def run_smooth(args, ds: dict) -> dict:
    train, valid = ds["train"], ds["valid"]
    targets = asymmetric_smoothed_targets(
        train["y"], train["play_time_ms"], train["duration_ms"],
        long_near=(args.long_pos, args.long_neg),
        short_near=(args.short_pos, args.short_neg), far=(args.far_pos, args.far_neg),
        width=args.boundary_width)
    model, training = train_fixed(train, ds["field_dims_total"], seed=args.seed,
                                  half_epochs=args.half_epochs, point_targets=targets,
                                  max_runtime=args.max_runtime)
    scores = _predict(model, torch.as_tensor(np.ascontiguousarray(valid["X"]), dtype=torch.long))
    config = {"idea": "threshold_label_smoothing", "seed": args.seed,
              "long_near": [args.long_pos, args.long_neg],
              "short_near": [args.short_pos, args.short_neg],
              "far": [args.far_pos, args.far_neg], "boundary_width": args.boundary_width,
              "half_epochs": args.half_epochs, "play_time_at_inference": False}
    return _write_result(args.out_dir, train, valid, scores, config, training)


def _representations(model: DimsDCN, x: np.ndarray) -> np.ndarray:
    indices = torch.as_tensor(np.ascontiguousarray(x[:, [1, 2]]), dtype=torch.long)
    with torch.no_grad():
        value = model.embedding(indices).sum(1).cpu().numpy()
    return value


def eb_adjustment(model: DimsDCN, history: dict, candidates: dict, tau: float) -> np.ndarray:
    """Frozen positive-minus-negative user prototype score adjustment."""
    history_repr = _representations(model, history["X"])
    size = int(max(np.max(history["user"]), np.max(candidates["user"]))) + 1
    positive_sum = np.zeros((size, history_repr.shape[1]), dtype=np.float64)
    negative_sum = np.zeros_like(positive_sum)
    positive_count = np.zeros(size, dtype=np.int64)
    negative_count = np.zeros(size, dtype=np.int64)
    positive = history["y"] > 0.5
    np.add.at(positive_sum, history["user"][positive], history_repr[positive])
    np.add.at(negative_sum, history["user"][~positive], history_repr[~positive])
    np.add.at(positive_count, history["user"][positive], 1)
    np.add.at(negative_count, history["user"][~positive], 1)
    positive_mean = positive_sum / np.maximum(positive_count[:, None], 1)
    negative_mean = negative_sum / np.maximum(negative_count[:, None], 1)
    prototypes = positive_mean - negative_mean
    history_count = positive_count + negative_count
    candidate_repr = _representations(model, candidates["X"])
    users = candidates["user"]
    dot = np.einsum("ij,ij->i", prototypes[users], candidate_repr)
    return (empirical_bayes_lambda(history_count[users], tau) * dot).astype(np.float32)


def _eb_tune(train: dict, total_dim: int, seed: int, max_runtime: int) -> tuple[float, list[dict]]:
    history = _subset(train, train["date"] <= 20220419)
    holdout = _subset(train, train["date"] >= 20220420)
    model, details = train_fixed(history, total_dim, seed=seed, half_epochs=4,
                                 max_runtime=max_runtime)
    x = torch.as_tensor(np.ascontiguousarray(holdout["X"]), dtype=torch.long)
    base = _predict(model, x)
    pseudo = []
    best = (-math.inf, 20.0)
    for tau in (20.0, 50.0):
        scores = base + eb_adjustment(model, history, holdout, tau)
        observed = _official_metrics(holdout["user"], holdout["y"], scores)
        pseudo.append({"tau": tau, **observed, "shared_base_runtime_s": details["runtime_s"]})
        if observed["primary"] > best[0]:
            best = observed["primary"], tau
    return best[1], pseudo


def run_eb(args, ds: dict) -> dict:
    train, valid = ds["train"], ds["valid"]
    tau, pseudo = ((args.tau, []) if args.tau is not None else
                   _eb_tune(train, ds["field_dims_total"], args.seed, args.max_runtime))
    model, training = train_fixed(train, ds["field_dims_total"], seed=args.seed,
                                  half_epochs=args.half_epochs, max_runtime=args.max_runtime)
    x = torch.as_tensor(np.ascontiguousarray(valid["X"]), dtype=torch.long)
    base = _predict(model, x)
    scores = base + args.adapter_scale * eb_adjustment(model, train, valid, tau)
    config = {"idea": "frozen_empirical_bayes_user_adapter", "seed": args.seed,
              "tau": tau, "adapter_scale": args.adapter_scale,
              "prototype": "positive mean minus negative mean of item+author embedding vectors",
              "half_epochs": args.half_epochs}
    return _write_result(args.out_dir, train, valid, scores, config, training, pseudo)


class SurvivalDCN(nn.Module):
    """Strong-L0 trunk with four conditional stop hazards."""

    def __init__(self, total_dim: int, fields: int, bins: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(total_dim, 16)
        nn.init.normal_(self.embedding.weight, std=0.01)
        self.embedding_dropout = nn.Dropout(0.1)
        width = fields * 16
        self.cross = nn.Linear(width, width)
        self.mlp = nn.Sequential(nn.Linear(width, 128), nn.ReLU(), nn.Dropout(0.2))
        self.hazard = nn.Linear(128, bins)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.embedding_dropout(self.embedding(x)).flatten(1)
        return self.hazard(self.mlp(x0 * self.cross(x0) + x0))


def _survival_score(hazards: torch.Tensor) -> torch.Tensor:
    return nn.functional.logsigmoid(-hazards).sum(1)


def run_survival(args, ds: dict) -> dict:
    started = time.time()
    rng = _set_seed(args.seed)
    train, valid = ds["train"], ds["valid"]
    x = torch.as_tensor(np.ascontiguousarray(train["X"]), dtype=torch.long)
    xv = torch.as_tensor(np.ascontiguousarray(valid["X"]), dtype=torch.long)
    event_np, risk_np = discrete_survival_targets(train["play_time_ms"], train["duration_ms"])
    event = torch.as_tensor(event_np)
    risk = torch.as_tensor(risk_np)
    weights = torch.as_tensor(_recency_weights(train["date"]))
    model = SurvivalDCN(ds["field_dims_total"], x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sampler = PairSampler(train["user"], train["y"], train["X"][:, 1])
    order = np.arange(len(x))
    losses = []
    pair_pos, pair_neg = sampler.sample(rng, 1, "uniform")
    for completed in range(args.half_epochs):
        if completed and completed % 2 == 0:
            optimizer.param_groups[0]["lr"] *= 0.5
            pair_pos, pair_neg = sampler.sample(rng, 1, "uniform")
        if completed % 2 == 0:
            order = rng.permutation(order)
        half = completed % 2
        half_size = math.ceil(len(order) / 2)
        rows = order[half * half_size:min(len(order), (half + 1) * half_size)]
        batches = math.ceil(len(rows) / 8192)
        half_losses = []
        for batch in range(batches):
            idx_np = rows[batch * 8192:(batch + 1) * 8192]
            idx = torch.as_tensor(idx_np, dtype=torch.long)
            hazards = model(x[idx])
            interval = nn.functional.binary_cross_entropy_with_logits(
                hazards, event[idx], reduction="none")
            nll = (interval * risk[idx]).sum(1) / risk[idx].sum(1)
            point_loss = (nll * weights[idx]).sum() / weights[idx].sum()
            p_all = pair_pos[half * len(pair_pos) // 2:(half + 1) * len(pair_pos) // 2]
            n_all = pair_neg[half * len(pair_neg) // 2:(half + 1) * len(pair_neg) // 2]
            lo, hi = batch * len(p_all) // batches, (batch + 1) * len(p_all) // batches
            p = torch.as_tensor(p_all[lo:hi], dtype=torch.long)
            n = torch.as_tensor(n_all[lo:hi], dtype=torch.long)
            if len(p):
                pair = nn.functional.softplus(_survival_score(model(x[n])) -
                                              _survival_score(model(x[p])))
                pair_weight = 0.5 * (weights[p] + weights[n])
                pair_loss = (pair * pair_weight).sum() / pair_weight.sum()
            else:
                pair_loss = point_loss * 0.0
            loss = 0.5 * point_loss + 0.5 * pair_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            half_losses.append(float(loss.detach()))
        losses.append(float(np.mean(half_losses)))
        print(f"half={completed + 1} loss={losses[-1]:.6f}", flush=True)
    model.eval()
    with torch.no_grad():
        scores = torch.cat([_survival_score(model(xv[i:i + 200_000]))
                            for i in range(0, len(xv), 200_000)]).cpu().numpy()
    training = {"runtime_s": round(time.time() - started, 1), "half_losses": losses,
                "half_epochs": args.half_epochs}
    config = {"idea": "four_bin_discrete_survival", "seed": args.seed, "bins": 4,
              "boundary": "min(duration_ms,18000)", "censoring": "right at boundary",
              "ranking_score": "log product of interval survival probabilities",
              "loss": "0.5 hazard NLL + 0.5 within-user BPR", "half_epochs": args.half_epochs,
              "play_time_at_inference": False}
    return _write_result(args.out_dir, train, valid, scores, config, training)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/real_ws")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--idea", required=True,
                    choices=("mask", "reweight", "smooth", "eb", "survival"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--half-epochs", type=int, default=5)
    ap.add_argument("--max-runtime", type=int, default=420)
    ap.add_argument("--p0", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--oldest-boost", type=float, default=0.0)
    ap.add_argument("--freeze-after-halves", type=int)
    ap.add_argument("--day-cap", type=float)
    ap.add_argument("--worst-days", type=int, nargs="*")
    ap.add_argument("--long-pos", type=float, default=0.10)
    ap.add_argument("--long-neg", type=float, default=0.03)
    ap.add_argument("--short-pos", type=float, default=0.05)
    ap.add_argument("--short-neg", type=float, default=0.015)
    ap.add_argument("--far-pos", type=float, default=0.01)
    ap.add_argument("--far-neg", type=float, default=0.003)
    ap.add_argument("--boundary-width", type=float, default=0.20)
    ap.add_argument("--tau", type=float)
    ap.add_argument("--adapter-scale", type=float, default=1.0)
    return ap


def run(args) -> dict:
    if args.half_epochs < 1:
        raise ValueError("half epochs must be positive")
    ds = load_data(args.data_dir, ())
    runners = {"mask": run_mask, "reweight": run_reweight,
               "smooth": run_smooth, "eb": run_eb, "survival": run_survival}
    return runners[args.idea](args, ds)


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
