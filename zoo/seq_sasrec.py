"""Validation-only SASRec/GRU sequence ranker with the zoo metrics contract."""

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
from typing import Any

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.official.evaluate import evaluate as official_evaluate
from zoo.seq_data import assert_no_leakage, load_or_prepare

FROZEN_BASELINE = 0.6047


def parser(description: str = __doc__) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--data-dir", default="data/real_ws")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--encoder", choices=("transformer", "gru"), default="transformer")
    ap.add_argument("--outcome-marks", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--max-history", type=int, choices=(20, 50), default=50)
    ap.add_argument("--k", type=int, choices=(16, 32), default=32)
    ap.add_argument("--blocks", type=int, choices=(1, 2), default=1)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--predict-batch-size", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--step-decay-factor", type=float, default=0.5)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--recency-half-life", type=float, default=7.0)
    ap.add_argument("--bpr-weight", type=float, default=0.5)
    ap.add_argument("--patience-halves", type=int, default=4)
    ap.add_argument("--max-runtime", type=int, default=470)
    ap.add_argument("--subsample", type=int, default=None)
    ap.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    return ap


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device(name: str) -> torch.device:
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.device(name)


def official_metrics(users: np.ndarray, labels: np.ndarray,
                     scores: np.ndarray) -> dict[str, float]:
    measured = official_evaluate(users.tolist(), labels.astype(int).tolist(), scores.tolist())
    return {"gauc": float(measured["GAUC"]), "ndcg5": float(measured["nDCG@5"]),
            "primary": float(measured["primary"])}


def recency_weights(dates: np.ndarray, half_life: float) -> np.ndarray:
    if half_life <= 0:
        raise ValueError("recency half-life must be positive")
    end = dt.date(2022, 4, 21)
    ages = np.asarray([(end - dt.date(int(value) // 10000,
                                    int(value) // 100 % 100,
                                    int(value) % 100)).days for value in dates])
    if np.any(ages < 0):
        raise ValueError("recency weighting saw a post-training date")
    result = np.exp2(-ages / half_life).astype(np.float32)
    return result / result.mean()


class PairSampler:
    def __init__(self, users: np.ndarray, labels: np.ndarray):
        order = np.argsort(users, kind="stable")
        sorted_users, sorted_labels = users[order], labels[order]
        starts = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1]])
        ends = np.r_[starts[1:], len(order)]
        positives, negatives, offsets, counts = [], [], [], []
        offset = 0
        for start, end in zip(starts, ends):
            rows = order[start:end]
            pos, neg = rows[sorted_labels[start:end] == 1], rows[sorted_labels[start:end] == 0]
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
        negative = self.negatives[self.offsets + rng.integers(0, self.counts)]
        permutation = rng.permutation(len(self.positives))
        return self.positives[permutation], negative[permutation]


class SequenceRanker(nn.Module):
    def __init__(self, n_video: int, n_author: int, n_duration: int, max_history: int,
                 k: int = 32, blocks: int = 1, dropout: float = 0.25,
                 outcome_marks: bool = False, encoder: str = "transformer"):
        super().__init__()
        self.max_history = max_history
        self.k = k
        self.outcome_marks = outcome_marks
        self.encoder_name = encoder
        self.video = nn.Embedding(n_video, k, padding_idx=0)
        self.author = nn.Embedding(n_author, k, padding_idx=0)
        self.duration = nn.Embedding(n_duration, k, padding_idx=0)
        self.context = nn.Linear(2 * k, k, bias=False)
        self.position = nn.Embedding(max_history, k)
        self.outcome = nn.Embedding(3, k, padding_idx=0) if outcome_marks else None
        self.cold = nn.Parameter(torch.zeros(k))
        self.input_dropout = nn.Dropout(dropout)
        if encoder == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=k, nhead=2, dim_feedforward=4 * k, dropout=dropout,
                activation="gelu", batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=blocks,
                                                 enable_nested_tensor=False)
            causal = torch.triu(torch.ones(max_history, max_history, dtype=torch.bool), diagonal=1)
            self.register_buffer("causal_mask", causal, persistent=False)
        elif encoder == "gru":
            self.encoder = nn.GRU(k, k, num_layers=blocks, batch_first=True,
                                  dropout=dropout if blocks > 1 else 0.0)
        else:
            raise ValueError(f"unknown encoder: {encoder}")
        self.norm = nn.LayerNorm(k)
        self.video_bias = nn.Embedding(n_video, 1, padding_idx=0)
        self.author_bias = nn.Embedding(n_author, 1, padding_idx=0)
        self.duration_bias = nn.Embedding(n_duration, 1, padding_idx=0)
        self.global_bias = nn.Parameter(torch.zeros(()))
        for embedding in (self.video, self.author, self.duration):
            nn.init.normal_(embedding.weight, std=0.02)

    def forward(self, hist_video: torch.Tensor, hist_author: torch.Tensor,
                hist_duration: torch.Tensor, hist_outcome: torch.Tensor,
                candidate_video: torch.Tensor, candidate_author: torch.Tensor,
                candidate_duration: torch.Tensor) -> torch.Tensor:
        padding = hist_video.eq(0)
        lengths = (~padding).sum(dim=1)
        position = torch.arange(hist_video.shape[1], device=hist_video.device)
        history = self.video(hist_video) + self.context(torch.cat(
            (self.author(hist_author), self.duration(hist_duration)), dim=-1))
        history = history + self.position(position)[None, :, :]
        if self.outcome is not None:
            history = history + self.outcome(hist_outcome)
        empty = lengths.eq(0)
        history = history.masked_fill(padding[:, :, None], 0.0)
        if empty.any():
            history = history.clone()
            history[empty, 0] = self.cold + self.position.weight[0]
        history = self.input_dropout(history)
        if self.encoder_name == "transformer":
            # Sequences are left-compacted and right-padded. A valid query can
            # only see positions <= itself under this mask, so every pad lies
            # in its future. Avoiding the redundant key-padding mask also avoids
            # NaN gradients in PyTorch's MPS fused attention kernel.
            encoded = self.encoder(history, mask=self.causal_mask[:history.shape[1], :history.shape[1]])
        else:
            encoded, _ = self.encoder(history)
        state_index = torch.clamp(lengths - 1, min=0)
        state = self.norm(encoded[torch.arange(len(lengths), device=lengths.device), state_index])
        candidate = self.video(candidate_video) + self.context(torch.cat(
            (self.author(candidate_author), self.duration(candidate_duration)), dim=-1))
        score = (state * candidate).sum(dim=-1) / math.sqrt(self.k)
        score = score + self.video_bias(candidate_video).squeeze(-1)
        score = score + self.author_bias(candidate_author).squeeze(-1)
        score = score + self.duration_bias(candidate_duration).squeeze(-1) + self.global_bias
        return score


def _batch(data: dict[str, np.ndarray], split: str, rows: np.ndarray,
           device: torch.device) -> tuple[torch.Tensor, ...]:
    history = data[f"{split}_history"][rows]
    lengths = (history >= 0).sum(axis=1)
    width = history.shape[1]
    positions = np.arange(width)[None, :]
    source_positions = positions + (width - lengths)[:, None]
    valid = positions < lengths[:, None]
    source_positions = np.minimum(source_positions, width - 1)
    compact = np.take_along_axis(history, source_positions, axis=1)
    compact = np.where(valid, compact, -1)
    safe = np.where(compact >= 0, compact, 0)

    def history_feature(name: str, outcome: bool = False) -> torch.Tensor:
        values = data[f"train_{name}"][safe]
        if outcome:
            values = values + 1
        values = np.where(valid, values, 0)
        return torch.as_tensor(values, dtype=torch.long, device=device)

    return (history_feature("video"), history_feature("author"), history_feature("duration"),
            history_feature("outcome", True),
            torch.as_tensor(data[f"{split}_video"][rows], dtype=torch.long, device=device),
            torch.as_tensor(data[f"{split}_author"][rows], dtype=torch.long, device=device),
            torch.as_tensor(data[f"{split}_duration"][rows], dtype=torch.long, device=device))


def _configuration(args: argparse.Namespace) -> dict[str, Any]:
    excluded = {"data_dir", "cache", "out_dir"}
    return {key: value for key, value in vars(args).items() if key not in excluded}


def train_and_report(data: dict[str, np.ndarray], args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    assert_no_leakage(data)
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = choose_device(args.device)
    n_train = len(data["train_user"]) if args.subsample is None else min(args.subsample, len(data["train_user"]))
    n_valid = len(data["valid_user"]) if args.subsample is None else min(args.subsample, len(data["valid_user"]))
    train_rows = np.arange(n_train, dtype=np.int64)
    valid_rows = np.arange(n_valid, dtype=np.int64)
    labels = torch.as_tensor(data["train_outcome"][:n_train], dtype=torch.float32, device=device)
    weights = torch.as_tensor(recency_weights(data["train_date"][:n_train], args.recency_half_life),
                              dtype=torch.float32, device=device)
    model = SequenceRanker(int(data["n_video"]), int(data["n_author"]),
                           int(data["n_duration"]), args.max_history, args.k, args.blocks,
                           args.dropout, args.outcome_marks, args.encoder).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sampler = PairSampler(data["train_user"][:n_train], data["train_outcome"][:n_train])

    def predict() -> np.ndarray:
        model.eval()
        chunks = []
        with torch.inference_mode():
            for start in range(0, n_valid, args.predict_batch_size):
                rows = valid_rows[start:start + args.predict_batch_size]
                chunks.append(model(*_batch(data, "valid", rows, device)).detach().cpu())
        model.train()
        return torch.cat(chunks).numpy()

    best_primary, best_epoch, best_state = -math.inf, 0.0, None
    history: list[dict[str, float]] = []
    bad = 0
    timed_out = False

    class RunTimeout(Exception):
        pass

    def timeout_handler(_signum: int, _frame: Any) -> None:
        raise RunTimeout

    previous_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(args.max_runtime)
    try:
        for epoch in range(args.epochs):
            order = rng.permutation(train_rows)
            pair_pos, pair_neg = sampler.sample(rng)
            half_size = math.ceil(n_train / 2)
            for half in range(2):
                selected = order[half * half_size:min(n_train, (half + 1) * half_size)]
                losses = []
                batches = math.ceil(len(selected) / args.batch_size)
                for batch_id in range(batches):
                    point_np = selected[batch_id * args.batch_size:(batch_id + 1) * args.batch_size]
                    pair_begin = ((half * batches + batch_id) * len(pair_pos)) // (2 * batches)
                    pair_end = ((half * batches + batch_id + 1) * len(pair_pos)) // (2 * batches)
                    pos_np, neg_np = pair_pos[pair_begin:pair_end], pair_neg[pair_begin:pair_end]
                    all_rows = np.concatenate((point_np, pos_np, neg_np))
                    logits = model(*_batch(data, "train", all_rows, device))
                    point_count, pair_count = len(point_np), len(pos_np)
                    point_raw = nn.functional.binary_cross_entropy_with_logits(
                        logits[:point_count], labels[point_np], reduction="none")
                    point_weight = weights[point_np]
                    point_loss = (point_raw * point_weight).sum() / point_weight.sum()
                    if pair_count:
                        pair_raw = nn.functional.softplus(
                            logits[point_count + pair_count:] - logits[point_count:point_count + pair_count])
                        pair_weight = 0.5 * (weights[pos_np] + weights[neg_np])
                        pair_loss = (pair_raw * pair_weight).sum() / pair_weight.sum()
                    else:
                        pair_loss = point_loss * 0
                    loss = (1 - args.bpr_weight) * point_loss + args.bpr_weight * pair_loss
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.detach()))
                epoch_value = epoch + (half + 1) / 2
                scores = predict()
                metrics = official_metrics(data["valid_user"][:n_valid],
                                           data["valid_label"][:n_valid], scores)
                entry = {"epoch": epoch_value, "train_loss": float(np.mean(losses)),
                         "lr": float(optimizer.param_groups[0]["lr"]),
                         "val_gauc": metrics["gauc"], "val_ndcg5": metrics["ndcg5"],
                         "val_primary": metrics["primary"]}
                history.append(entry)
                print(f"epoch {epoch_value:.1f} | loss {entry['train_loss']:.4f} | "
                      f"lr {entry['lr']:.7g} | valid primary {metrics['primary']:.6f}", flush=True)
                if metrics["primary"] > best_primary + 1e-5:
                    best_primary, best_epoch = metrics["primary"], epoch_value
                    best_state, bad = copy.deepcopy(model.state_dict()), 0
                else:
                    bad += 1
                optimizer.param_groups[0]["lr"] *= args.step_decay_factor
                if bad >= args.patience_halves:
                    break
            if bad >= args.patience_halves:
                break
    except RunTimeout:
        timed_out = True
        print(f"runtime alarm reached at {args.max_runtime}s; preserving best checkpoint", flush=True)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
    if best_state is None:
        raise RuntimeError("runtime cap expired before the first checkpoint")
    model.load_state_dict(best_state)
    scores = predict()
    metrics: dict[str, Any] = official_metrics(data["valid_user"][:n_valid],
                                               data["valid_label"][:n_valid], scores)
    metrics.update({"history": history, "runtime_s": round(time.time() - started, 1),
                    "best_epoch": best_epoch, "config": _configuration(args),
                    "device": str(device), "timed_out": timed_out,
                    "delta_vs_frozen_baseline": metrics["primary"] - FROZEN_BASELINE,
                    "leakage_check": "passed"})
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "scores.npy", scores)
    with (out / "predictions.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(("row_id", "user_id", "score"))
        for row_id, (user, score) in enumerate(zip(data["valid_user"][:n_valid], scores)):
            writer.writerow((row_id, int(user), f"{float(score):.10f}"))
    with (out / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, sort_keys=True)
        fh.write("\n")
    print("final:", json.dumps({key: value for key, value in metrics.items() if key != "history"},
                                sort_keys=True), flush=True)
    return metrics


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size < 2048:
        raise ValueError("batch size must be at least 2048")
    if not 0 <= args.bpr_weight <= 1:
        raise ValueError("BPR weight must be in [0, 1]")
    cache = args.cache or str(Path("/tmp") / f"mle_seq_cache_h{args.max_history}.npz")
    data = load_or_prepare(args.data_dir, args.max_history, cache)
    return train_and_report(data, args)


def per_user_ranks(users: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Return [0,1] ordinal ranks computed separately inside each user."""
    result = np.zeros(len(scores), dtype=np.float64)
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    starts = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1]])
    ends = np.r_[starts[1:], len(order)]
    for start, end in zip(starts, ends):
        rows = order[start:end]
        local_order = np.argsort(scores[rows], kind="stable")
        denominator = max(1, len(rows) - 1)
        result[rows[local_order]] = np.arange(len(rows), dtype=np.float64) / denominator
    return result


def rank_average(users: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if len(users) != len(left) or len(left) != len(right):
        raise ValueError("ensemble arrays have different lengths")
    return 0.5 * (per_user_ranks(users, left) + per_user_ranks(users, right))


def history_segments(users: np.ndarray, labels: np.ndarray, scores: np.ndarray,
                     history_counts: np.ndarray) -> dict[str, Any]:
    user_count: dict[int, int] = {}
    for user, count in zip(users, history_counts):
        user_count.setdefault(int(user), int(count))
    threshold = float(np.median(list(user_count.values())))
    low = np.asarray([user_count[int(user)] <= threshold for user in users])
    return {"threshold": threshold,
            "low": {**official_metrics(users[low], labels[low], scores[low]),
                    "users": int(len(np.unique(users[low]))), "rows": int(low.sum())},
            "high": {**official_metrics(users[~low], labels[~low], scores[~low]),
                     "users": int(len(np.unique(users[~low]))), "rows": int((~low).sum())}}


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
