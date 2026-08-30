"""Retrain the Pure champion and compare validation-only combination rules."""

import os
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.official.evaluate import evaluate
from tools.predict_test_bc07 import (
    load_npz,
    make_pair_pool,
    metric_values,
    per_user_ranks,
    train_variant,
)


TRAIN_PATH = ROOT / "data/real_ws/train.npz"
VAL_PATH = ROOT / "data/real_ws/val.npz"
SEEDS = (42, 43, 44)


def user_groups(users):
    users = np.asarray(users)
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    boundaries = np.r_[
        0,
        np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1,
        len(order),
    ]
    return order, boundaries


def margin_temperature(scores, order, boundaries):
    margins = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        local = np.asarray(scores[order[left:right]], dtype=np.float64)
        if len(local) < 2:
            continue
        diff = np.abs(local[:, None] - local[None, :])
        upper = diff[np.triu_indices(len(local), 1)]
        nonzero = upper[upper > 0.0]
        if len(nonzero):
            margins.append(nonzero)
    if not margins:
        raise RuntimeError("no nonzero within-user pair margins")
    temperature = float(np.median(np.concatenate(margins)))
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise RuntimeError(f"invalid margin temperature: {temperature}")
    return temperature


def soft_copeland(scores, temperature, order, boundaries):
    output = np.empty(len(scores), dtype=np.float64)
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        indices = order[left:right]
        local = np.asarray(scores[indices], dtype=np.float64)
        scaled = np.clip(
            (local[:, None] - local[None, :]) / temperature, -60.0, 60.0
        )
        # The self-win is the same 0.5 constant for every item, so omit it.
        output[indices] = (1.0 / (1.0 + np.exp(-scaled))).sum(axis=1) - 0.5
    return output


def tie_broken_user_ranks(aggregate, anchor, order, boundaries):
    output = np.empty(len(aggregate), dtype=np.float64)
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        indices = order[left:right]
        # lexsort's last key is primary: aggregate first, anchor on exact ties.
        local_order = np.lexsort((anchor[indices], aggregate[indices]))
        ranks = np.empty(len(indices), dtype=np.float64)
        ranks[local_order] = np.arange(len(indices), dtype=np.float64)
        if len(indices) > 1:
            ranks /= float(len(indices) - 1)
        else:
            ranks.fill(0.5)
        output[indices] = ranks
    return output


def jittered_user_ranks(scores, users, noise):
    jittered = np.asarray(scores, dtype=np.float64) + noise
    return per_user_ranks(jittered, users).astype(np.float64)


def primary(scores, users, labels):
    value = float(evaluate(users, labels, scores)["primary"])
    if not np.isfinite(value):
        raise RuntimeError("non-finite validation primary")
    return value


def main():
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    device = torch.device("cpu")

    tr = load_npz(TRAIN_PATH)
    va = load_npz(VAL_PATH)
    X = torch.from_numpy(np.asarray(tr["X"], dtype=np.int64))
    y = torch.from_numpy(np.asarray(tr["y"], dtype=np.float32))
    Xv = torch.from_numpy(np.asarray(va["X"], dtype=np.int64))
    vy = np.asarray(va["y"], dtype=np.int64)
    vu = np.asarray(va["user"])
    dates = np.asarray(tr["date"])
    train_users = np.asarray(tr["user"])
    pair_data = make_pair_pool(train_users, np.asarray(tr["y"]))
    total_dim = int(np.asarray(tr["field_dims"]).sum())

    config = {
        "dropout": 0.18,
        "weight_decay": 9e-05,
        "gamma": 0.57,
        "step_size": 2,
        "half_life": 7.0,
        "lr": 0.001,
        "total_dim": total_dim,
        "batch_size": 16384,
    }

    member_scores = []
    member_primaries = []
    for seed in SEEDS:
        metrics, validation_scores, _, _ = train_variant(
            config,
            seed,
            12,
            X,
            y,
            dates,
            pair_data,
            Xv,
            vu,
            vy,
            evaluate,
            device,
            Xv[:1],
            half_checkpoints=True,
        )
        validation_scores = np.asarray(validation_scores, dtype=np.float64)
        if not np.all(np.isfinite(validation_scores)):
            raise RuntimeError(f"seed {seed} produced non-finite validation scores")
        member_scores.append(validation_scores)
        member_primaries.append(float(metrics[2]))
        print(f"member seed={seed} primary={metrics[2]:.9f}", flush=True)

    if any(np.array_equal(member_scores[0], scores) for scores in member_scores[1:]):
        raise RuntimeError("two member prediction arrays are identical")

    hard_ranks = np.stack([per_user_ranks(scores, vu) for scores in member_scores])
    hard_primary = primary(hard_ranks.mean(axis=0), vu, vy)
    print(f"hard_rank_average primary={hard_primary:.9f}", flush=True)

    best_index = int(np.argmax(member_primaries))
    other_indices = [i for i in range(len(SEEDS)) if i != best_index]
    order, boundaries = user_groups(vu)
    temperatures = [
        margin_temperature(scores, order, boundaries) for scores in member_scores
    ]
    copeland = [
        soft_copeland(scores, temperature, order, boundaries)
        for scores, temperature in zip(member_scores, temperatures)
    ]
    soft_aggregate = (
        0.6 * copeland[best_index]
        + 0.2 * copeland[other_indices[0]]
        + 0.2 * copeland[other_indices[1]]
    )
    anchored_soft = tie_broken_user_ranks(
        soft_aggregate, member_scores[best_index], order, boundaries
    )
    anchored_primary = primary(anchored_soft, vu, vy)
    print(
        f"best_anchored_soft anchor_seed={SEEDS[best_index]} "
        f"primary={anchored_primary:.9f}",
        flush=True,
    )

    noise = np.random.default_rng(20260830).uniform(
        -0.5e-12, 0.5e-12, size=len(vu)
    )
    soft_ranks = [jittered_user_ranks(scores, vu, noise) for scores in member_scores]
    fallback = 0.6 * soft_ranks[best_index] + 0.4 * np.mean(
        [soft_ranks[i] for i in other_indices], axis=0
    )
    fallback_primary = primary(fallback, vu, vy)
    print(f"soft_rank_blend primary={fallback_primary:.9f}", flush=True)


if __name__ == "__main__":
    main()
