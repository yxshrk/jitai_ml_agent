"""Train fixed members on train only and create an official test submission.

This pipeline reads test FEATURES only to produce predictions (the required
submission artifact); labels are never loaded; it is intended to run only at
submission time.
"""

from __future__ import annotations

import argparse
import csv
import math
import shlex
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evidence.submission import check as check_submission
from zoo.ensemble_node import _within_user_ranks
from zoo.polish_stack import DCNLite, FROZEN_FIELDS, PairSampler, parser as member_parser
from zoo.polish_stack import recency_weights, set_seed

PURE_MEMBERS = (46, 74, 93, 91, 60)
ONE_K_MEMBERS = (42, 43, 44, 45, 46)
EXPECTED_PURE_TEST_ROWS = 170_588
ONE_K_DEFAULTS = {
    "lr": 0.00168,
    "dropout": 0.21,
    "weight_decay": 0.000037,
    "k": 24,
    "recency_half_life": 7.0,
    "epochs": 6,
}


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def parse_members(value: str) -> list[int]:
    result = []
    for token in value.split(","):
        cleaned = token.strip().lower()
        if cleaned.startswith("seed"):
            cleaned = cleaned[4:]
        if not cleaned:
            raise argparse.ArgumentTypeError("members must be a comma-separated seed list")
        try:
            result.append(int(cleaned))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid member seed {token!r}") from exc
    if len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("member seeds must be unique")
    return result


def _member_config(dataset: str, member_args: str) -> argparse.Namespace:
    ap = member_parser("Fixed train-only submission member")
    if dataset == "1k":
        ap.set_defaults(**ONE_K_DEFAULTS)
    args = ap.parse_args(["--out-dir", "/tmp/predict-test-unused", *shlex.split(member_args)])
    if args.subsample is not None:
        raise ValueError("--subsample is forbidden for final test prediction")
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    return args


def load_train_only(data_dir: Path) -> dict[str, Any]:
    """Load only train.npz. This function has no validation path or fallback."""
    path = data_dir / "train.npz"
    if not path.is_file():
        raise FileNotFoundError(f"missing training archive: {path}")
    with np.load(path, allow_pickle=False) as archive:
        required = {"X", "y", "user", "date", "field_dims"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{path} missing arrays: {sorted(missing)}")
        train = {name: np.asarray(archive[name]) for name in required}
    dates = train["date"]
    if dates.size and (int(dates.min()) < 20220408 or int(dates.max()) > 20220421):
        raise ValueError(f"forbidden date in train: {dates.min()}..{dates.max()}")
    if train["X"].ndim != 2 or train["X"].shape[1] != len(FROZEN_FIELDS):
        raise ValueError(f"expected five frozen fields, got shape {train['X'].shape}")
    return {
        "X": train["X"].astype(np.int32, copy=False),
        "y": train["y"].astype(np.float32, copy=False),
        "users": train["user"].astype(np.int64, copy=False),
        "date": dates.astype(np.int32, copy=False),
        "field_dims": train["field_dims"].astype(np.int64, copy=False),
    }


def load_test_features(path: Path, field_dims: np.ndarray) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"missing test feature archive: {path}")
    with np.load(path, allow_pickle=False) as archive:
        expected = {"X", "user_id", "video_id", "date"}
        if set(archive.files) != expected:
            raise ValueError(
                f"test archive must contain only {sorted(expected)}; got {archive.files}"
            )
        result = {name: np.asarray(archive[name]) for name in expected}
    lengths = {len(values) for values in result.values()}
    if len(lengths) != 1:
        raise ValueError("test feature arrays have different lengths")
    X = result["X"]
    if X.ndim != 2 or X.shape[1] != len(field_dims):
        raise ValueError(f"test X shape {X.shape} does not match {len(field_dims)} fields")
    if X.size and (X.min() < 0 or X.max() >= int(field_dims.sum())):
        raise ValueError("test X contains an out-of-vocabulary encoded index")
    dates = result["date"]
    if dates.size and (int(dates.min()) < 20220429 or int(dates.max()) > 20220508):
        raise ValueError(f"forbidden date in test: {dates.min()}..{dates.max()}")
    result["X"] = X.astype(np.int32, copy=False)
    result["user_id"] = result["user_id"].astype(np.int64, copy=False)
    result["video_id"] = result["video_id"].astype(np.int64, copy=False)
    return result


def train_member(train: dict[str, np.ndarray], test_x: np.ndarray,
                 args: argparse.Namespace, seed: int) -> np.ndarray:
    """Fit for a fixed epoch count; validation is neither loaded nor consulted."""
    set_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device(args.device)
    train_array = (train["X"] if train["X"].flags["C_CONTIGUOUS"]
                   else np.ascontiguousarray(train["X"]))
    test_array = test_x if test_x.flags["C_CONTIGUOUS"] else np.ascontiguousarray(test_x)
    train_x = torch.as_tensor(train_array, dtype=torch.int32)
    train_y = torch.as_tensor(train["y"], dtype=torch.float32)
    weights = torch.as_tensor(recency_weights(train["date"], args.recency_half_life))
    test_tensor = torch.as_tensor(test_array, dtype=torch.int32)
    model_device = None if device.type == "cpu" else device
    model = DCNLite(int(train["field_dims"].sum()), train_x.shape[1], args.k,
                    args.dropout, args.embedding_dropout, model_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sampler = PairSampler(train["users"], train["y"])

    def transfer(value: torch.Tensor) -> torch.Tensor:
        return value if device.type == "cpu" else value.to(device)

    order = np.arange(len(train_y))
    half_size = math.ceil(len(train_y) / 2)
    interval_halves = int(round(args.decay_every * 2))
    start_halves = int(round(args.decay_start_epoch * 2))
    completed_halves = 0
    model.train()
    for epoch in range(args.epochs):
        order = rng.permutation(order)
        pair_pos, pair_neg = sampler.sample(rng)
        for half in range(2):
            half_rows = order[half * half_size:min(len(order), (half + 1) * half_size)]
            batches = math.ceil(len(half_rows) / args.batch_size)
            for batch in range(batches):
                ids = torch.as_tensor(
                    half_rows[batch * args.batch_size:(batch + 1) * args.batch_size],
                    dtype=torch.long,
                )
                logits = model(transfer(train_x[ids]).long())
                batch_y, batch_weights = transfer(train_y[ids]), transfer(weights[ids])
                point = nn.functional.binary_cross_entropy_with_logits(
                    logits, batch_y, reduction="none")
                point_loss = (point * batch_weights).sum() / batch_weights.sum()
                pair_begin = ((half * batches + batch) * len(pair_pos)) // (2 * batches)
                pair_end = ((half * batches + batch + 1) * len(pair_pos)) // (2 * batches)
                if pair_end > pair_begin:
                    positive = torch.as_tensor(pair_pos[pair_begin:pair_end], dtype=torch.long)
                    negative = torch.as_tensor(pair_neg[pair_begin:pair_end], dtype=torch.long)
                    pair = nn.functional.softplus(
                        model(transfer(train_x[negative]).long())
                        - model(transfer(train_x[positive]).long())
                    )
                    pair_weights = 0.5 * transfer(weights[positive] + weights[negative])
                    pair_loss = (pair * pair_weights).sum() / pair_weights.sum()
                else:
                    pair_loss = point_loss * 0.0
                loss = (1.0 - args.bpr_weight) * point_loss + args.bpr_weight * pair_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            completed_halves += 1
            should_decay = (
                completed_halves >= start_halves
                and (completed_halves - start_halves) % interval_halves == 0
            )
            if should_decay:
                for group in optimizer.param_groups:
                    group["lr"] *= args.step_decay_factor
            print(f"seed {seed} epoch {epoch + (half + 1) / 2:.1f} complete", flush=True)

    model.eval()
    with torch.no_grad():
        chunks = [model(transfer(test_tensor[start:start + 200_000]).long()).cpu()
                  for start in range(0, len(test_tensor), 200_000)]
    return torch.cat(chunks).numpy() if chunks else np.empty(0, dtype=np.float32)


def write_submission(path: Path, users: np.ndarray, videos: np.ndarray,
                     scores: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("row_id", "user_id", "video_id", "score"))
        writer.writerows(
            (row_id, int(user), int(video), f"{float(score):.10f}")
            for row_id, (user, video, score) in enumerate(zip(users, videos, scores))
        )


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=("pure", "1k"), default="pure")
    ap.add_argument("--members", default=None, help="comma-separated seeds; 'seed42' is accepted")
    ap.add_argument("--single-member", action="store_true",
                    help="use only the first configured member")
    ap.add_argument("--member-args", default="", help="shell-style frozen-config overrides")
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--test-features", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("submission.csv"))
    return ap


def run(args: argparse.Namespace) -> int:
    defaults = PURE_MEMBERS if args.dataset == "pure" else ONE_K_MEMBERS
    members = parse_members(args.members) if args.members else list(defaults)
    if args.single_member:
        members = members[:1]
    config = _member_config(args.dataset, args.member_args)
    default_data = "data/real_ws" if args.dataset == "pure" else "data/real_ws_1k"
    data_dir = _resolve(args.data_dir or Path(default_data))
    feature_path = _resolve(args.test_features or Path(
        "data/test_features/test.npz" if args.dataset == "pure" else "data/test_features_1k/test.npz"
    ))
    train = load_train_only(data_dir)
    test = load_test_features(feature_path, train["field_dims"])
    if args.dataset == "pure" and len(test["X"]) != EXPECTED_PURE_TEST_ROWS:
        raise ValueError(f"Pure test rows {len(test['X']):,} != expected {EXPECTED_PURE_TEST_ROWS:,}")
    member_scores = [train_member(train, test["X"], config, seed) for seed in members]
    if len(member_scores) == 1:
        scores = member_scores[0].astype(np.float64)
    else:
        ranks = [_within_user_ranks(test["user_id"], values) for values in member_scores]
        scores = np.mean(np.column_stack(ranks), axis=1)
    out = _resolve(args.out)
    write_submission(out, test["user_id"], test["video_id"], scores)
    checked_rows = check_submission(out, feature_path, return_scores=False)
    if checked_rows != len(test["X"]):
        raise AssertionError("submission checker returned the wrong row count")
    print(f"wrote and validated {out}: {checked_rows:,} rows; members={members}")
    return checked_rows


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
