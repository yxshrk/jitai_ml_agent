"""Validation-only cumulative input-field ablation on a fixed DCN-lite model.

The runner reads only ``train.npz/csv`` and ``val.npz/csv``.  It never opens a
test file.  Every checkpoint is scored with ``data/official/evaluate.py`` at
half-epoch intervals and selected on the official PRIMARY metric.

Contract CLI::

    uv run python zoo/ablate_fields.py --data-dir data/real_ws \
        --out-dir <directory> --field-level 0 [--seed 42] [--regularized]
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
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.official.evaluate import evaluate as official_evaluate

TRAIN_MIN_DATE = 20220408
TRAIN_MAX_DATE = 20220421
VALID_MIN_DATE = 20220422
VALID_MAX_DATE = 20220428
BASELINE_PRIMARY = 0.6016

LEVEL_ADDITIONS: dict[int, tuple[str, ...]] = {
    0: ("user", "video", "author", "tab", "dur_bucket10"),
    1: ("hour", "day_of_week"),
    2: ("dur_bucket50", "dur_le_18s", "dur50_x_tab"),
    3: ("user_active_degree", "follow_user_num_range", "fans_user_num_range",
        "register_days_range", "is_video_author"),
    4: ("video_type", "upload_type", "music_id_top200", "first_tag",
        "aspect_ratio_bucket", "visible_status"),
    5: ("item_long_view_rate", "author_long_view_rate", "upload_age_bucket"),
}


def field_names(level: int) -> tuple[str, ...]:
    """Return the exact cumulative field list for a level."""
    if level not in LEVEL_ADDITIONS:
        raise ValueError("field level must be in 0..5")
    return tuple(name for current in range(level + 1) for name in LEVEL_ADDITIONS[current])


def parser(description: str = __doc__) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--data-dir", required=True,
                    help="directory containing train/val npz+csv; 'real' aliases data/real_ws")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--field-level", type=int, choices=range(6), required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--patience-halves", type=int, default=6)
    ap.add_argument("--subsample", type=int, default=None,
                    help="cap each split for smoke tests; date checks remain active")
    ap.add_argument("--raw-data-dir", default=None,
                    help="directory containing KuaiRand-Pure user/video side tables")
    ap.add_argument("--regularized", action="store_true",
                    help="use EXPERIMENTS_SWEEP r03 regularization package")
    ap.add_argument("--max-runtime", type=int, default=330,
                    help="training alarm in seconds; keeps the best completed half epoch")
    return ap


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _data_dir(value: str) -> Path:
    return ROOT / "data" / "real_ws" if value == "real" else Path(value)


def _raw_dir(value: str | None) -> Path:
    return Path(value) if value else ROOT.parent / "KuaiRand-Pure" / "data"


def _read_export_ids(path: Path, expected: int) -> tuple[np.ndarray, np.ndarray]:
    videos = np.empty(expected, dtype=np.int64)
    authors = np.empty(expected, dtype=np.int64)
    count = 0
    with path.open(newline="", encoding="utf-8") as fh:
        for count, row in enumerate(csv.DictReader(fh), start=1):
            if count > expected:
                raise ValueError(f"{path} has more rows than its npz")
            videos[count - 1] = int(row["video_id"])
            authors[count - 1] = int(row["author_id"])
    if count != expected:
        raise ValueError(f"{path} row mismatch: {count} != {expected}")
    return videos, authors


def load_validation_only(data_dir: str, subsample: int | None = None) -> dict:
    """Load only the frozen train/validation files and enforce their date windows."""
    base = _data_dir(data_dir)
    ds: dict = {"splits": ("train", "valid")}
    field_dims: np.ndarray | None = None
    specs = (("train", "train", TRAIN_MIN_DATE, TRAIN_MAX_DATE),
             ("valid", "val", VALID_MIN_DATE, VALID_MAX_DATE))
    for name, stem, min_date, max_date in specs:
        with np.load(base / f"{stem}.npz", allow_pickle=False) as z:
            split = {key: np.asarray(z[key]).copy() for key in z.files if key != "field_dims"}
            dims = np.asarray(z["field_dims"], dtype=np.int64)
        if field_dims is None:
            field_dims = dims
        elif not np.array_equal(field_dims, dims):
            raise ValueError("train and validation field dimensions differ")
        dates = split["date"]
        if dates.size and (int(dates.min()) < min_date or int(dates.max()) > max_date):
            raise ValueError(f"forbidden date in {name}: {int(dates.min())}..{int(dates.max())}")
        videos, authors = _read_export_ids(base / f"{stem}.csv", len(split["y"]))
        split["users"] = split.pop("user").astype(np.int64)
        split["videos"] = videos
        split["authors"] = authors
        split["tab_raw"] = split["X"][:, 3].astype(np.int64)
        if subsample is not None:
            split = {key: values[:subsample] for key, values in split.items()}
        split["X"] = split["X"].astype(np.int64)
        split["y"] = split["y"].astype(np.float32)
        ds[name] = split
    assert field_dims is not None
    ds["field_dims_total"] = int(field_dims.sum())
    ds["field_names"] = list(LEVEL_ADDITIONS[0])
    return ds


def _encode_append(ds: dict, train_values: np.ndarray, valid_values: np.ndarray,
                   name: str) -> None:
    """Fit a categorical vocabulary on train, append UNK-aware encoded columns."""
    train_values = np.asarray(train_values)
    valid_values = np.asarray(valid_values)
    vocab = {value: index for index, value in enumerate(dict.fromkeys(train_values.tolist()))}
    unknown = len(vocab)
    offset = int(ds["field_dims_total"])
    train_encoded = np.fromiter((vocab[value] + offset for value in train_values.tolist()),
                                dtype=np.int64, count=len(train_values))
    valid_encoded = np.fromiter((vocab.get(value, unknown) + offset
                                 for value in valid_values.tolist()),
                                dtype=np.int64, count=len(valid_values))
    ds["train"]["X"] = np.column_stack((ds["train"]["X"], train_encoded))
    ds["valid"]["X"] = np.column_stack((ds["valid"]["X"], valid_encoded))
    ds["field_dims_total"] = offset + unknown + 1
    ds["field_names"].append(name)


def _weekday(values: np.ndarray) -> np.ndarray:
    lookup = {int(value): dt.date(int(value) // 10000, int(value) // 100 % 100,
                                  int(value) % 100).weekday()
              for value in np.unique(values)}
    return np.fromiter((lookup[int(value)] for value in values),
                       dtype=np.int64, count=len(values))


def _add_level1(ds: dict) -> None:
    tr, va = ds["train"], ds["valid"]
    _encode_append(ds, tr["hourmin"] // 100, va["hourmin"] // 100, "hour")
    _encode_append(ds, _weekday(tr["date"]), _weekday(va["date"]), "day_of_week")


def _add_level2(ds: dict) -> None:
    tr, va = ds["train"], ds["valid"]
    edges = np.unique(np.quantile(tr["duration_ms"], np.linspace(0, 1, 51)[1:-1]))
    train_buckets = np.searchsorted(edges, tr["duration_ms"]).astype(np.int64)
    valid_buckets = np.searchsorted(edges, va["duration_ms"]).astype(np.int64)
    _encode_append(ds, train_buckets, valid_buckets, "dur_bucket50")
    _encode_append(ds, (tr["duration_ms"] <= 18_000).astype(np.int64),
                   (va["duration_ms"] <= 18_000).astype(np.int64), "dur_le_18s")
    _encode_append(ds, train_buckets * 100 + tr["tab_raw"],
                   valid_buckets * 100 + va["tab_raw"], "dur50_x_tab")


def _read_user_metadata(raw_dir: Path) -> dict[int, tuple[str, ...]]:
    names = LEVEL_ADDITIONS[3]
    result: dict[int, tuple[str, ...]] = {}
    with (raw_dir / "user_features_pure.csv").open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            result[int(row["user_id"])] = tuple(row.get(name) or "__MISSING__" for name in names)
    return result


def _add_level3(ds: dict, raw_dir: Path) -> None:
    metadata = _read_user_metadata(raw_dir)
    missing = ("__MISSING__",) * len(LEVEL_ADDITIONS[3])
    rows = {split: [metadata.get(int(user), missing) for user in ds[split]["users"]]
            for split in ds["splits"]}
    for index, name in enumerate(LEVEL_ADDITIONS[3]):
        _encode_append(ds, np.asarray([row[index] for row in rows["train"]]),
                       np.asarray([row[index] for row in rows["valid"]]), name)


def _aspect_bucket(width: str, height: str) -> str:
    try:
        ratio = float(width) / float(height)
        if not np.isfinite(ratio) or ratio <= 0:
            return "missing"
    except (TypeError, ValueError, ZeroDivisionError):
        return "missing"
    # Fixed semantic bins prevent validation/catalog distribution from fitting edges.
    return str(int(np.digitize(ratio, [0.56, 0.75, 1.0, 1.34, 1.78, 2.0])))


def _first_tag(value: str) -> str:
    cleaned = (value or "").strip().strip("[]")
    if not cleaned:
        return "__MISSING__"
    return cleaned.replace(";", ",").split(",")[0].strip() or "__MISSING__"


def _read_video_metadata(raw_dir: Path) -> dict[int, tuple[str, ...]]:
    result: dict[int, tuple[str, ...]] = {}
    with (raw_dir / "video_features_basic_pure.csv").open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            result[int(row["video_id"])] = (
                row.get("video_type") or "__MISSING__",
                row.get("upload_type") or "__MISSING__",
                row.get("music_id") or "__MISSING__",
                _first_tag(row.get("tag") or ""),
                _aspect_bucket(row.get("server_width", ""), row.get("server_height", "")),
                row.get("visible_status") or "__MISSING__",
                row.get("upload_dt") or "",
            )
    return result


def _video_rows(ds: dict, video_metadata: dict[int, tuple[str, ...]]) -> dict[str, list[tuple[str, ...]]]:
    missing = ("__MISSING__",) * 6 + ("",)
    return {split: [video_metadata.get(int(video), missing) for video in ds[split]["videos"]]
            for split in ds["splits"]}


def _add_level4(ds: dict, video_rows: dict[str, list[tuple[str, ...]]]) -> None:
    top_music = {value for value, _ in Counter(row[2] for row in video_rows["train"])
                 .most_common(200)}
    for index, name in enumerate(LEVEL_ADDITIONS[4]):
        train_values = np.asarray([row[index] if index != 2 or row[index] in top_music
                                   else "__OTHER__" for row in video_rows["train"]])
        valid_values = np.asarray([row[index] if index != 2 or row[index] in top_music
                                   else "__OTHER__" for row in video_rows["valid"]])
        _encode_append(ds, train_values, valid_values, name)


def _smoothed_rates(train_keys: np.ndarray, valid_keys: np.ndarray, labels: np.ndarray,
                    prior_strength: float = 20.0) -> tuple[np.ndarray, np.ndarray]:
    """LOO train rates and full-train validation rates, with no validation targets."""
    unique, inverse = np.unique(train_keys, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    positives = np.bincount(inverse, weights=labels.astype(np.float64))
    global_rate = float(labels.mean())
    train_rates = ((positives[inverse] - labels + prior_strength * global_rate) /
                   (counts[inverse] - 1.0 + prior_strength))
    positions = np.searchsorted(unique, valid_keys)
    seen = positions < len(unique)
    seen[seen] &= unique[positions[seen]] == valid_keys[seen]
    valid_rates = np.full(len(valid_keys), global_rate, dtype=np.float64)
    matched = positions[seen]
    valid_rates[seen] = ((positives[matched] + prior_strength * global_rate) /
                         (counts[matched] + prior_strength))
    return train_rates.astype(np.float32), valid_rates.astype(np.float32)


def _rate_buckets(train_rates: np.ndarray, valid_rates: np.ndarray,
                  n_buckets: int = 20) -> tuple[np.ndarray, np.ndarray]:
    edges = np.unique(np.quantile(train_rates, np.linspace(0, 1, n_buckets + 1)[1:-1]))
    return np.searchsorted(edges, train_rates), np.searchsorted(edges, valid_rates)


def _parse_upload_date(value: str) -> dt.date | None:
    try:
        return dt.date.fromisoformat(value[:10])
    except (TypeError, ValueError):
        return None


def _upload_age_bucket(dates: np.ndarray, rows: list[tuple[str, ...]]) -> np.ndarray:
    edges = np.asarray([0, 1, 3, 7, 14, 30, 60, 120, 365, 730], dtype=np.int64)
    values = np.empty(len(dates), dtype=np.int64)
    for index, (date_value, row) in enumerate(zip(dates, rows)):
        observation = dt.date(int(date_value) // 10000, int(date_value) // 100 % 100,
                              int(date_value) % 100)
        upload = _parse_upload_date(row[6])
        if upload is None:
            values[index] = len(edges) + 1
        else:
            age = max(0, (observation - upload).days)
            values[index] = int(np.searchsorted(edges, age, side="right"))
    return values


def _add_level5(ds: dict, video_rows: dict[str, list[tuple[str, ...]]]) -> None:
    tr, va = ds["train"], ds["valid"]
    for train_keys, valid_keys, name in (
        (tr["videos"], va["videos"], "item_long_view_rate"),
        (tr["authors"], va["authors"], "author_long_view_rate"),
    ):
        train_rates, valid_rates = _smoothed_rates(train_keys, valid_keys, tr["y"])
        train_buckets, valid_buckets = _rate_buckets(train_rates, valid_rates)
        _encode_append(ds, train_buckets, valid_buckets, name)
    _encode_append(ds, _upload_age_bucket(tr["date"], video_rows["train"]),
                   _upload_age_bucket(va["date"], video_rows["valid"]),
                   "upload_age_bucket")


def build_fields(ds: dict, level: int, raw_dir: Path | None = None) -> dict:
    """Mutate and return ``ds`` with exactly the requested cumulative fields."""
    if level not in LEVEL_ADDITIONS:
        raise ValueError("field level must be in 0..5")
    if level >= 1:
        _add_level1(ds)
    if level >= 2:
        _add_level2(ds)
    raw = raw_dir or _raw_dir(None)
    if level >= 3:
        _add_level3(ds, raw)
    video_rows = None
    if level >= 4:
        video_rows = _video_rows(ds, _read_video_metadata(raw))
        _add_level4(ds, video_rows)
    if level >= 5:
        assert video_rows is not None
        _add_level5(ds, video_rows)
    expected = field_names(level)
    if tuple(ds["field_names"]) != expected:
        raise AssertionError(f"field construction mismatch: {ds['field_names']} != {expected}")
    return ds


def recency_weights(dates: np.ndarray, half_life_days: float = 7.0) -> np.ndarray:
    """Seven-day exponential decay ending on the last legal training date."""
    end = dt.date(2022, 4, 21)
    ages = np.asarray([(end - dt.date(int(value) // 10000, int(value) // 100 % 100,
                                    int(value) % 100)).days for value in dates])
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
        self.positives = np.concatenate(positives) if positives else np.empty(0, dtype=np.int64)
        self.negatives = np.concatenate(negatives) if negatives else np.empty(0, dtype=np.int64)
        self.starts = np.concatenate(negative_starts) if negative_starts else np.empty(0, dtype=np.int64)
        self.counts = np.concatenate(negative_counts) if negative_counts else np.empty(0, dtype=np.int64)

    def sample(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        if not len(self.positives):
            return self.positives, self.positives
        negatives = self.negatives[self.starts + rng.integers(0, self.counts)]
        permutation = rng.permutation(len(self.positives))
        return self.positives[permutation], negatives[permutation]


class DCNLite(nn.Module):
    """Fixed one-cross-layer, MLP-128 DCN-lite architecture."""

    def __init__(self, total_dim: int, n_fields: int, k: int = 16,
                 dropout: float = 0.1, embedding_dropout: float = 0.0):
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
        hidden = self.mlp(x0 * self.cross(x0) + x0)
        return self.head(hidden).squeeze(1)


def official_metrics(users: np.ndarray, labels: np.ndarray,
                     scores: np.ndarray) -> dict[str, float]:
    result = official_evaluate(users.tolist(), labels.astype(int).tolist(), scores.tolist())
    return {"gauc": float(result["GAUC"]), "ndcg5": float(result["nDCG@5"]),
            "primary": float(result["primary"])}


def train_and_report(ds: dict, args: argparse.Namespace) -> dict:
    started = time.time()
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    tr, va = ds["train"], ds["valid"]
    train_x = torch.as_tensor(np.ascontiguousarray(tr["X"]), dtype=torch.long)
    train_y = torch.as_tensor(tr["y"], dtype=torch.float32)
    train_weights = torch.as_tensor(recency_weights(tr["date"]), dtype=torch.float32)
    valid_x = torch.as_tensor(np.ascontiguousarray(va["X"]), dtype=torch.long)
    dropout = 0.2 if args.regularized else 0.1
    embedding_dropout = 0.1 if args.regularized else 0.0
    model = DCNLite(int(ds["field_dims_total"]), train_x.shape[1], args.k,
                    dropout=dropout, embedding_dropout=embedding_dropout)
    if args.regularized:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = None
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
    best_primary = -math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0.0
    bad = 0
    history: list[dict[str, float]] = []
    timed_out = False

    class RunTimeout(Exception):
        pass

    def timeout_handler(_signum, _frame):
        raise RunTimeout

    previous_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(args.max_runtime)
    stop = False
    try:
        for epoch in range(1, args.epochs + 1):
            point_order = rng.permutation(point_order)
            pair_pos, pair_neg = sampler.sample(rng)
            for half in range(2):
                row_start = half * half_size
                row_end = min(len(train_y), (half + 1) * half_size)
                if row_start >= row_end:
                    continue
                half_rows = point_order[row_start:row_end]
                batches = math.ceil(len(half_rows) / args.batch_size)
                losses = []
                for batch in range(batches):
                    idx_np = half_rows[batch * args.batch_size:(batch + 1) * args.batch_size]
                    idx = torch.as_tensor(idx_np, dtype=torch.long)
                    logits = model(train_x[idx])
                    weights = train_weights[idx]
                    point = nn.functional.binary_cross_entropy_with_logits(
                        logits, train_y[idx], reduction="none")
                    point_loss = (point * weights).sum() / weights.sum()
                    pair_start = ((half * batches + batch) * len(pair_pos)) // (2 * batches)
                    pair_end = ((half * batches + batch + 1) * len(pair_pos)) // (2 * batches)
                    if pair_end > pair_start:
                        positive = torch.as_tensor(pair_pos[pair_start:pair_end], dtype=torch.long)
                        negative = torch.as_tensor(pair_neg[pair_start:pair_end], dtype=torch.long)
                        pair = nn.functional.softplus(
                            model(train_x[negative]) - model(train_x[positive]))
                        pair_weights = 0.5 * (train_weights[positive] + train_weights[negative])
                        pair_loss = (pair * pair_weights).sum() / pair_weights.sum()
                    else:
                        pair_loss = point_loss * 0.0
                    loss = 0.5 * point_loss + 0.5 * pair_loss
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.detach()))
                scores = predict()
                metrics = official_metrics(va["users"], va["y"], scores)
                epoch_value = epoch - 0.5 + half * 0.5
                entry = {"epoch": epoch_value, "train_loss": float(np.mean(losses)),
                         "val_gauc": metrics["gauc"], "val_ndcg5": metrics["ndcg5"],
                         "val_primary": metrics["primary"]}
                history.append(entry)
                print(f"epoch {epoch_value:.1f} | loss {entry['train_loss']:.4f} | "
                      f"valid gauc {metrics['gauc']:.6f} ndcg5 {metrics['ndcg5']:.6f} "
                      f"primary {metrics['primary']:.6f}", flush=True)
                if metrics["primary"] > best_primary + 1e-5:
                    best_primary = metrics["primary"]
                    best_state = copy.deepcopy(model.state_dict())
                    best_epoch = epoch_value
                    bad = 0
                else:
                    bad += 1
                    if bad >= args.patience_halves:
                        stop = True
                        break
            if scheduler is not None:
                scheduler.step()
            if stop:
                print(f"early stop after epoch {history[-1]['epoch']:.1f}", flush=True)
                break
    except RunTimeout:
        timed_out = True
        print(f"runtime alarm reached at {args.max_runtime}s; preserving best half epoch",
              flush=True)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
    if best_state is None:
        raise RuntimeError("runtime cap expired before a checkpoint completed")
    model.load_state_dict(best_state)
    scores = predict()
    metrics = official_metrics(va["users"], va["y"], scores)
    metrics.update({
        "history": history,
        "runtime_s": round(time.time() - started, 1),
        "best_epoch": best_epoch,
        "seed": args.seed,
        "field_level": args.field_level,
        "n_fields": train_x.shape[1],
        "field_names": list(ds["field_names"]),
        "regularized": bool(args.regularized),
        "delta_vs_baseline": metrics["primary"] - BASELINE_PRIMARY,
        "timed_out": timed_out,
    })
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
    print("final:", json.dumps({key: value for key, value in metrics.items()
                                 if key != "history"}, sort_keys=True), flush=True)
    return metrics


def run(args: argparse.Namespace) -> dict:
    ds = load_validation_only(args.data_dir, args.subsample)
    ds = build_fields(ds, args.field_level, _raw_dir(args.raw_data_dir))
    if ds["train"]["X"].shape[1] != len(field_names(args.field_level)):
        raise AssertionError("field count does not match declared configuration")
    return train_and_report(ds, args)


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
