"""Shared implementation for the validation-only final lever campaign.

Only ``train.npz``, ``val.npz`` and their aligned CSVs are read.  Dates are
asserted before training, and every score comes from data/official/evaluate.py.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import signal
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from data.official.evaluate import evaluate as official_evaluate

BASELINE = 0.6016
PRIMARY_CONTROL = 0.6041


def parser(description: str, default_variant: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--patience", type=int, default=6,
                    help="non-improving half-epoch evaluations")
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--bpr-weight", type=float, default=0.5)
    ap.add_argument("--half-life", type=int, choices=(2, 5), default=5)
    ap.add_argument("--variant", default=default_variant,
                    choices=("control", "shrinkage", "cross_ids", "freshness",
                             "specialists", "ffm", "finalmlp"))
    ap.add_argument("--subsample", type=int)
    ap.add_argument("--max-runtime", type=int, default=350)
    return ap


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _root(data_dir: str) -> Path:
    return ROOT / "data" / "real_ws" if data_dir == "real" else Path(data_dir)


def load_data(data_dir: str, subsample: int | None = None) -> dict:
    root = _root(data_dir)
    out: dict[str, dict[str, np.ndarray] | int] = {}
    for name, filename, csvname in (("train", "train.npz", "train.csv"),
                                     ("valid", "val.npz", "val.csv")):
        with np.load(root / filename, allow_pickle=False) as z:
            split = {key: z[key].copy() for key in z.files}
        n = len(split["y"])
        if name == "train" and subsample:
            n = min(n, subsample)
            split = {key: value[:n] if value.ndim and len(value) >= n and
                     key != "field_dims" else value for key, value in split.items()}
        rows = []
        with (root / csvname).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for i, row in enumerate(reader):
                if i >= n:
                    break
                rows.append(row)
        if len(rows) != n:
            raise ValueError(f"{csvname} alignment failure: {len(rows)} != {n}")
        for col in ("video_id", "author_id", "tab", "date", "duration_ms"):
            split[f"raw_{col}"] = np.asarray([int(float(row[col])) for row in rows],
                                               dtype=np.int64)
        dates = split["date"]
        if name == "train" and not (dates.min() >= 20220408 and dates.max() <= 20220421):
            raise ValueError("train dates violate 20220408..20220421")
        if name == "valid" and not (dates.min() >= 20220422 and dates.max() <= 20220428):
            raise ValueError("validation dates violate 20220422..20220428")
        if dates.max() >= 20220429:
            raise ValueError("forbidden date >= 20220429")
        out[name] = split
    tr = out["train"]
    assert isinstance(tr, dict)
    out["total_dim"] = int(np.max(tr["X"])) + 1
    return out


def metrics(users: np.ndarray, labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    result = official_evaluate(users.astype(int).tolist(), labels.astype(int).tolist(),
                               scores.astype(float).tolist())
    return {"gauc": float(result["GAUC"]), "ndcg5": float(result["nDCG@5"]),
            "primary": float(result["primary"])}


def add_frequency_crosses(ds: dict, minimum: int = 20) -> None:
    tr, va = ds["train"], ds["valid"]
    specs = (
        ("user_tab", lambda s: zip(s["user"], s["raw_tab"])),
        ("user_regime", lambda s: zip(s["user"], s["raw_duration_ms"] <= 18_000)),
        ("author_tab", lambda s: zip(s["raw_author_id"], s["raw_tab"])),
        ("tab_durbucket", lambda s: zip(s["raw_tab"], s["X"][:, 4])),
    )
    offset = int(ds["total_dim"])
    columns = {"train": [], "valid": []}
    for _name, getter in specs:
        train_keys = list(getter(tr))
        counts = Counter(train_keys)
        vocab = {key: i + 1 for i, key in enumerate(k for k, c in counts.items() if c >= minimum)}
        for split_name, split in (("train", tr), ("valid", va)):
            encoded = np.fromiter((offset + vocab.get(key, 0) for key in getter(split)),
                                  dtype=np.int64, count=len(split["y"]))
            columns[split_name].append(encoded)
        offset += len(vocab) + 1
    for split_name, split in (("train", tr), ("valid", va)):
        split["X"] = np.column_stack((split["X"], *columns[split_name]))
    ds["total_dim"] = offset


def add_freshness_velocity(ds: dict, half_life: int) -> None:
    """Add age plus causal per-video exposure/positive velocities as dense inputs."""
    upload: dict[int, int] = {}
    raw = ROOT.parent / "KuaiRand-Pure" / "data" / "video_features_basic_pure.csv"
    with raw.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            value = row.get("upload_dt", "")
            if value:
                upload[int(row["video_id"])] = int(value.replace("-", ""))

    import datetime as dt
    def ordinal(values: np.ndarray) -> np.ndarray:
        return np.asarray([dt.date(int(x)//10000, int(x)//100 % 100, int(x) % 100).toordinal()
                           for x in values], dtype=np.int64)
    tr = ds["train"]
    unique_days = sorted(np.unique(tr["date"]).tolist())
    day_video_exp: dict[int, Counter] = {}
    day_video_pos: dict[int, Counter] = {}
    for day in unique_days:
        mask = tr["date"] == day
        day_video_exp[day] = Counter(tr["raw_video_id"][mask].tolist())
        day_video_pos[day] = Counter(tr["raw_video_id"][mask][tr["y"][mask] == 1].tolist())
    decay = math.log(2.0) / half_life
    for split_name in ("train", "valid"):
        split = ds[split_name]
        dates_ord = ordinal(split["date"])
        up_values = np.asarray([upload.get(int(v), int(d)) for v, d in
                                zip(split["raw_video_id"], split["date"])], dtype=np.int64)
        ages = np.maximum(0, dates_ord - ordinal(up_values)).astype(np.float32)
        exposure = np.zeros(len(split["y"]), dtype=np.float32)
        positive = np.zeros(len(split["y"]), dtype=np.float32)
        for i, (date, video) in enumerate(zip(split["date"], split["raw_video_id"])):
            current = dt.date(int(date)//10000, int(date)//100 % 100, int(date) % 100).toordinal()
            for prior in unique_days:
                if prior >= int(date):
                    continue
                pord = dt.date(prior//10000, prior//100 % 100, prior % 100).toordinal()
                weight = math.exp(-decay * (current - pord))
                exposure[i] += weight * day_video_exp[prior][int(video)]
                positive[i] += weight * day_video_pos[prior][int(video)]
        split["dense"] = np.column_stack((np.log1p(ages) / 5.0,
                                           np.log1p(exposure),
                                           positive / np.maximum(exposure, 1.0))).astype(np.float32)


class DCN(nn.Module):
    def __init__(self, dim: int, fields: int, k: int, dropout: float,
                 dense: int = 0, mode: str = "control", users: int = 0,
                 cohorts: int = 10):
        super().__init__()
        self.mode = mode
        self.embedding = nn.Embedding(dim, k)
        nn.init.normal_(self.embedding.weight, std=0.01)
        self.cohort = nn.Embedding(cohorts, k) if mode == "shrinkage" else None
        self.gate = nn.Linear(1, 1) if mode == "shrinkage" else None
        width = fields * k + dense
        self.cross = nn.Linear(width, width)
        self.mlp = nn.Sequential(nn.Linear(width, 128), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(128, 1))

    def forward(self, x: torch.Tensor, dense: torch.Tensor | None = None,
                cohort: torch.Tensor | None = None,
                log_count: torch.Tensor | None = None) -> torch.Tensor:
        emb = self.embedding(x)
        if self.mode == "shrinkage":
            assert cohort is not None and log_count is not None and self.cohort is not None
            gate = torch.sigmoid(self.gate(log_count[:, None]))
            user = gate * emb[:, 0, :] + (1.0 - gate) * self.cohort(cohort)
            emb = torch.cat((user[:, None, :], emb[:, 1:, :]), dim=1)
        x0 = emb.flatten(1)
        if dense is not None:
            x0 = torch.cat((x0, dense), dim=1)
        crossed = x0 * self.cross(x0) + x0
        return self.mlp(crossed).squeeze(1)


class FFMHead(nn.Module):
    def __init__(self, dim: int, fields: int, k: int = 8, dropout: float = 0.1, **_):
        super().__init__()
        self.fields = fields
        self.emb = nn.ModuleList(nn.Embedding(dim, k) for _ in range(fields))
        for table in self.emb:
            nn.init.normal_(table.weight, std=0.01)
        self.linear = nn.Embedding(dim, 1)
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor, **_) -> torch.Tensor:
        score = self.linear(x).sum(1).squeeze(1) + self.bias
        for i in range(self.fields):
            for j in range(i + 1, self.fields):
                score = score + (self.emb[j](x[:, i]) * self.emb[i](x[:, j])).sum(1)
        return score


class FinalMLPHead(nn.Module):
    def __init__(self, dim: int, fields: int, k: int, dropout: float, **_):
        super().__init__()
        self.embedding = nn.Embedding(dim, k)
        nn.init.normal_(self.embedding.weight, std=0.01)
        width = fields * k
        self.a = nn.Sequential(nn.Linear(width, 128), nn.ReLU(), nn.Dropout(dropout),
                               nn.Linear(128, 64), nn.ReLU())
        self.b = nn.Sequential(nn.Linear(width, 128), nn.ReLU(), nn.Dropout(dropout),
                               nn.Linear(128, 64), nn.ReLU())
        self.fuse = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor, **_) -> torch.Tensor:
        flat = self.embedding(x).flatten(1)
        return self.fuse(torch.cat((self.a(flat), self.b(flat)), 1)).squeeze(1)


class PairSampler:
    def __init__(self, users: np.ndarray, labels: np.ndarray, user_uniform: bool = False):
        self.user_uniform = user_uniform
        self.groups = []
        order = np.argsort(users, kind="stable")
        sorted_users = users[order]
        starts = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1]])
        for start, end in zip(starts, np.r_[starts[1:], len(order)]):
            idx = order[start:end]
            pos, neg = idx[labels[idx] == 1], idx[labels[idx] == 0]
            if len(pos) and len(neg):
                self.groups.append((pos, neg))

    def sample(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        if self.user_uniform:
            p = np.asarray([rng.choice(pos) for pos, _ in self.groups])
            n = np.asarray([rng.choice(neg) for _, neg in self.groups])
        else:
            p = np.concatenate([pos for pos, _ in self.groups])
            n = np.concatenate([rng.choice(neg, len(pos)) for pos, neg in self.groups])
        order = rng.permutation(len(p))
        return p[order], n[order]


def _rank_by_user(users: np.ndarray, scores: np.ndarray) -> np.ndarray:
    result = np.empty(len(scores), dtype=np.float32)
    groups: dict[int, list[int]] = defaultdict(list)
    for i, user in enumerate(users):
        groups[int(user)].append(i)
    for indices in groups.values():
        order = np.argsort(scores[indices], kind="stable")
        ranks = np.empty(len(indices), dtype=np.float32)
        ranks[order] = np.arange(len(indices), dtype=np.float32)
        result[indices] = ranks / max(1, len(indices) - 1)
    return result


def _write(out_dir: str, va: dict, scores: np.ndarray, result: dict) -> None:
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    with (target / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("row_id", "user_id", "video_id", "score"))
        for i, (u, v, score) in enumerate(zip(va["user"], va["raw_video_id"], scores)):
            writer.writerow((i, int(u), int(v), f"{float(score):.10f}"))
    with (target / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, sort_keys=True)
        handle.write("\n")


def train_once(args: argparse.Namespace, ds: dict | None = None,
               specialist: str | None = None, write: bool = True) -> tuple[dict, np.ndarray]:
    started = time.time()
    set_seed(args.seed + (1 if specialist == "uniform" else 0))
    rng = np.random.default_rng(args.seed + (1 if specialist == "uniform" else 0))
    ds = load_data(args.data_dir, args.subsample) if ds is None else ds
    tr, va = ds["train"], ds["valid"]
    if args.variant == "cross_ids" and tr["X"].shape[1] == 5:
        add_frequency_crosses(ds)
    if args.variant == "freshness" and "dense" not in tr:
        add_freshness_velocity(ds, args.half_life)
    xtr = torch.as_tensor(np.ascontiguousarray(tr["X"]), dtype=torch.long)
    xva = torch.as_tensor(np.ascontiguousarray(va["X"]), dtype=torch.long)
    ytr = torch.as_tensor(tr["y"], dtype=torch.float32)
    dense_tr = torch.as_tensor(tr["dense"], dtype=torch.float32) if "dense" in tr else None
    dense_va = torch.as_tensor(va["dense"], dtype=torch.float32) if "dense" in va else None

    counts = np.bincount(tr["user"], minlength=int(max(tr["user"].max(), va["user"].max())) + 1)
    edges = np.quantile(counts[counts > 0], np.linspace(0, 1, 11)[1:-1])
    cohort_all = np.searchsorted(edges, counts).astype(np.int64)
    cohort_tr = torch.as_tensor(cohort_all[tr["user"]], dtype=torch.long)
    cohort_va = torch.as_tensor(cohort_all[va["user"]], dtype=torch.long)
    log_count_all = np.log1p(counts).astype(np.float32)
    scale = max(float(log_count_all.std()), 1e-6)
    log_count_all = (log_count_all - float(log_count_all.mean())) / scale
    count_tr = torch.as_tensor(log_count_all[tr["user"]])
    count_va = torch.as_tensor(log_count_all[va["user"]])

    if args.variant == "ffm":
        model: nn.Module = FFMHead(int(ds["total_dim"]), xtr.shape[1], k=8)
    elif args.variant == "finalmlp":
        model = FinalMLPHead(int(ds["total_dim"]), xtr.shape[1], args.k, args.dropout)
    else:
        model = DCN(int(ds["total_dim"]), xtr.shape[1], args.k, args.dropout,
                    dense=0 if dense_tr is None else dense_tr.shape[1], mode=args.variant)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    sampler = PairSampler(tr["user"], tr["y"], user_uniform=specialist == "uniform")

    def forward(indices, valid=False):
        x = xva[indices] if valid else xtr[indices]
        dense = dense_va[indices] if valid and dense_va is not None else (
            dense_tr[indices] if not valid and dense_tr is not None else None)
        cohort = cohort_va[indices] if valid else cohort_tr[indices]
        log_count = count_va[indices] if valid else count_tr[indices]
        return model(x, dense=dense, cohort=cohort, log_count=log_count)

    def predict() -> np.ndarray:
        model.eval()
        with torch.no_grad():
            values = torch.cat([forward(slice(i, i + 200_000), valid=True)
                                for i in range(0, len(xva), 200_000)]).numpy()
        model.train()
        return values

    best_primary = -math.inf
    best_state = None
    bad = 0
    history: list[dict] = []
    steps = math.ceil(len(ytr) / args.batch_size)
    best_step = 0.0
    class Timeout(Exception): pass
    def alarm(_sig, _frame): raise Timeout
    old_alarm = signal.signal(signal.SIGALRM, alarm)
    signal.alarm(args.max_runtime)
    timed_out = False
    try:
        for epoch in range(1, args.epochs + 1):
            pos, neg = sampler.sample(rng)
            order = rng.permutation(len(ytr))
            losses = []
            for batch in range(steps):
                idx = order[batch * args.batch_size:(batch + 1) * args.batch_size]
                logits = forward(idx)
                point = bce(logits, ytr[idx])
                if specialist == "uniform":
                    # Observable proxy for top-position value: short user lists get more weight.
                    weights = torch.as_tensor(1.0 / np.sqrt(np.maximum(counts[tr["user"][idx]], 1)),
                                              dtype=torch.float32)
                    point_loss = (point * weights / weights.mean()).mean()
                else:
                    point_loss = point.mean()
                lo, hi = batch * len(pos) // steps, (batch + 1) * len(pos) // steps
                pair_loss = torch.zeros((), dtype=torch.float32)
                if hi > lo:
                    pair_loss = nn.functional.softplus(forward(neg[lo:hi]) - forward(pos[lo:hi])).mean()
                bw = 0.7 if specialist == "positive" else args.bpr_weight
                loss = (1.0 - bw) * point_loss + bw * pair_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
                if batch + 1 in {math.ceil(steps / 2), steps}:
                    score = predict()
                    observed = metrics(va["user"], va["y"], score)
                    step = epoch - 0.5 if batch + 1 < steps else float(epoch)
                    record = {"epoch": step, "train_loss": float(np.mean(losses)),
                              "val_gauc": observed["gauc"],
                              "val_primary": observed["primary"]}
                    history.append(record)
                    print(f"step={step:.1f} loss={record['train_loss']:.5f} "
                          f"gauc={observed['gauc']:.6f} primary={observed['primary']:.6f}",
                          flush=True)
                    if observed["primary"] > best_primary + 1e-7:
                        best_primary, best_state, bad, best_step = observed["primary"], copy.deepcopy(model.state_dict()), 0, step
                    else:
                        bad += 1
                    if bad >= args.patience:
                        break
            if bad >= args.patience:
                break
    except Timeout:
        timed_out = True
        print("runtime cap reached; preserving best completed half-epoch", flush=True)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_alarm)
    if best_state is None:
        raise RuntimeError("no completed validation checkpoint")
    model.load_state_dict(best_state)
    scores = predict()
    result = {**metrics(va["user"], va["y"], scores), "history": history,
              "runtime_s": round(time.time() - started, 1), "best_step": best_step,
              "seed": args.seed, "variant": args.variant, "timed_out": timed_out}
    if write:
        _write(args.out_dir, va, scores, result)
    print("final:", json.dumps({k: v for k, v in result.items() if k != "history"}, sort_keys=True), flush=True)
    return result, scores


def run(args: argparse.Namespace) -> dict:
    if args.variant != "specialists":
        result, _ = train_once(args)
        return result
    ds = load_data(args.data_dir, args.subsample)
    left_args, right_args = copy.copy(args), copy.copy(args)
    left_args.variant = right_args.variant = "control"
    left, score_a = train_once(left_args, copy.deepcopy(ds), specialist="positive", write=False)
    right, score_b = train_once(right_args, copy.deepcopy(ds), specialist="uniform", write=False)
    va, tr = ds["valid"], ds["train"]
    rank_a, rank_b = _rank_by_user(va["user"], score_a), _rank_by_user(va["user"], score_b)
    list_sizes = np.bincount(va["user"], minlength=int(va["user"].max()) + 1)
    history_depth = np.bincount(tr["user"], minlength=max(len(list_sizes), int(tr["user"].max()) + 1))
    size = np.log1p(list_sizes[va["user"]]); depth = np.log1p(history_depth[va["user"]])
    size = (size - size.mean()) / max(size.std(), 1e-6)
    depth = (depth - depth.mean()) / max(depth.std(), 1e-6)
    candidates = []
    for intercept in (-1.0, 0.0, 1.0):
        for a in (-1.0, 0.0, 1.0):
            for b in (-1.0, 0.0, 1.0):
                mix = 1.0 / (1.0 + np.exp(-(intercept + a * size + b * depth)))
                score = mix * rank_a + (1.0 - mix) * rank_b
                candidates.append((metrics(va["user"], va["y"], score), score, (intercept, a, b)))
    observed, scores, gate = max(candidates, key=lambda item: item[0]["primary"])
    result = {**observed, "history": [{"specialist": "positive", **left},
                                       {"specialist": "uniform", **right}],
              "runtime_s": round(left["runtime_s"] + right["runtime_s"], 1),
              "seed": args.seed, "variant": "specialists", "gate": gate,
              "gate_fit": "27-point validation grid using list size and train-history depth"}
    _write(args.out_dir, va, scores, result)
    print("final:", json.dumps({k: v for k, v in result.items() if k != "history"}, sort_keys=True), flush=True)
    return result
