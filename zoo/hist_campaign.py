"""Validation-only user-history/data-feature campaign on the workspace NPZ fast path.

Contract CLI:
  uv run python zoo/hist_campaign.py --data-dir data/real_ws --out-dir <o> [--seed 42]

The default reproduces the frozen DCN-lite stack. Feature experiments are selected
with --experiment {affinity,user-cross,recency,covisit,din}. No test split is read.
All aggregates are fit on train dates 20220408..20220421; training target-derived
features use leave-one-out values, while validation uses the complete train window.
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import svds
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.official.evaluate import evaluate as official_evaluate

TRAIN_LO = 20220408
TRAIN_HI = 20220421
VALID_LO = 20220422
VALID_HI = 20220428
BASELINE = 0.6016


def parser(description: str = __doc__) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--subsample", type=int)
    ap.add_argument("--experiment", choices=("base", "affinity", "user-cross",
                                               "recency", "covisit", "din"),
                    default="base")
    ap.add_argument("--with-affinity", action="store_true")
    ap.add_argument("--half-life", type=float, choices=(3.0, 7.0, 14.0), default=7.0)
    ap.add_argument("--smoothing", type=float, default=10.0)
    ap.add_argument("--rate-buckets", type=int, default=20)
    return ap


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _load_video_ids(csv_path: Path, limit: int | None = None) -> np.ndarray:
    vals = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            vals.append(int(row["video_id"]))
            if limit is not None and len(vals) >= limit:
                break
    return np.asarray(vals, dtype=np.int64)


def load_workspace(data_dir: str, subsample: int | None = None) -> dict:
    """Load only train.npz and val.npz, rejecting dates outside frozen windows."""
    root = Path(data_dir)
    out = {"splits": ("train", "valid")}
    limits = {"train": None, "valid": None}
    if subsample:
        limits = {"train": max(1, int(subsample * 0.8)),
                  "valid": max(1, subsample - max(1, int(subsample * 0.8)))}
    for name, fname in (("train", "train.npz"), ("valid", "val.npz")):
        with np.load(root / fname, allow_pickle=False) as z:
            n = limits[name]
            sl = slice(None, n)
            split = {
                "X": z["X"][sl].astype(np.int64),
                "y": z["y"][sl].astype(np.float32),
                "users": z["user"][sl].astype(np.int64),
                "click": z["click"][sl].astype(np.float32),
                "play_time_ms": z["play_time_ms"][sl].astype(np.float32),
                "duration_ms": z["duration_ms"][sl].astype(np.float32),
                "hourmin": z["hourmin"][sl].astype(np.int32),
                "date": z["date"][sl].astype(np.int32),
            }
            field_dims = z["field_dims"].astype(np.int64)
        split["tab"] = split["X"][:, 3].copy()
        split["videos"] = _load_video_ids(root / ("train.csv" if name == "train" else "val.csv"), n)
        if len(split["videos"]) != len(split["y"]):
            raise ValueError(f"CSV/NPZ row mismatch for {name}")
        out[name] = split
    if not (np.all((out["train"]["date"] >= TRAIN_LO) & (out["train"]["date"] <= TRAIN_HI))):
        raise ValueError("train data outside 20220408..20220421")
    if not (np.all((out["valid"]["date"] >= VALID_LO) & (out["valid"]["date"] <= VALID_HI))):
        raise ValueError("validation data outside 20220422..20220428")
    out["field_dims"] = field_dims
    out["field_dims_total"] = int(field_dims.sum())
    return out


def _encode_column(train: np.ndarray, valid: np.ndarray, offset: int):
    vocab = {v: i for i, v in enumerate(dict.fromkeys(train.tolist()))}
    unk = len(vocab)
    tr = np.fromiter((vocab[v] for v in train.tolist()), dtype=np.int64, count=len(train)) + offset
    va = np.fromiter((vocab.get(v, unk) for v in valid.tolist()), dtype=np.int64,
                     count=len(valid)) + offset
    return tr, va, offset + unk + 1


def _append_columns(ds: dict, columns: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    offset = ds["field_dims_total"]
    tr_cols, va_cols = [], []
    for tr_raw, va_raw in columns:
        tr, va, offset = _encode_column(np.asarray(tr_raw), np.asarray(va_raw), offset)
        tr_cols.append(tr)
        va_cols.append(va)
    if tr_cols:
        ds["train"]["X"] = np.column_stack((ds["train"]["X"], *tr_cols))
        ds["valid"]["X"] = np.column_stack((ds["valid"]["X"], *va_cols))
    ds["field_dims_total"] = offset
    return ds


def add_best_features(ds: dict) -> dict:
    tr, va = ds["train"], ds["valid"]
    edges = np.quantile(tr["duration_ms"], np.linspace(0, 1, 51)[1:-1])
    btr = np.searchsorted(edges, tr["duration_ms"]).astype(np.int64)
    bva = np.searchsorted(edges, va["duration_ms"]).astype(np.int64)
    short_tr = (tr["duration_ms"] <= 18_000).astype(np.int64)
    short_va = (va["duration_ms"] <= 18_000).astype(np.int64)
    hour_tr, hour_va = tr["hourmin"] // 100, va["hourmin"] // 100

    def dow(a):
        return np.asarray([dt.date(int(x) // 10000, int(x) // 100 % 100,
                                   int(x) % 100).weekday() for x in a], dtype=np.int64)

    return _append_columns(ds, [
        (btr, bva),
        (short_tr, short_va),
        (btr * 100 + tr["tab"], bva * 100 + va["tab"]),
        (hour_tr, hour_va),
        (dow(tr["date"]), dow(va["date"])),
    ])


def _group_loo_rates(train_users: np.ndarray, train_keys: np.ndarray,
                     valid_users: np.ndarray, valid_keys: np.ndarray,
                     y: np.ndarray, smoothing: float):
    """Bayesian user-key rates: LOO for train, complete train mapping for valid."""
    users = train_users.astype(np.int64)
    vu = valid_users.astype(np.int64)
    n_users = int(max(users.max(initial=0), vu.max(initial=0))) + 1
    ucnt = np.bincount(users, minlength=n_users).astype(np.float64)
    upos = np.bincount(users, weights=y, minlength=n_users).astype(np.float64)
    global_mean = float(y.mean())
    denom = np.maximum(ucnt[users] - 1.0, 0.0)
    prior_tr = np.divide(upos[users] - y, denom, out=np.full(len(y), global_mean), where=denom > 0)
    prior_va = np.divide(upos[vu], ucnt[vu], out=np.full(len(vu), global_mean), where=ucnt[vu] > 0)

    all_keys = np.concatenate((train_keys, valid_keys)).astype(np.int64)
    _, key_codes = np.unique(all_keys, return_inverse=True)
    kt = key_codes[:len(train_keys)].astype(np.int64)
    kv = key_codes[len(train_keys):].astype(np.int64)
    n_keys = int(key_codes.max(initial=0)) + 1
    pair = users * n_keys + kt
    unique_pair, inverse, counts = np.unique(pair, return_inverse=True, return_counts=True)
    positives = np.bincount(inverse, weights=y).astype(np.float64)
    cnt_row, pos_row = counts[inverse].astype(np.float64), positives[inverse]
    train_rate = (pos_row - y + smoothing * prior_tr) / (cnt_row - 1.0 + smoothing)

    lookup_order = np.argsort(unique_pair)
    lookup_keys = unique_pair[lookup_order]
    valid_pair = vu * n_keys + kv
    at = np.searchsorted(lookup_keys, valid_pair)
    seen = at < len(lookup_keys)
    seen[seen] &= lookup_keys[at[seen]] == valid_pair[seen]
    valid_rate = prior_va.copy()
    matched = lookup_order[at[seen]]
    valid_rate[seen] = ((positives[matched] + smoothing * prior_va[seen]) /
                        (counts[matched] + smoothing))
    return train_rate.astype(np.float32), valid_rate.astype(np.float32)


def _bucket_rates(train: np.ndarray, valid: np.ndarray, n_buckets: int):
    edges = np.unique(np.quantile(train, np.linspace(0, 1, n_buckets + 1)[1:-1]))
    return np.searchsorted(edges, train), np.searchsorted(edges, valid)


def add_affinity_features(ds: dict, smoothing: float = 10.0,
                          n_buckets: int = 20) -> dict:
    tr, va = ds["train"], ds["valid"]
    duration_edges = np.quantile(tr["duration_ms"], np.linspace(0, 1, 11)[1:-1])
    keys = [
        (tr["X"][:, 2], va["X"][:, 2]),  # author
        (tr["tab"], va["tab"]),
        (np.searchsorted(duration_edges, tr["duration_ms"]),
         np.searchsorted(duration_edges, va["duration_ms"])),
    ]
    columns = []
    for kt, kv in keys:
        rt, rv = _group_loo_rates(tr["users"], kt, va["users"], kv, tr["y"], smoothing)
        columns.append(_bucket_rates(rt, rv, n_buckets))
    return _append_columns(ds, columns)


def add_user_cross_features(ds: dict, n_rate_buckets: int = 20) -> dict:
    tr, va = ds["train"], ds["valid"]
    users, vu, y = tr["users"].astype(int), va["users"].astype(int), tr["y"]
    n_users = int(max(users.max(), vu.max())) + 1
    cnt = np.bincount(users, minlength=n_users)
    pos = np.bincount(users, weights=y, minlength=n_users)
    # LOO train global rate; full train rate for validation.
    g = float(y.mean())
    den = cnt[users] - 1
    rtr = np.divide(pos[users] - y, den, out=np.full(len(y), g), where=den > 0)
    rva = np.divide(pos[vu], cnt[vu], out=np.full(len(vu), g), where=cnt[vu] > 0)
    redges = np.unique(np.quantile(rtr, np.linspace(0, 1, n_rate_buckets + 1)[1:-1]))
    rbtr, rbva = np.searchsorted(redges, rtr), np.searchsorted(redges, rva)
    cbtr = np.clip(np.floor(np.log2(np.maximum(cnt[users], 1))).astype(int), 0, 15)
    cbva = np.clip(np.floor(np.log2(np.maximum(cnt[vu], 1))).astype(int), 0, 15)
    dedges = np.quantile(tr["duration_ms"], np.linspace(0, 1, 11)[1:-1])
    dbtr = np.searchsorted(dedges, tr["duration_ms"])
    dbva = np.searchsorted(dedges, va["duration_ms"])
    # Crosses vary within user: count x duration, rate x author.
    author_tr, author_va = tr["X"][:, 2], va["X"][:, 2]
    return _append_columns(ds, [(cbtr * 16 + dbtr, cbva * 16 + dbva),
                                (rbtr.astype(np.int64) * 100_000 + author_tr,
                                 rbva.astype(np.int64) * 100_000 + author_va)])


class PairSampler:
    def __init__(self, users: np.ndarray, y: np.ndarray):
        order = np.argsort(users, kind="stable")
        us, ys = users[order], y[order]
        starts = np.flatnonzero(np.r_[True, us[1:] != us[:-1]])
        ends = np.r_[starts[1:], len(us)]
        pos, negs, nstart, ncount, off = [], [], [], [], 0
        for s, e in zip(starts, ends):
            idx = order[s:e]
            p, n = idx[ys[s:e] == 1], idx[ys[s:e] == 0]
            if len(p) and len(n):
                pos.append(p); negs.append(n)
                nstart.append(np.full(len(p), off, dtype=np.int64))
                ncount.append(np.full(len(p), len(n), dtype=np.int64)); off += len(n)
        self.pos = np.concatenate(pos) if pos else np.empty(0, dtype=np.int64)
        self.negs = np.concatenate(negs) if negs else np.empty(0, dtype=np.int64)
        self.nstart = np.concatenate(nstart) if nstart else np.empty(0, dtype=np.int64)
        self.ncount = np.concatenate(ncount) if ncount else np.empty(0, dtype=np.int64)

    def sample(self, rng):
        neg = self.negs[self.nstart + rng.integers(0, self.ncount)]
        p = rng.permutation(len(self.pos))
        return self.pos[p], neg[p]


class HistoryDCN(nn.Module):
    def __init__(self, dim: int, n_fields: int, k: int = 16, hidden: int = 128,
                 aux_names=("click", "effective_view")):
        super().__init__()
        self.emb = nn.Embedding(dim, k)
        nn.init.normal_(self.emb.weight, std=0.01)
        d = n_fields * k
        self.cross_w = nn.ModuleList([nn.Linear(d, d) for _ in range(2)])
        self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(0.1))
        self.heads = nn.ModuleDict({x: nn.Linear(hidden, 1) for x in ("main", *aux_names)})

    def forward(self, x):
        x0 = self.emb(x).flatten(1)
        xl = x0
        for layer in self.cross_w:
            xl = x0 * layer(xl) + xl
        h = self.mlp(xl)
        return {name: head(h).squeeze(1) for name, head in self.heads.items()}


def init_covisit_embeddings(model: HistoryDCN, ds: dict, k: int, seed: int) -> None:
    """Factorize the implicit item co-visitation matrix A.T@A with truncated SVD."""
    tr = ds["train"]
    users = tr["users"].astype(np.int64)
    items = tr["X"][:, 1].astype(np.int64)
    uvals, ui = np.unique(users, return_inverse=True)
    ivals, ii = np.unique(items, return_inverse=True)
    incidence = sparse.coo_matrix((np.ones(len(ii), dtype=np.float32), (ui, ii)),
                                  shape=(len(uvals), len(ivals))).tocsr()
    incidence.data[:] = 1.0
    # Right singular vectors are eigenvectors of item-item co-visitation A.T A.
    _, s, vt = svds(incidence, k=k, which="LM", random_state=seed)
    order = np.argsort(s)[::-1]
    item_vec = (vt[order].T * np.sqrt(s[order])).astype(np.float32)
    item_vec -= item_vec.mean(axis=0, keepdims=True)
    item_vec *= 0.01 / max(float(item_vec.std()), 1e-8)
    with torch.no_grad():
        model.emb.weight[torch.as_tensor(ivals)] = torch.as_tensor(item_vec)


def metrics(users, labels, scores):
    m = official_evaluate(list(users), list(labels), list(scores))
    return {"gauc": float(m["GAUC"]), "ndcg5": float(m["nDCG@5"]),
            "primary": float(m["primary"])}


def train(model: HistoryDCN, ds: dict, args, sample_weights: np.ndarray | None = None):
    start = time.time(); set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    tr, va = ds["train"], ds["valid"]
    Xtr = torch.as_tensor(np.ascontiguousarray(tr["X"]), dtype=torch.long)
    Xva = torch.as_tensor(np.ascontiguousarray(va["X"]), dtype=torch.long)
    y = torch.as_tensor(tr["y"], dtype=torch.float32)
    click = torch.as_tensor(tr["click"], dtype=torch.float32)
    effective = torch.as_tensor(tr["play_time_ms"] >= np.minimum(tr["duration_ms"], 18_000),
                                dtype=torch.float32)
    weights = torch.ones(len(y)) if sample_weights is None else torch.as_tensor(sample_weights, dtype=torch.float32)
    sampler = PairSampler(tr["users"], tr["y"])
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    def predict(X):
        model.eval()
        with torch.no_grad():
            ans = torch.cat([model(X[i:i + 200_000])["main"] for i in range(0, len(X), 200_000)])
        model.train()
        return ans.numpy()

    best_gauc, best_state, bad = -1.0, None, 0
    n, bs = len(y), args.batch_size
    for epoch in range(1, args.epochs + 1):
        t0 = time.time(); pp, pn = sampler.sample(rng); nb = math.ceil(n / bs)
        order = rng.permutation(n); losses = []
        for b in range(nb):
            idx = order[b * bs:(b + 1) * bs]
            out = model(Xtr[idx]); w = weights[idx]
            point = nn.functional.binary_cross_entropy_with_logits(out["main"], y[idx], reduction="none")
            loss = 0.5 * (point * w).sum() / w.sum()
            lo, hi = b * len(pp) // nb, (b + 1) * len(pp) // nb
            if hi > lo:
                zp = model(Xtr[pp[lo:hi]])["main"]
                zn = model(Xtr[pn[lo:hi]])["main"]
                pw = 0.5 * (weights[pp[lo:hi]] + weights[pn[lo:hi]])
                pair = nn.functional.softplus(zn - zp)
                loss = loss + 0.5 * (pair * pw).sum() / pw.sum()
            for name, target in (("click", click), ("effective_view", effective)):
                aux = nn.functional.binary_cross_entropy_with_logits(out[name], target[idx], reduction="none")
                loss = loss + 0.1 * (aux * w).sum() / w.sum()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            losses.append(float(loss))
        vm = metrics(va["users"], va["y"], predict(Xva))
        print(f"epoch {epoch:2d} | loss {np.mean(losses):.4f} | valid gauc {vm['gauc']:.4f} "
              f"primary {vm['primary']:.4f} | {time.time()-t0:.1f}s", flush=True)
        if vm["gauc"] > best_gauc + 1e-5:
            best_gauc, best_state, bad = vm["gauc"], copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop at epoch {epoch}", flush=True); break
    model.load_state_dict(best_state)
    scores = predict(Xva)
    result = metrics(va["users"], va["y"], scores)
    result["runtime_s"] = round(time.time() - start, 1)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    with (out / "predictions.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(("row_id", "user_id", "video_id", "score"))
        for i, (u, v, score) in enumerate(zip(va["users"], va["videos"], scores)):
            w.writerow((i, int(u), int(v), f"{float(score):.10f}"))
    with (out / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(result, fh, sort_keys=True); fh.write("\n")
    print("final:", result, flush=True)
    return result


def prepare(args) -> tuple[dict, np.ndarray | None]:
    ds = add_best_features(load_workspace(args.data_dir, args.subsample))
    affinity = args.experiment == "affinity" or args.with_affinity or args.experiment == "din"
    if affinity:
        ds = add_affinity_features(ds, args.smoothing, args.rate_buckets)
    if args.experiment == "user-cross":
        ds = add_user_cross_features(ds, args.rate_buckets)
    weights = None
    if args.experiment == "recency":
        age = TRAIN_HI - ds["train"]["date"].astype(np.int64)
        weights = np.exp2(-age / args.half_life).astype(np.float32)
        weights /= weights.mean()
    return ds, weights


def run(args):
    if args.experiment == "din":
        raise NotImplementedError("DIN-lite is gated on a confirmed affinity win")
    ds, weights = prepare(args)
    set_seed(args.seed)
    model = HistoryDCN(ds["field_dims_total"], ds["train"]["X"].shape[1], args.k)
    if args.experiment == "covisit":
        init_covisit_embeddings(model, ds, args.k, args.seed)
    return train(model, ds, args, weights)


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
