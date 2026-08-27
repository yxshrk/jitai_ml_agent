"""Shared infrastructure for zoo experiment scripts.

Every script follows CONTRACTS.md section 3: CLI `--data-dir <dir|real> --out-dir <o>
[--seed 42]`, deterministic per seed, writes predictions.csv (validation split) and
metrics.json scored with the vendored OFFICIAL evaluate.py (data/official/evaluate.py).
Training uses the hybrid loss 0.5*within-user-BPR + 0.5*logloss (MENU #1) and early
stopping on validation GAUC (MENU #2). The test split is never touched here.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
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
from data.real_loader import load_dataset


def make_parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--data-dir", required=True,
                    help="synthetic fixture dir, or 'real' for KuaiRand-Pure")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--subsample", type=int, default=None,
                    help="cap total train+valid rows (for smoke tests)")
    return ap


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


class PairSampler:
    """Within-user (positive, negative) pair sampler over the training split."""

    def __init__(self, users: np.ndarray, y: np.ndarray):
        order = np.argsort(users, kind="stable")
        u_sorted, y_sorted = users[order], y[order]
        starts = np.flatnonzero(np.r_[True, u_sorted[1:] != u_sorted[:-1]])
        ends = np.r_[starts[1:], len(u_sorted)]
        pos_list, neg_starts, neg_counts, neg_concat = [], [], [], []
        off = 0
        for s, e in zip(starts, ends):
            idx = order[s:e]
            pos = idx[y_sorted[s:e] == 1]
            neg = idx[y_sorted[s:e] == 0]
            if len(pos) and len(neg):
                pos_list.append(pos)
                neg_concat.append(neg)
                neg_starts.append(np.full(len(pos), off, dtype=np.int64))
                neg_counts.append(np.full(len(pos), len(neg), dtype=np.int64))
                off += len(neg)
        self.pos = np.concatenate(pos_list) if pos_list else np.empty(0, dtype=np.int64)
        self.neg_concat = np.concatenate(neg_concat) if neg_concat else np.empty(0, dtype=np.int64)
        self.neg_start = np.concatenate(neg_starts) if neg_starts else np.empty(0, dtype=np.int64)
        self.neg_count = np.concatenate(neg_counts) if neg_counts else np.empty(0, dtype=np.int64)

    def sample_epoch(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """One negative per eligible positive impression, shuffled."""
        if not len(self.pos):
            return self.pos, self.pos
        neg = self.neg_concat[self.neg_start + rng.integers(0, self.neg_count)]
        perm = rng.permutation(len(self.pos))
        return self.pos[perm], neg[perm]


def encode_extra_column(train_vals: np.ndarray, split_vals: dict[str, np.ndarray],
                        offset: int) -> tuple[dict[str, np.ndarray], int]:
    """Encode one extra categorical column with a train vocab + trailing UNK slot,
    mirroring the official UNK convention. Returns per-split id arrays and new offset."""
    vocab = {v: i for i, v in enumerate(dict.fromkeys(train_vals.tolist()))}
    unk = len(vocab)
    out = {}
    for name, vals in split_vals.items():
        out[name] = np.asarray([vocab.get(v, unk) for v in vals.tolist()],
                               dtype=np.int64) + offset
    return out, offset + unk + 1


def metrics_from_official(users, labels, scores) -> dict[str, float]:
    r = official_evaluate(list(users), list(labels), list(scores))
    return {"gauc": float(r["GAUC"]), "ndcg5": float(r["nDCG@5"]),
            "primary": float(r["primary"])}


def train_and_report(model: nn.Module, ds: dict, args, aux_targets=None,
                     aux_weight: float = 0.2, bpr_weight: float = 0.5) -> dict[str, float]:
    """Hybrid-loss training loop with early stopping on validation GAUC.

    aux_targets: optional dict name -> np.ndarray of train-split binary targets;
    the model must then return a dict {'main': logit, name: logit, ...}.
    """
    t_start = time.time()
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    tr, va = ds["train"], ds["valid"]
    Xtr = torch.as_tensor(np.ascontiguousarray(tr["X"]), dtype=torch.long)
    ytr = torch.as_tensor(tr["y"], dtype=torch.float32)
    Xva = torch.as_tensor(np.ascontiguousarray(va["X"]), dtype=torch.long)
    aux = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in (aux_targets or {}).items()}

    sampler = PairSampler(tr["users"], tr["y"])
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()

    def fwd(xb):
        out = model(xb)
        return out if isinstance(out, dict) else {"main": out}

    def predict(X, bs=200_000):
        model.eval()
        with torch.no_grad():
            s = torch.cat([fwd(X[i:i + bs])["main"] for i in range(0, len(X), bs)])
        model.train()
        return s.numpy()

    n = len(ytr)
    bs = args.batch_size
    best_gauc, best_state, bad = -1.0, None, 0
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        p_pos, p_neg = sampler.sample_epoch(rng)
        n_pairs = len(p_pos)
        point_order = rng.permutation(n)
        n_batches = (n + bs - 1) // bs
        losses = []
        for b in range(n_batches):
            idx = point_order[b * bs:(b + 1) * bs]
            xb = Xtr[idx]
            out = fwd(xb)
            loss = (1.0 - bpr_weight) * bce(out["main"], ytr[idx])
            if n_pairs and bpr_weight > 0:  # matching slice of the epoch's pair pool
                lo = (b * n_pairs) // n_batches
                hi = ((b + 1) * n_pairs) // n_batches
                if hi > lo:
                    zp = fwd(Xtr[p_pos[lo:hi]])["main"]
                    zn = fwd(Xtr[p_neg[lo:hi]])["main"]
                    loss = loss + bpr_weight * nn.functional.softplus(zn - zp).mean()
            for name, t in aux.items():
                loss = loss + aux_weight * bce(out[name], t[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss))
        va_m = metrics_from_official(va["users"], va["y"], predict(Xva))
        print(f"epoch {ep:2d} | loss {np.mean(losses):.4f} | valid gauc {va_m['gauc']:.4f} "
              f"primary {va_m['primary']:.4f} | {time.time() - t0:.1f}s", flush=True)
        if va_m["gauc"] > best_gauc + 1e-5:
            best_gauc, bad = va_m["gauc"], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop at epoch {ep}", flush=True)
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    scores = predict(Xva)
    metrics = metrics_from_official(va["users"], va["y"], scores)
    metrics["runtime_s"] = round(time.time() - t_start, 1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["row_id", "user_id", "video_id", "score"])
        for i, (u, v, s) in enumerate(zip(va["users"], va["videos"], scores)):
            w.writerow([i, int(u), int(v), f"{float(s):.10f}"])
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, sort_keys=True)
        fh.write("\n")
    print("final:", metrics, flush=True)
    return metrics


def load_for_args(args) -> dict:
    return load_dataset(args.data_dir, subsample=args.subsample)
