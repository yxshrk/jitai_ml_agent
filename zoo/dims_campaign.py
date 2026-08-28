"""Validation-only campaign for BPR sampling, auxiliary tasks, and optimizers.

Contract CLI:
  uv run python zoo/dims_campaign.py --data-dir data/real_ws --out-dir <o> [--seed 42]

The default is the confirmed strong-regularized five-field L0 DCN-lite control:
one cross layer, MLP128, 0.5 BPR + 0.5 logloss, no auxiliary loss, seven-day
recency weighting, MLP/embedding dropout 0.2/0.1, AdamW weight decay 1e-5, and
per-epoch 0.5 learning-rate decay. Only train.npz and val.npz are used for features
and labels; the matching CSV is read solely for raw video ids and auxiliary
columns that are not present in the NPZ export.
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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.official.evaluate import evaluate as official_evaluate

BASELINE_PRIMARY = 0.6016
AVAILABLE_NPZ_AUX = frozenset(("click", "effective_view", "play_fraction"))
KNOWN_AUX = frozenset((*AVAILABLE_NPZ_AUX, "like", "follow", "comment", "forward"))


def parser(description: str = __doc__) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--patience-halves", type=int, default=6)
    ap.add_argument("--max-runtime", type=int, default=330)
    ap.add_argument("--subsample", type=int)
    ap.add_argument("--negatives-per-positive", type=int, choices=(1, 3, 5), default=1)
    ap.add_argument("--negative-sampling",
                    choices=("uniform", "popularity", "hard", "hard-popularity"),
                    default="uniform")
    ap.add_argument("--aux-tasks", default="none",
                    help="comma-separated tasks, or 'none'")
    ap.add_argument("--aux-weights", default="none",
                    help="one comma-separated nonnegative weight per auxiliary task")
    ap.add_argument("--optimizer",
                    choices=("adamw", "adagrad", "sparse-adam", "sgd", "split-adagrad-adam"),
                    default="adamw")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--embedding-lr", type=float, default=0.05,
                    help="embedding LR for split-adagrad-adam")
    return ap


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _resolve_data_dir(value: str) -> Path:
    return ROOT / "data" / "real_ws" if value == "real" else Path(value)


def parse_aux(spec: str, weights: str) -> tuple[tuple[str, ...], tuple[float, ...]]:
    tasks = () if spec.strip().lower() in ("", "none") else tuple(x.strip() for x in spec.split(","))
    unknown = sorted(set(tasks) - KNOWN_AUX)
    if unknown:
        raise ValueError(f"unknown auxiliary task(s): {', '.join(unknown)}")
    if len(set(tasks)) != len(tasks):
        raise ValueError("auxiliary tasks must be unique")
    parsed = () if not tasks and weights.strip().lower() in ("", "none") else tuple(
        float(x.strip()) for x in weights.split(",") if x.strip())
    if not tasks and parsed:
        # The CLI default weights are harmless when --aux-tasks none is selected.
        parsed = ()
    if len(tasks) != len(parsed):
        raise ValueError("--aux-weights must contain one value per --aux-tasks entry")
    if any(x < 0 for x in parsed):
        raise ValueError("auxiliary weights must be nonnegative")
    return tasks, parsed


def _read_csv_columns(path: Path, columns: tuple[str, ...], expected: int) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [name for name in columns if name not in (reader.fieldnames or ())]
        if missing:
            raise ValueError(f"{path.name} missing requested auxiliary column(s): {', '.join(missing)}")
        values = {name: np.empty(expected, dtype=np.float32 if name != "video_id" else np.int64)
                  for name in columns}
        count = 0
        for count, row in enumerate(reader, start=1):
            if count > expected:
                raise ValueError(f"{path.name} has more than {expected} rows")
            for name in columns:
                values[name][count - 1] = row[name]
    if count != expected:
        raise ValueError(f"{path.name} row mismatch: {count} != {expected}")
    return values


def load_data(data_dir: str, aux_tasks: tuple[str, ...], subsample: int | None = None) -> dict:
    root = _resolve_data_dir(data_dir)
    result: dict = {}
    field_dims = None
    csv_aux = tuple(task for task in aux_tasks if task not in AVAILABLE_NPZ_AUX)
    for name, stem, low, high in (("train", "train", 20220408, 20220421),
                                  ("valid", "val", 20220422, 20220428)):
        with np.load(root / f"{stem}.npz", allow_pickle=False) as data:
            split = {key: np.asarray(data[key]).copy() for key in data.files if key != "field_dims"}
            dims = np.asarray(data["field_dims"], dtype=np.int64)
        if field_dims is None:
            field_dims = dims
        elif not np.array_equal(field_dims, dims):
            raise ValueError("train/validation field dimensions differ")
        dates = split["date"]
        if dates.size and (int(dates.min()) < low or int(dates.max()) > high):
            raise ValueError(f"{name} data outside {low}..{high}")
        needed = ("video_id", *csv_aux) if name == "train" else ("video_id",)
        csv_values = _read_csv_columns(root / f"{stem}.csv", needed, len(split["y"]))
        split["videos"] = csv_values.pop("video_id")
        split.update(csv_values)
        if subsample is not None:
            split = {key: value[:subsample] for key, value in split.items()}
        result[name] = split
    result["field_dims_total"] = int(field_dims.sum())
    return result


def make_aux_targets(split: dict, tasks: tuple[str, ...]) -> dict[str, torch.Tensor]:
    targets: dict[str, torch.Tensor] = {}
    for task in tasks:
        if task == "effective_view":
            value = split["play_time_ms"] >= np.minimum(split["duration_ms"], 18_000)
        elif task == "play_fraction":
            denom = np.minimum(split["duration_ms"], 18_000)
            value = np.divide(split["play_time_ms"], denom,
                              out=np.zeros_like(split["play_time_ms"], dtype=np.float32),
                              where=denom > 0)
            value = np.clip(value, 0.0, 3.0)
        else:
            value = split[task]
        targets[task] = torch.as_tensor(value, dtype=torch.float32)
    return targets


class PairSampler:
    """Sample within-user negatives with configurable count and candidate policy."""

    def __init__(self, users: np.ndarray, labels: np.ndarray, items: np.ndarray):
        exposure = np.bincount(items.astype(np.int64))
        self.groups: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        order = np.argsort(users, kind="stable")
        sorted_users = users[order]
        starts = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1]])
        ends = np.r_[starts[1:], len(order)]
        for start, end in zip(starts, ends):
            rows = order[start:end]
            pos, neg = rows[labels[rows] == 1], rows[labels[rows] == 0]
            if len(pos) and len(neg):
                pop = exposure[items[neg].astype(np.int64)].astype(np.float64) ** 0.75
                pop /= pop.sum()
                self.groups.append((pos, neg, pop))

    def sample(self, rng: np.random.Generator, count: int, mode: str,
               scores: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        positives, negatives = [], []
        hard = mode in ("hard", "hard-popularity")
        weighted = mode in ("popularity", "hard-popularity")
        if hard and scores is None:
            raise ValueError("hard-negative sampling requires current-model scores")
        for pos, neg, pop in self.groups:
            candidates, probabilities = neg, pop
            if hard:
                keep = max(1, math.ceil(len(neg) / 2))
                chosen = np.argpartition(scores[neg], -keep)[-keep:]
                candidates, probabilities = neg[chosen], pop[chosen]
                probabilities = probabilities / probabilities.sum()
            size = len(pos) * count
            draw = rng.choice(candidates, size=size, replace=True,
                              p=probabilities if weighted else None)
            positives.append(np.repeat(pos, count))
            negatives.append(draw)
        p = np.concatenate(positives) if positives else np.empty(0, dtype=np.int64)
        n = np.concatenate(negatives) if negatives else np.empty(0, dtype=np.int64)
        permutation = rng.permutation(len(p))
        return p[permutation], n[permutation]


class DimsDCN(nn.Module):
    def __init__(self, total_dim: int, fields: int, tasks: tuple[str, ...], k: int = 16,
                 sparse_embedding: bool = False):
        super().__init__()
        self.embedding = nn.Embedding(total_dim, k, sparse=sparse_embedding)
        nn.init.normal_(self.embedding.weight, std=0.01)
        self.embedding_dropout = nn.Dropout(0.1)
        width = fields * k
        self.cross = nn.Linear(width, width)
        self.mlp = nn.Sequential(nn.Linear(width, 128), nn.ReLU(), nn.Dropout(0.2))
        self.heads = nn.ModuleDict({name: nn.Linear(128, 1) for name in ("main", *tasks)})

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x0 = self.embedding_dropout(self.embedding(x)).flatten(1)
        hidden = self.mlp(x0 * self.cross(x0) + x0)
        return {name: head(hidden).squeeze(1) for name, head in self.heads.items()}


@dataclass
class Optimizers:
    values: tuple[torch.optim.Optimizer, ...]
    schedulers: tuple[torch.optim.lr_scheduler.LRScheduler, ...]

    def zero_grad(self) -> None:
        for value in self.values:
            value.zero_grad(set_to_none=True)

    def step(self) -> None:
        for value in self.values:
            value.step()

    def step_schedulers(self) -> None:
        for value in self.schedulers:
            value.step()


def make_optimizers(model: DimsDCN, args) -> Optimizers:
    embedding = list(model.embedding.parameters())
    dense = [parameter for name, parameter in model.named_parameters()
             if not name.startswith("embedding.")]
    if args.optimizer == "adamw":
        values = (torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5),)
    elif args.optimizer == "adagrad":
        values = (torch.optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=1e-5),)
    elif args.optimizer == "sgd":
        values = (torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                  weight_decay=1e-5),)
    elif args.optimizer == "sparse-adam":
        values = (torch.optim.SparseAdam(embedding, lr=args.lr),
                  torch.optim.Adam(dense, lr=args.lr))
    elif args.optimizer == "split-adagrad-adam":
        values = (torch.optim.Adagrad(embedding, lr=args.embedding_lr, weight_decay=1e-5),
                  torch.optim.Adam(dense, lr=args.lr))
    else:  # pragma: no cover - argparse owns the choices
        raise ValueError(args.optimizer)
    schedulers = tuple(torch.optim.lr_scheduler.StepLR(value, step_size=1, gamma=0.5)
                       for value in values)
    return Optimizers(values, schedulers)


def metrics(users: np.ndarray, labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    result = official_evaluate(users.tolist(), labels.astype(int).tolist(), scores.tolist())
    return {"gauc": float(result["GAUC"]), "ndcg5": float(result["nDCG@5"]),
            "primary": float(result["primary"])}


def _write(out_dir: str, valid: dict, scores: np.ndarray, result: dict) -> None:
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    with (target / "predictions.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(("row_id", "user_id", "video_id", "score"))
        for row_id, (user, video, score) in enumerate(zip(valid["user"], valid["videos"], scores)):
            writer.writerow((row_id, int(user), int(video), f"{float(score):.10f}"))
    with (target / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(result, fh, sort_keys=True)
        fh.write("\n")


def run(args) -> dict:
    started = time.time()
    tasks, task_weights = parse_aux(args.aux_tasks, args.aux_weights)
    if args.negatives_per_positive < 1:
        raise ValueError("negatives per positive must be positive")
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    ds = load_data(args.data_dir, tasks, args.subsample)
    train, valid = ds["train"], ds["valid"]
    xtr = torch.as_tensor(np.ascontiguousarray(train["X"]), dtype=torch.long)
    xva = torch.as_tensor(np.ascontiguousarray(valid["X"]), dtype=torch.long)
    ytr = torch.as_tensor(train["y"], dtype=torch.float32)
    aux = make_aux_targets(train, tasks)
    sample_weights_np = np.exp2(-(20220421 - train["date"].astype(np.int64)) / 7.0)
    sample_weights_np = (sample_weights_np / sample_weights_np.mean()).astype(np.float32)
    sample_weights = torch.as_tensor(sample_weights_np)
    sparse_embedding = args.optimizer == "sparse-adam"
    model = DimsDCN(ds["field_dims_total"], xtr.shape[1], tasks,
                    sparse_embedding=sparse_embedding)
    optimizers = make_optimizers(model, args)
    sampler = PairSampler(train["user"], train["y"], train["X"][:, 1])

    def predict(x: torch.Tensor) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            value = torch.cat([model(x[i:i + 200_000])["main"]
                               for i in range(0, len(x), 200_000)]).numpy()
        model.train()
        return value

    order = np.arange(len(ytr))
    best_primary, best_state, best_step, bad = -math.inf, None, 0.0, 0
    history: list[dict] = []
    timed_out = False

    class RunTimeout(Exception):
        pass

    def timeout_handler(_signum, _frame):
        raise RunTimeout

    previous_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(args.max_runtime)
    try:
        stop = False
        for epoch in range(1, args.epochs + 1):
            hard_scores = predict(xtr) if args.negative_sampling.startswith("hard") else None
            pair_pos, pair_neg = sampler.sample(
                rng, args.negatives_per_positive, args.negative_sampling, hard_scores)
            order = rng.permutation(order)
            half_size = math.ceil(len(order) / 2)
            for half in range(2):
                rows = order[half * half_size:min(len(order), (half + 1) * half_size)]
                batches = math.ceil(len(rows) / args.batch_size)
                losses = []
                for batch in range(batches):
                    idx_np = rows[batch * args.batch_size:(batch + 1) * args.batch_size]
                    idx = torch.as_tensor(idx_np, dtype=torch.long)
                    output = model(xtr[idx])
                    weight = sample_weights[idx]
                    point = nn.functional.binary_cross_entropy_with_logits(
                        output["main"], ytr[idx], reduction="none")
                    point_loss = (point * weight).sum() / weight.sum()
                    pair_batch = half * batches + batch
                    lo = pair_batch * len(pair_pos) // (2 * batches)
                    hi = (pair_batch + 1) * len(pair_pos) // (2 * batches)
                    if hi > lo:
                        p = torch.as_tensor(pair_pos[lo:hi], dtype=torch.long)
                        n = torch.as_tensor(pair_neg[lo:hi], dtype=torch.long)
                        pair = nn.functional.softplus(
                            model(xtr[n])["main"] - model(xtr[p])["main"])
                        pair_weight = 0.5 * (sample_weights[p] + sample_weights[n])
                        pair_loss = (pair * pair_weight).sum() / pair_weight.sum()
                    else:
                        pair_loss = point_loss * 0.0
                    loss = 0.5 * point_loss + 0.5 * pair_loss
                    for task, task_weight in zip(tasks, task_weights):
                        if task == "play_fraction":
                            task_loss = nn.functional.mse_loss(output[task], aux[task][idx], reduction="none")
                        else:
                            task_loss = nn.functional.binary_cross_entropy_with_logits(
                                output[task], aux[task][idx], reduction="none")
                        loss = loss + task_weight * (task_loss * weight).sum() / weight.sum()
                    optimizers.zero_grad()
                    loss.backward()
                    optimizers.step()
                    losses.append(float(loss.detach()))
                score = predict(xva)
                observed = metrics(valid["user"], valid["y"], score)
                step = epoch - 0.5 if half == 0 else float(epoch)
                record = {"epoch": step, "train_loss": float(np.mean(losses)),
                          "val_gauc": observed["gauc"], "val_primary": observed["primary"]}
                history.append(record)
                print(f"step={step:.1f} loss={record['train_loss']:.5f} "
                      f"gauc={observed['gauc']:.6f} primary={observed['primary']:.6f}",
                      flush=True)
                if observed["primary"] > best_primary + 1e-7:
                    best_primary, best_state, best_step, bad = (
                        observed["primary"], copy.deepcopy(model.state_dict()), step, 0)
                else:
                    bad += 1
                    if bad >= args.patience_halves:
                        stop = True
                        break
            optimizers.step_schedulers()
            if stop:
                break
    except RunTimeout:
        timed_out = True
        print("runtime cap reached; preserving best completed half-epoch", flush=True)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
    if best_state is None:
        raise RuntimeError("run ended before its first completed validation checkpoint")
    model.load_state_dict(best_state)
    scores = predict(xva)
    final = metrics(valid["user"], valid["y"], scores)
    result = {**final, "history": history, "runtime_s": round(time.time() - started, 1),
              "best_step": best_step, "seed": args.seed, "timed_out": timed_out,
              "delta": final["primary"] - BASELINE_PRIMARY,
              "config": {"negatives_per_positive": args.negatives_per_positive,
                         "negative_sampling": args.negative_sampling,
                         "aux_tasks": list(tasks), "aux_weights": list(task_weights),
                         "optimizer": args.optimizer, "lr": args.lr,
                         "embedding_lr": args.embedding_lr, "recency_half_life_days": 7.0,
                          "cross_layers": 1, "hidden": 128, "bpr_weight": 0.5,
                         "mlp_dropout": 0.2, "embedding_dropout": 0.1,
                         "weight_decay": (0.0 if args.optimizer == "sparse-adam" else 1e-5),
                         "dense_weight_decay": (0.0 if args.optimizer in
                                                  ("sparse-adam", "split-adagrad-adam")
                                                  else 1e-5),
                         "lr_step_gamma": 0.5}}
    _write(args.out_dir, valid, scores, result)
    print("final:", json.dumps({k: v for k, v in result.items() if k != "history"},
                                sort_keys=True), flush=True)
    return result


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
