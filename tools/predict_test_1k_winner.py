"""Train the KuaiRand-1K champion and build the label-free test submission."""

import csv
import datetime as dt
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.official.evaluate import evaluate


DATA_DIR = ROOT / "data/real_ws_1k"
TEST_PATH = ROOT / "data/test_features_1k/test.npz"
OUTPUT_PATH = ROOT / "evidence/test_submission_1k.csv"
CHECKER_PATH = ROOT / "evidence/submission.py"
SEEDS = (42, 1051)
FINAL_EPOCHS = 8


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(data_dir):
    train_path = os.path.join(data_dir, "train.npz")
    val_path = os.path.join(data_dir, "val.npz")
    if not (os.path.exists(train_path) and os.path.exists(val_path)):
        raise FileNotFoundError(f"expected NPZ train/validation data in {data_dir}")
    tr = np.load(train_path, allow_pickle=False)
    va = np.load(val_path, allow_pickle=False)
    field_dims = np.asarray(tr["field_dims"], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1])).astype(np.int64)
    val_video = np.asarray(va["X"][:, 1], dtype=np.int64) - offsets[1]
    return {
        "train_x": np.asarray(tr["X"], dtype=np.int64),
        "val_x": np.asarray(va["X"], dtype=np.int64),
        "train_y": np.asarray(tr["y"], dtype=np.float32),
        "val_y": np.asarray(va["y"], dtype=np.float32),
        "train_user": np.asarray(tr["user"]),
        "val_user": np.asarray(va["user"]),
        "val_video": val_video,
        "train_date": np.asarray(tr["date"])
        if "date" in tr.files
        else np.zeros(len(tr["y"]), dtype=np.int64),
        "field_dims": field_dims,
        "fast": True,
    }


def official_evaluate(data, scores):
    result = evaluate(data["val_user"], data["val_y"], scores)
    return {
        "gauc": float(result.get("GAUC", result.get("gauc"))),
        "ndcg5": float(result.get("nDCG@5", result.get("ndcg5"))),
        "primary": float(result["primary"]),
    }


def date_ordinals(values):
    values = np.asarray(values)
    unique = np.unique(values)
    converted = {}
    for value in unique:
        text = str(value)
        if text.endswith(".0"):
            text = text[:-2]
        digits = "".join(ch for ch in text if ch.isdigit())
        try:
            if len(digits) >= 8:
                converted[value] = dt.datetime.strptime(
                    digits[:8], "%Y%m%d"
                ).date().toordinal()
            else:
                converted[value] = int(float(text))
        except Exception:
            converted[value] = 0
    return np.asarray([converted[v] for v in values], dtype=np.float32)


def recency_weights(dates, half_life):
    if half_life is None:
        return np.ones(len(dates), dtype=np.float32)
    ordinal = date_ordinals(dates)
    age = np.max(ordinal) - ordinal
    weights = np.exp2(-age / float(half_life)).astype(np.float32)
    return weights / max(float(weights.mean()), 1e-6)


def make_pairs(users, labels, seed):
    users = np.asarray(users)
    labels = np.asarray(labels)
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    boundaries = np.flatnonzero(
        np.r_[True, sorted_users[1:] != sorted_users[:-1], True]
    )
    rng = np.random.RandomState(seed)
    positives = []
    negatives = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        idx = order[left:right]
        pos = idx[labels[idx] > 0.5]
        neg = idx[labels[idx] <= 0.5]
        if len(pos) and len(neg):
            positives.append(pos)
            negatives.append(rng.choice(neg, size=len(pos), replace=True))
    if not positives:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(positives), np.concatenate(negatives)


class RankModel(nn.Module):
    def __init__(self, field_dims, k, architecture, dropout, embedding_dropout):
        super().__init__()
        total = int(np.sum(field_dims))
        self.architecture = architecture
        self.embedding = nn.Embedding(total, k)
        self.linear = nn.Embedding(total, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.zeros_(self.linear.weight)
        if architecture == "dcn-lite":
            dim = len(field_dims) * k
            self.cross_w = nn.Parameter(torch.empty(dim))
            self.cross_b = nn.Parameter(torch.zeros(dim))
            nn.init.normal_(self.cross_w, std=0.01)
            self.mlp = nn.Sequential(
                nn.Linear(dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.head = nn.Linear(dim + 64, 1)

    def forward(self, x):
        emb = self.embedding_dropout(self.embedding(x))
        linear = self.linear(x).sum(dim=1).squeeze(-1) + self.bias
        if self.architecture == "fm":
            summed = emb.sum(dim=1)
            interaction = 0.5 * (
                summed.square() - emb.square().sum(dim=1)
            ).sum(dim=1)
            return linear + self.dropout(interaction.unsqueeze(1)).squeeze(1)
        flat = emb.flatten(1)
        cross = (
            flat * torch.sum(flat * self.cross_w, dim=1, keepdim=True)
            + self.cross_b
            + flat
        )
        deep = self.mlp(flat)
        return linear + self.head(
            torch.cat([self.dropout(cross), deep], dim=1)
        ).squeeze(1)


def predict(model, x, device, batch_size):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.as_tensor(
                x[start : start + batch_size], dtype=torch.long, device=device
            )
            outputs.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(outputs).astype(np.float64)


def train_run(data, config, seed, epochs, device, pair_pos, pair_neg):
    set_seed(seed)
    model = RankModel(
        data["field_dims"],
        config["k"],
        config["architecture"],
        config["dropout"],
        config["embedding_dropout"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=config["lr_gamma"]
    )
    train_x = data["train_x"]
    train_y = data["train_y"]
    weights = recency_weights(data["train_date"], config["half_life"])
    rng = np.random.RandomState(seed + 771)
    batch_size = 8192
    eval_batch = 65536
    best_metrics = None
    best_prediction = None
    best_step = 0.0
    curve = []
    pair_order = np.arange(len(pair_pos), dtype=np.int64)

    for epoch in range(epochs):
        order = rng.permutation(len(train_x))
        if len(pair_order):
            pair_order = rng.permutation(len(pair_pos))
        n_batches = (len(order) + batch_size - 1) // batch_size
        halfway = (n_batches + 1) // 2
        running_loss = 0.0
        seen = 0
        model.train()
        for batch_no, start in enumerate(
            range(0, len(order), batch_size), 1
        ):
            idx = order[start : start + batch_size]
            xb = torch.as_tensor(train_x[idx], dtype=torch.long, device=device)
            yb = torch.as_tensor(train_y[idx], dtype=torch.float32, device=device)
            wb = torch.as_tensor(weights[idx], dtype=torch.float32, device=device)
            logits = model(xb)
            point_loss = (
                F.binary_cross_entropy_with_logits(logits, yb, reduction="none")
                * wb
            ).mean()
            alpha = config["bpr_alpha"]
            if alpha > 0.0 and len(pair_order):
                pstart = ((batch_no - 1) * batch_size) % len(pair_order)
                take = np.arange(pstart, pstart + len(idx)) % len(pair_order)
                selected = pair_order[take]
                pi = pair_pos[selected]
                ni = pair_neg[selected]
                px = torch.as_tensor(train_x[pi], dtype=torch.long, device=device)
                nx = torch.as_tensor(train_x[ni], dtype=torch.long, device=device)
                pair_w = torch.as_tensor(
                    0.5 * (weights[pi] + weights[ni]),
                    dtype=torch.float32,
                    device=device,
                )
                bpr = F.softplus(-(model(px) - model(nx)))
                pair_loss = (bpr * pair_w).mean()
                loss = (1.0 - alpha) * point_loss + alpha * pair_loss
            else:
                loss = point_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running_loss += float(loss.detach().cpu()) * len(idx)
            seen += len(idx)

            if batch_no == halfway or batch_no == n_batches:
                scores = predict(model, data["val_x"], device, eval_batch)
                metrics = official_evaluate(data, scores)
                step = epoch + (0.5 if batch_no == halfway else 1.0)
                curve.append(
                    {
                        "epoch": step,
                        "train_loss": running_loss / max(seen, 1),
                        "gauc": metrics["gauc"],
                        "ndcg5": metrics["ndcg5"],
                        "primary": metrics["primary"],
                    }
                )
                print(
                    f"seed={seed} checkpoint={step:.1f} "
                    f"gauc={metrics['gauc']:.9f} "
                    f"ndcg5={metrics['ndcg5']:.9f} "
                    f"primary={metrics['primary']:.9f}",
                    flush=True,
                )
                if best_metrics is None or metrics["primary"] > best_metrics["primary"]:
                    best_metrics = metrics
                    best_step = step
                    best_prediction = scores.copy()
                model.train()
        scheduler.step()

    del model, optimizer, scheduler
    return {
        "metrics": best_metrics,
        "best_epoch": best_step,
        "curve": curve,
        "prediction": best_prediction,
    }


def per_user_ranks(scores, users):
    scores = np.asarray(scores)
    users = np.asarray(users)
    output = np.empty(len(scores), dtype=np.float32)
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    boundaries = np.r_[
        0,
        np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1,
        len(order),
    ]
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        indices = order[left:right]
        local_order = np.argsort(scores[indices], kind="stable")
        ranks = np.empty(len(indices), dtype=np.float32)
        ranks[local_order] = np.arange(len(indices), dtype=np.float32)
        if len(indices) > 1:
            ranks /= float(len(indices) - 1)
        else:
            ranks.fill(0.5)
        output[indices] = ranks
    return output


# Keep node 002's RankModel and train_run definitions byte-for-byte equivalent
# to the tracked reproduction. These wrappers capture the validation-best state
# without changing the recipe's optimization or checkpoint-selection logic.
_RECIPE_RANK_MODEL = RankModel
_RECIPE_OFFICIAL_EVALUATE = official_evaluate
_ACTIVE_MODEL = None
_CHECKPOINT_TRACKER = None


class _CheckpointTracker:
    def __init__(self):
        self.best_primary = None
        self.best_state = None

    def consider(self, metrics):
        global _ACTIVE_MODEL
        primary = metrics["primary"]
        if self.best_primary is not None and primary <= self.best_primary:
            return
        if _ACTIVE_MODEL is None:
            raise RuntimeError("checkpoint capture has no active model")
        state = _ACTIVE_MODEL.state_dict()
        if self.best_state is None:
            self.best_state = {
                key: value.detach().cpu().clone() for key, value in state.items()
            }
        else:
            for key, value in state.items():
                self.best_state[key].copy_(value.detach().cpu())
        self.best_primary = primary


def _tracked_rank_model(*args, **kwargs):
    global _ACTIVE_MODEL
    _ACTIVE_MODEL = _RECIPE_RANK_MODEL(*args, **kwargs)
    return _ACTIVE_MODEL


def _tracked_official_evaluate(data, scores):
    metrics = _RECIPE_OFFICIAL_EVALUATE(data, scores)
    if _CHECKPOINT_TRACKER is not None:
        _CHECKPOINT_TRACKER.consider(metrics)
    return metrics


RankModel = _tracked_rank_model
official_evaluate = _tracked_official_evaluate


def resembles_label_key(key):
    normalized = key.lower().strip()
    tokens = normalized.replace("-", "_").split("_")
    return (
        "label" in normalized
        or "y" in tokens
        or normalized in {"target", "targets", "long_view", "click"}
    )


def load_label_free_test(path):
    with np.load(path, allow_pickle=False) as archive:
        suspicious = [key for key in archive.files if resembles_label_key(key)]
        assert not suspicious, f"test archive contains label-like keys: {suspicious}"
        required = {"X", "user_id", "video_id"}
        missing = required.difference(archive.files)
        assert not missing, f"test archive missing required keys: {sorted(missing)}"
        # Read only the three permitted arrays from the hidden-test archive.
        test_x = np.asarray(archive["X"], dtype=np.int64)
        users = np.asarray(archive["user_id"])
        videos = np.asarray(archive["video_id"])
    assert test_x.ndim == 2 and test_x.shape[1] == 5
    assert len(test_x) == len(users) == len(videos)
    return test_x, users, videos


def write_submission(path, users, videos, scores):
    if not (len(users) == len(videos) == len(scores)):
        raise RuntimeError("submission arrays have different lengths")
    if not np.all(np.isfinite(scores)):
        raise RuntimeError("submission scores contain NaN or infinity")
    with open(path, "w", newline="", buffering=1024 * 1024) as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for row_id, (user, video, score) in enumerate(zip(users, videos, scores)):
            writer.writerow(
                [row_id, int(user), int(video), format(float(score), ".9g")]
            )


def main():
    global _ACTIVE_MODEL, _CHECKPOINT_TRACKER

    device = torch.device("cpu")
    data = load_data(DATA_DIR)
    test_x, test_users, test_videos = load_label_free_test(TEST_PATH)
    if int(test_x.min()) < 0 or int(test_x.max()) >= int(data["field_dims"].sum()):
        raise RuntimeError("test feature index falls outside the training field dimensions")

    config = {
        "architecture": "dcn-lite",
        "loss": "logloss",
        "weighting": "uniform",
        "regularization": "refined-mild",
        "k": 24,
        "lr": 0.00168,
        "dropout": 0.13,
        "embedding_dropout": 0.0,
        "weight_decay": 3.7e-5,
        "lr_gamma": 0.95,
        "bpr_alpha": 0.0,
        "half_life": None,
    }
    pair_pos, pair_neg = make_pairs(
        data["train_user"], data["train_y"], seed=42 + 991
    )

    validation_ranks = []
    test_ranks = []
    for seed in SEEDS:
        _CHECKPOINT_TRACKER = _CheckpointTracker()
        result = train_run(
            data, config, seed, FINAL_EPOCHS, device, pair_pos, pair_neg
        )
        metrics = result["metrics"]
        if _CHECKPOINT_TRACKER.best_state is None or _ACTIVE_MODEL is None:
            raise RuntimeError(f"seed {seed} did not produce a captured checkpoint")
        if not np.isclose(
            _CHECKPOINT_TRACKER.best_primary, metrics["primary"], rtol=0.0, atol=0.0
        ):
            raise RuntimeError(f"seed {seed} checkpoint capture disagrees with train_run")

        validation_scores = np.asarray(result["prediction"], dtype=np.float64)
        _ACTIVE_MODEL.load_state_dict(_CHECKPOINT_TRACKER.best_state)
        test_scores = predict(_ACTIVE_MODEL, test_x, device, 65536)
        if not np.all(np.isfinite(validation_scores)) or not np.all(
            np.isfinite(test_scores)
        ):
            raise RuntimeError(f"seed {seed} produced non-finite predictions")
        validation_ranks.append(per_user_ranks(validation_scores, data["val_user"]))
        test_ranks.append(per_user_ranks(test_scores, test_users))
        print(
            f"RESULT member seed={seed} validation "
            f"gauc={metrics['gauc']:.9f} ndcg5={metrics['ndcg5']:.9f} "
            f"primary={metrics['primary']:.9f} "
            f"best_checkpoint={result['best_epoch']:.1f}",
            flush=True,
        )

        _ACTIVE_MODEL = None
        _CHECKPOINT_TRACKER = None
        del validation_scores, test_scores, result

    validation_ensemble = np.stack(validation_ranks).mean(axis=0)
    ensemble_metrics = _RECIPE_OFFICIAL_EVALUATE(data, validation_ensemble)
    test_ensemble = np.stack(test_ranks).mean(axis=0)
    print(
        f"RESULT validation ensemble gauc={ensemble_metrics['gauc']:.9f} "
        f"ndcg5={ensemble_metrics['ndcg5']:.9f} "
        f"primary={ensemble_metrics['primary']:.9f}",
        flush=True,
    )

    write_submission(OUTPUT_PATH, test_users, test_videos, test_ensemble)
    print(f"WROTE {OUTPUT_PATH}: {len(test_ensemble):,d} rows", flush=True)
    command = [
        sys.executable,
        str(CHECKER_PATH),
        "--check",
        str(OUTPUT_PATH),
        str(TEST_PATH),
    ]
    print(
        "CHECKER COMMAND: evidence/submission.py --check "
        "evidence/test_submission_1k.csv data/test_features_1k/test.npz",
        flush=True,
    )
    checked = subprocess.run(
        command, cwd=ROOT, check=False, text=True, capture_output=True
    )
    if checked.stdout:
        print(checked.stdout, end="", flush=True)
    if checked.stderr:
        print(checked.stderr, end="", file=sys.stderr, flush=True)
    if checked.returncode != 0:
        raise subprocess.CalledProcessError(checked.returncode, command)


if __name__ == "__main__":
    main()
