"""Validation-only pairwise FM experiment for KuaiRand-Pure.

This keeps the organizer's data loader and evaluator intact. It warm-starts the
official pointwise FM, then fine-tunes it with BPR pairs sampled from a user's
training impressions. The objective is aligned with the benchmark's
within-user ranking metrics (GAUC and nDCG@5).
"""

import argparse
import collections
import json
from pathlib import Path
import time

import numpy as np

import baseline as organizer_baseline
from data import encode, load
from evaluate import evaluate


HYPOTHESIS = (
    "Fine-tuning the official FM with within-user positive/negative BPR pairs "
    "will improve ranking metrics over pointwise log loss alone."
)
CHANGE_DESCRIPTION = (
    "Warm-start the organizer FM with pointwise log loss, then optimize sampled "
    "within-user positive-versus-negative BPR pairs."
)


class BPRFM(organizer_baseline.FM):
    """Organizer FM with one BPR update method added; no evaluator changes."""

    def pair_step(self, positive_x, negative_x):
        """Minimize mean(-log(sigmoid(score(pos) - score(neg))))."""
        batch_size = len(positive_x)
        positive_logits, positive_e, positive_s = self.logits(positive_x)
        negative_logits, negative_e, negative_s = self.logits(negative_x)
        margins = positive_logits - negative_logits

        # d loss / d margin. The global bias cancels for pairs from one user.
        grad_margin = (organizer_baseline.sigmoid(margins) - 1.0) / batch_size
        grad_v = np.zeros_like(self.V)
        grad_w = np.zeros_like(self.W)
        np.add.at(grad_w, positive_x, grad_margin[:, None])
        np.add.at(grad_w, negative_x, -grad_margin[:, None])
        np.add.at(
            grad_v,
            positive_x,
            grad_margin[:, None, None] * (positive_s[:, None, :] - positive_e),
        )
        np.add.at(
            grad_v,
            negative_x,
            -grad_margin[:, None, None] * (negative_s[:, None, :] - negative_e),
        )
        grad_v += self.l2 * self.V
        grad_w += self.l2 * self.W

        self.t += 1
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        for parameter, gradient, momentum, variance in (
            (self.V, grad_v, self.mV, self.vV),
            (self.W, grad_w, self.mW, self.vW),
        ):
            momentum *= beta1
            momentum += (1 - beta1) * gradient
            variance *= beta2
            variance += (1 - beta2) * (gradient * gradient)
            parameter -= self.lr * (momentum / (1 - beta1**self.t)) / (
                np.sqrt(variance / (1 - beta2**self.t)) + epsilon
            )

        return float(np.mean(np.logaddexp(0.0, -margins)))


def build_pair_index(users, labels):
    """Return each positive training row and the same user's negative rows."""
    negatives_by_user = collections.defaultdict(list)
    positive_indices = []
    positive_users = []
    for index, (user_id, label) in enumerate(zip(users, labels)):
        if label:
            positive_indices.append(index)
            positive_users.append(user_id)
        else:
            negatives_by_user[user_id].append(index)

    usable_indices = []
    usable_users = []
    for index, user_id in zip(positive_indices, positive_users):
        if negatives_by_user[user_id]:
            usable_indices.append(index)
            usable_users.append(user_id)
    return (
        np.asarray(usable_indices, dtype=np.int64),
        usable_users,
        {user_id: np.asarray(indices, dtype=np.int64) for user_id, indices in negatives_by_user.items()},
    )


def pair_batches(rng, positive_indices, positive_users, negatives_by_user, batch_size):
    order = rng.permutation(len(positive_indices))
    for start in range(0, len(order), batch_size):
        selection = order[start : start + batch_size]
        positive_batch = positive_indices[selection]
        negative_batch = np.fromiter(
            (
                negatives_by_user[positive_users[index]][
                    rng.integers(len(negatives_by_user[positive_users[index]]))
                ]
                for index in selection
            ),
            dtype=np.int64,
            count=len(selection),
        )
        yield positive_batch, negative_batch


def metric_record(phase, epoch, loss, users, labels, scores):
    metrics = {name: float(value) for name, value in evaluate(users, labels, scores).items()}
    return {
        "phase": phase,
        "iteration": epoch,
        "hypothesis": HYPOTHESIS,
        "change": CHANGE_DESCRIPTION,
        "train_loss": round(float(loss), 7),
        "metrics": metrics,
        "error_or_recovery": None,
        "manual_interventions": 0,
    }


def append_log(path, record):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def train(args):
    splits = load(args.data_dir)
    encoded, dimension = encode(splits)
    train_x, train_y, train_users = encoded["train"]
    valid_x, valid_y, valid_users = encoded["valid"]
    model = BPRFM(dimension, k=args.embedding_dim, lr=args.pointwise_lr, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    log_path = Path(args.run_log) if args.run_log else None
    if log_path is not None and log_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing run log: {log_path}")

    best_primary = -np.inf
    best_state = None
    best_record = None

    def validate(phase, epoch, loss):
        nonlocal best_primary, best_record, best_state
        record = metric_record(phase, epoch, loss, valid_users, valid_y, model.predict(valid_x))
        append_log(log_path, record)
        metrics = record["metrics"]
        print(
            f"{phase:9s} {epoch:2d} | loss {record['train_loss']:.4f} | "
            f"GAUC {metrics['GAUC']:.4f} | nDCG@5 {metrics['nDCG@5']:.4f} | "
            f"primary {metrics['primary']:.4f}"
        )
        if metrics["primary"] > best_primary + 1e-5:
            best_primary = metrics["primary"]
            best_record = record
            best_state = (model.V.copy(), model.W.copy(), np.float32(model.b))

    for epoch in range(1, args.warmup_epochs + 1):
        order = rng.permutation(len(train_y))
        losses = [
            model.step(train_x[order[start : start + args.batch_size]], train_y[order[start : start + args.batch_size]])
            for start in range(0, len(order), args.batch_size)
        ]
        validate("pointwise", epoch, np.mean(losses))

    positive_indices, positive_users, negatives_by_user = build_pair_index(train_users, train_y)
    print(f"pairwise pool: {len(positive_indices):,} positive rows across {len(negatives_by_user):,} users")
    model.lr = args.pairwise_lr
    for epoch in range(1, args.pairwise_epochs + 1):
        losses = [
            model.pair_step(train_x[positive_batch], train_x[negative_batch])
            for positive_batch, negative_batch in pair_batches(
                rng, positive_indices, positive_users, negatives_by_user, args.batch_size
            )
        ]
        validate("pairwise", epoch, np.mean(losses))

    model.V, model.W, model.b = best_state
    print("\nBest validation result")
    print(json.dumps(best_record, indent=2, sort_keys=True))
    return best_record


def self_test():
    model = BPRFM(dim=3, k=4, lr=0.05, l2=0.0, seed=7)
    positive_x = np.asarray([[0, 1]], dtype=np.int32)
    negative_x = np.asarray([[0, 2]], dtype=np.int32)
    before = float(model.logits(positive_x)[0][0] - model.logits(negative_x)[0][0])
    for _ in range(30):
        model.pair_step(positive_x, negative_x)
    after = float(model.logits(positive_x)[0][0] - model.logits(negative_x)[0][0])
    if after <= before:
        raise AssertionError(f"BPR margin did not improve: {before} -> {after}")
    print(f"BPR gradient self-test passed: margin {before:.4f} -> {after:.4f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./KuaiRand-Pure/data")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--pointwise_lr", type=float, default=0.001)
    parser.add_argument("--pairwise_lr", type=float, default=0.0005)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--pairwise_epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--run_log", default=None, help="Optional JSONL iteration log path")
    parser.add_argument("--self_test", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.self_test:
        self_test()
    else:
        started = time.time()
        train(arguments)
        print(f"elapsed_seconds={time.time() - started:.1f}")
