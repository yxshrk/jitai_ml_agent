"""Validation-only multi-task FM with a long_view ranking head and click auxiliary.

The click task shares embeddings with long_view but has its own linear head. The
validation checkpoint is selected only by the fixed long_view GAUC/nDCG@5
evaluator, never by auxiliary-task accuracy or test data.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from evaluate import evaluate
from temporal_fm import HYPOTHESIS as CONTEXT_HYPOTHESIS
from temporal_fm import encode, load_rows


TASKS = ("long_view", "is_click")
HYPOTHESIS = (
    "A lightweight click auxiliary task will improve the long_view ranking head by "
    "training shared user-item-context embeddings on additional engagement signal."
)


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30, 30)))


class MultiTaskFM:
    def __init__(self, dimension, embedding_dim, learning_rate, task_weights, seed):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dimension, embedding_dim)).astype(np.float32)
        self.W = np.zeros((len(TASKS), dimension), dtype=np.float32)
        self.b = np.zeros(len(TASKS), dtype=np.float32)
        self.task_weights = np.asarray(task_weights, dtype=np.float32)
        self.learning_rate = learning_rate
        self.l2 = 1e-6
        self.mV = np.zeros_like(self.V)
        self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)
        self.vb = np.zeros_like(self.b)
        self.t = 0

    def logits(self, features):
        embeddings = self.V[features]
        summed_embeddings = embeddings.sum(1)
        interaction = 0.5 * (
            (summed_embeddings**2).sum(1) - (embeddings**2).sum((1, 2))
        )
        linear = self.W[:, features].sum(2).T
        return linear + interaction[:, None] + self.b, embeddings, summed_embeddings

    def step(self, features, labels):
        logits, embeddings, summed_embeddings = self.logits(features)
        batch_size = len(labels)
        gradients = (sigmoid(logits) - labels) * self.task_weights / batch_size
        shared_gradient = gradients.sum(axis=1)

        grad_v = np.zeros_like(self.V)
        grad_w = np.zeros_like(self.W)
        for task_index in range(len(TASKS)):
            np.add.at(grad_w[task_index], features, gradients[:, task_index, None])
        np.add.at(
            grad_v,
            features,
            shared_gradient[:, None, None] * (summed_embeddings[:, None, :] - embeddings),
        )
        grad_v += self.l2 * self.V
        grad_w += self.l2 * self.W
        grad_b = gradients.sum(axis=0)

        self.t += 1
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        for parameter, gradient, momentum, variance in (
            (self.V, grad_v, self.mV, self.vV),
            (self.W, grad_w, self.mW, self.vW),
            (self.b, grad_b, self.mb, self.vb),
        ):
            momentum *= beta1
            momentum += (1 - beta1) * gradient
            variance *= beta2
            variance += (1 - beta2) * (gradient * gradient)
            parameter -= self.learning_rate * (momentum / (1 - beta1**self.t)) / (
                np.sqrt(variance / (1 - beta2**self.t)) + epsilon
            )
        return float(
            np.mean(
                -self.task_weights
                * (labels * np.log(sigmoid(logits) + 1e-9) + (1 - labels) * np.log(1 - sigmoid(logits) + 1e-9))
            )
        )

    def predict_long_view(self, features, batch_size=200_000):
        return np.concatenate(
            [self.logits(features[start : start + batch_size])[0][:, 0] for start in range(0, len(features), batch_size)]
        )


def task_labels(rows):
    return {
        split: np.asarray([[row["targets"][task] for task in TASKS] for row in split_rows], dtype=np.float32)
        for split, split_rows in rows.items()
    }


def append_log(path, record):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def run(args):
    extra_fields = tuple(field.strip() for field in args.extra_features.split(",") if field.strip())
    rows = load_rows(args.data_dir)
    encoded, dimension, fields = encode(rows, extra_fields)
    labels = task_labels(rows)
    train_x, _, _ = encoded["train"]
    valid_x, valid_long_view, valid_users = encoded["valid"]
    model = MultiTaskFM(
        dimension=dimension,
        embedding_dim=args.embedding_dim,
        learning_rate=args.learning_rate,
        task_weights=(1.0, args.click_weight),
        seed=args.seed,
    )
    rng = np.random.default_rng(args.seed)
    log_path = Path(args.run_log) if args.run_log else None
    if log_path is not None and log_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing run log: {log_path}")

    best_primary = -np.inf
    best_state = None
    best_record = None
    stalled_epochs = 0
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(train_x))
        losses = [
            model.step(train_x[order[start : start + args.batch_size]], labels["train"][order[start : start + args.batch_size]])
            for start in range(0, len(order), args.batch_size)
        ]
        metrics = {
            name: float(value)
            for name, value in evaluate(valid_users, valid_long_view, model.predict_long_view(valid_x)).items()
        }
        record = {
            "phase": "multitask_fm",
            "iteration": epoch,
            "hypothesis": HYPOTHESIS,
            "context_hypothesis": CONTEXT_HYPOTHESIS,
            "change": f"FM fields: {fields}; shared tasks: {TASKS}; click_weight={args.click_weight}",
            "train_loss": round(float(np.mean(losses)), 7),
            "metrics": metrics,
            "error_or_recovery": None,
            "manual_interventions": 0,
        }
        append_log(log_path, record)
        print(
            f"epoch {epoch:2d} | loss {record['train_loss']:.4f} | GAUC {metrics['GAUC']:.4f} | "
            f"nDCG@5 {metrics['nDCG@5']:.4f} | primary {metrics['primary']:.4f}"
        )
        if metrics["primary"] > best_primary + 1e-5:
            best_primary = metrics["primary"]
            best_state = (model.V.copy(), model.W.copy(), model.b.copy())
            best_record = record
            stalled_epochs = 0
        else:
            stalled_epochs += 1
            if stalled_epochs >= args.patience:
                print(f"early stop at epoch {epoch}")
                break

    model.V, model.W, model.b = best_state
    print("\nBest validation result")
    print(json.dumps(best_record, indent=2, sort_keys=True))
    return best_record


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./KuaiRand-Pure/data")
    parser.add_argument("--extra_features", default="hour,weekday,is_rand")
    parser.add_argument("--click_weight", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--run_log", default=None, help="Optional JSONL iteration log path")
    return parser.parse_args()


if __name__ == "__main__":
    started = time.time()
    run(parse_args())
    print(f"elapsed_seconds={time.time() - started:.1f}")
