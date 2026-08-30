"""Validation-only contextual FM with a class-weighted long_view objective."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

import baseline as organizer_baseline
from evaluate import evaluate
from temporal_fm import encode, load_rows


HYPOTHESIS = (
    "A modest positive-class weight will improve top-of-list long_view ordering "
    "without destabilizing GAUC."
)


class WeightedFM(organizer_baseline.FM):
    def step(self, features, labels, positive_weight):
        logits, embeddings, summed_embeddings = self.logits(features)
        example_weights = np.where(labels > 0, positive_weight, 1.0).astype(np.float32)
        gradients = (organizer_baseline.sigmoid(logits) - labels) * example_weights / example_weights.sum()
        grad_v = np.zeros_like(self.V)
        grad_w = np.zeros_like(self.W)
        np.add.at(grad_w, features, gradients[:, None])
        np.add.at(grad_v, features, gradients[:, None, None] * (summed_embeddings[:, None, :] - embeddings))
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
        self.b -= self.lr * gradients.sum()
        loss = -np.mean(
            example_weights
            * (labels * np.log(organizer_baseline.sigmoid(logits) + 1e-9)
               + (1 - labels) * np.log(1 - organizer_baseline.sigmoid(logits) + 1e-9))
        )
        return float(loss)


def append_log(path, record):
    if path is None:
        return
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing run log: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def run(args):
    rows = load_rows(args.data_dir)
    extra_fields = tuple(field.strip() for field in args.extra_features.split(",") if field.strip())
    encoded, dimension, fields = encode(rows, extra_fields)
    train_x, train_y, _ = encoded["train"]
    valid_x, valid_y, valid_users = encoded["valid"]
    model = WeightedFM(dimension, k=args.embedding_dim, lr=args.learning_rate, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    best_primary = -np.inf
    best_state = None
    best_event = None
    trajectory = []
    stalled_epochs = 0
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(train_y))
        losses = [
            model.step(
                train_x[order[start : start + args.batch_size]],
                train_y[order[start : start + args.batch_size]],
                args.positive_weight,
            )
            for start in range(0, len(order), args.batch_size)
        ]
        metrics = {name: float(value) for name, value in evaluate(valid_users, valid_y, model.predict(valid_x)).items()}
        event = {"epoch": epoch, "train_loss": round(float(np.mean(losses)), 7), "metrics": metrics}
        trajectory.append(event)
        print(
            f"epoch {epoch:2d} | loss {event['train_loss']:.4f} | GAUC {metrics['GAUC']:.4f} | "
            f"nDCG@5 {metrics['nDCG@5']:.4f} | primary {metrics['primary']:.4f}"
        )
        if metrics["primary"] > best_primary + 1e-5:
            best_primary = metrics["primary"]
            best_state = (model.V.copy(), model.W.copy(), np.float32(model.b))
            best_event = event
            stalled_epochs = 0
        else:
            stalled_epochs += 1
            if stalled_epochs >= args.patience:
                print(f"early stop at epoch {epoch}")
                break
    record = {
        "phase": "weighted_contextual_fm",
        "hypothesis": HYPOTHESIS,
        "positive_weight": args.positive_weight,
        "fields": fields,
        "best": best_event,
        "trajectory": trajectory,
        "error_or_recovery": None,
        "manual_interventions": 0,
        "test_data_used": False,
    }
    append_log(Path(args.run_log) if args.run_log else None, record)
    print("\nBest validation result")
    print(json.dumps(best_event, indent=2, sort_keys=True))
    return record


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./KuaiRand-Pure/data")
    parser.add_argument("--extra_features", default="hour,weekday,is_rand")
    parser.add_argument("--positive_weight", type=float, default=1.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--run_log", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    started = time.time()
    run(parse_args())
    print(f"elapsed_seconds={time.time() - started:.1f}")
