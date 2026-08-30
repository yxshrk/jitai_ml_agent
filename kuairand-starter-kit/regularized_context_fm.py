"""Validation-only regularized contextual factorization machine.

This experiment preserves the organizer's FM scoring path but applies field
dropout during training, Adam weight decay, and a deterministic step-wise
learning-rate schedule.  It uses only train/validation dates and refuses to
overwrite an experiment log.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np

import baseline as organizer_baseline
from evaluate import evaluate
from temporal_fm import ALLOWED_EXTRAS, encode, load_rows


HYPOTHESIS = (
    "Field dropout and scheduled regularization reduce contextual FM overfit "
    "while retaining its efficient within-user ranking signal."
)


class RegularizedFM(organizer_baseline.FM):
    """Organizer FM with training-only field dropout and controllable L2."""

    def step(self, features, labels, *, dropout, rng):
        batch_size, field_count = features.shape
        embeddings = self.V[features]
        field_mask = (rng.random((batch_size, field_count, 1)) >= dropout).astype(np.float32)
        masked_embeddings = embeddings * field_mask
        summed = masked_embeddings.sum(axis=1)
        interactions = 0.5 * ((summed**2).sum(axis=1) - (masked_embeddings**2).sum(axis=(1, 2)))
        masked_linear = self.W[features] * field_mask.squeeze(axis=2)
        logits = self.b + masked_linear.sum(axis=1) + interactions
        gradients = ((organizer_baseline.sigmoid(logits) - labels) / batch_size).astype(np.float32)

        gradient_v = np.zeros_like(self.V)
        gradient_w = np.zeros_like(self.W)
        np.add.at(gradient_w, features, gradients[:, None] * field_mask.squeeze(axis=2))
        np.add.at(
            gradient_v,
            features,
            gradients[:, None, None] * field_mask * (summed[:, None, :] - masked_embeddings),
        )
        gradient_v += self.l2 * self.V
        gradient_w += self.l2 * self.W
        self.t += 1
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        for parameter, gradient, first, second in (
            (self.V, gradient_v, self.mV, self.vV),
            (self.W, gradient_w, self.mW, self.vW),
        ):
            first *= beta1
            first += (1 - beta1) * gradient
            second *= beta2
            second += (1 - beta2) * (gradient * gradient)
            parameter -= self.lr * (first / (1 - beta1**self.t)) / (
                np.sqrt(second / (1 - beta2**self.t)) + epsilon
            )
        self.b -= self.lr * gradients.sum()
        return float(
            -np.mean(
                labels * np.log(organizer_baseline.sigmoid(logits) + 1e-9)
                + (1 - labels) * np.log(1 - organizer_baseline.sigmoid(logits) + 1e-9)
            )
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./KuaiRand-Pure/data")
    parser.add_argument("--extra_features", default="hour,weekday,is_rand")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.00009)
    parser.add_argument("--dropout", type=float, default=0.18)
    parser.add_argument("--lr_decay", type=float, default=0.57)
    parser.add_argument("--lr_step_epochs", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--run_log", required=True)
    return parser.parse_args()


def main(args):
    extra_fields = tuple(part.strip() for part in args.extra_features.split(",") if part.strip())
    unsupported = set(extra_fields) - ALLOWED_EXTRAS
    if unsupported:
        raise ValueError(f"Unsupported context fields: {sorted(unsupported)}")
    if not 0 <= args.dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    if not 0 < args.lr_decay <= 1:
        raise ValueError("lr_decay must be in (0, 1]")
    log_path = Path(args.run_log)
    if log_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing run log: {log_path}")

    rows = load_rows(args.data_dir)
    encoded, dimension, fields = encode(rows, extra_fields)
    train_x, train_y, _ = encoded["train"]
    valid_x, valid_y, valid_users = encoded["valid"]
    model = RegularizedFM(
        dimension, k=args.embedding_dim, lr=args.learning_rate, l2=args.weight_decay, seed=args.seed
    )
    rng = np.random.default_rng(args.seed)
    best_primary, best_state, best_record, stalled = -np.inf, None, None, 0
    trajectory = []
    for epoch in range(1, args.epochs + 1):
        model.lr = args.learning_rate * args.lr_decay ** ((epoch - 1) // args.lr_step_epochs)
        order = rng.permutation(len(train_y))
        losses = [
            model.step(train_x[order[start : start + args.batch_size]], train_y[order[start : start + args.batch_size]], dropout=args.dropout, rng=rng)
            for start in range(0, len(order), args.batch_size)
        ]
        metrics = {key: float(value) for key, value in evaluate(valid_users, valid_y, model.predict(valid_x)).items()}
        record = {
            "epoch": epoch,
            "learning_rate": model.lr,
            "train_loss": float(np.mean(losses)),
            "metrics": metrics,
        }
        trajectory.append(record)
        print(f"epoch {epoch:2d} | lr {model.lr:.6g} | loss {record['train_loss']:.4f} | "
              f"GAUC {metrics['GAUC']:.4f} | nDCG@5 {metrics['nDCG@5']:.4f} | primary {metrics['primary']:.4f}")
        if metrics["primary"] > best_primary + 1e-5:
            best_primary, stalled, best_record = metrics["primary"], 0, record
            best_state = (model.V.copy(), model.W.copy(), np.float32(model.b))
        else:
            stalled += 1
            if stalled >= args.patience:
                print(f"early stop at epoch {epoch}")
                break

    model.V, model.W, model.b = best_state
    summary = {
        "phase": "regularized_context_fm",
        "hypothesis": HYPOTHESIS,
        "selection_split": "validation",
        "test_data_used": False,
        "manual_interventions": 0,
        "fields": fields,
        "config": {
            "seed": args.seed, "embedding_dim": args.embedding_dim, "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay, "dropout": args.dropout, "lr_decay": args.lr_decay,
            "lr_step_epochs": args.lr_step_epochs, "batch_size": args.batch_size,
        },
        "best": best_record,
        "trajectory": trajectory,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("\nBest validation result")
    print(json.dumps(best_record, indent=2, sort_keys=True))


if __name__ == "__main__":
    started = time.time()
    main(parse_args())
    print(f"elapsed_seconds={time.time() - started:.1f}")
