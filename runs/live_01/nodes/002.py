"""Recency-weighted factorization-machine ranking baseline.

Reads training, validation, and video feature files from --data-dir.
Writes validation predictions and metrics to --out-dir, plus optional
predictions for --score-extra.
"""
import argparse
import csv
import json
import os
import time
from datetime import date, datetime

import numpy as np
from evaluate import evaluate

FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']
RECENCY_REFERENCE_DATE = date(2022, 4, 21)
RECENCY_HALF_LIFE_DAYS = 7.0


def read_rows(path, cols):
    """Read selected columns from a CSV as strings, preserving file order."""
    with open(path, newline='') as fh:
        reader = csv.reader(fh)
        header = next(reader)
        indices = [header.index(col) for col in cols]
        return [[record[i] for i in indices] for record in reader]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def make_recency_weights(date_values):
    """Return mean-one exponential weights with a seven-day half-life."""
    unique_dates = set(date_values)
    by_date = {}
    for value in unique_dates:
        row_date = datetime.strptime(value, '%Y%m%d').date()
        age_days = max(0, (RECENCY_REFERENCE_DATE - row_date).days)
        by_date[value] = 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)

    weights = np.fromiter(
        (by_date[value] for value in date_values),
        dtype=np.float64,
        count=len(date_values),
    )
    weights /= weights.mean()
    return weights.astype(np.float32)


class FM:
    """Factorization machine trained with weighted logloss and Adam."""

    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr = lr
        self.l2 = l2

        self.mV = np.zeros_like(self.V)
        self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.t = 0

    def logits(self, X):
        E = self.V[X]
        S = E.sum(axis=1)
        inter = 0.5 * (
            (S ** 2).sum(axis=1) - (E ** 2).sum(axis=(1, 2))
        )
        return self.b + self.W[X].sum(axis=1) + inter, E, S

    def step(self, X, y, weights):
        z, E, S = self.logits(X)
        probabilities = sigmoid(z)
        weight_sum = max(float(np.sum(weights, dtype=np.float64)), 1e-12)

        g = (
            weights * (probabilities - y) / weight_sum
        ).astype(np.float32)

        gV = np.zeros_like(self.V)
        gW = np.zeros_like(self.W)
        np.add.at(gW, X, g[:, None])
        np.add.at(
            gV,
            X,
            g[:, None, None] * (S[:, None, :] - E),
        )

        gV += self.l2 * self.V
        gW += self.l2 * self.W

        self.t += 1
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        for parameter, gradient, first_moment, second_moment in (
            (self.V, gV, self.mV, self.vV),
            (self.W, gW, self.mW, self.vW),
        ):
            first_moment *= beta1
            first_moment += (1.0 - beta1) * gradient
            second_moment *= beta2
            second_moment += (1.0 - beta2) * (gradient * gradient)

            corrected_first = first_moment / (1.0 - beta1 ** self.t)
            corrected_second = second_moment / (1.0 - beta2 ** self.t)
            parameter -= self.lr * corrected_first / (
                np.sqrt(corrected_second) + epsilon
            )

        self.b -= self.lr * g.sum()

        row_losses = -(
            y * np.log(probabilities + 1e-9)
            + (1.0 - y) * np.log(1.0 - probabilities + 1e-9)
        )
        return float(
            np.sum(weights * row_losses, dtype=np.float64) / weight_sum
        )

    def predict(self, X, batch_size=200_000):
        return np.concatenate([
            self.logits(X[start:start + batch_size])[0]
            for start in range(0, len(X), batch_size)
        ])


def write_predictions(path, rows, scores):
    with open(path, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['row_id', 'user_id', 'video_id', 'score'])
        for row, score in zip(rows, scores):
            writer.writerow([
                row[0],
                row[1],
                row[2],
                f'{float(score):.9g}',
            ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--score-extra', default=None)
    parser.add_argument('--k', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch', type=int, default=8192)
    parser.add_argument('--patience', type=int, default=4)
    args = parser.parse_args()

    smoke_epochs = int(os.environ.get('SMOKE_EPOCHS', '0') or 0)
    epochs = (
        min(args.epochs, smoke_epochs)
        if smoke_epochs > 0
        else args.epochs
    )

    os.makedirs(args.out_dir, exist_ok=True)
    start_time = time.time()

    video_to_author = dict(read_rows(
        os.path.join(args.data_dir, 'video_features_basic.csv'),
        ['video_id', 'author_id'],
    ))
    train_rows = read_rows(
        os.path.join(args.data_dir, 'train.csv'),
        [
            'user_id',
            'video_id',
            'date',
            'tab',
            'duration_ms',
            'long_view',
        ],
    )
    valid_rows = read_rows(
        os.path.join(args.data_dir, 'valid.csv'),
        [
            'row_id',
            'user_id',
            'video_id',
            'tab',
            'duration_ms',
            'long_view',
        ],
    )

    duration_edges = np.quantile(
        np.array([float(row[4]) for row in train_rows]),
        np.linspace(0, 1, 11)[1:-1],
    )

    def raw_fields(user, video, tab, duration):
        bucket = int(np.searchsorted(duration_edges, float(duration)))
        return [
            user,
            video,
            video_to_author.get(video, 'UNK'),
            tab,
            str(bucket),
        ]

    vocabularies = [dict() for _ in FIELDS]
    for row in train_rows:
        values = raw_fields(row[0], row[1], row[3], row[4])
        for field_index, value in enumerate(values):
            vocabulary = vocabularies[field_index]
            if value not in vocabulary:
                vocabulary[value] = len(vocabulary)

    unknown_ids = [len(vocabulary) for vocabulary in vocabularies]
    field_dimensions = [
        len(vocabulary) + 1 for vocabulary in vocabularies
    ]
    offsets = np.cumsum(
        [0] + field_dimensions[:-1]
    ).astype(np.int32)
    total_dimension = int(sum(field_dimensions))

    def encode(rows, user_index, video_index, tab_index, duration_index):
        X = np.empty((len(rows), len(FIELDS)), dtype=np.int32)
        for row_number, row in enumerate(rows):
            values = raw_fields(
                row[user_index],
                row[video_index],
                row[tab_index],
                row[duration_index],
            )
            for field_index, value in enumerate(values):
                X[row_number, field_index] = (
                    vocabularies[field_index].get(
                        value,
                        unknown_ids[field_index],
                    )
                    + offsets[field_index]
                )
        return X

    X_train = encode(train_rows, 0, 1, 3, 4)
    y_train = np.array(
        [1.0 if row[5] != '0' else 0.0 for row in train_rows],
        dtype=np.float32,
    )
    train_weights = make_recency_weights(
        [row[2] for row in train_rows]
    )

    X_valid = encode(valid_rows, 1, 2, 3, 4)
    y_valid = [1 if row[5] != '0' else 0 for row in valid_rows]
    valid_users = [row[1] for row in valid_rows]

    print(
        f'loaded+encoded in {time.time() - start_time:.0f}s: '
        f'train {len(train_rows):,} valid {len(valid_rows):,} '
        f'dim {total_dimension:,}',
        flush=True,
    )

    model = FM(
        total_dimension,
        k=args.k,
        lr=args.lr,
        seed=args.seed,
    )
    rng = np.random.default_rng(args.seed)

    best_primary = -1.0
    best_state = None
    bad_epochs = 0
    history = []

    for epoch in range(1, epochs + 1):
        permutation = rng.permutation(len(y_train))
        losses = []

        for start in range(0, len(permutation), args.batch):
            batch_indices = permutation[start:start + args.batch]
            losses.append(model.step(
                X_train[batch_indices],
                y_train[batch_indices],
                train_weights[batch_indices],
            ))

        mean_loss = float(np.mean(losses))
        result = evaluate(
            valid_users,
            y_valid,
            model.predict(X_valid),
        )
        history.append({
            'epoch': epoch,
            'train_loss': mean_loss,
            'val_gauc': result['GAUC'],
            'val_ndcg5': result['nDCG@5'],
            'val_primary': result['primary'],
        })

        print(
            f"epoch {epoch:2d} | loss {mean_loss:.4f} | "
            f"valid GAUC {result['GAUC']:.4f} "
            f"nDCG@5 {result['nDCG@5']:.4f} "
            f"primary {result['primary']:.4f}",
            flush=True,
        )

        if result['primary'] > best_primary + 1e-5:
            best_primary = result['primary']
            bad_epochs = 0
            best_state = (
                model.V.copy(),
                model.W.copy(),
                np.float32(model.b),
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break

    model.V, model.W, model.b = best_state

    valid_scores = model.predict(X_valid)
    final_result = evaluate(valid_users, y_valid, valid_scores)

    write_predictions(
        os.path.join(args.out_dir, 'predictions.csv'),
        valid_rows,
        valid_scores,
    )

    best_epoch = int(
        np.argmax([entry['val_primary'] for entry in history]) + 1
    )
    metrics = {
        'gauc': final_result['GAUC'],
        'ndcg5': final_result['nDCG@5'],
        'primary': final_result['primary'],
        'best_epoch': best_epoch,
        'history': history,
        'seed': args.seed,
        'duration_s': time.time() - start_time,
    }
    with open(
        os.path.join(args.out_dir, 'metrics.json'),
        'w',
    ) as fh:
        json.dump(metrics, fh, indent=1)

    if args.score_extra:
        extra_rows = read_rows(
            args.score_extra,
            [
                'row_id',
                'user_id',
                'video_id',
                'tab',
                'duration_ms',
            ],
        )
        X_extra = encode(extra_rows, 1, 2, 3, 4)
        write_predictions(
            os.path.join(args.out_dir, 'predictions_extra.csv'),
            extra_rows,
            model.predict(X_extra),
        )

    print(
        f"done: valid primary {final_result['primary']:.4f} "
        f"in {time.time() - start_time:.0f}s"
    )


if __name__ == '__main__':
    main()
