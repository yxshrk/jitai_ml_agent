"""Within-user BPR candidate based on node_000.

Reads   data/train.csv, data/valid.csv, data/video_features_basic.csv
Writes  <out>/predictions.csv, <out>/metrics.json, and optionally
        <out>/predictions_extra.csv.
Model   The parent 5-field factorization machine trained with within-user BPR:
        for every eligible positive row, sample one negative row from the same
        user each epoch and minimize -log(sigmoid(score_pos-score_neg)).
"""
import argparse
import csv
import json
import os
import time

import numpy as np
from evaluate import evaluate

FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']


def read_rows(path, cols):
    """Read selected columns as strings, preserving file order."""
    with open(path, newline='') as fh:
        reader = csv.reader(fh)
        header = next(reader)
        indices = [header.index(c) for c in cols]
        return [[record[i] for i in indices] for record in reader]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class FM:
    """Factorization machine optimized with within-user BPR and Adam."""

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

    def step_pair(self, X_pos, X_neg):
        """Apply one Adam update for -mean(log sigmoid(s_pos-s_neg))."""
        batch_size = len(X_pos)

        z_pos, E_pos, S_pos = self.logits(X_pos)
        z_neg, E_neg, S_neg = self.logits(X_neg)
        difference = z_pos - z_neg

        # d[-log(sigmoid(d))]/dd = -sigmoid(-d)
        g = (-sigmoid(-difference) / batch_size).astype(np.float32)

        gV = np.zeros_like(self.V)
        gW = np.zeros_like(self.W)

        # Positive score receives g; negative score receives -g.
        np.add.at(gW, X_pos, g[:, None])
        np.add.at(gW, X_neg, -g[:, None])

        np.add.at(
            gV,
            X_pos,
            g[:, None, None] * (S_pos[:, None, :] - E_pos)
        )
        np.add.at(
            gV,
            X_neg,
            -g[:, None, None] * (S_neg[:, None, :] - E_neg)
        )

        # Keep the parent's regularization and optimizer unchanged.
        gV += self.l2 * self.V
        gW += self.l2 * self.W

        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for parameter, gradient, first, second in (
            (self.V, gV, self.mV, self.vV),
            (self.W, gW, self.mW, self.vW),
        ):
            first *= beta1
            first += (1.0 - beta1) * gradient
            second *= beta2
            second += (1.0 - beta2) * (gradient * gradient)

            first_hat = first / (1.0 - beta1 ** self.t)
            second_hat = second / (1.0 - beta2 ** self.t)
            parameter -= self.lr * first_hat / (np.sqrt(second_hat) + eps)

        # The global intercept cancels exactly in every within-user difference.
        return float(np.mean(np.logaddexp(0.0, -difference)))

    def predict(self, X, bs=200_000):
        return np.concatenate([
            self.logits(X[i:i + bs])[0]
            for i in range(0, len(X), bs)
        ])


def write_predictions(path, rows, scores):
    with open(path, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['row_id', 'user_id', 'video_id', 'score'])
        for row, score in zip(rows, scores):
            writer.writerow([
                row[0], row[1], row[2], f'{float(score):.9g}'
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
        if smoke_epochs > 0 else args.epochs
    )

    os.makedirs(args.out_dir, exist_ok=True)
    start_time = time.time()

    # ---- load ----
    video_to_author = dict(read_rows(
        f'{args.data_dir}/video_features_basic.csv',
        ['video_id', 'author_id']
    ))
    train_rows = read_rows(
        f'{args.data_dir}/train.csv',
        ['user_id', 'video_id', 'tab', 'duration_ms', 'long_view']
    )
    valid_rows = read_rows(
        f'{args.data_dir}/valid.csv',
        [
            'row_id', 'user_id', 'video_id', 'tab',
            'duration_ms', 'long_view'
        ]
    )

    # ---- encode exactly the parent's five categorical fields ----
    duration_edges = np.quantile(
        np.array([float(row[3]) for row in train_rows]),
        np.linspace(0, 1, 11)[1:-1]
    )

    def raw(user, video, tab, duration):
        return [
            user,
            video,
            video_to_author.get(video, 'UNK'),
            tab,
            str(int(np.searchsorted(duration_edges, float(duration))))
        ]

    vocabs = [dict() for _ in FIELDS]
    for row in train_rows:
        values = raw(row[0], row[1], row[2], row[3])
        for field, value in enumerate(values):
            if value not in vocabs[field]:
                vocabs[field][value] = len(vocabs[field])

    unknown = [len(vocab) for vocab in vocabs]
    dimensions = [len(vocab) + 1 for vocab in vocabs]
    offsets = np.cumsum([0] + dimensions[:-1]).astype(np.int32)
    total_dimension = int(sum(dimensions))

    def encode(rows, user_col, video_col, tab_col, duration_col):
        X = np.empty((len(rows), len(FIELDS)), dtype=np.int32)
        for n, row in enumerate(rows):
            values = raw(
                row[user_col],
                row[video_col],
                row[tab_col],
                row[duration_col]
            )
            for field, value in enumerate(values):
                X[n, field] = (
                    vocabs[field].get(value, unknown[field])
                    + offsets[field]
                )
        return X

    X_train = encode(train_rows, 0, 1, 2, 3)
    y_train = np.array(
        [1.0 if row[4] != '0' else 0.0 for row in train_rows],
        dtype=np.float32
    )

    X_valid = encode(valid_rows, 1, 2, 3, 4)
    y_valid = [1 if row[5] != '0' else 0 for row in valid_rows]
    valid_users = [row[1] for row in valid_rows]

    # ---- precompute vectorized same-user pair-sampling structures ----
    # The first encoded field is user_id and has offset zero.
    train_user_codes = X_train[:, 0]
    positive_indices = np.flatnonzero(y_train > 0.5)
    negative_indices = np.flatnonzero(y_train < 0.5)

    negative_order = np.argsort(
        train_user_codes[negative_indices], kind='stable'
    )
    negatives_by_user = negative_indices[negative_order]

    number_of_users = dimensions[0]
    negative_counts = np.bincount(
        train_user_codes[negative_indices],
        minlength=number_of_users
    ).astype(np.int64)

    negative_starts = np.empty(number_of_users, dtype=np.int64)
    negative_starts[0] = 0
    if number_of_users > 1:
        negative_starts[1:] = np.cumsum(negative_counts[:-1])

    positive_users = train_user_codes[positive_indices]
    eligible = negative_counts[positive_users] > 0
    pair_positive_indices = positive_indices[eligible]
    pair_users = train_user_codes[pair_positive_indices]
    pair_negative_starts = negative_starts[pair_users]
    pair_negative_counts = negative_counts[pair_users]

    print(
        f'loaded+encoded in {time.time() - start_time:.0f}s: '
        f'train {len(train_rows):,} valid {len(valid_rows):,} '
        f'dim {total_dimension:,} BPR anchors {len(pair_positive_indices):,}',
        flush=True
    )

    # ---- train with within-user BPR and early stopping on valid primary ----
    model = FM(
        total_dimension,
        k=args.k,
        lr=args.lr,
        seed=args.seed
    )
    rng = np.random.default_rng(args.seed)

    best = -1.0
    best_state = None
    bad_epochs = 0
    history = []

    for epoch in range(1, epochs + 1):
        # Sample one negative uniformly from the same user's negative rows for
        # every eligible positive anchor. All row-scale work is vectorized.
        sampled_offsets = (
            rng.random(len(pair_positive_indices)) * pair_negative_counts
        ).astype(np.int64)
        sampled_negative_indices = negatives_by_user[
            pair_negative_starts + sampled_offsets
        ]

        order = rng.permutation(len(pair_positive_indices))
        epoch_positive = pair_positive_indices[order]
        epoch_negative = sampled_negative_indices[order]

        losses = []
        for begin in range(0, len(epoch_positive), args.batch):
            end = begin + args.batch
            losses.append(model.step_pair(
                X_train[epoch_positive[begin:end]],
                X_train[epoch_negative[begin:end]]
            ))

        valid_scores = model.predict(X_valid)
        result = evaluate(valid_users, y_valid, valid_scores)
        mean_loss = float(np.mean(losses))

        history.append({
            'epoch': epoch,
            'train_loss': mean_loss,
            'val_gauc': result['GAUC'],
            'val_ndcg5': result['nDCG@5'],
            'val_primary': result['primary']
        })

        print(
            f"epoch {epoch:2d} | BPR loss {mean_loss:.4f} | "
            f"valid GAUC {result['GAUC']:.4f} "
            f"nDCG@5 {result['nDCG@5']:.4f} "
            f"primary {result['primary']:.4f}",
            flush=True
        )

        if result['primary'] > best + 1e-5:
            best = result['primary']
            bad_epochs = 0
            best_state = (
                model.V.copy(),
                model.W.copy(),
                np.float32(model.b)
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break

    model.V, model.W, model.b = best_state

    # ---- outputs ----
    valid_scores = model.predict(X_valid)
    result = evaluate(valid_users, y_valid, valid_scores)
    write_predictions(
        f'{args.out_dir}/predictions.csv',
        valid_rows,
        valid_scores
    )

    best_epoch = history[int(np.argmax([
        item['val_primary'] for item in history
    ]))]['epoch']

    with open(f'{args.out_dir}/metrics.json', 'w') as fh:
        json.dump({
            'gauc': result['GAUC'],
            'ndcg5': result['nDCG@5'],
            'primary': result['primary'],
            'best_epoch': int(best_epoch),
            'history': history,
            'seed': args.seed,
            'duration_s': time.time() - start_time
        }, fh, indent=1)

    if args.score_extra:
        extra_rows = read_rows(
            args.score_extra,
            [
                'row_id', 'user_id', 'video_id', 'tab',
                'duration_ms'
            ]
        )
        X_extra = encode(extra_rows, 1, 2, 3, 4)
        write_predictions(
            f'{args.out_dir}/predictions_extra.csv',
            extra_rows,
            model.predict(X_extra)
        )

    print(
        f"done: valid primary {result['primary']:.4f} "
        f"in {time.time() - start_time:.0f}s",
        flush=True
    )


if __name__ == '__main__':
    main()
