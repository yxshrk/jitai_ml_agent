"""Within-user recency-weighted BPR candidate based on node_001.

Reads   data/train.csv, data/valid.csv, data/video_features_basic.csv
Writes  <out>/predictions.csv, <out>/metrics.json, and optionally
        <out>/predictions_extra.csv.
Model   The parent 5-field factorization machine trained with within-user BPR.
        Each sampled pair is weighted by the geometric mean of fixed seven-day
        exponential recency weights for its positive and negative impressions.
"""
import argparse
import csv
import json
import os
import time
from datetime import date

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


def parse_date(value):
    """Parse compact YYYYMMDD dates."""
    value = str(value)
    return date(
        int(value[0:4]),
        int(value[4:6]),
        int(value[6:8])
    )


def unique_within_user_rank_scores(users, raw_scores):
    """Convert raw scores to unique scores preserving stable within-user order.

    For each user, rows are stably sorted by descending raw score. Thus raw
    ties retain input order, exactly matching evaluate.py's stable nDCG sort.
    Distinct integer scores are then assigned within the user.
    """
    raw_scores = np.asarray(raw_scores)
    if not np.all(np.isfinite(raw_scores)):
        raise ValueError('Model produced NaN or infinite scores')

    groups = {}
    for index, user in enumerate(users):
        groups.setdefault(user, []).append(index)

    rank_scores = np.empty(len(raw_scores), dtype=np.int64)
    for indices in groups.values():
        ordered = sorted(indices, key=lambda i: -float(raw_scores[i]))
        group_size = len(ordered)
        for rank, index in enumerate(ordered):
            rank_scores[index] = group_size - rank

    return rank_scores


class FM:
    """Factorization machine optimized with recency-weighted BPR and Adam."""

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

    def step_pair(self, X_pos, X_neg, pair_weights):
        """Apply one Adam update for weighted -log sigmoid(s_pos-s_neg)."""
        pair_weights = np.asarray(pair_weights, dtype=np.float32)
        weight_sum = float(pair_weights.sum())
        if weight_sum <= 0.0:
            raise ValueError('Pair weights must have positive total weight')

        z_pos, E_pos, S_pos = self.logits(X_pos)
        z_neg, E_neg, S_neg = self.logits(X_neg)
        difference = z_pos - z_neg

        # d[-log(sigmoid(d))]/dd = -sigmoid(-d). Normalizing by the
        # batch's total pair weight recovers the parent's gradient exactly
        # when all pair weights are one.
        g = (
            -sigmoid(-difference) * pair_weights / weight_sum
        ).astype(np.float32)

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
        losses = np.logaddexp(0.0, -difference)
        return float(np.sum(pair_weights * losses) / weight_sum)

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
                row[0], row[1], row[2], str(int(score))
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
        [
            'user_id', 'video_id', 'tab', 'duration_ms',
            'long_view', 'date'
        ]
    )
    valid_rows = read_rows(
        f'{args.data_dir}/valid.csv',
        [
            'row_id', 'user_id', 'video_id', 'tab',
            'duration_ms', 'long_view'
        ]
    )

    # Fixed seven-day half-life measured backward from the last train date.
    reference_date = date(2022, 4, 21)
    unique_train_dates = {row[5] for row in train_rows}
    date_to_weight = {}
    for value in unique_train_dates:
        age_days = max(0, (reference_date - parse_date(value)).days)
        date_to_weight[value] = 0.5 ** (age_days / 7.0)

    train_recency_weights = np.fromiter(
        (date_to_weight[row[5]] for row in train_rows),
        dtype=np.float32,
        count=len(train_rows)
    )
    train_recency_weights /= train_recency_weights.mean()

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
        f'dim {total_dimension:,} BPR anchors {len(pair_positive_indices):,} '
        f'recency range {train_recency_weights.min():.3f}-'
        f'{train_recency_weights.max():.3f}',
        flush=True
    )

    # ---- train with weighted within-user BPR and valid early stopping ----
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

        # A pair's recency is the geometric mean of its two impression
        # weights, so neither side alone determines its temporal importance.
        sampled_pair_weights = np.sqrt(
            train_recency_weights[pair_positive_indices]
            * train_recency_weights[sampled_negative_indices]
        ).astype(np.float32)

        order = rng.permutation(len(pair_positive_indices))
        epoch_positive = pair_positive_indices[order]
        epoch_negative = sampled_negative_indices[order]
        epoch_pair_weights = sampled_pair_weights[order]

        losses = []
        for begin in range(0, len(epoch_positive), args.batch):
            end = begin + args.batch
            losses.append(model.step_pair(
                X_train[epoch_positive[begin:end]],
                X_train[epoch_negative[begin:end]],
                epoch_pair_weights[begin:end]
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
            f"epoch {epoch:2d} | weighted BPR loss {mean_loss:.4f} | "
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
    # Raw scores are retained for official metric computation. Prediction
    # files receive unique within-user rank scores to eliminate emitted ties.
    valid_raw_scores = model.predict(X_valid)
    result = evaluate(valid_users, y_valid, valid_raw_scores)
    valid_rank_scores = unique_within_user_rank_scores(
        valid_users,
        valid_raw_scores
    )
    write_predictions(
        f'{args.out_dir}/predictions.csv',
        valid_rows,
        valid_rank_scores
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
        extra_raw_scores = model.predict(X_extra)
        extra_users = [row[1] for row in extra_rows]
        extra_rank_scores = unique_within_user_rank_scores(
            extra_users,
            extra_raw_scores
        )
        write_predictions(
            f'{args.out_dir}/predictions_extra.csv',
            extra_rows,
            extra_rank_scores
        )

    print(
        f"done: valid primary {result['primary']:.4f} "
        f"in {time.time() - start_time:.0f}s",
        flush=True
    )


if __name__ == '__main__':
    main()
