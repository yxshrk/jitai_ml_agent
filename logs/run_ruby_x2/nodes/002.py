import argparse
import csv
import datetime
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from data.official.evaluate import evaluate as official_evaluate
except ImportError:
    official_evaluate = None
try:
    from harness.evaluate_provisional import evaluate as provisional_evaluate
except ImportError:
    provisional_evaluate = None


def metric_value(metrics, *names):
    for name in names:
        if name in metrics:
            return float(metrics[name])
    raise KeyError(names)


def run_evaluator(user_ids, labels, scores, fast_path):
    evaluator = official_evaluate if fast_path and official_evaluate is not None else provisional_evaluate
    if evaluator is None:
        evaluator = official_evaluate
    return evaluator(user_ids, labels.astype(int), scores)


def date_ordinals(values):
    out = np.zeros(len(values), dtype=np.float32)
    cache = {}
    for i, value in enumerate(values):
        text = str(value)
        if text.endswith('.0'):
            text = text[:-2]
        text = text.replace('-', '')
        if len(text) >= 8 and text[:8].isdigit():
            key = text[:8]
            if key not in cache:
                try:
                    cache[key] = datetime.date(
                        int(key[:4]), int(key[4:6]), int(key[6:8])
                    ).toordinal()
                except ValueError:
                    cache[key] = 0
            out[i] = cache[key]
    return out


def read_csv_rows(path):
    rows = []
    with open(path, 'r', newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append({
                'user_id': row['user_id'],
                'video_id': row['video_id'],
                'tab': row.get('tab', '0'),
                'duration_ms': float(row.get('duration_ms', 0.0) or 0.0),
                'date': row.get('date', '0'),
                'long_view': float(row['long_view'])
            })
    return rows


def load_csv_data(data_dir):
    train_rows = read_csv_rows(os.path.join(data_dir, 'train.csv'))
    val_rows = read_csv_rows(os.path.join(data_dir, 'val.csv'))
    durations = np.asarray([r['duration_ms'] for r in train_rows], dtype=np.float64)
    if len(durations):
        quantiles = np.quantile(durations, np.linspace(0.1, 0.9, 9))
    else:
        quantiles = np.zeros(9, dtype=np.float64)

    def vocabulary(key):
        values = sorted({r[key] for r in train_rows})
        return {value: i + 1 for i, value in enumerate(values)}

    user_map = vocabulary('user_id')
    video_map = vocabulary('video_id')
    tab_map = vocabulary('tab')
    field_dims = np.asarray([
        len(user_map) + 1,
        len(video_map) + 1,
        1,
        len(tab_map) + 1,
        10
    ], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1]))

    def encode(rows):
        x = np.empty((len(rows), 5), dtype=np.int64)
        for i, row in enumerate(rows):
            x[i, 0] = user_map.get(row['user_id'], 0)
            x[i, 1] = video_map.get(row['video_id'], 0)
            x[i, 2] = 0
            x[i, 3] = tab_map.get(row['tab'], 0)
            x[i, 4] = int(np.searchsorted(
                quantiles, row['duration_ms'], side='right'
            ))
        x += offsets
        return x

    train = {
        'X': encode(train_rows),
        'y': np.asarray([r['long_view'] for r in train_rows], dtype=np.float32),
        'user': np.asarray([r['user_id'] for r in train_rows]),
        'date': np.asarray([r['date'] for r in train_rows]),
        'field_dims': field_dims
    }
    val = {
        'X': encode(val_rows),
        'y': np.asarray([r['long_view'] for r in val_rows], dtype=np.float32),
        'user': np.asarray([r['user_id'] for r in val_rows]),
        'video': np.asarray([r['video_id'] for r in val_rows])
    }
    return train, val


def load_data(data_dir):
    train_npz = os.path.join(data_dir, 'train.npz')
    val_npz = os.path.join(data_dir, 'val.npz')
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        with np.load(train_npz) as archive:
            train = {key: archive[key] for key in archive.files}
        with np.load(val_npz) as archive:
            val = {key: archive[key] for key in archive.files}
        if 'video' not in val:
            val['video'] = np.zeros(len(val['y']), dtype=np.int64)
        return train, val, True
    train, val = load_csv_data(data_dir)
    return train, val, False


class DCNLite(torch.nn.Module):
    def __init__(self, total_dim, fields, embedding_dim, dropout):
        super().__init__()
        width = fields * embedding_dim
        self.emb = torch.nn.Embedding(total_dim, embedding_dim)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.gauge_bias = torch.nn.Parameter(torch.zeros(1))
        self.emb_drop = torch.nn.Dropout(dropout)
        self.cross_w = torch.nn.Parameter(torch.empty(width))
        self.cross_b = torch.nn.Parameter(torch.zeros(width))
        self.cross_head = torch.nn.Linear(width, 1, bias=False)
        self.deep = torch.nn.Sequential(
            torch.nn.Linear(width, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(64, 1)
        )
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)
        torch.nn.init.normal_(self.cross_w, std=0.01)
        torch.nn.init.zeros_(self.cross_head.weight)
        for layer in self.deep:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
        torch.nn.init.zeros_(self.deep[-1].weight)

    def forward(self, x):
        raw = self.emb(x)
        embedded = self.emb_drop(raw)
        summed = embedded.sum(dim=1)
        fm = 0.5 * (
            summed.square() - embedded.square().sum(dim=1)
        ).sum(dim=1)
        linear = self.lin(x).sum(dim=(1, 2)) + self.bias
        flat = embedded.flatten(1)
        cross = (
            flat
            + flat * torch.sum(flat * self.cross_w, dim=1, keepdim=True)
            + self.cross_b
        )
        return (
            linear
            + fm
            + self.cross_head(cross).squeeze(1)
            + self.deep(flat).squeeze(1)
        )


def make_user_groups(users):
    order = np.argsort(users, kind='mergesort')
    sorted_users = users[order]
    boundaries = np.flatnonzero(
        np.r_[True, sorted_users[1:] != sorted_users[:-1], True]
    )
    return [
        torch.from_numpy(order[left:right].astype(np.int64, copy=False))
        for left, right in zip(boundaries[:-1], boundaries[1:])
    ]


def complete_user_batches(user_groups, group_order, batch_size):
    batches = []
    current_groups = []
    current_size = 0

    def finish(groups):
        indices = torch.cat(groups)
        local_group = torch.cat([
            torch.full((len(group),), i, dtype=torch.long)
            for i, group in enumerate(groups)
        ])
        return indices, local_group

    for group_number in group_order.tolist():
        group = user_groups[group_number]
        if current_groups and current_size + len(group) > batch_size:
            batches.append(finish(current_groups))
            current_groups = []
            current_size = 0
        current_groups.append(group)
        current_size += len(group)
    if current_groups:
        batches.append(finish(current_groups))
    return batches


def centered_user_logits(logits, local_group, global_bias):
    group_count = int(local_group.max().item()) + 1
    sums = torch.zeros(group_count, dtype=logits.dtype, device=logits.device)
    counts = torch.zeros(group_count, dtype=logits.dtype, device=logits.device)
    sums.index_add_(0, local_group, logits)
    counts.index_add_(0, local_group, torch.ones_like(logits))
    means = sums / counts.clamp_min(1.0)
    return logits - means[local_group] + global_bias


def make_pairs(users, labels, seed):
    order = np.argsort(users, kind='mergesort')
    sorted_users = users[order]
    boundaries = np.flatnonzero(
        np.r_[True, sorted_users[1:] != sorted_users[:-1], True]
    )
    rng = np.random.default_rng(seed)
    positives = []
    negatives = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        indices = order[left:right]
        pos = indices[labels[indices] > 0.5]
        neg = indices[labels[indices] <= 0.5]
        if len(pos) and len(neg):
            positives.append(pos)
            negatives.append(neg[rng.integers(0, len(neg), size=len(pos))])
    if not positives:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    return (
        np.concatenate(positives).astype(np.int64),
        np.concatenate(negatives).astype(np.int64)
    )


def prediction(model, x_cpu, device):
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(x_cpu), 65536):
            xb = x_cpu[start:start + 65536].to(device, non_blocking=True)
            parts.append(model(xb).detach().cpu().numpy())
    return np.concatenate(parts).astype(np.float64)


def train_one(config, seed, epochs, arrays, evaluator, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

    (
        x_train,
        y_train,
        x_val,
        val_user,
        val_y,
        recency,
        pair_pos,
        pair_neg,
        total_dim,
        user_groups
    ) = arrays

    model = DCNLite(
        total_dim,
        x_train.shape[1],
        16,
        config['dropout']
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['step_size'],
        gamma=config['gamma']
    )
    pair_count = len(pair_pos)
    batch_size = 8192 if device.type == 'cuda' else 4096
    best_primary = -1.0
    best_scores = None
    best_metrics = None
    trace = []

    for epoch in range(epochs):
        group_permutation = torch.randperm(len(user_groups))
        midpoint = (len(user_groups) + 1) // 2
        halves = (
            group_permutation[:midpoint],
            group_permutation[midpoint:]
        )
        for half_index, group_order in enumerate(halves):
            model.train()
            losses = []
            batches = complete_user_batches(
                user_groups, group_order, batch_size
            )
            for indices, local_group_cpu in batches:
                xb = x_train[indices].to(device, non_blocking=True)
                yb = y_train[indices].to(device, non_blocking=True)
                wb = recency[indices].to(device, non_blocking=True)
                local_group = local_group_cpu.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                raw_logits = model(xb)
                point_logits = centered_user_logits(
                    raw_logits, local_group, model.gauge_bias
                )
                point_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    point_logits, yb, reduction='none'
                )
                point_loss = (
                    (point_loss * wb).sum() / wb.sum().clamp_min(1e-8)
                )

                if pair_count:
                    selected = torch.randint(0, pair_count, (len(indices),))
                    positive_indices = pair_pos[selected]
                    negative_indices = pair_neg[selected]
                    xp = x_train[positive_indices].to(device, non_blocking=True)
                    xn = x_train[negative_indices].to(device, non_blocking=True)
                    pair_weights = (
                        (recency[positive_indices] + recency[negative_indices])
                        .mul(0.5)
                        .to(device, non_blocking=True)
                    )
                    rank_loss = torch.nn.functional.softplus(
                        -(model(xp) - model(xn))
                    )
                    rank_loss = (
                        (rank_loss * pair_weights).sum()
                        / pair_weights.sum().clamp_min(1e-8)
                    )
                    loss = 0.5 * point_loss + 0.5 * rank_loss
                else:
                    loss = point_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))

            scores = prediction(model, x_val, device)
            metrics = evaluator(val_user, val_y, scores)
            primary = metric_value(metrics, 'primary')
            trace.append({
                'epoch': epoch + 0.5 * (half_index + 1),
                'train_loss': round(
                    float(np.mean(losses)) if losses else 0.0, 6
                ),
                'lr': float(optimizer.param_groups[0]['lr']),
                'val_gauc': round(
                    metric_value(metrics, 'GAUC', 'gauc'), 6
                ),
                'val_primary': round(primary, 6)
            })
            if primary > best_primary + 1e-8:
                best_primary = primary
                best_scores = scores.copy()
                best_metrics = metrics
        scheduler.step()

    return best_primary, best_scores, best_metrics, trace


def rank_average(score_list, users):
    result = np.zeros(len(users), dtype=np.float64)
    order = np.argsort(users, kind='mergesort')
    sorted_users = users[order]
    boundaries = np.flatnonzero(
        np.r_[True, sorted_users[1:] != sorted_users[:-1], True]
    )
    for scores in score_list:
        ranks = np.empty(len(users), dtype=np.float64)
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            indices = order[left:right]
            local_order = np.argsort(scores[indices], kind='mergesort')
            local_ranks = np.empty(len(indices), dtype=np.float64)
            local_ranks[local_order] = np.arange(len(indices), dtype=np.float64)
            ranks[indices] = local_ranks / max(1, len(indices) - 1)
        result += ranks
    return result / len(score_list)


def serializable_config(config):
    return {
        key: (int(value) if key == 'step_size' else float(value))
        for key, value in config.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=12)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    progress_path = os.path.join(args.out_dir, 'progress.log')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    train, val, fast_path = load_data(args.data_dir)
    evaluator = lambda users, labels, scores: run_evaluator(
        users, labels, scores, fast_path
    )
    x_train = torch.from_numpy(train['X'].astype(np.int64))
    y_train_np = train['y'].astype(np.float32)
    y_train = torch.from_numpy(y_train_np)
    x_val = torch.from_numpy(val['X'].astype(np.int64))
    train_users = np.asarray(train['user'])
    val_user = np.asarray(val['user'])
    val_y = val['y'].astype(np.float32)
    total_dim = int(np.asarray(train['field_dims']).sum())

    if 'date' in train:
        ordinals = date_ordinals(train['date'])
    else:
        ordinals = np.zeros(len(y_train_np), dtype=np.float32)
    max_ordinal = float(ordinals.max()) if len(ordinals) else 0.0
    ages = np.maximum(0.0, max_ordinal - ordinals)

    pair_pos_np, pair_neg_np = make_pairs(
        train_users, y_train_np, args.seed
    )
    pair_pos = torch.from_numpy(pair_pos_np)
    pair_neg = torch.from_numpy(pair_neg_np)
    user_groups = make_user_groups(train_users)

    smoke = int(os.environ.get('SMOKE_EPOCHS', '0') or 0)
    coarse_epochs = min(3, smoke) if smoke > 0 else 3
    refine_epochs = min(5, smoke) if smoke > 0 else 5
    final_epochs = min(args.epochs, smoke) if smoke > 0 else args.epochs
    coarse_count = 2 if smoke > 0 else 40
    refine_count = 1 if smoke > 0 else 18
    final_seed_count = 1 if smoke > 0 else 5

    rng = np.random.default_rng(args.seed + 991)
    coarse_configs = []
    for _ in range(coarse_count):
        coarse_configs.append({
            'dropout': float(rng.uniform(0.12, 0.43)),
            'weight_decay': float(10.0 ** rng.uniform(
                math.log10(2e-5), math.log10(4e-3)
            )),
            'lr': float(10.0 ** rng.uniform(
                math.log10(3.5e-4), math.log10(1.5e-3)
            )),
            'gamma': float(rng.choice(np.asarray([
                0.32, 0.45, 0.58, 0.72, 0.84
            ]))),
            'step_size': int(rng.choice(np.asarray([1, 1, 1, 2, 2, 3]))),
            'half_life': float(rng.choice(np.asarray([
                3.0, 4.5, 6.5, 9.0, 13.0, 17.0
            ])))
        })

    history = []
    best_primary = -1.0
    best_config = None

    def execute_probe(stage, probe_index, config, epochs):
        nonlocal best_primary, best_config
        weights = np.exp(
            -math.log(2.0) * ages / config['half_life']
        ).astype(np.float32)
        weights /= max(float(weights.mean()), 1e-8)
        recency = torch.from_numpy(weights)
        arrays = (
            x_train,
            y_train,
            x_val,
            val_user,
            val_y,
            recency,
            pair_pos,
            pair_neg,
            total_dim,
            user_groups
        )
        probe_seed = args.seed + probe_index + (
            0 if stage == 'coarse' else 10000
        )
        primary, _, metrics, trace = train_one(
            config,
            probe_seed,
            epochs,
            arrays,
            evaluator,
            device
        )
        record = {
            'stage': stage,
            'probe': probe_index,
            'config': serializable_config(config),
            'epochs': epochs,
            'gauc': metric_value(metrics, 'GAUC', 'gauc'),
            'ndcg5': metric_value(metrics, 'nDCG@5', 'ndcg5'),
            'primary': primary,
            'best_checkpoint': max(
                trace, key=lambda item: item['val_primary']
            )['epoch']
        }
        history.append(record)
        with open(progress_path, 'a') as fh:
            fh.write(json.dumps(record, sort_keys=True) + '\n')
        if primary > best_primary:
            best_primary = primary
            best_config = dict(config)

    for i, config in enumerate(coarse_configs):
        execute_probe('coarse', i, config, coarse_epochs)

    center = dict(best_config)
    refine_configs = []
    for _ in range(refine_count):
        refine_configs.append({
            'dropout': float(np.clip(
                center['dropout'] + rng.normal(0.0, 0.045),
                0.08,
                0.48
            )),
            'weight_decay': float(np.clip(
                center['weight_decay'] * math.exp(rng.normal(0.0, 0.48)),
                1e-5,
                8e-3
            )),
            'lr': float(np.clip(
                center['lr'] * math.exp(rng.normal(0.0, 0.25)),
                2.5e-4,
                2e-3
            )),
            'gamma': float(np.clip(
                center['gamma'] + rng.normal(0.0, 0.075),
                0.25,
                0.92
            )),
            'step_size': int(np.clip(
                center['step_size']
                + rng.choice(np.asarray([-1, 0, 0, 0, 1])),
                1,
                3
            )),
            'half_life': float(np.clip(
                center['half_life'] * math.exp(rng.normal(0.0, 0.22)),
                2.5,
                21.0
            ))
        })

    for i, config in enumerate(refine_configs):
        execute_probe('refine', i, config, refine_epochs)

    final_weights = np.exp(
        -math.log(2.0) * ages / best_config['half_life']
    ).astype(np.float32)
    final_weights /= max(float(final_weights.mean()), 1e-8)
    final_recency = torch.from_numpy(final_weights)
    final_arrays = (
        x_train,
        y_train,
        x_val,
        val_user,
        val_y,
        final_recency,
        pair_pos,
        pair_neg,
        total_dim,
        user_groups
    )

    final_scores = []
    final_runs = []
    for offset in range(final_seed_count):
        run_seed = args.seed + offset
        primary, scores, metrics, trace = train_one(
            best_config,
            run_seed,
            final_epochs,
            final_arrays,
            evaluator,
            device
        )
        final_scores.append(scores)
        final_runs.append({
            'seed': run_seed,
            'best_primary': primary,
            'gauc': metric_value(metrics, 'GAUC', 'gauc'),
            'ndcg5': metric_value(metrics, 'nDCG@5', 'ndcg5'),
            'trace': trace
        })

    if len(final_scores) == 1:
        blended_scores = final_scores[0]
    else:
        blended_scores = rank_average(final_scores, val_user)
    final_metrics = evaluator(val_user, val_y, blended_scores)
    output_metrics = {
        'gauc': metric_value(final_metrics, 'GAUC', 'gauc'),
        'ndcg5': metric_value(final_metrics, 'nDCG@5', 'ndcg5'),
        'primary': metric_value(final_metrics, 'primary'),
        'best_config': serializable_config(best_config),
        'history': history,
        'final_runs': final_runs
    }
    with open(os.path.join(args.out_dir, 'metrics.json'), 'w') as fh:
        json.dump(output_metrics, fh)

    video = val.get('video', np.zeros(len(val_y), dtype=np.int64))
    with open(os.path.join(args.out_dir, 'predictions.csv'), 'w') as fh:
        fh.write('row_id,user_id,video_id,score\n')
        for i, score in enumerate(blended_scores):
            fh.write(f'{i},{val_user[i]},{video[i]},{score:.9g}\n')


if __name__ == '__main__':
    main()
