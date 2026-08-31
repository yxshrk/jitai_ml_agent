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


class DCNLite(torch.nn.Module):
    def __init__(self, total_dim, fields=5, k=16, hidden=128, dropout=0.25):
        super().__init__()
        self.fields = fields
        self.k = k
        width = fields * k
        self.emb = torch.nn.Embedding(total_dim, k)
        self.linear = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.emb_dropout = torch.nn.Dropout(dropout)
        self.cross1 = torch.nn.Linear(width, 1)
        self.cross2 = torch.nn.Linear(width, 1)
        self.cross_out = torch.nn.Linear(width, 1)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(width, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, hidden // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden // 2, 1),
        )
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.linear.weight)
        for layer in (self.cross1, self.cross2, self.cross_out):
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)

    def forward(self, x):
        e = self.emb_dropout(self.emb(x))
        x0 = e.reshape(e.shape[0], -1)
        x1 = x0 * self.cross1(x0) + x0
        x2 = x0 * self.cross2(x1) + x1
        linear_term = self.linear(x).sum((1, 2))
        return self.bias + linear_term + self.cross_out(x2).squeeze(1) + self.mlp(x0).squeeze(1)


def parse_date_ordinals(values):
    values = np.asarray(values)
    unique, inverse = np.unique(values, return_inverse=True)
    mapped = np.empty(len(unique), dtype=np.float64)
    for i, value in enumerate(unique):
        text = str(value.decode() if isinstance(value, bytes) else value)
        text = text.split('.')[0].replace('-', '')
        try:
            mapped[i] = datetime.date(int(text[:4]), int(text[4:6]), int(text[6:8])).toordinal()
        except Exception:
            mapped[i] = float(i)
    return mapped[inverse]


def load_npz(data_dir):
    from data.official.evaluate import evaluate
    tr = np.load(os.path.join(data_dir, 'train.npz'))
    va = np.load(os.path.join(data_dir, 'val.npz'))
    data = {
        'Xt': tr['X'].astype(np.int64),
        'yt': tr['y'].astype(np.float32),
        'ut': tr['user'],
        'date': tr['date'],
        'Xv': va['X'].astype(np.int64),
        'yv': va['y'].astype(np.int64),
        'uv': va['user'],
        'video': np.zeros(len(va['y']), dtype=np.int64),
        'total_dim': int(tr['field_dims'].sum()),
    }
    return data, evaluate


def load_csv_data(data_dir):
    from harness.evaluate_provisional import evaluate
    train_rows = []
    with open(os.path.join(data_dir, 'train.csv'), newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            train_rows.append({
                'user_id': row['user_id'],
                'video_id': row['video_id'],
                'tab': row['tab'],
                'duration_ms': float(row['duration_ms']),
                'date': row['date'],
                'long_view': float(row['long_view']),
            })
    durations = np.asarray([r['duration_ms'] for r in train_rows], dtype=np.float64)
    edges = np.unique(np.quantile(durations, np.linspace(0.1, 0.9, 9)))
    fields = [
        [r['user_id'] for r in train_rows],
        [r['video_id'] for r in train_rows],
        ['0' for _ in train_rows],
        [r['tab'] for r in train_rows],
        [str(int(np.searchsorted(edges, r['duration_ms'], side='right'))) for r in train_rows],
    ]
    maps = []
    dims = []
    for values in fields:
        mapping = {v: i + 1 for i, v in enumerate(sorted(set(values)))}
        maps.append(mapping)
        dims.append(len(mapping) + 1)
    offsets = np.cumsum([0] + dims[:-1]).astype(np.int64)

    def encode_row(row):
        vals = [
            row['user_id'], row['video_id'], '0', row['tab'],
            str(int(np.searchsorted(edges, float(row['duration_ms']), side='right'))),
        ]
        return [offsets[j] + maps[j].get(vals[j], 0) for j in range(5)]

    Xt = np.asarray([encode_row(r) for r in train_rows], dtype=np.int64)
    yt = np.asarray([r['long_view'] for r in train_rows], dtype=np.float32)
    ut = np.asarray([r['user_id'] for r in train_rows])
    dates = np.asarray([r['date'] for r in train_rows])
    Xv_list, yv, uv, videos = [], [], [], []
    with open(os.path.join(data_dir, 'val.csv'), newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            feature_row = {
                'user_id': row['user_id'],
                'video_id': row['video_id'],
                'tab': row['tab'],
                'duration_ms': float(row['duration_ms']),
            }
            Xv_list.append(encode_row(feature_row))
            yv.append(int(float(row['long_view'])))
            uv.append(row['user_id'])
            videos.append(row['video_id'])
    data = {
        'Xt': Xt,
        'yt': yt,
        'ut': ut,
        'date': dates,
        'Xv': np.asarray(Xv_list, dtype=np.int64),
        'yv': np.asarray(yv, dtype=np.int64),
        'uv': np.asarray(uv),
        'video': np.asarray(videos),
        'total_dim': int(sum(dims)),
    }
    return data, evaluate


def build_pair_pool(users, labels, seed):
    rng = np.random.default_rng(seed)
    order = np.argsort(users, kind='stable')
    sorted_users = np.asarray(users)[order]
    boundaries = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1, len(order)]
    positives, negatives = [], []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        group = order[left:right]
        pos = group[labels[group] > 0.5]
        neg = group[labels[group] <= 0.5]
        if len(pos) == 0 or len(neg) == 0:
            continue
        count = min(64, max(len(pos), len(neg)))
        positives.append(rng.choice(pos, size=count, replace=len(pos) < count))
        negatives.append(rng.choice(neg, size=count, replace=len(neg) < count))
    if not positives:
        pos = np.flatnonzero(labels > 0.5)
        neg = np.flatnonzero(labels <= 0.5)
        count = min(len(labels), max(1, min(len(pos), len(neg))))
        return rng.choice(pos, count), rng.choice(neg, count)
    return np.concatenate(positives).astype(np.int64), np.concatenate(negatives).astype(np.int64)


def build_user_groups(users, indices):
    indices = np.asarray(indices, dtype=np.int64)
    local_users = np.asarray(users)[indices]
    order = np.argsort(local_users, kind='stable')
    sorted_indices = indices[order]
    sorted_users = local_users[order]
    boundaries = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1, len(order)]
    return [sorted_indices[left:right] for left, right in zip(boundaries[:-1], boundaries[1:])]


def complete_slate_batches(groups, rng, max_rows):
    group_order = rng.permutation(len(groups))
    batch_groups = []
    batch_rows = 0
    for group_number in group_order:
        group = groups[int(group_number)]
        if batch_groups and batch_rows + len(group) > max_rows:
            idx = np.concatenate(batch_groups).astype(np.int64, copy=False)
            gid = np.concatenate([
                np.full(len(g), j, dtype=np.int64) for j, g in enumerate(batch_groups)
            ])
            yield idx, gid
            batch_groups = []
            batch_rows = 0
        batch_groups.append(group)
        batch_rows += len(group)
        if batch_rows >= max_rows:
            idx = np.concatenate(batch_groups).astype(np.int64, copy=False)
            gid = np.concatenate([
                np.full(len(g), j, dtype=np.int64) for j, g in enumerate(batch_groups)
            ])
            yield idx, gid
            batch_groups = []
            batch_rows = 0
    if batch_groups:
        idx = np.concatenate(batch_groups).astype(np.int64, copy=False)
        gid = np.concatenate([
            np.full(len(g), j, dtype=np.int64) for j, g in enumerate(batch_groups)
        ])
        yield idx, gid


def gauge_fixed_logits(raw_logits, group_ids):
    group_count = int(group_ids[-1].item()) + 1
    sums = torch.zeros(group_count, dtype=raw_logits.dtype, device=raw_logits.device)
    counts = torch.zeros(group_count, dtype=raw_logits.dtype, device=raw_logits.device)
    sums.scatter_add_(0, group_ids, raw_logits)
    counts.scatter_add_(0, group_ids, torch.ones_like(raw_logits))
    means = sums / counts.clamp_min(1.0)
    return raw_logits - means[group_ids]


def metric_values(result):
    return (
        float(result.get('GAUC', result.get('gauc'))),
        float(result.get('nDCG@5', result.get('ndcg5'))),
        float(result['primary']),
    )


def lr_multiplier(name, epoch):
    if name == 'step1_055':
        return 0.55 ** epoch
    if name == 'step2_038':
        return 0.38 ** (epoch // 2)
    if name == 'frontload':
        return (0.68 ** min(epoch, 2)) * (0.43 ** max(0, epoch - 2))
    return 0.22 ** (epoch // 3)


def score_model(model, Xv, device):
    model.eval()
    pieces = []
    with torch.no_grad():
        for start in range(0, len(Xv), 65536):
            xb = torch.as_tensor(Xv[start:start + 65536], dtype=torch.long, device=device)
            pieces.append(model(xb).detach().cpu().numpy())
    return np.concatenate(pieces)


def train_candidate(config, seed, epochs, train_indices, data, pair_pos, pair_neg,
                    recency_age, evaluate_fn, device, half_epoch=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    model = DCNLite(data['total_dim'], dropout=float(config['dropout'])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1.0e-3, weight_decay=float(config['weight_decay'])
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda e: lr_multiplier(config['schedule'], e)
    )
    rng = np.random.default_rng(seed + 1709)
    half_life = float(config['half_life'])
    weights = np.exp2(-recency_age / half_life).astype(np.float32)
    weights /= max(float(weights.mean()), 1e-8)
    pair_weights = np.sqrt(weights[pair_pos] * weights[pair_neg]).astype(np.float32)
    user_groups = build_user_groups(data['ut'], train_indices)
    bs = 8192
    best_primary = -1.0
    best_scores = None
    curve = []
    for epoch in range(epochs):
        model.train()
        batches = list(complete_slate_batches(user_groups, rng, bs))
        steps = len(batches)
        checkpoints = {steps - 1}
        if half_epoch and steps > 1:
            checkpoints.add(max(0, steps // 2 - 1))
        running = 0.0
        seen = 0
        for step, (idx, group_ids_np) in enumerate(batches):
            pair_choice = rng.integers(0, len(pair_pos), size=len(idx))
            pidx = pair_pos[pair_choice]
            nidx = pair_neg[pair_choice]
            xb = torch.as_tensor(data['Xt'][idx], dtype=torch.long, device=device)
            yb = torch.as_tensor(data['yt'][idx], dtype=torch.float32, device=device)
            wb = torch.as_tensor(weights[idx], dtype=torch.float32, device=device)
            group_ids = torch.as_tensor(group_ids_np, dtype=torch.long, device=device)
            xp = torch.as_tensor(data['Xt'][pidx], dtype=torch.long, device=device)
            xn = torch.as_tensor(data['Xt'][nidx], dtype=torch.long, device=device)
            pw = torch.as_tensor(pair_weights[pair_choice], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            raw_logits = model(xb)
            centered_logits = gauge_fixed_logits(raw_logits, group_ids) + model.bias
            point_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                centered_logits, yb, reduction='none'
            )
            point_loss = (point_loss * wb).sum() / wb.sum().clamp_min(1e-8)
            pair_loss = torch.nn.functional.softplus(-(model(xp) - model(xn)))
            pair_loss = (pair_loss * pw).sum() / pw.sum().clamp_min(1e-8)
            loss = 0.5 * point_loss + 0.5 * pair_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(loss.detach().cpu()) * len(idx)
            seen += len(idx)
            if step in checkpoints:
                scores = score_model(model, data['Xv'], device)
                result = evaluate_fn(data['uv'], data['yv'], scores)
                gauc, ndcg5, primary = metric_values(result)
                curve.append({
                    'epoch': epoch + (step + 1) / max(steps, 1),
                    'train_loss': running / max(seen, 1),
                    'gauc': gauc,
                    'ndcg5': ndcg5,
                    'primary': primary,
                })
                if primary > best_primary:
                    best_primary = primary
                    best_scores = scores.copy()
                model.train()
        scheduler.step()
    del model, optimizer, scheduler
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_primary, best_scores, curve


def config_dict(dropout, weight_decay, schedule, half_life):
    return {
        'dropout': float(dropout),
        'weight_decay': float(weight_decay),
        'schedule': str(schedule),
        'half_life': float(half_life),
    }


def normalized_rank(scores):
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = (np.arange(len(scores), dtype=np.float64) + 0.5) / max(len(scores), 1)
    return ranks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=12)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    fast_path = (
        os.path.exists(os.path.join(args.data_dir, 'train.npz')) and
        os.path.exists(os.path.join(args.data_dir, 'val.npz'))
    )
    if fast_path:
        data, evaluate_fn = load_npz(args.data_dir)
    else:
        data, evaluate_fn = load_csv_data(args.data_dir)
    smoke_raw = os.environ.get('SMOKE_EPOCHS')
    smoke_cap = int(smoke_raw) if smoke_raw is not None else None
    coarse_epochs = min(3, smoke_cap) if smoke_cap is not None else 3
    refine_epochs = min(5, smoke_cap) if smoke_cap is not None else 5
    final_epochs = min(args.epochs, smoke_cap) if smoke_cap is not None else args.epochs
    coarse_count = 2 if smoke_cap is not None else 72
    refine_count = 1 if smoke_cap is not None else 32
    final_seed_count = 1 if smoke_cap is not None else 5
    date_ord = parse_date_ordinals(data['date'])
    recency_age = np.max(date_ord) - date_ord
    pair_pos, pair_neg = build_pair_pool(data['ut'], data['yt'], args.seed + 91)
    rng = np.random.default_rng(args.seed + 3101)
    all_indices = np.arange(len(data['yt']), dtype=np.int64)
    coarse_size = max(1, int(0.68 * len(all_indices)))
    coarse_indices = np.sort(rng.choice(all_indices, size=coarse_size, replace=False))
    schedules = ['step1_055', 'step2_038', 'frontload', 'step3_022']
    half_lives = [3.5, 7.0, 14.0]
    coarse_configs = []
    anchors = [
        config_dict(0.16, 4.0e-5, 'step1_055', 3.5),
        config_dict(0.39, 2.5e-3, 'step3_022', 14.0),
        config_dict(0.27, 3.2e-4, 'frontload', 7.0),
        config_dict(0.33, 9.0e-4, 'step2_038', 7.0),
    ]
    for cfg in anchors[:coarse_count]:
        coarse_configs.append(cfg)
    while len(coarse_configs) < coarse_count:
        coarse_configs.append(config_dict(
            rng.uniform(0.16, 0.39),
            math.exp(rng.uniform(math.log(4.0e-5), math.log(2.5e-3))),
            schedules[int(rng.integers(0, len(schedules)))],
            half_lives[int(rng.integers(0, len(half_lives)))],
        ))
    history = []
    progress_path = os.path.join(args.out_dir, 'progress.log')
    best_config = None
    best_probe = -1.0
    with open(progress_path, 'a') as progress:
        for probe_id, cfg in enumerate(coarse_configs):
            primary, _, curve = train_candidate(
                cfg, args.seed + 1000 + probe_id, coarse_epochs, coarse_indices,
                data, pair_pos, pair_neg, recency_age, evaluate_fn, device
            )
            record = {
                'stage': 'coarse', 'probe': probe_id, 'config': cfg,
                'primary': float(primary), 'epochs': coarse_epochs,
                'row_fraction': float(len(coarse_indices) / len(all_indices)),
                'objective': 'gauge_fixed_bce_plus_bpr',
            }
            history.append(record)
            progress.write(json.dumps(record, sort_keys=True) + '\n')
            progress.flush()
            if primary > best_probe:
                best_probe = primary
                best_config = dict(cfg)
        drop_grid = np.clip(np.asarray([
            best_config['dropout'] - 0.05,
            best_config['dropout'] - 0.025,
            best_config['dropout'],
            best_config['dropout'] + 0.025,
            best_config['dropout'] + 0.05,
        ]), 0.14, 0.42)
        wd_grid = np.clip(
            best_config['weight_decay'] * np.asarray([0.5, 0.72, 1.0, 1.4, 2.0]),
            3.0e-5, 3.0e-3
        )
        half_order = sorted(
            half_lives, key=lambda x: abs(math.log(x / best_config['half_life']))
        )
        schedule_order = [best_config['schedule']] + [
            s for s in schedules if s != best_config['schedule']
        ]
        dense = []
        for dropout in drop_grid:
            for weight_decay in wd_grid:
                for half_life in half_order:
                    for schedule in schedule_order:
                        dense.append(config_dict(dropout, weight_decay, schedule, half_life))
        winner_key = json.dumps(best_config, sort_keys=True)
        dense.sort(key=lambda c: (
            json.dumps(c, sort_keys=True) != winner_key,
            abs(c['dropout'] - best_config['dropout']) +
            abs(math.log(c['weight_decay'] / best_config['weight_decay'])) +
            0.12 * abs(math.log(c['half_life'] / best_config['half_life'])) +
            (0.08 if c['schedule'] != best_config['schedule'] else 0.0),
        ))
        if len(dense) > 1:
            head = dense[:1]
            pool = dense[1:]
            order = rng.permutation(len(pool))
            refine_configs = head + [pool[i] for i in order[:max(0, refine_count - 1)]]
        else:
            refine_configs = dense
        refined_best = -1.0
        refined_config = dict(best_config)
        for probe_id, cfg in enumerate(refine_configs):
            primary, _, curve = train_candidate(
                cfg, args.seed + 5000 + probe_id, refine_epochs, all_indices,
                data, pair_pos, pair_neg, recency_age, evaluate_fn, device
            )
            record = {
                'stage': 'refine', 'probe': probe_id, 'config': cfg,
                'primary': float(primary), 'epochs': refine_epochs,
                'row_fraction': 1.0,
                'objective': 'gauge_fixed_bce_plus_bpr',
            }
            history.append(record)
            progress.write(json.dumps(record, sort_keys=True) + '\n')
            progress.flush()
            if primary > refined_best:
                refined_best = primary
                refined_config = dict(cfg)
        final_rank_scores = []
        final_records = []
        for seed_offset in range(final_seed_count):
            final_seed = args.seed + seed_offset
            primary, scores, curve = train_candidate(
                refined_config, final_seed, final_epochs, all_indices,
                data, pair_pos, pair_neg, recency_age, evaluate_fn, device,
                half_epoch=True
            )
            final_rank_scores.append(normalized_rank(scores))
            record = {
                'stage': 'final',
                'seed': final_seed,
                'config': refined_config,
                'best_primary': float(primary),
                'epochs': final_epochs,
                'objective': 'gauge_fixed_bce_plus_bpr',
                'checkpoints': curve,
            }
            final_records.append(record)
            progress.write(json.dumps({
                'stage': 'final', 'seed': final_seed,
                'config': refined_config, 'primary': float(primary),
                'objective': 'gauge_fixed_bce_plus_bpr',
            }, sort_keys=True) + '\n')
            progress.flush()
    best_scores = np.mean(np.stack(final_rank_scores, axis=0), axis=0)
    result = evaluate_fn(data['uv'], data['yv'], best_scores)
    gauc, ndcg5, primary = metric_values(result)
    metrics = {
        'gauc': gauc,
        'ndcg5': ndcg5,
        'primary': primary,
        'method': 'gauge-fixed-bce',
        'selected_config': refined_config,
        'history': history,
        'final_history': final_records,
        'ensemble_seeds': [args.seed + i for i in range(final_seed_count)],
    }
    with open(os.path.join(args.out_dir, 'metrics.json'), 'w') as fh:
        json.dump(metrics, fh)
    with open(os.path.join(args.out_dir, 'predictions.csv'), 'w') as fh:
        fh.write('row_id,user_id,video_id,score\n')
        for i, score in enumerate(best_scores):
            fh.write('{},{},{},{:.9g}\n'.format(i, data['uv'][i], data['video'][i], score))


if __name__ == '__main__':
    main()
