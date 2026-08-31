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
        text = text.split('.')[0].replace('-', '').replace('/', '')
        try:
            mapped[i] = datetime.date(int(text[:4]), int(text[4:6]), int(text[6:8])).toordinal()
        except Exception:
            mapped[i] = float(i)
    return mapped[inverse]


def parse_hour_values(values):
    result = np.zeros(len(values), dtype=np.float64)
    for i, value in enumerate(values):
        text = str(value.decode() if isinstance(value, bytes) else value).strip()
        try:
            if ':' in text:
                parts = text.split(':')
                hour = int(parts[0])
                minute = int(parts[1])
            else:
                number = int(float(text))
                hour = number // 100
                minute = number % 100
            if hour < 0 or hour > 23 or minute < 0 or minute > 59:
                hour = 0
                minute = 0
            result[i] = hour + minute / 60.0
        except Exception:
            result[i] = 0.0
    return result


def load_npz(data_dir):
    from data.official.evaluate import evaluate
    tr = np.load(os.path.join(data_dir, 'train.npz'))
    va = np.load(os.path.join(data_dir, 'val.npz'))
    data = {
        'Xt': tr['X'].astype(np.int64),
        'yt': tr['y'].astype(np.float32),
        'ut': tr['user'],
        'date': tr['date'],
        'hourmin': tr['hourmin'],
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
                'hourmin': row['hourmin'],
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
            row['user_id'],
            row['video_id'],
            '0',
            row['tab'],
            str(int(np.searchsorted(edges, float(row['duration_ms']), side='right'))),
        ]
        return [offsets[j] + maps[j].get(vals[j], 0) for j in range(5)]

    Xt = np.asarray([encode_row(r) for r in train_rows], dtype=np.int64)
    yt = np.asarray([r['long_view'] for r in train_rows], dtype=np.float32)
    ut = np.asarray([r['user_id'] for r in train_rows])
    dates = np.asarray([r['date'] for r in train_rows])
    hourmin = np.asarray([r['hourmin'] for r in train_rows])
    Xv_list = []
    yv = []
    uv = []
    videos = []
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
        'hourmin': hourmin,
        'Xv': np.asarray(Xv_list, dtype=np.int64),
        'yv': np.asarray(yv, dtype=np.int64),
        'uv': np.asarray(uv),
        'video': np.asarray(videos),
        'total_dim': int(sum(dims)),
    }
    return data, evaluate


def compute_previous_gap_hours(users, dates, hourmin):
    day = parse_date_ordinals(dates)
    hour = parse_hour_values(hourmin)
    timestamp = day * 24.0 + hour
    users = np.asarray(users)
    row_index = np.arange(len(users), dtype=np.int64)
    order = np.lexsort((row_index, timestamp, users))
    gaps = np.full(len(users), np.inf, dtype=np.float64)
    if len(order) > 1:
        previous = order[:-1]
        current = order[1:]
        same_user = users[current] == users[previous]
        valid_current = current[same_user]
        valid_previous = previous[same_user]
        gaps[valid_current] = np.maximum(
            0.0, timestamp[valid_current] - timestamp[valid_previous])
    return gaps


def build_pair_pool(users, labels, seed):
    rng = np.random.default_rng(seed)
    order = np.argsort(users, kind='stable')
    sorted_users = np.asarray(users)[order]
    boundaries = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1, len(order)]
    positives = []
    negatives = []
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
        count = max(1, min(len(pos), len(neg)))
        return rng.choice(pos, count), rng.choice(neg, count)
    return np.concatenate(positives).astype(np.int64), np.concatenate(negatives).astype(np.int64)


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


def make_recency_weights(recency_age, gap_hours, config):
    short_cut = float(config.get('gap_short_hours', 1.0))
    long_cut = float(config.get('gap_long_hours', 24.0))
    multipliers = np.asarray([
        float(config.get('gap_decay_short', 1.0)),
        float(config.get('gap_decay_medium', 1.0)),
        float(config.get('gap_decay_long', 1.0)),
    ], dtype=np.float64)
    bucket = np.zeros(len(gap_hours), dtype=np.int64)
    bucket[gap_hours > short_cut] = 1
    bucket[gap_hours > long_cut] = 2
    decay_exponent = multipliers[bucket]
    half_life = float(config['half_life'])
    weights = np.exp2(-(recency_age / half_life) * decay_exponent).astype(np.float32)
    weights /= max(float(weights.mean()), 1e-8)
    return weights


def train_candidate(config, seed, epochs, train_indices, data, pair_pos, pair_neg,
                    recency_age, gap_hours, evaluate_fn, device, half_epoch=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    model = DCNLite(data['total_dim'], dropout=float(config['dropout'])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1.0e-3, weight_decay=float(config['weight_decay']))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda e: lr_multiplier(config['schedule'], e))
    rng = np.random.default_rng(seed + 1709)
    weights = make_recency_weights(recency_age, gap_hours, config)
    pair_weights = np.sqrt(weights[pair_pos] * weights[pair_neg]).astype(np.float32)
    batch_size = 8192
    best_primary = -1.0
    best_scores = None
    curve = []
    for epoch in range(epochs):
        model.train()
        shuffled = rng.permutation(train_indices)
        steps = int(math.ceil(len(shuffled) / batch_size))
        checkpoints = {steps - 1}
        if half_epoch and steps > 1:
            checkpoints.add(max(0, steps // 2 - 1))
        running = 0.0
        seen = 0
        for step, start in enumerate(range(0, len(shuffled), batch_size)):
            idx = shuffled[start:start + batch_size]
            pair_choice = rng.integers(0, len(pair_pos), size=len(idx))
            pidx = pair_pos[pair_choice]
            nidx = pair_neg[pair_choice]
            xb = torch.as_tensor(data['Xt'][idx], dtype=torch.long, device=device)
            yb = torch.as_tensor(data['yt'][idx], dtype=torch.float32, device=device)
            wb = torch.as_tensor(weights[idx], dtype=torch.float32, device=device)
            xp = torch.as_tensor(data['Xt'][pidx], dtype=torch.long, device=device)
            xn = torch.as_tensor(data['Xt'][nidx], dtype=torch.long, device=device)
            pw = torch.as_tensor(pair_weights[pair_choice], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            point_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, yb, reduction='none')
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
                    'epoch': float(epoch + (step + 1) / max(steps, 1)),
                    'train_loss': float(running / max(seen, 1)),
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


def core_config(dropout, weight_decay, schedule, half_life):
    return {
        'dropout': float(dropout),
        'weight_decay': float(weight_decay),
        'schedule': str(schedule),
        'half_life': float(half_life),
        'gap_short_hours': 1.0,
        'gap_long_hours': 24.0,
        'gap_decay_short': 1.0,
        'gap_decay_medium': 1.0,
        'gap_decay_long': 1.0,
    }


def with_gap(core, short_hours, long_hours, short_decay, medium_decay, long_decay):
    result = dict(core)
    result.update({
        'gap_short_hours': float(short_hours),
        'gap_long_hours': float(long_hours),
        'gap_decay_short': float(short_decay),
        'gap_decay_medium': float(medium_decay),
        'gap_decay_long': float(long_decay),
    })
    return result


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
    probe_epochs = min(args.epochs, smoke_cap) if smoke_cap is not None else args.epochs
    final_epochs = min(args.epochs, smoke_cap) if smoke_cap is not None else args.epochs
    coarse_count = 2 if smoke_cap is not None else 24
    refine_count = 1 if smoke_cap is not None else 12
    gap_seed_count = 1 if smoke_cap is not None else 4
    final_seed_count = 1 if smoke_cap is not None else 5

    date_ord = parse_date_ordinals(data['date'])
    recency_age = np.max(date_ord) - date_ord
    gap_hours = compute_previous_gap_hours(data['ut'], data['date'], data['hourmin'])
    pair_pos, pair_neg = build_pair_pool(data['ut'], data['yt'], args.seed + 91)
    all_indices = np.arange(len(data['yt']), dtype=np.int64)
    rng = np.random.default_rng(args.seed + 3101)
    coarse_size = max(1, int(0.68 * len(all_indices)))
    coarse_indices = np.sort(rng.choice(all_indices, size=coarse_size, replace=False))

    schedules = ['step1_055', 'step2_038', 'frontload', 'step3_022']
    half_lives = [3.5, 7.0, 14.0]
    anchors = [
        core_config(0.16, 4.0e-5, 'step1_055', 3.5),
        core_config(0.39, 2.5e-3, 'step3_022', 14.0),
        core_config(0.27, 3.2e-4, 'frontload', 7.0),
        core_config(0.33, 9.0e-4, 'step2_038', 7.0),
    ]
    coarse_configs = anchors[:coarse_count]
    while len(coarse_configs) < coarse_count:
        coarse_configs.append(core_config(
            rng.uniform(0.16, 0.39),
            math.exp(rng.uniform(math.log(4.0e-5), math.log(2.5e-3))),
            schedules[int(rng.integers(0, len(schedules)))],
            half_lives[int(rng.integers(0, len(half_lives)))],
        ))

    history = []
    progress_path = os.path.join(args.out_dir, 'progress.log')
    best_core = None
    best_coarse = -1.0

    with open(progress_path, 'a') as progress:
        for probe_id, config in enumerate(coarse_configs):
            primary, _, curve = train_candidate(
                config, args.seed + 1000 + probe_id, coarse_epochs, coarse_indices,
                data, pair_pos, pair_neg, recency_age, gap_hours, evaluate_fn, device)
            record = {
                'stage': 'inherited_core_coarse',
                'probe': probe_id,
                'config': config,
                'primary': float(primary),
                'epochs': coarse_epochs,
                'row_fraction': float(len(coarse_indices) / len(all_indices)),
                'checkpoints': curve,
            }
            history.append(record)
            progress.write(json.dumps({k: v for k, v in record.items() if k != 'checkpoints'}, sort_keys=True) + '\n')
            progress.flush()
            if primary > best_coarse:
                best_coarse = primary
                best_core = dict(config)

        refine_configs = [dict(best_core)]
        while len(refine_configs) < refine_count:
            refine_configs.append(core_config(
                np.clip(best_core['dropout'] + rng.normal(0.0, 0.035), 0.14, 0.42),
                np.clip(best_core['weight_decay'] * math.exp(rng.normal(0.0, 0.45)), 3.0e-5, 3.0e-3),
                best_core['schedule'] if rng.random() < 0.6 else schedules[int(rng.integers(0, len(schedules)))],
                best_core['half_life'] if rng.random() < 0.6 else half_lives[int(rng.integers(0, len(half_lives)))],
            ))

        best_refine = -1.0
        fixed_core = dict(best_core)
        for probe_id, config in enumerate(refine_configs):
            primary, _, curve = train_candidate(
                config, args.seed + 5000 + probe_id, refine_epochs, all_indices,
                data, pair_pos, pair_neg, recency_age, gap_hours, evaluate_fn, device)
            record = {
                'stage': 'inherited_core_refine',
                'probe': probe_id,
                'config': config,
                'primary': float(primary),
                'epochs': refine_epochs,
                'row_fraction': 1.0,
                'checkpoints': curve,
            }
            history.append(record)
            progress.write(json.dumps({k: v for k, v in record.items() if k != 'checkpoints'}, sort_keys=True) + '\n')
            progress.flush()
            if primary > best_refine:
                best_refine = primary
                fixed_core = dict(config)

        gap_specs = [
            (1.0, 24.0, 1.00, 1.00, 1.00),
            (1.0, 24.0, 0.80, 1.00, 1.40),
            (1.0, 24.0, 0.67, 1.00, 2.00),
            (1.0, 24.0, 0.50, 1.00, 2.50),
            (2.0, 24.0, 0.67, 1.00, 2.00),
            (4.0, 24.0, 0.67, 1.00, 2.00),
            (1.0, 48.0, 0.67, 1.00, 2.00),
            (2.0, 48.0, 0.67, 1.00, 2.00),
            (1.0, 24.0, 0.67, 0.85, 2.00),
            (1.0, 24.0, 0.67, 1.15, 2.00),
            (2.0, 48.0, 0.80, 1.00, 1.40),
            (4.0, 48.0, 0.50, 0.85, 2.50),
        ]
        if smoke_cap is not None:
            gap_specs = gap_specs[:2]

        gap_summaries = []
        selected_config = None
        selected_mean = -1.0
        for variant_id, spec in enumerate(gap_specs):
            config = with_gap(fixed_core, *spec)
            variant_scores = []
            for seed_offset in range(gap_seed_count):
                probe_seed = args.seed + seed_offset
                primary, _, curve = train_candidate(
                    config, probe_seed, probe_epochs, all_indices, data,
                    pair_pos, pair_neg, recency_age, gap_hours, evaluate_fn, device,
                    half_epoch=True)
                variant_scores.append(float(primary))
                record = {
                    'stage': 'gap_conditioned_probe',
                    'variant': variant_id,
                    'seed': probe_seed,
                    'config': config,
                    'primary': float(primary),
                    'epochs': probe_epochs,
                    'row_fraction': 1.0,
                    'checkpoints': curve,
                }
                history.append(record)
                progress.write(json.dumps({k: v for k, v in record.items() if k != 'checkpoints'}, sort_keys=True) + '\n')
                progress.flush()
            mean_primary = float(np.mean(variant_scores))
            std_primary = float(np.std(variant_scores))
            summary = {
                'variant': variant_id,
                'config': config,
                'mean_primary': mean_primary,
                'std_primary': std_primary,
                'seed_primaries': variant_scores,
            }
            gap_summaries.append(summary)
            if mean_primary > selected_mean:
                selected_mean = mean_primary
                selected_config = dict(config)

        final_rank_scores = []
        final_records = []
        for seed_offset in range(final_seed_count):
            final_seed = args.seed + seed_offset
            primary, scores, curve = train_candidate(
                selected_config, final_seed, final_epochs, all_indices, data,
                pair_pos, pair_neg, recency_age, gap_hours, evaluate_fn, device,
                half_epoch=True)
            final_rank_scores.append(normalized_rank(scores))
            final_record = {
                'stage': 'final',
                'seed': final_seed,
                'config': selected_config,
                'best_primary': float(primary),
                'epochs': final_epochs,
                'checkpoints': curve,
            }
            final_records.append(final_record)
            progress.write(json.dumps({
                'stage': 'final',
                'seed': final_seed,
                'config': selected_config,
                'primary': float(primary),
            }, sort_keys=True) + '\n')
            progress.flush()

    best_scores = np.mean(np.stack(final_rank_scores, axis=0), axis=0)
    result = evaluate_fn(data['uv'], data['yv'], best_scores)
    gauc, ndcg5, primary = metric_values(result)
    metrics = {
        'gauc': gauc,
        'ndcg5': ndcg5,
        'primary': primary,
        'diagnosis': 'parent learning curve unavailable; test heterogeneous temporal data shift',
        'fixed_parent_core': fixed_core,
        'selected_config': selected_config,
        'gap_variant_summaries': gap_summaries,
        'history': history,
        'final_history': final_records,
        'ensemble_seeds': [args.seed + i for i in range(final_seed_count)],
    }
    with open(os.path.join(args.out_dir, 'metrics.json'), 'w') as fh:
        json.dump(metrics, fh)
    with open(os.path.join(args.out_dir, 'predictions.csv'), 'w') as fh:
        fh.write('row_id,user_id,video_id,score\n')
        for i, score in enumerate(best_scores):
            fh.write(f'{i},{data["uv"][i]},{data["video"][i]},{score:.9g}\n')


if __name__ == '__main__':
    main()
