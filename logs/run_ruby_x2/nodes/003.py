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


def run_evaluator(user, labels, scores, fast_path):
    evaluator = official_evaluate if fast_path and official_evaluate is not None else provisional_evaluate
    if evaluator is None:
        evaluator = official_evaluate
    return evaluator(user, labels.astype(int), scores)


def parse_date(value):
    s = str(value)
    if s.endswith('.0'):
        s = s[:-2]
    s = s.replace('-', '')
    if len(s) >= 8 and s[:8].isdigit():
        try:
            return datetime.date(int(s[:4]), int(s[4:6]), int(s[6:8])).toordinal()
        except ValueError:
            return 0
    return 0


def date_ordinals(values):
    cache = {}
    out = np.zeros(len(values), dtype=np.int64)
    for i, value in enumerate(values):
        key = str(value)
        if key not in cache:
            cache[key] = parse_date(value)
        out[i] = cache[key]
    return out


def hour_minutes(values):
    out = np.zeros(len(values), dtype=np.int64)
    for i, value in enumerate(values):
        try:
            v = int(float(value))
        except (TypeError, ValueError):
            v = 0
        hour = max(0, min(23, v // 100))
        minute = max(0, min(59, v % 100))
        out[i] = hour * 60 + minute
    return out


def read_csv(path, training):
    rows = []
    with open(path, 'r', newline='') as fh:
        for row in csv.DictReader(fh):
            record = {
                'user_id': row['user_id'],
                'video_id': row['video_id'],
                'tab': row.get('tab', '0'),
                'duration_ms': float(row.get('duration_ms', 0.0) or 0.0),
                'hourmin': row.get('hourmin', '0'),
                'date': row.get('date', '0'),
                'long_view': float(row['long_view'])
            }
            if training:
                record['play_time_ms'] = float(row.get('play_time_ms', 0.0) or 0.0)
            rows.append(record)
    return rows


def load_csv_data(data_dir):
    train_rows = read_csv(os.path.join(data_dir, 'train.csv'), True)
    val_rows = read_csv(os.path.join(data_dir, 'val.csv'), False)
    durations = np.asarray([r['duration_ms'] for r in train_rows], dtype=np.float64)
    cuts = np.quantile(durations, np.linspace(0.1, 0.9, 9)) if len(durations) else np.zeros(9)

    def vocab(key):
        return {v: i + 1 for i, v in enumerate(sorted({r[key] for r in train_rows}))}

    user_map = vocab('user_id')
    video_map = vocab('video_id')
    tab_map = vocab('tab')
    field_dims = np.asarray([len(user_map) + 1, len(video_map) + 1, 1, len(tab_map) + 1, 10], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1]))

    def encode(rows):
        x = np.empty((len(rows), 5), dtype=np.int64)
        for i, row in enumerate(rows):
            x[i, 0] = user_map.get(row['user_id'], 0)
            x[i, 1] = video_map.get(row['video_id'], 0)
            x[i, 2] = 0
            x[i, 3] = tab_map.get(row['tab'], 0)
            x[i, 4] = int(np.searchsorted(cuts, row['duration_ms'], side='right'))
        return x + offsets

    train = {
        'X': encode(train_rows),
        'y': np.asarray([r['long_view'] for r in train_rows], dtype=np.float32),
        'user': np.asarray([r['user_id'] for r in train_rows]),
        'video': np.asarray([r['video_id'] for r in train_rows]),
        'duration_ms': np.asarray([r['duration_ms'] for r in train_rows], dtype=np.float32),
        'play_time_ms': np.asarray([r['play_time_ms'] for r in train_rows], dtype=np.float32),
        'hourmin': np.asarray([r['hourmin'] for r in train_rows]),
        'date': np.asarray([r['date'] for r in train_rows]),
        'field_dims': field_dims
    }
    val = {
        'X': encode(val_rows),
        'y': np.asarray([r['long_view'] for r in val_rows], dtype=np.float32),
        'user': np.asarray([r['user_id'] for r in val_rows]),
        'video': np.asarray([r['video_id'] for r in val_rows]),
        'duration_ms': np.asarray([r['duration_ms'] for r in val_rows], dtype=np.float32),
        'hourmin': np.asarray([r['hourmin'] for r in val_rows]),
        'date': np.asarray([r['date'] for r in val_rows])
    }
    return train, val


def load_data(data_dir):
    train_path = os.path.join(data_dir, 'train.npz')
    val_path = os.path.join(data_dir, 'val.npz')
    if os.path.exists(train_path) and os.path.exists(val_path):
        with np.load(train_path) as z:
            train = {k: z[k] for k in z.files}
        with np.load(val_path) as z:
            val = {k: z[k] for k in z.files}
        if 'video' not in val:
            val['video'] = np.zeros(len(val['y']), dtype=np.int64)
        return train, val, True
    train, val = load_csv_data(data_dir)
    return train, val, False


def local_fields(x, field_dims):
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1])).astype(np.int64)
    return x.astype(np.int64) - offsets[None, :]


def chronological_features(train, val, history_len=12):
    field_dims = np.asarray(train['field_dims'], dtype=np.int64)
    tr_local = local_fields(train['X'], field_dims)
    va_local = local_fields(val['X'], field_dims)
    tr_dates = date_ordinals(train.get('date', np.zeros(len(tr_local))))
    va_dates = date_ordinals(val.get('date', np.zeros(len(va_local))))
    tr_hm = hour_minutes(train.get('hourmin', np.zeros(len(tr_local))))
    va_hm = hour_minutes(val.get('hourmin', np.zeros(len(va_local))))
    tr_time = tr_dates * 1440 + tr_hm
    va_time = va_dates * 1440 + va_hm
    tr_users = np.asarray(train['user'])
    va_users = np.asarray(val['user'])
    author_offset = int(field_dims[0] + field_dims[1])
    unknown_author = author_offset
    tr_hist = np.full((len(tr_local), history_len), unknown_author, dtype=np.int32)
    va_hist = np.full((len(va_local), history_len), unknown_author, dtype=np.int32)
    tr_mask = np.zeros((len(tr_local), history_len), dtype=np.float32)
    va_mask = np.zeros((len(va_local), history_len), dtype=np.float32)
    tr_gap = np.zeros(len(tr_local), dtype=np.int64)
    va_gap = np.zeros(len(va_local), dtype=np.int64)
    tr_pos = np.zeros(len(tr_local), dtype=np.int64)
    va_pos = np.zeros(len(va_local), dtype=np.int64)
    gap_edges = np.asarray([1, 2, 5, 10, 30, 60, 180, 720], dtype=np.int64)
    state = {}

    def process(users, times, local, hist, mask, gap, pos):
        order = np.lexsort((np.arange(len(users), dtype=np.int64), times))
        for idx in order:
            key = users[idx].item() if isinstance(users[idx], np.generic) else users[idx]
            previous = state.get(key)
            if previous is None:
                authors = []
                previous_time = None
                session_pos = 0
            else:
                authors, previous_time, session_pos = previous
            count = min(history_len, len(authors))
            if count:
                hist[idx, history_len - count:] = np.asarray(authors[-count:], dtype=np.int32)
                mask[idx, history_len - count:] = 1.0
            if previous_time is None:
                delta = 1000000
                session_pos = 0
            else:
                delta = max(0, int(times[idx] - previous_time))
                session_pos = session_pos + 1 if delta <= 30 else 0
            gap[idx] = int(np.searchsorted(gap_edges, delta, side='right'))
            pos[idx] = min(session_pos, 15)
            updated = (authors + [int(local[idx, 2] + author_offset)])[-history_len:]
            state[key] = (updated, int(times[idx]), session_pos)

    process(tr_users, tr_time, tr_local, tr_hist, tr_mask, tr_gap, tr_pos)
    process(va_users, va_time, va_local, va_hist, va_mask, va_gap, va_pos)

    tr_hour = tr_hm // 60
    va_hour = va_hm // 60
    tr_weekday = np.asarray([(datetime.date.fromordinal(int(d)).weekday() if d > 0 else 0) for d in tr_dates], dtype=np.int64)
    va_weekday = np.asarray([(datetime.date.fromordinal(int(d)).weekday() if d > 0 else 0) for d in va_dates], dtype=np.int64)
    tr_rand = (tr_local[:, 3] == 1).astype(np.int64)
    va_rand = (va_local[:, 3] == 1).astype(np.int64)
    extra_dims = np.asarray([24, 7, 2, 9, 16], dtype=np.int64)
    all_dims = np.concatenate((field_dims, extra_dims))
    all_offsets = np.concatenate(([0], np.cumsum(all_dims)[:-1])).astype(np.int64)

    def expanded(local, hour, weekday, is_rand, gap, pos):
        raw = np.column_stack((local, hour, weekday, is_rand, gap, pos)).astype(np.int64)
        return (raw + all_offsets[None, :]).astype(np.int64)

    return expanded(tr_local, tr_hour, tr_weekday, tr_rand, tr_gap, tr_pos), expanded(va_local, va_hour, va_weekday, va_rand, va_gap, va_pos), tr_hist, va_hist, tr_mask, va_mask, all_dims


def make_pairs(users, labels, seed):
    order = np.argsort(users, kind='mergesort')
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    rng = np.random.default_rng(seed)
    positives = []
    negatives = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        idx = order[left:right]
        pos = idx[labels[idx] > 0.5]
        neg = idx[labels[idx] <= 0.5]
        if len(pos) and len(neg):
            positives.append(pos)
            negatives.append(neg[rng.integers(0, len(neg), size=len(pos))])
    if not positives:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(positives).astype(np.int64), np.concatenate(negatives).astype(np.int64)


class SequenceDeepFM(torch.nn.Module):
    def __init__(self, total_dim, fields, embedding_dim, dropout):
        super().__init__()
        self.embedding = torch.nn.Embedding(total_dim, embedding_dim)
        self.linear = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        width = (fields + 1) * embedding_dim
        self.embedding_dropout = torch.nn.Dropout(dropout)
        self.deep = torch.nn.Sequential(
            torch.nn.Linear(width, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout)
        )
        self.main_head = torch.nn.Linear(64, 1)
        self.watch_head = torch.nn.Linear(64, 1)
        torch.nn.init.normal_(self.embedding.weight, std=0.01)
        torch.nn.init.zeros_(self.linear.weight)
        for layer in self.deep:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
        torch.nn.init.zeros_(self.main_head.weight)
        torch.nn.init.zeros_(self.main_head.bias)
        torch.nn.init.xavier_uniform_(self.watch_head.weight)
        torch.nn.init.zeros_(self.watch_head.bias)

    def forward(self, x, history, history_mask):
        field_emb = self.embedding_dropout(self.embedding(x))
        summed = field_emb.sum(dim=1)
        fm = 0.5 * (summed.square() - field_emb.square().sum(dim=1)).sum(dim=1)
        linear = self.linear(x).sum(dim=(1, 2)) + self.bias
        hist_emb = self.embedding(history)
        denom = history_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (hist_emb * history_mask.unsqueeze(-1)).sum(dim=1) / denom
        pooled = self.embedding_dropout(pooled)
        deep_input = torch.cat((field_emb.flatten(1), pooled), dim=1)
        representation = self.deep(deep_input)
        logits = linear + fm + self.main_head(representation).squeeze(1)
        watch = torch.nn.functional.softplus(self.watch_head(representation).squeeze(1))
        return logits, watch


def watch_targets(train):
    play = np.maximum(0.0, np.asarray(train.get('play_time_ms', np.zeros(len(train['y']))), dtype=np.float32))
    duration = np.maximum(0.0, np.asarray(train.get('duration_ms', np.zeros(len(train['y']))), dtype=np.float32))
    clipped = np.minimum(play, duration)
    scale = float(np.log1p(300.0))
    target = np.log1p(clipped / 1000.0) / scale
    censored = ((duration > 0.0) & (play >= duration)).astype(np.float32)
    valid = (duration > 0.0).astype(np.float32)
    return target.astype(np.float32), censored, valid


def predict(model, x, history, mask, device):
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(x), 65536):
            end = start + 65536
            xb = x[start:end].to(device, non_blocking=True)
            hb = history[start:end].to(device, non_blocking=True).long()
            mb = mask[start:end].to(device, non_blocking=True)
            logits, _ = model(xb, hb, mb)
            parts.append(logits.cpu().numpy())
    return np.concatenate(parts).astype(np.float64)


def train_one(config, seed, epochs, arrays, evaluator, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    x_train, y_train, tr_hist, tr_mask, watch_y, censored, watch_valid, x_val, va_hist, va_mask, val_user, val_y, recency, pair_pos, pair_neg, total_dim = arrays
    model = SequenceDeepFM(total_dim, x_train.shape[1], 16, config['dropout']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'])
    batch_size = 8192 if device.type == 'cuda' else 4096
    n = len(y_train)
    pair_n = len(pair_pos)
    best_primary = -1.0
    best_scores = None
    best_metrics = None
    trace = []
    for epoch in range(epochs):
        permutation = torch.randperm(n)
        midpoint = (n + 1) // 2
        for half, (left, right) in enumerate(((0, midpoint), (midpoint, n))):
            model.train()
            losses = []
            segment = permutation[left:right]
            for start in range(0, len(segment), batch_size):
                idx = segment[start:start + batch_size]
                xb = x_train[idx].to(device, non_blocking=True)
                hb = tr_hist[idx].to(device, non_blocking=True).long()
                mb = tr_mask[idx].to(device, non_blocking=True)
                yb = y_train[idx].to(device, non_blocking=True)
                wb = recency[idx].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits, watch_pred = model(xb, hb, mb)
                point = torch.nn.functional.binary_cross_entropy_with_logits(logits, yb, reduction='none')
                point = (point * wb).sum() / wb.sum().clamp_min(1e-8)
                target = watch_y[idx].to(device, non_blocking=True)
                censor = censored[idx].to(device, non_blocking=True)
                valid = watch_valid[idx].to(device, non_blocking=True)
                symmetric = torch.nn.functional.smooth_l1_loss(watch_pred, target, reduction='none')
                one_sided = torch.nn.functional.smooth_l1_loss(torch.minimum(watch_pred, target), target, reduction='none')
                auxiliary = ((symmetric * (1.0 - censor) + one_sided * censor) * valid * wb).sum()
                auxiliary = auxiliary / (valid * wb).sum().clamp_min(1e-8)
                if pair_n:
                    selected = torch.randint(0, pair_n, (len(idx),))
                    pi = pair_pos[selected]
                    ni = pair_neg[selected]
                    xp = x_train[pi].to(device, non_blocking=True)
                    xn = x_train[ni].to(device, non_blocking=True)
                    hp = tr_hist[pi].to(device, non_blocking=True).long()
                    hn = tr_hist[ni].to(device, non_blocking=True).long()
                    mp = tr_mask[pi].to(device, non_blocking=True)
                    mn = tr_mask[ni].to(device, non_blocking=True)
                    pos_logits, _ = model(xp, hp, mp)
                    neg_logits, _ = model(xn, hn, mn)
                    pair_weight = ((recency[pi] + recency[ni]) * 0.5).to(device, non_blocking=True)
                    rank = torch.nn.functional.softplus(-(pos_logits - neg_logits))
                    rank = (rank * pair_weight).sum() / pair_weight.sum().clamp_min(1e-8)
                    main_loss = (1.0 - config['bpr_mix']) * point + config['bpr_mix'] * rank
                else:
                    main_loss = point
                loss = main_loss + config['aux_weight'] * auxiliary
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            scores = predict(model, x_val, va_hist, va_mask, device)
            metrics = evaluator(val_user, val_y, scores)
            primary = metric_value(metrics, 'primary')
            trace.append({
                'epoch': epoch + 0.5 * (half + 1),
                'train_loss': round(float(np.mean(losses)) if losses else 0.0, 6),
                'lr': float(optimizer.param_groups[0]['lr']),
                'val_gauc': round(metric_value(metrics, 'GAUC', 'gauc'), 6),
                'val_primary': round(primary, 6)
            })
            if primary > best_primary:
                best_primary = primary
                best_scores = scores.copy()
                best_metrics = metrics
        scheduler.step()
    return best_primary, best_scores, best_metrics, trace


def serializable_config(config):
    result = {}
    for key, value in config.items():
        result[key] = int(value) if key == 'step_size' else float(value)
    return result


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
    evaluator = lambda u, y, s: run_evaluator(u, y, s, fast_path)
    x_tr_np, x_va_np, hist_tr_np, hist_va_np, mask_tr_np, mask_va_np, field_dims = chronological_features(train, val, 12)
    x_train = torch.from_numpy(x_tr_np)
    x_val = torch.from_numpy(x_va_np)
    tr_hist = torch.from_numpy(hist_tr_np)
    va_hist = torch.from_numpy(hist_va_np)
    tr_mask = torch.from_numpy(mask_tr_np)
    va_mask = torch.from_numpy(mask_va_np)
    y_train_np = np.asarray(train['y'], dtype=np.float32)
    y_train = torch.from_numpy(y_train_np)
    val_y = np.asarray(val['y'], dtype=np.float32)
    val_user = np.asarray(val['user'])
    total_dim = int(field_dims.sum())
    target_np, censored_np, valid_np = watch_targets(train)
    watch_y = torch.from_numpy(target_np)
    censored = torch.from_numpy(censored_np)
    watch_valid = torch.from_numpy(valid_np)
    ordinals = date_ordinals(train.get('date', np.zeros(len(y_train_np)))).astype(np.float32)
    ages = np.maximum(0.0, float(ordinals.max()) - ordinals) if len(ordinals) else np.zeros(0, dtype=np.float32)
    pair_pos_np, pair_neg_np = make_pairs(np.asarray(train['user']), y_train_np, args.seed)
    pair_pos = torch.from_numpy(pair_pos_np)
    pair_neg = torch.from_numpy(pair_neg_np)

    smoke = int(os.environ.get('SMOKE_EPOCHS', '0') or 0)
    coarse_epochs = min(3, smoke) if smoke > 0 else 3
    refine_epochs = min(5, smoke) if smoke > 0 else 5
    final_epochs = min(args.epochs, smoke) if smoke > 0 else args.epochs
    if smoke > 0:
        coarse_count, refine_count, final_seed_count = 2, 1, 1
    elif device.type == 'cuda':
        coarse_count, refine_count, final_seed_count = 80, 30, 3
    else:
        coarse_count, refine_count, final_seed_count = 48, 20, 3

    rng = np.random.default_rng(args.seed + 2719)
    coarse_configs = []
    for _ in range(coarse_count):
        coarse_configs.append({
            'dropout': float(rng.uniform(0.12, 0.38)),
            'weight_decay': float(10.0 ** rng.uniform(math.log10(2e-5), math.log10(2e-3))),
            'lr': float(10.0 ** rng.uniform(math.log10(3.5e-4), math.log10(1.4e-3))),
            'gamma': float(rng.choice(np.asarray([0.35, 0.45, 0.58, 0.70, 0.82]))),
            'step_size': int(rng.choice(np.asarray([1, 1, 1, 2, 2, 3]))),
            'half_life': float(rng.choice(np.asarray([3.0, 4.5, 6.5, 9.0, 13.0]))),
            'aux_weight': float(10.0 ** rng.uniform(math.log10(0.025), math.log10(0.18))),
            'bpr_mix': float(rng.choice(np.asarray([0.35, 0.45, 0.50, 0.55, 0.65])))
        })

    history = []
    best_primary = -1.0
    best_config = None

    def execute(stage, index, config, epochs):
        nonlocal best_primary, best_config
        weights = np.exp(-math.log(2.0) * ages / config['half_life']).astype(np.float32)
        weights /= max(float(weights.mean()), 1e-8)
        recency = torch.from_numpy(weights)
        arrays = (x_train, y_train, tr_hist, tr_mask, watch_y, censored, watch_valid, x_val, va_hist, va_mask, val_user, val_y, recency, pair_pos, pair_neg, total_dim)
        seed_offset = index + (0 if stage == 'coarse' else 10000)
        primary, _, metrics, trace = train_one(config, args.seed + seed_offset, epochs, arrays, evaluator, device)
        record = {
            'stage': stage,
            'probe': index,
            'config': serializable_config(config),
            'epochs': epochs,
            'gauc': metric_value(metrics, 'GAUC', 'gauc'),
            'ndcg5': metric_value(metrics, 'nDCG@5', 'ndcg5'),
            'primary': primary,
            'best_checkpoint': max(trace, key=lambda item: item['val_primary'])['epoch']
        }
        history.append(record)
        with open(progress_path, 'a') as fh:
            fh.write(json.dumps(record, sort_keys=True) + '\n')
        if primary > best_primary:
            best_primary = primary
            best_config = dict(config)

    for i, config in enumerate(coarse_configs):
        execute('coarse', i, config, coarse_epochs)

    center = dict(best_config)
    refine_configs = []
    for _ in range(refine_count):
        refine_configs.append({
            'dropout': float(np.clip(center['dropout'] + rng.normal(0.0, 0.035), 0.08, 0.44)),
            'weight_decay': float(np.clip(center['weight_decay'] * math.exp(rng.normal(0.0, 0.40)), 1e-5, 4e-3)),
            'lr': float(np.clip(center['lr'] * math.exp(rng.normal(0.0, 0.20)), 2.5e-4, 1.8e-3)),
            'gamma': float(np.clip(center['gamma'] + rng.normal(0.0, 0.06), 0.25, 0.90)),
            'step_size': int(np.clip(center['step_size'] + rng.choice(np.asarray([-1, 0, 0, 0, 1])), 1, 3)),
            'half_life': float(np.clip(center['half_life'] * math.exp(rng.normal(0.0, 0.18)), 2.5, 18.0)),
            'aux_weight': float(np.clip(center['aux_weight'] * math.exp(rng.normal(0.0, 0.28)), 0.015, 0.25)),
            'bpr_mix': float(np.clip(center['bpr_mix'] + rng.normal(0.0, 0.06), 0.25, 0.75))
        })
    for i, config in enumerate(refine_configs):
        execute('refine', i, config, refine_epochs)

    final_weights = np.exp(-math.log(2.0) * ages / best_config['half_life']).astype(np.float32)
    final_weights /= max(float(final_weights.mean()), 1e-8)
    final_recency = torch.from_numpy(final_weights)
    final_arrays = (x_train, y_train, tr_hist, tr_mask, watch_y, censored, watch_valid, x_val, va_hist, va_mask, val_user, val_y, final_recency, pair_pos, pair_neg, total_dim)
    score_members = []
    final_runs = []
    for offset in range(final_seed_count):
        run_seed = args.seed + offset
        primary, scores, metrics, trace = train_one(best_config, run_seed, final_epochs, final_arrays, evaluator, device)
        score_members.append(scores)
        final_runs.append({
            'seed': run_seed,
            'best_primary': primary,
            'gauc': metric_value(metrics, 'GAUC', 'gauc'),
            'ndcg5': metric_value(metrics, 'nDCG@5', 'ndcg5'),
            'trace': trace
        })

    blended_scores = np.mean(np.stack(score_members, axis=0), axis=0)
    final_metrics = evaluator(val_user, val_y, blended_scores)
    output = {
        'gauc': metric_value(final_metrics, 'GAUC', 'gauc'),
        'ndcg5': metric_value(final_metrics, 'nDCG@5', 'ndcg5'),
        'primary': metric_value(final_metrics, 'primary'),
        'best_config': serializable_config(best_config),
        'history': history,
        'final_runs': final_runs,
        'ensemble': {'method': 'mean_logit', 'seeds': [args.seed + i for i in range(final_seed_count)]}
    }
    with open(os.path.join(args.out_dir, 'metrics.json'), 'w') as fh:
        json.dump(output, fh)

    video = val.get('video', np.zeros(len(val_y), dtype=np.int64))
    with open(os.path.join(args.out_dir, 'predictions.csv'), 'w') as fh:
        fh.write('row_id,user_id,video_id,score\n')
        for i, score in enumerate(blended_scores):
            fh.write(f'{i},{val_user[i]},{video[i]},{score:.9g}\n')


if __name__ == '__main__':
    main()
