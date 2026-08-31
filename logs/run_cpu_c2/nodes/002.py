import argparse
import csv
import datetime
import json
import os
import sys
import time
from collections import deque

import numpy as np
import torch


class SequenceDeepFM(torch.nn.Module):
    def __init__(self, total_dim, fields=10, k=16, hidden=256, dropout=0.25,
                 history_dropout=0.15, pad_id=0):
        super().__init__()
        self.fields = fields
        self.k = k
        self.pad_id = int(pad_id)
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.field_dropout = torch.nn.Dropout(dropout)
        self.history_dropout = torch.nn.Dropout(history_dropout)
        d = (fields + 1) * k
        self.deep = torch.nn.Sequential(
            torch.nn.Linear(d, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.main_head = torch.nn.Linear(64, 1)
        self.watch_head = torch.nn.Linear(64, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)
        for layer in self.deep:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
        torch.nn.init.xavier_uniform_(self.main_head.weight)
        torch.nn.init.zeros_(self.main_head.bias)
        torch.nn.init.xavier_uniform_(self.watch_head.weight)
        torch.nn.init.zeros_(self.watch_head.bias)

    def forward(self, x, history):
        field_e = self.field_dropout(self.emb(x))
        hist_mask = history.ne(self.pad_id).unsqueeze(-1)
        hist_e = self.emb(history)
        hist_sum = (hist_e * hist_mask).sum(1)
        hist_count = hist_mask.sum(1).clamp_min(1)
        hist_mean = self.history_dropout(hist_sum / hist_count)
        all_e = torch.cat([field_e, hist_mean.unsqueeze(1)], dim=1)
        summed = all_e.sum(1)
        fm = 0.5 * (summed.square() - all_e.square().sum(1)).sum(1)
        hist_linear = (self.lin(history).squeeze(-1) * hist_mask.squeeze(-1)).sum(1)
        hist_linear = hist_linear / hist_mask.squeeze(-1).sum(1).clamp_min(1)
        linear = self.bias + self.lin(x).sum((1, 2)) + hist_linear
        shared = self.deep(all_e.flatten(1))
        logits = linear + fm + self.main_head(shared).squeeze(1)
        watch = self.watch_head(shared).squeeze(1)
        return logits, watch

    def score(self, x, history):
        return self.forward(x, history)[0]


def parse_day(value):
    s = str(value)
    if s.endswith('.0'):
        s = s[:-2]
    digits = ''.join(ch for ch in s if ch.isdigit())[:8]
    try:
        return datetime.date(int(digits[:4]), int(digits[4:6]),
                             int(digits[6:8])).toordinal()
    except Exception:
        return 0


def parse_hourmin(values):
    arr = np.asarray(values)
    out_hour = np.zeros(len(arr), dtype=np.int64)
    out_minute = np.zeros(len(arr), dtype=np.int64)
    for i, value in enumerate(arr):
        try:
            v = int(float(value))
        except Exception:
            v = 0
        hour = v // 100
        minute = v % 100
        if hour < 0 or hour > 23 or minute < 0 or minute > 59:
            total = max(0, v) % 1440
            hour, minute = total // 60, total % 60
        out_hour[i] = hour
        out_minute[i] = minute
    return out_hour, out_minute


def timestamps(dates, hourmin):
    values = np.asarray(dates)
    unique, inv = np.unique(values.astype(str), return_inverse=True)
    ordinals = np.asarray([parse_day(v) for v in unique], dtype=np.int64)
    hour, minute = parse_hourmin(hourmin)
    return ordinals[inv] * 1440 + hour * 60 + minute, hour, ordinals[inv] % 7


def recency_weights(dates, half_life):
    arr = np.asarray(dates)
    unique, inv = np.unique(arr.astype(str), return_inverse=True)
    ordinals = np.asarray([parse_day(x) for x in unique], dtype=np.float32)
    newest = float(ordinals.max()) if len(ordinals) else 0.0
    age = newest - ordinals[inv]
    weights = np.exp2(-age / float(half_life)).astype(np.float32)
    return weights / max(float(weights.mean()), 1e-8)


def encode_field(train_values, val_values):
    mapping = {}
    train_encoded = np.empty(len(train_values), dtype=np.int64)
    for i, value in enumerate(train_values):
        key = str(value)
        if key not in mapping:
            mapping[key] = len(mapping) + 1
        train_encoded[i] = mapping[key]
    val_encoded = np.asarray([mapping.get(str(v), 0) for v in val_values], dtype=np.int64)
    return train_encoded, val_encoded, len(mapping) + 1


def load_csv_fallback(data_dir):
    def read_file(path, training):
        names = ['user_id', 'video_id', 'tab', 'duration_ms', 'date',
                 'hourmin', 'long_view']
        if training:
            names.append('play_time_ms')
        columns = {name: [] for name in names}
        with open(path, newline='') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                for name in names:
                    columns[name].append(row[name])
        return columns

    tr = read_file(os.path.join(data_dir, 'train.csv'), True)
    va = read_file(os.path.join(data_dir, 'val.csv'), False)
    duration_train = np.asarray(tr['duration_ms'], dtype=np.float32)
    duration_val = np.asarray(va['duration_ms'], dtype=np.float32)
    cuts = np.unique(np.quantile(duration_train, np.linspace(0.1, 0.9, 9)))
    train_raw = [tr['user_id'], tr['video_id'], tr['video_id'], tr['tab'],
                 np.searchsorted(cuts, duration_train, side='right').astype(str)]
    val_raw = [va['user_id'], va['video_id'], va['video_id'], va['tab'],
               np.searchsorted(cuts, duration_val, side='right').astype(str)]
    tx, vx, dims = [], [], []
    offset = 0
    for train_col, val_col in zip(train_raw, val_raw):
        te, ve, dim = encode_field(train_col, val_col)
        tx.append(te + offset)
        vx.append(ve + offset)
        dims.append(dim)
        offset += dim
    return {
        'Xt': np.stack(tx, axis=1).astype(np.int64),
        'yt': np.asarray(tr['long_view'], dtype=np.float32),
        'train_user': np.asarray(tr['user_id']),
        'train_date': np.asarray(tr['date']),
        'train_hourmin': np.asarray(tr['hourmin']),
        'train_play': np.asarray(tr['play_time_ms'], dtype=np.float32),
        'train_duration': duration_train,
        'Xv': np.stack(vx, axis=1).astype(np.int64),
        'yv': np.asarray(va['long_view'], dtype=np.float32),
        'val_user': np.asarray(va['user_id']),
        'val_video': np.asarray(va['video_id']),
        'val_date': np.asarray(va['date']),
        'val_hourmin': np.asarray(va['hourmin']),
        'val_duration': duration_val,
        'field_dims': np.asarray(dims, dtype=np.int64),
        'fast': False,
    }


def load_data(data_dir):
    train_path = os.path.join(data_dir, 'train.npz')
    val_path = os.path.join(data_dir, 'val.npz')
    if os.path.exists(train_path) and os.path.exists(val_path):
        tr = np.load(train_path)
        va = np.load(val_path)
        dims = tr['field_dims'].astype(np.int64)
        video_offset = int(dims[0])
        return {
            'Xt': tr['X'].astype(np.int64),
            'yt': tr['y'].astype(np.float32),
            'train_user': np.asarray(tr['user']),
            'train_date': np.asarray(tr['date']),
            'train_hourmin': np.asarray(tr['hourmin']),
            'train_play': np.asarray(tr['play_time_ms'], dtype=np.float32),
            'train_duration': np.asarray(tr['duration_ms'], dtype=np.float32),
            'Xv': va['X'].astype(np.int64),
            'yv': va['y'].astype(np.float32),
            'val_user': np.asarray(va['user']),
            'val_video': va['X'][:, 1].astype(np.int64) - video_offset,
            'val_date': np.asarray(va['date']),
            'val_hourmin': np.asarray(va['hourmin']),
            'val_duration': np.asarray(va['duration_ms'], dtype=np.float32),
            'field_dims': dims,
            'fast': True,
        }
    return load_csv_fallback(data_dir)


def build_sequence_features(data, history_length=12):
    train_ts, train_hour, train_weekday = timestamps(
        data['train_date'], data['train_hourmin'])
    val_ts, val_hour, val_weekday = timestamps(data['val_date'], data['val_hourmin'])
    n_train = len(data['Xt'])
    n_val = len(data['Xv'])
    train_hist = np.empty((n_train, history_length), dtype=np.int32)
    val_hist = np.empty((n_val, history_length), dtype=np.int32)
    train_gap = np.zeros(n_train, dtype=np.int64)
    val_gap = np.zeros(n_val, dtype=np.int64)
    train_pos = np.zeros(n_train, dtype=np.int64)
    val_pos = np.zeros(n_val, dtype=np.int64)
    base_dim = int(data['field_dims'].sum())
    pad_id = base_dim
    train_hist.fill(pad_id)
    val_hist.fill(pad_id)
    histories = {}
    last_time = {}
    session_position = {}
    gap_edges = np.asarray([1, 2, 5, 10, 20, 30, 60, 180, 720], dtype=np.int64)

    def process(users, authors, ts, output_hist, output_gap, output_pos):
        order = np.argsort(ts, kind='stable')
        for idx in order:
            user = str(users[idx])
            history = histories.get(user)
            if history is None:
                history = deque(maxlen=history_length)
                histories[user] = history
            if history:
                values = np.fromiter(history, dtype=np.int32, count=len(history))
                output_hist[idx, history_length - len(values):] = values
            previous = last_time.get(user)
            if previous is None:
                gap = 10
                position = 0
            else:
                delta = max(0, int(ts[idx] - previous))
                gap = int(np.searchsorted(gap_edges, delta, side='right'))
                if delta > 30:
                    position = 0
                else:
                    position = min(session_position[user] + 1, 9)
            output_gap[idx] = gap
            output_pos[idx] = position
            session_position[user] = position
            last_time[user] = int(ts[idx])
            history.append(int(authors[idx]))

    process(data['train_user'], data['Xt'][:, 2], train_ts,
            train_hist, train_gap, train_pos)
    process(data['val_user'], data['Xv'][:, 2], val_ts,
            val_hist, val_gap, val_pos)

    tab_offset = int(data['field_dims'][:3].sum())
    train_tab_local = data['Xt'][:, 3] - tab_offset
    val_tab_local = data['Xv'][:, 3] - tab_offset
    train_rand = (train_tab_local == 1).astype(np.int64)
    val_rand = (val_tab_local == 1).astype(np.int64)
    context_dims = [24, 7, 2, 11, 10]

    def add_context(X, hour, weekday, is_rand, gap, position):
        result = [X]
        local_offset = base_dim + 1
        for values, dim in zip([hour, weekday, is_rand, gap, position], context_dims):
            result.append((np.asarray(values, dtype=np.int64) + local_offset)[:, None])
            local_offset += dim
        return np.concatenate(result, axis=1).astype(np.int64)

    data['Xt'] = add_context(data['Xt'], train_hour, train_weekday,
                             train_rand, train_gap, train_pos)
    data['Xv'] = add_context(data['Xv'], val_hour, val_weekday,
                             val_rand, val_gap, val_pos)
    data['Ht'] = train_hist.astype(np.int64)
    data['Hv'] = val_hist.astype(np.int64)
    data['pad_id'] = pad_id
    data['total_dim'] = base_dim + 1 + sum(context_dims)
    data['watch_target'] = (
        np.log1p(np.minimum(np.maximum(data['train_play'], 0.0),
                            np.maximum(data['train_duration'], 0.0))) / 12.0
    ).astype(np.float32)
    data['watch_censored'] = (
        data['train_play'] >= np.maximum(data['train_duration'], 1.0)
    ).astype(np.float32)
    return data


def make_pairs(users, labels, seed):
    order = np.argsort(users, kind='stable')
    sorted_users = users[order]
    bounds = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    rng = np.random.RandomState(seed)
    positives, negatives = [], []
    for left, right in zip(bounds[:-1], bounds[1:]):
        group = order[left:right]
        pos = group[labels[group] > 0.5]
        neg = group[labels[group] <= 0.5]
        if len(pos) and len(neg):
            pos = pos.copy()
            neg = neg.copy()
            rng.shuffle(pos)
            rng.shuffle(neg)
            positives.append(pos)
            negatives.append(np.resize(neg, len(pos)))
    if not positives:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(positives), np.concatenate(negatives)


def metric_dict(evaluate_fn, users, labels, scores):
    metrics = evaluate_fn(users, labels.astype(int), scores)
    return {
        'gauc': float(metrics.get('GAUC', metrics.get('gauc'))),
        'ndcg5': float(metrics.get('nDCG@5', metrics.get('ndcg5'))),
        'primary': float(metrics['primary']),
    }


def predict(model, X, H, device, batch_size=32768):
    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            stop = start + batch_size
            xb = torch.as_tensor(X[start:stop], dtype=torch.long, device=device)
            hb = torch.as_tensor(H[start:stop], dtype=torch.long, device=device)
            result.append(model.score(xb, hb).detach().cpu().numpy())
    return np.concatenate(result).astype(np.float64)


def censored_watch_loss(prediction, target, censored):
    uncensored_loss = torch.nn.functional.smooth_l1_loss(
        prediction, target, reduction='none')
    censored_loss = torch.nn.functional.smooth_l1_loss(
        torch.maximum(prediction, target), target, reduction='none')
    return ((1.0 - censored) * uncensored_loss + censored * censored_loss).mean()


def train_once(data, pairs, config, seed, epochs, evaluate_fn, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    model = SequenceDeepFM(
        data['total_dim'], dropout=float(config['dropout']),
        history_dropout=float(config['history_dropout']),
        pad_id=data['pad_id']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['lr']),
                                  weight_decay=float(config['weight_decay']))
    weights = recency_weights(data['train_date'], float(config['half_life']))
    pos_idx, neg_idx = pairs
    pair_n = len(pos_idx)
    n = len(data['yt'])
    batch_size = 32768
    rng = np.random.RandomState(seed + 1907)
    best_primary = -1.0
    best_scores = None
    curve = []
    pair_cursor = 0
    for epoch in range(epochs):
        model.train()
        permutation = rng.permutation(n)
        pair_perm = rng.permutation(pair_n) if pair_n else None
        decay_power = int(epoch // float(config['step_every']))
        lr = float(config['lr']) * float(config['gamma']) ** decay_power
        for group in optimizer.param_groups:
            group['lr'] = lr
        last_loss = 0.0
        for start in range(0, n, batch_size):
            idx_np = permutation[start:start + batch_size]
            xb = torch.as_tensor(data['Xt'][idx_np], dtype=torch.long, device=device)
            hb = torch.as_tensor(data['Ht'][idx_np], dtype=torch.long, device=device)
            yb = torch.as_tensor(data['yt'][idx_np], dtype=torch.float32, device=device)
            wb = torch.as_tensor(weights[idx_np], dtype=torch.float32, device=device)
            watch_target = torch.as_tensor(data['watch_target'][idx_np],
                                           dtype=torch.float32, device=device)
            censored = torch.as_tensor(data['watch_censored'][idx_np],
                                       dtype=torch.float32, device=device)
            logits, watch_prediction = model(xb, hb)
            raw_bce = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, yb, reduction='none')
            bce = (raw_bce * wb).sum() / wb.sum().clamp_min(1e-8)
            auxiliary = censored_watch_loss(watch_prediction, watch_target, censored)
            if pair_n:
                take = len(idx_np)
                locations = np.arange(pair_cursor, pair_cursor + take) % pair_n
                chosen = pair_perm[locations]
                pair_cursor = int((pair_cursor + take) % pair_n)
                pi, ni = pos_idx[chosen], neg_idx[chosen]
                xp = torch.as_tensor(data['Xt'][pi], dtype=torch.long, device=device)
                hp = torch.as_tensor(data['Ht'][pi], dtype=torch.long, device=device)
                xn = torch.as_tensor(data['Xt'][ni], dtype=torch.long, device=device)
                hn = torch.as_tensor(data['Ht'][ni], dtype=torch.long, device=device)
                pair_weight = torch.as_tensor(weights[pi], dtype=torch.float32,
                                              device=device)
                pair_raw = torch.nn.functional.softplus(
                    -(model.score(xp, hp) - model.score(xn, hn)))
                pair_loss = (pair_raw * pair_weight).sum() / pair_weight.sum().clamp_min(1e-8)
                main_loss = float(config['bce_mix']) * bce + \
                    (1.0 - float(config['bce_mix'])) * pair_loss
            else:
                main_loss = bce
            loss = main_loss + float(config['aux_weight']) * auxiliary
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            last_loss = float(loss.detach().cpu())
        scores = predict(model, data['Xv'], data['Hv'], device)
        metrics = metric_dict(evaluate_fn, data['val_user'], data['yv'], scores)
        curve.append({'epoch': epoch + 1, 'train_loss': round(last_loss, 6),
                      'lr': lr, 'val_gauc': round(metrics['gauc'], 6),
                      'val_primary': round(metrics['primary'], 6)})
        if metrics['primary'] > best_primary + 1e-9:
            best_primary = metrics['primary']
            best_scores = scores.copy()
    del model, optimizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_primary, best_scores, curve


def append_progress(path, record):
    with open(path, 'a') as fh:
        fh.write(json.dumps(record, sort_keys=True) + '\n')


def coarse_configs(seed):
    rng = np.random.RandomState(seed + 733)
    configs = []
    for i in range(4):
        configs.append({
            'dropout': float(rng.uniform(0.16, 0.38)),
            'history_dropout': float(rng.uniform(0.05, 0.30)),
            'weight_decay': float(10 ** rng.uniform(-4.5, -2.7)),
            'lr': float(10 ** rng.uniform(np.log10(4.5e-4), np.log10(1.25e-3))),
            'gamma': float([0.24, 0.34, 0.46, 0.58][i % 4]),
            'step_every': float([1.5, 2.0, 2.5][(i // 4) % 3]),
            'half_life': float([3.5, 7.0, 10.5, 14.0][(i * 3) % 4]),
            'aux_weight': float([0.03, 0.06, 0.10, 0.16][(i // 2) % 4]),
            'bce_mix': float([0.45, 0.55, 0.65][i % 3]),
        })
    return configs


def refine_configs(winner):
    variants = []
    changes = [
        (-0.04, -0.04, 0.60, 0.82, 0.85, 0.80),
        (-0.02, 0.00, 0.80, 0.92, 1.00, 0.85),
        (0.00, -0.05, 1.00, 0.82, 1.15, 1.00),
        (0.00, 0.00, 1.00, 1.00, 1.00, 1.00),
    ]
    for dd, hd, wd, lr, aux, half in changes:
        config = dict(winner)
        config['dropout'] = float(np.clip(config['dropout'] + dd, 0.10, 0.45))
        config['history_dropout'] = float(np.clip(config['history_dropout'] + hd, 0.02, 0.40))
        config['weight_decay'] = float(np.clip(config['weight_decay'] * wd, 1e-5, 5e-3))
        config['lr'] = float(np.clip(config['lr'] * lr, 3e-4, 1.8e-3))
        config['aux_weight'] = float(np.clip(config['aux_weight'] * aux, 0.015, 0.25))
        config['half_life'] = float(np.clip(config['half_life'] * half, 2.5, 18.0))
        variants.append(config)
    return variants


def select_config(configs, history, stage):
    best_index, best_mean = 0, -1.0
    for index in range(len(configs)):
        values = [entry['primary'] for entry in history
                  if entry['stage'] == stage and entry['config_index'] == index]
        value = float(np.mean(values)) if values else -1.0
        if value > best_mean:
            best_index, best_mean = index, value
    return configs[best_index], best_index, best_mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=14)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    progress_path = os.path.join(args.out_dir, 'progress.log')
    if os.path.exists(progress_path):
        os.remove(progress_path)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        device = torch.device('cpu')
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    data = build_sequence_features(load_data(args.data_dir), 12)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, root)
    if data['fast']:
        from data.official.evaluate import evaluate as evaluate_fn
    else:
        from harness.evaluate_provisional import evaluate as evaluate_fn

    pairs = make_pairs(data['train_user'], data['yt'], args.seed + 71)

    def capped(value):
        return max(1, min(value, 2))

    history = []
    coarse = coarse_configs(args.seed)[:2]
    coarse_seeds = [args.seed]
    for config_index, config in enumerate(coarse):
        for probe_seed in coarse_seeds:
            started = time.time()
            primary, _, curve = train_once(data, pairs, config, probe_seed,
                                           capped(2), evaluate_fn, device)
            record = {'stage': 'coarse', 'config_index': config_index,
                      'seed': probe_seed, 'config': config, 'primary': primary,
                      'epochs': capped(2), 'seconds': round(time.time() - started, 3),
                      'curve': curve}
            history.append(record)
            append_progress(progress_path, {k: v for k, v in record.items() if k != 'curve'})

    coarse_winner, coarse_index, coarse_mean = select_config(coarse, history, 'coarse')
    refined = refine_configs(coarse_winner)[:2]
    refine_seeds = [args.seed + 1]
    for config_index, config in enumerate(refined):
        for probe_seed in refine_seeds:
            started = time.time()
            primary, _, curve = train_once(data, pairs, config, probe_seed,
                                           capped(2), evaluate_fn, device)
            record = {'stage': 'refine', 'config_index': config_index,
                      'seed': probe_seed, 'config': config, 'primary': primary,
                      'epochs': capped(2), 'seconds': round(time.time() - started, 3),
                      'curve': curve}
            history.append(record)
            append_progress(progress_path, {k: v for k, v in record.items() if k != 'curve'})

    final_config, refine_index, refine_mean = select_config(refined, history, 'refine')
    final_seeds = [args.seed + 2]
    final_scores = []
    for final_seed in final_seeds:
        started = time.time()
        primary, scores, curve = train_once(data, pairs, final_config, final_seed,
                                            capped(args.epochs), evaluate_fn, device)
        final_scores.append(scores)
        record = {'stage': 'final', 'seed': final_seed, 'config': final_config,
                  'primary': primary, 'epochs': capped(args.epochs),
                  'seconds': round(time.time() - started, 3), 'curve': curve}
        history.append(record)
        append_progress(progress_path, {k: v for k, v in record.items() if k != 'curve'})

    chosen_scores = np.mean(np.stack(final_scores), axis=0)
    chosen_metrics = metric_dict(evaluate_fn, data['val_user'], data['yv'], chosen_scores)
    history.append({'stage': 'ensemble_close', 'member_count': len(final_scores),
                    'method': 'mean_logits', 'metrics': chosen_metrics})
    output = {
        'gauc': chosen_metrics['gauc'],
        'ndcg5': chosen_metrics['ndcg5'],
        'primary': chosen_metrics['primary'],
        'selected_output': 'mean_logits_' + str(len(final_scores)) + '_members',
        'selected_config': final_config,
        'search_summary': {
            'coarse_winner_index': coarse_index,
            'coarse_winner_mean_primary': coarse_mean,
            'refine_winner_index': refine_index,
            'refine_winner_mean_primary': refine_mean,
            'pair_count': int(len(pairs[0])),
            'history_length': 12,
        },
        'history': history,
    }
    with open(os.path.join(args.out_dir, 'metrics.json'), 'w') as fh:
        json.dump(output, fh)
    with open(os.path.join(args.out_dir, 'predictions.csv'), 'w') as fh:
        fh.write('row_id,user_id,video_id,score\n')
        for i, score in enumerate(chosen_scores):
            fh.write(f'{i},{data["val_user"][i]},{data["val_video"][i]},{score:.9g}\n')


if __name__ == '__main__':
    main()
