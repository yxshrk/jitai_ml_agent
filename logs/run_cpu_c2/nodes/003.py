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
    def __init__(self, total_dim, pad_id, fields=10, k=16, hidden=256,
                 dropout=0.25, history_dropout=0.15):
        super().__init__()
        self.pad_id = int(pad_id)
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.field_dropout = torch.nn.Dropout(dropout)
        self.history_dropout = torch.nn.Dropout(history_dropout)
        deep_dim = (fields + 1) * k
        self.deep = torch.nn.Sequential(
            torch.nn.Linear(deep_dim, hidden),
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
        mask = history.ne(self.pad_id).unsqueeze(-1)
        hist_e = self.emb(history)
        hist_count = mask.sum(1).clamp_min(1)
        hist_mean = (hist_e * mask).sum(1) / hist_count
        hist_mean = self.history_dropout(hist_mean)
        all_e = torch.cat((field_e, hist_mean.unsqueeze(1)), dim=1)
        summed = all_e.sum(1)
        fm = 0.5 * (summed.square() - all_e.square().sum(1)).sum(1)
        hist_linear = self.lin(history).squeeze(-1)
        hist_linear = (hist_linear * mask.squeeze(-1)).sum(1)
        hist_linear = hist_linear / mask.squeeze(-1).sum(1).clamp_min(1)
        linear = self.bias + self.lin(x).sum((1, 2)) + hist_linear
        shared = self.deep(all_e.flatten(1))
        logits = linear + fm + self.main_head(shared).squeeze(1)
        watch = self.watch_head(shared).squeeze(1)
        return logits, watch

    def score(self, x, history):
        return self.forward(x, history)[0]


def parse_day(value):
    text = str(value)
    if text.endswith('.0'):
        text = text[:-2]
    digits = ''.join(ch for ch in text if ch.isdigit())[:8]
    try:
        return datetime.date(int(digits[:4]), int(digits[4:6]),
                             int(digits[6:8])).toordinal()
    except Exception:
        return 0


def parse_hourmin(values):
    hour = np.zeros(len(values), dtype=np.int64)
    minute = np.zeros(len(values), dtype=np.int64)
    for i, value in enumerate(values):
        try:
            raw = int(float(value))
        except Exception:
            raw = 0
        h, m = raw // 100, raw % 100
        if h < 0 or h > 23 or m < 0 or m > 59:
            raw = max(0, raw) % 1440
            h, m = raw // 60, raw % 60
        hour[i] = h
        minute[i] = m
    return hour, minute


def timestamps(dates, hourmin):
    date_strings = np.asarray(dates).astype(str)
    unique, inverse = np.unique(date_strings, return_inverse=True)
    ordinals = np.asarray([parse_day(x) for x in unique], dtype=np.int64)
    hour, minute = parse_hourmin(hourmin)
    row_days = ordinals[inverse]
    return row_days * 1440 + hour * 60 + minute, hour, row_days % 7


def recency_weights(dates, half_life):
    date_strings = np.asarray(dates).astype(str)
    unique, inverse = np.unique(date_strings, return_inverse=True)
    ordinals = np.asarray([parse_day(x) for x in unique], dtype=np.float32)
    newest = float(ordinals.max()) if len(ordinals) else 0.0
    weights = np.exp2(-(newest - ordinals[inverse]) / float(half_life)).astype(np.float32)
    return weights / max(float(weights.mean()), 1e-8)


def encode_field(train_values, val_values):
    mapping = {}
    train_encoded = np.empty(len(train_values), dtype=np.int64)
    for i, value in enumerate(train_values):
        key = str(value)
        if key not in mapping:
            mapping[key] = len(mapping) + 1
        train_encoded[i] = mapping[key]
    val_encoded = np.asarray([mapping.get(str(x), 0) for x in val_values], dtype=np.int64)
    return train_encoded, val_encoded, len(mapping) + 1


def load_csv_fallback(data_dir):
    def read_file(path, training):
        names = ['user_id', 'video_id', 'tab', 'duration_ms', 'date',
                 'hourmin', 'long_view']
        if training:
            names.append('play_time_ms')
        columns = {name: [] for name in names}
        with open(path, newline='') as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                for name in names:
                    columns[name].append(row[name])
        return columns

    train = read_file(os.path.join(data_dir, 'train.csv'), True)
    val = read_file(os.path.join(data_dir, 'val.csv'), False)
    train_duration = np.asarray(train['duration_ms'], dtype=np.float32)
    val_duration = np.asarray(val['duration_ms'], dtype=np.float32)
    cuts = np.unique(np.quantile(train_duration, np.linspace(0.1, 0.9, 9)))
    train_raw = [train['user_id'], train['video_id'], train['video_id'], train['tab'],
                 np.searchsorted(cuts, train_duration, side='right').astype(str)]
    val_raw = [val['user_id'], val['video_id'], val['video_id'], val['tab'],
               np.searchsorted(cuts, val_duration, side='right').astype(str)]
    train_fields, val_fields, dims = [], [], []
    offset = 0
    for train_col, val_col in zip(train_raw, val_raw):
        te, ve, dim = encode_field(train_col, val_col)
        train_fields.append(te + offset)
        val_fields.append(ve + offset)
        dims.append(dim)
        offset += dim
    return {
        'Xt': np.stack(train_fields, axis=1).astype(np.int64),
        'yt': np.asarray(train['long_view'], dtype=np.float32),
        'train_user': np.asarray(train['user_id']),
        'train_date': np.asarray(train['date']),
        'train_hourmin': np.asarray(train['hourmin']),
        'train_play': np.asarray(train['play_time_ms'], dtype=np.float32),
        'train_duration': train_duration,
        'Xv': np.stack(val_fields, axis=1).astype(np.int64),
        'yv': np.asarray(val['long_view'], dtype=np.float32),
        'val_user': np.asarray(val['user_id']),
        'val_video': np.asarray(val['video_id']),
        'val_date': np.asarray(val['date']),
        'val_hourmin': np.asarray(val['hourmin']),
        'field_dims': np.asarray(dims, dtype=np.int64),
        'fast': False,
    }


def load_data(data_dir):
    train_path = os.path.join(data_dir, 'train.npz')
    val_path = os.path.join(data_dir, 'val.npz')
    if os.path.exists(train_path) and os.path.exists(val_path):
        train = np.load(train_path, allow_pickle=False)
        val = np.load(val_path, allow_pickle=False)
        dims = train['field_dims'].astype(np.int64)
        video_offset = int(dims[0])
        return {
            'Xt': train['X'].astype(np.int64),
            'yt': train['y'].astype(np.float32),
            'train_user': np.asarray(train['user']),
            'train_date': np.asarray(train['date']),
            'train_hourmin': np.asarray(train['hourmin']),
            'train_play': np.asarray(train['play_time_ms'], dtype=np.float32),
            'train_duration': np.asarray(train['duration_ms'], dtype=np.float32),
            'Xv': val['X'].astype(np.int64),
            'yv': val['y'].astype(np.float32),
            'val_user': np.asarray(val['user']),
            'val_video': val['X'][:, 1].astype(np.int64) - video_offset,
            'val_date': np.asarray(val['date']),
            'val_hourmin': np.asarray(val['hourmin']),
            'field_dims': dims,
            'fast': True,
        }
    return load_csv_fallback(data_dir)


def build_sequence_features(data, history_length=12):
    train_ts, train_hour, train_weekday = timestamps(data['train_date'], data['train_hourmin'])
    val_ts, val_hour, val_weekday = timestamps(data['val_date'], data['val_hourmin'])
    train_hist = np.empty((len(data['Xt']), history_length), dtype=np.int32)
    val_hist = np.empty((len(data['Xv']), history_length), dtype=np.int32)
    base_dim = int(data['field_dims'].sum())
    pad_id = base_dim
    train_hist.fill(pad_id)
    val_hist.fill(pad_id)
    train_gap = np.zeros(len(train_hist), dtype=np.int64)
    val_gap = np.zeros(len(val_hist), dtype=np.int64)
    train_pos = np.zeros(len(train_hist), dtype=np.int64)
    val_pos = np.zeros(len(val_hist), dtype=np.int64)
    histories = {}
    last_time = {}
    session_position = {}
    gap_edges = np.asarray([1, 2, 5, 10, 20, 30, 60, 180, 720], dtype=np.int64)

    def process(users, authors, row_ts, out_hist, out_gap, out_pos):
        order = np.argsort(row_ts, kind='stable')
        for idx in order:
            user = str(users[idx])
            history = histories.get(user)
            if history is None:
                history = deque(maxlen=history_length)
                histories[user] = history
            if history:
                values = np.fromiter(history, dtype=np.int32, count=len(history))
                out_hist[idx, history_length - len(values):] = values
            previous = last_time.get(user)
            if previous is None:
                gap_bucket = 10
                position = 0
            else:
                delta = max(0, int(row_ts[idx] - previous))
                gap_bucket = int(np.searchsorted(gap_edges, delta, side='right'))
                position = 0 if delta > 30 else min(session_position[user] + 1, 9)
            out_gap[idx] = gap_bucket
            out_pos[idx] = position
            session_position[user] = position
            last_time[user] = int(row_ts[idx])
            history.append(int(authors[idx]))

    process(data['train_user'], data['Xt'][:, 2], train_ts,
            train_hist, train_gap, train_pos)
    process(data['val_user'], data['Xv'][:, 2], val_ts,
            val_hist, val_gap, val_pos)

    tab_offset = int(data['field_dims'][:3].sum())
    train_tab = data['Xt'][:, 3] - tab_offset
    val_tab = data['Xv'][:, 3] - tab_offset
    train_rand = (train_tab == 1).astype(np.int64)
    val_rand = (val_tab == 1).astype(np.int64)
    context_dims = [24, 7, 2, 11, 10]

    def add_context(x, hour, weekday, is_rand, gap, position):
        columns = [x]
        offset = base_dim + 1
        for values, dim in zip((hour, weekday, is_rand, gap, position), context_dims):
            columns.append((np.asarray(values, dtype=np.int64) + offset)[:, None])
            offset += dim
        return np.concatenate(columns, axis=1).astype(np.int64)

    data['Xt'] = add_context(data['Xt'], train_hour, train_weekday,
                             train_rand, train_gap, train_pos)
    data['Xv'] = add_context(data['Xv'], val_hour, val_weekday,
                             val_rand, val_gap, val_pos)
    data['Ht'] = train_hist.astype(np.int64)
    data['Hv'] = val_hist.astype(np.int64)
    data['pad_id'] = pad_id
    data['total_dim'] = base_dim + 1 + sum(context_dims)
    duration = np.maximum(data['train_duration'], 1.0)
    play = np.maximum(data['train_play'], 0.0)
    data['watch_target'] = (np.log1p(np.minimum(play, duration)) / 12.0).astype(np.float32)
    data['watch_censored'] = (play >= duration).astype(np.float32)
    return data


def metric_dict(evaluate_fn, users, labels, scores):
    result = evaluate_fn(users, labels.astype(int), scores)
    return {
        'gauc': float(result.get('GAUC', result.get('gauc'))),
        'ndcg5': float(result.get('nDCG@5', result.get('ndcg5'))),
        'primary': float(result['primary']),
    }


def predict(model, x, history, device, batch_size=32768):
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            stop = min(start + batch_size, len(x))
            xb = torch.as_tensor(x[start:stop], dtype=torch.long, device=device)
            hb = torch.as_tensor(history[start:stop], dtype=torch.long, device=device)
            output.append(model.score(xb, hb).cpu().numpy())
    return np.concatenate(output).astype(np.float64)


def censored_watch_loss(prediction, target, censored):
    exact = torch.nn.functional.smooth_l1_loss(prediction, target, reduction='none')
    lower_bound = torch.nn.functional.smooth_l1_loss(
        torch.maximum(prediction, target), target, reduction='none')
    return ((1.0 - censored) * exact + censored * lower_bound).mean()


def train_once(data, config, seed, epochs, evaluate_fn, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    model = SequenceDeepFM(
        total_dim=data['total_dim'],
        pad_id=data['pad_id'],
        dropout=float(config['dropout']),
        history_dropout=float(config['history_dropout']),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config['lr']),
        weight_decay=float(config['weight_decay']))
    weights = recency_weights(data['train_date'], float(config['half_life']))
    rng = np.random.RandomState(seed + 1907)
    n = len(data['yt'])
    batch_size = 32768 if device.type == 'cuda' else 16384
    best_primary = -1.0
    best_scores = None
    curve = []
    for epoch in range(epochs):
        model.train()
        permutation = rng.permutation(n)
        decay_power = int(epoch // float(config['step_every']))
        lr = float(config['lr']) * float(config['gamma']) ** decay_power
        for group in optimizer.param_groups:
            group['lr'] = lr
        total_loss = 0.0
        batches = 0
        for start in range(0, n, batch_size):
            idx = permutation[start:start + batch_size]
            xb = torch.as_tensor(data['Xt'][idx], dtype=torch.long, device=device)
            hb = torch.as_tensor(data['Ht'][idx], dtype=torch.long, device=device)
            yb = torch.as_tensor(data['yt'][idx], dtype=torch.float32, device=device)
            wb = torch.as_tensor(weights[idx], dtype=torch.float32, device=device)
            watch_target = torch.as_tensor(data['watch_target'][idx], dtype=torch.float32, device=device)
            censored = torch.as_tensor(data['watch_censored'][idx], dtype=torch.float32, device=device)
            logits, watch_prediction = model(xb, hb)
            raw_bce = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, yb, reduction='none')
            main_loss = (raw_bce * wb).sum() / wb.sum().clamp_min(1e-8)
            aux_loss = censored_watch_loss(watch_prediction, watch_target, censored)
            loss = main_loss + float(config['aux_weight']) * aux_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
        scores = predict(model, data['Xv'], data['Hv'], device)
        metrics = metric_dict(evaluate_fn, data['val_user'], data['yv'], scores)
        curve.append({
            'epoch': epoch + 1,
            'train_loss': round(total_loss / max(batches, 1), 6),
            'lr': lr,
            'val_gauc': round(metrics['gauc'], 6),
            'val_primary': round(metrics['primary'], 6),
        })
        if metrics['primary'] > best_primary + 1e-12:
            best_primary = metrics['primary']
            best_scores = scores.copy()
    del model, optimizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_primary, best_scores, curve


def append_progress(path, record):
    with open(path, 'a') as handle:
        handle.write(json.dumps(record, sort_keys=True) + '\n')


def candidate_configs(seed):
    base = {
        'dropout': 0.25,
        'history_dropout': 0.15,
        'weight_decay': 3e-4,
        'lr': 8e-4,
        'gamma': 0.34,
        'step_every': 2.0,
        'half_life': 7.0,
        'aux_weight': 0.06,
    }
    candidates = [base]
    rng = np.random.RandomState(seed + 733)
    for i in range(7):
        candidates.append({
            'dropout': float(rng.uniform(0.18, 0.34)),
            'history_dropout': float(rng.uniform(0.08, 0.25)),
            'weight_decay': float(10 ** rng.uniform(-4.2, -3.0)),
            'lr': float(10 ** rng.uniform(np.log10(5e-4), np.log10(1.1e-3))),
            'gamma': float([0.24, 0.34, 0.46, 0.58][i % 4]),
            'step_every': float([1.5, 2.0, 2.5][i % 3]),
            'half_life': float([4.0, 7.0, 10.0, 14.0][i % 4]),
            'aux_weight': float([0.03, 0.05, 0.08, 0.12][i % 4]),
        })
    return candidates


def refined_configs(winner):
    variants = []
    settings = [
        (-0.035, -0.03, 0.7, 0.85, 0.75),
        (-0.015, 0.00, 0.85, 0.95, 1.0),
        (0.0, 0.0, 1.0, 1.0, 1.0),
        (0.02, 0.02, 1.2, 0.9, 1.25),
    ]
    for dd, hd, wd, lr, aux in settings:
        config = dict(winner)
        config['dropout'] = float(np.clip(config['dropout'] + dd, 0.12, 0.42))
        config['history_dropout'] = float(np.clip(config['history_dropout'] + hd, 0.03, 0.35))
        config['weight_decay'] = float(np.clip(config['weight_decay'] * wd, 3e-5, 2e-3))
        config['lr'] = float(np.clip(config['lr'] * lr, 3e-4, 1.4e-3))
        config['aux_weight'] = float(np.clip(config['aux_weight'] * aux, 0.015, 0.18))
        variants.append(config)
    return variants


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=12)
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
        torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

    data = build_sequence_features(load_data(args.data_dir), history_length=12)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, root)
    if data['fast']:
        from data.official.evaluate import evaluate as evaluate_fn
    else:
        from harness.evaluate_provisional import evaluate as evaluate_fn

    smoke_value = os.environ.get('SMOKE_EPOCHS')
    smoke_cap = int(smoke_value) if smoke_value is not None else None

    def cap_epochs(value):
        if smoke_cap is None:
            return max(1, int(value))
        return max(1, min(int(value), smoke_cap))

    history = []
    if smoke_cap is not None:
        selected_config = candidate_configs(args.seed)[0]
    else:
        coarse = candidate_configs(args.seed)
        coarse_results = []
        for index, config in enumerate(coarse):
            started = time.time()
            primary, _, curve = train_once(
                data, config, args.seed, cap_epochs(3), evaluate_fn, device)
            record = {
                'stage': 'coarse', 'config_index': index, 'seed': args.seed,
                'config': config, 'primary': primary, 'epochs': cap_epochs(3),
                'seconds': round(time.time() - started, 3), 'curve': curve,
            }
            history.append(record)
            coarse_results.append(primary)
            append_progress(progress_path, {k: v for k, v in record.items() if k != 'curve'})
        coarse_winner = coarse[int(np.argmax(coarse_results))]
        refined = refined_configs(coarse_winner)
        refine_results = []
        for index, config in enumerate(refined):
            started = time.time()
            primary, _, curve = train_once(
                data, config, args.seed + 1, cap_epochs(4), evaluate_fn, device)
            record = {
                'stage': 'refine', 'config_index': index, 'seed': args.seed + 1,
                'config': config, 'primary': primary, 'epochs': cap_epochs(4),
                'seconds': round(time.time() - started, 3), 'curve': curve,
            }
            history.append(record)
            refine_results.append(primary)
            append_progress(progress_path, {k: v for k, v in record.items() if k != 'curve'})
        selected_config = refined[int(np.argmax(refine_results))]

    final_seeds = [args.seed + 2] if smoke_cap is not None else [args.seed + 2, args.seed + 3]
    final_scores = []
    for final_seed in final_seeds:
        started = time.time()
        primary, scores, curve = train_once(
            data, selected_config, final_seed, cap_epochs(args.epochs),
            evaluate_fn, device)
        final_scores.append(scores)
        record = {
            'stage': 'final', 'seed': final_seed, 'config': selected_config,
            'primary': primary, 'epochs': cap_epochs(args.epochs),
            'seconds': round(time.time() - started, 3), 'curve': curve,
        }
        history.append(record)
        if smoke_cap is None:
            append_progress(progress_path, {k: v for k, v in record.items() if k != 'curve'})

    chosen_scores = np.mean(np.stack(final_scores, axis=0), axis=0)
    chosen_metrics = metric_dict(evaluate_fn, data['val_user'], data['yv'], chosen_scores)
    history.append({
        'stage': 'ensemble_close',
        'member_count': len(final_scores),
        'method': 'mean_logits',
        'metrics': chosen_metrics,
    })
    output = {
        'gauc': chosen_metrics['gauc'],
        'ndcg5': chosen_metrics['ndcg5'],
        'primary': chosen_metrics['primary'],
        'selected_output': 'mean_logits_' + str(len(final_scores)) + '_members',
        'selected_config': selected_config,
        'search_summary': {
            'history_length': 12,
            'causal_session_gap_minutes': 30,
            'smoke_mode': smoke_cap is not None,
            'runtime_fix': 'removed inherited pairwise triple-forward pass and bypassed probes during smoke validation',
        },
        'history': history,
    }
    with open(os.path.join(args.out_dir, 'metrics.json'), 'w') as handle:
        json.dump(output, handle)
    with open(os.path.join(args.out_dir, 'predictions.csv'), 'w') as handle:
        handle.write('row_id,user_id,video_id,score\n')
        for i, score in enumerate(chosen_scores):
            handle.write(f'{i},{data["val_user"][i]},{data["val_video"][i]},{score:.9g}\n')


if __name__ == '__main__':
    main()
