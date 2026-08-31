import argparse
import csv
import datetime
import gc
import json
import math
import os
import sys
from collections import deque

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def date_ordinal(value):
    s = str(value)
    if s.endswith('.0'):
        s = s[:-2]
    digits = ''.join(ch for ch in s if ch.isdigit())
    if len(digits) >= 8:
        try:
            return datetime.date(int(digits[:4]), int(digits[4:6]), int(digits[6:8])).toordinal()
        except ValueError:
            pass
    return datetime.date(2022, 4, 21).toordinal()


def parse_hourmin(value):
    try:
        v = int(float(value))
    except (TypeError, ValueError):
        return 0, 0
    hour = max(0, min(23, v // 100))
    minute = max(0, min(59, v % 100))
    return hour, minute


def temporal_arrays(dates, hourmins):
    n = len(dates)
    ordinals = np.empty(n, dtype=np.int64)
    hours = np.empty(n, dtype=np.int64)
    timestamps = np.empty(n, dtype=np.int64)
    weekdays = np.empty(n, dtype=np.int64)
    for i in range(n):
        ordinal = date_ordinal(dates[i])
        hour, minute = parse_hourmin(hourmins[i])
        ordinals[i] = ordinal
        hours[i] = hour
        weekdays[i] = (ordinal - 1) % 7
        timestamps[i] = ordinal * 1440 + hour * 60 + minute
    return ordinals, hours, weekdays, timestamps


def load_npz(data_dir):
    tr = np.load(os.path.join(data_dir, 'train.npz'))
    va = np.load(os.path.join(data_dir, 'val.npz'))
    field_dims = tr['field_dims'].astype(np.int64)
    xtr = tr['X'].astype(np.int64)
    xva = va['X'].astype(np.int64)
    ytr = tr['y'].astype(np.float32)
    yva = va['y'].astype(np.float32)
    utr = tr['user']
    uva = va['user']
    dates_tr = tr['date'] if 'date' in tr.files else np.zeros(len(ytr), dtype=np.int64)
    dates_va = va['date'] if 'date' in va.files else np.zeros(len(yva), dtype=np.int64)
    hm_tr = tr['hourmin'] if 'hourmin' in tr.files else np.zeros(len(ytr), dtype=np.int64)
    hm_va = va['hourmin'] if 'hourmin' in va.files else np.zeros(len(yva), dtype=np.int64)
    play = tr['play_time_ms'].astype(np.float32) if 'play_time_ms' in tr.files else np.zeros(len(ytr), dtype=np.float32)
    duration = tr['duration_ms'].astype(np.float32) if 'duration_ms' in tr.files else np.ones(len(ytr), dtype=np.float32)
    video_offset = int(field_dims[0])
    vva = xva[:, 1] - video_offset
    return {
        'xtr': xtr, 'xva': xva, 'ytr': ytr, 'yva': yva,
        'utr': utr, 'uva': uva, 'vva': vva, 'dates_tr': dates_tr,
        'dates_va': dates_va, 'hm_tr': hm_tr, 'hm_va': hm_va,
        'play': play, 'duration': duration, 'field_dims': field_dims,
        'pair_cache': {}
    }


def read_csv_rows(path, training):
    rows = []
    with open(path, 'r', newline='') as fh:
        for row in csv.DictReader(fh):
            item = {
                'user': row['user_id'],
                'video': row['video_id'],
                'tab': row.get('tab', ''),
                'duration': float(row.get('duration_ms', 0) or 0),
                'date': row.get('date', ''),
                'hourmin': row.get('hourmin', 0),
                'y': float(row['long_view'])
            }
            if training:
                item['play'] = float(row.get('play_time_ms', 0) or 0)
            rows.append(item)
    return rows


def make_mapping(values):
    return {v: i for i, v in enumerate(sorted(set(values)))}


def load_csv(data_dir):
    tr_rows = read_csv_rows(os.path.join(data_dir, 'train.csv'), True)
    va_rows = read_csv_rows(os.path.join(data_dir, 'val.csv'), False)
    user_map = make_mapping([r['user'] for r in tr_rows])
    video_map = make_mapping([r['video'] for r in tr_rows])
    tab_map = make_mapping([r['tab'] for r in tr_rows])
    durations = np.asarray([r['duration'] for r in tr_rows], dtype=np.float64)
    cuts = np.unique(np.quantile(durations, np.linspace(0.1, 0.9, 9))) if len(durations) else np.array([])
    field_dims = np.asarray([
        len(user_map) + 1, len(video_map) + 1, 1,
        len(tab_map) + 1, len(cuts) + 1
    ], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1])).astype(np.int64)

    def encode(rows, training):
        n = len(rows)
        x = np.empty((n, 5), dtype=np.int64)
        y = np.empty(n, dtype=np.float32)
        users = np.empty(n, dtype=object)
        videos = np.empty(n, dtype=object)
        dates = np.empty(n, dtype=object)
        hourmins = np.empty(n, dtype=object)
        play = np.zeros(n, dtype=np.float32)
        duration = np.empty(n, dtype=np.float32)
        for i, r in enumerate(rows):
            vals = [
                user_map.get(r['user'], len(user_map)),
                video_map.get(r['video'], len(video_map)),
                0,
                tab_map.get(r['tab'], len(tab_map)),
                int(np.searchsorted(cuts, r['duration'], side='right'))
            ]
            x[i] = np.asarray(vals, dtype=np.int64) + offsets
            y[i] = r['y']
            users[i] = r['user']
            videos[i] = r['video']
            dates[i] = r['date']
            hourmins[i] = r['hourmin']
            duration[i] = r['duration']
            if training:
                play[i] = r['play']
        return x, y, users, videos, dates, hourmins, play, duration

    xtr, ytr, utr, _, dtr, htr, play, duration = encode(tr_rows, True)
    xva, yva, uva, vva, dva, hva, _, _ = encode(va_rows, False)
    return {
        'xtr': xtr, 'xva': xva, 'ytr': ytr, 'yva': yva,
        'utr': utr, 'uva': uva, 'vva': vva, 'dates_tr': dtr,
        'dates_va': dva, 'hm_tr': htr, 'hm_va': hva, 'play': play,
        'duration': duration, 'field_dims': field_dims, 'pair_cache': {}
    }


def build_sequence_context(data, history_len=12):
    _, hour_tr, weekday_tr, time_tr = temporal_arrays(data['dates_tr'], data['hm_tr'])
    _, hour_va, weekday_va, time_va = temporal_arrays(data['dates_va'], data['hm_va'])
    tab_offset = int(np.sum(data['field_dims'][:3]))
    tab_tr = data['xtr'][:, 3] - tab_offset
    tab_va = data['xva'][:, 3] - tab_offset
    rand_tr = (tab_tr == 1).astype(np.int64)
    rand_va = (tab_va == 1).astype(np.int64)
    gap_cuts = np.asarray([1, 5, 15, 30, 60, 180, 720, 1440], dtype=np.int64)
    context_dims = np.asarray([24, 7, 2, 10, 16], dtype=np.int64)
    context_offsets = np.concatenate(([0], np.cumsum(context_dims)[:-1])).astype(np.int64)
    histories = {}
    last_times = {}
    session_positions = {}

    def process(users, times, hours, weekdays, random_flags, authors):
        n = len(users)
        hist = np.full((n, history_len), -1, dtype=np.int64)
        ctx = np.empty((n, 5), dtype=np.int64)
        index = np.arange(n, dtype=np.int64)
        order = np.lexsort((index, times, users))
        for idx in order:
            user = users[idx].item() if isinstance(users[idx], np.generic) else users[idx]
            previous = last_times.get(user)
            if previous is None:
                gap_bucket = 0
                position = 0
            else:
                gap = max(0, int(times[idx]) - int(previous))
                gap_bucket = 1 + int(np.searchsorted(gap_cuts, gap, side='right'))
                gap_bucket = min(gap_bucket, 9)
                if gap > 30:
                    position = 0
                else:
                    position = min(int(session_positions.get(user, 0)) + 1, 15)
            prior = histories.get(user)
            if prior is not None and len(prior):
                values = list(prior)
                hist[idx, history_len - len(values):] = values
            ctx[idx] = np.asarray([
                int(hours[idx]), int(weekdays[idx]), int(random_flags[idx]),
                gap_bucket, position
            ], dtype=np.int64) + context_offsets
            if prior is None:
                prior = deque(maxlen=history_len)
                histories[user] = prior
            prior.append(int(authors[idx]))
            last_times[user] = int(times[idx])
            session_positions[user] = position
        return hist, ctx

    author_tr = data['xtr'][:, 2]
    author_va = data['xva'][:, 2]
    hist_tr, ctx_tr = process(data['utr'], time_tr, hour_tr, weekday_tr, rand_tr, author_tr)
    hist_va, ctx_va = process(data['uva'], time_va, hour_va, weekday_va, rand_va, author_va)
    data['hist_tr'] = hist_tr
    data['hist_va'] = hist_va
    data['ctx_tr'] = ctx_tr
    data['ctx_va'] = ctx_va
    data['context_dims'] = context_dims


def recency_weights(dates, half_life=7.0):
    anchor = datetime.date(2022, 4, 21).toordinal()
    ordinals = np.fromiter((date_ordinal(v) for v in dates), dtype=np.int64, count=len(dates))
    age = np.maximum(0, anchor - ordinals).astype(np.float32)
    weights = np.exp2(-age / float(half_life)).astype(np.float32)
    weights /= max(float(weights.mean()), 1e-6)
    return weights


def make_pairs(users, labels, seed):
    rng = np.random.RandomState(seed)
    order = np.argsort(users, kind='mergesort')
    pos_parts = []
    neg_parts = []
    start = 0
    while start < len(order):
        end = start + 1
        user = users[order[start]]
        while end < len(order) and users[order[end]] == user:
            end += 1
        group = order[start:end]
        pos = group[labels[group] > 0.5]
        neg = group[labels[group] <= 0.5]
        if len(pos) and len(neg):
            chosen = neg.copy()
            rng.shuffle(chosen)
            pos_parts.append(pos.astype(np.int64))
            neg_parts.append(chosen[np.arange(len(pos)) % len(chosen)].astype(np.int64))
        start = end
    if not pos_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(pos_parts), np.concatenate(neg_parts)


class SequenceDeepFM(torch.nn.Module):
    def __init__(self, total_base, total_context, embedding_dim, dropout):
        super().__init__()
        self.base_embedding = torch.nn.Embedding(total_base, embedding_dim)
        self.context_embedding = torch.nn.Embedding(total_context, embedding_dim)
        self.base_linear = torch.nn.Embedding(total_base, 1)
        self.context_linear = torch.nn.Embedding(total_context, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.embedding_dropout = torch.nn.Dropout(dropout)
        width = 11 * embedding_dim
        self.deep1 = torch.nn.Linear(width, 128)
        self.deep2 = torch.nn.Linear(128, 64)
        self.dropout = torch.nn.Dropout(dropout)
        self.main_head = torch.nn.Linear(64, 1, bias=False)
        self.watch_head = torch.nn.Linear(64, 1)
        torch.nn.init.normal_(self.base_embedding.weight, std=0.01)
        torch.nn.init.normal_(self.context_embedding.weight, std=0.01)
        torch.nn.init.zeros_(self.base_linear.weight)
        torch.nn.init.zeros_(self.context_linear.weight)
        torch.nn.init.xavier_uniform_(self.deep1.weight)
        torch.nn.init.xavier_uniform_(self.deep2.weight)
        torch.nn.init.xavier_uniform_(self.main_head.weight)
        torch.nn.init.xavier_uniform_(self.watch_head.weight)

    def forward(self, x, context, history):
        base = self.base_embedding(x)
        ctx = self.context_embedding(context)
        mask = history.ge(0)
        safe_history = history.clamp_min(0)
        history_vectors = self.base_embedding(safe_history) * mask.unsqueeze(-1)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(history_vectors.dtype)
        pooled = history_vectors.sum(dim=1) / denom
        fields = torch.cat([base, ctx, pooled.unsqueeze(1)], dim=1)
        fields = self.embedding_dropout(fields)
        summed = fields.sum(dim=1)
        fm = 0.5 * (summed.square() - fields.square().sum(dim=1)).sum(dim=1)
        linear = self.base_linear(x).sum(dim=(1, 2)) + self.context_linear(context).sum(dim=(1, 2))
        flat = fields.reshape(fields.shape[0], -1)
        hidden = self.dropout(torch.relu(self.deep1(flat)))
        hidden = self.dropout(torch.relu(self.deep2(hidden)))
        main = self.bias + linear + fm + self.main_head(hidden).squeeze(1)
        watch = self.watch_head(hidden).squeeze(1)
        return main, watch


def get_metrics(evaluate_fn, users, labels, scores):
    result = evaluate_fn(users, labels.astype(int), scores)
    return {
        'gauc': float(result['GAUC'] if 'GAUC' in result else result['gauc']),
        'ndcg5': float(result['nDCG@5'] if 'nDCG@5' in result else result['ndcg5']),
        'primary': float(result['primary'])
    }


def predict(model, x, context, history, device, batch_size=65536):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = start + batch_size
            xb = torch.from_numpy(x[start:end]).to(device)
            cb = torch.from_numpy(context[start:end]).to(device)
            hb = torch.from_numpy(history[start:end]).to(device)
            logits, _ = model(xb, cb, hb)
            outputs.append(logits.cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def watch_targets(play, duration):
    safe_duration = np.maximum(duration.astype(np.float32), 1.0)
    observed = np.minimum(np.maximum(play.astype(np.float32), 0.0), safe_duration)
    target = np.log1p(observed) / np.maximum(np.log1p(safe_duration), 1e-6)
    censored = ((duration > 0) & (play >= duration)).astype(np.float32)
    valid = (duration > 0).astype(np.float32)
    return target.astype(np.float32), censored, valid


def config_json(config):
    return {
        'dropout': round(float(config['dropout']), 7),
        'weight_decay': float(config['weight_decay']),
        'lr': float(config['lr']),
        'step_gamma': round(float(config['step_gamma']), 7),
        'aux_weight': round(float(config['aux_weight']), 7),
        'bpr_mix': round(float(config['bpr_mix']), 7)
    }


def append_progress(path, record):
    with open(path, 'a') as fh:
        fh.write(json.dumps(record, sort_keys=True) + '\n')


def train_one(data, config, epochs, seed, device, evaluate_fn, keep_trace):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    model = SequenceDeepFM(
        int(data['field_dims'].sum()), int(data['context_dims'].sum()), 16,
        float(config['dropout'])
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config['lr']), weight_decay=float(config['weight_decay'])
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=float(config['step_gamma']))
    weights = data['sample_weights']
    watch_target = data['watch_target']
    watch_censored = data['watch_censored']
    watch_valid = data['watch_valid']
    pair_key = int(seed + 991)
    if pair_key not in data['pair_cache']:
        data['pair_cache'][pair_key] = make_pairs(data['utr'], data['ytr'], pair_key)
    pair_pos, pair_neg = data['pair_cache'][pair_key]
    rng = np.random.RandomState(seed + 17)
    n = len(data['ytr'])
    batch_size = 8192 if device.type == 'cuda' else 4096
    best_primary = -1.0
    best_scores = None
    best_metrics = None
    trace = []
    for epoch in range(epochs):
        permutation = rng.permutation(n)
        pair_order = rng.permutation(len(pair_pos)) if len(pair_pos) else np.empty(0, dtype=np.int64)
        halves = np.array_split(permutation, 2)
        pair_halves = np.array_split(pair_order, 2) if len(pair_order) else [pair_order, pair_order]
        for half_index, indices in enumerate(halves):
            model.train()
            losses = []
            pair_indices = pair_halves[half_index]
            pair_cursor = 0
            for start in range(0, len(indices), batch_size):
                idx = indices[start:start + batch_size]
                xb = torch.from_numpy(data['xtr'][idx]).to(device)
                cb = torch.from_numpy(data['ctx_tr'][idx]).to(device)
                hb = torch.from_numpy(data['hist_tr'][idx]).to(device)
                yb = torch.from_numpy(data['ytr'][idx]).to(device)
                wb = torch.from_numpy(weights[idx]).to(device)
                target_b = torch.from_numpy(watch_target[idx]).to(device)
                censored_b = torch.from_numpy(watch_censored[idx]).to(device)
                valid_b = torch.from_numpy(watch_valid[idx]).to(device)
                logits, watch_prediction = model(xb, cb, hb)
                bce_each = torch.nn.functional.binary_cross_entropy_with_logits(logits, yb, reduction='none')
                bce_loss = (bce_each * wb).sum() / wb.sum().clamp_min(1e-6)
                exact_loss = torch.nn.functional.smooth_l1_loss(watch_prediction, target_b, reduction='none')
                lower_bound_loss = torch.relu(target_b - watch_prediction).square()
                aux_each = (1.0 - censored_b) * exact_loss + censored_b * lower_bound_loss
                aux_weighting = wb * valid_b
                aux_loss = (aux_each * aux_weighting).sum() / aux_weighting.sum().clamp_min(1e-6)
                bpr_mix = float(config['bpr_mix'])
                if bpr_mix > 0 and len(pair_indices):
                    count = min(len(idx), len(pair_indices))
                    if pair_cursor + count <= len(pair_indices):
                        selected = pair_indices[pair_cursor:pair_cursor + count]
                    else:
                        selected = np.resize(pair_indices, count)
                    pair_cursor += count
                    pi = pair_pos[selected]
                    ni = pair_neg[selected]
                    xp = torch.from_numpy(data['xtr'][pi]).to(device)
                    cp = torch.from_numpy(data['ctx_tr'][pi]).to(device)
                    hp = torch.from_numpy(data['hist_tr'][pi]).to(device)
                    xn = torch.from_numpy(data['xtr'][ni]).to(device)
                    cn = torch.from_numpy(data['ctx_tr'][ni]).to(device)
                    hn = torch.from_numpy(data['hist_tr'][ni]).to(device)
                    positive_logits, _ = model(xp, cp, hp)
                    negative_logits, _ = model(xn, cn, hn)
                    pair_weights = torch.from_numpy(0.5 * (weights[pi] + weights[ni])).to(device)
                    pair_each = torch.nn.functional.softplus(-(positive_logits - negative_logits))
                    pair_loss = (pair_each * pair_weights).sum() / pair_weights.sum().clamp_min(1e-6)
                    main_loss = (1.0 - bpr_mix) * bce_loss + bpr_mix * pair_loss
                else:
                    main_loss = bce_loss
                loss = main_loss + float(config['aux_weight']) * aux_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            scores = predict(model, data['xva'], data['ctx_va'], data['hist_va'], device)
            metrics = get_metrics(evaluate_fn, data['uva'], data['yva'], scores)
            if metrics['primary'] > best_primary:
                best_primary = metrics['primary']
                best_scores = scores.copy()
                best_metrics = metrics
            if keep_trace:
                trace.append({
                    'checkpoint': epoch + 0.5 * (half_index + 1),
                    'loss': round(float(np.mean(losses)) if losses else 0.0, 6),
                    'lr': float(optimizer.param_groups[0]['lr']),
                    **metrics
                })
        scheduler.step()
    del optimizer, scheduler, model
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_metrics, best_scores, trace


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
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        device = torch.device('cpu')

    fast = os.path.exists(os.path.join(args.data_dir, 'train.npz')) and os.path.exists(os.path.join(args.data_dir, 'val.npz'))
    if fast:
        from data.official.evaluate import evaluate as evaluate_fn
        data = load_npz(args.data_dir)
    else:
        from harness.evaluate_provisional import evaluate as evaluate_fn
        data = load_csv(args.data_dir)

    build_sequence_context(data, history_len=12)
    data['sample_weights'] = recency_weights(data['dates_tr'], 7.0)
    data['watch_target'], data['watch_censored'], data['watch_valid'] = watch_targets(data['play'], data['duration'])

    smoke_text = os.environ.get('SMOKE_EPOCHS')
    smoke_cap = int(smoke_text) if smoke_text is not None else None
    probe_epochs = min(3, smoke_cap) if smoke_cap is not None else 3
    refine_epochs = min(6, smoke_cap) if smoke_cap is not None else 6
    final_epochs = min(args.epochs, smoke_cap) if smoke_cap is not None else args.epochs
    if smoke_cap is not None:
        coarse_count, refine_count, final_count = 1, 0, 1
    elif device.type == 'cuda':
        coarse_count, refine_count, final_count = 56, 24, 2
    else:
        coarse_count, refine_count, final_count = 28, 12, 2

    rng = np.random.RandomState(args.seed + 4103)
    coarse_configs = []
    for _ in range(coarse_count):
        coarse_configs.append({
            'dropout': float(rng.uniform(0.12, 0.42)),
            'weight_decay': float(10.0 ** rng.uniform(math.log10(1e-5), math.log10(3e-3))),
            'lr': float(10.0 ** rng.uniform(math.log10(3e-4), math.log10(1.6e-3))),
            'step_gamma': float(rng.uniform(0.28, 0.7)),
            'aux_weight': float(rng.uniform(0.03, 0.22)),
            'bpr_mix': float(rng.choice([0.3, 0.5, 0.5, 0.7]))
        })

    history = []
    best_config = None
    best_primary = -1.0
    for probe, config in enumerate(coarse_configs):
        metrics, _, _ = train_one(data, config, probe_epochs, args.seed, device, evaluate_fn, False)
        record = {'stage': 'coarse', 'probe': probe, 'epochs': probe_epochs, 'config': config_json(config), **metrics}
        history.append(record)
        append_progress(progress_path, record)
        if metrics['primary'] > best_primary:
            best_primary = metrics['primary']
            best_config = dict(config)

    refined_best = dict(best_config)
    refined_primary = best_primary
    for probe in range(refine_count):
        config = {
            'dropout': float(np.clip(rng.normal(best_config['dropout'], 0.04), 0.08, 0.5)),
            'weight_decay': float(np.clip(best_config['weight_decay'] * math.exp(rng.normal(0, 0.5)), 5e-6, 6e-3)),
            'lr': float(np.clip(best_config['lr'] * math.exp(rng.normal(0, 0.25)), 2e-4, 2.2e-3)),
            'step_gamma': float(np.clip(rng.normal(best_config['step_gamma'], 0.08), 0.2, 0.8)),
            'aux_weight': float(np.clip(rng.normal(best_config['aux_weight'], 0.035), 0.015, 0.3)),
            'bpr_mix': float(np.clip(rng.normal(best_config['bpr_mix'], 0.1), 0.2, 0.8))
        }
        metrics, _, _ = train_one(data, config, refine_epochs, args.seed, device, evaluate_fn, False)
        record = {'stage': 'refine', 'probe': probe, 'epochs': refine_epochs, 'config': config_json(config), **metrics}
        history.append(record)
        append_progress(progress_path, record)
        if metrics['primary'] > refined_primary:
            refined_primary = metrics['primary']
            refined_best = dict(config)

    member_scores = []
    member_metrics = []
    member_seeds = []
    for offset in range(final_count):
        member_seed = args.seed + offset
        metrics, scores, trace = train_one(
            data, refined_best, final_epochs, member_seed, device, evaluate_fn, True
        )
        member_scores.append(scores)
        member_metrics.append(metrics)
        member_seeds.append(member_seed)
        record = {
            'stage': 'final_member', 'seed': member_seed, 'epochs': final_epochs,
            'config': config_json(refined_best), 'checkpoints': trace, **metrics
        }
        history.append(record)
        append_progress(progress_path, {k: v for k, v in record.items() if k != 'checkpoints'})

    final_scores = np.mean(np.stack(member_scores, axis=0), axis=0).astype(np.float32)
    final_metrics = get_metrics(evaluate_fn, data['uva'], data['yva'], final_scores)
    ensemble_record = {
        'stage': 'ensemble', 'kind': 'mean_logit', 'seeds': member_seeds,
        'members': len(member_scores), **final_metrics
    }
    history.append(ensemble_record)
    append_progress(progress_path, ensemble_record)

    output = {
        'gauc': final_metrics['gauc'],
        'ndcg5': final_metrics['ndcg5'],
        'primary': final_metrics['primary'],
        'selected': 'predeclared_mean_logit',
        'winning_config': config_json(refined_best),
        'history': history
    }
    with open(os.path.join(args.out_dir, 'metrics.json'), 'w') as fh:
        json.dump(output, fh)
    with open(os.path.join(args.out_dir, 'predictions.csv'), 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['row_id', 'user_id', 'video_id', 'score'])
        for i, score in enumerate(final_scores):
            writer.writerow([i, data['uva'][i], data['vva'][i], format(float(score), '.9g')])


if __name__ == '__main__':
    main()
