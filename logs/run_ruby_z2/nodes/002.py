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


class FM(torch.nn.Module):
    def __init__(self, total_dim, k=16):
        super().__init__()
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)

    def forward(self, x):
        e = self.emb(x)
        summed = e.sum(1)
        pair = 0.5 * (summed * summed - (e * e).sum(1)).sum(1)
        return self.bias + self.lin(x).sum((1, 2)) + pair


class DCNLite(torch.nn.Module):
    def __init__(self, total_dim, fields=5, k=16, dropout=0.25):
        super().__init__()
        width = fields * k
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.emb_drop = torch.nn.Dropout(dropout)
        self.cross_w = torch.nn.ParameterList([
            torch.nn.Parameter(torch.empty(width)) for _ in range(2)
        ])
        self.cross_b = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros(width)) for _ in range(2)
        ])
        self.deep = torch.nn.Sequential(
            torch.nn.Linear(width, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.cross_out = torch.nn.Linear(width, 1, bias=False)
        self.deep_out = torch.nn.Linear(64, 1, bias=False)
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)
        for w in self.cross_w:
            torch.nn.init.normal_(w, std=0.01)

    def forward(self, x):
        e = self.emb_drop(self.emb(x))
        summed = e.sum(1)
        fm = 0.5 * (summed * summed - (e * e).sum(1)).sum(1)
        linear = self.lin(x).sum((1, 2))
        x0 = e.flatten(1)
        cross = x0
        for w, b in zip(self.cross_w, self.cross_b):
            cross = x0 * (cross * w).sum(1, keepdim=True) + b + cross
        deep = self.deep(x0)
        return self.bias + linear + fm + self.cross_out(cross).squeeze(1) + self.deep_out(deep).squeeze(1)


def parse_date_ord(values):
    arr = np.asarray(values)
    out = np.zeros(len(arr), dtype=np.int32)
    cache = {}
    for i, value in enumerate(arr):
        text = str(value.decode() if isinstance(value, bytes) else value)
        text = text.split('.')[0].replace('-', '')
        if text not in cache:
            try:
                cache[text] = datetime.date(int(text[:4]), int(text[4:6]), int(text[6:8])).toordinal()
            except Exception:
                cache[text] = 0
        out[i] = cache[text]
    return out


def parse_hour(values):
    arr = np.asarray(values)
    out = np.zeros(len(arr), dtype=np.int16)
    for i, value in enumerate(arr):
        text = str(value.decode() if isinstance(value, bytes) else value).strip()
        try:
            if ':' in text:
                hour = int(text.split(':')[0])
            else:
                number = int(float(text))
                if 0 <= number <= 23:
                    hour = number
                elif 0 <= number <= 2359 and number % 100 < 60:
                    hour = number // 100
                else:
                    hour = (number // 60) % 24
            out[i] = int(np.clip(hour, 0, 23))
        except Exception:
            out[i] = 0
    return out


def load_csv_data(data_dir):
    def read_rows(path):
        rows = []
        with open(path, newline='') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows.append({
                    'user': row['user_id'],
                    'video': row['video_id'],
                    'tab': row['tab'],
                    'duration': float(row['duration_ms']),
                    'date': row['date'],
                    'hourmin': row['hourmin'],
                    'y': float(row['long_view'])
                })
        return rows

    tr_rows = read_rows(os.path.join(data_dir, 'train.csv'))
    va_rows = read_rows(os.path.join(data_dir, 'val.csv'))
    durations = np.asarray([r['duration'] for r in tr_rows], dtype=np.float64)
    edges = np.unique(np.quantile(durations, np.linspace(0.1, 0.9, 9)))
    maps = []
    for key in ('user', 'video', 'author', 'tab', 'dur'):
        if key == 'author':
            vals = ['0']
        elif key == 'dur':
            vals = [str(int(np.searchsorted(edges, r['duration'], side='right'))) for r in tr_rows]
        else:
            vals = [r[key] for r in tr_rows]
        maps.append({v: i + 1 for i, v in enumerate(sorted(set(vals)))})
    field_dims = np.asarray([len(m) + 1 for m in maps], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1]))

    def encode(rows):
        x = np.zeros((len(rows), 5), dtype=np.int32)
        raw_user = np.empty(len(rows), dtype=object)
        raw_video = np.empty(len(rows), dtype=object)
        for i, row in enumerate(rows):
            duration_bucket = str(int(np.searchsorted(edges, row['duration'], side='right')))
            values = [row['user'], row['video'], '0', row['tab'], duration_bucket]
            for j, value in enumerate(values):
                x[i, j] = offsets[j] + maps[j].get(value, 0)
            raw_user[i] = row['user']
            raw_video[i] = row['video']
        return {
            'X': x,
            'y': np.asarray([r['y'] for r in rows], dtype=np.float32),
            'user': raw_user,
            'video_raw': raw_video,
            'date': np.asarray([r['date'] for r in rows]),
            'hourmin': np.asarray([r['hourmin'] for r in rows]),
            'field_dims': field_dims
        }

    return encode(tr_rows), encode(va_rows), False


def load_data(data_dir):
    train_path = os.path.join(data_dir, 'train.npz')
    val_path = os.path.join(data_dir, 'val.npz')
    if os.path.exists(train_path) and os.path.exists(val_path):
        with np.load(train_path) as data:
            tr = {k: data[k] for k in data.files}
        with np.load(val_path) as data:
            va = {k: data[k] for k in data.files}
        va['video_raw'] = np.asarray(va['X'])[:, 1]
        return tr, va, True
    return load_csv_data(data_dir)


def metric_values(evaluator, users, labels, scores):
    result = evaluator(users, labels.astype(int), scores)
    return {
        'gauc': float(result['GAUC'] if 'GAUC' in result else result['gauc']),
        'ndcg5': float(result['nDCG@5'] if 'nDCG@5' in result else result['ndcg5']),
        'primary': float(result['primary'])
    }


def predict(model, xv, device, batch_size=65536):
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(xv), batch_size):
            xb = xv[start:start + batch_size].to(device, non_blocking=True)
            parts.append(model(xb).detach().cpu().numpy())
    return np.concatenate(parts).astype(np.float64)


class SortedNegativePool:
    def __init__(self, keys, negative_indices):
        neg_keys = np.asarray(keys, dtype=np.int64)[negative_indices]
        order = np.argsort(neg_keys, kind='stable')
        self.keys = neg_keys[order]
        self.indices = negative_indices[order]

    def sample(self, query_keys, rng):
        query_keys = np.asarray(query_keys, dtype=np.int64)
        left = np.searchsorted(self.keys, query_keys, side='left')
        right = np.searchsorted(self.keys, query_keys, side='right')
        counts = right - left
        available = counts > 0
        result = np.full(len(query_keys), -1, dtype=np.int64)
        if np.any(available):
            offsets = (rng.random(int(available.sum())) * counts[available]).astype(np.int64)
            result[available] = self.indices[left[available] + offsets]
        return result, available


class ContextPairIndex:
    def __init__(self, users, labels, dates, hours, tabs):
        _, user_code = np.unique(np.asarray(users), return_inverse=True)
        _, date_code = np.unique(np.asarray(dates), return_inverse=True)
        _, hour_code = np.unique(np.asarray(hours), return_inverse=True)
        _, tab_code = np.unique(np.asarray(tabs), return_inverse=True)
        self.user_code = user_code.astype(np.int64)
        self.date_code = date_code.astype(np.int64)
        self.hour_code = hour_code.astype(np.int64)
        self.tab_code = tab_code.astype(np.int64)
        n_date = int(self.date_code.max()) + 1 if len(labels) else 1
        n_hour = int(self.hour_code.max()) + 1 if len(labels) else 1
        n_tab = int(self.tab_code.max()) + 1 if len(labels) else 1
        self.day_key = self.user_code * n_date + self.date_code
        self.hour_key = self.day_key * n_hour + self.hour_code
        self.tab_key = self.day_key * n_tab + self.tab_code
        negative = np.flatnonzero(labels < 0.5).astype(np.int64)
        self.user_pool = SortedNegativePool(self.user_code, negative)
        self.day_pool = SortedNegativePool(self.day_key, negative)
        self.hour_pool = SortedNegativePool(self.hour_key, negative)
        self.tab_pool = SortedNegativePool(self.tab_key, negative)
        user_left = np.searchsorted(self.user_pool.keys, self.user_code, side='left')
        user_right = np.searchsorted(self.user_pool.keys, self.user_code, side='right')
        self.positive = np.flatnonzero((labels >= 0.5) & (user_right > user_left)).astype(np.int64)

    def sample(self, positives, context_fraction, rng):
        uniform, available = self.user_pool.sample(self.user_code[positives], rng)
        if not np.all(available):
            raise RuntimeError('A sampled positive has no within-user negative')
        if context_fraction <= 0.0:
            return uniform
        target = rng.random(len(positives)) < float(context_fraction)
        unresolved = np.flatnonzero(target)
        result = uniform.copy()
        for pool, keys in ((self.hour_pool, self.hour_key),
                           (self.tab_pool, self.tab_key),
                           (self.day_pool, self.day_key)):
            if len(unresolved) == 0:
                break
            sampled, found = pool.sample(keys[positives[unresolved]], rng)
            if np.any(found):
                destination = unresolved[found]
                result[destination] = sampled[found]
            unresolved = unresolved[~found]
        return result


def recency_weights(dates, half_life, cache):
    key = float(half_life)
    if key in cache:
        return cache[key]
    ordinals = parse_date_ord(dates)
    valid = ordinals > 0
    newest = int(ordinals[valid].max()) if np.any(valid) else 0
    age = np.maximum(newest - ordinals, 0).astype(np.float32)
    weights = np.exp(-math.log(2.0) * age / key).astype(np.float32)
    weights /= max(float(weights.mean()), 1e-8)
    cache[key] = torch.from_numpy(weights)
    return cache[key]


def train_package(config, context_fraction, seed, epochs, xt, yt, xv, train_dates,
                  val_users, val_y, total_dim, device, evaluator, pair_index,
                  recency_cache, half_checkpoints=False, keep_state=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = DCNLite(total_dim, dropout=float(config['dropout'])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config['lr']), weight_decay=float(config['weight_decay']))
    weights = recency_weights(train_dates, config['half_life'], recency_cache)
    n = len(yt)
    batch_size = 8192
    best_primary = -1.0
    best_scores = None
    best_state = None
    checkpoints = []
    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed + 17011)
    rng = np.random.default_rng(seed + 29023)
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(n, generator=generator)
        running_loss = 0.0
        batches = 0
        evaluated_half = False
        for start in range(0, n, batch_size):
            idx = permutation[start:start + batch_size]
            xb = xt[idx].to(device, non_blocking=True)
            yb = yt[idx].to(device, non_blocking=True)
            wb = weights[idx].to(device, non_blocking=True)
            logits = model(xb)
            bce_each = torch.nn.functional.binary_cross_entropy_with_logits(logits, yb, reduction='none')
            bce_loss = (bce_each * wb).sum() / wb.sum().clamp_min(1e-8)

            pair_n = max(1, len(idx) // 2)
            positives = pair_index.positive[rng.integers(0, len(pair_index.positive), size=pair_n)]
            negatives = pair_index.sample(positives, context_fraction, rng)
            positive_cpu = torch.from_numpy(positives)
            negative_cpu = torch.from_numpy(negatives)
            pair_x = torch.cat((xt[positive_cpu], xt[negative_cpu]), dim=0).to(device, non_blocking=True)
            pair_logits = model(pair_x)
            positive_logits = pair_logits[:pair_n]
            negative_logits = pair_logits[pair_n:]
            pair_weights = 0.5 * (weights[positive_cpu] + weights[negative_cpu])
            pair_weights_device = pair_weights.to(device, non_blocking=True)
            pair_each = torch.nn.functional.softplus(-(positive_logits - negative_logits))
            bpr_loss = (pair_each * pair_weights_device).sum() / pair_weights_device.sum().clamp_min(1e-8)
            loss = 0.5 * bce_loss + 0.5 * bpr_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running_loss += float(loss.detach().cpu())
            batches += 1

            processed = min(start + len(idx), n)
            if half_checkpoints and not evaluated_half and processed >= (n + 1) // 2:
                scores = predict(model, xv, device)
                metrics = metric_values(evaluator, val_users, val_y, scores)
                checkpoints.append({
                    'epoch': epoch + 0.5,
                    'train_loss': running_loss / batches,
                    'val_gauc': metrics['gauc'],
                    'val_primary': metrics['primary']
                })
                if metrics['primary'] > best_primary:
                    best_primary = metrics['primary']
                    best_scores = scores.copy()
                    if keep_state:
                        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                model.train()
                evaluated_half = True

        if (epoch + 1) % int(config['step_every']) == 0:
            for group in optimizer.param_groups:
                group['lr'] *= float(config['gamma'])
        scores = predict(model, xv, device)
        metrics = metric_values(evaluator, val_users, val_y, scores)
        checkpoints.append({
            'epoch': epoch + 1.0,
            'train_loss': running_loss / max(batches, 1),
            'val_gauc': metrics['gauc'],
            'val_primary': metrics['primary']
        })
        if metrics['primary'] > best_primary:
            best_primary = metrics['primary']
            best_scores = scores.copy()
            if keep_state:
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if keep_state and best_state is not None:
        model.load_state_dict(best_state)
        best_scores = predict(model, xv, device)
        best_primary = metric_values(evaluator, val_users, val_y, best_scores)['primary']
    return best_scores, best_primary, checkpoints


def train_parent_reference(seed, epochs, xt, yt, xv, val_users, val_y,
                           total_dim, device, evaluator):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = FM(total_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    n = len(yt)
    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed)
    best_primary = -1.0
    best_scores = None
    patience = 0
    for _ in range(epochs):
        model.train()
        permutation = torch.randperm(n, generator=generator)
        for start in range(0, n, 8192):
            idx = permutation[start:start + 8192]
            xb = xt[idx].to(device, non_blocking=True)
            yb = yt[idx].to(device, non_blocking=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(xb), yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scores = predict(model, xv, device)
        primary = metric_values(evaluator, val_users, val_y, scores)['primary']
        if primary > best_primary + 1e-6:
            best_primary = primary
            best_scores = scores.copy()
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break
    return best_scores, best_primary


def rank_within_user(users, scores):
    users = np.asarray(users)
    scores = np.asarray(scores)
    order = np.argsort(users, kind='stable')
    sorted_users = users[order]
    output = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_users[end] == sorted_users[start]:
            end += 1
        indices = order[start:end]
        local_order = np.argsort(scores[indices], kind='mergesort')
        ranks = np.empty(end - start, dtype=np.float64)
        ranks[local_order] = np.arange(end - start, dtype=np.float64)
        if end - start > 1:
            ranks /= float(end - start - 1)
        else:
            ranks[:] = 0.5
        output[indices] = ranks
        start = end
    return output


def append_progress(path, record):
    with open(path, 'a') as fh:
        fh.write(json.dumps(record, sort_keys=True) + '\n')


def coarse_configs(seed):
    rng = np.random.default_rng(seed + 8101)
    drops = np.linspace(0.15, 0.40, 12)
    decays = np.logspace(math.log10(3e-5), math.log10(3e-3), 12)
    rng.shuffle(drops)
    rng.shuffle(decays)
    half_lives = [3.5, 7.0, 14.0]
    schedules = [(1, 0.45), (1, 0.65), (2, 0.40), (2, 0.60), (3, 0.50), (3, 0.72)]
    learning_rates = [0.00055, 0.00075, 0.0010, 0.0013]
    configs = []
    for i in range(12):
        step_every, gamma = schedules[i % len(schedules)]
        configs.append({
            'dropout': float(drops[i]),
            'weight_decay': float(decays[i]),
            'half_life': float(half_lives[(i * 2) % 3]),
            'step_every': int(step_every),
            'gamma': float(gamma),
            'lr': float(learning_rates[(i * 3) % 4])
        })
    return configs


def refine_configs(winner, seed):
    rng = np.random.default_rng(seed + 19001)
    drop_offsets = np.linspace(-0.07, 0.07, 10)
    decay_factors = np.exp(np.linspace(-0.9, 0.9, 10))
    learning_rate_factors = np.linspace(0.72, 1.28, 10)
    gamma_offsets = np.linspace(-0.12, 0.12, 10)
    rng.shuffle(drop_offsets)
    rng.shuffle(decay_factors)
    rng.shuffle(learning_rate_factors)
    rng.shuffle(gamma_offsets)
    half_candidates = sorted(set([3.5, 5.0, 7.0, 10.0, 14.0, float(winner['half_life'])]))
    configs = []
    for i in range(10):
        configs.append({
            'dropout': float(np.clip(winner['dropout'] + drop_offsets[i], 0.10, 0.48)),
            'weight_decay': float(np.clip(winner['weight_decay'] * decay_factors[i], 1e-5, 6e-3)),
            'half_life': float(half_candidates[i % len(half_candidates)]),
            'step_every': int(max(1, min(3, winner['step_every'] + ((i % 3) - 1)))),
            'gamma': float(np.clip(winner['gamma'] + gamma_offsets[i], 0.28, 0.82)),
            'lr': float(np.clip(winner['lr'] * learning_rate_factors[i], 0.00035, 0.0017))
        })
    configs[0] = dict(winner)
    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=15)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    progress_path = os.path.join(args.out_dir, 'progress.log')
    if os.path.exists(progress_path):
        os.remove(progress_path)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tr, va, fast_path = load_data(args.data_dir)
    if fast_path:
        from data.official.evaluate import evaluate as evaluator
    else:
        from harness.evaluate_provisional import evaluate as evaluator

    xt = torch.from_numpy(np.asarray(tr['X'], dtype=np.int64))
    yt = torch.from_numpy(np.asarray(tr['y'], dtype=np.float32))
    xv = torch.from_numpy(np.asarray(va['X'], dtype=np.int64))
    train_users = np.asarray(tr['user'])
    val_users = np.asarray(va['user'])
    val_y = np.asarray(va['y'], dtype=np.float32)
    train_dates = np.asarray(tr['date'])
    train_hours = parse_hour(tr['hourmin'])
    train_tabs = np.asarray(tr['X'])[:, 3]
    total_dim = int(np.asarray(tr['field_dims']).sum())
    pair_index = ContextPairIndex(
        train_users, np.asarray(tr['y'], dtype=np.float32),
        parse_date_ord(train_dates), train_hours, train_tabs)
    if len(pair_index.positive) == 0:
        raise RuntimeError('No within-user positive-negative training pairs are available')

    smoke_text = os.environ.get('SMOKE_EPOCHS')
    smoke_cap = int(smoke_text) if smoke_text is not None else None
    coarse_epochs = min(3, smoke_cap) if smoke_cap is not None else 3
    refine_epochs = min(6, smoke_cap) if smoke_cap is not None else 6
    dial_epochs = min(8, smoke_cap) if smoke_cap is not None else 8
    final_epochs = min(args.epochs, smoke_cap) if smoke_cap is not None else args.epochs
    reference_epochs = min(12, smoke_cap) if smoke_cap is not None else 12
    history = []
    recency_cache = {}

    coarse_list = coarse_configs(args.seed)
    if smoke_cap is not None:
        coarse_list = coarse_list[:2]
    best_config = None
    best_probe = -1.0
    for i, config in enumerate(coarse_list):
        _, primary, checkpoints = train_package(
            config, 0.0, args.seed + 101 + i, coarse_epochs, xt, yt, xv,
            train_dates, val_users, val_y, total_dim, device, evaluator,
            pair_index, recency_cache)
        record = {'stage': 'coarse', 'probe': i + 1, 'config': config,
                  'context_fraction': 0.0, 'epochs': coarse_epochs,
                  'primary': float(primary), 'checkpoints': checkpoints}
        history.append(record)
        append_progress(progress_path, {'stage': 'coarse', 'probe': i + 1,
                                        'config': config, 'primary': float(primary)})
        if primary > best_probe:
            best_probe = primary
            best_config = dict(config)

    refine_list = refine_configs(best_config, args.seed)
    if smoke_cap is not None:
        refine_list = refine_list[:1]
    refined_config = dict(best_config)
    refined_primary = best_probe
    for i, config in enumerate(refine_list):
        _, primary, checkpoints = train_package(
            config, 0.0, args.seed + 401 + i, refine_epochs, xt, yt, xv,
            train_dates, val_users, val_y, total_dim, device, evaluator,
            pair_index, recency_cache)
        record = {'stage': 'refine', 'probe': i + 1, 'config': config,
                  'context_fraction': 0.0, 'epochs': refine_epochs,
                  'primary': float(primary), 'checkpoints': checkpoints}
        history.append(record)
        append_progress(progress_path, {'stage': 'refine', 'probe': i + 1,
                                        'config': config, 'primary': float(primary)})
        if primary > refined_primary:
            refined_primary = primary
            refined_config = dict(config)

    if smoke_cap is None:
        fractions = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 1.0]
        dial_repeats = 5
    else:
        fractions = [0.0, 0.30]
        dial_repeats = 1
    fraction_results = {}
    for fraction_index, fraction in enumerate(fractions):
        values = []
        for repeat in range(dial_repeats):
            probe_seed = args.seed + 1001 + fraction_index * 31 + repeat
            _, primary, checkpoints = train_package(
                refined_config, fraction, probe_seed, dial_epochs, xt, yt, xv,
                train_dates, val_users, val_y, total_dim, device, evaluator,
                pair_index, recency_cache)
            values.append(float(primary))
            record = {'stage': 'context_dial', 'context_fraction': float(fraction),
                      'repeat': repeat + 1, 'seed': probe_seed, 'config': refined_config,
                      'epochs': dial_epochs, 'primary': float(primary),
                      'checkpoints': checkpoints}
            history.append(record)
            append_progress(progress_path, {
                'stage': 'context_dial', 'context_fraction': float(fraction),
                'repeat': repeat + 1, 'seed': probe_seed, 'primary': float(primary)})
        fraction_results[float(fraction)] = {
            'mean_primary': float(np.mean(values)),
            'std_primary': float(np.std(values)),
            'primaries': values
        }

    selected_fraction = max(
        fractions, key=lambda value: fraction_results[float(value)]['mean_primary'])
    history.append({'stage': 'context_selection',
                    'selected_context_fraction': float(selected_fraction),
                    'results': fraction_results})
    append_progress(progress_path, {
        'stage': 'context_selection',
        'selected_context_fraction': float(selected_fraction),
        'mean_primary': fraction_results[float(selected_fraction)]['mean_primary']})

    parent_scores, parent_primary = train_parent_reference(
        args.seed, reference_epochs, xt, yt, xv, val_users, val_y,
        total_dim, device, evaluator)
    history.append({'stage': 'parent_reference', 'seed': args.seed,
                    'epochs': reference_epochs, 'primary': float(parent_primary)})
    append_progress(progress_path, {'stage': 'parent_reference',
                                    'seed': args.seed, 'primary': float(parent_primary)})

    member_count = 5 if smoke_cap is None else 2
    member_scores = []
    member_primaries = []
    for member in range(member_count):
        member_seed = args.seed + member
        scores, primary, checkpoints = train_package(
            refined_config, selected_fraction, member_seed, final_epochs,
            xt, yt, xv, train_dates, val_users, val_y, total_dim, device,
            evaluator, pair_index, recency_cache, True, True)
        if np.allclose(scores, parent_scores, rtol=1e-6, atol=1e-7):
            raise AssertionError('Final member is identical to parent predictions')
        for previous in member_scores:
            if np.allclose(scores, previous, rtol=1e-6, atol=1e-7):
                raise AssertionError('Distinct-seed members produced identical predictions')
        member_scores.append(scores)
        member_primaries.append(float(primary))
        record = {'stage': 'final_member', 'member': member + 1,
                  'seed': member_seed, 'config': refined_config,
                  'context_fraction': float(selected_fraction),
                  'epochs': final_epochs, 'primary': float(primary),
                  'checkpoints': checkpoints}
        history.append(record)
        append_progress(progress_path, {
            'stage': 'final_member', 'member': member + 1,
            'seed': member_seed, 'primary': float(primary)})

    ranked_members = [rank_within_user(val_users, scores) for scores in member_scores]
    final_scores = np.mean(np.stack(ranked_members, axis=0), axis=0)
    parent_ranked = rank_within_user(val_users, parent_scores)
    if np.allclose(final_scores, parent_ranked, rtol=1e-6, atol=1e-7):
        raise AssertionError('Final ensemble is identical to parent ranked predictions')
    metrics = metric_values(evaluator, val_users, val_y, final_scores)
    history.append({'stage': 'ensemble', 'members': member_count,
                    'seeds': [args.seed + i for i in range(member_count)],
                    'context_fraction': float(selected_fraction),
                    'primary': metrics['primary']})
    append_progress(progress_path, {'stage': 'ensemble', 'members': member_count,
                                    'primary': metrics['primary']})

    output_metrics = {
        'gauc': metrics['gauc'],
        'ndcg5': metrics['ndcg5'],
        'primary': metrics['primary'],
        'selected_config': refined_config,
        'selected_context_fraction': float(selected_fraction),
        'context_fraction_results': fraction_results,
        'parent_reference_primary': float(parent_primary),
        'member_primaries': member_primaries,
        'history': history
    }
    with open(os.path.join(args.out_dir, 'metrics.json'), 'w') as fh:
        json.dump(output_metrics, fh)

    video_values = np.asarray(va.get('video_raw', np.asarray(va['X'])[:, 1]))
    with open(os.path.join(args.out_dir, 'predictions.csv'), 'w') as fh:
        fh.write('row_id,user_id,video_id,score\n')
        for i, score in enumerate(final_scores):
            fh.write(f'{i},{val_users[i]},{video_values[i]},{score:.9g}\n')


if __name__ == '__main__':
    main()
