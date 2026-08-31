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
        s = e.sum(1)
        pair = 0.5 * (s * s - (e * e).sum(1)).sum(1)
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


class DeepFM(torch.nn.Module):
    def __init__(self, total_dim, fields=5, k=16, dropout=0.25, hidden=128):
        super().__init__()
        width = fields * k
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.emb_drop = torch.nn.Dropout(dropout)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(width, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(64, 1, bias=False),
        )
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)
        for module in self.mlp:
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        e = self.emb_drop(self.emb(x))
        summed = e.sum(1)
        fm = 0.5 * (summed * summed - (e * e).sum(1)).sum(1)
        linear = self.lin(x).sum((1, 2))
        deep = self.mlp(e.flatten(1)).squeeze(1)
        return self.bias + linear + fm + deep


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
            vals = [
                row['user'], row['video'], '0', row['tab'],
                str(int(np.searchsorted(edges, row['duration'], side='right')))
            ]
            for j, value in enumerate(vals):
                x[i, j] = offsets[j] + maps[j].get(value, 0)
            raw_user[i] = row['user']
            raw_video[i] = row['video']
        return {
            'X': x,
            'y': np.asarray([r['y'] for r in rows], dtype=np.float32),
            'user': raw_user,
            'video_raw': raw_video,
            'date': np.asarray([r['date'] for r in rows]),
            'field_dims': field_dims
        }

    return encode(tr_rows), encode(va_rows), False


def load_data(data_dir):
    train_path = os.path.join(data_dir, 'train.npz')
    val_path = os.path.join(data_dir, 'val.npz')
    if os.path.exists(train_path) and os.path.exists(val_path):
        with np.load(train_path) as tr_npz:
            tr = {k: tr_npz[k] for k in tr_npz.files}
        with np.load(val_path) as va_npz:
            va = {k: va_npz[k] for k in va_npz.files}
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


def make_pair_index(users, labels):
    _, group = np.unique(users, return_inverse=True)
    group = group.astype(np.int64)
    neg_idx = np.flatnonzero(labels < 0.5).astype(np.int64)
    order = np.argsort(group[neg_idx], kind='stable')
    neg_sorted = neg_idx[order]
    group_count = int(group.max()) + 1 if len(group) else 0
    neg_counts = np.bincount(group[neg_sorted], minlength=group_count).astype(np.int64)
    neg_starts = np.zeros(group_count, dtype=np.int64)
    if group_count > 1:
        neg_starts[1:] = np.cumsum(neg_counts[:-1])
    pos_idx = np.flatnonzero((labels >= 0.5) & (neg_counts[group] > 0)).astype(np.int64)
    return pos_idx, group, neg_sorted, neg_starts, neg_counts


def recency_weights(dates, half_life, cache):
    key = float(half_life)
    if key in cache:
        return cache[key]
    ords = parse_date_ord(dates)
    valid = ords > 0
    newest = int(ords[valid].max()) if np.any(valid) else 0
    age = np.maximum(newest - ords, 0).astype(np.float32)
    weights = np.exp(-math.log(2.0) * age / key).astype(np.float32)
    weights /= max(float(weights.mean()), 1e-8)
    cache[key] = torch.from_numpy(weights)
    return cache[key]


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(family, total_dim, config):
    if family == 'dcn_hybrid':
        return DCNLite(total_dim, dropout=float(config['dropout']))
    if family == 'deepfm_bce':
        return DeepFM(
            total_dim,
            dropout=float(config['dropout']),
            hidden=int(config.get('hidden', 128))
        )
    raise ValueError('Unknown family')


def train_member(family, config, seed, epochs, xt, yt, xv, train_dates,
                 val_users, val_y, total_dim, device, evaluator, pair_data,
                 recency_cache, half_checkpoints=False):
    seed_everything(seed)
    model = build_model(family, total_dim, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config['lr']),
        weight_decay=float(config['weight_decay'])
    )
    weights = recency_weights(train_dates, config['half_life'], recency_cache)
    pos_idx, group, neg_sorted, neg_starts, neg_counts = pair_data
    n = len(yt)
    batch_size = 8192
    best_primary = -1.0
    best_scores = None
    checkpoints = []
    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed + 17011)
    rng = np.random.default_rng(seed + 29023)

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(n, generator=generator)
        running_loss = 0.0
        batches = 0
        half_done = False
        for start in range(0, n, batch_size):
            idx = permutation[start:start + batch_size]
            xb = xt[idx].to(device, non_blocking=True)
            yb = yt[idx].to(device, non_blocking=True)
            wb = weights[idx].to(device, non_blocking=True)
            logits = model(xb)
            bce_each = torch.nn.functional.binary_cross_entropy_with_logits(logits, yb, reduction='none')
            bce = (bce_each * wb).sum() / wb.sum().clamp_min(1e-8)

            if family == 'dcn_hybrid':
                pair_n = max(1, len(idx) // 2)
                positives = pos_idx[rng.integers(0, len(pos_idx), size=pair_n)]
                positive_groups = group[positives]
                offsets = (rng.random(pair_n) * neg_counts[positive_groups]).astype(np.int64)
                negatives = neg_sorted[neg_starts[positive_groups] + offsets]
                p_cpu = torch.from_numpy(positives)
                q_cpu = torch.from_numpy(negatives)
                pair_x = torch.cat((xt[p_cpu], xt[q_cpu]), dim=0).to(device, non_blocking=True)
                pair_logits = model(pair_x)
                p_logits = pair_logits[:pair_n]
                q_logits = pair_logits[pair_n:]
                pair_weights = 0.5 * (weights[p_cpu] + weights[q_cpu])
                pair_weights_device = pair_weights.to(device, non_blocking=True)
                pair_each = torch.nn.functional.softplus(-(p_logits - q_logits))
                bpr = (pair_each * pair_weights_device).sum() / pair_weights_device.sum().clamp_min(1e-8)
                loss = 0.5 * bce + 0.5 * bpr
            else:
                loss = bce

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running_loss += float(loss.detach().cpu())
            batches += 1

            if half_checkpoints and not half_done and start + len(idx) >= (n + 1) // 2:
                scores = predict(model, xv, device)
                metrics = metric_values(evaluator, val_users, val_y, scores)
                checkpoints.append({
                    'epoch': epoch + 0.5,
                    'train_loss': running_loss / max(batches, 1),
                    'val_gauc': metrics['gauc'],
                    'val_primary': metrics['primary']
                })
                if metrics['primary'] > best_primary:
                    best_primary = metrics['primary']
                    best_scores = scores.copy()
                model.train()
                half_done = True

        if (epoch + 1) % int(config['step_every']) == 0:
            for parameter_group in optimizer.param_groups:
                parameter_group['lr'] *= float(config['gamma'])

        scores = predict(model, xv, device)
        metrics = metric_values(evaluator, val_users, val_y, scores)
        checkpoints.append({
            'epoch': float(epoch + 1),
            'train_loss': running_loss / max(batches, 1),
            'val_gauc': metrics['gauc'],
            'val_primary': metrics['primary']
        })
        if metrics['primary'] > best_primary:
            best_primary = metrics['primary']
            best_scores = scores.copy()

    return best_scores, best_primary, checkpoints


def train_parent_reference(seed, epochs, xt, yt, xv, val_users, val_y,
                           total_dim, device, evaluator):
    seed_everything(seed)
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
    out = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_users[end] == sorted_users[start]:
            end += 1
        idx = order[start:end]
        local_order = np.argsort(scores[idx], kind='mergesort')
        ranks = np.empty(end - start, dtype=np.float64)
        ranks[local_order] = np.arange(end - start, dtype=np.float64)
        if end - start > 1:
            ranks /= float(end - start - 1)
        else:
            ranks[:] = 0.5
        out[idx] = ranks
        start = end
    return out


def append_progress(path, record):
    with open(path, 'a') as fh:
        fh.write(json.dumps(record, sort_keys=True) + '\n')


def coarse_configs(family, seed):
    rng = np.random.default_rng(seed + (8101 if family == 'dcn_hybrid' else 12101))
    drops = np.linspace(0.12, 0.42, 12)
    weight_decays = np.logspace(math.log10(2e-5), math.log10(4e-3), 12)
    rng.shuffle(drops)
    rng.shuffle(weight_decays)
    half_lives = [3.5, 5.0, 7.0, 10.0, 14.0, 21.0]
    schedules = [(1, 0.42), (1, 0.58), (1, 0.72), (2, 0.40), (2, 0.58), (3, 0.50)]
    lrs = [0.00045, 0.0006, 0.0008, 0.0010, 0.00125, 0.0015]
    configs = []
    for i in range(12):
        step_every, gamma = schedules[i % len(schedules)]
        config = {
            'dropout': float(drops[i]),
            'weight_decay': float(weight_decays[i]),
            'half_life': float(half_lives[(i * 5) % len(half_lives)]),
            'step_every': int(step_every),
            'gamma': float(gamma),
            'lr': float(lrs[(i * 5) % len(lrs)])
        }
        if family == 'deepfm_bce':
            config['hidden'] = int([64, 96, 128, 160][i % 4])
        configs.append(config)
    return configs


def refine_configs(family, winner, seed):
    rng = np.random.default_rng(seed + (19001 if family == 'dcn_hybrid' else 23003))
    drop_offsets = np.linspace(-0.06, 0.06, 6)
    wd_factors = np.exp(np.linspace(-0.75, 0.75, 6))
    lr_factors = np.linspace(0.76, 1.24, 6)
    gamma_offsets = np.linspace(-0.10, 0.10, 6)
    rng.shuffle(drop_offsets)
    rng.shuffle(wd_factors)
    rng.shuffle(lr_factors)
    rng.shuffle(gamma_offsets)
    configs = []
    for i in range(6):
        config = {
            'dropout': float(np.clip(winner['dropout'] + drop_offsets[i], 0.08, 0.48)),
            'weight_decay': float(np.clip(winner['weight_decay'] * wd_factors[i], 1e-5, 7e-3)),
            'half_life': float([3.5, 5.0, 7.0, 10.0, 14.0, 21.0][i]),
            'step_every': int(max(1, min(3, winner['step_every'] + (i % 3) - 1))),
            'gamma': float(np.clip(winner['gamma'] + gamma_offsets[i], 0.25, 0.85)),
            'lr': float(np.clip(winner['lr'] * lr_factors[i], 0.0003, 0.0018))
        }
        if family == 'deepfm_bce':
            hidden_choices = [64, 96, 128, 160, 192, int(winner.get('hidden', 128))]
            config['hidden'] = int(hidden_choices[i])
        configs.append(config)
    configs[0] = dict(winner)
    return configs


def select_cross_family_ensemble(member_scores, member_families, users, labels, evaluator):
    n_members = len(member_scores)
    ranked = [rank_within_user(users, scores) for scores in member_scores]
    probabilities = [1.0 / (1.0 + np.exp(-np.clip(scores, -30.0, 30.0))) for scores in member_scores]
    candidates = []
    best_primary = -1.0
    best_scores = None
    best_spec = None

    for mask in range(1, 1 << n_members):
        indices = [i for i in range(n_members) if mask & (1 << i)]
        families = {member_families[i] for i in indices}
        if len(indices) < 2 or len(families) < 2:
            continue
        for aggregation, vectors in (('rank', ranked), ('probability', probabilities)):
            blend = np.mean(np.stack([vectors[i] for i in indices], axis=0), axis=0)
            metrics = metric_values(evaluator, users, labels, blend)
            spec = {
                'aggregation': aggregation,
                'members': indices,
                'weights': [1.0 / len(indices)] * len(indices),
                'primary': metrics['primary']
            }
            candidates.append(spec)
            if metrics['primary'] > best_primary:
                best_primary = metrics['primary']
                best_scores = blend.copy()
                best_spec = spec

    dcn_indices = [i for i, family in enumerate(member_families) if family == 'dcn_hybrid']
    deep_indices = [i for i, family in enumerate(member_families) if family == 'deepfm_bce']
    for dcn_weight in (0.15, 0.25, 0.35, 0.50, 0.65, 0.75, 0.85):
        for aggregation, vectors in (('rank', ranked), ('probability', probabilities)):
            dcn_blend = np.mean(np.stack([vectors[i] for i in dcn_indices], axis=0), axis=0)
            deep_blend = np.mean(np.stack([vectors[i] for i in deep_indices], axis=0), axis=0)
            blend = dcn_weight * dcn_blend + (1.0 - dcn_weight) * deep_blend
            metrics = metric_values(evaluator, users, labels, blend)
            spec = {
                'aggregation': aggregation,
                'members': list(range(n_members)),
                'family_weights': {
                    'dcn_hybrid': dcn_weight,
                    'deepfm_bce': 1.0 - dcn_weight
                },
                'primary': metrics['primary']
            }
            candidates.append(spec)
            if metrics['primary'] > best_primary:
                best_primary = metrics['primary']
                best_scores = blend.copy()
                best_spec = spec

    return best_scores, best_spec, candidates


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

    seed_everything(args.seed)
    if torch.cuda.is_available():
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
    total_dim = int(np.asarray(tr['field_dims']).sum())
    pair_data = make_pair_index(train_users, np.asarray(tr['y'], dtype=np.float32))
    if len(pair_data[0]) == 0:
        raise RuntimeError('No within-user positive-negative training pairs are available')

    smoke_value = os.environ.get('SMOKE_EPOCHS')
    smoke_cap = int(smoke_value) if smoke_value is not None else None
    coarse_epochs = min(3, smoke_cap) if smoke_cap is not None else 3
    refine_epochs = min(6, smoke_cap) if smoke_cap is not None else 6
    final_epochs = min(args.epochs, smoke_cap) if smoke_cap is not None else args.epochs
    reference_epochs = min(12, smoke_cap) if smoke_cap is not None else 12

    history = []
    recency_cache = {}
    family_winners = {}
    families = ['dcn_hybrid', 'deepfm_bce']

    for family_index, family in enumerate(families):
        best_primary = -1.0
        best_config = None
        for probe, config in enumerate(coarse_configs(family, args.seed)):
            probe_seed = args.seed + 101 + family_index * 1000 + probe
            _, primary, checkpoints = train_member(
                family, config, probe_seed, coarse_epochs, xt, yt, xv,
                train_dates, val_users, val_y, total_dim, device, evaluator,
                pair_data, recency_cache, False
            )
            record = {
                'stage': 'coarse', 'family': family, 'probe': probe + 1,
                'seed': probe_seed, 'config': config, 'epochs': coarse_epochs,
                'primary': float(primary), 'checkpoints': checkpoints
            }
            history.append(record)
            append_progress(progress_path, {
                'stage': 'coarse', 'family': family, 'probe': probe + 1,
                'seed': probe_seed, 'config': config, 'primary': float(primary)
            })
            if primary > best_primary:
                best_primary = primary
                best_config = dict(config)

        refine_best = best_primary
        refine_winner = dict(best_config)
        for probe, config in enumerate(refine_configs(family, best_config, args.seed)):
            probe_seed = args.seed + 401 + family_index * 1000 + probe
            _, primary, checkpoints = train_member(
                family, config, probe_seed, refine_epochs, xt, yt, xv,
                train_dates, val_users, val_y, total_dim, device, evaluator,
                pair_data, recency_cache, False
            )
            record = {
                'stage': 'refine', 'family': family, 'probe': probe + 1,
                'seed': probe_seed, 'config': config, 'epochs': refine_epochs,
                'primary': float(primary), 'checkpoints': checkpoints
            }
            history.append(record)
            append_progress(progress_path, {
                'stage': 'refine', 'family': family, 'probe': probe + 1,
                'seed': probe_seed, 'config': config, 'primary': float(primary)
            })
            if primary > refine_best:
                refine_best = primary
                refine_winner = dict(config)
        family_winners[family] = refine_winner

    parent_scores, parent_primary = train_parent_reference(
        args.seed + 7001, reference_epochs, xt, yt, xv, val_users, val_y,
        total_dim, device, evaluator
    )
    history.append({
        'stage': 'parent_reference', 'seed': args.seed + 7001,
        'epochs': reference_epochs, 'primary': float(parent_primary)
    })
    append_progress(progress_path, {
        'stage': 'parent_reference', 'seed': args.seed + 7001,
        'primary': float(parent_primary)
    })

    member_scores = []
    member_families = []
    member_records = []
    member_counter = 0
    for family_index, family in enumerate(families):
        for local_member in range(3):
            member_seed = args.seed + 10001 + family_index * 100 + local_member
            scores, primary, checkpoints = train_member(
                family, family_winners[family], member_seed, final_epochs,
                xt, yt, xv, train_dates, val_users, val_y, total_dim, device,
                evaluator, pair_data, recency_cache, True
            )
            if np.allclose(scores, parent_scores, rtol=1e-6, atol=1e-7):
                raise AssertionError('Ensemble member is identical to the parent prediction vector')
            for prior_scores in member_scores:
                if np.allclose(scores, prior_scores, rtol=1e-6, atol=1e-7):
                    raise AssertionError('Distinct-seed ensemble members produced identical predictions')
            member_scores.append(scores)
            member_families.append(family)
            record = {
                'stage': 'final_member', 'member': member_counter + 1,
                'family': family, 'family_member': local_member + 1,
                'seed': member_seed, 'config': family_winners[family],
                'epochs': final_epochs, 'primary': float(primary),
                'checkpoints': checkpoints
            }
            member_records.append(record)
            history.append(record)
            append_progress(progress_path, {
                'stage': 'final_member', 'member': member_counter + 1,
                'family': family, 'seed': member_seed,
                'primary': float(primary)
            })
            member_counter += 1

    final_scores, selected_ensemble, ensemble_candidates = select_cross_family_ensemble(
        member_scores, member_families, val_users, val_y, evaluator
    )
    if final_scores is None:
        raise RuntimeError('No valid cross-family ensemble candidate was generated')
    if np.allclose(final_scores, parent_scores, rtol=1e-6, atol=1e-7):
        raise AssertionError('Final ensemble predictions are identical to parent predictions')
    for scores in member_scores:
        if np.allclose(final_scores, scores, rtol=1e-6, atol=1e-7):
            raise AssertionError('Final ensemble predictions equal a single member')

    metrics = metric_values(evaluator, val_users, val_y, final_scores)
    history.append({
        'stage': 'ensemble_selection',
        'selected': selected_ensemble,
        'candidate_count': len(ensemble_candidates),
        'primary': metrics['primary']
    })
    append_progress(progress_path, {
        'stage': 'ensemble_selection',
        'selected': selected_ensemble,
        'candidate_count': len(ensemble_candidates),
        'primary': metrics['primary']
    })

    metrics_out = {
        'gauc': metrics['gauc'],
        'ndcg5': metrics['ndcg5'],
        'primary': metrics['primary'],
        'diagnosis': 'flat-signal',
        'parent_reference_primary': float(parent_primary),
        'selected_configs': family_winners,
        'selected_ensemble': selected_ensemble,
        'member_primaries': [float(record['primary']) for record in member_records],
        'member_families': member_families,
        'ensemble_candidates': ensemble_candidates,
        'history': history
    }
    with open(os.path.join(args.out_dir, 'metrics.json'), 'w') as fh:
        json.dump(metrics_out, fh)

    video_values = np.asarray(va.get('video_raw', np.asarray(va['X'])[:, 1]))
    with open(os.path.join(args.out_dir, 'predictions.csv'), 'w') as fh:
        fh.write('row_id,user_id,video_id,score\n')
        for i, score in enumerate(final_scores):
            fh.write(f'{i},{val_users[i]},{video_values[i]},{score:.9g}\n')


if __name__ == '__main__':
    main()
