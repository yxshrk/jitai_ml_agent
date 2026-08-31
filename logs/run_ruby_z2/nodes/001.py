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
    def read_rows(path, training):
        rows = []
        with open(path, newline='') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                item = {
                    'user': row['user_id'],
                    'video': row['video_id'],
                    'tab': row['tab'],
                    'duration': float(row['duration_ms']),
                    'date': row['date'],
                    'y': float(row['long_view'])
                }
                rows.append(item)
        return rows

    tr_rows = read_rows(os.path.join(data_dir, 'train.csv'), True)
    va_rows = read_rows(os.path.join(data_dir, 'val.csv'), False)
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
        for i, r in enumerate(rows):
            vals = [r['user'], r['video'], '0', r['tab'], str(int(np.searchsorted(edges, r['duration'], side='right')))]
            for j, value in enumerate(vals):
                x[i, j] = offsets[j] + maps[j].get(value, 0)
            raw_user[i] = r['user']
            raw_video[i] = r['video']
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
        tr_npz = np.load(train_path)
        va_npz = np.load(val_path)
        tr = {k: tr_npz[k] for k in tr_npz.files}
        va = {k: va_npz[k] for k in va_npz.files}
        va['video_raw'] = np.zeros(len(va['y']), dtype=np.int64)
        return tr, va, True
    return load_csv_data(data_dir)


def metric_values(evaluator, users, labels, scores):
    m = evaluator(users, labels.astype(int), scores)
    return {
        'gauc': float(m['GAUC'] if 'GAUC' in m else m['gauc']),
        'ndcg5': float(m['nDCG@5'] if 'nDCG@5' in m else m['ndcg5']),
        'primary': float(m['primary'])
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
    w = np.exp(-math.log(2.0) * age / key).astype(np.float32)
    w /= max(float(w.mean()), 1e-8)
    cache[key] = torch.from_numpy(w)
    return cache[key]


def train_package(config, seed, epochs, xt, yt, xv, train_users, train_dates,
                  val_users, val_y, total_dim, device, evaluator, pair_data,
                  recency_cache, half_checkpoints=False, keep_state=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = DCNLite(total_dim, dropout=float(config['dropout'])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(config['lr']), weight_decay=float(config['weight_decay']))
    weights = recency_weights(train_dates, config['half_life'], recency_cache)
    pos_idx, group, neg_sorted, neg_starts, neg_counts = pair_data
    n = len(yt)
    batch_size = 8192
    best_primary = -1.0
    best_scores = None
    best_state = None
    checkpoints = []
    tg = torch.Generator(device='cpu')
    tg.manual_seed(seed + 17011)
    rng = np.random.default_rng(seed + 29023)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=tg)
        evaluated_half = False
        running_loss = 0.0
        batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb = xt[idx].to(device, non_blocking=True)
            yb = yt[idx].to(device, non_blocking=True)
            wb = weights[idx].to(device, non_blocking=True)
            logits = model(xb)
            bce_each = torch.nn.functional.binary_cross_entropy_with_logits(logits, yb, reduction='none')
            bce_loss = (bce_each * wb).sum() / wb.sum().clamp_min(1e-8)
            pair_n = max(1, len(idx) // 2)
            chosen = pos_idx[rng.integers(0, len(pos_idx), size=pair_n)]
            chosen_group = group[chosen]
            offsets = (rng.random(pair_n) * neg_counts[chosen_group]).astype(np.int64)
            negatives = neg_sorted[neg_starts[chosen_group] + offsets]
            p_cpu = torch.from_numpy(chosen)
            q_cpu = torch.from_numpy(negatives)
            pair_x = torch.cat((xt[p_cpu], xt[q_cpu]), dim=0).to(device, non_blocking=True)
            pair_logits = model(pair_x)
            p_logit, q_logit = pair_logits[:pair_n], pair_logits[pair_n:]
            pair_w = 0.5 * (weights[p_cpu] + weights[q_cpu])
            pair_each = torch.nn.functional.softplus(-(p_logit - q_logit))
            pair_w_dev = pair_w.to(device, non_blocking=True)
            bpr_loss = (pair_each * pair_w_dev).sum() / pair_w_dev.sum().clamp_min(1e-8)
            loss = 0.5 * bce_loss + 0.5 * bpr_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            running_loss += float(loss.detach().cpu())
            batches += 1
            processed = min(start + len(idx), n)
            if half_checkpoints and not evaluated_half and processed >= (n + 1) // 2:
                scores = predict(model, xv, device)
                metrics = metric_values(evaluator, val_users, val_y, scores)
                checkpoints.append({'epoch': epoch + 0.5, 'train_loss': running_loss / batches,
                                    'val_gauc': metrics['gauc'], 'val_primary': metrics['primary']})
                if metrics['primary'] > best_primary:
                    best_primary = metrics['primary']
                    best_scores = scores.copy()
                    if keep_state:
                        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                model.train()
                evaluated_half = True
        if (epoch + 1) % int(config['step_every']) == 0:
            for group_opt in opt.param_groups:
                group_opt['lr'] *= float(config['gamma'])
        scores = predict(model, xv, device)
        metrics = metric_values(evaluator, val_users, val_y, scores)
        checkpoints.append({'epoch': epoch + 1.0, 'train_loss': running_loss / max(batches, 1),
                            'val_gauc': metrics['gauc'], 'val_primary': metrics['primary']})
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


def train_parent_reference(seed, epochs, xt, yt, xv, val_users, val_y, total_dim, device, evaluator):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = FM(total_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n = len(yt)
    batch_size = 8192
    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed)
    best = -1.0
    best_scores = None
    patience = 0
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=generator)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb = xt[idx].to(device, non_blocking=True)
            yb = yt[idx].to(device, non_blocking=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        scores = predict(model, xv, device)
        primary = metric_values(evaluator, val_users, val_y, scores)['primary']
        if primary > best + 1e-6:
            best = primary
            best_scores = scores.copy()
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break
    return best_scores, best


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


def coarse_configs(seed):
    rng = np.random.default_rng(seed + 8101)
    drops = np.linspace(0.15, 0.40, 12)
    wds = np.logspace(math.log10(3e-5), math.log10(3e-3), 12)
    rng.shuffle(drops)
    rng.shuffle(wds)
    half_lives = [3.5, 7.0, 14.0]
    schedules = [(1, 0.45), (1, 0.65), (2, 0.40), (2, 0.60), (3, 0.50), (3, 0.72)]
    lrs = [0.00055, 0.00075, 0.0010, 0.0013]
    configs = []
    for i in range(12):
        step_every, gamma = schedules[i % len(schedules)]
        configs.append({
            'dropout': float(drops[i]),
            'weight_decay': float(wds[i]),
            'half_life': float(half_lives[(i * 2) % 3]),
            'step_every': int(step_every),
            'gamma': float(gamma),
            'lr': float(lrs[(i * 3) % 4])
        })
    return configs


def refine_configs(winner, seed):
    rng = np.random.default_rng(seed + 19001)
    configs = []
    drop_offsets = np.linspace(-0.07, 0.07, 10)
    wd_factors = np.exp(np.linspace(-0.9, 0.9, 10))
    lr_factors = np.linspace(0.72, 1.28, 10)
    rng.shuffle(drop_offsets)
    rng.shuffle(wd_factors)
    rng.shuffle(lr_factors)
    half_candidates = sorted(set([3.5, 5.0, 7.0, 10.0, 14.0, float(winner['half_life'])]))
    gamma_offsets = np.linspace(-0.12, 0.12, 10)
    rng.shuffle(gamma_offsets)
    for i in range(10):
        configs.append({
            'dropout': float(np.clip(winner['dropout'] + drop_offsets[i], 0.10, 0.48)),
            'weight_decay': float(np.clip(winner['weight_decay'] * wd_factors[i], 1e-5, 6e-3)),
            'half_life': float(half_candidates[i % len(half_candidates)]),
            'step_every': int(max(1, min(3, winner['step_every'] + ((i % 3) - 1)))),
            'gamma': float(np.clip(winner['gamma'] + gamma_offsets[i], 0.28, 0.82)),
            'lr': float(np.clip(winner['lr'] * lr_factors[i], 0.00035, 0.0017))
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
    total_dim = int(np.asarray(tr['field_dims']).sum())
    pair_data = make_pair_index(train_users, np.asarray(tr['y'], dtype=np.float32))
    if len(pair_data[0]) == 0:
        raise RuntimeError('No within-user positive-negative training pairs are available')

    smoke = os.environ.get('SMOKE_EPOCHS')
    smoke_cap = int(smoke) if smoke is not None else None
    coarse_epochs = min(3, smoke_cap) if smoke_cap is not None else 3
    refine_epochs = min(6, smoke_cap) if smoke_cap is not None else 6
    final_epochs = min(args.epochs, smoke_cap) if smoke_cap is not None else args.epochs
    reference_epochs = min(12, smoke_cap) if smoke_cap is not None else 12
    recency_cache = {}
    history = []

    best_config = None
    best_probe = -1.0
    for i, config in enumerate(coarse_configs(args.seed)):
        _, score, checkpoints = train_package(
            config, args.seed + 101 + i, coarse_epochs, xt, yt, xv, train_users,
            train_dates, val_users, val_y, total_dim, device, evaluator, pair_data,
            recency_cache, False, False)
        record = {'stage': 'coarse', 'probe': i + 1, 'config': config,
                  'epochs': coarse_epochs, 'primary': float(score), 'checkpoints': checkpoints}
        history.append(record)
        append_progress(progress_path, {'stage': 'coarse', 'probe': i + 1,
                                        'config': config, 'primary': float(score)})
        if score > best_probe:
            best_probe = score
            best_config = dict(config)

    refine_best = best_probe
    refine_config = dict(best_config)
    for i, config in enumerate(refine_configs(best_config, args.seed)):
        _, score, checkpoints = train_package(
            config, args.seed + 401 + i, refine_epochs, xt, yt, xv, train_users,
            train_dates, val_users, val_y, total_dim, device, evaluator, pair_data,
            recency_cache, False, False)
        record = {'stage': 'refine', 'probe': i + 1, 'config': config,
                  'epochs': refine_epochs, 'primary': float(score), 'checkpoints': checkpoints}
        history.append(record)
        append_progress(progress_path, {'stage': 'refine', 'probe': i + 1,
                                        'config': config, 'primary': float(score)})
        if score > refine_best:
            refine_best = score
            refine_config = dict(config)

    parent_scores, parent_primary = train_parent_reference(
        args.seed, reference_epochs, xt, yt, xv, val_users, val_y,
        total_dim, device, evaluator)
    append_progress(progress_path, {'stage': 'parent_reference',
                                    'seed': args.seed, 'primary': float(parent_primary)})
    history.append({'stage': 'parent_reference', 'seed': args.seed,
                    'epochs': reference_epochs, 'primary': float(parent_primary)})

    member_scores = []
    member_records = []
    for member in range(5):
        member_seed = args.seed + member
        scores, primary, checkpoints = train_package(
            refine_config, member_seed, final_epochs, xt, yt, xv, train_users,
            train_dates, val_users, val_y, total_dim, device, evaluator, pair_data,
            recency_cache, True, True)
        if np.allclose(scores, parent_scores, rtol=1e-6, atol=1e-7):
            raise AssertionError('Final ensemble member is identical to the parent prediction vector')
        for prior in member_scores:
            if np.allclose(scores, prior, rtol=1e-6, atol=1e-7):
                raise AssertionError('Distinct-seed ensemble members produced identical predictions')
        member_scores.append(scores)
        member_record = {'stage': 'final_member', 'member': member + 1,
                         'seed': member_seed, 'config': refine_config,
                         'epochs': final_epochs, 'primary': float(primary),
                         'checkpoints': checkpoints}
        member_records.append(member_record)
        history.append(member_record)
        append_progress(progress_path, {'stage': 'final_member', 'member': member + 1,
                                        'seed': member_seed, 'primary': float(primary)})

    ranked = [rank_within_user(val_users, scores) for scores in member_scores]
    final_scores = np.mean(np.stack(ranked, axis=0), axis=0)
    if np.allclose(final_scores, rank_within_user(val_users, parent_scores), rtol=1e-6, atol=1e-7):
        raise AssertionError('Final ensemble predictions are identical to parent ranked predictions')
    metrics = metric_values(evaluator, val_users, val_y, final_scores)
    history.append({'stage': 'ensemble', 'members': 5, 'seeds': [args.seed + i for i in range(5)],
                    'config': refine_config, 'primary': metrics['primary']})
    append_progress(progress_path, {'stage': 'ensemble', 'members': 5,
                                    'primary': metrics['primary']})

    metrics_out = {
        'gauc': metrics['gauc'],
        'ndcg5': metrics['ndcg5'],
        'primary': metrics['primary'],
        'selected_config': refine_config,
        'parent_reference_primary': float(parent_primary),
        'member_primaries': [float(r['primary']) for r in member_records],
        'history': history
    }
    with open(os.path.join(args.out_dir, 'metrics.json'), 'w') as fh:
        json.dump(metrics_out, fh)

    video_values = np.asarray(va.get('video_raw', np.zeros(len(final_scores), dtype=np.int64)))
    with open(os.path.join(args.out_dir, 'predictions.csv'), 'w') as fh:
        fh.write('row_id,user_id,video_id,score\n')
        for i, score in enumerate(final_scores):
            fh.write(f'{i},{val_users[i]},{video_values[i]},{score:.9g}\n')


if __name__ == '__main__':
    main()
