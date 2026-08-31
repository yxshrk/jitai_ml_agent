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
    def __init__(self, total_dim, n_fields=5, k=16, hidden=128, dropout=0.25):
        super().__init__()
        self.n_fields = n_fields
        self.k = k
        d = n_fields * k
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.emb_drop = torch.nn.Dropout(dropout)
        self.cross_w = torch.nn.Parameter(torch.empty(d))
        self.cross_b = torch.nn.Parameter(torch.zeros(d))
        self.cross_out = torch.nn.Linear(d, 1, bias=False)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(d, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, hidden // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden // 2, 1),
        )
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)
        torch.nn.init.normal_(self.cross_w, std=0.01)
        torch.nn.init.xavier_uniform_(self.cross_out.weight)
        for layer in self.mlp:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

    def forward(self, x):
        raw = self.emb(x)
        fm_sum = raw.sum(1)
        fm = 0.5 * (fm_sum.square() - raw.square().sum(1)).sum(1)
        z0 = self.emb_drop(raw).reshape(x.shape[0], -1)
        cross = z0 * (z0 @ self.cross_w).unsqueeze(1) + self.cross_b + z0
        linear = self.lin(x).sum((1, 2))
        return self.bias + linear + fm + self.cross_out(cross).squeeze(1) + self.mlp(z0).squeeze(1)


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def date_ordinals(values):
    a = np.asarray(values)
    out = np.empty(len(a), dtype=np.float32)
    cache = {}
    for i, value in enumerate(a):
        try:
            key = int(value)
        except Exception:
            key = 0
        if key not in cache:
            text = str(key)
            try:
                if len(text) == 8:
                    cache[key] = datetime.date(int(text[:4]), int(text[4:6]), int(text[6:8])).toordinal()
                else:
                    cache[key] = key
            except Exception:
                cache[key] = key
        out[i] = cache[key]
    return out


def encode_column(train_values, val_values):
    mapping = {}
    train_encoded = np.empty(len(train_values), dtype=np.int64)
    for i, value in enumerate(train_values):
        key = str(value)
        if key not in mapping:
            mapping[key] = len(mapping)
        train_encoded[i] = mapping[key]
    oov = len(mapping)
    val_encoded = np.asarray([mapping.get(str(v), oov) for v in val_values], dtype=np.int64)
    return train_encoded, val_encoded, oov + 1


def load_csv_fallback(data_dir):
    def read_split(path, training):
        columns = {k: [] for k in ['user_id', 'video_id', 'tab', 'duration_ms', 'date', 'long_view']}
        with open(path, 'r', newline='') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                columns['user_id'].append(row['user_id'])
                columns['video_id'].append(row['video_id'])
                columns['tab'].append(row['tab'])
                columns['duration_ms'].append(float(row['duration_ms']))
                columns['date'].append(row['date'])
                columns['long_view'].append(float(row['long_view']))
        return columns

    tr = read_split(os.path.join(data_dir, 'train.csv'), True)
    va = read_split(os.path.join(data_dir, 'val.csv'), False)
    tu, vu, du = encode_column(tr['user_id'], va['user_id'])
    tv, vv, dv = encode_column(tr['video_id'], va['video_id'])
    tt, vt, dt = encode_column(tr['tab'], va['tab'])
    author_train = tv.copy()
    author_val = vv.copy()
    da = dv
    train_duration = np.asarray(tr['duration_ms'], dtype=np.float64)
    val_duration = np.asarray(va['duration_ms'], dtype=np.float64)
    quantiles = np.unique(np.quantile(train_duration, np.linspace(0.1, 0.9, 9)))
    td = np.searchsorted(quantiles, train_duration, side='right').astype(np.int64)
    vd = np.searchsorted(quantiles, val_duration, side='right').astype(np.int64)
    dd = len(quantiles) + 1
    dims = np.asarray([du, dv, da, dt, dd], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(dims)[:-1]))
    Xt = np.stack([tu, tv, author_train, tt, td], axis=1) + offsets
    Xv = np.stack([vu, vv, author_val, vt, vd], axis=1) + offsets
    train = {
        'X': Xt.astype(np.int64),
        'y': np.asarray(tr['long_view'], dtype=np.float32),
        'user': np.asarray(tr['user_id']),
        'date': np.asarray(tr['date']),
        'field_dims': dims,
    }
    val = {
        'X': Xv.astype(np.int64),
        'y': np.asarray(va['long_view'], dtype=np.float32),
        'user': np.asarray(va['user_id']),
        'video': np.asarray(va['video_id']),
    }
    return train, val, False


def load_data(data_dir):
    train_npz = os.path.join(data_dir, 'train.npz')
    val_npz = os.path.join(data_dir, 'val.npz')
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        tr_file = np.load(train_npz)
        va_file = np.load(val_npz)
        train = {k: tr_file[k] for k in tr_file.files}
        val = {k: va_file[k] for k in va_file.files}
        val['video'] = np.zeros(len(val['y']), dtype=np.int64)
        return train, val, True
    return load_csv_fallback(data_dir)


def build_pair_pools(users, labels):
    users = np.asarray(users)
    labels = np.asarray(labels)
    order = np.argsort(users, kind='mergesort')
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    positives = []
    negatives = []
    pos_offsets = []
    neg_offsets = []
    pos_counts = []
    neg_counts = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        idx = order[left:right]
        pos = idx[labels[idx] > 0.5]
        neg = idx[labels[idx] <= 0.5]
        if len(pos) and len(neg):
            pos_offsets.append(len(positives))
            neg_offsets.append(len(negatives))
            pos_counts.append(len(pos))
            neg_counts.append(len(neg))
            positives.extend(pos.tolist())
            negatives.extend(neg.tolist())
    return {
        'pos': np.asarray(positives, dtype=np.int64),
        'neg': np.asarray(negatives, dtype=np.int64),
        'po': np.asarray(pos_offsets, dtype=np.int64),
        'no': np.asarray(neg_offsets, dtype=np.int64),
        'pc': np.asarray(pos_counts, dtype=np.int64),
        'nc': np.asarray(neg_counts, dtype=np.int64),
    }


def sample_pairs(pool, count, rng):
    group = rng.integers(0, len(pool['po']), size=count)
    pslot = pool['po'][group] + np.floor(rng.random(count) * pool['pc'][group]).astype(np.int64)
    nslot = pool['no'][group] + np.floor(rng.random(count) * pool['nc'][group]).astype(np.int64)
    return pool['pos'][pslot], pool['neg'][nslot]


def metric_values(evaluator, users, labels, scores):
    result = evaluator(users, labels.astype(int), scores)
    return {
        'gauc': float(result['GAUC'] if 'GAUC' in result else result['gauc']),
        'ndcg5': float(result.get('nDCG@5', result.get('ndcg5'))),
        'primary': float(result['primary']),
    }


def predict(model, Xv, device, batch_size=65536):
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(Xv), batch_size):
            xb = torch.as_tensor(Xv[start:start + batch_size], dtype=torch.long, device=device)
            chunks.append(model(xb).detach().cpu().numpy())
    return np.concatenate(chunks)


def set_learning_rate(optimizer, base_lr, completed_epochs, step_epochs, gamma):
    exponent = int(math.floor((completed_epochs + 1e-9) / step_epochs))
    lr = base_lr * (gamma ** exponent)
    for group in optimizer.param_groups:
        group['lr'] = lr
    return lr


def train_candidate(config, train, val, pair_pool, recency_age, evaluator, device,
                    seed, epochs, checkpoint_half_epochs):
    seed_all(seed)
    rng = np.random.default_rng(seed)
    X = np.asarray(train['X'], dtype=np.int64)
    y_np = np.asarray(train['y'], dtype=np.float32)
    Xv = np.asarray(val['X'], dtype=np.int64)
    total_dim = int(np.asarray(train['field_dims']).sum())
    model = DCNLite(total_dim, n_fields=X.shape[1], k=16, hidden=128,
                    dropout=float(config['dropout'])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['lr']),
                                  weight_decay=float(config['weight_decay']))
    n = len(y_np)
    batch_size = 8192 if device.type == 'cuda' else 4096
    pair_batch = max(512, batch_size // 2)
    y = torch.as_tensor(y_np, dtype=torch.float32, device=device)
    recency_np = np.exp2(-recency_age / float(config['half_life'])).astype(np.float32)
    weights = torch.as_tensor(recency_np, dtype=torch.float32, device=device)
    best_primary = -1.0
    best_scores = None
    best_metrics = None
    curve = []
    global_batches = int(math.ceil(n / batch_size))
    checkpoints = {global_batches - 1}
    if checkpoint_half_epochs:
        checkpoints.add(max(0, global_batches // 2 - 1))
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(n, device=device)
        pair_count = global_batches * pair_batch
        pos_idx, neg_idx = sample_pairs(pair_pool, pair_count, rng)
        last_loss = 0.0
        for batch_number, start in enumerate(range(0, n, batch_size)):
            completed = epoch + batch_number / max(1, global_batches)
            current_lr = set_learning_rate(optimizer, float(config['lr']), completed,
                                           float(config['step_epochs']), float(config['gamma']))
            idx = permutation[start:start + batch_size]
            xb = torch.as_tensor(X[idx.detach().cpu().numpy()], dtype=torch.long, device=device)
            logits = model(xb)
            raw_bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, y[idx], reduction='none')
            bce = (raw_bce * weights[idx]).sum() / weights[idx].sum().clamp_min(1e-8)
            p0 = batch_number * pair_batch
            p1 = p0 + pair_batch
            pi_np = pos_idx[p0:p1]
            ni_np = neg_idx[p0:p1]
            pi = torch.as_tensor(pi_np, dtype=torch.long, device=device)
            ni = torch.as_tensor(ni_np, dtype=torch.long, device=device)
            pair_x = np.concatenate([X[pi_np], X[ni_np]], axis=0)
            pair_logits = model(torch.as_tensor(pair_x, dtype=torch.long, device=device))
            pscore = pair_logits[:len(pi_np)]
            nscore = pair_logits[len(pi_np):]
            pair_weight = 0.5 * (weights[pi] + weights[ni])
            raw_pair = torch.nn.functional.softplus(-(pscore - nscore))
            bpr = (raw_pair * pair_weight).sum() / pair_weight.sum().clamp_min(1e-8)
            loss = 0.5 * bce + 0.5 * bpr
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            last_loss = float(loss.detach().cpu())
            if batch_number in checkpoints:
                scores = predict(model, Xv, device)
                metrics = metric_values(evaluator, np.asarray(val['user']), np.asarray(val['y']), scores)
                point = {
                    'epoch': round(epoch + (batch_number + 1) / global_batches, 3),
                    'train_loss': round(last_loss, 6),
                    'lr': float(current_lr),
                    'primary': metrics['primary'],
                }
                curve.append(point)
                if metrics['primary'] > best_primary:
                    best_primary = metrics['primary']
                    best_scores = scores.copy()
                    best_metrics = metrics
                model.train()
    del optimizer
    model.to('cpu')
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_metrics, best_scores, curve


def coarse_configs(seed):
    rng = np.random.default_rng(seed + 1771)
    configs = []
    half_lives = [3.5, 7.0, 14.0]
    step_choices = [0.75, 1.0, 1.5, 2.0, 2.75]
    gamma_choices = [0.35, 0.45, 0.55, 0.68, 0.78]
    for i in range(12):
        configs.append({
            'dropout': float(rng.uniform(0.15, 0.40)),
            'weight_decay': float(10 ** rng.uniform(math.log10(3e-5), math.log10(3e-3))),
            'lr': float(10 ** rng.uniform(math.log10(3.5e-4), math.log10(1.6e-3))),
            'step_epochs': float(step_choices[i % len(step_choices)]),
            'gamma': float(gamma_choices[(i * 2) % len(gamma_choices)]),
            'half_life': float(half_lives[(i * 2) % len(half_lives)]),
        })
    return configs


def refined_configs(winner):
    patterns = [
        (0.00, 1.00, 1.00, 1.00, 0.00, 1.00),
        (-0.025, 0.55, 0.82, 0.82, -0.06, 0.75),
        (-0.012, 0.75, 1.18, 1.18, 0.04, 1.00),
        (0.012, 1.35, 0.90, 0.92, -0.03, 1.25),
        (0.025, 1.80, 1.10, 1.08, 0.06, 1.50),
        (-0.035, 1.20, 1.00, 1.32, 0.02, 0.75),
        (0.035, 0.70, 1.28, 0.72, -0.08, 1.25),
        (0.000, 1.55, 0.74, 1.45, 0.08, 1.00),
    ]
    result = []
    for dd, wm, lm, sm, dg, hm in patterns:
        result.append({
            'dropout': float(np.clip(winner['dropout'] + dd, 0.12, 0.45)),
            'weight_decay': float(np.clip(winner['weight_decay'] * wm, 2e-5, 5e-3)),
            'lr': float(np.clip(winner['lr'] * lm, 2.5e-4, 2e-3)),
            'step_epochs': float(np.clip(winner['step_epochs'] * sm, 0.6, 3.5)),
            'gamma': float(np.clip(winner['gamma'] + dg, 0.25, 0.85)),
            'half_life': float(np.clip(winner['half_life'] * hm, 2.5, 18.0)),
        })
    return result


def append_progress(path, record):
    with open(path, 'a') as fh:
        fh.write(json.dumps(record, sort_keys=True) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=16)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    progress_path = os.path.join(args.out_dir, 'progress.log')
    if os.path.exists(progress_path):
        os.remove(progress_path)
    seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train, val, fast_path = load_data(args.data_dir)
    if fast_path:
        from data.official.evaluate import evaluate as evaluator
    else:
        from harness.evaluate_provisional import evaluate as evaluator
    ages = date_ordinals(train['date'])
    recency_age = ages.max() - ages
    pair_pool = build_pair_pools(train['user'], train['y'])
    if len(pair_pool['po']) == 0:
        raise RuntimeError('No users with both positive and negative labels for BPR training')
    smoke = os.environ.get('SMOKE_EPOCHS')
    smoke_cap = int(smoke) if smoke is not None else None
    coarse_epochs = min(3, smoke_cap) if smoke_cap is not None else 3
    refine_epochs = min(5, smoke_cap) if smoke_cap is not None else 5
    final_epochs = min(args.epochs, smoke_cap) if smoke_cap is not None else args.epochs
    repetitions = 1 if smoke_cap is not None else 6
    coarse = coarse_configs(args.seed)
    if smoke_cap is not None:
        coarse = coarse[:2]
    history = []
    coarse_summary = []
    probe_number = 0
    for config_id, config in enumerate(coarse):
        scores = []
        for rep in range(repetitions):
            probe_seed = args.seed + 1000 + config_id * 31 + rep
            metrics, _, curve = train_candidate(config, train, val, pair_pool, recency_age,
                                                  evaluator, device, probe_seed,
                                                  coarse_epochs, False)
            probe_number += 1
            record = {
                'stage': 'coarse', 'probe': probe_number, 'config_id': config_id,
                'replicate': rep, 'seed': probe_seed, 'config': config,
                'primary': metrics['primary'], 'gauc': metrics['gauc'], 'curve': curve,
            }
            history.append(record)
            append_progress(progress_path, record)
            scores.append(metrics['primary'])
        coarse_summary.append({'config': config, 'mean': float(np.mean(scores)),
                               'std': float(np.std(scores)), 'scores': scores})
    coarse_summary.sort(key=lambda x: x['mean'], reverse=True)
    stage1_winner = coarse_summary[0]['config']
    refined = refined_configs(stage1_winner)
    if smoke_cap is not None:
        refined = refined[:2]
    refine_summary = []
    for config_id, config in enumerate(refined):
        scores = []
        for rep in range(repetitions):
            probe_seed = args.seed + 100000 + config_id * 37 + rep
            metrics, _, curve = train_candidate(config, train, val, pair_pool, recency_age,
                                                  evaluator, device, probe_seed,
                                                  refine_epochs, True)
            probe_number += 1
            record = {
                'stage': 'refine', 'probe': probe_number, 'config_id': config_id,
                'replicate': rep, 'seed': probe_seed, 'config': config,
                'primary': metrics['primary'], 'gauc': metrics['gauc'], 'curve': curve,
            }
            history.append(record)
            append_progress(progress_path, record)
            scores.append(metrics['primary'])
        refine_summary.append({'config': config, 'mean': float(np.mean(scores)),
                               'std': float(np.std(scores)), 'scores': scores})
    refine_summary.sort(key=lambda x: x['mean'], reverse=True)
    winning_config = refine_summary[0]['config']
    final_seed = args.seed + 900001
    final_metrics, final_scores, final_curve = train_candidate(
        winning_config, train, val, pair_pool, recency_age, evaluator, device,
        final_seed, final_epochs, True)
    final_record = {
        'stage': 'final', 'seed': final_seed, 'epochs': final_epochs,
        'config': winning_config, 'primary': final_metrics['primary'],
        'gauc': final_metrics['gauc'], 'curve': final_curve,
    }
    history.append(final_record)
    append_progress(progress_path, final_record)
    output_metrics = {
        'gauc': final_metrics['gauc'],
        'ndcg5': final_metrics['ndcg5'],
        'primary': final_metrics['primary'],
        'winning_config': winning_config,
        'coarse_summary': coarse_summary,
        'refine_summary': refine_summary,
        'history': history,
    }
    with open(os.path.join(args.out_dir, 'metrics.json'), 'w') as fh:
        json.dump(output_metrics, fh)
    users = np.asarray(val['user'])
    videos = np.asarray(val['video'])
    with open(os.path.join(args.out_dir, 'predictions.csv'), 'w') as fh:
        fh.write('row_id,user_id,video_id,score\n')
        for i, score in enumerate(final_scores):
            fh.write(f'{i},{users[i]},{videos[i]},{score:.8g}\n')


if __name__ == '__main__':
    main()
