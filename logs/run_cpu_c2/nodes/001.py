import argparse
import csv
import datetime
import json
import os
import sys
import time

import numpy as np
import torch


class DCNLite(torch.nn.Module):
    def __init__(self, total_dim, fields=5, k=16, hidden=128, dropout=0.25):
        super().__init__()
        self.fields = fields
        self.k = k
        d = fields * k
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.emb_drop = torch.nn.Dropout(dropout)
        self.cross_w = torch.nn.ParameterList([
            torch.nn.Parameter(torch.empty(d)) for _ in range(2)
        ])
        self.cross_b = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros(d)) for _ in range(2)
        ])
        self.cross_out = torch.nn.Linear(d, 1)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(d, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(64, 1),
        )
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)
        for w in self.cross_w:
            torch.nn.init.normal_(w, std=0.01)
        torch.nn.init.xavier_uniform_(self.cross_out.weight)
        torch.nn.init.zeros_(self.cross_out.bias)
        for layer in self.mlp:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

    def forward(self, x):
        e0 = self.emb(x)
        e = self.emb_drop(e0)
        summed = e.sum(1)
        fm = 0.5 * (summed.square() - e.square().sum(1)).sum(1)
        linear = self.bias + self.lin(x).sum((1, 2))
        x0 = e.flatten(1)
        xl = x0
        for w, b in zip(self.cross_w, self.cross_b):
            xl = xl + x0 * (xl * w).sum(1, keepdim=True) + b
        return linear + fm + self.cross_out(xl).squeeze(1) + self.mlp(x0).squeeze(1)


def parse_day(value):
    s = str(value)
    if s.endswith('.0'):
        s = s[:-2]
    digits = ''.join(ch for ch in s if ch.isdigit())[:8]
    try:
        return datetime.date(int(digits[:4]), int(digits[4:6]), int(digits[6:8])).toordinal()
    except Exception:
        return 0


def recency_weights(dates, half_life):
    arr = np.asarray(dates)
    unique, inv = np.unique(arr.astype(str), return_inverse=True)
    ordinals = np.asarray([parse_day(x) for x in unique], dtype=np.float32)
    newest = float(ordinals.max()) if len(ordinals) else 0.0
    age = newest - ordinals[inv]
    w = np.exp2(-age / float(half_life)).astype(np.float32)
    return w / max(float(w.mean()), 1e-8)


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
        columns = {'user_id': [], 'video_id': [], 'tab': [], 'duration_ms': [],
                   'date': [], 'long_view': []}
        with open(path, newline='') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                columns['user_id'].append(row['user_id'])
                columns['video_id'].append(row['video_id'])
                columns['tab'].append(row['tab'])
                columns['duration_ms'].append(float(row['duration_ms']))
                columns['date'].append(row['date'])
                columns['long_view'].append(float(row['long_view']))
        return columns

    tr = read_file(os.path.join(data_dir, 'train.csv'), True)
    va = read_file(os.path.join(data_dir, 'val.csv'), False)
    duration_train = np.asarray(tr['duration_ms'], dtype=np.float64)
    duration_val = np.asarray(va['duration_ms'], dtype=np.float64)
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
        'Xv': np.stack(vx, axis=1).astype(np.int64),
        'yv': np.asarray(va['long_view'], dtype=np.float32),
        'val_user': np.asarray(va['user_id']),
        'val_video': np.asarray(va['video_id']),
        'field_dims': np.asarray(dims, dtype=np.int64),
        'fast': False,
    }


def load_data(data_dir):
    train_npz = os.path.join(data_dir, 'train.npz')
    val_npz = os.path.join(data_dir, 'val.npz')
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        tr = np.load(train_npz)
        va = np.load(val_npz)
        dims = tr['field_dims'].astype(np.int64)
        video_offset = int(dims[0])
        val_video = va['X'][:, 1].astype(np.int64) - video_offset
        return {
            'Xt': tr['X'].astype(np.int64),
            'yt': tr['y'].astype(np.float32),
            'train_user': np.asarray(tr['user']),
            'train_date': np.asarray(tr['date']),
            'Xv': va['X'].astype(np.int64),
            'yv': va['y'].astype(np.float32),
            'val_user': np.asarray(va['user']),
            'val_video': val_video,
            'field_dims': dims,
            'fast': True,
        }
    return load_csv_fallback(data_dir)


def make_pairs(users, labels, seed):
    order = np.argsort(users, kind='stable')
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    rng = np.random.RandomState(seed)
    positives = []
    negatives = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
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
    return np.concatenate(positives).astype(np.int64), np.concatenate(negatives).astype(np.int64)


def metric_dict(evaluate_fn, users, labels, scores):
    m = evaluate_fn(users, labels.astype(int), scores)
    return {
        'gauc': float(m['GAUC'] if 'GAUC' in m else m['gauc']),
        'ndcg5': float(m.get('nDCG@5', m.get('ndcg5'))),
        'primary': float(m['primary']),
    }


def predict(model, Xv, device, batch_size=65536):
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(Xv), batch_size):
            xb = torch.as_tensor(Xv[start:start + batch_size], dtype=torch.long, device=device)
            output.append(model(xb).detach().cpu().numpy())
    return np.concatenate(output).astype(np.float64)


def train_once(data, pairs, config, seed, epochs, evaluate_fn, device,
               half_epoch_checkpoints=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    model = DCNLite(int(data['field_dims'].sum()), dropout=float(config['dropout'])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['lr']),
                                  weight_decay=float(config['weight_decay']))
    Xt = data['Xt']
    yt = data['yt']
    weights = recency_weights(data['train_date'], float(config['half_life']))
    pos_idx, neg_idx = pairs
    n = len(yt)
    pair_n = len(pos_idx)
    batch_size = 8192
    rng = np.random.RandomState(seed + 9173)
    best_primary = -1.0
    best_scores = None
    curve = []
    global_step = 0
    checks_per_epoch = 2 if half_epoch_checkpoints else 1
    for epoch in range(epochs):
        permutation = rng.permutation(n)
        pair_perm = rng.permutation(pair_n) if pair_n else np.empty(0, dtype=np.int64)
        chunks = np.array_split(permutation, checks_per_epoch)
        for half, chunk in enumerate(chunks):
            model.train()
            last_loss = 0.0
            progress = epoch + half / checks_per_epoch
            decay_power = int(progress // float(config['step_every']))
            lr = float(config['lr']) * (float(config['gamma']) ** decay_power)
            for group in optimizer.param_groups:
                group['lr'] = lr
            for start in range(0, len(chunk), batch_size):
                idx_np = chunk[start:start + batch_size]
                idx = torch.as_tensor(idx_np, dtype=torch.long, device=device)
                xb = torch.as_tensor(Xt[idx_np], dtype=torch.long, device=device)
                yb = torch.as_tensor(yt[idx_np], dtype=torch.float32, device=device)
                wb = torch.as_tensor(weights[idx_np], dtype=torch.float32, device=device)
                logits = model(xb)
                bce = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, yb, reduction='none')
                bce_loss = (bce * wb).sum() / wb.sum().clamp_min(1e-8)
                if pair_n:
                    take = len(idx_np)
                    base = (global_step * batch_size) % pair_n
                    locations = np.arange(base, base + take, dtype=np.int64) % pair_n
                    chosen = pair_perm[locations]
                    pi = pos_idx[chosen]
                    ni = neg_idx[chosen]
                    xp = torch.as_tensor(Xt[pi], dtype=torch.long, device=device)
                    xn = torch.as_tensor(Xt[ni], dtype=torch.long, device=device)
                    pair_w = torch.as_tensor(weights[pi], dtype=torch.float32, device=device)
                    pair_loss_raw = torch.nn.functional.softplus(-(model(xp) - model(xn)))
                    pair_loss = (pair_loss_raw * pair_w).sum() / pair_w.sum().clamp_min(1e-8)
                    loss = 0.5 * bce_loss + 0.5 * pair_loss
                else:
                    loss = bce_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                last_loss = float(loss.detach().cpu())
                global_step += 1
            scores = predict(model, data['Xv'], device)
            metrics = metric_dict(evaluate_fn, data['val_user'], data['yv'], scores)
            checkpoint = epoch + (half + 1) / checks_per_epoch
            curve.append({'checkpoint': checkpoint, 'train_loss': round(last_loss, 6),
                          'lr': lr, 'val_gauc': round(metrics['gauc'], 6),
                          'val_primary': round(metrics['primary'], 6)})
            if metrics['primary'] > best_primary + 1e-8:
                best_primary = metrics['primary']
                best_scores = scores.copy()
    del model, optimizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_primary, best_scores, curve


def append_progress(path, record):
    with open(path, 'a') as fh:
        fh.write(json.dumps(record, sort_keys=True) + '\n')


def stage1_configs(seed):
    rng = np.random.RandomState(seed + 441)
    configs = []
    half_lives = [3.5, 7.0, 14.0]
    gammas = [0.24, 0.36, 0.52]
    steps = [1.5, 2.0, 2.75]
    for i in range(12):
        configs.append({
            'dropout': float(rng.uniform(0.15, 0.40)),
            'weight_decay': float(10.0 ** rng.uniform(np.log10(3e-5), np.log10(3e-3))),
            'lr': float(10.0 ** rng.uniform(np.log10(5.5e-4), np.log10(1.45e-3))),
            'gamma': float(gammas[i % len(gammas)]),
            'step_every': float(steps[(i // 3) % len(steps)]),
            'half_life': float(half_lives[(i * 2 + i // 4) % len(half_lives)]),
        })
    return configs


def refine_configs(winner):
    adjustments = [
        (-0.055, 0.50, 0.84, 0.88, 0.82, 0.80),
        (-0.025, 0.72, 1.00, 1.00, 1.00, 1.00),
        (0.000, 1.00, 0.82, 1.12, 0.85, 1.00),
        (0.000, 1.00, 1.00, 1.00, 1.00, 1.00),
        (0.000, 1.00, 1.18, 0.88, 1.18, 1.00),
        (0.025, 1.38, 1.00, 1.10, 1.00, 1.00),
        (0.055, 1.90, 0.86, 1.00, 1.16, 1.25),
        (0.030, 0.62, 1.16, 0.92, 0.90, 0.75),
    ]
    result = []
    for dd, wd, lr, gamma, step, half in adjustments:
        result.append({
            'dropout': float(np.clip(winner['dropout'] + dd, 0.10, 0.46)),
            'weight_decay': float(np.clip(winner['weight_decay'] * wd, 1e-5, 6e-3)),
            'lr': float(np.clip(winner['lr'] * lr, 3e-4, 2e-3)),
            'gamma': float(np.clip(winner['gamma'] * gamma, 0.16, 0.68)),
            'step_every': float(np.clip(winner['step_every'] * step, 1.0, 3.5)),
            'half_life': float(np.clip(winner['half_life'] * half, 2.5, 18.0)),
        })
    return result


def rank_within_user(users, scores):
    result = np.empty(len(scores), dtype=np.float64)
    order = np.argsort(users, kind='stable')
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        idx = order[left:right]
        local_order = np.argsort(scores[idx], kind='mergesort')
        ranks = np.empty(len(idx), dtype=np.float64)
        if len(idx) == 1:
            ranks[0] = 0.5
        else:
            ranks[local_order] = np.arange(len(idx), dtype=np.float64) / (len(idx) - 1.0)
        result[idx] = ranks
    return result


def select_config(configs, records, stage):
    best_index = 0
    best_mean = -1.0
    for i in range(len(configs)):
        values = [r['primary'] for r in records if r['stage'] == stage and r['config_index'] == i]
        mean_value = float(np.mean(values))
        if mean_value > best_mean:
            best_mean = mean_value
            best_index = i
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
        torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

    data = load_data(args.data_dir)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, root)
    if data['fast']:
        from data.official.evaluate import evaluate as evaluate_fn
    else:
        from harness.evaluate_provisional import evaluate as evaluate_fn

    pairs = make_pairs(data['train_user'], data['yt'], args.seed + 71)
    smoke_value = os.environ.get('SMOKE_EPOCHS')
    smoke = smoke_value is not None
    smoke_cap = int(smoke_value) if smoke else None

    def capped(value):
        return max(1, min(value, smoke_cap)) if smoke else value

    history = []
    coarse = stage1_configs(args.seed)
    coarse_seeds = [args.seed, args.seed + 1]
    if smoke:
        coarse = coarse[:1]
        coarse_seeds = coarse_seeds[:1]
    for config_index, config in enumerate(coarse):
        for probe_seed in coarse_seeds:
            started = time.time()
            primary, _, curve = train_once(data, pairs, config, probe_seed, capped(2),
                                            evaluate_fn, device, False)
            record = {'stage': 'coarse', 'config_index': config_index,
                      'seed': probe_seed, 'config': config, 'primary': primary,
                      'epochs': capped(2), 'seconds': round(time.time() - started, 3),
                      'curve': curve}
            history.append(record)
            append_progress(progress_path, {k: v for k, v in record.items() if k != 'curve'})

    coarse_winner, coarse_index, coarse_mean = select_config(coarse, history, 'coarse')
    refined = refine_configs(coarse_winner)
    refine_seeds = [args.seed + 2, args.seed + 3]
    if smoke:
        refined = refined[3:4]
        refine_seeds = refine_seeds[:1]
    for config_index, config in enumerate(refined):
        for probe_seed in refine_seeds:
            started = time.time()
            primary, _, curve = train_once(data, pairs, config, probe_seed, capped(5),
                                            evaluate_fn, device, False)
            record = {'stage': 'refine', 'config_index': config_index,
                      'seed': probe_seed, 'config': config, 'primary': primary,
                      'epochs': capped(5), 'seconds': round(time.time() - started, 3),
                      'curve': curve}
            history.append(record)
            append_progress(progress_path, {k: v for k, v in record.items() if k != 'curve'})

    final_config, refine_index, refine_mean = select_config(refined, history, 'refine')
    final_seeds = [args.seed + i for i in range(4, 9)]
    if smoke:
        final_seeds = final_seeds[:1]
    final_scores = []
    final_candidates = []
    for final_seed in final_seeds:
        started = time.time()
        primary, scores, curve = train_once(data, pairs, final_config, final_seed,
                                            capped(args.epochs), evaluate_fn, device, True)
        metrics = metric_dict(evaluate_fn, data['val_user'], data['yv'], scores)
        final_scores.append(scores)
        final_candidates.append((metrics, scores, 'single_seed_' + str(final_seed)))
        record = {'stage': 'final', 'seed': final_seed, 'config': final_config,
                  'primary': primary, 'epochs': capped(args.epochs),
                  'seconds': round(time.time() - started, 3), 'curve': curve}
        history.append(record)
        append_progress(progress_path, {k: v for k, v in record.items() if k != 'curve'})

    if len(final_scores) > 1:
        mean_scores = np.mean(np.stack(final_scores), axis=0)
        mean_metrics = metric_dict(evaluate_fn, data['val_user'], data['yv'], mean_scores)
        final_candidates.append((mean_metrics, mean_scores, 'mean_logits'))
        ranked = [rank_within_user(data['val_user'], scores) for scores in final_scores]
        rank_scores = np.mean(np.stack(ranked), axis=0)
        rank_metrics = metric_dict(evaluate_fn, data['val_user'], data['yv'], rank_scores)
        final_candidates.append((rank_metrics, rank_scores, 'within_user_rank_average'))
        history.append({'stage': 'ensemble_close', 'mean_logits': mean_metrics,
                        'rank_average': rank_metrics})

    chosen_metrics, chosen_scores, chosen_name = max(
        final_candidates, key=lambda item: item[0]['primary'])
    metrics_output = {
        'gauc': chosen_metrics['gauc'],
        'ndcg5': chosen_metrics['ndcg5'],
        'primary': chosen_metrics['primary'],
        'selected_output': chosen_name,
        'selected_config': final_config,
        'search_summary': {
            'coarse_winner_index': coarse_index,
            'coarse_winner_mean_primary': coarse_mean,
            'refine_winner_index': refine_index,
            'refine_winner_mean_primary': refine_mean,
            'pair_count': int(len(pairs[0])),
        },
        'history': history,
    }
    with open(os.path.join(args.out_dir, 'metrics.json'), 'w') as fh:
        json.dump(metrics_output, fh)

    with open(os.path.join(args.out_dir, 'predictions.csv'), 'w') as fh:
        fh.write('row_id,user_id,video_id,score\n')
        for i, score in enumerate(chosen_scores):
            fh.write(f'{i},{data["val_user"][i]},{data["val_video"][i]},{score:.9g}\n')


if __name__ == '__main__':
    main()
