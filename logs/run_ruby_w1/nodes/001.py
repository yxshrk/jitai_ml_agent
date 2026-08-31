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
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        d = n_fields * k
        self.cross_w = torch.nn.Parameter(torch.empty(2, d))
        self.cross_b = torch.nn.Parameter(torch.zeros(2, d))
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(d, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, hidden // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden // 2, 1),
        )
        self.cross_out = torch.nn.Linear(d, 1)
        self.input_dropout = torch.nn.Dropout(dropout)
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)
        torch.nn.init.normal_(self.cross_w, std=0.01)
        torch.nn.init.zeros_(self.cross_out.bias)

    def forward(self, x):
        e = self.emb(x)
        s = e.sum(dim=1)
        fm_pair = 0.5 * (s.square() - e.square().sum(dim=1)).sum(dim=1)
        fm = self.bias + self.lin(x).sum(dim=(1, 2)) + fm_pair
        x0 = self.input_dropout(e).flatten(1)
        xl = x0
        for layer in range(2):
            scalar = (xl * self.cross_w[layer]).sum(dim=1, keepdim=True)
            xl = xl + x0 * scalar + self.cross_b[layer]
        return fm + self.cross_out(xl).squeeze(1) + self.mlp(x0).squeeze(1)


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode_column(train_values, val_values):
    mapping = {}
    train_encoded = np.empty(len(train_values), dtype=np.int64)
    for i, value in enumerate(train_values):
        key = str(value)
        if key not in mapping:
            mapping[key] = len(mapping) + 1
        train_encoded[i] = mapping[key]
    val_encoded = np.fromiter((mapping.get(str(v), 0) for v in val_values), dtype=np.int64,
                              count=len(val_values))
    return train_encoded, val_encoded, len(mapping) + 1


def load_csv(path, training):
    rows = []
    with open(path, newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            item = {
                'user_id': row['user_id'],
                'video_id': row['video_id'],
                'author_id': row.get('author_id', '0'),
                'tab': row['tab'],
                'duration_ms': float(row['duration_ms']),
                'date': row.get('date', '0'),
                'long_view': float(row['long_view']),
            }
            rows.append(item)
    return rows


def load_data(data_dir):
    train_npz = os.path.join(data_dir, 'train.npz')
    val_npz = os.path.join(data_dir, 'val.npz')
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        trf = np.load(train_npz)
        vaf = np.load(val_npz)
        tr = {k: trf[k] for k in trf.files}
        va = {k: vaf[k] for k in vaf.files}
        trf.close()
        vaf.close()
        field_dims = tr['field_dims'].astype(np.int64)
        offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1]))
        video_out = va['X'][:, 1].astype(np.int64) - int(offsets[1])
        user_out = va['user']
        return tr, va, user_out, video_out, True

    train_rows = load_csv(os.path.join(data_dir, 'train.csv'), True)
    val_rows = load_csv(os.path.join(data_dir, 'val.csv'), False)
    train_cols = [[r[name] for r in train_rows] for name in
                  ('user_id', 'video_id', 'author_id', 'tab')]
    val_cols = [[r[name] for r in val_rows] for name in
                ('user_id', 'video_id', 'author_id', 'tab')]
    encoded_train = []
    encoded_val = []
    dims = []
    for train_col, val_col in zip(train_cols, val_cols):
        et, ev, dim = encode_column(train_col, val_col)
        encoded_train.append(et)
        encoded_val.append(ev)
        dims.append(dim)
    train_duration = np.asarray([r['duration_ms'] for r in train_rows], dtype=np.float64)
    val_duration = np.asarray([r['duration_ms'] for r in val_rows], dtype=np.float64)
    edges = np.unique(np.quantile(train_duration, np.linspace(0.1, 0.9, 9)))
    encoded_train.append(np.searchsorted(edges, train_duration, side='right').astype(np.int64))
    encoded_val.append(np.searchsorted(edges, val_duration, side='right').astype(np.int64))
    dims.append(len(edges) + 1)
    field_dims = np.asarray(dims, dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1]))
    Xtr = np.stack(encoded_train, axis=1) + offsets
    Xva = np.stack(encoded_val, axis=1) + offsets
    tr = {
        'X': Xtr.astype(np.int32),
        'y': np.asarray([r['long_view'] for r in train_rows], dtype=np.float32),
        'user': np.asarray(train_cols[0]),
        'date': np.asarray([r['date'] for r in train_rows]),
        'field_dims': field_dims,
    }
    va = {
        'X': Xva.astype(np.int32),
        'y': np.asarray([r['long_view'] for r in val_rows], dtype=np.float32),
        'user': np.asarray(val_cols[0]),
        'field_dims': field_dims,
    }
    return tr, va, np.asarray(val_cols[0]), np.asarray(val_cols[1]), False


def get_evaluator(fast_path):
    if fast_path:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    return evaluate


def date_ordinals(values):
    arr = np.asarray(values)
    unique, inverse = np.unique(arr.astype(str), return_inverse=True)
    converted = np.zeros(len(unique), dtype=np.float64)
    valid = []
    for i, text in enumerate(unique):
        digits = ''.join(ch for ch in str(text) if ch.isdigit())
        try:
            if len(digits) >= 8:
                dt = datetime.date(int(digits[:4]), int(digits[4:6]), int(digits[6:8]))
                converted[i] = float(dt.toordinal())
                valid.append(converted[i])
        except ValueError:
            converted[i] = 0.0
    if not valid:
        return np.zeros(len(arr), dtype=np.float32)
    floor = min(valid)
    converted[converted == 0.0] = floor
    return converted[inverse].astype(np.float32)


def recency_weights(date_values, half_life):
    days = date_ordinals(date_values)
    age = float(days.max()) - days
    weights = np.exp2(-age / float(half_life)).astype(np.float32)
    weights /= max(float(weights.mean()), 1e-8)
    return weights


def make_pair_structure(users, labels):
    _, user_code = np.unique(np.asarray(users), return_inverse=True)
    user_code = user_code.astype(np.int64)
    n_users = int(user_code.max()) + 1
    positive = np.flatnonzero(np.asarray(labels) > 0.5).astype(np.int64)
    negative = np.flatnonzero(np.asarray(labels) <= 0.5).astype(np.int64)
    neg_users = user_code[negative]
    order = np.argsort(neg_users, kind='stable')
    negative = negative[order]
    counts = np.bincount(neg_users, minlength=n_users).astype(np.int64)
    starts = np.zeros(n_users, dtype=np.int64)
    if n_users > 1:
        starts[1:] = np.cumsum(counts[:-1])
    keep = counts[user_code[positive]] > 0
    positive = positive[keep]
    return positive, negative, user_code, starts, counts


def metric_values(evaluate, users, labels, scores):
    result = evaluate(users, labels.astype(int), scores)
    return {
        'gauc': float(result['GAUC'] if 'GAUC' in result else result['gauc']),
        'ndcg5': float(result.get('nDCG@5', result.get('ndcg5'))),
        'primary': float(result['primary']),
    }


def predict(model, Xv, device, batch_size=65536):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(Xv), batch_size):
            outputs.append(model(Xv[start:start + batch_size].to(device)).detach().cpu().numpy())
    return np.concatenate(outputs).astype(np.float64)


def train_once(data, config, seed, epochs, device, evaluate, final_checkpoints=False):
    seed_everything(seed)
    X_cpu, y_cpu, w_cpu, Xv_cpu, val_users, val_y = data['arrays']
    total_dim = data['total_dim']
    model = DCNLite(total_dim, dropout=float(config['dropout'])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                  weight_decay=float(config['weight_decay']))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=int(config['step_size']), gamma=float(config['gamma']))
    bce = torch.nn.BCEWithLogitsLoss(reduction='none')
    n = len(y_cpu)
    batch_size = 8192
    pos_np, neg_np, user_code, neg_starts, neg_counts = data['pairs']
    pos = torch.from_numpy(pos_np).to(device)
    negative = torch.from_numpy(neg_np).to(device)
    pos_users = torch.from_numpy(user_code[pos_np]).to(device)
    starts = torch.from_numpy(neg_starts).to(device)
    counts = torch.from_numpy(neg_counts).to(device)
    X = X_cpu.to(device)
    y = y_cpu.to(device)
    weights = w_cpu.to(device)
    Xv = Xv_cpu
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 9187)
    best_primary = -1.0
    best_scores = None
    checkpoint_history = []

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=generator, device=device)
        num_batches = int(math.ceil(n / batch_size))
        half_batch = int(math.ceil(num_batches / 2.0))
        half_done = False
        loss_value = 0.0
        for batch_no, start in enumerate(range(0, n, batch_size), 1):
            idx = perm[start:start + batch_size]
            pair_count = min(len(idx), len(pos))
            pair_choice = torch.randint(0, len(pos), (pair_count,), generator=generator,
                                        device=device)
            pidx = pos[pair_choice]
            pu = pos_users[pair_choice]
            nc = counts[pu]
            offsets = torch.floor(torch.rand(pair_count, generator=generator, device=device) *
                                  nc.to(torch.float32)).to(torch.long)
            nidx = negative[starts[pu] + offsets]
            optimizer.zero_grad(set_to_none=True)
            logits = model(X[idx])
            point_loss = (bce(logits, y[idx]) * weights[idx]).mean()
            margin = model(X[pidx]) - model(X[nidx])
            pair_weight = 0.5 * (weights[pidx] + weights[nidx])
            pair_loss = (torch.nn.functional.softplus(-margin) * pair_weight).mean()
            loss = 0.5 * point_loss + 0.5 * pair_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_value = float(loss.detach().cpu())
            if final_checkpoints and batch_no == half_batch:
                scores = predict(model, Xv, device)
                metrics = metric_values(evaluate, val_users, val_y, scores)
                checkpoint_history.append({
                    'epoch': epoch + 0.5,
                    'train_loss': round(loss_value, 6),
                    'lr': float(optimizer.param_groups[0]['lr']),
                    'gauc': metrics['gauc'],
                    'ndcg5': metrics['ndcg5'],
                    'primary': metrics['primary'],
                })
                if metrics['primary'] > best_primary:
                    best_primary = metrics['primary']
                    best_scores = scores.copy()
                model.train()
                half_done = True
        scheduler.step()
        if final_checkpoints:
            scores = predict(model, Xv, device)
            metrics = metric_values(evaluate, val_users, val_y, scores)
            checkpoint_history.append({
                'epoch': float(epoch + 1),
                'train_loss': round(loss_value, 6),
                'lr': float(optimizer.param_groups[0]['lr']),
                'gauc': metrics['gauc'],
                'ndcg5': metrics['ndcg5'],
                'primary': metrics['primary'],
            })
            if metrics['primary'] > best_primary:
                best_primary = metrics['primary']
                best_scores = scores.copy()
        elif epoch == epochs - 1:
            scores = predict(model, Xv, device)
            metrics = metric_values(evaluate, val_users, val_y, scores)
            best_primary = metrics['primary']
            best_scores = scores

    return best_primary, best_scores, checkpoint_history


def coarse_configs(rng, count):
    schedules = [(1, 0.45), (1, 0.60), (1, 0.75), (2, 0.50),
                 (2, 0.68), (2, 0.82), (3, 0.58), (3, 0.74)]
    half_lives = [3.5, 7.0, 14.0]
    configs = []
    for i in range(count):
        step_size, gamma = schedules[i % len(schedules)]
        configs.append({
            'dropout': float(rng.uniform(0.15, 0.40)),
            'weight_decay': float(10.0 ** rng.uniform(math.log10(3e-5), math.log10(3e-3))),
            'step_size': int(step_size),
            'gamma': float(gamma),
            'half_life': float(half_lives[i % len(half_lives)]),
        })
    return configs


def refined_configs(rng, winner, count):
    configs = []
    dropout_offsets = np.linspace(-0.07, 0.07, count)
    wd_multipliers = np.exp(np.linspace(-0.9, 0.9, count))
    half_options = sorted(set([3.5, 5.0, 7.0, 10.0, 14.0, winner['half_life']]))
    for i in range(count):
        gamma = float(np.clip(winner['gamma'] + rng.uniform(-0.13, 0.13), 0.35, 0.90))
        step = int(np.clip(winner['step_size'] + int(rng.choice([-1, 0, 0, 1])), 1, 3))
        configs.append({
            'dropout': float(np.clip(winner['dropout'] + dropout_offsets[i], 0.10, 0.45)),
            'weight_decay': float(np.clip(winner['weight_decay'] * wd_multipliers[i], 1e-5, 6e-3)),
            'step_size': step,
            'gamma': gamma,
            'half_life': float(half_options[i % len(half_options)]),
        })
    configs[0] = dict(winner)
    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=14)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    progress_path = os.path.join(args.out_dir, 'progress.log')
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tr, va, output_users, output_videos, fast_path = load_data(args.data_dir)
    evaluate = get_evaluator(fast_path)

    Xtr = torch.from_numpy(tr['X'].astype(np.int64))
    ytr = torch.from_numpy(tr['y'].astype(np.float32))
    Xva = torch.from_numpy(va['X'].astype(np.int64))
    val_y = va['y'].astype(np.float32)
    total_dim = int(np.asarray(tr['field_dims']).sum())
    pairs = make_pair_structure(tr['user'], tr['y'])
    smoke = os.environ.get('SMOKE_EPOCHS')
    cap = int(smoke) if smoke is not None else None
    coarse_epochs = min(3, cap) if cap is not None else 3
    refine_epochs = min(5, cap) if cap is not None else 5
    final_epochs = min(args.epochs, cap) if cap is not None else args.epochs
    coarse_epochs = max(1, coarse_epochs)
    refine_epochs = max(1, refine_epochs)
    final_epochs = max(1, final_epochs)

    rng = np.random.default_rng(args.seed + 1103)
    history = []
    cached_weights = {}

    def make_data(half_life):
        key = float(half_life)
        if key not in cached_weights:
            if 'date' in tr:
                values = recency_weights(tr['date'], key)
            else:
                values = np.ones(len(ytr), dtype=np.float32)
            cached_weights[key] = torch.from_numpy(values)
        return {
            'arrays': (Xtr, ytr, cached_weights[key], Xva, va['user'], val_y),
            'total_dim': total_dim,
            'pairs': pairs,
        }

    coarse = coarse_configs(rng, 32)
    best_config = None
    best_probe = -1.0
    for i, config in enumerate(coarse):
        probe_seed = args.seed + 1000 + i
        primary, _, _ = train_once(make_data(config['half_life']), config, probe_seed,
                                   coarse_epochs, device, evaluate, False)
        record = {'stage': 'coarse', 'probe': i + 1, 'seed': probe_seed,
                  'epochs': coarse_epochs, 'config': config, 'primary': float(primary)}
        history.append(record)
        with open(progress_path, 'a') as fh:
            fh.write(json.dumps(record, sort_keys=True) + '\n')
        if primary > best_probe:
            best_probe = primary
            best_config = dict(config)

    refined = refined_configs(rng, best_config, 16)
    for i, config in enumerate(refined):
        probe_seed = args.seed + 3000 + i
        primary, _, _ = train_once(make_data(config['half_life']), config, probe_seed,
                                   refine_epochs, device, evaluate, False)
        record = {'stage': 'refine', 'probe': i + 1, 'seed': probe_seed,
                  'epochs': refine_epochs, 'config': config, 'primary': float(primary)}
        history.append(record)
        with open(progress_path, 'a') as fh:
            fh.write(json.dumps(record, sort_keys=True) + '\n')
        if primary > best_probe:
            best_probe = primary
            best_config = dict(config)

    final_primary, best_scores, checkpoints = train_once(
        make_data(best_config['half_life']), best_config, args.seed, final_epochs,
        device, evaluate, True)
    final_metrics = metric_values(evaluate, va['user'], val_y, best_scores)
    history.append({
        'stage': 'final',
        'seed': args.seed,
        'epochs': final_epochs,
        'config': best_config,
        'selected_probe_primary': float(best_probe),
        'best_checkpoint_primary': float(final_primary),
        'checkpoints': checkpoints,
    })

    metrics = {
        'gauc': final_metrics['gauc'],
        'ndcg5': final_metrics['ndcg5'],
        'primary': final_metrics['primary'],
        'history': history,
    }
    with open(os.path.join(args.out_dir, 'metrics.json'), 'w') as fh:
        json.dump(metrics, fh)
    with open(os.path.join(args.out_dir, 'predictions.csv'), 'w') as fh:
        fh.write('row_id,user_id,video_id,score\n')
        for i, score in enumerate(best_scores):
            fh.write(f'{i},{output_users[i]},{output_videos[i]},{score:.9g}\n')


if __name__ == '__main__':
    main()
