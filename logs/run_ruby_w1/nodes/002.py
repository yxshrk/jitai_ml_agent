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
        self.centered_bias = torch.nn.Parameter(torch.zeros(1))
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
    val_encoded = np.fromiter(
        (mapping.get(str(v), 0) for v in val_values),
        dtype=np.int64,
        count=len(val_values),
    )
    return train_encoded, val_encoded, len(mapping) + 1


def load_csv(path):
    rows = []
    with open(path, newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append({
                'user_id': row['user_id'],
                'video_id': row['video_id'],
                'author_id': row.get('author_id', '0'),
                'tab': row['tab'],
                'duration_ms': float(row['duration_ms']),
                'date': row.get('date', '0'),
                'long_view': float(row['long_view']),
            })
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
        return tr, va, va['user'], video_out, True

    train_rows = load_csv(os.path.join(data_dir, 'train.csv'))
    val_rows = load_csv(os.path.join(data_dir, 'val.csv'))
    names = ('user_id', 'video_id', 'author_id', 'tab')
    train_cols = [[r[name] for r in train_rows] for name in names]
    val_cols = [[r[name] for r in val_rows] for name in names]
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


def make_user_groups(user_code):
    order = np.argsort(user_code, kind='stable')
    sorted_users = user_code[order]
    cuts = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
    return [part.astype(np.int64, copy=False) for part in np.split(order, cuts)]


def complete_slate_batches(groups, order, batch_size):
    batches = []
    current = []
    current_size = 0
    for user_position in order:
        group = groups[int(user_position)]
        group_size = len(group)
        if current and current_size + group_size > batch_size:
            batches.append(np.concatenate(current))
            current = []
            current_size = 0
        current.append(group)
        current_size += group_size
        if current_size >= batch_size:
            batches.append(np.concatenate(current))
            current = []
            current_size = 0
    if current:
        batches.append(np.concatenate(current))
    return batches


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
            batch = Xv[start:start + batch_size].to(device)
            outputs.append(model(batch).detach().cpu().numpy())
    return np.concatenate(outputs).astype(np.float64)


def centered_logits(logits, user_codes, global_bias):
    _, inverse = torch.unique(user_codes, sorted=False, return_inverse=True)
    group_count = int(inverse.max().item()) + 1
    sums = torch.zeros(group_count, dtype=logits.dtype, device=logits.device)
    counts = torch.zeros(group_count, dtype=logits.dtype, device=logits.device)
    sums.scatter_add_(0, inverse, logits)
    counts.scatter_add_(0, inverse, torch.ones_like(logits))
    means = sums / counts.clamp_min(1.0)
    return logits - means[inverse] + global_bias


def train_once(data, config, seed, epochs, device, evaluate, centered, checkpoints=False):
    seed_everything(seed)
    X_cpu, y_cpu, w_cpu, Xv_cpu, val_users, val_y = data['arrays']
    model = DCNLite(data['total_dim'], dropout=float(config['dropout'])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=float(config['weight_decay'])
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=int(config['step_size']), gamma=float(config['gamma'])
    )
    bce = torch.nn.BCEWithLogitsLoss(reduction='none')
    pos_np, neg_np, user_code_np, neg_starts, neg_counts = data['pairs']
    groups = data['groups']
    pos = torch.from_numpy(pos_np).to(device)
    negative = torch.from_numpy(neg_np).to(device)
    user_code = torch.from_numpy(user_code_np).to(device)
    pos_users = torch.from_numpy(user_code_np[pos_np]).to(device)
    starts = torch.from_numpy(neg_starts).to(device)
    counts = torch.from_numpy(neg_counts).to(device)
    X = X_cpu.to(device)
    y = y_cpu.to(device)
    weights = w_cpu.to(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 9187)
    batch_size = 8192
    best_primary = -1.0
    best_scores = None
    checkpoint_history = []

    for epoch in range(epochs):
        model.train()
        user_order = torch.randperm(len(groups), generator=generator, device=device).cpu().numpy()
        batches = complete_slate_batches(groups, user_order, batch_size)
        half_batch = int(math.ceil(len(batches) / 2.0))
        loss_value = 0.0
        for batch_no, batch_np in enumerate(batches, 1):
            idx = torch.from_numpy(batch_np).to(device)
            pair_count = min(len(batch_np), len(pos_np))
            pair_choice = torch.randint(
                0, len(pos_np), (pair_count,), generator=generator, device=device
            )
            pidx = pos[pair_choice]
            pu = pos_users[pair_choice]
            nc = counts[pu]
            offsets = torch.floor(
                torch.rand(pair_count, generator=generator, device=device) * nc.to(torch.float32)
            ).to(torch.long)
            nidx = negative[starts[pu] + offsets]

            optimizer.zero_grad(set_to_none=True)
            raw_logits = model(X[idx])
            if centered:
                point_logits = centered_logits(raw_logits, user_code[idx], model.centered_bias)
            else:
                point_logits = raw_logits
            point_loss = (bce(point_logits, y[idx]) * weights[idx]).mean()
            margin = model(X[pidx]) - model(X[nidx])
            pair_weight = 0.5 * (weights[pidx] + weights[nidx])
            pair_loss = (torch.nn.functional.softplus(-margin) * pair_weight).mean()
            loss = 0.5 * point_loss + 0.5 * pair_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_value = float(loss.detach().cpu())

            if checkpoints and batch_no == half_batch:
                scores = predict(model, Xv_cpu, device)
                metrics = metric_values(evaluate, val_users, val_y, scores)
                checkpoint_history.append({
                    'epoch': float(epoch) + 0.5,
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

        scheduler.step()
        if checkpoints:
            scores = predict(model, Xv_cpu, device)
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
            scores = predict(model, Xv_cpu, device)
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
            'weight_decay': float(np.clip(
                winner['weight_decay'] * wd_multipliers[i], 1e-5, 6e-3
            )),
            'step_size': step,
            'gamma': gamma,
            'half_life': float(half_options[i % len(half_options)]),
        })
    configs[0] = dict(winner)
    return configs


def append_progress(path, record):
    with open(path, 'a') as fh:
        fh.write(json.dumps(record, sort_keys=True) + '\n')


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
    groups = make_user_groups(pairs[2])

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
            'groups': groups,
        }

    best_config = None
    best_probe = -1.0
    for i, config in enumerate(coarse_configs(rng, 32)):
        probe_seed = args.seed + 1000 + i
        primary, _, _ = train_once(
            make_data(config['half_life']), config, probe_seed, coarse_epochs,
            device, evaluate, True, False
        )
        record = {
            'stage': 'coarse_centered',
            'probe': i + 1,
            'seed': probe_seed,
            'epochs': coarse_epochs,
            'config': config,
            'primary': float(primary),
        }
        history.append(record)
        append_progress(progress_path, record)
        if primary > best_probe:
            best_probe = primary
            best_config = dict(config)

    for i, config in enumerate(refined_configs(rng, best_config, 16)):
        probe_seed = args.seed + 3000 + i
        primary, _, _ = train_once(
            make_data(config['half_life']), config, probe_seed, refine_epochs,
            device, evaluate, True, False
        )
        record = {
            'stage': 'refine_centered',
            'probe': i + 1,
            'seed': probe_seed,
            'epochs': refine_epochs,
            'config': config,
            'primary': float(primary),
        }
        history.append(record)
        append_progress(progress_path, record)
        if primary > best_probe:
            best_probe = primary
            best_config = dict(config)

    paired = []
    paired_deltas = []
    output_scores = None
    output_checkpoints = None
    for i in range(5):
        paired_seed = args.seed + i
        baseline_primary, baseline_scores, baseline_curve = train_once(
            make_data(best_config['half_life']), best_config, paired_seed,
            final_epochs, device, evaluate, False, True
        )
        centered_primary, centered_scores, centered_curve = train_once(
            make_data(best_config['half_life']), best_config, paired_seed,
            final_epochs, device, evaluate, True, True
        )
        baseline_metrics = metric_values(evaluate, va['user'], val_y, baseline_scores)
        centered_metrics = metric_values(evaluate, va['user'], val_y, centered_scores)
        delta = centered_metrics['primary'] - baseline_metrics['primary']
        paired_deltas.append(float(delta))
        record = {
            'stage': 'paired_confirmation',
            'pair': i + 1,
            'seed': paired_seed,
            'epochs': final_epochs,
            'config': best_config,
            'baseline': baseline_metrics,
            'centered': centered_metrics,
            'primary_delta': float(delta),
            'baseline_checkpoints': baseline_curve,
            'centered_checkpoints': centered_curve,
            'baseline_best_primary': float(baseline_primary),
            'centered_best_primary': float(centered_primary),
        }
        paired.append(record)
        history.append(record)
        append_progress(progress_path, {
            'stage': 'paired_confirmation',
            'pair': i + 1,
            'seed': paired_seed,
            'baseline_primary': baseline_metrics['primary'],
            'centered_primary': centered_metrics['primary'],
            'primary_delta': float(delta),
        })
        if i == 0:
            output_scores = centered_scores.copy()
            output_checkpoints = centered_curve

    deltas = np.asarray(paired_deltas, dtype=np.float64)
    mean_delta = float(deltas.mean())
    std_delta = float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0
    se_delta = float(std_delta / math.sqrt(len(deltas))) if len(deltas) > 1 else 0.0
    ci_low = float(mean_delta - 1.96 * se_delta)
    ci_high = float(mean_delta + 1.96 * se_delta)
    positive_pairs = int(np.sum(deltas > 0.0))
    stability = {
        'paired_seeds': 5,
        'deltas': paired_deltas,
        'mean_delta': mean_delta,
        'sample_std': std_delta,
        'standard_error': se_delta,
        'normal_95ci': [ci_low, ci_high],
        'positive_pairs': positive_pairs,
        'stable_positive': bool(ci_low > 0.0 and positive_pairs >= 4),
        'clears_validation_noise': bool(mean_delta >= 0.002),
    }
    history.append({
        'stage': 'paired_summary',
        'selected_probe_primary': float(best_probe),
        'config': best_config,
        'stability': stability,
        'output_seed': args.seed,
        'output_centered_checkpoints': output_checkpoints,
    })

    final_metrics = metric_values(evaluate, va['user'], val_y, output_scores)
    metrics = {
        'gauc': final_metrics['gauc'],
        'ndcg5': final_metrics['ndcg5'],
        'primary': final_metrics['primary'],
        'paired_confirmation': stability,
        'history': history,
    }
    with open(os.path.join(args.out_dir, 'metrics.json'), 'w') as fh:
        json.dump(metrics, fh)
    with open(os.path.join(args.out_dir, 'predictions.csv'), 'w') as fh:
        fh.write('row_id,user_id,video_id,score\n')
        for i, score in enumerate(output_scores):
            fh.write(f'{i},{output_users[i]},{output_videos[i]},{score:.9g}\n')


if __name__ == '__main__':
    main()
