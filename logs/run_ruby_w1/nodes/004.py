import argparse
import csv
import datetime
import gc
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FM(torch.nn.Module):
    def __init__(self, total_dim, k=16, dropout=0.15):
        super().__init__()
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.dropout = torch.nn.Dropout(dropout)
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)

    def forward(self, x):
        e = self.dropout(self.emb(x))
        summed = e.sum(dim=1)
        pair = 0.5 * (summed.square() - e.square().sum(dim=1)).sum(dim=1)
        return self.bias + self.lin(x).sum(dim=(1, 2)) + pair


class DCNLite(torch.nn.Module):
    def __init__(self, total_dim, n_fields, k=16, hidden=128, dropout=0.25,
                 cross_layers=2):
        super().__init__()
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        d = n_fields * k
        self.input_dropout = torch.nn.Dropout(dropout)
        self.cross_w = torch.nn.Parameter(torch.empty(cross_layers, d))
        self.cross_b = torch.nn.Parameter(torch.zeros(cross_layers, d))
        self.cross_out = torch.nn.Linear(d, 1)
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
        torch.nn.init.zeros_(self.cross_out.bias)

    def forward(self, x):
        e = self.emb(x)
        summed = e.sum(dim=1)
        pair = 0.5 * (summed.square() - e.square().sum(dim=1)).sum(dim=1)
        fm = self.bias + self.lin(x).sum(dim=(1, 2)) + pair
        x0 = self.input_dropout(e).flatten(1)
        xl = x0
        for layer in range(self.cross_w.shape[0]):
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
        (mapping.get(str(value), 0) for value in val_values),
        dtype=np.int64,
        count=len(val_values),
    )
    return train_encoded, val_encoded, len(mapping) + 1


def load_csv(path):
    rows = []
    with open(path, newline='') as fh:
        for row in csv.DictReader(fh):
            rows.append({
                'user_id': row['user_id'],
                'video_id': row['video_id'],
                'author_id': row.get('author_id', '0'),
                'tab': row['tab'],
                'hourmin': row.get('hourmin', '0'),
                'date': row.get('date', '0'),
                'duration_ms': float(row['duration_ms']),
                'long_view': float(row['long_view']),
            })
    return rows


def load_data(data_dir):
    train_npz = os.path.join(data_dir, 'train.npz')
    val_npz = os.path.join(data_dir, 'val.npz')
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        trf = np.load(train_npz)
        vaf = np.load(val_npz)
        tr = {key: trf[key] for key in trf.files}
        va = {key: vaf[key] for key in vaf.files}
        trf.close()
        vaf.close()
        dims = tr['field_dims'].astype(np.int64)
        offsets = np.concatenate(([0], np.cumsum(dims)[:-1]))
        output_videos = va['X'][:, 1].astype(np.int64) - int(offsets[1])
        return tr, va, va['user'], output_videos, True

    train_rows = load_csv(os.path.join(data_dir, 'train.csv'))
    val_rows = load_csv(os.path.join(data_dir, 'val.csv'))
    names = ('user_id', 'video_id', 'author_id', 'tab')
    encoded_train = []
    encoded_val = []
    dims = []
    for name in names:
        et, ev, dim = encode_column(
            [row[name] for row in train_rows],
            [row[name] for row in val_rows],
        )
        encoded_train.append(et)
        encoded_val.append(ev)
        dims.append(dim)
    train_duration = np.asarray([row['duration_ms'] for row in train_rows])
    val_duration = np.asarray([row['duration_ms'] for row in val_rows])
    edges = np.unique(np.quantile(train_duration, np.linspace(0.1, 0.9, 9)))
    encoded_train.append(np.searchsorted(edges, train_duration, side='right'))
    encoded_val.append(np.searchsorted(edges, val_duration, side='right'))
    dims.append(len(edges) + 1)
    field_dims = np.asarray(dims, dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1]))
    tr = {
        'X': (np.stack(encoded_train, axis=1) + offsets).astype(np.int32),
        'y': np.asarray([row['long_view'] for row in train_rows], dtype=np.float32),
        'user': np.asarray([row['user_id'] for row in train_rows]),
        'date': np.asarray([row['date'] for row in train_rows]),
        'hourmin': np.asarray([row['hourmin'] for row in train_rows]),
        'field_dims': field_dims,
    }
    va = {
        'X': (np.stack(encoded_val, axis=1) + offsets).astype(np.int32),
        'y': np.asarray([row['long_view'] for row in val_rows], dtype=np.float32),
        'user': np.asarray([row['user_id'] for row in val_rows]),
        'date': np.asarray([row['date'] for row in val_rows]),
        'hourmin': np.asarray([row['hourmin'] for row in val_rows]),
        'field_dims': field_dims,
    }
    return tr, va, va['user'], np.asarray([row['video_id'] for row in val_rows]), False


def get_evaluator(fast_path):
    if fast_path:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    return evaluate


def metric_values(evaluate, users, labels, scores):
    result = evaluate(users, labels.astype(int), scores)
    return {
        'gauc': float(result.get('GAUC', result.get('gauc'))),
        'ndcg5': float(result.get('nDCG@5', result.get('ndcg5'))),
        'primary': float(result['primary']),
    }


def date_ordinals(values):
    values = np.asarray(values).astype(str)
    unique, inverse = np.unique(values, return_inverse=True)
    converted = np.zeros(len(unique), dtype=np.float64)
    valid = []
    for i, text in enumerate(unique):
        digits = ''.join(ch for ch in text if ch.isdigit())
        try:
            if len(digits) >= 8:
                dt = datetime.date(int(digits[:4]), int(digits[4:6]), int(digits[6:8]))
                converted[i] = dt.toordinal()
                valid.append(converted[i])
        except ValueError:
            pass
    if not valid:
        return np.zeros(len(values), dtype=np.float32)
    converted[converted == 0] = min(valid)
    return converted[inverse].astype(np.float32)


def recency_weights(values, half_life):
    days = date_ordinals(values)
    age = float(days.max()) - days
    weights = np.exp2(-age / float(half_life)).astype(np.float32)
    weights /= max(float(weights.mean()), 1e-8)
    return weights


def hour_buckets(values):
    result = np.zeros(len(values), dtype=np.int64)
    for i, value in enumerate(np.asarray(values).astype(str)):
        digits = ''.join(ch for ch in value if ch.isdigit())
        try:
            number = int(digits) if digits else 0
            result[i] = (number // 100) % 24
        except ValueError:
            result[i] = 0
    return result


def weekday_buckets(values):
    ordinals = date_ordinals(values).astype(np.int64)
    if np.all(ordinals == 0):
        return np.zeros(len(ordinals), dtype=np.int64)
    return ordinals % 7


def build_sequence_features(tr, va):
    base_dim = int(np.asarray(tr['field_dims']).sum())
    offsets = np.concatenate(([0], np.cumsum(tr['field_dims'].astype(np.int64))[:-1]))
    unknown_author = int(offsets[2])
    depth = 12
    tr_hist = np.full((len(tr['X']), depth), unknown_author, dtype=np.int64)
    va_hist = np.full((len(va['X']), depth), unknown_author, dtype=np.int64)
    histories = {}

    def consume(users, authors, output):
        for i in range(len(users)):
            key = str(users[i])
            history = histories.get(key)
            if history is None:
                history = []
                histories[key] = history
            take = history[-depth:]
            if take:
                output[i, depth - len(take):] = take
            history.append(int(authors[i]))
            if len(history) > depth:
                del history[0]

    consume(tr['user'], tr['X'][:, 2], tr_hist)
    consume(va['user'], va['X'][:, 2], va_hist)
    tr_hour = hour_buckets(tr.get('hourmin', np.zeros(len(tr['X'])))) + base_dim
    va_hour = hour_buckets(va.get('hourmin', np.zeros(len(va['X'])))) + base_dim
    tr_day = weekday_buckets(tr.get('date', np.zeros(len(tr['X'])))) + base_dim + 24
    va_day = weekday_buckets(va.get('date', np.zeros(len(va['X'])))) + base_dim + 24
    Xtr = np.concatenate((tr['X'].astype(np.int64), tr_hist,
                          tr_hour[:, None], tr_day[:, None]), axis=1)
    Xva = np.concatenate((va['X'].astype(np.int64), va_hist,
                          va_hour[:, None], va_day[:, None]), axis=1)
    return Xtr, Xva, base_dim + 31


def make_pair_structure(users, labels, context=None):
    _, user_code = np.unique(np.asarray(users).astype(str), return_inverse=True)
    user_code = user_code.astype(np.int64)
    if context is None:
        key_code = user_code
    else:
        pairs = np.stack((user_code, np.asarray(context).astype(np.int64)), axis=1)
        _, key_code = np.unique(pairs, axis=0, return_inverse=True)
        key_code = key_code.astype(np.int64)
    positive = np.flatnonzero(np.asarray(labels) > 0.5).astype(np.int64)
    negative = np.flatnonzero(np.asarray(labels) <= 0.5).astype(np.int64)
    neg_keys = key_code[negative]
    order = np.argsort(neg_keys, kind='stable')
    negative = negative[order]
    n_keys = int(key_code.max()) + 1
    counts = np.bincount(neg_keys, minlength=n_keys).astype(np.int64)
    starts = np.zeros(n_keys, dtype=np.int64)
    if n_keys > 1:
        starts[1:] = np.cumsum(counts[:-1])
    positive = positive[counts[key_code[positive]] > 0]
    return positive, negative, key_code, starts, counts


def predict(model, X, device, batch_size=65536):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = X[start:start + batch_size].to(device)
            outputs.append(model(batch).detach().cpu().numpy())
    return np.concatenate(outputs).astype(np.float64)


def build_model(config, total_dim, n_fields):
    if config['architecture'] == 'fm':
        return FM(total_dim, k=16, dropout=float(config['dropout']))
    return DCNLite(
        total_dim,
        n_fields=n_fields,
        k=16,
        hidden=int(config['hidden']),
        dropout=float(config['dropout']),
        cross_layers=int(config['cross_layers']),
    )


def train_once(bundle, config, seed, epochs, device, evaluate, checkpoints):
    seed_everything(seed)
    X_cpu = bundle['Xtr']
    Xv_cpu = bundle['Xva']
    y_cpu = bundle['y']
    weights_cpu = bundle['weights']
    val_users = bundle['val_users']
    val_y = bundle['val_y']
    model = build_model(config, bundle['total_dim'], X_cpu.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config['lr']),
        weight_decay=float(config['weight_decay']))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=int(config['step_size']), gamma=float(config['gamma']))
    bce = torch.nn.BCEWithLogitsLoss(reduction='none')
    X = X_cpu.to(device)
    y = y_cpu.to(device)
    weights = weights_cpu.to(device)
    positive_np, negative_np, key_code, starts_np, counts_np = bundle['pairs']
    positive = torch.from_numpy(positive_np).to(device)
    negative = torch.from_numpy(negative_np).to(device)
    positive_keys = torch.from_numpy(key_code[positive_np]).to(device)
    starts = torch.from_numpy(starts_np).to(device)
    counts = torch.from_numpy(counts_np).to(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 9187)
    batch_size = 8192
    best_primary = -1.0
    best_scores = None
    curve = []
    n = len(y_cpu)

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(n, generator=generator, device=device)
        last_loss = 0.0
        for start in range(0, n, batch_size):
            idx = permutation[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(X[idx])
            point_loss = (bce(logits, y[idx]) * weights[idx]).mean()
            pair_count = min(len(idx), len(positive))
            if pair_count > 0 and float(config['pair_mix']) > 0:
                chosen = torch.randint(0, len(positive), (pair_count,),
                                       generator=generator, device=device)
                pidx = positive[chosen]
                pkeys = positive_keys[chosen]
                offsets = torch.floor(
                    torch.rand(pair_count, generator=generator, device=device) *
                    counts[pkeys].to(torch.float32)).to(torch.long)
                nidx = negative[starts[pkeys] + offsets]
                margin = model(X[pidx]) - model(X[nidx])
                pair_weight = 0.5 * (weights[pidx] + weights[nidx])
                pair_loss = (torch.nn.functional.softplus(-margin) * pair_weight).mean()
                mix = float(config['pair_mix'])
                loss = (1.0 - mix) * point_loss + mix * pair_loss
            else:
                loss = point_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            last_loss = float(loss.detach().cpu())
        scheduler.step()
        if checkpoints or epoch == epochs - 1:
            scores = predict(model, Xv_cpu, device)
            metrics = metric_values(evaluate, val_users, val_y, scores)
            curve.append({
                'epoch': epoch + 1,
                'train_loss': last_loss,
                'lr': float(optimizer.param_groups[0]['lr']),
                'gauc': metrics['gauc'],
                'ndcg5': metrics['ndcg5'],
                'primary': metrics['primary'],
            })
            if metrics['primary'] > best_primary:
                best_primary = metrics['primary']
                best_scores = scores.copy()
    del model, optimizer, scheduler, X, y, weights
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_primary, best_scores, curve


def family_configs(family, rng, count):
    schedules = [(1, 0.45), (1, 0.60), (1, 0.75), (2, 0.50),
                 (2, 0.68), (2, 0.82), (3, 0.58), (3, 0.74)]
    configs = []
    for i in range(count):
        step, gamma = schedules[i % len(schedules)]
        config = {
            'family': family,
            'architecture': 'fm' if family == 'recency_fm' else 'dcn',
            'dropout': float(rng.uniform(0.16, 0.38)),
            'weight_decay': float(10 ** rng.uniform(-4.6, -2.6)),
            'lr': float(10 ** rng.uniform(-3.2, -2.85)),
            'step_size': int(step),
            'gamma': float(gamma),
            'pair_mix': 0.5,
            'hidden': int(rng.choice([96, 128, 160])),
            'cross_layers': 1 if family == 'sequence_deepfm' else 2,
            'half_life': None,
        }
        if family == 'recency_fm':
            config['half_life'] = float([3.5, 5.0, 7.0, 10.0, 14.0][i % 5])
            config['dropout'] = float(rng.uniform(0.05, 0.25))
        elif family == 'sequence_deepfm':
            config['pair_mix'] = float([0.3, 0.4, 0.5, 0.6][i % 4])
        elif family == 'regularized_dcn':
            config['pair_mix'] = float([0.4, 0.5, 0.6][i % 3])
        elif family == 'temporal_pairs':
            config['pair_mix'] = float([0.4, 0.5, 0.6, 0.7][i % 4])
        configs.append(config)
    return configs


def global_rank(scores):
    order = np.argsort(scores, kind='mergesort')
    result = np.empty(len(scores), dtype=np.float64)
    result[order] = np.arange(len(scores), dtype=np.float64)
    if len(scores) > 1:
        result /= len(scores) - 1
    return result


def per_user_rank(users, scores):
    _, codes = np.unique(np.asarray(users).astype(str), return_inverse=True)
    order = np.argsort(codes, kind='stable')
    sorted_codes = codes[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_codes[1:] != sorted_codes[:-1], True])
    result = np.zeros(len(scores), dtype=np.float64)
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        idx = order[left:right]
        local_order = np.argsort(scores[idx], kind='mergesort')
        ranks = np.empty(len(idx), dtype=np.float64)
        ranks[local_order] = np.arange(len(idx), dtype=np.float64)
        if len(idx) > 1:
            ranks /= len(idx) - 1
        result[idx] = ranks
    return result


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

    base_Xtr = torch.from_numpy(tr['X'].astype(np.int64))
    base_Xva = torch.from_numpy(va['X'].astype(np.int64))
    seq_Xtr_np, seq_Xva_np, seq_total_dim = build_sequence_features(tr, va)
    seq_Xtr = torch.from_numpy(seq_Xtr_np)
    seq_Xva = torch.from_numpy(seq_Xva_np)
    del seq_Xtr_np, seq_Xva_np
    ytr = torch.from_numpy(tr['y'].astype(np.float32))
    val_y = va['y'].astype(np.float32)
    total_dim = int(np.asarray(tr['field_dims']).sum())
    uniform_weights = torch.ones(len(ytr), dtype=torch.float32)
    standard_pairs = make_pair_structure(tr['user'], tr['y'])
    temporal_pairs = make_pair_structure(tr['user'], tr['y'], tr['X'][:, 3])
    recency_cache = {}

    smoke_value = os.environ.get('SMOKE_EPOCHS')
    smoke_cap = int(smoke_value) if smoke_value is not None else None
    probe_epochs = max(1, min(3, smoke_cap)) if smoke_cap is not None else 3
    final_epochs = max(1, min(args.epochs, smoke_cap)) if smoke_cap is not None else args.epochs
    probes_per_family = 1 if smoke_cap is not None else 12
    families = ['regularized_dcn', 'temporal_pairs', 'sequence_deepfm', 'recency_fm']
    rng = np.random.default_rng(args.seed + 17041)
    history = []
    winners = {}

    def bundle_for(config):
        family = config['family']
        if config['half_life'] is None:
            weights = uniform_weights
        else:
            half_life = float(config['half_life'])
            if half_life not in recency_cache:
                if 'date' in tr:
                    values = recency_weights(tr['date'], half_life)
                else:
                    values = np.ones(len(ytr), dtype=np.float32)
                recency_cache[half_life] = torch.from_numpy(values)
            weights = recency_cache[half_life]
        if family == 'sequence_deepfm':
            Xtr, Xva, dim = seq_Xtr, seq_Xva, seq_total_dim
        else:
            Xtr, Xva, dim = base_Xtr, base_Xva, total_dim
        pairs = temporal_pairs if family == 'temporal_pairs' else standard_pairs
        return {
            'Xtr': Xtr,
            'Xva': Xva,
            'y': ytr,
            'weights': weights,
            'val_users': va['user'],
            'val_y': val_y,
            'pairs': pairs,
            'total_dim': dim,
        }

    probe_number = 0
    for family in families:
        best_primary = -1.0
        best_config = None
        for local_index, config in enumerate(family_configs(family, rng, probes_per_family)):
            probe_number += 1
            probe_seed = args.seed + 1000 + probe_number
            primary, _, curve = train_once(
                bundle_for(config), config, probe_seed, probe_epochs,
                device, evaluate, False)
            record = {
                'stage': 'probe',
                'family': family,
                'probe': local_index + 1,
                'seed': probe_seed,
                'epochs': probe_epochs,
                'config': config,
                'primary': float(primary),
                'curve': curve,
            }
            history.append(record)
            append_progress(progress_path, record)
            if primary > best_primary:
                best_primary = primary
                best_config = dict(config)
        winners[family] = {'config': best_config, 'probe_primary': best_primary}

    member_scores = []
    member_records = []
    for family_index, family in enumerate(families):
        config = winners[family]['config']
        member_seed = args.seed + 50000 + family_index * 997
        primary, scores, curve = train_once(
            bundle_for(config), config, member_seed, final_epochs,
            device, evaluate, True)
        metrics = metric_values(evaluate, va['user'], val_y, scores)
        record = {
            'stage': 'final_member',
            'family': family,
            'seed': member_seed,
            'epochs': final_epochs,
            'config': config,
            'selected_probe_primary': float(winners[family]['probe_primary']),
            'gauc': metrics['gauc'],
            'ndcg5': metrics['ndcg5'],
            'primary': metrics['primary'],
            'admitted': bool(metrics['primary'] >= 0.6040),
            'checkpoints': curve,
        }
        history.append(record)
        member_records.append(record)
        member_scores.append(scores)
        append_progress(progress_path, record)

    for i in range(len(member_scores)):
        for j in range(i):
            assert not np.allclose(member_scores[i], member_scores[j])

    admitted = [i for i, record in enumerate(member_records) if record['admitted']]
    if len(admitted) < 2:
        admitted = sorted(
            range(len(member_records)),
            key=lambda i: member_records[i]['primary'], reverse=True)[:3]

    per_user_members = [per_user_rank(va['user'], member_scores[i]) for i in admitted]
    global_members = [global_rank(member_scores[i]) for i in admitted]
    per_user_blend = np.mean(np.stack(per_user_members), axis=0)
    global_blend = np.mean(np.stack(global_members), axis=0)
    per_user_metrics = metric_values(evaluate, va['user'], val_y, per_user_blend)
    global_metrics = metric_values(evaluate, va['user'], val_y, global_blend)
    if per_user_metrics['primary'] >= global_metrics['primary']:
        best_scores = per_user_blend
        final_metrics = per_user_metrics
        blend_kind = 'per_user_rank_average'
    else:
        best_scores = global_blend
        final_metrics = global_metrics
        blend_kind = 'global_rank_average'

    assert all(not np.allclose(best_scores, member_scores[i]) for i in admitted)
    blend_record = {
        'stage': 'blend',
        'admitted_indices': admitted,
        'admitted_families': [families[i] for i in admitted],
        'per_user_rank_metrics': per_user_metrics,
        'global_rank_metrics': global_metrics,
        'selected': blend_kind,
    }
    history.append(blend_record)
    append_progress(progress_path, blend_record)

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
