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
    def __init__(self, total_dim, fields=5, k=16, hidden=128, dropout=0.25, cross_layers=2):
        super().__init__()
        self.fields = fields
        self.k = k
        width = fields * k
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.cross_w = torch.nn.ParameterList([
            torch.nn.Parameter(torch.empty(width)) for _ in range(cross_layers)
        ])
        self.cross_b = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros(width)) for _ in range(cross_layers)
        ])
        self.cross_out = torch.nn.Linear(width, 1, bias=False)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(width, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, hidden // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden // 2, 1),
        )
        self.input_dropout = torch.nn.Dropout(dropout)
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)
        for w in self.cross_w:
            torch.nn.init.normal_(w, std=0.01)
        torch.nn.init.zeros_(self.cross_out.weight)
        for layer in self.mlp:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight, gain=0.5)
                torch.nn.init.zeros_(layer.bias)

    def forward(self, x):
        e = self.emb(x)
        summed = e.sum(1)
        fm = 0.5 * (summed.square() - e.square().sum(1)).sum(1)
        linear = self.lin(x).sum((1, 2))
        x0 = self.input_dropout(e.reshape(e.shape[0], -1))
        cross = x0
        for w, b in zip(self.cross_w, self.cross_b):
            scalar = (cross * w).sum(1, keepdim=True)
            cross = cross + x0 * scalar + b
        deep = self.mlp(x0).squeeze(1)
        return self.bias + linear + fm + self.cross_out(cross).squeeze(1) + deep


def parse_day(values):
    out = np.zeros(len(values), dtype=np.float32)
    for i, value in enumerate(values):
        text = str(value)
        if text.endswith('.0'):
            text = text[:-2]
        digits = ''.join(ch for ch in text if ch.isdigit())
        try:
            if len(digits) >= 8:
                d = datetime.date(int(digits[:4]), int(digits[4:6]), int(digits[6:8]))
                out[i] = float(d.toordinal())
            else:
                out[i] = float(value)
        except Exception:
            out[i] = 0.0
    return out


def load_npz(data_dir):
    tr = np.load(os.path.join(data_dir, 'train.npz'))
    va = np.load(os.path.join(data_dir, 'val.npz'))
    train = {
        'X': tr['X'].astype(np.int64),
        'y': tr['y'].astype(np.float32),
        'user': np.asarray(tr['user']),
        'date': np.asarray(tr['date']),
        'field_dims': tr['field_dims'].astype(np.int64),
    }
    val = {
        'X': va['X'].astype(np.int64),
        'y': va['y'].astype(np.int64),
        'user': np.asarray(va['user']),
    }
    first_offset = int(train['field_dims'][0])
    val['video'] = val['X'][:, 1] - first_offset
    return train, val, True


def load_csv_data(data_dir):
    train_path = os.path.join(data_dir, 'train.csv')
    val_path = os.path.join(data_dir, 'val.csv')
    train_rows = []
    durations = []
    with open(train_path, newline='') as fh:
        for row in csv.DictReader(fh):
            train_rows.append({
                'user_id': row['user_id'],
                'video_id': row['video_id'],
                'tab': row['tab'],
                'duration_ms': float(row['duration_ms']),
                'date': row['date'],
                'long_view': float(row['long_view']),
            })
            durations.append(float(row['duration_ms']))
    edges = np.quantile(np.asarray(durations, dtype=np.float64), np.linspace(0.1, 0.9, 9))
    maps = []
    for key in ('user_id', 'video_id', 'author_id', 'tab', 'dur_bucket'):
        if key == 'author_id':
            vals = ['__unknown_author__']
        elif key == 'dur_bucket':
            vals = [str(i) for i in range(10)]
        else:
            vals = sorted({r[key] for r in train_rows})
        maps.append({v: i for i, v in enumerate(vals)})
    field_dims = np.asarray([len(m) + 1 for m in maps], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1]))

    def encode(row):
        raw = [
            row['user_id'], row['video_id'], '__unknown_author__', row['tab'],
            str(int(np.searchsorted(edges, float(row['duration_ms']), side='right'))),
        ]
        return [offsets[j] + maps[j].get(raw[j], len(maps[j])) for j in range(5)]

    tx = np.asarray([encode(r) for r in train_rows], dtype=np.int64)
    ty = np.asarray([r['long_view'] for r in train_rows], dtype=np.float32)
    tu = np.asarray([r['user_id'] for r in train_rows])
    td = np.asarray([r['date'] for r in train_rows])
    val_rows = []
    with open(val_path, newline='') as fh:
        for row in csv.DictReader(fh):
            val_rows.append({
                'user_id': row['user_id'],
                'video_id': row['video_id'],
                'tab': row['tab'],
                'duration_ms': float(row['duration_ms']),
                'long_view': float(row['long_view']),
            })
    vx = np.asarray([encode(r) for r in val_rows], dtype=np.int64)
    vy = np.asarray([r['long_view'] for r in val_rows], dtype=np.int64)
    vu = np.asarray([r['user_id'] for r in val_rows])
    vv = np.asarray([r['video_id'] for r in val_rows])
    train = {'X': tx, 'y': ty, 'user': tu, 'date': td, 'field_dims': field_dims}
    val = {'X': vx, 'y': vy, 'user': vu, 'video': vv}
    return train, val, False


def make_pairs(users, labels, seed):
    order = np.argsort(users, kind='stable')
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    rng = np.random.default_rng(seed)
    pos_parts = []
    neg_parts = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        idx = order[left:right]
        pos = idx[labels[idx] > 0.5]
        neg = idx[labels[idx] <= 0.5]
        if len(pos) and len(neg):
            pos_parts.append(pos)
            neg_parts.append(neg[rng.integers(0, len(neg), size=len(pos))])
    if not pos_parts:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.concatenate(pos_parts).astype(np.int64), np.concatenate(neg_parts).astype(np.int64)


def metric_values(metric):
    return {
        'gauc': float(metric.get('GAUC', metric.get('gauc'))),
        'ndcg5': float(metric.get('nDCG@5', metric.get('ndcg5'))),
        'primary': float(metric['primary']),
    }


def predict(model, xv, device):
    model.eval()
    parts = []
    with torch.no_grad():
        for left in range(0, len(xv), 65536):
            xb = torch.from_numpy(xv[left:left + 65536]).to(device)
            parts.append(model(xb).detach().cpu().numpy())
    return np.concatenate(parts).astype(np.float64)


def train_once(train, val, config, epochs, seed, device, evaluator, pair_pos, pair_neg,
               eval_half_epochs=False, retain_scores=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    model = DCNLite(
        int(train['field_dims'].sum()), dropout=float(config['dropout']),
        cross_layers=2, hidden=128, k=16,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(config['lr']),
                            weight_decay=float(config['weight_decay']))
    x = train['X']
    y = train['y']
    days = parse_day(train['date'])
    max_day = float(days.max()) if len(days) else 0.0
    half_life = float(config['half_life'])
    recency = np.exp2(-(max_day - days) / max(half_life, 0.1)).astype(np.float32)
    recency /= max(float(recency.mean()), 1e-6)
    n = len(y)
    batch_size = 8192
    rng = np.random.default_rng(seed + 991)
    best_primary = -1.0
    best_scores = None
    curve = []
    update_count = 0
    total_steps = int(math.ceil(n / batch_size))
    for epoch in range(epochs):
        perm = rng.permutation(n)
        model.train()
        running = 0.0
        seen = 0
        for step, left in enumerate(range(0, n, batch_size)):
            idx = perm[left:left + batch_size]
            xb = torch.from_numpy(x[idx]).to(device)
            yb = torch.from_numpy(y[idx]).to(device)
            wb = torch.from_numpy(recency[idx]).to(device)
            logits = model(xb)
            point_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, yb, reduction='none')
            point_loss = (point_loss * wb).sum() / wb.sum().clamp_min(1e-6)
            if len(pair_pos):
                pair_count = max(1, len(idx) // 2)
                chosen = rng.integers(0, len(pair_pos), size=pair_count)
                pi = pair_pos[chosen]
                ni = pair_neg[chosen]
                px = torch.from_numpy(x[pi]).to(device)
                nx = torch.from_numpy(x[ni]).to(device)
                pair_weight = torch.from_numpy(recency[pi]).to(device)
                pair_raw = torch.nn.functional.softplus(-(model(px) - model(nx)))
                pair_loss = (pair_raw * pair_weight).sum() / pair_weight.sum().clamp_min(1e-6)
                loss = 0.5 * point_loss + 0.5 * pair_loss
            else:
                loss = point_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            update_count += 1
            running += float(loss.detach().cpu()) * len(idx)
            seen += len(idx)
            at_half = step + 1 == max(1, total_steps // 2)
            at_end = step + 1 == total_steps
            if eval_half_epochs and (at_half or at_end):
                scores = predict(model, val['X'], device)
                met = metric_values(evaluator(val['user'], val['y'], scores))
                position = epoch + (0.5 if at_half and not at_end else 1.0)
                curve.append({'epoch': position, 'train_loss': running / max(seen, 1),
                              'val_gauc': met['gauc'], 'val_primary': met['primary']})
                if met['primary'] > best_primary:
                    best_primary = met['primary']
                    best_scores = scores.copy()
                model.train()
        if not eval_half_epochs:
            scores = predict(model, val['X'], device)
            met = metric_values(evaluator(val['user'], val['y'], scores))
            curve.append({'epoch': epoch + 1, 'train_loss': running / max(seen, 1),
                          'val_gauc': met['gauc'], 'val_primary': met['primary']})
            if met['primary'] > best_primary:
                best_primary = met['primary']
                best_scores = scores.copy()
        if (epoch + 1) % int(config['step_size']) == 0:
            for group in opt.param_groups:
                group['lr'] *= float(config['gamma'])
    result_scores = best_scores if retain_scores else None
    del model, opt
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_primary, result_scores, curve


def append_progress(path, record):
    with open(path, 'a') as fh:
        fh.write(json.dumps(record, sort_keys=True) + '\n')


def coarse_configs(seed, count):
    rng = np.random.default_rng(seed + 1701)
    half_lives = np.asarray([3.5, 7.0, 14.0])
    gammas = np.asarray([0.35, 0.5, 0.65, 0.8])
    steps = np.asarray([1, 2, 3])
    configs = []
    for _ in range(count):
        configs.append({
            'dropout': float(rng.uniform(0.15, 0.40)),
            'weight_decay': float(10 ** rng.uniform(math.log10(3e-5), math.log10(3e-3))),
            'lr': float(10 ** rng.uniform(math.log10(4e-4), math.log10(1.4e-3))),
            'gamma': float(rng.choice(gammas)),
            'step_size': int(rng.choice(steps)),
            'half_life': float(rng.choice(half_lives)),
        })
    return configs


def refined_configs(winner, count):
    patterns = [
        (-0.06, -0.55, -0.22, 0.82, -1),
        (-0.03, -0.28, 0.00, 0.92, 0),
        (0.00, 0.00, -0.12, 1.00, 0),
        (0.00, 0.00, 0.12, 1.00, 0),
        (0.03, 0.28, 0.00, 1.08, 0),
        (0.06, 0.55, 0.22, 1.18, 1),
        (-0.025, 0.35, -0.16, 0.88, 1),
        (0.025, -0.35, 0.16, 1.12, -1),
        (-0.045, 0.12, 0.08, 0.95, 0),
        (0.045, -0.12, -0.08, 1.05, 0),
    ]
    out = []
    half_options = np.asarray([3.5, 7.0, 14.0])
    half_index = int(np.argmin(np.abs(half_options - float(winner['half_life']))))
    for dd, dwd, dlr, gmul, dh in patterns[:count]:
        out.append({
            'dropout': float(np.clip(float(winner['dropout']) + dd, 0.10, 0.45)),
            'weight_decay': float(np.clip(float(winner['weight_decay']) * (10 ** dwd), 1e-5, 6e-3)),
            'lr': float(np.clip(float(winner['lr']) * (10 ** dlr), 2.5e-4, 1.8e-3)),
            'gamma': float(np.clip(float(winner['gamma']) * gmul, 0.25, 0.88)),
            'step_size': int(winner['step_size']),
            'half_life': float(half_options[int(np.clip(half_index + dh, 0, 2))]),
        })
    return out


def rank_transform(scores, users):
    order = np.argsort(users, kind='stable')
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    ranks = np.empty(len(scores), dtype=np.float64)
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        idx = order[left:right]
        local_order = np.argsort(scores[idx], kind='stable')
        local_rank = np.empty(len(idx), dtype=np.float64)
        local_rank[local_order] = np.arange(len(idx), dtype=np.float64)
        if len(idx) > 1:
            local_rank /= float(len(idx) - 1)
        else:
            local_rank[:] = 0.5
        ranks[idx] = local_rank
    return ranks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=14)
    args = ap.parse_args()
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

    fast = os.path.exists(os.path.join(args.data_dir, 'train.npz')) and os.path.exists(
        os.path.join(args.data_dir, 'val.npz'))
    if fast:
        train, val, _ = load_npz(args.data_dir)
        from data.official.evaluate import evaluate as evaluator
    else:
        train, val, _ = load_csv_data(args.data_dir)
        from harness.evaluate_provisional import evaluate as evaluator

    smoke_raw = os.environ.get('SMOKE_EPOCHS')
    smoke_cap = int(smoke_raw) if smoke_raw is not None else None
    coarse_epochs = 3 if smoke_cap is None else min(3, smoke_cap)
    refine_epochs = 5 if smoke_cap is None else min(5, smoke_cap)
    final_epochs = args.epochs if smoke_cap is None else min(args.epochs, smoke_cap)
    coarse_count = 12 if smoke_cap is None else 2
    refine_count = 8 if smoke_cap is None else 2
    repeats = 2 if smoke_cap is None else 1
    final_seed_count = 5 if smoke_cap is None else 1

    pair_pos, pair_neg = make_pairs(train['user'], train['y'], args.seed + 313)
    history = []
    coarse_results = []
    for config_id, config in enumerate(coarse_configs(args.seed, coarse_count)):
        values = []
        for repeat in range(repeats):
            run_seed = args.seed + 1000 + config_id * 17 + repeat
            score, _, curve = train_once(
                train, val, config, coarse_epochs, run_seed, device, evaluator,
                pair_pos, pair_neg, False, False)
            record = {'stage': 'coarse', 'config_id': config_id, 'repeat': repeat,
                      'seed': run_seed, 'config': config, 'primary': float(score),
                      'curve': curve}
            history.append(record)
            append_progress(progress_path, {'stage': 'coarse', 'config_id': config_id,
                                             'repeat': repeat, 'primary': float(score),
                                             'config': config})
            values.append(score)
        coarse_results.append((float(np.mean(values)), config))
    coarse_results.sort(key=lambda z: z[0], reverse=True)
    coarse_winner = coarse_results[0][1]

    refine_results = []
    for config_id, config in enumerate(refined_configs(coarse_winner, refine_count)):
        values = []
        for repeat in range(repeats):
            run_seed = args.seed + 3000 + config_id * 19 + repeat
            score, _, curve = train_once(
                train, val, config, refine_epochs, run_seed, device, evaluator,
                pair_pos, pair_neg, False, False)
            record = {'stage': 'refine', 'config_id': config_id, 'repeat': repeat,
                      'seed': run_seed, 'config': config, 'primary': float(score),
                      'curve': curve}
            history.append(record)
            append_progress(progress_path, {'stage': 'refine', 'config_id': config_id,
                                             'repeat': repeat, 'primary': float(score),
                                             'config': config})
            values.append(score)
        refine_results.append((float(np.mean(values)), config))
    refine_results.sort(key=lambda z: z[0], reverse=True)
    winning_config = refine_results[0][1]

    rank_sum = np.zeros(len(val['y']), dtype=np.float64)
    final_runs = []
    for offset in range(final_seed_count):
        run_seed = args.seed + offset
        score, scores, curve = train_once(
            train, val, winning_config, final_epochs, run_seed, device, evaluator,
            pair_pos, pair_neg, True, True)
        rank_sum += rank_transform(scores, val['user'])
        final_record = {'stage': 'final', 'seed': run_seed, 'config': winning_config,
                        'best_primary': float(score), 'curve': curve}
        final_runs.append(final_record)
        history.append(final_record)
        append_progress(progress_path, {'stage': 'final', 'seed': run_seed,
                                         'primary': float(score), 'config': winning_config})
    final_scores = rank_sum / float(final_seed_count)
    metrics = metric_values(evaluator(val['user'], val['y'], final_scores))
    output_metrics = {
        'gauc': metrics['gauc'],
        'ndcg5': metrics['ndcg5'],
        'primary': metrics['primary'],
        'winning_config': winning_config,
        'coarse_winner_mean_primary': coarse_results[0][0],
        'refine_winner_mean_primary': refine_results[0][0],
        'ensemble_seeds': [args.seed + i for i in range(final_seed_count)],
        'history': history,
    }
    with open(os.path.join(args.out_dir, 'metrics.json'), 'w') as fh:
        json.dump(output_metrics, fh)
    with open(os.path.join(args.out_dir, 'predictions.csv'), 'w') as fh:
        fh.write('row_id,user_id,video_id,score\n')
        for i, score in enumerate(final_scores):
            fh.write(f'{i},{val["user"][i]},{val["video"][i]},{score:.9g}\n')


if __name__ == '__main__':
    main()
