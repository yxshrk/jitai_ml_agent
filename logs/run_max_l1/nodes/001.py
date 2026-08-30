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
    def __init__(self, total_dim, fields=5, k=16, hidden=128, dropout=0.2, cross_layers=2):
        super().__init__()
        self.fields = fields
        self.k = k
        width = fields * k
        self.emb = torch.nn.Embedding(total_dim, k)
        self.linear = torch.nn.Embedding(total_dim, 1)
        self.emb_dropout = torch.nn.Dropout(dropout)
        self.cross_w = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.empty(width)) for _ in range(cross_layers)]
        )
        self.cross_b = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.zeros(width)) for _ in range(cross_layers)]
        )
        self.cross_out = torch.nn.Linear(width, 1, bias=False)
        self.deep = torch.nn.Sequential(
            torch.nn.Linear(width, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, hidden // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden // 2, 1),
        )
        self.bias = torch.nn.Parameter(torch.zeros(1))
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.linear.weight)
        for w in self.cross_w:
            torch.nn.init.normal_(w, std=0.01)
        torch.nn.init.xavier_uniform_(self.cross_out.weight)
        for module in self.deep:
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        e = self.emb_dropout(self.emb(x))
        x0 = e.reshape(e.shape[0], -1)
        xl = x0
        for w, b in zip(self.cross_w, self.cross_b):
            scalar = torch.sum(xl * w, dim=1, keepdim=True)
            xl = x0 * scalar + b + xl
        linear = self.linear(x).sum(dim=(1, 2))
        return self.bias + linear + self.cross_out(xl).squeeze(1) + self.deep(x0).squeeze(1)


def date_ages(values):
    vals = np.asarray(values)
    parsed = []
    for value in vals:
        text = str(value.decode() if isinstance(value, bytes) else value)
        text = text.split('.')[0]
        try:
            parsed.append(datetime.datetime.strptime(text, '%Y%m%d').date().toordinal())
        except ValueError:
            try:
                parsed.append(int(float(text)))
            except ValueError:
                parsed.append(0)
    parsed = np.asarray(parsed, dtype=np.float32)
    return parsed.max() - parsed


def load_csv_data(data_dir):
    def rows(path, training):
        result = []
        with open(path, 'r', newline='') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                result.append({
                    'user': row['user_id'],
                    'video': row['video_id'],
                    'tab': row['tab'],
                    'duration': float(row['duration_ms']),
                    'date': row['date'],
                    'y': float(row['long_view']),
                })
        return result

    train_rows = rows(os.path.join(data_dir, 'train.csv'), True)
    val_rows = rows(os.path.join(data_dir, 'val.csv'), False)
    user_values = sorted({r['user'] for r in train_rows})
    video_values = sorted({r['video'] for r in train_rows})
    tab_values = sorted({r['tab'] for r in train_rows})
    user_map = {v: i for i, v in enumerate(user_values)}
    video_map = {v: i for i, v in enumerate(video_values)}
    tab_map = {v: i for i, v in enumerate(tab_values)}
    durations = np.asarray([r['duration'] for r in train_rows], dtype=np.float64)
    quantiles = np.quantile(durations, np.linspace(0.1, 0.9, 9))
    dims = np.asarray([len(user_map) + 1, len(video_map) + 1, 1,
                       len(tab_map) + 1, 10], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(dims)[:-1]))

    def encode(records):
        x = np.empty((len(records), 5), dtype=np.int64)
        users = np.empty(len(records), dtype=object)
        videos = np.empty(len(records), dtype=object)
        dates = np.empty(len(records), dtype=object)
        labels = np.empty(len(records), dtype=np.float32)
        for i, r in enumerate(records):
            users[i] = r['user']
            videos[i] = r['video']
            dates[i] = r['date']
            labels[i] = r['y']
            x[i, 0] = user_map.get(r['user'], len(user_map)) + offsets[0]
            x[i, 1] = video_map.get(r['video'], len(video_map)) + offsets[1]
            x[i, 2] = offsets[2]
            x[i, 3] = tab_map.get(r['tab'], len(tab_map)) + offsets[3]
            x[i, 4] = int(np.searchsorted(quantiles, r['duration'], side='right')) + offsets[4]
        return {'X': x, 'y': labels, 'user': users, 'video': videos,
                'date': dates, 'field_dims': dims}

    return encode(train_rows), encode(val_rows), False


def load_data(data_dir):
    train_npz = os.path.join(data_dir, 'train.npz')
    val_npz = os.path.join(data_dir, 'val.npz')
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        tr_file = np.load(train_npz, allow_pickle=False)
        va_file = np.load(val_npz, allow_pickle=False)
        tr = {key: tr_file[key] for key in tr_file.files}
        va = {key: va_file[key] for key in va_file.files}
        va['video'] = np.zeros(len(va['y']), dtype=np.int64)
        return tr, va, True
    return load_csv_data(data_dir)


def make_pairs(users, labels, seed):
    rng = np.random.RandomState(seed)
    users = np.asarray(users)
    labels = np.asarray(labels) > 0.5
    order = np.argsort(users, kind='mergesort')
    sorted_users = users[order]
    changes = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [len(order)]))
    pos_out = np.empty(len(order), dtype=np.int64)
    neg_out = np.empty(len(order), dtype=np.int64)
    cursor = 0
    for start, end in zip(starts, ends):
        group = order[start:end]
        pos = group[labels[group]]
        neg = group[~labels[group]]
        if len(pos) == 0 or len(neg) == 0:
            continue
        count = end - start
        pos_out[cursor:cursor + count] = pos[rng.randint(0, len(pos), size=count)]
        neg_out[cursor:cursor + count] = neg[rng.randint(0, len(neg), size=count)]
        cursor += count
    if cursor == 0:
        raise RuntimeError('No users with both positive and negative labels for BPR')
    return pos_out[:cursor], neg_out[:cursor]


def metric_values(evaluator, users, labels, scores):
    result = evaluator(users, labels.astype(int), scores)
    return {
        'gauc': float(result.get('GAUC', result.get('gauc'))),
        'ndcg5': float(result.get('nDCG@5', result.get('ndcg5'))),
        'primary': float(result['primary']),
    }


def predict(model, x, device, batch_size=65536):
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = x[start:start + batch_size].to(device)
            chunks.append(model(xb).detach().cpu().numpy())
    return np.concatenate(chunks).astype(np.float64)


def train_once(config, epochs, seed, tr, va, ages, pair_pos, pair_neg,
               evaluator, device, half_checkpoints, retain_scores):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    model = DCNLite(
        int(np.asarray(tr['field_dims']).sum()),
        fields=tr['X'].shape[1],
        k=16,
        hidden=128,
        dropout=float(config['dropout']),
        cross_layers=2,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config['lr']), weight_decay=float(config['weight_decay'])
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=int(config['step_size']), gamma=float(config['gamma'])
    )
    x_train = torch.from_numpy(np.asarray(tr['X'], dtype=np.int64))
    y_train = torch.from_numpy(np.asarray(tr['y'], dtype=np.float32))
    x_val = torch.from_numpy(np.asarray(va['X'], dtype=np.int64))
    recency = np.exp(-math.log(2.0) * ages / float(config['half_life'])).astype(np.float32)
    recency /= max(float(recency.mean()), 1e-8)
    weights = torch.from_numpy(recency)
    pair_pos_t = torch.from_numpy(pair_pos)
    pair_neg_t = torch.from_numpy(pair_neg)
    n = len(y_train)
    batch_size = 16384 if device.type == 'cuda' else 8192
    pair_batch_size = max(1024, batch_size // 2)
    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed + 991)
    best_primary = -1.0
    best_metrics = None
    best_scores = None
    curve = []
    global_step = 0
    phases = 2 if half_checkpoints else 1
    for epoch in range(epochs):
        permutation = torch.randperm(n, generator=generator)
        boundaries = np.linspace(0, n, phases + 1, dtype=np.int64)
        for phase in range(phases):
            model.train()
            loss_sum = 0.0
            batches = 0
            phase_idx = permutation[boundaries[phase]:boundaries[phase + 1]]
            for start in range(0, len(phase_idx), batch_size):
                idx = phase_idx[start:start + batch_size]
                xb = x_train[idx].to(device)
                yb = y_train[idx].to(device)
                wb = weights[idx].to(device)
                pair_ids = torch.randint(
                    0, len(pair_pos_t), (min(pair_batch_size, len(idx)),), generator=generator
                )
                pidx = pair_pos_t[pair_ids]
                nidx = pair_neg_t[pair_ids]
                xp = x_train[pidx].to(device)
                xn = x_train[nidx].to(device)
                pair_w = (0.5 * (weights[pidx] + weights[nidx])).to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                point_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, yb, reduction='none'
                )
                point_loss = torch.sum(point_loss * wb) / torch.clamp(torch.sum(wb), min=1e-8)
                pair_margin = model(xp) - model(xn)
                bpr_each = torch.nn.functional.softplus(-pair_margin)
                bpr_loss = torch.sum(bpr_each * pair_w) / torch.clamp(torch.sum(pair_w), min=1e-8)
                loss = 0.5 * point_loss + 0.5 * bpr_loss
                loss.backward()
                optimizer.step()
                global_step += 1
                loss_sum += float(loss.detach().cpu())
                batches += 1
            scores = predict(model, x_val, device)
            metrics = metric_values(evaluator, np.asarray(va['user']), np.asarray(va['y']), scores)
            checkpoint = epoch + float(phase + 1) / phases
            curve.append({
                'checkpoint': round(checkpoint, 2),
                'train_loss': round(loss_sum / max(batches, 1), 6),
                'lr': float(optimizer.param_groups[0]['lr']),
                'gauc': round(metrics['gauc'], 6),
                'ndcg5': round(metrics['ndcg5'], 6),
                'primary': round(metrics['primary'], 6),
            })
            if metrics['primary'] > best_primary + 1e-8:
                best_primary = metrics['primary']
                best_metrics = metrics
                if retain_scores:
                    best_scores = scores.copy()
        scheduler.step()
    if retain_scores and best_scores is None:
        best_scores = predict(model, x_val, device)
    del model, optimizer, scheduler
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_metrics, best_scores, curve


def broad_configs(rng, count):
    configs = []
    gammas = [0.32, 0.42, 0.53, 0.65, 0.78, 0.88]
    steps = [1, 2, 3, 4]
    half_lives = [3.0, 4.5, 6.5, 9.0, 13.0, 18.0]
    for _ in range(count):
        configs.append({
            'dropout': float(rng.uniform(0.12, 0.43)),
            'weight_decay': float(10 ** rng.uniform(math.log10(2e-5), math.log10(4e-3))),
            'lr': float(10 ** rng.uniform(math.log10(3e-4), math.log10(2.2e-3))),
            'gamma': float(gammas[rng.randint(len(gammas))]),
            'step_size': int(steps[rng.randint(len(steps))]),
            'half_life': float(half_lives[rng.randint(len(half_lives))]),
        })
    return configs


def refined_configs(rng, winner, count):
    configs = [dict(winner)]
    while len(configs) < count:
        configs.append({
            'dropout': float(np.clip(winner['dropout'] + rng.normal(0.0, 0.035), 0.08, 0.48)),
            'weight_decay': float(np.clip(winner['weight_decay'] * math.exp(rng.normal(0.0, 0.42)),
                                          1e-5, 8e-3)),
            'lr': float(np.clip(winner['lr'] * math.exp(rng.normal(0.0, 0.22)), 2e-4, 3e-3)),
            'gamma': float(np.clip(winner['gamma'] + rng.normal(0.0, 0.075), 0.25, 0.95)),
            'step_size': int(np.clip(winner['step_size'] + rng.randint(-1, 2), 1, 5)),
            'half_life': float(np.clip(winner['half_life'] * math.exp(rng.normal(0.0, 0.26)),
                                       2.0, 24.0)),
        })
    return configs


def rank_vector(scores):
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(len(scores), dtype=np.float64)
    if len(scores) > 1:
        ranks /= float(len(scores) - 1)
    return ranks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    progress_path = os.path.join(args.out_dir, 'progress.log')
    with open(progress_path, 'w'):
        pass

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        device = torch.device('cpu')
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    tr, va, fast_path = load_data(args.data_dir)
    if fast_path:
        from data.official.evaluate import evaluate as evaluator
    else:
        from harness.evaluate_provisional import evaluate as evaluator

    ages = date_ages(tr['date'])
    pair_pos, pair_neg = make_pairs(tr['user'], tr['y'], args.seed + 17)
    smoke_value = os.environ.get('SMOKE_EPOCHS')
    smoke = smoke_value is not None
    smoke_cap = int(smoke_value) if smoke else None
    coarse_epochs = min(3, smoke_cap) if smoke else 3
    refine_epochs = min(6, smoke_cap) if smoke else 6
    final_epochs = min(args.epochs, smoke_cap) if smoke else args.epochs
    coarse_epochs = max(1, coarse_epochs)
    refine_epochs = max(1, refine_epochs)
    final_epochs = max(1, final_epochs)

    if smoke:
        coarse_count, refine_count, final_seed_count = 2, 1, 1
    elif device.type == 'cuda':
        coarse_count, refine_count, final_seed_count = 72, 28, 5
    else:
        coarse_count, refine_count, final_seed_count = 48, 16, 5

    rng = np.random.RandomState(args.seed + 101)
    history = []
    coarse = broad_configs(rng, coarse_count)
    coarse_results = []
    for probe_id, config in enumerate(coarse):
        metrics, _, curve = train_once(
            config, coarse_epochs, args.seed + 1000 + probe_id, tr, va, ages,
            pair_pos, pair_neg, evaluator, device, False, False
        )
        record = {
            'stage': 'coarse', 'probe': probe_id, 'epochs': coarse_epochs,
            'config': config, 'gauc': metrics['gauc'], 'ndcg5': metrics['ndcg5'],
            'primary': metrics['primary'], 'curve': curve,
        }
        history.append(record)
        coarse_results.append((metrics['primary'], config))
        with open(progress_path, 'a') as fh:
            fh.write(json.dumps(record, sort_keys=True) + '\n')

    coarse_results.sort(key=lambda item: item[0], reverse=True)
    coarse_winner = dict(coarse_results[0][1])
    refined = refined_configs(rng, coarse_winner, refine_count)
    refined_results = []
    for probe_id, config in enumerate(refined):
        metrics, _, curve = train_once(
            config, refine_epochs, args.seed + 5000 + probe_id, tr, va, ages,
            pair_pos, pair_neg, evaluator, device, False, False
        )
        record = {
            'stage': 'refine', 'probe': probe_id, 'epochs': refine_epochs,
            'config': config, 'gauc': metrics['gauc'], 'ndcg5': metrics['ndcg5'],
            'primary': metrics['primary'], 'curve': curve,
        }
        history.append(record)
        refined_results.append((metrics['primary'], config))
        with open(progress_path, 'a') as fh:
            fh.write(json.dumps(record, sort_keys=True) + '\n')

    refined_results.sort(key=lambda item: item[0], reverse=True)
    winning_config = dict(refined_results[0][1])
    member_scores = []
    member_records = []
    for member in range(final_seed_count):
        member_seed = args.seed + member
        metrics, scores, curve = train_once(
            winning_config, final_epochs, member_seed, tr, va, ages,
            pair_pos, pair_neg, evaluator, device, True, True
        )
        member_scores.append(scores)
        record = {
            'stage': 'final', 'member': member, 'seed': member_seed,
            'epochs': final_epochs, 'config': winning_config,
            'gauc': metrics['gauc'], 'ndcg5': metrics['ndcg5'],
            'primary': metrics['primary'], 'curve': curve,
        }
        member_records.append(record)
        history.append(record)
        with open(progress_path, 'a') as fh:
            fh.write(json.dumps(record, sort_keys=True) + '\n')

    ranked = [rank_vector(scores) for scores in member_scores]
    running = np.zeros(len(va['y']), dtype=np.float64)
    best_ensemble_metrics = None
    best_ensemble_scores = None
    ensemble_history = []
    for count, member_rank in enumerate(ranked, 1):
        running += member_rank
        ensemble_scores = running / count
        metrics = metric_values(evaluator, np.asarray(va['user']), np.asarray(va['y']), ensemble_scores)
        record = {'members': count, 'gauc': metrics['gauc'],
                  'ndcg5': metrics['ndcg5'], 'primary': metrics['primary']}
        ensemble_history.append(record)
        if best_ensemble_metrics is None or metrics['primary'] > best_ensemble_metrics['primary'] + 1e-8:
            best_ensemble_metrics = metrics
            best_ensemble_scores = ensemble_scores.copy()

    metrics_payload = {
        'gauc': best_ensemble_metrics['gauc'],
        'ndcg5': best_ensemble_metrics['ndcg5'],
        'primary': best_ensemble_metrics['primary'],
        'winning_config': winning_config,
        'coarse_probe_count': coarse_count,
        'refine_probe_count': refine_count,
        'final_seed_count': final_seed_count,
        'ensemble_history': ensemble_history,
        'history': history,
    }
    with open(os.path.join(args.out_dir, 'metrics.json'), 'w') as fh:
        json.dump(metrics_payload, fh)

    users = np.asarray(va['user'])
    videos = np.asarray(va['video'])
    with open(os.path.join(args.out_dir, 'predictions.csv'), 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['row_id', 'user_id', 'video_id', 'score'])
        for i, score in enumerate(best_ensemble_scores):
            user_value = users[i].item() if hasattr(users[i], 'item') else users[i]
            video_value = videos[i].item() if hasattr(videos[i], 'item') else videos[i]
            writer.writerow([i, user_value, video_value, format(float(score), '.9g')])


if __name__ == '__main__':
    main()
