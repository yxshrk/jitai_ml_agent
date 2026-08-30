import argparse
import csv
import datetime
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RankModel(torch.nn.Module):
    def __init__(self, total_dim, dropout=0.30, k=16):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)
        d = 5 * k
        self.cross_w = torch.nn.ParameterList([
            torch.nn.Parameter(torch.empty(d)) for _ in range(2)
        ])
        self.cross_b = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros(d)) for _ in range(2)
        ])
        for w in self.cross_w:
            torch.nn.init.normal_(w, std=0.01)
        self.cross_out = torch.nn.Linear(d, 1, bias=False)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(d, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(128, 1),
        )

    def forward(self, x):
        e = self.dropout(self.emb(x))
        summed = e.sum(1)
        pair = 0.5 * (summed.square() - e.square().sum(1)).sum(1)
        fm = self.bias + self.lin(x).sum((1, 2)) + pair
        x0 = e.reshape(e.shape[0], -1)
        cross = x0
        for w, b in zip(self.cross_w, self.cross_b):
            cross = x0 * (cross * w).sum(1, keepdim=True) + b + cross
        return fm + self.cross_out(cross).squeeze(1) + self.mlp(x0).squeeze(1)


def date_number(value):
    s = str(value)
    if s.endswith('.0'):
        s = s[:-2]
    s = s.replace('-', '')
    try:
        return datetime.datetime.strptime(s[:8], '%Y%m%d').date().toordinal()
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return 0


def make_csv_data(data_dir):
    train_rows = []
    val_rows = []
    with open(os.path.join(data_dir, 'train.csv'), newline='') as fh:
        for r in csv.DictReader(fh):
            train_rows.append((r['user_id'], r['video_id'], r['tab'],
                               float(r['duration_ms']), float(r['long_view']), r['date']))
    with open(os.path.join(data_dir, 'val.csv'), newline='') as fh:
        for r in csv.DictReader(fh):
            val_rows.append((r['user_id'], r['video_id'], r['tab'],
                             float(r['duration_ms']), float(r['long_view']), r['date']))

    durations = np.asarray([r[3] for r in train_rows], dtype=np.float64)
    edges = np.unique(np.quantile(durations, np.linspace(0.1, 0.9, 9)))
    maps = []
    for column in (0, 1, 2):
        values = sorted(set(r[column] for r in train_rows))
        maps.append({v: i + 1 for i, v in enumerate(values)})
    dims = [len(maps[0]) + 1, len(maps[1]) + 1, len(maps[1]) + 1,
            len(maps[2]) + 1, 10]
    offsets = np.cumsum([0] + dims[:-1]).astype(np.int64)

    def encode(rows):
        x = np.zeros((len(rows), 5), dtype=np.int64)
        for i, r in enumerate(rows):
            user_code = maps[0].get(r[0], 0)
            video_code = maps[1].get(r[1], 0)
            tab_code = maps[2].get(r[2], 0)
            bucket = min(int(np.searchsorted(edges, r[3], side='right')), 9)
            x[i] = np.asarray([user_code, video_code, video_code, tab_code, bucket]) + offsets
        return x

    tr = {
        'X': encode(train_rows),
        'y': np.asarray([r[4] for r in train_rows], dtype=np.float32),
        'user': np.asarray([r[0] for r in train_rows]),
        'date': np.asarray([r[5] for r in train_rows]),
        'field_dims': np.asarray(dims, dtype=np.int64),
    }
    va = {
        'X': encode(val_rows),
        'y': np.asarray([r[4] for r in val_rows], dtype=np.float32),
        'user': np.asarray([r[0] for r in val_rows]),
        'video_raw': np.asarray([r[1] for r in val_rows]),
    }
    return tr, va, False


def load_data(data_dir):
    train_path = os.path.join(data_dir, 'train.npz')
    val_path = os.path.join(data_dir, 'val.npz')
    if os.path.exists(train_path) and os.path.exists(val_path):
        with np.load(train_path) as z:
            tr = {k: z[k] for k in ('X', 'y', 'user', 'date', 'field_dims')}
        with np.load(val_path) as z:
            va = {k: z[k] for k in ('X', 'y', 'user')}
        video_offset = int(tr['field_dims'][0])
        va['video_raw'] = va['X'][:, 1].astype(np.int64) - video_offset
        return tr, va, True
    return make_csv_data(data_dir)


def build_recency_weights(dates):
    ordinals = np.asarray([date_number(v) for v in dates], dtype=np.float64)
    latest = float(ordinals.max()) if len(ordinals) else 0.0
    ages = np.maximum(0.0, latest - ordinals)
    weights = np.exp2(-ages / 7.0)
    weights /= max(float(weights.mean()), 1e-8)
    return weights.astype(np.float32)


def build_pairs(users, labels, seed):
    user_text = users.astype(str)
    order = np.argsort(user_text, kind='stable')
    sorted_users = user_text[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    rng = np.random.RandomState(seed)
    pos_parts = []
    neg_parts = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        idx = order[left:right]
        pos = idx[labels[idx] > 0.5]
        neg = idx[labels[idx] <= 0.5]
        if len(pos) and len(neg):
            pos_parts.append(pos)
            neg_parts.append(rng.choice(neg, size=len(pos), replace=True))
    if not pos_parts:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    return np.concatenate(pos_parts).astype(np.int64), np.concatenate(neg_parts).astype(np.int64)


def build_rank_metadata(users, labels):
    _, group_codes = np.unique(users.astype(str), return_inverse=True)
    group_codes = group_codes.astype(np.int64)
    group_count = int(group_codes.max()) + 1 if len(group_codes) else 0
    positive_counts = np.bincount(group_codes, weights=(labels > 0.5).astype(np.float64),
                                  minlength=group_count).astype(np.int64)
    discounts = 1.0 / np.log2(np.arange(2, 7, dtype=np.float64))
    prefix = np.r_[0.0, np.cumsum(discounts)]
    idcg = prefix[np.minimum(positive_counts, 5)]
    idcg[idcg <= 0.0] = 1.0
    return group_codes, idcg.astype(np.float32)


def predict(model, x, chunk_size=65536):
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, x.shape[0], chunk_size):
            parts.append(model(x[start:start + chunk_size]).detach().cpu().numpy())
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def lambda_pair_weights(model, x_train, group_codes, idcg, pair_pos_np, pair_neg_np,
                        lambda_alpha, device):
    if len(pair_pos_np) == 0:
        return torch.zeros(0, dtype=torch.float32, device=device)
    if lambda_alpha <= 0.0:
        return torch.ones(len(pair_pos_np), dtype=torch.float32, device=device)
    scores = predict(model, x_train)
    order = np.lexsort((-scores, group_codes))
    sorted_groups = group_codes[order]
    starts = np.flatnonzero(np.r_[True, sorted_groups[1:] != sorted_groups[:-1]])
    group_starts = np.empty(int(group_codes.max()) + 1, dtype=np.int64)
    group_starts[sorted_groups[starts]] = starts
    sorted_ranks = np.arange(len(order), dtype=np.int64) - group_starts[sorted_groups]
    ranks = np.empty(len(order), dtype=np.int64)
    ranks[order] = sorted_ranks
    rp = ranks[pair_pos_np]
    rn = ranks[pair_neg_np]
    dp = np.where(rp < 5, 1.0 / np.log2(rp.astype(np.float64) + 2.0), 0.0)
    dn = np.where(rn < 5, 1.0 / np.log2(rn.astype(np.float64) + 2.0), 0.0)
    raw = np.abs(dp - dn) / idcg[group_codes[pair_pos_np]]
    positive = raw[raw > 0.0]
    scale = float(positive.mean()) if len(positive) else 1.0
    normalized = np.clip(raw / max(scale, 1e-8), 0.0, 20.0)
    mixed = (1.0 - lambda_alpha) + lambda_alpha * normalized
    mean = float(mixed.mean())
    mixed /= max(mean, 1e-8)
    return torch.from_numpy(mixed.astype(np.float32)).to(device)


def metric_values(evaluator, users, labels, scores):
    result = evaluator(users, labels.astype(int), scores)
    return {
        'gauc': float(result.get('GAUC', result.get('gauc'))),
        'ndcg5': float(result.get('nDCG@5', result.get('ndcg5'))),
        'primary': float(result['primary']),
    }


def train_once(lambda_alpha, seed, epochs, data, evaluator, device, keep_scores):
    (x_train, y_train, x_val, users_val, labels_val, recency, pair_pos, pair_neg,
     pair_pos_np, pair_neg_np, group_codes, idcg, total_dim) = data
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    model = RankModel(total_dim, dropout=0.30, k=16).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.65)
    n = int(y_train.shape[0])
    batch_size = 8192
    steps = (n + batch_size - 1) // batch_size
    best_primary = -1.0
    best_metrics = None
    best_scores = None
    learning = []

    for epoch in range(epochs):
        dynamic_weights = lambda_pair_weights(
            model, x_train, group_codes, idcg, pair_pos_np, pair_neg_np,
            lambda_alpha, device
        )
        model.train()
        permutation = torch.randperm(n, device=device)
        epoch_loss = 0.0
        seen = 0
        checkpoints = {max(1, steps // 2), steps}
        for step in range(steps):
            idx = permutation[step * batch_size:min((step + 1) * batch_size, n)]
            logits = model(x_train[idx])
            pointwise = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, y_train[idx], reduction='none'
            )
            bce_loss = (pointwise * recency[idx]).mean()
            if pair_pos.numel() > 0:
                pick = torch.randint(0, pair_pos.numel(), (idx.numel(),), device=device)
                pidx = pair_pos[pick]
                nidx = pair_neg[pick]
                margin = model(x_train[pidx]) - model(x_train[nidx])
                pairwise = torch.nn.functional.softplus(-margin)
                pair_loss = (pairwise * recency[pidx] * dynamic_weights[pick]).mean()
                loss = 0.5 * bce_loss + 0.5 * pair_loss
            else:
                loss = bce_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            count = int(idx.numel())
            epoch_loss += float(loss.detach().item()) * count
            seen += count

            if step + 1 in checkpoints:
                scores = predict(model, x_val)
                metrics = metric_values(evaluator, users_val, labels_val, scores)
                learning.append({
                    'epoch': epoch + 1,
                    'fraction': 0.5 if step + 1 < steps else 1.0,
                    'train_loss': round(epoch_loss / max(seen, 1), 6),
                    'val_gauc': round(metrics['gauc'], 6),
                    'val_primary': round(metrics['primary'], 6),
                })
                if metrics['primary'] > best_primary + 1e-8:
                    best_primary = metrics['primary']
                    best_metrics = metrics
                    if keep_scores:
                        best_scores = scores.copy()
                model.train()
        scheduler.step()
    return best_metrics, best_scores, learning


def append_progress(path, payload):
    with open(path, 'a') as fh:
        fh.write(json.dumps(payload, sort_keys=True) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=14)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        device = torch.device('cpu')

    tr, va, fast_path = load_data(args.data_dir)
    if fast_path:
        from data.official.evaluate import evaluate as evaluator
    else:
        from harness.evaluate_provisional import evaluate as evaluator

    x_train = torch.from_numpy(tr['X'].astype(np.int64)).to(device)
    y_train_np = tr['y'].astype(np.float32)
    y_train = torch.from_numpy(y_train_np).to(device)
    x_val = torch.from_numpy(va['X'].astype(np.int64)).to(device)
    recency = torch.from_numpy(build_recency_weights(tr['date'])).to(device)
    pair_pos_np, pair_neg_np = build_pairs(tr['user'], y_train_np, args.seed + 271)
    pair_pos = torch.from_numpy(pair_pos_np).to(device)
    pair_neg = torch.from_numpy(pair_neg_np).to(device)
    group_codes, idcg = build_rank_metadata(tr['user'], y_train_np)
    total_dim = int(np.asarray(tr['field_dims']).sum())
    data = (x_train, y_train, x_val, va['user'], va['y'], recency,
            pair_pos, pair_neg, pair_pos_np, pair_neg_np, group_codes, idcg,
            total_dim)

    smoke_value = os.environ.get('SMOKE_EPOCHS')
    smoke_cap = int(smoke_value) if smoke_value is not None else None
    if smoke_cap is not None:
        probe_epochs = min(1, smoke_cap)
        refine_epochs = min(1, smoke_cap)
        final_epochs = min(args.epochs, smoke_cap)
        alpha_grid = [0.0, 1.0]
        probe_seeds = [args.seed]
        refine_seeds = [args.seed + 1001]
        finalist_count = 1
    else:
        probe_epochs = 12
        refine_epochs = 14
        final_epochs = args.epochs
        alpha_grid = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
        probe_seeds = [args.seed + 101 * i for i in range(5)]
        refine_seeds = [args.seed + 1001, args.seed + 1102, args.seed + 1203]
        finalist_count = 3

    history = []
    progress_path = os.path.join(args.out_dir, 'progress.log')
    probe_summary = []
    for alpha in alpha_grid:
        values = []
        for run_seed in probe_seeds:
            metrics, _, learning = train_once(
                alpha, run_seed, probe_epochs, data, evaluator, device, False
            )
            record = {
                'stage': 'lambda_probe',
                'lambda_alpha': alpha,
                'seed': run_seed,
                'epochs': probe_epochs,
                'gauc': metrics['gauc'],
                'ndcg5': metrics['ndcg5'],
                'primary': metrics['primary'],
                'learning': learning,
            }
            history.append(record)
            values.append(metrics['primary'])
            append_progress(progress_path, {
                'stage': 'lambda_probe', 'lambda_alpha': alpha,
                'seed': run_seed, 'primary': metrics['primary']
            })
        probe_summary.append({
            'lambda_alpha': alpha,
            'mean_primary': float(np.mean(values)),
            'std_primary': float(np.std(values)),
        })

    probe_summary.sort(key=lambda r: (-r['mean_primary'], r['std_primary']))
    finalists = [r['lambda_alpha'] for r in probe_summary[:finalist_count]]
    refinement_summary = []
    for alpha in finalists:
        values = []
        for run_seed in refine_seeds:
            metrics, _, learning = train_once(
                alpha, run_seed, refine_epochs, data, evaluator, device, False
            )
            record = {
                'stage': 'lambda_refinement',
                'lambda_alpha': alpha,
                'seed': run_seed,
                'epochs': refine_epochs,
                'gauc': metrics['gauc'],
                'ndcg5': metrics['ndcg5'],
                'primary': metrics['primary'],
                'learning': learning,
            }
            history.append(record)
            values.append(metrics['primary'])
            append_progress(progress_path, {
                'stage': 'lambda_refinement', 'lambda_alpha': alpha,
                'seed': run_seed, 'primary': metrics['primary']
            })
        refinement_summary.append({
            'lambda_alpha': alpha,
            'mean_primary': float(np.mean(values)),
            'std_primary': float(np.std(values)),
        })

    refinement_summary.sort(key=lambda r: (-r['mean_primary'], r['std_primary']))
    selected_alpha = float(refinement_summary[0]['lambda_alpha'])
    final_metrics, final_scores, final_learning = train_once(
        selected_alpha, args.seed, final_epochs, data, evaluator, device, True
    )
    history.append({
        'stage': 'final',
        'lambda_alpha': selected_alpha,
        'seed': args.seed,
        'epochs': final_epochs,
        'gauc': final_metrics['gauc'],
        'ndcg5': final_metrics['ndcg5'],
        'primary': final_metrics['primary'],
        'learning': final_learning,
    })

    payload = {
        'gauc': final_metrics['gauc'],
        'ndcg5': final_metrics['ndcg5'],
        'primary': final_metrics['primary'],
        'selected_config': {
            'architecture': 'dcn-lite',
            'loss': '0.5-bce+0.5-bpr',
            'pair_weighting': 'current-rank-absolute-delta-ndcg5',
            'lambda_alpha': selected_alpha,
            'weighting': 'recency-7d',
            'dropout': 0.30,
            'weight_decay': 0.001,
            'lr_gamma': 0.65,
        },
        'probe_summary': probe_summary,
        'refinement_summary': refinement_summary,
        'history': history,
    }
    with open(os.path.join(args.out_dir, 'metrics.json'), 'w') as fh:
        json.dump(payload, fh)

    with open(os.path.join(args.out_dir, 'predictions.csv'), 'w') as fh:
        fh.write('row_id,user_id,video_id,score\n')
        for i, score in enumerate(final_scores):
            fh.write(f'{i},{va["user"][i]},{va["video_raw"][i]},{float(score):.8g}\n')


if __name__ == '__main__':
    main()
