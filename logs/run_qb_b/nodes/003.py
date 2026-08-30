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
    def __init__(self, total_dim, dropout=0.30, k=16, duration_heads=False):
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
        self.mlp_hidden = torch.nn.Sequential(
            torch.nn.Linear(d, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.mlp_out = torch.nn.Linear(128, 1)
        self.ordinal_head = torch.nn.Linear(128, 4)
        self.watch_head = torch.nn.Linear(128, 1)
        self.duration_heads = duration_heads
        if duration_heads:
            self.regime_head = torch.nn.Linear(128, 2)

    def forward(self, x, short_regime=None):
        e = self.dropout(self.emb(x))
        summed = e.sum(1)
        pair = 0.5 * (summed.square() - e.square().sum(1)).sum(1)
        fm = self.bias + self.lin(x).sum((1, 2)) + pair
        x0 = e.reshape(e.shape[0], -1)
        cross = x0
        for w, b in zip(self.cross_w, self.cross_b):
            cross = x0 * (cross * w).sum(1, keepdim=True) + b + cross
        hidden = self.mlp_hidden(x0)
        logits = fm + self.cross_out(cross).squeeze(1) + self.mlp_out(hidden).squeeze(1)
        if self.duration_heads:
            regime_logits = self.regime_head(hidden)
            if short_regime is None:
                short_regime = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            logits = logits + regime_logits.gather(1, short_regime.long().view(-1, 1)).squeeze(1)
        return logits, self.ordinal_head(hidden), self.watch_head(hidden).squeeze(1)


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
            train_rows.append({
                'user': r['user_id'],
                'video': r['video_id'],
                'tab': r['tab'],
                'duration': float(r['duration_ms']),
                'label': float(r['long_view']),
                'play': float(r['play_time_ms']),
                'date': r['date'],
            })
    with open(os.path.join(data_dir, 'val.csv'), newline='') as fh:
        for r in csv.DictReader(fh):
            val_rows.append({
                'user': r['user_id'],
                'video': r['video_id'],
                'tab': r['tab'],
                'duration': float(r['duration_ms']),
                'label': float(r['long_view']),
            })
    durations = np.asarray([r['duration'] for r in train_rows], dtype=np.float64)
    edges = np.unique(np.quantile(durations, np.linspace(0.1, 0.9, 9)))
    user_values = sorted(set(r['user'] for r in train_rows))
    video_values = sorted(set(r['video'] for r in train_rows))
    tab_values = sorted(set(r['tab'] for r in train_rows))
    user_map = {v: i + 1 for i, v in enumerate(user_values)}
    video_map = {v: i + 1 for i, v in enumerate(video_values)}
    tab_map = {v: i + 1 for i, v in enumerate(tab_values)}
    dims = [len(user_map) + 1, len(video_map) + 1, len(video_map) + 1,
            len(tab_map) + 1, 10]
    offsets = np.cumsum([0] + dims[:-1]).astype(np.int64)

    def encode(rows):
        x = np.zeros((len(rows), 5), dtype=np.int64)
        for i, r in enumerate(rows):
            bucket = min(int(np.searchsorted(edges, r['duration'], side='right')), 9)
            values = [user_map.get(r['user'], 0), video_map.get(r['video'], 0),
                      video_map.get(r['video'], 0), tab_map.get(r['tab'], 0), bucket]
            x[i] = np.asarray(values, dtype=np.int64) + offsets
        return x

    tr = {
        'X': encode(train_rows),
        'y': np.asarray([r['label'] for r in train_rows], dtype=np.float32),
        'user': np.asarray([r['user'] for r in train_rows]),
        'date': np.asarray([r['date'] for r in train_rows]),
        'play_time_ms': np.asarray([r['play'] for r in train_rows], dtype=np.float32),
        'duration_ms': np.asarray([r['duration'] for r in train_rows], dtype=np.float32),
        'field_dims': np.asarray(dims, dtype=np.int64),
    }
    va = {
        'X': encode(val_rows),
        'y': np.asarray([r['label'] for r in val_rows], dtype=np.float32),
        'user': np.asarray([r['user'] for r in val_rows]),
        'duration_ms': np.asarray([r['duration'] for r in val_rows], dtype=np.float32),
        'video_raw': np.asarray([r['video'] for r in val_rows]),
    }
    return tr, va, False


def load_data(data_dir):
    train_path = os.path.join(data_dir, 'train.npz')
    val_path = os.path.join(data_dir, 'val.npz')
    if os.path.exists(train_path) and os.path.exists(val_path):
        with np.load(train_path) as z:
            tr = {k: z[k] for k in ('X', 'y', 'user', 'date', 'field_dims',
                                     'play_time_ms', 'duration_ms')}
        with np.load(val_path) as z:
            va = {k: z[k] for k in ('X', 'y', 'user', 'duration_ms')}
        offset = int(np.asarray(tr['field_dims'])[0])
        va['video_raw'] = va['X'][:, 1].astype(np.int64) - offset
        return tr, va, True
    return make_csv_data(data_dir)


def build_recency_weights(dates, half_life):
    ordinals = np.asarray([date_number(v) for v in dates], dtype=np.float64)
    latest = float(ordinals.max()) if len(ordinals) else 0.0
    age = np.maximum(0.0, latest - ordinals)
    weights = np.exp2(-age / float(half_life))
    weights /= max(float(weights.mean()), 1e-8)
    return weights.astype(np.float32)


def build_pairs(users, labels, seed):
    text_users = users.astype(str)
    order = np.argsort(text_users, kind='stable')
    sorted_users = text_users[order]
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


def metric_values(evaluator, users, labels, scores):
    result = evaluator(users, labels.astype(int), scores)
    return {
        'gauc': float(result.get('GAUC', result.get('gauc'))),
        'ndcg5': float(result.get('nDCG@5', result.get('ndcg5'))),
        'primary': float(result['primary']),
    }


def canonical_config(config):
    return json.dumps(config, sort_keys=True, separators=(',', ':'))


def base_config():
    return {
        'name': 'champion',
        'ordinal_weight': 0.0,
        'cwm_weight': 0.0,
        'duration_heads': False,
        'recency_half_life': 7.0,
        'dropout': 0.30,
        'weight_decay': 1e-3,
        'lr_gamma': 0.65,
    }


def merge_addons(first, second):
    merged = base_config()
    for source in (first, second):
        if source['ordinal_weight'] > 0:
            merged['ordinal_weight'] = source['ordinal_weight']
        if source['cwm_weight'] > 0:
            merged['cwm_weight'] = source['cwm_weight']
        if source['duration_heads']:
            merged['duration_heads'] = True
        if source['recency_half_life'] != 7.0:
            merged['recency_half_life'] = source['recency_half_life']
    merged['name'] = first['family'] + '+' + second['family']
    merged['family'] = merged['name']
    return merged


def train_once(config, seed, epochs, data, evaluator, device, keep_scores):
    (xt, yt, xv, users_v, labels_v, duration_t, duration_v, ordinal_targets,
     watch_targets, completed, recency_cache, pair_pos, pair_neg, total_dim) = data
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    model = RankModel(total_dim, dropout=config['dropout'],
                      duration_heads=config['duration_heads']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                  weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['lr_gamma'])
    weights = recency_cache[float(config['recency_half_life'])]
    short_t = (duration_t <= 18000.0).long()
    short_v = (duration_v <= 18000.0).long()
    n = int(yt.shape[0])
    batch_size = 8192
    steps = (n + batch_size - 1) // batch_size
    best_primary = -1.0
    best_metrics = None
    best_scores = None
    learning = []
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(n, device=device)
        epoch_loss = 0.0
        seen = 0
        checkpoints = {max(1, steps // 2), steps}
        for step in range(steps):
            idx = permutation[step * batch_size:min((step + 1) * batch_size, n)]
            logits, ordinal_logits, watch_logits = model(xt[idx], short_t[idx])
            point = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, yt[idx], reduction='none')
            bce_loss = (point * weights[idx]).mean()
            if pair_pos.numel() > 0:
                pick = torch.randint(0, pair_pos.numel(), (idx.numel(),), device=device)
                pidx = pair_pos[pick]
                nidx = pair_neg[pick]
                pos_logits = model(xt[pidx], short_t[pidx])[0]
                neg_logits = model(xt[nidx], short_t[nidx])[0]
                pair_loss = (torch.nn.functional.softplus(-(pos_logits - neg_logits)) *
                             weights[pidx]).mean()
                loss = 0.5 * bce_loss + 0.5 * pair_loss
            else:
                loss = bce_loss
            if config['ordinal_weight'] > 0:
                ordinal_element = torch.nn.functional.binary_cross_entropy_with_logits(
                    ordinal_logits, ordinal_targets[idx], reduction='none').mean(1)
                ordinal_loss = (ordinal_element * weights[idx]).mean()
                loss = loss + float(config['ordinal_weight']) * ordinal_loss
            if config['cwm_weight'] > 0:
                watch_prediction = torch.sigmoid(watch_logits)
                exact_loss = torch.nn.functional.smooth_l1_loss(
                    watch_prediction, watch_targets[idx], reduction='none')
                censored_loss = torch.relu(watch_targets[idx] - watch_prediction).square()
                cwm_element = torch.where(completed[idx], censored_loss, exact_loss)
                cwm_loss = (cwm_element * weights[idx]).mean()
                loss = loss + float(config['cwm_weight']) * cwm_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += float(loss.detach().item()) * int(idx.numel())
            seen += int(idx.numel())
            if step + 1 in checkpoints:
                model.eval()
                chunks = []
                with torch.no_grad():
                    for left in range(0, xv.shape[0], 65536):
                        right = min(left + 65536, xv.shape[0])
                        chunks.append(model(xv[left:right], short_v[left:right])[0].cpu().numpy())
                scores = np.concatenate(chunks)
                metrics = metric_values(evaluator, users_v, labels_v, scores)
                learning.append({
                    'epoch': epoch + 1,
                    'fraction': 0.5 if step + 1 < steps else 1.0,
                    'train_loss': round(epoch_loss / max(seen, 1), 6),
                    'val_gauc': round(metrics['gauc'], 6),
                    'val_primary': round(metrics['primary'], 6),
                })
                if metrics['primary'] > best_primary + 1e-10:
                    best_primary = metrics['primary']
                    best_metrics = metrics
                    if keep_scores:
                        best_scores = scores.copy()
                model.train()
        scheduler.step()
    return best_metrics, best_scores, learning


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=16)
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

    y_np = tr['y'].astype(np.float32)
    duration_np = np.maximum(tr['duration_ms'].astype(np.float32), 1.0)
    play_np = np.maximum(tr['play_time_ms'].astype(np.float32), 0.0)
    ratio_denominator = np.minimum(duration_np, 18000.0)
    watch_ratio = play_np / np.maximum(ratio_denominator, 1.0)
    thresholds = np.asarray([0.25, 0.50, 0.75, 1.00], dtype=np.float32)
    ordinal_np = (watch_ratio[:, None] >= thresholds[None, :]).astype(np.float32)
    scale_denominator = np.log1p(np.maximum(duration_np, 18000.0))
    observed_or_bound = np.minimum(play_np, duration_np)
    watch_target_np = np.log1p(observed_or_bound) / np.maximum(scale_denominator, 1e-6)
    watch_target_np = np.clip(watch_target_np, 0.0, 1.0).astype(np.float32)
    completed_np = play_np >= duration_np

    xt = torch.from_numpy(tr['X'].astype(np.int64)).to(device)
    yt = torch.from_numpy(y_np).to(device)
    xv = torch.from_numpy(va['X'].astype(np.int64)).to(device)
    duration_t = torch.from_numpy(duration_np).to(device)
    duration_v = torch.from_numpy(np.maximum(va['duration_ms'].astype(np.float32), 1.0)).to(device)
    ordinal_targets = torch.from_numpy(ordinal_np).to(device)
    watch_targets = torch.from_numpy(watch_target_np).to(device)
    completed = torch.from_numpy(completed_np.astype(np.bool_)).to(device)
    pos_np, neg_np = build_pairs(tr['user'], y_np, args.seed + 271)
    pair_pos = torch.from_numpy(pos_np).to(device)
    pair_neg = torch.from_numpy(neg_np).to(device)
    total_dim = int(np.asarray(tr['field_dims']).sum())

    half_lives = [3.0, 5.0, 7.0, 10.0, 14.0, 21.0, 28.0]
    recency_cache = {
        value: torch.from_numpy(build_recency_weights(tr['date'], value)).to(device)
        for value in half_lives
    }
    data = (xt, yt, xv, va['user'], va['y'], duration_t, duration_v,
            ordinal_targets, watch_targets, completed, recency_cache,
            pair_pos, pair_neg, total_dim)

    smoke_value = os.environ.get('SMOKE_EPOCHS')
    smoke_cap = int(smoke_value) if smoke_value is not None else None
    probe_epochs = 5
    refine_epochs = 12
    finalist_epochs = args.epochs
    final_epochs = args.epochs
    if smoke_cap is not None:
        probe_epochs = min(probe_epochs, smoke_cap)
        refine_epochs = min(refine_epochs, smoke_cap)
        finalist_epochs = min(finalist_epochs, smoke_cap)
        final_epochs = min(final_epochs, smoke_cap)

    candidates = []
    champion = base_config()
    champion['family'] = 'champion'
    candidates.append(champion)
    ordinal_weights = [0.05, 0.10, 0.20, 0.30, 0.40, 0.60]
    cwm_weights = [0.05, 0.10, 0.20, 0.30, 0.40, 0.60]
    recency_values = [3.0, 5.0, 10.0, 14.0, 21.0, 28.0]
    if smoke_cap is not None:
        ordinal_weights = [0.20]
        cwm_weights = [0.20]
        recency_values = [14.0]
    for weight in ordinal_weights:
        config = base_config()
        config.update({'name': 'ordinal-' + str(weight), 'family': 'ordinal',
                       'ordinal_weight': weight})
        candidates.append(config)
    duration_config = base_config()
    duration_config.update({'name': 'duration-regime-heads', 'family': 'duration',
                            'duration_heads': True})
    candidates.append(duration_config)
    for weight in cwm_weights:
        config = base_config()
        config.update({'name': 'cwm-' + str(weight), 'family': 'cwm',
                       'cwm_weight': weight})
        candidates.append(config)
    for value in recency_values:
        config = base_config()
        config.update({'name': 'recency-' + str(value), 'family': 'recency',
                       'recency_half_life': value})
        candidates.append(config)

    if smoke_cap is not None:
        probe_seeds = [args.seed]
        refine_seeds = [args.seed + 401]
        finalist_seeds = [args.seed + 701]
    elif device.type == 'cuda':
        probe_seeds = [args.seed + 101 * i for i in range(8)]
        refine_seeds = [args.seed + 2001 + 101 * i for i in range(5)]
        finalist_seeds = [args.seed + 4001 + 101 * i for i in range(5)]
    else:
        probe_seeds = [args.seed + 101 * i for i in range(4)]
        refine_seeds = [args.seed + 2001 + 101 * i for i in range(3)]
        finalist_seeds = [args.seed + 4001 + 101 * i for i in range(3)]

    history = []
    progress_path = os.path.join(args.out_dir, 'progress.log')

    def run_group(stage, configs, seeds, epochs):
        summaries = []
        for config_id, config in enumerate(configs):
            values = []
            for run_seed in seeds:
                metrics, _, learning = train_once(config, run_seed, epochs, data,
                                                  evaluator, device, False)
                record = {
                    'stage': stage,
                    'cell': config_id,
                    'seed': run_seed,
                    'epochs': epochs,
                    'config': config,
                    'gauc': metrics['gauc'],
                    'ndcg5': metrics['ndcg5'],
                    'primary': metrics['primary'],
                    'learning': learning,
                }
                history.append(record)
                values.append(metrics['primary'])
                with open(progress_path, 'a') as fh:
                    fh.write(json.dumps({'stage': stage, 'cell': config_id,
                                         'seed': run_seed, 'config': config,
                                         'primary': metrics['primary']}) + '\n')
            summaries.append({
                'mean_primary': float(np.mean(values)),
                'std_primary': float(np.std(values)),
                'config': config,
            })
        summaries.sort(key=lambda item: (-item['mean_primary'], item['std_primary'],
                                         canonical_config(item['config'])))
        return summaries

    single_summary = run_group('single_addon_probe', candidates, probe_seeds, probe_epochs)
    family_best = {}
    for item in single_summary:
        family = item['config']['family']
        if family in ('ordinal', 'duration', 'cwm', 'recency') and family not in family_best:
            family_best[family] = item['config']
    family_names = sorted(family_best)
    pair_configs = []
    for i in range(len(family_names)):
        for j in range(i + 1, len(family_names)):
            pair_configs.append(merge_addons(family_best[family_names[i]],
                                             family_best[family_names[j]]))
    pair_summary = run_group('promising_pair_probe', pair_configs, probe_seeds, probe_epochs)

    all_ranked = single_summary + pair_summary
    all_ranked.sort(key=lambda item: (-item['mean_primary'], item['std_primary'],
                                      canonical_config(item['config'])))
    unique_configs = []
    seen_configs = set()
    refine_count = 2 if smoke_cap is not None else 8
    for item in all_ranked:
        key = canonical_config(item['config'])
        if key not in seen_configs:
            seen_configs.add(key)
            unique_configs.append(item['config'])
        if len(unique_configs) >= refine_count:
            break
    refine_summary = run_group('full_length_refinement', unique_configs,
                               refine_seeds, refine_epochs)
    finalist_count = 1 if smoke_cap is not None else min(3, len(refine_summary))
    finalist_configs = [item['config'] for item in refine_summary[:finalist_count]]
    finalist_summary = run_group('paired_finalist_confirmation', finalist_configs,
                                 finalist_seeds, finalist_epochs)
    selected_config = finalist_summary[0]['config']

    final_seed = args.seed + 9001
    final_metrics, final_scores, final_learning = train_once(
        selected_config, final_seed, final_epochs, data, evaluator, device, True)
    history.append({
        'stage': 'final',
        'seed': final_seed,
        'epochs': final_epochs,
        'config': selected_config,
        'gauc': final_metrics['gauc'],
        'ndcg5': final_metrics['ndcg5'],
        'primary': final_metrics['primary'],
        'learning': final_learning,
    })

    payload = {
        'gauc': final_metrics['gauc'],
        'ndcg5': final_metrics['ndcg5'],
        'primary': final_metrics['primary'],
        'selected_config': selected_config,
        'single_addon_summary': single_summary,
        'pair_summary': pair_summary,
        'refinement_summary': refine_summary,
        'finalist_summary': finalist_summary,
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
