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
    def read_rows(path):
        rows = []
        with open(path, newline='') as fh:
            for row in csv.DictReader(fh):
                rows.append({
                    'user': row['user_id'],
                    'video': row['video_id'],
                    'tab': row['tab'],
                    'duration': float(row['duration_ms']),
                    'date': row['date'],
                    'y': float(row['long_view'])
                })
        return rows

    train_rows = read_rows(os.path.join(data_dir, 'train.csv'))
    val_rows = read_rows(os.path.join(data_dir, 'val.csv'))
    durations = np.asarray([r['duration'] for r in train_rows], dtype=np.float64)
    edges = np.unique(np.quantile(durations, np.linspace(0.1, 0.9, 9)))
    user_values = sorted(set(r['user'] for r in train_rows))
    video_values = sorted(set(r['video'] for r in train_rows))
    tab_values = sorted(set(r['tab'] for r in train_rows))
    user_map = {v: i + 1 for i, v in enumerate(user_values)}
    video_map = {v: i + 1 for i, v in enumerate(video_values)}
    author_map = {'0': 1}
    tab_map = {v: i + 1 for i, v in enumerate(tab_values)}
    dur_map = {str(i): i + 1 for i in range(10)}
    maps = [user_map, video_map, author_map, tab_map, dur_map]
    field_dims = np.asarray([len(m) + 1 for m in maps], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1]))

    def encode(rows):
        x = np.zeros((len(rows), 5), dtype=np.int32)
        users = np.empty(len(rows), dtype=object)
        videos = np.empty(len(rows), dtype=object)
        for i, row in enumerate(rows):
            bucket = str(int(np.searchsorted(edges, row['duration'], side='right')))
            values = [row['user'], row['video'], '0', row['tab'], bucket]
            for j, value in enumerate(values):
                x[i, j] = int(offsets[j] + maps[j].get(value, 0))
            users[i] = row['user']
            videos[i] = row['video']
        return {
            'X': x,
            'y': np.asarray([r['y'] for r in rows], dtype=np.float32),
            'user': users,
            'video_raw': videos,
            'date': np.asarray([r['date'] for r in rows]),
            'field_dims': field_dims
        }

    return encode(train_rows), encode(val_rows), False


def load_data(data_dir):
    train_path = os.path.join(data_dir, 'train.npz')
    val_path = os.path.join(data_dir, 'val.npz')
    if os.path.exists(train_path) and os.path.exists(val_path):
        with np.load(train_path) as src:
            train = {k: src[k] for k in src.files}
        with np.load(val_path) as src:
            val = {k: src[k] for k in src.files}
        val['video_raw'] = np.asarray(val['X'])[:, 1].astype(np.int64)
        return train, val, True
    return load_csv_data(data_dir)


def metric_values(evaluator, users, labels, scores):
    result = evaluator(users, labels.astype(int), scores)
    return {
        'gauc': float(result['GAUC'] if 'GAUC' in result else result['gauc']),
        'ndcg5': float(result['nDCG@5'] if 'nDCG@5' in result else result['ndcg5']),
        'primary': float(result['primary'])
    }


def append_progress(path, record):
    with open(path, 'a') as fh:
        fh.write(json.dumps(record, sort_keys=True) + '\n')


def predict(model, x, device, batch_size=65536):
    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = x[start:start + batch_size].to(device, non_blocking=True)
            result.append(model(xb).detach().cpu().numpy())
    return np.concatenate(result).astype(np.float64)


def rank_within_user(users, scores):
    users = np.asarray(users)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(users, kind='stable')
    sorted_users = users[order]
    result = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_users[end] == sorted_users[start]:
            end += 1
        idx = order[start:end]
        local = np.argsort(scores[idx], kind='mergesort')
        ranks = np.empty(end - start, dtype=np.float64)
        ranks[local] = np.arange(end - start, dtype=np.float64)
        if end - start > 1:
            ranks /= float(end - start - 1)
        else:
            ranks[:] = 0.5
        result[idx] = ranks
        start = end
    return result


def recency_weights(dates, half_life=7.0):
    ords = parse_date_ord(dates)
    valid = ords > 0
    newest = int(ords[valid].max()) if np.any(valid) else 0
    age = np.maximum(newest - ords, 0).astype(np.float32)
    weights = np.exp(-math.log(2.0) * age / float(half_life)).astype(np.float32)
    weights /= max(float(weights.mean()), 1e-8)
    return weights


def make_pair_index(users, labels, indices):
    indices = np.asarray(indices, dtype=np.int64)
    local_users = np.asarray(users)[indices]
    local_labels = np.asarray(labels)[indices]
    _, groups = np.unique(local_users, return_inverse=True)
    groups = groups.astype(np.int64)
    neg_local = np.flatnonzero(local_labels < 0.5).astype(np.int64)
    order = np.argsort(groups[neg_local], kind='stable')
    neg_sorted = neg_local[order]
    group_count = int(groups.max()) + 1 if len(groups) else 0
    counts = np.bincount(groups[neg_sorted], minlength=group_count).astype(np.int64)
    starts = np.zeros(group_count, dtype=np.int64)
    if group_count > 1:
        starts[1:] = np.cumsum(counts[:-1])
    positives = np.flatnonzero((local_labels >= 0.5) & (counts[groups] > 0)).astype(np.int64)
    return positives, groups, neg_sorted, starts, counts


def train_champion(seed, epochs, train_indices, xt, y_array, train_users, train_dates,
                   eval_x, eval_users, eval_y, total_dim, device, evaluator,
                   select_best=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = DCNLite(total_dim, dropout=0.25).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=3e-4)
    train_indices = np.asarray(train_indices, dtype=np.int64)
    pair_data = make_pair_index(train_users, y_array, train_indices)
    positives, groups, neg_sorted, starts, counts = pair_data
    if len(positives) == 0:
        raise RuntimeError('No within-user positive-negative pairs are available')
    all_weights = recency_weights(train_dates, 7.0)
    local_weights = torch.from_numpy(all_weights[train_indices])
    local_y = torch.from_numpy(np.asarray(y_array[train_indices], dtype=np.float32))
    local_x = xt[torch.from_numpy(train_indices)]
    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed + 17011)
    rng = np.random.default_rng(seed + 29023)
    best_primary = -1.0
    best_scores = None
    checkpoints = []
    batch_size = 8192
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(len(train_indices), generator=generator)
        running = 0.0
        batches = 0
        for start in range(0, len(train_indices), batch_size):
            local_idx = permutation[start:start + batch_size]
            xb = local_x[local_idx].to(device, non_blocking=True)
            yb = local_y[local_idx].to(device, non_blocking=True)
            wb = local_weights[local_idx].to(device, non_blocking=True)
            logits = model(xb)
            bce_each = torch.nn.functional.binary_cross_entropy_with_logits(logits, yb, reduction='none')
            bce = (bce_each * wb).sum() / wb.sum().clamp_min(1e-8)
            pair_n = max(1, len(local_idx) // 2)
            pos = positives[rng.integers(0, len(positives), size=pair_n)]
            pos_group = groups[pos]
            offsets = (rng.random(pair_n) * counts[pos_group]).astype(np.int64)
            neg = neg_sorted[starts[pos_group] + offsets]
            pos_t = torch.from_numpy(pos)
            neg_t = torch.from_numpy(neg)
            pair_x = torch.cat((local_x[pos_t], local_x[neg_t]), dim=0).to(device, non_blocking=True)
            pair_logits = model(pair_x)
            pos_logits = pair_logits[:pair_n]
            neg_logits = pair_logits[pair_n:]
            pair_weights = 0.5 * (local_weights[pos_t] + local_weights[neg_t])
            pair_weights_device = pair_weights.to(device, non_blocking=True)
            pair_each = torch.nn.functional.softplus(-(pos_logits - neg_logits))
            bpr = (pair_each * pair_weights_device).sum() / pair_weights_device.sum().clamp_min(1e-8)
            loss = 0.5 * bce + 0.5 * bpr
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(loss.detach().cpu())
            batches += 1
        for group in optimizer.param_groups:
            group['lr'] *= 0.5
        scores = predict(model, eval_x, device)
        metrics = metric_values(evaluator, eval_users, eval_y, scores)
        checkpoints.append({
            'epoch': epoch + 1,
            'train_loss': running / max(batches, 1),
            'primary': metrics['primary'],
            'gauc': metrics['gauc']
        })
        if not select_best or metrics['primary'] > best_primary:
            best_primary = metrics['primary']
            best_scores = scores.copy()
    return best_scores, float(best_primary), checkpoints


def compressed_ids(values):
    _, inverse = np.unique(np.asarray(values), return_inverse=True)
    return inverse.astype(np.int64)


def build_signed_sketch(x, labels, users_raw, dates, seed, dim=64):
    x = np.asarray(x)
    labels = np.asarray(labels, dtype=np.float64)
    user_codes = compressed_ids(users_raw)
    video_codes = compressed_ids(x[:, 1])
    n_users = int(user_codes.max()) + 1 if len(user_codes) else 0
    n_videos = int(video_codes.max()) + 1 if len(video_codes) else 0
    weights = recency_weights(dates, 7.0).astype(np.float64)
    weighted_sum = np.bincount(user_codes, weights=weights * labels, minlength=n_users)
    weight_sum = np.bincount(user_codes, weights=weights, minlength=n_users)
    global_mean = float(np.sum(weights * labels) / max(np.sum(weights), 1e-12))
    user_mean = np.divide(weighted_sum, weight_sum, out=np.full(n_users, global_mean), where=weight_sum > 0)
    residual = np.sqrt(weights) * (labels - user_mean[user_codes])
    rng = np.random.default_rng(seed + 44021)
    hashes = rng.integers(0, 2, size=(n_users, dim), dtype=np.int8).astype(np.float32)
    hashes = hashes * 2.0 - 1.0
    hashes /= math.sqrt(float(dim))
    video_sketch = np.zeros((n_videos, dim), dtype=np.float32)
    np.add.at(video_sketch, video_codes, residual[:, None].astype(np.float32) * hashes[user_codes])
    norms = np.linalg.norm(video_sketch, axis=1, keepdims=True)
    video_sketch = np.divide(video_sketch, np.maximum(norms, 1e-12), out=np.zeros_like(video_sketch))
    taste = np.zeros((n_users, dim), dtype=np.float32)
    np.add.at(taste, user_codes, residual[:, None].astype(np.float32) * video_sketch[video_codes])
    taste_norms = np.linalg.norm(taste, axis=1, keepdims=True)
    taste = np.divide(taste, np.maximum(taste_norms, 1e-12), out=np.zeros_like(taste))
    user_lookup = {value: i for i, value in enumerate(np.unique(np.asarray(users_raw)))}
    video_lookup = {value: i for i, value in enumerate(np.unique(np.asarray(x)[:, 1]))}
    return taste, video_sketch, user_lookup, video_lookup


def score_signed_sketch(model_data, query_x, query_users):
    taste, video_sketch, user_lookup, video_lookup = model_data
    query_x = np.asarray(query_x)
    query_users = np.asarray(query_users)
    result = np.zeros(len(query_x), dtype=np.float64)
    for i in range(len(query_x)):
        user_index = user_lookup.get(query_users[i])
        video_index = video_lookup.get(query_x[i, 1])
        if user_index is not None and video_index is not None:
            result[i] = float(np.dot(taste[user_index], video_sketch[video_index]))
    return result


def rolling_split(dates):
    ords = parse_date_ord(dates)
    valid = np.unique(ords[ords > 0])
    if len(valid) >= 3:
        hold_days = max(1, int(math.ceil(len(valid) * 0.2)))
        cutoff = valid[-hold_days]
        early = np.flatnonzero((ords > 0) & (ords < cutoff))
        hold = np.flatnonzero(ords >= cutoff)
    else:
        split = max(1, int(len(ords) * 0.8))
        early = np.arange(split, dtype=np.int64)
        hold = np.arange(split, len(ords), dtype=np.int64)
    if len(early) == 0 or len(hold) == 0:
        split = max(1, int(len(ords) * 0.8))
        early = np.arange(split, dtype=np.int64)
        hold = np.arange(split, len(ords), dtype=np.int64)
    return early.astype(np.int64), hold.astype(np.int64)


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

    train, val, fast_path = load_data(args.data_dir)
    if fast_path:
        from data.official.evaluate import evaluate as evaluator
    else:
        from harness.evaluate_provisional import evaluate as evaluator

    train_x_np = np.asarray(train['X'], dtype=np.int64)
    val_x_np = np.asarray(val['X'], dtype=np.int64)
    train_y = np.asarray(train['y'], dtype=np.float32)
    val_y = np.asarray(val['y'], dtype=np.float32)
    train_users = np.asarray(train['user'])
    val_users = np.asarray(val['user'])
    train_dates = np.asarray(train['date'])
    xt = torch.from_numpy(train_x_np)
    xv = torch.from_numpy(val_x_np)
    total_dim = int(np.asarray(train['field_dims']).sum())

    smoke_value = os.environ.get('SMOKE_EPOCHS')
    smoke_cap = int(smoke_value) if smoke_value is not None else None
    rolling_epochs = min(5, smoke_cap) if smoke_cap is not None else 5
    final_epochs = min(args.epochs, smoke_cap) if smoke_cap is not None else args.epochs
    history = []

    early_idx, hold_idx = rolling_split(train_dates)
    hold_x = xt[torch.from_numpy(hold_idx)]
    hold_users = train_users[hold_idx]
    hold_y = train_y[hold_idx]
    rolling_scores, rolling_primary, rolling_checkpoints = train_champion(
        args.seed + 7001, rolling_epochs, early_idx, xt, train_y, train_users,
        train_dates, hold_x, hold_users, hold_y, total_dim, device, evaluator, True)
    rolling_graph_model = build_signed_sketch(
        train_x_np[early_idx], train_y[early_idx], train_users[early_idx],
        train_dates[early_idx], args.seed, 64)
    rolling_graph_scores = score_signed_sketch(
        rolling_graph_model, train_x_np[hold_idx], hold_users)
    rolling_champion_rank = rank_within_user(hold_users, rolling_scores)
    rolling_graph_rank = rank_within_user(hold_users, rolling_graph_scores)
    alpha_results = []
    best_alpha = 0.05
    best_alpha_primary = -1.0
    for alpha in (0.05, 0.1, 0.2):
        blended = rolling_champion_rank + alpha * rolling_graph_rank
        metrics = metric_values(evaluator, hold_users, hold_y, blended)
        record = {
            'stage': 'rolling_alpha_probe',
            'alpha': alpha,
            'primary': metrics['primary'],
            'gauc': metrics['gauc'],
            'ndcg5': metrics['ndcg5']
        }
        alpha_results.append(record)
        history.append(record)
        append_progress(progress_path, record)
        if metrics['primary'] > best_alpha_primary:
            best_alpha_primary = metrics['primary']
            best_alpha = alpha
    history.append({
        'stage': 'rolling_champion',
        'seed': args.seed + 7001,
        'epochs': rolling_epochs,
        'primary': rolling_primary,
        'checkpoints': rolling_checkpoints
    })

    all_indices = np.arange(len(train_y), dtype=np.int64)
    member_scores = []
    member_primaries = []
    for member in range(5):
        member_seed = args.seed + member
        scores, primary, checkpoints = train_champion(
            member_seed, final_epochs, all_indices, xt, train_y, train_users,
            train_dates, xv, val_users, val_y, total_dim, device, evaluator, True)
        for previous in member_scores:
            if np.allclose(scores, previous, rtol=1e-6, atol=1e-7):
                raise AssertionError('Distinct-seed ensemble members produced identical predictions')
        member_scores.append(scores)
        member_primaries.append(primary)
        record = {
            'stage': 'final_member',
            'member': member + 1,
            'seed': member_seed,
            'epochs': final_epochs,
            'primary': primary,
            'checkpoints': checkpoints
        }
        history.append(record)
        append_progress(progress_path, {
            'stage': 'final_member',
            'member': member + 1,
            'seed': member_seed,
            'primary': primary
        })

    champion_ranks = [rank_within_user(val_users, scores) for scores in member_scores]
    champion_scores = np.mean(np.stack(champion_ranks, axis=0), axis=0)
    for scores in member_scores:
        if np.allclose(scores, champion_scores, rtol=1e-6, atol=1e-7):
            raise AssertionError('An ensemble member is identical to the parent ensemble prediction')

    final_graph_model = build_signed_sketch(
        train_x_np, train_y, train_users, train_dates, args.seed, 64)
    graph_scores = score_signed_sketch(final_graph_model, val_x_np, val_users)
    graph_ranks = rank_within_user(val_users, graph_scores)
    final_scores = champion_scores + best_alpha * graph_ranks
    if np.allclose(final_scores, champion_scores, rtol=1e-6, atol=1e-7):
        raise AssertionError('Signed-sketch blend is identical to the parent champion predictions')

    champion_metrics = metric_values(evaluator, val_users, val_y, champion_scores)
    final_metrics = metric_values(evaluator, val_users, val_y, final_scores)
    history.append({
        'stage': 'champion_ensemble',
        'members': 5,
        'primary': champion_metrics['primary']
    })
    history.append({
        'stage': 'signed_sketch_blend',
        'dimension': 64,
        'alpha': best_alpha,
        'primary': final_metrics['primary']
    })
    append_progress(progress_path, {
        'stage': 'signed_sketch_blend',
        'alpha': best_alpha,
        'champion_primary': champion_metrics['primary'],
        'primary': final_metrics['primary']
    })

    metrics_out = {
        'gauc': final_metrics['gauc'],
        'ndcg5': final_metrics['ndcg5'],
        'primary': final_metrics['primary'],
        'selected_alpha': best_alpha,
        'sketch_dimension': 64,
        'champion_primary': champion_metrics['primary'],
        'member_primaries': member_primaries,
        'rolling_alpha_results': alpha_results,
        'history': history
    }
    with open(os.path.join(args.out_dir, 'metrics.json'), 'w') as fh:
        json.dump(metrics_out, fh)

    video_values = np.asarray(val.get('video_raw', val_x_np[:, 1]))
    with open(os.path.join(args.out_dir, 'predictions.csv'), 'w') as fh:
        fh.write('row_id,user_id,video_id,score\n')
        for i, score in enumerate(final_scores):
            fh.write(f'{i},{val_users[i]},{video_values[i]},{score:.9g}\n')


if __name__ == '__main__':
    main()
