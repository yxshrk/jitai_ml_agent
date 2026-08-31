import argparse
import csv
import datetime as dt
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def date_ordinal(value):
    s = str(value).strip()
    try:
        if '.' in s:
            s = str(int(float(s)))
        s = s.replace('-', '')
        return dt.date(int(s[:4]), int(s[4:6]), int(s[6:8])).toordinal()
    except Exception:
        return 0


def parse_hourmin(value):
    try:
        v = int(float(str(value).strip()))
        hour, minute = v // 100, v % 100
        if 0 <= hour < 24 and 0 <= minute < 60:
            return hour, minute
    except Exception:
        pass
    return 24, 0


def load_fast(data_dir):
    tr = np.load(Path(data_dir) / 'train.npz', allow_pickle=False)
    va = np.load(Path(data_dir) / 'val.npz', allow_pickle=False)
    Xtr = np.asarray(tr['X'], dtype=np.int64)
    Xva = np.asarray(va['X'], dtype=np.int64)
    ytr = np.asarray(tr['y'], dtype=np.float32)
    yva = np.asarray(va['y'], dtype=np.float32)
    dims = np.asarray(tr['field_dims'], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(dims[:-1]))).astype(np.int64)
    return {
        'Xtr': Xtr,
        'ytr': ytr,
        'utr': np.asarray(tr['user']),
        'dates': np.asarray(tr['date']) if 'date' in tr.files else np.zeros(len(ytr), dtype=np.int64),
        'hourmin_tr': np.asarray(tr['hourmin']) if 'hourmin' in tr.files else np.zeros(len(ytr), dtype=np.int64),
        'Xva': Xva,
        'yva': yva,
        'uva': np.asarray(va['user']),
        'dates_va': np.asarray(va['date']) if 'date' in va.files else np.zeros(len(yva), dtype=np.int64),
        'hourmin_va': np.asarray(va['hourmin']) if 'hourmin' in va.files else np.zeros(len(yva), dtype=np.int64),
        'video_out': Xva[:, 1] - offsets[1],
        'field_dims': dims,
        'fast': True,
    }


def read_csv_rows(path):
    rows = []
    with open(path, 'r', newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rows.append({
                'user_id': row['user_id'],
                'video_id': row['video_id'],
                'author_id': row.get('author_id', '__missing_author__'),
                'tab': row.get('tab', '0'),
                'duration_ms': float(row.get('duration_ms', 0) or 0),
                'date': row.get('date', '0'),
                'hourmin': row.get('hourmin', '0'),
                'long_view': float(row['long_view']),
            })
    return rows


def load_csv(data_dir):
    train_rows = read_csv_rows(Path(data_dir) / 'train.csv')
    val_rows = read_csv_rows(Path(data_dir) / 'val.csv')
    durations = np.asarray([r['duration_ms'] for r in train_rows], dtype=np.float64)
    cuts = np.quantile(durations, np.linspace(0.1, 0.9, 9)) if len(durations) else np.zeros(9)

    def raw(rows):
        return [[r['user_id'], r['video_id'], r['author_id'], r['tab'],
                 str(int(np.searchsorted(cuts, r['duration_ms'], side='right')))] for r in rows]

    raw_tr, raw_va = raw(train_rows), raw(val_rows)
    maps, dims = [], []
    for j in range(5):
        values = sorted({r[j] for r in raw_tr})
        mapping = {v: i + 1 for i, v in enumerate(values)}
        maps.append(mapping)
        dims.append(len(mapping) + 1)
    offsets = np.concatenate(([0], np.cumsum(dims[:-1]))).astype(np.int64)

    def encode(rows):
        X = np.empty((len(rows), 5), dtype=np.int64)
        for i, row in enumerate(rows):
            for j in range(5):
                X[i, j] = maps[j].get(row[j], 0) + offsets[j]
        return X

    return {
        'Xtr': encode(raw_tr),
        'ytr': np.asarray([r['long_view'] for r in train_rows], dtype=np.float32),
        'utr': np.asarray([r['user_id'] for r in train_rows]),
        'dates': np.asarray([r['date'] for r in train_rows]),
        'hourmin_tr': np.asarray([r['hourmin'] for r in train_rows]),
        'Xva': encode(raw_va),
        'yva': np.asarray([r['long_view'] for r in val_rows], dtype=np.float32),
        'uva': np.asarray([r['user_id'] for r in val_rows]),
        'dates_va': np.asarray([r['date'] for r in val_rows]),
        'hourmin_va': np.asarray([r['hourmin'] for r in val_rows]),
        'video_out': np.asarray([r['video_id'] for r in val_rows]),
        'field_dims': np.asarray(dims, dtype=np.int64),
        'fast': False,
    }


def add_session_time_features(data):
    base_dims = np.asarray(data['field_dims'], dtype=np.int64)
    base_offsets = np.concatenate(([0], np.cumsum(base_dims[:-1]))).astype(np.int64)
    tab_dim = int(base_dims[3])
    gap_edges = np.asarray([1, 5, 15, 30, 60, 180, 720], dtype=np.float64)
    position_edges = np.asarray([1, 2, 3, 5, 8, 16], dtype=np.int64)
    extra_dims = np.asarray([9, 7, 25 * tab_dim, 8 * tab_dim], dtype=np.int64)
    cache = {}

    def ordinal(value):
        key = str(value)
        if key not in cache:
            cache[key] = date_ordinal(value)
        return cache[key]

    def derive(users, dates, hourmins, X, state):
        local = np.empty((len(users), 4), dtype=np.int64)
        tabs = np.clip(X[:, 3] - base_offsets[3], 0, tab_dim - 1).astype(np.int64)
        for i in range(len(users)):
            user = users[i].item() if isinstance(users[i], np.generic) else users[i]
            day = ordinal(dates[i])
            hour, minute = parse_hourmin(hourmins[i])
            timestamp = day * 1440 + hour * 60 + minute if day > 0 and hour < 24 else None
            previous = state.get(user)
            gap_code, position = 0, 1
            if previous is not None and timestamp is not None and previous[0] is not None:
                gap = timestamp - previous[0]
                if gap >= 0:
                    gap_code = 1 + int(np.searchsorted(gap_edges, float(gap), side='right'))
                    position = previous[1] + 1 if gap <= 30 else 1
            weekday = (day - 1) % 7 if day > 0 else 7
            local[i] = [gap_code,
                        int(np.searchsorted(position_edges, position, side='right')),
                        hour * tab_dim + tabs[i],
                        weekday * tab_dim + tabs[i]]
            state[user] = (timestamp, position)
        return local

    state = {}
    tr = derive(data['utr'], data['dates'], data['hourmin_tr'], data['Xtr'], state)
    va = derive(data['uva'], data['dates_va'], data['hourmin_va'], data['Xva'], state)
    offsets = int(np.sum(base_dims)) + np.concatenate(([0], np.cumsum(extra_dims[:-1]))).astype(np.int64)
    data['Xtr'] = np.concatenate([data['Xtr'], tr + offsets], axis=1)
    data['Xva'] = np.concatenate([data['Xva'], va + offsets], axis=1)
    data['field_dims'] = np.concatenate([base_dims, extra_dims])


def make_recency_weights(dates, half_life=7.0):
    ords = np.asarray([date_ordinal(x) for x in dates], dtype=np.int64)
    valid = ords > 0
    if not np.any(valid):
        return np.ones(len(dates), dtype=np.float32)
    newest = int(np.max(ords[valid]))
    ages = np.maximum(0, newest - ords)
    weights = np.exp(-math.log(2.0) * ages / half_life).astype(np.float32)
    weights[~valid] = 1.0
    weights /= max(float(weights.mean()), 1e-8)
    return weights


def make_pairs(users, labels, seed):
    order = np.argsort(users, kind='mergesort')
    sorted_users = users[order]
    rng = np.random.default_rng(seed)
    positives, negatives = [], []
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_users[end] == sorted_users[start]:
            end += 1
        idx = order[start:end]
        pos = idx[labels[idx] > 0.5]
        neg = idx[labels[idx] <= 0.5]
        if len(pos) and len(neg):
            positives.append(pos.astype(np.int64, copy=False))
            negatives.append(neg[rng.integers(0, len(neg), size=len(pos))].astype(np.int64, copy=False))
        start = end
    if not positives:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    return np.concatenate(positives), np.concatenate(negatives)


class RankModel(nn.Module):
    def __init__(self, n_vocab, n_fields, dropout):
        super().__init__()
        k = 16
        self.embedding = nn.Embedding(n_vocab, k)
        self.linear = nn.Embedding(n_vocab, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        self.embed_dropout = nn.Dropout(dropout)
        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.zeros_(self.linear.weight)
        d = n_fields * k
        self.cross_scalar = nn.Linear(d, 1, bias=False)
        self.cross_bias = nn.Parameter(torch.zeros(d))
        self.deep = nn.Sequential(
            nn.Linear(d, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout),
        )
        self.head = nn.Linear(d + 32, 1)

    def forward(self, x):
        emb = self.embed_dropout(self.embedding(x))
        linear = self.linear(x).sum(dim=1).squeeze(-1) + self.bias
        x0 = emb.reshape(emb.shape[0], -1)
        cross = x0 * self.cross_scalar(x0) + x0 + self.cross_bias
        deep = self.deep(x0)
        return linear + self.head(torch.cat([cross, deep], dim=1)).squeeze(-1)


def metric_function(fast):
    if fast:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    return evaluate


def normalize_metrics(result):
    return {
        'gauc': float(result.get('GAUC', result.get('gauc'))),
        'ndcg5': float(result.get('nDCG@5', result.get('ndcg5'))),
        'primary': float(result.get('primary')),
    }


def predict(model, X, batch_size, device):
    model.eval()
    pieces = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.as_tensor(X[start:start + batch_size], dtype=torch.long, device=device)
            pieces.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(pieces).astype(np.float64)


def centered_bce(logits, labels, weights, user_codes):
    _, inverse = torch.unique(user_codes, sorted=False, return_inverse=True)
    counts = torch.zeros(int(inverse.max().item()) + 1, dtype=logits.dtype, device=logits.device)
    sums = torch.zeros_like(counts)
    counts.scatter_add_(0, inverse, torch.ones_like(logits))
    sums.scatter_add_(0, inverse, logits)
    centered = logits - sums[inverse] / counts[inverse].clamp_min(1.0)
    return (F.binary_cross_entropy_with_logits(centered, labels, reduction='none') * weights).mean()


def lambda_pair_weights(model, data, pair_pos, pair_neg, batch_size, device):
    if not len(pair_pos):
        return np.empty(0, dtype=np.float32)
    scores = predict(model, data['Xtr'], batch_size, device)
    users = data['utr']
    labels = data['ytr']
    order = np.argsort(users, kind='mergesort')
    ranks = np.empty(len(users), dtype=np.int64)
    idcg = np.ones(len(users), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and users[order[end]] == users[order[start]]:
            end += 1
        idx = order[start:end]
        ranked = idx[np.argsort(-scores[idx], kind='mergesort')]
        ranks[ranked] = np.arange(len(idx), dtype=np.int64)
        positives = int(np.sum(labels[idx] > 0.5))
        if positives > 0:
            ideal = np.sum(1.0 / np.log2(np.arange(positives, dtype=np.float64) + 2.0))
            idcg[idx] = max(float(ideal), 1e-8)
        start = end
    dp = 1.0 / np.log2(ranks[pair_pos].astype(np.float64) + 2.0)
    dn = 1.0 / np.log2(ranks[pair_neg].astype(np.float64) + 2.0)
    weights = np.abs(dp - dn) / idcg[pair_pos]
    weights = np.maximum(weights, 1e-4)
    weights /= max(float(weights.mean()), 1e-8)
    return weights.astype(np.float32)


def train_member(data, objective, seed, epochs, device, evaluator, pair_pos, pair_neg):
    seed_all(seed)
    Xtr, ytr = data['Xtr'], data['ytr']
    n_vocab = max(int(np.sum(data['field_dims'])), int(Xtr.max()) + 1, int(data['Xva'].max()) + 1)
    model = RankModel(n_vocab, Xtr.shape[1], 0.21).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00168, weight_decay=3.7e-5)
    batch_size = 16384 if device.type == 'cuda' else 8192
    recency = data['recency7']
    _, user_codes = np.unique(data['utr'], return_inverse=True)
    user_codes = user_codes.astype(np.int64)
    rng = np.random.default_rng(seed + 991)
    best_primary = -float('inf')
    best_metrics, best_predictions, best_state = None, None, None
    best_checkpoint = 0.0
    trajectory = []

    for epoch in range(epochs):
        model.train()
        point_order = rng.permutation(len(Xtr))
        pair_order = rng.permutation(len(pair_pos)) if len(pair_pos) else np.empty(0, dtype=np.int64)
        pair_cursor = 0
        lambda_weights = None
        if objective == 'lambda-hybrid':
            lambda_weights = lambda_pair_weights(model, data, pair_pos, pair_neg, batch_size, device)
            model.train()
        total_steps = max(1, math.ceil(len(point_order) / batch_size))
        for step in range(total_steps):
            idx = point_order[step * batch_size:(step + 1) * batch_size]
            xb = torch.as_tensor(Xtr[idx], dtype=torch.long, device=device)
            yb = torch.as_tensor(ytr[idx], dtype=torch.float32, device=device)
            wb = torch.as_tensor(recency[idx], dtype=torch.float32, device=device)
            logits = model(xb)
            if objective == 'gauge-hybrid':
                ub = torch.as_tensor(user_codes[idx], dtype=torch.long, device=device)
                point_loss = centered_bce(logits, yb, wb, ub)
            else:
                point_loss = (F.binary_cross_entropy_with_logits(logits, yb, reduction='none') * wb).mean()

            if len(pair_pos):
                need = len(idx)
                if pair_cursor + need > len(pair_order):
                    pair_order = rng.permutation(len(pair_pos))
                    pair_cursor = 0
                selected = pair_order[pair_cursor:pair_cursor + need]
                pair_cursor += len(selected)
                pi, ni = pair_pos[selected], pair_neg[selected]
                xp = torch.as_tensor(Xtr[pi], dtype=torch.long, device=device)
                xn = torch.as_tensor(Xtr[ni], dtype=torch.long, device=device)
                pair_weight = 0.5 * (recency[pi] + recency[ni])
                if lambda_weights is not None:
                    pair_weight = pair_weight * lambda_weights[selected]
                pw = torch.as_tensor(pair_weight, dtype=torch.float32, device=device)
                pair_loss = (F.softplus(-(model(xp) - model(xn))) * pw).mean()
                loss = 0.5 * point_loss + 0.5 * pair_loss
            else:
                loss = point_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        predictions = predict(model, data['Xva'], batch_size, device)
        metrics = normalize_metrics(evaluator(data['uva'], data['yva'], predictions))
        trajectory.append({'checkpoint': float(epoch + 1), **metrics})
        if metrics['primary'] > best_primary:
            best_primary = metrics['primary']
            best_metrics = metrics
            best_predictions = predictions.copy()
            best_checkpoint = float(epoch + 1)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        for group in optimizer.param_groups:
            group['lr'] *= 0.72

    if best_state is not None:
        model.load_state_dict(best_state)
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return {
        'objective': objective,
        'metrics': best_metrics,
        'predictions': best_predictions,
        'best_checkpoint': best_checkpoint,
        'trajectory': trajectory,
    }


def within_user_ranks(users, scores):
    order = np.argsort(users, kind='mergesort')
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and users[order[end]] == users[order[start]]:
            end += 1
        idx = order[start:end]
        ranked = idx[np.argsort(scores[idx], kind='mergesort')]
        if len(idx) == 1:
            ranks[idx] = 0.5
        else:
            ranks[ranked] = np.arange(len(idx), dtype=np.float64) / float(len(idx) - 1)
        start = end
    return ranks


def append_progress(path, record):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, sort_keys=True) + '\n')


def main():
    args = parse_args()
    seed_all(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / 'progress.log'
    if progress_path.exists():
        progress_path.unlink()

    data_dir = Path(args.data_dir)
    fast = (data_dir / 'train.npz').exists() and (data_dir / 'val.npz').exists()
    data = load_fast(data_dir) if fast else load_csv(data_dir)
    add_session_time_features(data)
    data['recency7'] = make_recency_weights(data['dates'], 7.0)
    evaluator = metric_function(fast)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pair_pos, pair_neg = make_pairs(data['utr'], data['ytr'], args.seed + 17)

    smoke = os.environ.get('SMOKE_EPOCHS')
    smoke_cap = int(smoke) if smoke is not None else None
    epochs = 1
    if smoke_cap is not None:
        epochs = max(1, min(epochs, smoke_cap))

    objectives = ['bpr-hybrid', 'gauge-hybrid', 'lambda-hybrid']
    results = []
    history = []
    for member_id, objective in enumerate(objectives):
        result = train_member(data, objective, args.seed + member_id * 1009, epochs,
                              device, evaluator, pair_pos, pair_neg)
        results.append(result)
        entry = {
            'phase': 'objective-member',
            'member': member_id,
            'seed': args.seed + member_id * 1009,
            'objective': objective,
            'best_checkpoint': result['best_checkpoint'],
            'metrics': result['metrics'],
            'trajectory': result['trajectory'],
        }
        history.append(entry)
        append_progress(progress_path, {
            'phase': 'objective-member',
            'member': member_id,
            'objective': objective,
            'primary': result['metrics']['primary'],
        })

    champion = results[0]
    competence_margin = 0.005
    competent = [0]
    for i in range(1, len(results)):
        if results[i]['metrics']['primary'] >= champion['metrics']['primary'] - competence_margin:
            competent.append(i)

    candidate_metrics = None
    candidate_predictions = None
    if len(competent) >= 2:
        ranked = [within_user_ranks(data['uva'], results[i]['predictions']) for i in competent]
        candidate_predictions = np.mean(np.stack(ranked, axis=0), axis=0)
        candidate_metrics = normalize_metrics(evaluator(data['uva'], data['yva'], candidate_predictions))

    required_primary_gain = 0.002
    max_component_harm = 0.001
    accepted = False
    if candidate_metrics is not None:
        accepted = (
            candidate_metrics['primary'] >= champion['metrics']['primary'] + required_primary_gain
            and candidate_metrics['gauc'] >= champion['metrics']['gauc'] - max_component_harm
            and candidate_metrics['ndcg5'] >= champion['metrics']['ndcg5'] - max_component_harm
        )

    if accepted:
        predictions = candidate_predictions
        final_metrics = candidate_metrics
        selected = 'objective-diverse-rank-ensemble'
    else:
        predictions = champion['predictions']
        final_metrics = champion['metrics']
        selected = 'bpr-hybrid-champion-fallback'

    gate_record = {
        'phase': 'ensemble-gate',
        'competent_members': competent,
        'candidate_metrics': candidate_metrics,
        'champion_metrics': champion['metrics'],
        'required_primary_gain': required_primary_gain,
        'max_component_harm': max_component_harm,
        'accepted': accepted,
        'selected': selected,
    }
    history.append(gate_record)
    append_progress(progress_path, gate_record)

    with open(out_dir / 'predictions.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['row_id', 'user_id', 'video_id', 'score'])
        for i, (user, video, score) in enumerate(zip(data['uva'], data['video_out'], predictions)):
            writer.writerow([i, user, video, format(float(score), '.10g')])

    payload = {
        'gauc': final_metrics['gauc'],
        'ndcg5': final_metrics['ndcg5'],
        'primary': final_metrics['primary'],
        'selected': selected,
        'ensemble_recipe': {
            'objectives': objectives,
            'blend': 'equal within-user ranks among competent members',
            'competence_margin': competence_margin,
            'required_primary_gain': required_primary_gain,
            'max_component_harm': max_component_harm,
            'weights_swept': False,
        },
        'base_config': {
            'architecture': 'dcn-lite',
            'embedding_dim': 16,
            'dropout': 0.21,
            'weight_decay': 3.7e-5,
            'learning_rate': 0.00168,
            'lr_decay': 0.72,
            'weighting': 'recency-7d',
            'session_time_features': True,
        },
        'history': history,
    }
    with open(out_dir / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(payload, f, sort_keys=True)


if __name__ == '__main__':
    main()
