import argparse
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def as_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_csv_rows(path):
    with open(path, 'r', newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def make_mapping(values):
    unique = sorted(set(values))
    return {value: i + 1 for i, value in enumerate(unique)}


def date_ordinal(value):
    text = str(value).replace('-', '')
    return as_int(text, 0)


def load_csv_data(data_dir):
    train_rows = read_csv_rows(Path(data_dir) / 'train.csv')
    val_rows = read_csv_rows(Path(data_dir) / 'val.csv')

    user_map = make_mapping([r.get('user_id', '') for r in train_rows])
    video_map = make_mapping([r.get('video_id', '') for r in train_rows])
    tab_map = make_mapping([r.get('tab', '') for r in train_rows])
    train_durations = np.asarray([as_float(r.get('duration_ms', 0)) for r in train_rows], dtype=np.float64)
    if len(train_durations):
        edges = np.unique(np.quantile(train_durations, np.linspace(0.1, 0.9, 9)))
    else:
        edges = np.asarray([], dtype=np.float64)

    dims = np.asarray([len(user_map) + 1, len(video_map) + 1, 1, len(tab_map) + 1, len(edges) + 1], dtype=np.int64)
    offsets = np.concatenate([np.asarray([0], dtype=np.int64), np.cumsum(dims[:-1])])

    def encode(rows):
        x = np.zeros((len(rows), 5), dtype=np.int64)
        duration = np.zeros(len(rows), dtype=np.float32)
        users = np.zeros(len(rows), dtype=np.int64)
        videos = np.zeros(len(rows), dtype=np.int64)
        dates = np.zeros(len(rows), dtype=np.int64)
        for i, row in enumerate(rows):
            uid_text = row.get('user_id', '')
            vid_text = row.get('video_id', '')
            dur = as_float(row.get('duration_ms', 0))
            local = np.asarray([
                user_map.get(uid_text, 0),
                video_map.get(vid_text, 0),
                0,
                tab_map.get(row.get('tab', ''), 0),
                int(np.searchsorted(edges, dur, side='right'))
            ], dtype=np.int64)
            x[i] = local + offsets
            duration[i] = dur
            users[i] = as_int(uid_text, user_map.get(uid_text, 0))
            videos[i] = as_int(vid_text, video_map.get(vid_text, 0))
            dates[i] = date_ordinal(row.get('date', 0))
        return x, duration, users, videos, dates

    x_train, dur_train, user_train, video_train, date_train = encode(train_rows)
    x_val, dur_val, user_val, video_val, date_val = encode(val_rows)
    y_train = np.asarray([as_float(r.get('long_view', 0)) for r in train_rows], dtype=np.float32)
    y_val = np.asarray([as_float(r.get('long_view', 0)) for r in val_rows], dtype=np.float32)
    return {
        'x_train': x_train,
        'y_train': y_train,
        'user_train': user_train,
        'duration_train': dur_train,
        'date_train': date_train,
        'x_val': x_val,
        'y_val': y_val,
        'user_val': user_val,
        'video_val': video_val,
        'duration_val': dur_val,
        'field_dims': dims,
        'fast': False
    }


def load_npz_data(data_dir):
    train = np.load(Path(data_dir) / 'train.npz', allow_pickle=False)
    val = np.load(Path(data_dir) / 'val.npz', allow_pickle=False)
    field_dims = np.asarray(train['field_dims'] if 'field_dims' in train.files else val['field_dims'], dtype=np.int64)
    x_train = np.asarray(train['X'], dtype=np.int64)
    x_val = np.asarray(val['X'], dtype=np.int64)
    user_train = np.asarray(train['user'], dtype=np.int64)
    user_val = np.asarray(val['user'], dtype=np.int64)
    video_val = np.asarray(x_val[:, 1], dtype=np.int64)
    return {
        'x_train': x_train,
        'y_train': np.asarray(train['y'], dtype=np.float32),
        'user_train': user_train,
        'duration_train': np.asarray(train['duration_ms'], dtype=np.float32),
        'date_train': np.asarray(train['date'], dtype=np.int64),
        'x_val': x_val,
        'y_val': np.asarray(val['y'], dtype=np.float32),
        'user_val': user_val,
        'video_val': video_val,
        'duration_val': np.asarray(val['duration_ms'], dtype=np.float32),
        'field_dims': field_dims,
        'fast': True
    }


def load_data(data_dir):
    if (Path(data_dir) / 'train.npz').is_file() and (Path(data_dir) / 'val.npz').is_file():
        return load_npz_data(data_dir)
    return load_csv_data(data_dir)


def recency_weights(dates):
    dates = np.asarray(dates, dtype=np.int64)
    if not len(dates):
        return np.ones(0, dtype=np.float32)
    maximum = int(np.max(dates))
    try:
        from datetime import datetime
        max_day = datetime.strptime(str(maximum), '%Y%m%d')
        age = np.asarray([(max_day - datetime.strptime(str(int(d)), '%Y%m%d')).days for d in dates], dtype=np.float32)
    except (ValueError, OverflowError):
        age = (maximum - dates).astype(np.float32)
    weights = np.exp2(-np.maximum(age, 0.0) / 7.0)
    weights /= max(float(weights.mean()), 1e-6)
    return weights.astype(np.float32)


def build_pairs(users, labels, seed, cap_per_user=4):
    users = np.asarray(users)
    labels = np.asarray(labels)
    order = np.argsort(users, kind='mergesort')
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    rng = np.random.default_rng(seed)
    positives = []
    negatives = []
    for j in range(len(boundaries) - 1):
        idx = order[boundaries[j]:boundaries[j + 1]]
        pos = idx[labels[idx] > 0.5]
        neg = idx[labels[idx] <= 0.5]
        if len(pos) == 0 or len(neg) == 0:
            continue
        count = min(cap_per_user, max(len(pos), len(neg)))
        positives.extend(rng.choice(pos, size=count, replace=len(pos) < count).tolist())
        negatives.extend(rng.choice(neg, size=count, replace=len(neg) < count).tolist())
    if not positives:
        return np.asarray([0], dtype=np.int64), np.asarray([0], dtype=np.int64)
    return np.asarray(positives, dtype=np.int64), np.asarray(negatives, dtype=np.int64)


class DCNLite(nn.Module):
    def __init__(self, field_dims, embed_dim=16, hidden=128, cross_layers=2, dropout=0.2):
        super().__init__()
        self.field_count = len(field_dims)
        self.embed_dim = embed_dim
        width = self.field_count * embed_dim
        self.embedding = nn.Embedding(int(np.sum(field_dims)), embed_dim)
        nn.init.normal_(self.embedding.weight, std=0.01)
        self.input_dropout = nn.Dropout(dropout)
        self.cross_w = nn.ParameterList([nn.Parameter(torch.empty(width)) for _ in range(cross_layers)])
        self.cross_b = nn.ParameterList([nn.Parameter(torch.zeros(width)) for _ in range(cross_layers)])
        for weight in self.cross_w:
            nn.init.normal_(weight, std=0.01)
        self.mlp = nn.Sequential(
            nn.Linear(width, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.base_head = nn.Linear(width + hidden // 2, 1)
        self.short_residual = nn.Linear(width, 1)
        self.long_residual = nn.Linear(width, 1)
        nn.init.zeros_(self.short_residual.weight)
        nn.init.zeros_(self.short_residual.bias)
        nn.init.zeros_(self.long_residual.weight)
        nn.init.zeros_(self.long_residual.bias)

    def forward(self, x, duration):
        emb = self.embedding(x).reshape(x.shape[0], -1)
        x0 = self.input_dropout(emb)
        crossed = x0
        for weight, bias in zip(self.cross_w, self.cross_b):
            crossed = x0 * torch.sum(crossed * weight, dim=1, keepdim=True) + bias + crossed
        deep = self.mlp(x0)
        base = self.base_head(torch.cat([crossed, deep], dim=1)).squeeze(1)
        short = self.short_residual(x0).squeeze(1)
        long = self.long_residual(x0).squeeze(1)
        gate = (duration <= 18000.0).to(base.dtype)
        return base + gate * short + (1.0 - gate) * long


def evaluate_scores(fast, users, labels, scores):
    if fast:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    result = evaluate(users, labels, scores)
    return {
        'gauc': float(result.get('GAUC', result.get('gauc'))),
        'ndcg5': float(result.get('nDCG@5', result.get('ndcg5'))),
        'primary': float(result.get('primary'))
    }


def predict(model, x, duration, device, batch_size):
    model.eval()
    outputs = np.empty(len(x), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            xb = torch.as_tensor(x[start:end], dtype=torch.long, device=device)
            db = torch.as_tensor(duration[start:end], dtype=torch.float32, device=device)
            outputs[start:end] = torch.sigmoid(model(xb, db)).cpu().numpy()
    return outputs


def train_model(data, config, epochs, seed, device):
    seed_all(seed)
    model = DCNLite(
        data['field_dims'], embed_dim=16, hidden=128, cross_layers=2,
        dropout=config['dropout']
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=config['lr_gamma'])
    x = data['x_train']
    y = data['y_train']
    duration = data['duration_train']
    sample_weight = recency_weights(data['date_train'])
    pair_pos, pair_neg = build_pairs(data['user_train'], y, seed)
    rng = np.random.default_rng(seed)
    batch_size = 8192 if device.type == 'cuda' else 4096
    pair_batch = max(512, batch_size // 2)
    best_gauc = -1.0
    best_state = None
    best_metrics = None
    best_epoch = 0
    stale = 0

    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(x))
        pair_order = rng.permutation(len(pair_pos))
        pair_cursor = 0
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            if pair_cursor + pair_batch > len(pair_order):
                pair_order = rng.permutation(len(pair_pos))
                pair_cursor = 0
            selected = pair_order[pair_cursor:pair_cursor + min(pair_batch, len(pair_order))]
            pair_cursor += len(selected)
            pi = pair_pos[selected]
            ni = pair_neg[selected]

            xb = torch.as_tensor(x[idx], dtype=torch.long, device=device)
            yb = torch.as_tensor(y[idx], dtype=torch.float32, device=device)
            db = torch.as_tensor(duration[idx], dtype=torch.float32, device=device)
            wb = torch.as_tensor(sample_weight[idx], dtype=torch.float32, device=device)
            pos_x = torch.as_tensor(x[pi], dtype=torch.long, device=device)
            neg_x = torch.as_tensor(x[ni], dtype=torch.long, device=device)
            pos_d = torch.as_tensor(duration[pi], dtype=torch.float32, device=device)
            neg_d = torch.as_tensor(duration[ni], dtype=torch.float32, device=device)

            logits = model(xb, db)
            point_loss = (F.binary_cross_entropy_with_logits(logits, yb, reduction='none') * wb).mean()
            pos_logits = model(pos_x, pos_d)
            neg_logits = model(neg_x, neg_d)
            pair_loss = F.softplus(-(pos_logits - neg_logits)).mean()
            used_rows = torch.unique(torch.cat([xb.reshape(-1), pos_x.reshape(-1), neg_x.reshape(-1)]))
            row_l2 = model.embedding.weight[used_rows].pow(2).sum(dim=1).mean()
            residual_l2 = model.short_residual.weight.pow(2).mean() + model.long_residual.weight.pow(2).mean()
            loss = 0.5 * point_loss + 0.5 * pair_loss + config['row_l2'] * row_l2 + config['residual_l2'] * residual_l2
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        scheduler.step()
        val_scores = predict(model, data['x_val'], data['duration_val'], device, batch_size * 2)
        metrics = evaluate_scores(data['fast'], data['user_val'], data['y_val'], val_scores)
        if metrics['gauc'] > best_gauc + 1e-8:
            best_gauc = metrics['gauc']
            best_metrics = metrics
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 2:
            break

    model.load_state_dict(best_state)
    final_scores = predict(model, data['x_val'], data['duration_val'], device, batch_size * 2)
    final_metrics = evaluate_scores(data['fast'], data['user_val'], data['y_val'], final_scores)
    final_metrics['best_epoch'] = best_epoch
    return final_scores, final_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = load_data(args.data_dir)
    smoke = os.environ.get('SMOKE_EPOCHS')
    smoke_cap = max(1, int(smoke)) if smoke is not None else None
    probe_epochs = min(3, smoke_cap) if smoke_cap is not None else 3
    final_epochs = min(10, smoke_cap) if smoke_cap is not None else 10

    candidates = [
        {'name': 'balanced', 'dropout': 0.20, 'weight_decay': 3e-4, 'row_l2': 1e-6, 'residual_l2': 3e-4, 'lr': 1.0e-3, 'lr_gamma': 0.55},
        {'name': 'strong', 'dropout': 0.30, 'weight_decay': 1e-3, 'row_l2': 3e-6, 'residual_l2': 1e-3, 'lr': 1.0e-3, 'lr_gamma': 0.50},
        {'name': 'low_lr', 'dropout': 0.25, 'weight_decay': 5e-4, 'row_l2': 2e-6, 'residual_l2': 5e-4, 'lr': 6.0e-4, 'lr_gamma': 0.65}
    ]

    history = []
    progress_path = out_dir / 'progress.log'
    best_config = None
    best_primary = -1.0
    for i, config in enumerate(candidates):
        _, metrics = train_model(data, config, probe_epochs, args.seed + i * 101, device)
        record = {'config': config, 'epochs': probe_epochs, 'gauc': metrics['gauc'], 'ndcg5': metrics['ndcg5'], 'primary': metrics['primary'], 'best_epoch': metrics['best_epoch']}
        history.append(record)
        with open(progress_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, sort_keys=True) + '\n')
        if metrics['primary'] > best_primary:
            best_primary = metrics['primary']
            best_config = config

    scores, metrics = train_model(data, best_config, final_epochs, args.seed + 1009, device)
    with open(out_dir / 'predictions.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['row_id', 'user_id', 'video_id', 'score'])
        for i, (user, video, score) in enumerate(zip(data['user_val'], data['video_val'], scores)):
            writer.writerow([i, int(user), int(video), format(float(score), '.9g')])

    output_metrics = {
        'gauc': metrics['gauc'],
        'ndcg5': metrics['ndcg5'],
        'primary': metrics['primary'],
        'history': history,
        'selected_config': best_config,
        'final_best_epoch': metrics['best_epoch']
    }
    with open(out_dir / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(output_metrics, f, sort_keys=True)


if __name__ == '__main__':
    main()
