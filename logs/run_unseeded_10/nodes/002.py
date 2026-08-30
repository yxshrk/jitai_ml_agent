import argparse
import csv
import json
import os
import random
from datetime import date as Date

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def date_ordinals(values):
    arr = np.asarray(values)
    result = np.zeros(len(arr), dtype=np.float32)
    cache = {}
    for i, value in enumerate(arr):
        text = str(int(value)) if isinstance(value, (int, np.integer, float, np.floating)) else str(value)
        text = text.replace('-', '')[:8]
        if text not in cache:
            try:
                cache[text] = Date(int(text[:4]), int(text[4:6]), int(text[6:8])).toordinal()
            except Exception:
                cache[text] = 0
        result[i] = cache[text]
    return result


def recency_weights(dates):
    ords = date_ordinals(dates)
    valid = ords > 0
    if not np.any(valid):
        return np.ones(len(ords), dtype=np.float32)
    newest = float(np.max(ords[valid]))
    ages = np.maximum(0.0, newest - ords)
    weights = np.exp(-np.log(2.0) * ages / 7.0)
    return weights.astype(np.float32)


def load_npz(data_dir):
    train_file = np.load(os.path.join(data_dir, 'train.npz'), allow_pickle=False)
    val_file = np.load(os.path.join(data_dir, 'val.npz'), allow_pickle=False)
    train_x = np.asarray(train_file['X'], dtype=np.int64)
    val_x = np.asarray(val_file['X'], dtype=np.int64)
    train_y = np.asarray(train_file['y'], dtype=np.float32)
    val_y = np.asarray(val_file['y'], dtype=np.float32)
    train_user = np.asarray(train_file['user'])
    val_user = np.asarray(val_file['user'])
    field_dims = np.asarray(train_file['field_dims'], dtype=np.int64)
    train_play = np.asarray(train_file['play_time_ms'], dtype=np.float32)
    train_duration = np.asarray(train_file['duration_ms'], dtype=np.float32)
    train_dates = np.asarray(train_file['date'])
    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1]))).astype(np.int64)
    if train_x.size and np.max(train_x[:, -1]) >= field_dims[-1]:
        train_x = train_x - offsets[None, :]
        val_x = val_x - offsets[None, :]
    train_x = np.maximum(train_x, 0)
    val_x = np.maximum(val_x, 0)
    for j, dim in enumerate(field_dims):
        train_x[:, j] = np.minimum(train_x[:, j], int(dim) - 1)
        val_x[:, j] = np.minimum(val_x[:, j], int(dim) - 1)
    video_ids = val_x[:, 1].copy()
    return {
        'train_x': train_x,
        'train_y': train_y,
        'val_x': val_x,
        'val_y': val_y,
        'train_user': train_user,
        'val_user': val_user,
        'video_ids': video_ids,
        'field_dims': field_dims,
        'train_play': train_play,
        'train_duration': train_duration,
        'train_dates': train_dates,
        'fast': True,
    }


def read_csv_rows(path, training):
    rows = []
    with open(path, 'r', newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            item = {
                'user_id': row['user_id'],
                'video_id': row['video_id'],
                'author_id': row.get('author_id', row['video_id']),
                'tab': row['tab'],
                'duration_ms': float(row['duration_ms']),
                'date': row['date'],
                'long_view': float(row['long_view']),
            }
            if training:
                item['play_time_ms'] = float(row['play_time_ms'])
            rows.append(item)
    return rows


def make_mapping(values):
    mapping = {}
    for value in values:
        if value not in mapping:
            mapping[value] = len(mapping) + 1
    return mapping


def load_csv(data_dir):
    train_rows = read_csv_rows(os.path.join(data_dir, 'train.csv'), True)
    val_rows = read_csv_rows(os.path.join(data_dir, 'val.csv'), False)
    fields = ['user_id', 'video_id', 'author_id', 'tab']
    mappings = {field: make_mapping([row[field] for row in train_rows]) for field in fields}
    train_durations = np.asarray([row['duration_ms'] for row in train_rows], dtype=np.float64)
    if len(train_durations):
        edges = np.unique(np.quantile(train_durations, np.linspace(0.1, 0.9, 9)))
    else:
        edges = np.asarray([], dtype=np.float64)

    def encode(rows):
        x = np.zeros((len(rows), 5), dtype=np.int64)
        for i, row in enumerate(rows):
            for j, field in enumerate(fields):
                x[i, j] = mappings[field].get(row[field], 0)
            x[i, 4] = int(np.searchsorted(edges, row['duration_ms'], side='right'))
        return x

    train_x = encode(train_rows)
    val_x = encode(val_rows)
    field_dims = np.asarray([len(mappings[field]) + 1 for field in fields] + [len(edges) + 1], dtype=np.int64)
    return {
        'train_x': train_x,
        'train_y': np.asarray([row['long_view'] for row in train_rows], dtype=np.float32),
        'val_x': val_x,
        'val_y': np.asarray([row['long_view'] for row in val_rows], dtype=np.float32),
        'train_user': np.asarray([row['user_id'] for row in train_rows]),
        'val_user': np.asarray([row['user_id'] for row in val_rows]),
        'video_ids': np.asarray([row['video_id'] for row in val_rows]),
        'field_dims': field_dims,
        'train_play': np.asarray([row['play_time_ms'] for row in train_rows], dtype=np.float32),
        'train_duration': np.asarray([row['duration_ms'] for row in train_rows], dtype=np.float32),
        'train_dates': np.asarray([row['date'] for row in train_rows]),
        'fast': False,
    }


def build_pairs(users, labels, seed):
    rng = np.random.RandomState(seed)
    order = np.argsort(users, kind='mergesort')
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    positives = []
    negatives = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        group = order[left:right]
        pos = group[labels[group] > 0.5]
        neg = group[labels[group] <= 0.5]
        if len(pos) == 0 or len(neg) == 0:
            continue
        count = max(len(pos), len(neg))
        positives.append(rng.choice(pos, size=count, replace=len(pos) < count))
        negatives.append(rng.choice(neg, size=count, replace=len(neg) < count))
    if not positives:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(positives), np.concatenate(negatives)


class DCNCensoredModel(nn.Module):
    def __init__(self, field_dims, embed_dim=16, dropout=0.30):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(int(dim), embed_dim) for dim in field_dims])
        width = len(field_dims) * embed_dim
        self.embed_dropout = nn.Dropout(dropout)
        self.cross_w = nn.ParameterList([nn.Parameter(torch.empty(width)) for _ in range(2)])
        self.cross_b = nn.ParameterList([nn.Parameter(torch.zeros(width)) for _ in range(2)])
        self.mlp = nn.Sequential(
            nn.Linear(width, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.long_head = nn.Linear(width + 64, 1)
        self.watch_head = nn.Linear(width + 64, 1)
        self.reset_parameters()

    def reset_parameters(self):
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight, std=0.01)
        for weight in self.cross_w:
            nn.init.normal_(weight, std=0.01)
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.long_head.weight)
        nn.init.zeros_(self.long_head.bias)
        nn.init.xavier_uniform_(self.watch_head.weight)
        nn.init.zeros_(self.watch_head.bias)

    def forward(self, x):
        x0 = torch.cat([embedding(x[:, i]) for i, embedding in enumerate(self.embeddings)], dim=1)
        x0 = self.embed_dropout(x0)
        cross = x0
        for weight, bias in zip(self.cross_w, self.cross_b):
            scalar = torch.sum(cross * weight, dim=1, keepdim=True)
            cross = x0 * scalar + bias + cross
        deep = self.mlp(x0)
        representation = torch.cat([cross, deep], dim=1)
        return self.long_head(representation).squeeze(1), self.watch_head(representation).squeeze(1)


def predict(model, x, device, batch_size=8192):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.as_tensor(x[start:start + batch_size], dtype=torch.long, device=device)
            logits, _ = model(xb)
            outputs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(outputs).astype(np.float64)


def evaluate_scores(fast, users, labels, scores):
    if fast:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    return evaluate(users, labels, scores)


def metric_value(metrics, upper, lower):
    if upper in metrics:
        return float(metrics[upper])
    return float(metrics[lower])


def train_model(data, seed, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DCNCensoredModel(data['field_dims']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.55)
    x = data['train_x']
    y = data['train_y']
    sample_weights = recency_weights(data['train_dates'])
    play = np.maximum(data['train_play'], 0.0)
    duration = np.maximum(data['train_duration'], 1.0)
    observed = np.minimum(play, duration)
    watch_target = (np.log1p(observed) / 10.0).astype(np.float32)
    censored = (play >= duration).astype(np.float32)
    pair_pos, pair_neg = build_pairs(data['train_user'], y, seed)
    rng = np.random.RandomState(seed)
    batch_size = 4096
    pair_batch_size = 2048
    best_primary = -1.0
    best_state = None
    stale = 0

    for epoch in range(epochs):
        model.train()
        permutation = rng.permutation(len(x))
        if len(pair_pos):
            pair_perm = rng.permutation(len(pair_pos))
        else:
            pair_perm = np.empty(0, dtype=np.int64)
        pair_cursor = 0
        for start in range(0, len(x), batch_size):
            indices = permutation[start:start + batch_size]
            xb = torch.as_tensor(x[indices], dtype=torch.long, device=device)
            yb = torch.as_tensor(y[indices], dtype=torch.float32, device=device)
            wb = torch.as_tensor(sample_weights[indices], dtype=torch.float32, device=device)
            target_b = torch.as_tensor(watch_target[indices], dtype=torch.float32, device=device)
            censored_b = torch.as_tensor(censored[indices], dtype=torch.float32, device=device)
            logits, watch_pred = model(xb)
            bce_each = F.binary_cross_entropy_with_logits(logits, yb, reduction='none')
            bce_loss = torch.sum(bce_each * wb) / torch.clamp(torch.sum(wb), min=1.0)
            exact_error = (watch_pred - target_b).pow(2)
            lower_bound_error = F.relu(target_b - watch_pred).pow(2)
            watch_each = (1.0 - censored_b) * exact_error + censored_b * lower_bound_error
            watch_loss = torch.sum(watch_each * wb) / torch.clamp(torch.sum(wb), min=1.0)

            if len(pair_pos):
                if pair_cursor + pair_batch_size > len(pair_perm):
                    pair_perm = rng.permutation(len(pair_pos))
                    pair_cursor = 0
                selected = pair_perm[pair_cursor:pair_cursor + pair_batch_size]
                pair_cursor += pair_batch_size
                pos_idx = pair_pos[selected]
                neg_idx = pair_neg[selected]
                pos_x = torch.as_tensor(x[pos_idx], dtype=torch.long, device=device)
                neg_x = torch.as_tensor(x[neg_idx], dtype=torch.long, device=device)
                pos_logits, _ = model(pos_x)
                neg_logits, _ = model(neg_x)
                pair_w = torch.as_tensor(sample_weights[pos_idx], dtype=torch.float32, device=device)
                bpr_each = F.softplus(-(pos_logits - neg_logits))
                bpr_loss = torch.sum(bpr_each * pair_w) / torch.clamp(torch.sum(pair_w), min=1.0)
            else:
                bpr_loss = torch.zeros((), device=device)

            loss = 0.5 * bce_loss + 0.5 * bpr_loss + 0.3 * watch_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        scores = predict(model, data['val_x'], device)
        metrics = evaluate_scores(data['fast'], data['val_user'], data['val_y'], scores)
        primary = metric_value(metrics, 'primary', 'primary')
        if primary > best_primary + 1e-12:
            best_primary = primary
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        scheduler.step()
        if stale >= 2:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, device


def write_outputs(out_dir, data, scores, metrics):
    os.makedirs(out_dir, exist_ok=True)
    prediction_path = os.path.join(out_dir, 'predictions.csv')
    with open(prediction_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['row_id', 'user_id', 'video_id', 'score'])
        for i, (user_id, video_id, score) in enumerate(zip(data['val_user'], data['video_ids'], scores)):
            if isinstance(user_id, np.generic):
                user_id = user_id.item()
            if isinstance(video_id, np.generic):
                video_id = video_id.item()
            writer.writerow([i, user_id, video_id, format(float(score), '.12g')])
    normalized = {
        'gauc': metric_value(metrics, 'GAUC', 'gauc'),
        'ndcg5': metric_value(metrics, 'nDCG@5', 'ndcg5'),
        'primary': metric_value(metrics, 'primary', 'primary'),
    }
    with open(os.path.join(out_dir, 'metrics.json'), 'w', encoding='utf-8') as handle:
        json.dump(normalized, handle, sort_keys=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    seed_everything(args.seed)
    fast = os.path.exists(os.path.join(args.data_dir, 'train.npz')) and os.path.exists(os.path.join(args.data_dir, 'val.npz'))
    data = load_npz(args.data_dir) if fast else load_csv(args.data_dir)
    epochs = 8
    smoke = os.environ.get('SMOKE_EPOCHS')
    if smoke is not None:
        epochs = min(epochs, max(1, int(smoke)))
    model, device = train_model(data, args.seed, epochs)
    scores = predict(model, data['val_x'], device)
    metrics = evaluate_scores(data['fast'], data['val_user'], data['val_y'], scores)
    write_outputs(args.out_dir, data, scores, metrics)


if __name__ == '__main__':
    main()
