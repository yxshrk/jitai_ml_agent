import argparse
import contextlib
import csv
import io
import json
import os
import random
from copy import deepcopy
from datetime import datetime

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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def date_ord(value):
    text = str(value).strip()
    if text.endswith('.0'):
        text = text[:-2]
    try:
        return datetime.strptime(text, '%Y%m%d').toordinal()
    except Exception:
        try:
            return int(float(text))
        except Exception:
            return 0


def load_npz(data_dir):
    train_path = os.path.join(data_dir, 'train.npz')
    val_path = os.path.join(data_dir, 'val.npz')
    with np.load(train_path, allow_pickle=False) as z:
        x_train = z['X'].astype(np.int64, copy=False)
        y_train = z['y'].astype(np.float32, copy=False)
        users_train = z['user']
        play = z['play_time_ms'].astype(np.float32, copy=False)
        duration = z['duration_ms'].astype(np.float32, copy=False)
        dates = z['date'] if 'date' in z.files else np.zeros(len(y_train), dtype=np.int64)
        field_dims = z['field_dims'].astype(np.int64, copy=False)
    with np.load(val_path, allow_pickle=False) as z:
        x_val = z['X'].astype(np.int64, copy=False)
        y_val = z['y'].astype(np.float32, copy=False)
        users_val = z['user']
        if 'video' in z.files:
            videos_val = z['video']
        elif 'video_id' in z.files:
            videos_val = z['video_id']
        else:
            video_offset = int(field_dims[0])
            videos_val = x_val[:, 1].astype(np.int64) - video_offset
    effective_duration = np.maximum(np.minimum(duration, 18000.0), 1.0)
    ratio = play / effective_duration
    thresholds = np.asarray([0.10, 0.25, 0.50, 0.75, 1.00], dtype=np.float32)
    ordinal = (ratio[:, None] >= thresholds[None, :]).astype(np.float32)
    ord_dates = np.asarray([date_ord(v) for v in dates], dtype=np.int64)
    if len(ord_dates) and np.max(ord_dates) > 0:
        age = np.maximum(np.max(ord_dates) - ord_dates, 0)
        weights = np.power(0.5, age.astype(np.float32) / 7.0).astype(np.float32)
        weights /= max(float(np.mean(weights)), 1e-6)
    else:
        weights = np.ones(len(y_train), dtype=np.float32)
    return {
        'x_train': x_train,
        'y_train': y_train,
        'users_train': users_train,
        'ordinal': ordinal,
        'weights': weights,
        'x_val': x_val,
        'y_val': y_val,
        'users_val': users_val,
        'videos_val': videos_val,
        'field_dims': field_dims,
        'npz': True,
    }


def read_csv_rows(path, training):
    rows = []
    with open(path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = {
                'user_id': row.get('user_id', ''),
                'video_id': row.get('video_id', ''),
                'author_id': row.get('author_id', ''),
                'tab': row.get('tab', ''),
                'duration_ms': float(row.get('duration_ms', 0) or 0),
                'long_view': float(row.get('long_view', 0) or 0),
            }
            if training:
                item['play_time_ms'] = float(row.get('play_time_ms', 0) or 0)
                item['date'] = row.get('date', '0')
            rows.append(item)
    return rows


def load_csv(data_dir):
    train_rows = read_csv_rows(os.path.join(data_dir, 'train.csv'), True)
    val_rows = read_csv_rows(os.path.join(data_dir, 'val.csv'), False)
    train_duration = np.asarray([r['duration_ms'] for r in train_rows], dtype=np.float64)
    if len(train_duration):
        cuts = np.unique(np.quantile(train_duration, np.linspace(0.1, 0.9, 9)))
    else:
        cuts = np.asarray([], dtype=np.float64)

    def raw_fields(row):
        author = row['author_id'] if row['author_id'] != '' else '__missing_author__'
        dur_bucket = str(int(np.searchsorted(cuts, row['duration_ms'], side='right')))
        return [row['user_id'], row['video_id'], author, row['tab'], dur_bucket]

    maps = []
    for field in range(5):
        values = sorted({raw_fields(r)[field] for r in train_rows})
        maps.append({v: i + 1 for i, v in enumerate(values)})
    field_dims = np.asarray([len(m) + 1 for m in maps], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1])).astype(np.int64)

    def encode(rows):
        x = np.zeros((len(rows), 5), dtype=np.int64)
        for i, row in enumerate(rows):
            fields = raw_fields(row)
            for j in range(5):
                x[i, j] = offsets[j] + maps[j].get(fields[j], 0)
        return x

    x_train = encode(train_rows)
    x_val = encode(val_rows)
    y_train = np.asarray([r['long_view'] for r in train_rows], dtype=np.float32)
    y_val = np.asarray([r['long_view'] for r in val_rows], dtype=np.float32)
    users_train = np.asarray([r['user_id'] for r in train_rows])
    users_val = np.asarray([r['user_id'] for r in val_rows])
    videos_val = np.asarray([r['video_id'] for r in val_rows])
    play = np.asarray([r['play_time_ms'] for r in train_rows], dtype=np.float32)
    duration = np.asarray([r['duration_ms'] for r in train_rows], dtype=np.float32)
    effective_duration = np.maximum(np.minimum(duration, 18000.0), 1.0)
    ratio = play / effective_duration
    thresholds = np.asarray([0.10, 0.25, 0.50, 0.75, 1.00], dtype=np.float32)
    ordinal = (ratio[:, None] >= thresholds[None, :]).astype(np.float32)
    ord_dates = np.asarray([date_ord(r['date']) for r in train_rows], dtype=np.int64)
    if len(ord_dates) and np.max(ord_dates) > 0:
        age = np.maximum(np.max(ord_dates) - ord_dates, 0)
        weights = np.power(0.5, age.astype(np.float32) / 7.0).astype(np.float32)
        weights /= max(float(np.mean(weights)), 1e-6)
    else:
        weights = np.ones(len(train_rows), dtype=np.float32)
    return {
        'x_train': x_train,
        'y_train': y_train,
        'users_train': users_train,
        'ordinal': ordinal,
        'weights': weights,
        'x_val': x_val,
        'y_val': y_val,
        'users_val': users_val,
        'videos_val': videos_val,
        'field_dims': field_dims,
        'npz': False,
    }


def make_pairs(users, labels, rng):
    order = np.argsort(users, kind='mergesort')
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    pos_parts = []
    neg_parts = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        group = order[a:b]
        positives = group[labels[group] > 0.5]
        negatives = group[labels[group] <= 0.5]
        if len(positives) and len(negatives):
            chosen = negatives[rng.integers(0, len(negatives), size=len(positives))]
            pos_parts.append(positives)
            neg_parts.append(chosen)
    if not pos_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(pos_parts), np.concatenate(neg_parts)


class DCNLite(nn.Module):
    def __init__(self, field_dims, embed_dim=16, hidden=128, dropout=0.30):
        super().__init__()
        total = int(np.sum(field_dims))
        width = len(field_dims) * embed_dim
        self.embedding = nn.Embedding(total, embed_dim)
        self.cross_w = nn.Parameter(torch.empty(width))
        self.cross_b = nn.Parameter(torch.zeros(width))
        self.mlp = nn.Sequential(
            nn.Linear(width, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.embedding_dropout = nn.Dropout(dropout)
        self.main_head = nn.Linear(width + hidden // 2, 1)
        self.ordinal_head = nn.Linear(width + hidden // 2, 5)
        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.normal_(self.cross_w, std=0.01)

    def forward(self, x):
        base = self.embedding_dropout(self.embedding(x).flatten(1))
        cross = base + base * torch.sum(base * self.cross_w, dim=1, keepdim=True) + self.cross_b
        deep = self.mlp(base)
        representation = torch.cat([cross, deep], dim=1)
        return self.main_head(representation).squeeze(1), self.ordinal_head(representation)


def metric_values(data, scores):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        if data['npz']:
            from data.official.evaluate import evaluate
        else:
            from harness.evaluate_provisional import evaluate
        result = evaluate(data['users_val'], data['y_val'], scores)
    gauc = float(result.get('GAUC', result.get('gauc')))
    ndcg = float(result.get('nDCG@5', result.get('ndcg5')))
    primary = float(result.get('primary', (gauc + ndcg) / 2.0))
    return {'gauc': gauc, 'ndcg5': ndcg, 'primary': primary}


def predict(model, x, device, batch_size=16384):
    model.eval()
    pieces = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.as_tensor(x[start:start + batch_size], dtype=torch.long, device=device)
            logits, _ = model(xb)
            pieces.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(pieces).astype(np.float64) if pieces else np.empty(0, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    seed_everything(args.seed)

    if os.path.exists(os.path.join(args.data_dir, 'train.npz')) and os.path.exists(os.path.join(args.data_dir, 'val.npz')):
        data = load_npz(args.data_dir)
    else:
        data = load_csv(args.data_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DCNLite(data['field_dims']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    rng = np.random.default_rng(args.seed)
    pair_pos, pair_neg = make_pairs(data['users_train'], data['y_train'], rng)

    epochs = 8
    smoke = os.environ.get('SMOKE_EPOCHS')
    if smoke is not None:
        epochs = min(epochs, max(1, int(smoke)))
    batch_size = 8192
    best_gauc = -float('inf')
    best_state = deepcopy(model.state_dict())
    stale = 0

    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(data['y_train']))
        pair_order = rng.permutation(len(pair_pos)) if len(pair_pos) else pair_pos
        for step, start in enumerate(range(0, len(order), batch_size)):
            idx = order[start:start + batch_size]
            xb = torch.as_tensor(data['x_train'][idx], dtype=torch.long, device=device)
            yb = torch.as_tensor(data['y_train'][idx], dtype=torch.float32, device=device)
            ob = torch.as_tensor(data['ordinal'][idx], dtype=torch.float32, device=device)
            wb = torch.as_tensor(data['weights'][idx], dtype=torch.float32, device=device)
            logits, ordinal_logits = model(xb)
            point_loss = (F.binary_cross_entropy_with_logits(logits, yb, reduction='none') * wb).mean()
            ordinal_loss = (F.binary_cross_entropy_with_logits(ordinal_logits, ob, reduction='none').mean(dim=1) * wb).mean()
            if len(pair_pos):
                pstart = (step * batch_size) % len(pair_pos)
                psel = pair_order[pstart:min(pstart + len(idx), len(pair_pos))]
                if len(psel) < len(idx):
                    extra = pair_order[:len(idx) - len(psel)]
                    psel = np.concatenate([psel, extra])
                xp = torch.as_tensor(data['x_train'][pair_pos[psel]], dtype=torch.long, device=device)
                xn = torch.as_tensor(data['x_train'][pair_neg[psel]], dtype=torch.long, device=device)
                pos_logits, _ = model(xp)
                neg_logits, _ = model(xn)
                pair_loss = F.softplus(-(pos_logits - neg_logits)).mean()
            else:
                pair_loss = point_loss.new_zeros(())
            loss = 0.5 * point_loss + 0.5 * pair_loss + 0.3 * ordinal_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        scheduler.step()
        scores = predict(model, data['x_val'], device)
        current = metric_values(data, scores)
        if current['gauc'] > best_gauc + 1e-7:
            best_gauc = current['gauc']
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= 2 and smoke is None:
            break

    model.load_state_dict(best_state)
    scores = predict(model, data['x_val'], device)
    metrics = metric_values(data, scores)

    prediction_path = os.path.join(args.out_dir, 'predictions.csv')
    with open(prediction_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['row_id', 'user_id', 'video_id', 'score'])
        for i, (user, video, score) in enumerate(zip(data['users_val'], data['videos_val'], scores)):
            writer.writerow([i, user.item() if isinstance(user, np.generic) else user, video.item() if isinstance(video, np.generic) else video, float(score)])

    with open(os.path.join(args.out_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, separators=(',', ':'))


if __name__ == '__main__':
    main()
