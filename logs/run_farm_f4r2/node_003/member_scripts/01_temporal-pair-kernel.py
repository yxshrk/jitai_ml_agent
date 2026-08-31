import argparse
import csv
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def clean_metrics(r):
    return {'gauc': float(r['GAUC']), 'ndcg5': float(r['nDCG@5']), 'primary': float(r['primary'])}


def scalar(x):
    return x.item() if isinstance(x, np.generic) else x


def make_context(hourmin, date):
    h = np.asarray(hourmin, dtype=np.int64)
    h = np.where(h >= 100, h // 100, h) % 24
    d = np.asarray(date, dtype=np.int64)
    day = (d % 100 + 2 * ((d // 100) % 100) + 3) % 7
    return np.column_stack([h, day]).astype(np.int64)


def load_data(data_dir):
    if (data_dir / 'train.npz').exists() and (data_dir / 'val.npz').exists():
        with np.load(data_dir / 'train.npz', allow_pickle=False) as z:
            xt = np.asarray(z['X'], np.int64)
            yt = np.asarray(z['y'], np.float32).reshape(-1)
            ut = np.asarray(z['user']).reshape(-1)
            ct = make_context(z['hourmin'], z['date'])
            dims = np.asarray(z['field_dims'], np.int64)
        with np.load(data_dir / 'val.npz', allow_pickle=False) as z:
            xv = np.asarray(z['X'], np.int64)
            yv = np.asarray(z['y'], np.float32).reshape(-1)
            uv = np.asarray(z['user']).reshape(-1)
            cv = make_context(z['hourmin'], z['date'])
        videos = xv[:, 1] - int(dims[0])
        total = max(int(dims.sum()), int(max(xt.max(), xv.max())) + 1)
        return xt, yt, ut, ct, xv, yv, uv, cv, videos, total, True

    def read_csv(path):
        rows, labels, users, videos, hours, dates = [], [], [], [], [], []
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            has_author = 'author_id' in (reader.fieldnames or [])
            for r in reader:
                duration = float(r.get('duration_ms', '0') or 0)
                author = r['author_id'] if has_author else '__missing_author__'
                rows.append((r['user_id'], r['video_id'], author, r.get('tab', '0'), duration))
                labels.append(float(r['long_view']))
                users.append(r['user_id'])
                videos.append(r['video_id'])
                hours.append(int(float(r.get('hourmin', '0') or 0)))
                dates.append(int(float(r.get('date', '0') or 0)))
        return rows, np.asarray(labels, np.float32), np.asarray(users), np.asarray(videos), make_context(hours, dates)

    tr, yt, ut, _, ct = read_csv(data_dir / 'train.csv')
    va, yv, uv, videos, cv = read_csv(data_dir / 'val.csv')
    cuts = np.quantile([r[4] for r in tr], np.linspace(0.1, 0.9, 9))
    train_cols, val_cols, offset = [], [], 0
    for j in range(4):
        mapping = {}
        for r in tr:
            if r[j] not in mapping:
                mapping[r[j]] = len(mapping) + 1
        dim = len(mapping) + 1
        train_cols.append(np.asarray([mapping.get(r[j], 0) + offset for r in tr]))
        val_cols.append(np.asarray([mapping.get(r[j], 0) + offset for r in va]))
        offset += dim
    train_cols.append(np.searchsorted(cuts, [r[4] for r in tr]) + offset)
    val_cols.append(np.searchsorted(cuts, [r[4] for r in va]) + offset)
    offset += 10
    return np.column_stack(train_cols).astype(np.int64), yt, ut, ct, np.column_stack(val_cols).astype(np.int64), yv, uv, cv, videos, offset, False


class TemporalFM(nn.Module):
    def __init__(self, total, bias):
        super().__init__()
        k = 16
        self.emb = nn.Embedding(total, k)
        self.linear = nn.Embedding(total, 1)
        self.hour = nn.Embedding(24, k)
        self.day = nn.Embedding(7, k)
        self.context_head = nn.Sequential(nn.Linear(2 * k, 32), nn.ReLU(), nn.Dropout(0.15), nn.Linear(32, 1))
        self.bias = nn.Parameter(torch.tensor(float(bias)))
        nn.init.normal_(self.emb.weight, std=0.01)
        nn.init.normal_(self.hour.weight, std=0.01)
        nn.init.normal_(self.day.weight, std=0.01)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x, context):
        e = F.dropout(self.emb(x), p=0.18, training=self.training)
        summed = e.sum(1)
        fm = 0.5 * (summed.square() - e.square().sum(1)).sum(1)
        ctx = self.hour(context[:, 0]) + self.day(context[:, 1])
        item = e[:, 1]
        author = e[:, 2]
        pair = (item * ctx).sum(1) + 0.5 * (author * self.hour(context[:, 0])).sum(1)
        return self.linear(x).sum(1).squeeze(1) + fm + pair + self.context_head(torch.cat([item, ctx], 1)).squeeze(1) + self.bias


def predict(model, x, context, device):
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(x), 131072):
            xb = torch.as_tensor(x[start:start + 131072], device=device)
            cb = torch.as_tensor(context[start:start + 131072], device=device)
            parts.append(model(xb, cb).cpu().numpy())
    return np.concatenate(parts).astype(np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    seed_all(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xt, yt, ut, ct, xv, yv, uv, cv, videos, total, fast = load_data(Path(args.data_dir))
    if fast:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    rate = float(np.clip(yt.mean(), 1e-5, 1 - 1e-5))
    model = TemporalFM(total, math.log(rate / (1 - rate))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0028, weight_decay=4e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4, 7, 10], gamma=0.35)
    epochs = 14
    if os.getenv('SMOKE_EPOCHS') is not None:
        epochs = min(epochs, max(1, int(os.environ['SMOKE_EPOCHS'])))
    rng = np.random.default_rng(args.seed + 71)
    best_scores, best_metrics, best_epoch, history = None, None, 0, []
    for epoch in range(epochs):
        order = rng.permutation(len(xt))
        model.train()
        for start in range(0, len(order), 49152):
            idx = order[start:start + 49152]
            xb = torch.as_tensor(xt[idx], device=device)
            cb = torch.as_tensor(ct[idx], device=device)
            yb = torch.as_tensor(yt[idx], device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(xb, cb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        scheduler.step()
        scores = predict(model, xv, cv, device)
        current = clean_metrics(evaluate(uv, yv, scores))
        history.append({'epoch': epoch + 1, **current})
        if best_metrics is None or current['primary'] > best_metrics['primary']:
            best_scores, best_metrics, best_epoch = scores.copy(), current, epoch + 1
    with open(out_dir / 'predictions.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['row_id', 'user_id', 'video_id', 'score'])
        for i, (u, v, s) in enumerate(zip(uv, videos, best_scores)):
            writer.writerow([i, scalar(u), scalar(v), format(float(s), '.12g')])
    with open(out_dir / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump({**best_metrics, 'history': history, 'best_epoch': best_epoch, 'family': 'temporal-pair-kernel', 'seed': args.seed}, f, sort_keys=True)


if __name__ == '__main__':
    main()
