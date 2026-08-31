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


def ordinal_date(values):
    out = []
    for value in values:
        s = str(int(value))
        if len(s) >= 8:
            year, month, day = int(s[:4]), int(s[4:6]), int(s[6:8])
        else:
            year, month, day = 2022, 1, int(value) % 100
        out.append(year * 372 + month * 31 + day)
    return np.asarray(out, dtype=np.float32)


def load_data(data_dir):
    if (data_dir / 'train.npz').exists() and (data_dir / 'val.npz').exists():
        with np.load(data_dir / 'train.npz', allow_pickle=False) as z:
            xt = np.asarray(z['X'], np.int64)
            yt = np.asarray(z['y'], np.float32).reshape(-1)
            ut = np.asarray(z['user']).reshape(-1)
            dates = np.asarray(z['date']).reshape(-1)
            dims = np.asarray(z['field_dims'], np.int64)
        with np.load(data_dir / 'val.npz', allow_pickle=False) as z:
            xv = np.asarray(z['X'], np.int64)
            yv = np.asarray(z['y'], np.float32).reshape(-1)
            uv = np.asarray(z['user']).reshape(-1)
        videos = xv[:, 1] - int(dims[0])
        total = max(int(dims.sum()), int(max(xt.max(), xv.max())) + 1)
        return xt, yt, ut, dates, xv, yv, uv, videos, total, True

    def read_csv(path):
        rows, labels, users, videos, dates = [], [], [], [], []
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
                dates.append(int(float(r.get('date', '0') or 0)))
        return rows, np.asarray(labels, np.float32), np.asarray(users), np.asarray(videos), np.asarray(dates)

    tr, yt, ut, _, dates = read_csv(data_dir / 'train.csv')
    va, yv, uv, videos, _ = read_csv(data_dir / 'val.csv')
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
    return np.column_stack(train_cols).astype(np.int64), yt, ut, dates, np.column_stack(val_cols).astype(np.int64), yv, uv, videos, offset, False


class BilinearFM(nn.Module):
    def __init__(self, total, fields, bias):
        super().__init__()
        k = 16
        self.emb = nn.Embedding(total, k)
        self.linear = nn.Embedding(total, 1)
        self.transforms = nn.Parameter(torch.stack([torch.eye(k) for _ in range(fields)]))
        self.bias = nn.Parameter(torch.tensor(float(bias)))
        nn.init.normal_(self.emb.weight, std=0.01)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x):
        e = F.dropout(self.emb(x), p=0.12, training=self.training)
        transformed = torch.einsum('bfk,fkl->bfl', e, self.transforms)
        interaction = torch.zeros(x.shape[0], device=x.device)
        for i in range(e.shape[1]):
            for j in range(i + 1, e.shape[1]):
                interaction = interaction + (transformed[:, i] * e[:, j]).sum(1)
        return self.linear(x).sum(1).squeeze(1) + interaction + self.bias


def predict(model, x, device):
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(x), 131072):
            xb = torch.as_tensor(x[start:start + 131072], device=device)
            parts.append(model(xb).cpu().numpy())
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
    xt, yt, ut, dates, xv, yv, uv, videos, total, fast = load_data(Path(args.data_dir))
    if fast:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    day = ordinal_date(dates)
    weights = np.exp2(-(day.max() - day) / 7.0).astype(np.float32)
    weights /= max(float(weights.mean()), 1e-8)
    rate = float(np.clip(np.average(yt, weights=weights), 1e-5, 1 - 1e-5))
    model = BilinearFM(total, xt.shape[1], math.log(rate / (1 - rate))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0025, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 6, 9, 12], gamma=0.4)
    epochs = 14
    if os.getenv('SMOKE_EPOCHS') is not None:
        epochs = min(epochs, max(1, int(os.environ['SMOKE_EPOCHS'])))
    rng = np.random.default_rng(args.seed + 911)
    best_scores, best_metrics, best_epoch, history = None, None, 0, []
    for epoch in range(epochs):
        order = rng.permutation(len(xt))
        model.train()
        for start in range(0, len(order), 49152):
            idx = order[start:start + 49152]
            xb = torch.as_tensor(xt[idx], device=device)
            yb = torch.as_tensor(yt[idx], device=device)
            wb = torch.as_tensor(weights[idx], device=device)
            optimizer.zero_grad(set_to_none=True)
            losses = F.binary_cross_entropy_with_logits(model(xb), yb, reduction='none')
            loss = (losses * wb).sum() / wb.sum().clamp_min(1e-8)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        scheduler.step()
        scores = predict(model, xv, device)
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
        json.dump({**best_metrics, 'history': history, 'best_epoch': best_epoch, 'family': 'recency-weighted-bilinear-fm', 'seed': args.seed, 'half_life_days': 7}, f, sort_keys=True)


if __name__ == '__main__':
    main()
