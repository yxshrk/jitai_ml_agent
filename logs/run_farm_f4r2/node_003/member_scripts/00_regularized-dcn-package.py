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
    if isinstance(x, np.generic):
        x = x.item()
    return x


def load_data(data_dir):
    train_npz = data_dir / 'train.npz'
    val_npz = data_dir / 'val.npz'
    if train_npz.exists() and val_npz.exists():
        with np.load(train_npz, allow_pickle=False) as z:
            xt = np.asarray(z['X'], dtype=np.int64)
            yt = np.asarray(z['y'], dtype=np.float32).reshape(-1)
            ut = np.asarray(z['user']).reshape(-1)
            dims = np.asarray(z['field_dims'], dtype=np.int64)
        with np.load(val_npz, allow_pickle=False) as z:
            xv = np.asarray(z['X'], dtype=np.int64)
            yv = np.asarray(z['y'], dtype=np.float32).reshape(-1)
            uv = np.asarray(z['user']).reshape(-1)
        videos = xv[:, 1] - int(dims[0])
        total = max(int(dims.sum()), int(max(xt.max(), xv.max())) + 1)
        return xt, yt, ut, xv, yv, uv, videos, total, True

    def read_csv(path):
        rows, labels, users, videos = [], [], [], []
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
        return rows, np.asarray(labels, np.float32), np.asarray(users), np.asarray(videos)

    tr, yt, ut, _ = read_csv(data_dir / 'train.csv')
    va, yv, uv, videos = read_csv(data_dir / 'val.csv')
    cuts = np.quantile(np.asarray([r[4] for r in tr]), np.linspace(0.1, 0.9, 9))
    train_cols, val_cols, offset = [], [], 0
    for j in range(4):
        mapping = {}
        for row in tr:
            if row[j] not in mapping:
                mapping[row[j]] = len(mapping) + 1
        dim = len(mapping) + 1
        train_cols.append(np.asarray([mapping.get(r[j], 0) + offset for r in tr]))
        val_cols.append(np.asarray([mapping.get(r[j], 0) + offset for r in va]))
        offset += dim
    train_cols.append(np.searchsorted(cuts, [r[4] for r in tr]).astype(np.int64) + offset)
    val_cols.append(np.searchsorted(cuts, [r[4] for r in va]).astype(np.int64) + offset)
    offset += 10
    return np.column_stack(train_cols).astype(np.int64), yt, ut, np.column_stack(val_cols).astype(np.int64), yv, uv, videos, offset, False


class DCN(nn.Module):
    def __init__(self, total, fields, bias):
        super().__init__()
        k = 16
        width = fields * k
        self.emb = nn.Embedding(total, k)
        self.linear = nn.Embedding(total, 1)
        self.bias = nn.Parameter(torch.tensor(float(bias)))
        self.cross_w = nn.ParameterList([nn.Parameter(torch.empty(width)) for _ in range(2)])
        self.cross_b = nn.ParameterList([nn.Parameter(torch.zeros(width)) for _ in range(2)])
        self.cross_out = nn.Linear(width, 1, bias=False)
        self.deep = nn.Sequential(nn.Linear(width, 128), nn.ReLU(), nn.Dropout(0.30), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.20), nn.Linear(64, 1))
        nn.init.normal_(self.emb.weight, std=0.01)
        nn.init.zeros_(self.linear.weight)
        for p in self.cross_w:
            nn.init.normal_(p, std=0.01)

    def forward(self, x):
        x0 = F.dropout(self.emb(x), p=0.20, training=self.training).flatten(1)
        z = x0
        for w, b in zip(self.cross_w, self.cross_b):
            z = x0 * (z * w).sum(1, keepdim=True) + b + z
        return self.linear(x).sum(1).squeeze(1) + self.cross_out(z).squeeze(1) + self.deep(x0).squeeze(1) + self.bias


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
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xt, yt, ut, xv, yv, uv, videos, total, fast = load_data(data_dir)
    if fast:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    rate = float(np.clip(yt.mean(), 1e-5, 1 - 1e-5))
    model = DCN(total, xt.shape[1], math.log(rate / (1 - rate))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0025, weight_decay=8e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 6, 9, 12], gamma=0.4)
    epochs = 14
    if os.getenv('SMOKE_EPOCHS') is not None:
        epochs = min(epochs, max(1, int(os.environ['SMOKE_EPOCHS'])))
    rng = np.random.default_rng(args.seed + 17)
    best_scores, best_metrics, best_epoch, history = None, None, 0, []
    for epoch in range(epochs):
        order = rng.permutation(len(xt))
        model.train()
        for start in range(0, len(order), 32768):
            idx = order[start:start + 32768]
            xb = torch.as_tensor(xt[idx], device=device)
            yb = torch.as_tensor(yt[idx], device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            targets = yb * 0.985 + 0.0075
            loss = F.binary_cross_entropy_with_logits(logits, targets)
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
        json.dump({**best_metrics, 'history': history, 'best_epoch': best_epoch, 'family': 'regularized-dcn-package', 'seed': args.seed}, f, sort_keys=True)


if __name__ == '__main__':
    main()
