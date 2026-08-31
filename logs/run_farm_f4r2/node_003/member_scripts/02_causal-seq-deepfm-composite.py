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


def previous_items(xt, train_users, xv, val_users, unknown):
    state = {}
    train_prev = np.empty(len(xt), dtype=np.int64)
    for i, (u, row) in enumerate(zip(train_users, xt)):
        key = scalar(u)
        train_prev[i] = state.get(key, unknown)
        state[key] = int(row[1])
    val_prev = np.empty(len(xv), dtype=np.int64)
    for i, (u, row) in enumerate(zip(val_users, xv)):
        key = scalar(u)
        val_prev[i] = state.get(key, unknown)
        state[key] = int(row[1])
    return train_prev, val_prev


def load_data(data_dir):
    if (data_dir / 'train.npz').exists() and (data_dir / 'val.npz').exists():
        with np.load(data_dir / 'train.npz', allow_pickle=False) as z:
            xt = np.asarray(z['X'], np.int64)
            yt = np.asarray(z['y'], np.float32).reshape(-1)
            click = np.asarray(z['click'], np.float32).reshape(-1)
            ut = np.asarray(z['user']).reshape(-1)
            dims = np.asarray(z['field_dims'], np.int64)
        with np.load(data_dir / 'val.npz', allow_pickle=False) as z:
            xv = np.asarray(z['X'], np.int64)
            yv = np.asarray(z['y'], np.float32).reshape(-1)
            uv = np.asarray(z['user']).reshape(-1)
        unknown = int(dims[0])
        train_prev, val_prev = previous_items(xt, ut, xv, uv, unknown)
        videos = xv[:, 1] - int(dims[0])
        total = max(int(dims.sum()), int(max(xt.max(), xv.max())) + 1)
        return xt, yt, click, ut, train_prev, xv, yv, uv, val_prev, videos, total, True

    def read_csv(path, training):
        rows, labels, clicks, users, videos = [], [], [], [], []
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            has_author = 'author_id' in (reader.fieldnames or [])
            for r in reader:
                duration = float(r.get('duration_ms', '0') or 0)
                author = r['author_id'] if has_author else '__missing_author__'
                rows.append((r['user_id'], r['video_id'], author, r.get('tab', '0'), duration))
                labels.append(float(r['long_view']))
                clicks.append(float(r.get('click', '0') or 0) if training else 0.0)
                users.append(r['user_id'])
                videos.append(r['video_id'])
        return rows, np.asarray(labels, np.float32), np.asarray(clicks, np.float32), np.asarray(users), np.asarray(videos)

    tr, yt, click, ut, _ = read_csv(data_dir / 'train.csv', True)
    va, yv, _, uv, videos = read_csv(data_dir / 'val.csv', False)
    cuts = np.quantile([r[4] for r in tr], np.linspace(0.1, 0.9, 9))
    train_cols, val_cols, offset, user_dim = [], [], 0, 0
    for j in range(4):
        mapping = {}
        for r in tr:
            if r[j] not in mapping:
                mapping[r[j]] = len(mapping) + 1
        dim = len(mapping) + 1
        if j == 0:
            user_dim = dim
        train_cols.append(np.asarray([mapping.get(r[j], 0) + offset for r in tr]))
        val_cols.append(np.asarray([mapping.get(r[j], 0) + offset for r in va]))
        offset += dim
    train_cols.append(np.searchsorted(cuts, [r[4] for r in tr]) + offset)
    val_cols.append(np.searchsorted(cuts, [r[4] for r in va]) + offset)
    offset += 10
    xt = np.column_stack(train_cols).astype(np.int64)
    xv = np.column_stack(val_cols).astype(np.int64)
    train_prev, val_prev = previous_items(xt, ut, xv, uv, user_dim)
    return xt, yt, click, ut, train_prev, xv, yv, uv, val_prev, videos, offset, False


class SeqDeepFM(nn.Module):
    def __init__(self, total, fields, bias):
        super().__init__()
        k = 16
        self.emb = nn.Embedding(total, k)
        self.linear = nn.Embedding(total, 1)
        self.deep = nn.Sequential(nn.Linear((fields + 1) * k, 128), nn.ReLU(), nn.Dropout(0.25), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.15))
        self.main = nn.Linear(65, 1)
        self.aux = nn.Linear(64, 1)
        self.bias = nn.Parameter(torch.tensor(float(bias)))
        nn.init.normal_(self.emb.weight, std=0.01)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x, previous):
        e = F.dropout(self.emb(x), p=0.15, training=self.training)
        pe = self.emb(previous)
        summed = e.sum(1)
        fm = 0.5 * (summed.square() - e.square().sum(1)).sum(1)
        sequence_pair = (e[:, 1] * pe).sum(1, keepdim=True)
        hidden = self.deep(torch.cat([e.flatten(1), pe], 1))
        main = self.linear(x).sum(1).squeeze(1) + fm + self.main(torch.cat([hidden, sequence_pair], 1)).squeeze(1) + self.bias
        return main, self.aux(hidden).squeeze(1)


def predict(model, x, previous, device):
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(x), 131072):
            xb = torch.as_tensor(x[start:start + 131072], device=device)
            pb = torch.as_tensor(previous[start:start + 131072], device=device)
            parts.append(model(xb, pb)[0].cpu().numpy())
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
    xt, yt, clicks, ut, train_prev, xv, yv, uv, val_prev, videos, total, fast = load_data(Path(args.data_dir))
    if fast:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    rate = float(np.clip(yt.mean(), 1e-5, 1 - 1e-5))
    model = SeqDeepFM(total, xt.shape[1], math.log(rate / (1 - rate))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0022, weight_decay=6e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4, 7, 10], gamma=0.4)
    epochs = 14
    if os.getenv('SMOKE_EPOCHS') is not None:
        epochs = min(epochs, max(1, int(os.environ['SMOKE_EPOCHS'])))
    rng = np.random.default_rng(args.seed + 313)
    best_scores, best_metrics, best_epoch, history = None, None, 0, []
    for epoch in range(epochs):
        order = rng.permutation(len(xt))
        model.train()
        for start in range(0, len(order), 32768):
            idx = order[start:start + 32768]
            xb = torch.as_tensor(xt[idx], device=device)
            pb = torch.as_tensor(train_prev[idx], device=device)
            yb = torch.as_tensor(yt[idx], device=device)
            cb = torch.as_tensor(clicks[idx], device=device)
            optimizer.zero_grad(set_to_none=True)
            logits, aux = model(xb, pb)
            loss = F.binary_cross_entropy_with_logits(logits, yb) + 0.1 * F.binary_cross_entropy_with_logits(aux, cb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        scheduler.step()
        scores = predict(model, xv, val_prev, device)
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
        json.dump({**best_metrics, 'history': history, 'best_epoch': best_epoch, 'family': 'causal-seq-deepfm-composite', 'seed': args.seed}, f, sort_keys=True)


if __name__ == '__main__':
    main()
