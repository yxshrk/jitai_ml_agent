import argparse
import csv
import json
import os
import random
from datetime import datetime

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))


def date_number(value):
    text = str(value).strip()
    if text.endswith('.0'):
        text = text[:-2]
    try:
        return datetime.strptime(text, '%Y%m%d').toordinal()
    except ValueError:
        try:
            return int(float(text))
        except ValueError:
            return 0


def load_npz(data_dir):
    train_path = os.path.join(data_dir, 'train.npz')
    val_path = os.path.join(data_dir, 'val.npz')
    with np.load(train_path, allow_pickle=False) as tr:
        x_train = np.asarray(tr['X'], dtype=np.int64)
        y_train = np.asarray(tr['y'], dtype=np.float32)
        user_train = np.asarray(tr['user'])
        date_train = np.asarray(tr['date'])
        field_dims = np.asarray(tr['field_dims'], dtype=np.int64)
    with np.load(val_path, allow_pickle=False) as va:
        x_val = np.asarray(va['X'], dtype=np.int64)
        y_val = np.asarray(va['y'], dtype=np.float32)
        user_val = np.asarray(va['user'])
    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1], dtype=np.int64)))
    video_val = x_val[:, 1] - offsets[1]
    return x_train, y_train, user_train, date_train, x_val, y_val, user_val, video_val, field_dims, True


def read_csv_rows(path):
    rows = []
    with open(path, 'r', newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({
                'user_id': row['user_id'],
                'video_id': row['video_id'],
                'tab': row['tab'],
                'duration_ms': float(row['duration_ms']),
                'date': row['date'],
                'long_view': float(row['long_view'])
            })
    return rows


def make_mapping(values):
    unique = sorted(set(values))
    return {value: index + 1 for index, value in enumerate(unique)}


def load_csv(data_dir):
    train_rows = read_csv_rows(os.path.join(data_dir, 'train.csv'))
    val_rows = read_csv_rows(os.path.join(data_dir, 'val.csv'))
    user_map = make_mapping([r['user_id'] for r in train_rows])
    video_map = make_mapping([r['video_id'] for r in train_rows])
    tab_map = make_mapping([r['tab'] for r in train_rows])
    durations = np.asarray([r['duration_ms'] for r in train_rows], dtype=np.float64)
    quantiles = np.quantile(durations, np.linspace(0.1, 0.9, 9)) if len(durations) else np.zeros(9)
    field_dims = np.asarray([len(user_map) + 1, len(video_map) + 1, 2, len(tab_map) + 1, 10], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1], dtype=np.int64)))

    def encode(rows):
        x = np.zeros((len(rows), 5), dtype=np.int64)
        for i, row in enumerate(rows):
            x[i, 0] = user_map.get(row['user_id'], 0)
            x[i, 1] = video_map.get(row['video_id'], 0)
            x[i, 2] = 0
            x[i, 3] = tab_map.get(row['tab'], 0)
            x[i, 4] = int(np.searchsorted(quantiles, row['duration_ms'], side='right'))
        x += offsets.reshape(1, -1)
        return x

    x_train = encode(train_rows)
    x_val = encode(val_rows)
    y_train = np.asarray([r['long_view'] for r in train_rows], dtype=np.float32)
    y_val = np.asarray([r['long_view'] for r in val_rows], dtype=np.float32)
    user_train = np.asarray([r['user_id'] for r in train_rows])
    user_val = np.asarray([r['user_id'] for r in val_rows])
    video_val = np.asarray([r['video_id'] for r in val_rows])
    date_train = np.asarray([r['date'] for r in train_rows])
    return x_train, y_train, user_train, date_train, x_val, y_val, user_val, video_val, field_dims, False


class RegularizedDCN(nn.Module):
    def __init__(self, field_dims, embedding_dim=24, hidden_dim=128, dropout=0.30, cross_layers=2):
        super().__init__()
        total = int(np.sum(field_dims))
        self.embedding = nn.Embedding(total, embedding_dim)
        self.linear_embedding = nn.Embedding(total, 1)
        input_dim = int(len(field_dims) * embedding_dim)
        self.cross_w = nn.ParameterList([nn.Parameter(torch.empty(input_dim)) for _ in range(cross_layers)])
        self.cross_b = nn.ParameterList([nn.Parameter(torch.zeros(input_dim)) for _ in range(cross_layers)])
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.output = nn.Linear(input_dim + hidden_dim // 2, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.zeros_(self.linear_embedding.weight)
        for weight in self.cross_w:
            nn.init.normal_(weight, std=0.01)

    def forward(self, x):
        embedded = self.embedding(x)
        flat = embedded.flatten(1)
        crossed = flat
        for weight, bias in zip(self.cross_w, self.cross_b):
            scale = torch.sum(crossed * weight, dim=1, keepdim=True)
            crossed = flat * scale + bias + crossed
        deep = self.mlp(flat)
        interaction = self.output(torch.cat([crossed, deep], dim=1)).squeeze(1)
        linear = self.linear_embedding(x).sum(dim=1).squeeze(1)
        return interaction + linear + self.bias

    def accessed_row_l2(self, x):
        emb = self.embedding(x)
        linear = self.linear_embedding(x)
        return emb.square().sum(dim=(1, 2)).mean() + linear.square().sum(dim=(1, 2)).mean()


def recency_weights(dates, half_life=7.0):
    ordinals = np.asarray([date_number(v) for v in dates], dtype=np.float64)
    newest = float(np.max(ordinals)) if len(ordinals) else 0.0
    ages = np.maximum(0.0, newest - ordinals)
    weights = np.exp2(-ages / half_life).astype(np.float32)
    mean = float(weights.mean()) if len(weights) else 1.0
    return weights / max(mean, 1e-8)


def build_user_groups(users, labels):
    groups = {}
    for index, user in enumerate(users.tolist()):
        groups.setdefault(user, [[], []])[int(labels[index] >= 0.5)].append(index)
    mixed = [(np.asarray(v[1], dtype=np.int64), np.asarray(v[0], dtype=np.int64)) for v in groups.values() if v[0] and v[1]]
    return mixed


def make_pairs(groups, rng):
    positives = []
    negatives = []
    for pos, neg in groups:
        positives.append(pos)
        negatives.append(rng.choice(neg, size=len(pos), replace=True))
    if not positives:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    pos = np.concatenate(positives)
    neg = np.concatenate(negatives)
    order = rng.permutation(len(pos))
    return pos[order], neg[order]


def predict_logits(model, x, batch_size=8192):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start:start + batch_size]).long()
            outputs.append(model(xb).cpu().numpy())
    return np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float32)


def official_metrics(npz_mode, users, labels, scores):
    if npz_mode:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    result = evaluate(users, labels, scores)
    return {
        'gauc': float(result.get('GAUC', result.get('gauc'))),
        'ndcg5': float(result.get('nDCG@5', result.get('ndcg5'))),
        'primary': float(result.get('primary'))
    }


def train_model(x_train, y_train, users_train, dates_train, x_val, y_val, users_val, field_dims, seed, epochs, npz_mode):
    model = RegularizedDCN(field_dims, embedding_dim=24, hidden_dim=128, dropout=0.30, cross_layers=2)
    embedding_parameters = list(model.embedding.parameters()) + list(model.linear_embedding.parameters())
    embedding_ids = {id(p) for p in embedding_parameters}
    dense_parameters = [p for p in model.parameters() if id(p) not in embedding_ids]
    optimizer = torch.optim.AdamW([
        {'params': embedding_parameters, 'weight_decay': 0.0},
        {'params': dense_parameters, 'weight_decay': 1e-3}
    ], lr=0.00168)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    weights = recency_weights(dates_train, half_life=7.0)
    groups = build_user_groups(users_train, y_train)
    rng = np.random.default_rng(seed)
    batch_size = min(4096, max(64, len(x_train)))
    best_gauc = -float('inf')
    best_state = None

    for _ in range(epochs):
        order = rng.permutation(len(x_train))
        pair_pos, pair_neg = make_pairs(groups, rng)
        split = max(1, (len(order) + 1) // 2)
        for phase_start in range(0, len(order), split):
            phase_indices = order[phase_start:phase_start + split]
            model.train()
            pair_cursor = 0
            for start in range(0, len(phase_indices), batch_size):
                idx = phase_indices[start:start + batch_size]
                xb = torch.from_numpy(x_train[idx]).long()
                yb = torch.from_numpy(y_train[idx]).float()
                wb = torch.from_numpy(weights[idx]).float()
                logits = model(xb)
                point_losses = F.binary_cross_entropy_with_logits(logits, yb, reduction='none')
                point_loss = (point_losses * wb).sum() / wb.sum().clamp_min(1e-8)

                if len(pair_pos):
                    count = len(idx)
                    positions = np.arange(pair_cursor, pair_cursor + count) % len(pair_pos)
                    pair_cursor += count
                    pidx = pair_pos[positions]
                    nidx = pair_neg[positions]
                    px = torch.from_numpy(x_train[pidx]).long()
                    nx = torch.from_numpy(x_train[nidx]).long()
                    pair_loss = F.softplus(-(model(px) - model(nx))).mean()
                    accessed = torch.cat([xb, px, nx], dim=0)
                else:
                    pair_loss = point_loss.detach() * 0.0
                    accessed = xb

                row_l2 = model.accessed_row_l2(accessed)
                loss = 0.5 * point_loss + 0.5 * pair_loss + 1e-4 * row_l2
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            val_logits = predict_logits(model, x_val)
            metrics = official_metrics(npz_mode, users_val, y_val, val_logits)
            if metrics['gauc'] > best_gauc:
                best_gauc = metrics['gauc']
                best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        scheduler.step()

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def write_predictions(path, users, videos, scores):
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['row_id', 'user_id', 'video_id', 'score'])
        for i, (user, video, score) in enumerate(zip(users, videos, scores)):
            writer.writerow([i, user, video, format(float(score), '.10g')])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    seed_everything(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    npz_mode = os.path.isfile(os.path.join(args.data_dir, 'train.npz')) and os.path.isfile(os.path.join(args.data_dir, 'val.npz'))
    if npz_mode:
        data = load_npz(args.data_dir)
    else:
        data = load_csv(args.data_dir)
    x_train, y_train, user_train, date_train, x_val, y_val, user_val, video_val, field_dims, npz_mode = data

    epochs = 3
    smoke = os.environ.get('SMOKE_EPOCHS')
    if smoke is not None:
        epochs = min(epochs, max(1, int(smoke)))

    model = train_model(x_train, y_train, user_train, date_train, x_val, y_val, user_val, field_dims, args.seed, epochs, npz_mode)
    logits = predict_logits(model, x_val)
    scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    metrics = official_metrics(npz_mode, user_val, y_val, logits)
    write_predictions(os.path.join(args.out_dir, 'predictions.csv'), user_val, video_val, scores)
    with open(os.path.join(args.out_dir, 'metrics.json'), 'w', encoding='utf-8') as handle:
        json.dump(metrics, handle, separators=(',', ':'))


if __name__ == '__main__':
    main()
