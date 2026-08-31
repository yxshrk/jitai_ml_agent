import argparse
import csv
import datetime
import json
import math
import os
import sys
from collections import deque

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SequenceDeepFM(torch.nn.Module):
    def __init__(self, total_dim, n_fields, k=16, hidden=128, dropout=0.2):
        super().__init__()
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        input_dim = (n_fields + 1) * k
        self.input_dropout = torch.nn.Dropout(dropout)
        self.shared = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, hidden // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.main_head = torch.nn.Linear(hidden // 2, 1)
        self.watch_head = torch.nn.Linear(hidden // 2, 1)
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)
        torch.nn.init.zeros_(self.main_head.bias)
        torch.nn.init.zeros_(self.watch_head.bias)

    def forward(self, x, history, history_len):
        e = self.emb(x)
        he = self.emb(history)
        mask = (torch.arange(history.shape[1], device=history.device)[None, :] <
                history_len[:, None]).to(he.dtype)
        pooled = (he * mask.unsqueeze(-1)).sum(dim=1)
        pooled = pooled / history_len.clamp_min(1).to(he.dtype).unsqueeze(1)
        pooled = pooled * (history_len > 0).to(he.dtype).unsqueeze(1)
        all_e = torch.cat([e, pooled.unsqueeze(1)], dim=1)
        summed = all_e.sum(dim=1)
        fm_pair = 0.5 * (summed.square() - all_e.square().sum(dim=1)).sum(dim=1)
        linear = self.bias + self.lin(x).sum(dim=(1, 2))
        shared = self.shared(self.input_dropout(all_e).flatten(1))
        logits = linear + fm_pair + self.main_head(shared).squeeze(1)
        watch = torch.sigmoid(self.watch_head(shared).squeeze(1))
        return logits, watch


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def encode_column(train_values, val_values):
    mapping = {}
    train_encoded = np.empty(len(train_values), dtype=np.int64)
    for i, value in enumerate(train_values):
        key = str(value)
        if key not in mapping:
            mapping[key] = len(mapping) + 1
        train_encoded[i] = mapping[key]
    val_encoded = np.fromiter((mapping.get(str(v), 0) for v in val_values),
                              dtype=np.int64, count=len(val_values))
    return train_encoded, val_encoded, len(mapping) + 1


def load_csv_rows(path, training):
    rows = []
    with open(path, newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            item = {
                'user_id': row['user_id'],
                'video_id': row['video_id'],
                'author_id': row.get('author_id', '0'),
                'tab': row['tab'],
                'duration_ms': float(row['duration_ms']),
                'hourmin': row.get('hourmin', '0'),
                'date': row.get('date', '0'),
                'long_view': float(row['long_view']),
            }
            if training:
                item['play_time_ms'] = float(row.get('play_time_ms', '0') or 0.0)
            rows.append(item)
    return rows


def load_data(data_dir):
    train_npz = os.path.join(data_dir, 'train.npz')
    val_npz = os.path.join(data_dir, 'val.npz')
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        with np.load(train_npz) as fh:
            tr = {k: fh[k] for k in fh.files}
        with np.load(val_npz) as fh:
            va = {k: fh[k] for k in fh.files}
        dims = tr['field_dims'].astype(np.int64)
        offsets = np.concatenate(([0], np.cumsum(dims)[:-1]))
        output_users = va['user']
        output_videos = va['X'][:, 1].astype(np.int64) - int(offsets[1])
        return tr, va, output_users, output_videos, True

    train_rows = load_csv_rows(os.path.join(data_dir, 'train.csv'), True)
    val_rows = load_csv_rows(os.path.join(data_dir, 'val.csv'), False)
    names = ('user_id', 'video_id', 'author_id', 'tab')
    encoded_train = []
    encoded_val = []
    dims = []
    for name in names:
        et, ev, dim = encode_column([r[name] for r in train_rows],
                                    [r[name] for r in val_rows])
        encoded_train.append(et)
        encoded_val.append(ev)
        dims.append(dim)
    train_duration = np.asarray([r['duration_ms'] for r in train_rows], dtype=np.float64)
    val_duration = np.asarray([r['duration_ms'] for r in val_rows], dtype=np.float64)
    edges = np.unique(np.quantile(train_duration, np.linspace(0.1, 0.9, 9)))
    encoded_train.append(np.searchsorted(edges, train_duration, side='right').astype(np.int64))
    encoded_val.append(np.searchsorted(edges, val_duration, side='right').astype(np.int64))
    dims.append(len(edges) + 1)
    dims = np.asarray(dims, dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(dims)[:-1]))
    tr = {
        'X': (np.stack(encoded_train, axis=1) + offsets).astype(np.int32),
        'y': np.asarray([r['long_view'] for r in train_rows], dtype=np.float32),
        'user': np.asarray([r['user_id'] for r in train_rows]),
        'duration_ms': train_duration.astype(np.float32),
        'play_time_ms': np.asarray([r['play_time_ms'] for r in train_rows], dtype=np.float32),
        'hourmin': np.asarray([r['hourmin'] for r in train_rows]),
        'date': np.asarray([r['date'] for r in train_rows]),
        'field_dims': dims,
    }
    va = {
        'X': (np.stack(encoded_val, axis=1) + offsets).astype(np.int32),
        'y': np.asarray([r['long_view'] for r in val_rows], dtype=np.float32),
        'user': np.asarray([r['user_id'] for r in val_rows]),
        'duration_ms': val_duration.astype(np.float32),
        'hourmin': np.asarray([r['hourmin'] for r in val_rows]),
        'date': np.asarray([r['date'] for r in val_rows]),
        'field_dims': dims,
    }
    return tr, va, np.asarray([r['user_id'] for r in val_rows]), np.asarray(
        [r['video_id'] for r in val_rows]), False


def get_evaluator(fast_path):
    if fast_path:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    return evaluate


def parse_date(value):
    digits = ''.join(ch for ch in str(value) if ch.isdigit())
    try:
        if len(digits) >= 8:
            return datetime.date(int(digits[:4]), int(digits[4:6]), int(digits[6:8]))
    except ValueError:
        pass
    return datetime.date(2022, 1, 1)


def parse_hour_minute(value):
    digits = ''.join(ch for ch in str(value) if ch.isdigit())
    try:
        number = int(digits) if digits else 0
    except ValueError:
        number = 0
    hour = max(0, min(23, number // 100))
    minute = max(0, min(59, number % 100))
    return hour, minute


def temporal_arrays(date_values, hour_values):
    n = len(date_values)
    hour = np.empty(n, dtype=np.int64)
    weekday = np.empty(n, dtype=np.int64)
    timestamp = np.empty(n, dtype=np.int64)
    epoch = datetime.date(2020, 1, 1).toordinal()
    for i in range(n):
        day = parse_date(date_values[i])
        h, m = parse_hour_minute(hour_values[i])
        hour[i] = h
        weekday[i] = day.weekday()
        timestamp[i] = (day.toordinal() - epoch) * 1440 + h * 60 + m
    return hour, weekday, timestamp


def build_causal_features(tr, va):
    base_dims = np.asarray(tr['field_dims'], dtype=np.int64)
    base_total = int(base_dims.sum())
    base_offsets = np.concatenate(([0], np.cumsum(base_dims)[:-1]))
    author_pad = int(base_offsets[2])
    train_hour, train_weekday, train_time = temporal_arrays(tr['date'], tr['hourmin'])
    val_hour, val_weekday, val_time = temporal_arrays(va['date'], va['hourmin'])
    tab_offset = int(base_offsets[3])
    train_tab_local = tr['X'][:, 3].astype(np.int64) - tab_offset
    val_tab_local = va['X'][:, 3].astype(np.int64) - tab_offset
    train_rand = (train_tab_local == 2).astype(np.int64)
    val_rand = (val_tab_local == 2).astype(np.int64)

    histories = {}
    previous_time = {}
    session_position = {}
    gap_edges = np.asarray([1, 5, 15, 30, 60, 180, 720, 1440, 4320], dtype=np.int64)
    pos_edges = np.asarray([1, 2, 3, 4, 5, 8, 16, 32], dtype=np.int64)

    def process(users, authors, timestamps):
        n = len(users)
        history = np.full((n, 12), author_pad, dtype=np.int32)
        history_len = np.zeros(n, dtype=np.int16)
        gaps = np.zeros(n, dtype=np.int64)
        positions = np.zeros(n, dtype=np.int64)
        for i in range(n):
            key = str(users[i])
            queue = histories.get(key)
            if queue is None:
                queue = deque(maxlen=12)
                histories[key] = queue
            length = len(queue)
            history_len[i] = length
            if length:
                history[i, :length] = np.asarray(queue, dtype=np.int32)
            prev = previous_time.get(key)
            if prev is None:
                gap = 1000000
                position = 0
            else:
                gap = max(0, int(timestamps[i]) - int(prev))
                if gap > 30:
                    position = 0
                else:
                    position = session_position.get(key, 0) + 1
            gaps[i] = np.searchsorted(gap_edges, gap, side='right')
            positions[i] = np.searchsorted(pos_edges, position, side='right')
            previous_time[key] = int(timestamps[i])
            session_position[key] = position
            queue.append(int(authors[i]))
        return history, history_len, gaps, positions

    tr_hist, tr_hlen, tr_gap, tr_pos = process(tr['user'], tr['X'][:, 2], train_time)
    va_hist, va_hlen, va_gap, va_pos = process(va['user'], va['X'][:, 2], val_time)
    extra_dims = np.asarray([24, 7, 2, len(gap_edges) + 1, len(pos_edges) + 1], dtype=np.int64)
    extra_offsets = base_total + np.concatenate(([0], np.cumsum(extra_dims)[:-1]))
    tr_extra = np.stack([train_hour, train_weekday, train_rand, tr_gap, tr_pos], axis=1)
    va_extra = np.stack([val_hour, val_weekday, val_rand, va_gap, va_pos], axis=1)
    Xtr = np.concatenate([tr['X'].astype(np.int64), tr_extra + extra_offsets], axis=1)
    Xva = np.concatenate([va['X'].astype(np.int64), va_extra + extra_offsets], axis=1)
    return {
        'Xtr': Xtr.astype(np.int32),
        'Xva': Xva.astype(np.int32),
        'Htr': tr_hist,
        'Hva': va_hist,
        'Ltr': tr_hlen,
        'Lva': va_hlen,
        'total_dim': int(base_total + extra_dims.sum()),
        'n_fields': Xtr.shape[1],
    }


def metric_values(evaluate, users, labels, scores):
    result = evaluate(users, labels.astype(int), scores)
    return {
        'gauc': float(result['GAUC'] if 'GAUC' in result else result['gauc']),
        'ndcg5': float(result.get('nDCG@5', result.get('ndcg5'))),
        'primary': float(result['primary']),
    }


def predict(model, arrays, device, batch_size=32768):
    model.eval()
    Xv, Hv, Lv = arrays
    outputs = []
    with torch.no_grad():
        for start in range(0, len(Xv), batch_size):
            end = min(start + batch_size, len(Xv))
            x = torch.as_tensor(Xv[start:end], dtype=torch.long, device=device)
            h = torch.as_tensor(Hv[start:end], dtype=torch.long, device=device)
            length = torch.as_tensor(Lv[start:end], dtype=torch.long, device=device)
            logits, _ = model(x, h, length)
            outputs.append(logits.detach().cpu().numpy())
    return np.concatenate(outputs).astype(np.float64)


def train_once(data, config, seed, epochs, device, evaluate, checkpointing):
    seed_everything(seed)
    model = SequenceDeepFM(data['total_dim'], data['n_fields'], k=16,
                           hidden=int(config['hidden']), dropout=float(config['dropout'])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['lr']),
                                  weight_decay=float(config['weight_decay']))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(config['step_size']),
                                                 gamma=float(config['gamma']))
    X = data['Xtr']
    H = data['Htr']
    L = data['Ltr']
    y = data['y']
    watch = data['watch']
    censored = data['censored']
    val_arrays = (data['Xva'], data['Hva'], data['Lva'])
    n = len(y)
    batch_size = int(config['batch_size'])
    rng = np.random.default_rng(seed + 991)
    best_primary = -1.0
    best_scores = None
    checkpoints = []
    bce = torch.nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        permutation = rng.permutation(n)
        last_loss = 0.0
        for start in range(0, n, batch_size):
            idx = permutation[start:start + batch_size]
            xb = torch.as_tensor(X[idx], dtype=torch.long, device=device)
            hb = torch.as_tensor(H[idx], dtype=torch.long, device=device)
            lb = torch.as_tensor(L[idx], dtype=torch.long, device=device)
            yb = torch.as_tensor(y[idx], dtype=torch.float32, device=device)
            wb = torch.as_tensor(watch[idx], dtype=torch.float32, device=device)
            cb = torch.as_tensor(censored[idx], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits, watch_pred = model(xb, hb, lb)
            main_loss = bce(logits, yb)
            uncensored_loss = (1.0 - cb) * (watch_pred - wb).square()
            censored_loss = cb * torch.relu(wb - watch_pred).square()
            aux_loss = (uncensored_loss + censored_loss).mean()
            loss = main_loss + float(config['aux_weight']) * aux_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            last_loss = float(loss.detach().cpu())
        scheduler.step()
        if checkpointing or epoch == epochs - 1:
            scores = predict(model, val_arrays, device)
            metrics = metric_values(evaluate, data['val_users'], data['val_y'], scores)
            checkpoints.append({
                'epoch': epoch + 1,
                'train_loss': last_loss,
                'lr': float(optimizer.param_groups[0]['lr']),
                'gauc': metrics['gauc'],
                'ndcg5': metrics['ndcg5'],
                'primary': metrics['primary'],
            })
            if metrics['primary'] > best_primary:
                best_primary = metrics['primary']
                best_scores = scores.copy()
    return best_primary, best_scores, checkpoints


def make_configs(seed, count):
    rng = np.random.default_rng(seed + 1709)
    configs = []
    for i in range(count):
        configs.append({
            'hidden': int([96, 128, 160][i % 3]),
            'dropout': float(rng.uniform(0.12, 0.32)),
            'weight_decay': float(10.0 ** rng.uniform(-5.0, -3.1)),
            'lr': float(10.0 ** rng.uniform(math.log10(4e-4), math.log10(1.4e-3))),
            'step_size': int([1, 2, 2, 3][i % 4]),
            'gamma': float(rng.uniform(0.52, 0.82)),
            'aux_weight': float([0.03, 0.05, 0.08, 0.12, 0.16][i % 5]),
            'batch_size': 8192,
        })
    return configs


def load_parent_scores(path, expected_length):
    scores = []
    with open(path, newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            scores.append(float(row['score']))
    result = np.asarray(scores, dtype=np.float64)
    if len(result) != expected_length:
        raise RuntimeError('Parent prediction length mismatch')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=12)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    progress_path = os.path.join(args.out_dir, 'progress.log')
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tr, va, output_users, output_videos, fast_path = load_data(args.data_dir)
    evaluate = get_evaluator(fast_path)
    causal = build_causal_features(tr, va)

    duration = np.maximum(np.asarray(tr['duration_ms'], dtype=np.float32), 1.0)
    play = np.maximum(np.asarray(tr.get('play_time_ms', np.zeros(len(duration))),
                                 dtype=np.float32), 0.0)
    watch = np.clip(play / duration, 0.0, 1.0).astype(np.float32)
    censored = (play >= duration).astype(np.float32)
    data = dict(causal)
    data.update({
        'y': np.asarray(tr['y'], dtype=np.float32),
        'watch': watch,
        'censored': censored,
        'val_users': va['user'],
        'val_y': np.asarray(va['y'], dtype=np.float32),
    })

    smoke_value = os.environ.get('SMOKE_EPOCHS')
    smoke_cap = int(smoke_value) if smoke_value is not None else None
    probe_epochs = max(1, min(3, smoke_cap)) if smoke_cap is not None else 3
    final_epochs = max(1, min(args.epochs, smoke_cap)) if smoke_cap is not None else args.epochs
    if smoke_cap is not None:
        probe_count = 2
    else:
        probe_count = 32 if device.type == 'cuda' else 20

    history = []
    configs = make_configs(args.seed, probe_count)
    best_config = None
    best_probe = -1.0
    for i, config in enumerate(configs):
        probe_seed = args.seed + 1000 + i
        primary, _, checkpoints = train_once(data, config, probe_seed, probe_epochs,
                                              device, evaluate, False)
        record = {
            'stage': 'probe',
            'probe': i + 1,
            'seed': probe_seed,
            'epochs': probe_epochs,
            'config': config,
            'primary': float(primary),
            'checkpoints': checkpoints,
        }
        history.append(record)
        with open(progress_path, 'a') as fh:
            fh.write(json.dumps(record, sort_keys=True) + '\n')
        if primary > best_probe:
            best_probe = primary
            best_config = dict(config)

    member_scores = []
    member_metrics = []
    member_seeds = [args.seed, args.seed + 1]
    for member_index, member_seed in enumerate(member_seeds):
        primary, scores, checkpoints = train_once(data, best_config, member_seed,
                                                   final_epochs, device, evaluate, True)
        metrics = metric_values(evaluate, va['user'], data['val_y'], scores)
        member_scores.append(scores)
        member_metrics.append(metrics)
        record = {
            'stage': 'final_member',
            'member': member_index + 1,
            'seed': member_seed,
            'epochs': final_epochs,
            'config': best_config,
            'best_checkpoint_primary': float(primary),
            'gauc': metrics['gauc'],
            'ndcg5': metrics['ndcg5'],
            'primary': metrics['primary'],
            'checkpoints': checkpoints,
        }
        history.append(record)
        with open(progress_path, 'a') as fh:
            fh.write(json.dumps(record, sort_keys=True) + '\n')

    if np.allclose(member_scores[0], member_scores[1], rtol=1e-7, atol=1e-9):
        raise RuntimeError('Distinct-seed ensemble members produced identical scores')
    parent_path = os.environ.get('PARENT_PREDICTIONS')
    if parent_path:
        parent_scores = load_parent_scores(parent_path, len(member_scores[0]))
        for scores in member_scores:
            if np.allclose(scores, parent_scores, rtol=1e-7, atol=1e-9):
                raise RuntimeError('Ensemble member is identical to parent predictions')

    final_scores = np.mean(np.stack(member_scores, axis=0), axis=0)
    final_metrics = metric_values(evaluate, va['user'], data['val_y'], final_scores)
    history.append({
        'stage': 'mean_logit_close',
        'member_seeds': member_seeds,
        'member_primaries': [m['primary'] for m in member_metrics],
        'gauc': final_metrics['gauc'],
        'ndcg5': final_metrics['ndcg5'],
        'primary': final_metrics['primary'],
    })

    metrics = {
        'gauc': final_metrics['gauc'],
        'ndcg5': final_metrics['ndcg5'],
        'primary': final_metrics['primary'],
        'history': history,
    }
    with open(os.path.join(args.out_dir, 'metrics.json'), 'w') as fh:
        json.dump(metrics, fh)
    with open(os.path.join(args.out_dir, 'predictions.csv'), 'w') as fh:
        fh.write('row_id,user_id,video_id,score\n')
        for i, score in enumerate(final_scores):
            fh.write(f'{i},{output_users[i]},{output_videos[i]},{score:.9g}\n')


if __name__ == '__main__':
    main()
