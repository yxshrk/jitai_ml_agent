import argparse
import csv
import datetime
import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RankModel(torch.nn.Module):
    def __init__(self, total_dim, n_fields=5, k=16, dropout=0.30):
        super().__init__()
        self.n_fields = n_fields
        self.k = k
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.dropout = torch.nn.Dropout(dropout)
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)
        dim = n_fields * k
        self.cross_w = torch.nn.Parameter(torch.empty(2, dim))
        self.cross_b = torch.nn.Parameter(torch.zeros(2, dim))
        torch.nn.init.normal_(self.cross_w, std=0.01)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(64, 1),
        )
        self.cross_out = torch.nn.Linear(dim, 1, bias=False)

    def forward(self, x):
        e = self.dropout(self.emb(x))
        wide = self.bias + self.lin(x).sum((1, 2))
        x0 = e.reshape(e.shape[0], -1)
        xl = x0
        for layer in range(2):
            scalar = (xl * self.cross_w[layer]).sum(1, keepdim=True)
            xl = x0 * scalar + self.cross_b[layer] + xl
        return wide + self.cross_out(xl).squeeze(1) + self.mlp(x0).squeeze(1)


def date_ord(value):
    text = str(int(value)) if isinstance(value, (int, np.integer, float, np.floating)) else str(value)
    text = text.strip().replace('-', '')
    try:
        return datetime.datetime.strptime(text[:8], '%Y%m%d').date().toordinal()
    except Exception:
        try:
            return int(float(text))
        except Exception:
            return 0


def recency_weights(dates, half_life=7.0):
    vals = np.asarray([date_ord(x) for x in dates], dtype=np.float32)
    latest = float(vals.max()) if len(vals) else 0.0
    result = np.exp2(-(latest - vals) / half_life).astype(np.float32)
    result /= max(float(result.mean()), 1e-8)
    return result


def encode_map(train_values, val_values):
    mapping = {}
    encoded_train = np.empty(len(train_values), dtype=np.int64)
    for i, value in enumerate(train_values):
        key = str(value)
        if key not in mapping:
            mapping[key] = len(mapping)
        encoded_train[i] = mapping[key]
    unknown = len(mapping)
    encoded_val = np.asarray([mapping.get(str(v), unknown) for v in val_values], dtype=np.int64)
    return encoded_train, encoded_val, unknown + 1


def load_csv_data(data_dir):
    feature_names = ['user_id', 'video_id', 'tab', 'duration_ms', 'date']

    def read_file(path):
        cols = {name: [] for name in feature_names}
        labels = []
        with open(path, 'r', newline='') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                for name in feature_names:
                    cols[name].append(row[name])
                labels.append(float(row['long_view']))
        return cols, np.asarray(labels, dtype=np.float32)

    trc, ty = read_file(os.path.join(data_dir, 'train.csv'))
    vac, vy = read_file(os.path.join(data_dir, 'val.csv'))
    durations = np.asarray([float(x) for x in trc['duration_ms']], dtype=np.float64)
    val_durations = np.asarray([float(x) for x in vac['duration_ms']], dtype=np.float64)
    cuts = np.unique(np.quantile(durations, np.linspace(0.1, 0.9, 9)))
    train_bucket = np.searchsorted(cuts, durations, side='right').astype(str)
    val_bucket = np.searchsorted(cuts, val_durations, side='right').astype(str)
    raw_train = [trc['user_id'], trc['video_id'], trc['video_id'], trc['tab'], train_bucket]
    raw_val = [vac['user_id'], vac['video_id'], vac['video_id'], vac['tab'], val_bucket]
    train_fields = []
    val_fields = []
    dims = []
    offset = 0
    for tv, vv in zip(raw_train, raw_val):
        et, ev, dim = encode_map(tv, vv)
        train_fields.append(et + offset)
        val_fields.append(ev + offset)
        dims.append(dim)
        offset += dim
    return {
        'Xt': np.stack(train_fields, axis=1).astype(np.int64),
        'yt': ty,
        'ut': np.asarray(trc['user_id']),
        'dates': np.asarray(trc['date']),
        'Xv': np.stack(val_fields, axis=1).astype(np.int64),
        'yv': vy,
        'uv': np.asarray(vac['user_id']),
        'video': np.asarray(vac['video_id']),
        'field_dims': np.asarray(dims, dtype=np.int64),
        'official': False,
    }


def load_data(data_dir):
    train_npz = os.path.join(data_dir, 'train.npz')
    val_npz = os.path.join(data_dir, 'val.npz')
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        tr = np.load(train_npz)
        va = np.load(val_npz)
        dims = tr['field_dims'].astype(np.int64)
        video_offset = int(dims[0])
        return {
            'Xt': tr['X'].astype(np.int64),
            'yt': tr['y'].astype(np.float32),
            'ut': tr['user'],
            'dates': tr['date'],
            'Xv': va['X'].astype(np.int64),
            'yv': va['y'].astype(np.float32),
            'uv': va['user'],
            'video': va['X'][:, 1].astype(np.int64) - video_offset,
            'field_dims': dims,
            'official': True,
        }
    return load_csv_data(data_dir)


def build_pair_index(users, labels):
    users_array = np.asarray(users)
    order = np.argsort(users_array, kind='mergesort')
    sorted_users = users_array[order]
    boundaries = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1, len(order)]
    pos_parts = []
    start_parts = []
    count_parts = []
    neg_parts = []
    idcg = np.ones(len(labels), dtype=np.float32)
    neg_cursor = 0
    discounts = 1.0 / np.log2(np.arange(2, 7, dtype=np.float64))
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        idx = order[left:right]
        positive = idx[labels[idx] > 0.5]
        negative = idx[labels[idx] <= 0.5]
        ideal = float(discounts[:min(len(positive), 5)].sum())
        idcg[idx] = max(ideal, 1e-8)
        if len(positive) and len(negative):
            pos_parts.append(positive.astype(np.int64, copy=False))
            start_parts.append(np.full(len(positive), neg_cursor, dtype=np.int64))
            count_parts.append(np.full(len(positive), len(negative), dtype=np.int64))
            neg_parts.append(negative.astype(np.int64, copy=False))
            neg_cursor += len(negative)
    if not pos_parts:
        return None
    return {
        'positive': np.concatenate(pos_parts),
        'neg_start': np.concatenate(start_parts),
        'neg_count': np.concatenate(count_parts),
        'negative': np.concatenate(neg_parts),
        'idcg': idcg,
    }


def normalized_metrics(evaluator, users, labels, scores):
    result = evaluator(users, labels.astype(int), scores)
    return {
        'gauc': float(result.get('GAUC', result.get('gauc'))),
        'ndcg5': float(result.get('nDCG@5', result.get('ndcg5'))),
        'primary': float(result['primary']),
    }


def predict(model, features, device, batch_size=65536):
    model.eval()
    pieces = []
    with torch.no_grad():
        for left in range(0, len(features), batch_size):
            xb = torch.from_numpy(features[left:left + batch_size]).to(device)
            pieces.append(model(xb).detach().cpu().numpy())
    return np.concatenate(pieces).astype(np.float64)


def current_ranks(users, scores):
    users_array = np.asarray(users)
    order = np.lexsort((-scores, users_array))
    sorted_users = users_array[order]
    starts = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1]
    ends = np.r_[starts[1:], len(order)]
    group_sizes = ends - starts
    repeated_starts = np.repeat(starts, group_sizes)
    rank_sorted = np.arange(len(order), dtype=np.int64) - repeated_starts
    ranks = np.empty(len(order), dtype=np.int64)
    ranks[order] = rank_sorted
    return ranks


def rank_discount(ranks):
    ranks = np.asarray(ranks, dtype=np.int64)
    result = np.zeros(len(ranks), dtype=np.float32)
    mask = ranks < 5
    result[mask] = (1.0 / np.log2(ranks[mask].astype(np.float64) + 2.0)).astype(np.float32)
    return result


def append_progress(path, record):
    with open(path, 'a') as fh:
        fh.write(json.dumps(record, sort_keys=True) + '\n')
        fh.flush()


def train_once(data, evaluator, pair_index, alpha, seed, epochs, device, keep_scores=False):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    model = RankModel(int(data['field_dims'].sum()), dropout=0.30).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    bce = torch.nn.BCEWithLogitsLoss(reduction='none')
    Xt = data['Xt']
    yt = data['yt']
    weights = recency_weights(data['dates'], 7.0)
    n = len(yt)
    batch_size = 8192
    best_primary = -1.0
    best_scores = None
    best_metrics = None
    best_epoch = 0.0
    curve = []
    pos_all = pair_index['positive'] if pair_index is not None else None
    neg_start = pair_index['neg_start'] if pair_index is not None else None
    neg_count = pair_index['neg_count'] if pair_index is not None else None
    neg_all = pair_index['negative'] if pair_index is not None else None
    idcg = pair_index['idcg'] if pair_index is not None else None
    for epoch in range(epochs):
        train_scores = predict(model, Xt, device)
        ranks = current_ranks(data['ut'], train_scores)
        discounts = rank_discount(ranks)
        model.train()
        permutation = rng.permutation(n)
        running_loss = 0.0
        batches = 0
        for left in range(0, n, batch_size):
            ids = permutation[left:left + batch_size]
            xb = torch.from_numpy(Xt[ids]).to(device)
            yb = torch.from_numpy(yt[ids]).to(device)
            wb = torch.from_numpy(weights[ids]).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            point_each = bce(logits, yb)
            point_loss = (point_each * wb).sum() / wb.sum().clamp_min(1e-8)
            if pair_index is not None:
                chosen = rng.integers(0, len(pos_all), size=len(ids), endpoint=False)
                offsets = (rng.random(len(ids)) * neg_count[chosen]).astype(np.int64)
                pos_ids = pos_all[chosen]
                neg_ids = neg_all[neg_start[chosen] + offsets]
                xp = torch.from_numpy(Xt[pos_ids]).to(device)
                xn = torch.from_numpy(Xt[neg_ids]).to(device)
                pair_each = torch.nn.functional.softplus(-(model(xp) - model(xn)))
                recency_pair = 0.5 * (weights[pos_ids] + weights[neg_ids])
                delta = np.abs(discounts[pos_ids] - discounts[neg_ids]) / idcg[pos_ids]
                positive_delta = delta[delta > 0]
                scale = float(positive_delta.mean()) if len(positive_delta) else 1.0
                normalized_delta = delta / max(scale, 1e-8)
                lambda_factor = (1.0 - alpha) + alpha * normalized_delta
                pair_weights_np = (recency_pair * lambda_factor).astype(np.float32, copy=False)
                pair_weights = torch.from_numpy(pair_weights_np).to(device)
                pair_loss = (pair_each * pair_weights).sum() / pair_weights.sum().clamp_min(1e-8)
                loss = 0.5 * point_loss + 0.5 * pair_loss
            else:
                loss = point_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running_loss += float(loss.detach().cpu())
            batches += 1
        scheduler.step()
        scores = predict(model, data['Xv'], device)
        metrics = normalized_metrics(evaluator, data['uv'], data['yv'], scores)
        curve.append({
            'epoch': epoch + 1,
            'train_loss': round(running_loss / max(batches, 1), 6),
            'gauc': round(metrics['gauc'], 6),
            'ndcg5': round(metrics['ndcg5'], 6),
            'primary': round(metrics['primary'], 6),
        })
        if metrics['primary'] > best_primary + 1e-8:
            best_primary = metrics['primary']
            best_metrics = metrics
            best_epoch = float(epoch + 1)
            if keep_scores:
                best_scores = scores.copy()
    if keep_scores and best_scores is None:
        best_scores = predict(model, data['Xv'], device)
        best_metrics = normalized_metrics(evaluator, data['uv'], data['yv'], best_scores)
        best_primary = best_metrics['primary']
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return {
        'primary': float(best_primary),
        'metrics': best_metrics,
        'best_epoch': best_epoch,
        'curve': curve,
        'scores': best_scores,
    }


def summarize(alpha, phase_scores):
    values = phase_scores.get(alpha, [])
    return {
        'lambda_alpha': alpha,
        'mean': round(float(np.mean(values)), 6) if values else None,
        'std': round(float(np.std(values)), 6) if values else None,
        'scores': [round(float(x), 6) for x in values],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=16)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    progress_path = os.path.join(args.out_dir, 'progress.log')
    if os.path.exists(progress_path):
        os.remove(progress_path)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = load_data(args.data_dir)
    if data['official']:
        from data.official.evaluate import evaluate as evaluator
    else:
        from harness.evaluate_provisional import evaluate as evaluator
    pair_index = build_pair_index(data['ut'], data['yt'])
    smoke_text = os.environ.get('SMOKE_EPOCHS')
    smoke_cap = int(smoke_text) if smoke_text is not None else None
    probe_epochs = 10 if device.type == 'cuda' else 9
    refine_epochs = 14 if device.type == 'cuda' else 12
    final_epochs = args.epochs
    if smoke_cap is not None:
        probe_epochs = min(probe_epochs, smoke_cap)
        refine_epochs = min(refine_epochs, smoke_cap)
        final_epochs = min(final_epochs, smoke_cap)
    probe_seed_count = 12 if device.type == 'cuda' else 8
    refine_seed_count = 8 if device.type == 'cuda' else 5
    if smoke_cap is not None:
        probe_seed_count = 1
        refine_seed_count = 1
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    history = []
    probe_scores = {alpha: [] for alpha in alphas}
    probe_number = 0
    for seed_index in range(probe_seed_count):
        run_seed = args.seed + 1009 * seed_index
        for alpha in alphas:
            result = train_once(data, evaluator, pair_index, alpha, run_seed, probe_epochs, device, False)
            record = {
                'phase': 'paired_probe',
                'probe': probe_number,
                'lambda_alpha': alpha,
                'seed': run_seed,
                'epochs': probe_epochs,
                'best_epoch': result['best_epoch'],
                'gauc': round(result['metrics']['gauc'], 6),
                'ndcg5': round(result['metrics']['ndcg5'], 6),
                'primary': round(result['primary'], 6),
            }
            history.append(record)
            probe_scores[alpha].append(result['primary'])
            append_progress(progress_path, record)
            probe_number += 1
    ranked_alphas = sorted(alphas, key=lambda a: float(np.mean(probe_scores[a])), reverse=True)
    finalist_count = 2 if smoke_cap is None else min(2, len(ranked_alphas))
    finalists = ranked_alphas[:finalist_count]
    refine_scores = {alpha: [] for alpha in finalists}
    for seed_index in range(refine_seed_count):
        run_seed = args.seed + 50021 + 1291 * seed_index
        for alpha in finalists:
            result = train_once(data, evaluator, pair_index, alpha, run_seed, refine_epochs, device, False)
            record = {
                'phase': 'paired_refinement',
                'probe': probe_number,
                'lambda_alpha': alpha,
                'seed': run_seed,
                'epochs': refine_epochs,
                'best_epoch': result['best_epoch'],
                'gauc': round(result['metrics']['gauc'], 6),
                'ndcg5': round(result['metrics']['ndcg5'], 6),
                'primary': round(result['primary'], 6),
            }
            history.append(record)
            refine_scores[alpha].append(result['primary'])
            append_progress(progress_path, record)
            probe_number += 1
    chosen_alpha = max(finalists, key=lambda a: float(np.mean(refine_scores[a])))
    final_seed = args.seed + 900001
    final_result = train_once(data, evaluator, pair_index, chosen_alpha, final_seed, final_epochs, device, True)
    final_record = {
        'phase': 'final',
        'lambda_alpha': chosen_alpha,
        'seed': final_seed,
        'epochs': final_epochs,
        'best_epoch': final_result['best_epoch'],
        'gauc': round(final_result['metrics']['gauc'], 6),
        'ndcg5': round(final_result['metrics']['ndcg5'], 6),
        'primary': round(final_result['primary'], 6),
        'curve': final_result['curve'],
    }
    history.append(final_record)
    append_progress(progress_path, final_record)
    scores = final_result['scores']
    metrics = final_result['metrics']
    with open(os.path.join(args.out_dir, 'predictions.csv'), 'w') as fh:
        fh.write('row_id,user_id,video_id,score\n')
        for i, score in enumerate(scores):
            fh.write(f"{i},{data['uv'][i]},{data['video'][i]},{score:.8g}\n")
    probe_summary = [summarize(alpha, probe_scores) for alpha in alphas]
    probe_summary.sort(key=lambda row: row['mean'], reverse=True)
    refinement_summary = [summarize(alpha, refine_scores) for alpha in finalists]
    refinement_summary.sort(key=lambda row: row['mean'], reverse=True)
    output = {
        'gauc': metrics['gauc'],
        'ndcg5': metrics['ndcg5'],
        'primary': metrics['primary'],
        'chosen_config': {
            'architecture': 'dcn-lite',
            'loss': '0.5-bce+0.5-bpr',
            'pair_weighting': 'lambda-delta-ndcg5',
            'lambda_alpha': chosen_alpha,
            'weighting': 'recency-7d',
            'regularization': 'dropout-0.30+adamw-1e-3+step-decay',
        },
        'probe_summary': probe_summary,
        'refinement_summary': refinement_summary,
        'history': history,
    }
    with open(os.path.join(args.out_dir, 'metrics.json'), 'w') as fh:
        json.dump(output, fh)


if __name__ == '__main__':
    main()
