import argparse
import copy
import csv
import datetime
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.official.evaluate import evaluate as official_evaluate
from harness.evaluate_provisional import evaluate as provisional_evaluate


class DCNLite(torch.nn.Module):
    def __init__(self, total_dim, fields=5, k=16, hidden=128, dropout=0.25):
        super().__init__()
        self.fields = fields
        self.k = k
        d = fields * k
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.emb_drop = torch.nn.Dropout(dropout)
        self.cross_w = torch.nn.Parameter(torch.empty(d))
        self.cross_b = torch.nn.Parameter(torch.zeros(d))
        self.deep = torch.nn.Sequential(
            torch.nn.Linear(d, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, hidden // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden // 2, 1),
        )
        self.cross_out = torch.nn.Linear(d, 1)
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)
        torch.nn.init.normal_(self.cross_w, std=0.01)
        torch.nn.init.xavier_uniform_(self.cross_out.weight)
        torch.nn.init.zeros_(self.cross_out.bias)
        for mod in self.deep:
            if isinstance(mod, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(mod.weight)
                torch.nn.init.zeros_(mod.bias)

    def forward(self, x):
        e = self.emb_drop(self.emb(x))
        summed = e.sum(1)
        fm = 0.5 * (summed.square() - e.square().sum(1)).sum(1)
        linear = self.lin(x).sum((1, 2))
        flat = e.reshape(e.shape[0], -1)
        scalar = (flat * self.cross_w).sum(1, keepdim=True)
        cross = flat + flat * scalar + self.cross_b
        return self.bias + linear + fm + self.cross_out(cross).squeeze(1) + self.deep(flat).squeeze(1)


class ParentFM(torch.nn.Module):
    def __init__(self, total_dim, k=16):
        super().__init__()
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)

    def forward(self, x):
        e = self.emb(x)
        s = e.sum(1)
        pair = 0.5 * (s.square() - e.square().sum(1)).sum(1)
        return self.bias + self.lin(x).sum((1, 2)) + 0.5 * (s.square() - e.square().sum(1)).sum(1)


def metric_values(m):
    return float(m.get('GAUC', m.get('gauc'))), float(m.get('nDCG@5', m.get('ndcg5'))), float(m['primary'])


def date_ordinals(values):
    vals = np.asarray(values).reshape(-1)
    unique, inv = np.unique(vals.astype(str), return_inverse=True)
    converted = np.empty(len(unique), dtype=np.float64)
    for i, value in enumerate(unique):
        text = str(value).replace('-', '')[:8]
        try:
            converted[i] = datetime.date(int(text[:4]), int(text[4:6]), int(text[6:8])).toordinal()
        except Exception:
            converted[i] = float(i)
    return converted[inv]


def recency_weights(dates, half_life):
    ords = date_ordinals(dates)
    age = ords.max() - ords
    w = np.power(0.5, age / float(half_life)).astype(np.float32)
    w /= max(float(w.mean()), 1e-8)
    return w


def make_pool(keys, labels):
    neg = np.flatnonzero(~labels).astype(np.int64)
    neg_keys = keys[neg]
    order = np.argsort(neg_keys, kind='stable')
    sorted_idx = neg[order]
    sorted_keys = neg_keys[order]
    unique, starts, counts = np.unique(sorted_keys, return_index=True, return_counts=True)
    return unique.astype(np.int64), starts.astype(np.int64), counts.astype(np.int64), sorted_idx


def sample_pool(pool, query_keys, rng):
    unique, starts, counts, sorted_idx = pool
    query_keys = np.asarray(query_keys, dtype=np.int64)
    loc = np.searchsorted(unique, query_keys)
    valid = loc < len(unique)
    safe = np.minimum(loc, max(len(unique) - 1, 0))
    if len(unique) == 0:
        return np.full(len(query_keys), -1, dtype=np.int64)
    valid &= unique[safe] == query_keys
    out = np.full(len(query_keys), -1, dtype=np.int64)
    if np.any(valid):
        lv = loc[valid]
        offsets = (rng.random_sample(int(valid.sum())) * counts[lv]).astype(np.int64)
        out[valid] = sorted_idx[starts[lv] + offsets]
    return out


def make_pair_data(users, labels, dates, hourmin, tabs):
    labels = np.asarray(labels).reshape(-1) > 0.5
    _, user_code = np.unique(np.asarray(users).reshape(-1), return_inverse=True)
    _, date_code = np.unique(np.asarray(dates).reshape(-1), return_inverse=True)
    raw_hour = np.asarray(hourmin).reshape(-1).astype(np.int64)
    hour = raw_hour if raw_hour.max(initial=0) <= 23 else raw_hour // 100
    _, hour_code = np.unique(hour, return_inverse=True)
    _, tab_code = np.unique(np.asarray(tabs).reshape(-1), return_inverse=True)
    nu = int(user_code.max()) + 1
    nd = int(date_code.max()) + 1
    nh = int(hour_code.max()) + 1
    nt = int(tab_code.max()) + 1
    user_key = user_code.astype(np.int64)
    day_key = user_key * nd + date_code.astype(np.int64)
    hour_key = day_key * nh + hour_code.astype(np.int64)
    tab_key = day_key * nt + tab_code.astype(np.int64)
    user_pool = make_pool(user_key, labels)
    user_neg_counts = np.zeros(nu, dtype=np.int64)
    np.add.at(user_neg_counts, user_code[~labels], 1)
    pos = np.flatnonzero(labels & (user_neg_counts[user_code] > 0)).astype(np.int64)
    return {
        'pos': pos,
        'user_key': user_key,
        'day_key': day_key,
        'hour_key': hour_key,
        'tab_key': tab_key,
        'user_pool': user_pool,
        'day_pool': make_pool(day_key, labels),
        'hour_pool': make_pool(hour_key, labels),
        'tab_pool': make_pool(tab_key, labels),
    }


def sample_negatives(chosen, pair_data, context_fraction, rng):
    negatives = sample_pool(pair_data['user_pool'], pair_data['user_key'][chosen], rng)
    targeted = np.flatnonzero(rng.random_sample(len(chosen)) < context_fraction)
    if len(targeted) == 0:
        return negatives
    chosen_target = chosen[targeted]
    use_hour = rng.random_sample(len(targeted)) < 0.5
    contextual = np.full(len(targeted), -1, dtype=np.int64)
    if np.any(use_hour):
        contextual[use_hour] = sample_pool(pair_data['hour_pool'], pair_data['hour_key'][chosen_target[use_hour]], rng)
    if np.any(~use_hour):
        contextual[~use_hour] = sample_pool(pair_data['tab_pool'], pair_data['tab_key'][chosen_target[~use_hour]], rng)
    missing = contextual < 0
    if np.any(missing):
        contextual[missing] = sample_pool(pair_data['day_pool'], pair_data['day_key'][chosen_target[missing]], rng)
    found = contextual >= 0
    negatives[targeted[found]] = contextual[found]
    return negatives


def predict(model, X, device, batch=65536):
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(X), batch):
            xb = X[start:start + batch].to(device, non_blocking=True)
            chunks.append(model(xb).detach().cpu().numpy())
    return np.concatenate(chunks).astype(np.float64)


def lr_multiplier(schedule, progress):
    if schedule == 'rapid_half':
        return 0.55 ** int(progress / 0.5)
    if schedule == 'rapid_one':
        return 0.42 ** int(progress)
    if schedule == 'stair_1p5':
        return 0.35 ** int(progress / 1.5)
    if schedule == 'two_drop':
        return 0.12 if progress >= 3.0 else (0.38 if progress >= 1.5 else 1.0)
    return 1.0


def train_package(cfg, seed, epochs, Xt, yt, Xv, val_user, val_y, train_weights, pair_data, total_dim, device, evaluator, half_checkpoints):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    rng = np.random.RandomState(seed)
    model = DCNLite(total_dim, dropout=float(cfg['dropout'])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg['lr']), weight_decay=float(cfg['weight_decay']))
    pos_pool = pair_data['pos']
    n = len(yt)
    batch_size = 16384
    checks = 2 if half_checkpoints else 1
    best_primary = -1.0
    best_scores = None
    trace = []
    last_loss = 0.0
    for epoch in range(epochs):
        permutation = rng.permutation(n)
        edges = [0, n // 2, n] if half_checkpoints else [0, n]
        for segment in range(checks):
            model.train()
            left, right = edges[segment], edges[segment + 1]
            progress = epoch + segment / float(checks)
            current_lr = float(cfg['lr']) * lr_multiplier(cfg['schedule'], progress)
            for pg in opt.param_groups:
                pg['lr'] = current_lr
            for start in range(left, right, batch_size):
                idx_np = permutation[start:min(start + batch_size, right)]
                if len(idx_np) == 0:
                    continue
                idx = torch.from_numpy(idx_np).long()
                xb = Xt[idx].to(device, non_blocking=True)
                yb = yt[idx].to(device, non_blocking=True)
                wb = train_weights[idx].to(device, non_blocking=True)
                logits = model(xb)
                bce_each = torch.nn.functional.binary_cross_entropy_with_logits(logits, yb, reduction='none')
                bce_loss = (bce_each * wb).sum() / wb.sum().clamp_min(1e-6)
                pair_n = len(idx_np)
                chosen = pos_pool[rng.randint(0, len(pos_pool), size=pair_n)]
                negatives = sample_negatives(chosen, pair_data, float(cfg['context_fraction']), rng)
                pair_indices = np.concatenate([chosen, negatives])
                pair_x = Xt[torch.from_numpy(pair_indices).long()].to(device, non_blocking=True)
                pair_scores = model(pair_x)
                pair_w = train_weights[torch.from_numpy(chosen).long()].to(device, non_blocking=True)
                bpr_each = torch.nn.functional.softplus(-(pair_scores[:pair_n] - pair_scores[pair_n:]))
                bpr_loss = (bpr_each * pair_w).sum() / pair_w.sum().clamp_min(1e-6)
                loss = 0.5 * bce_loss + 0.5 * bpr_loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                last_loss = float(loss.detach().cpu())
            scores = predict(model, Xv, device)
            gauc, ndcg, primary = metric_values(evaluator(val_user, val_y, scores))
            checkpoint = epoch + (segment + 1) / float(checks)
            trace.append({'checkpoint': checkpoint, 'train_loss': round(last_loss, 6), 'lr': current_lr, 'val_gauc': round(gauc, 6), 'val_ndcg5': round(ndcg, 6), 'val_primary': round(primary, 6)})
            if primary > best_primary + 1e-8:
                best_primary = primary
                best_scores = scores.copy()
    model.to('cpu')
    del model, opt
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_primary, best_scores, trace


def train_parent_reference(seed, epochs, Xt, yt, Xv, val_user, val_y, total_dim, device, evaluator):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    rng = np.random.RandomState(seed)
    model = ParentFM(total_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best = -1.0
    best_scores = None
    patience = 0
    for _ in range(epochs):
        model.train()
        perm = rng.permutation(len(yt))
        for start in range(0, len(yt), 8192):
            idx = torch.from_numpy(perm[start:start + 8192]).long()
            xb = Xt[idx].to(device, non_blocking=True)
            yb = yt[idx].to(device, non_blocking=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        scores = predict(model, Xv, device)
        primary = metric_values(evaluator(val_user, val_y, scores))[2]
        if primary > best + 1e-8:
            best = primary
            best_scores = scores.copy()
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break
    model.to('cpu')
    del model, opt
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return best_scores


def rank_transform(scores):
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(len(scores), dtype=np.float64)
    if len(scores) > 1:
        ranks /= float(len(scores) - 1)
    return ranks


def append_progress(path, record):
    with open(path, 'a') as fh:
        fh.write(json.dumps(record, sort_keys=True) + '\n')


def read_csv_rows(path, training):
    rows = []
    with open(path, newline='') as fh:
        for row in csv.DictReader(fh):
            item = {
                'user': row['user_id'],
                'video': row['video_id'],
                'author': row.get('author_id', row['video_id']),
                'tab': row['tab'],
                'duration': float(row['duration_ms']),
                'date': row['date'],
                'hourmin': int(float(row['hourmin'])),
                'y': float(row['long_view']),
            }
            rows.append(item)
    return rows


def load_data(data_dir):
    train_npz = os.path.join(data_dir, 'train.npz')
    val_npz = os.path.join(data_dir, 'val.npz')
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        tr = np.load(train_npz)
        va = np.load(val_npz)
        return {
            'Xt': tr['X'].astype(np.int64), 'yt': tr['y'].astype(np.float32),
            'train_user': np.asarray(tr['user']), 'train_date': np.asarray(tr['date']),
            'train_hourmin': np.asarray(tr['hourmin']), 'Xv': va['X'].astype(np.int64),
            'val_y': np.asarray(va['y']).astype(int), 'val_user': np.asarray(va['user']),
            'val_video': np.asarray(va['X'])[:, 1], 'field_dims': np.asarray(tr['field_dims']),
            'evaluator': official_evaluate,
        }
    train_rows = read_csv_rows(os.path.join(data_dir, 'train.csv'), True)
    val_rows = read_csv_rows(os.path.join(data_dir, 'val.csv'), False)
    quantiles = np.quantile(np.asarray([r['duration'] for r in train_rows]), np.linspace(0.1, 0.9, 9))
    names = ['user', 'video', 'author', 'tab']
    maps = []
    for name in names:
        values = sorted(set(r[name] for r in train_rows))
        maps.append({v: i + 1 for i, v in enumerate(values)})
    dims = [len(m) + 1 for m in maps] + [10]
    offsets = np.cumsum([0] + dims[:-1]).astype(np.int64)
    def encode(rows):
        X = np.empty((len(rows), 5), dtype=np.int64)
        for i, row in enumerate(rows):
            for j, name in enumerate(names):
                X[i, j] = offsets[j] + maps[j].get(row[name], 0)
            X[i, 4] = offsets[4] + int(np.searchsorted(quantiles, row['duration'], side='right'))
        return X
    return {
        'Xt': encode(train_rows), 'yt': np.asarray([r['y'] for r in train_rows], dtype=np.float32),
        'train_user': np.asarray([r['user'] for r in train_rows]), 'train_date': np.asarray([r['date'] for r in train_rows]),
        'train_hourmin': np.asarray([r['hourmin'] for r in train_rows]), 'Xv': encode(val_rows),
        'val_y': np.asarray([r['y'] for r in val_rows], dtype=int), 'val_user': np.asarray([r['user'] for r in val_rows]),
        'val_video': np.asarray([r['video'] for r in val_rows]), 'field_dims': np.asarray(dims),
        'evaluator': provisional_evaluate,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=18)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    progress_path = os.path.join(args.out_dir, 'progress.log')
    if os.path.exists(progress_path):
        os.remove(progress_path)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    data = load_data(args.data_dir)
    Xt = torch.from_numpy(data['Xt'])
    yt = torch.from_numpy(data['yt'])
    Xv = torch.from_numpy(data['Xv'])
    val_user = data['val_user']
    val_y = data['val_y']
    evaluator = data['evaluator']
    total_dim = int(data['field_dims'].sum())
    pair_data = make_pair_data(data['train_user'], data['yt'], data['train_date'], data['train_hourmin'], data['Xt'][:, 3])
    if len(pair_data['pos']) == 0:
        raise RuntimeError('no eligible within-user positive-negative pairs')
    smoke_value = os.environ.get('SMOKE_EPOCHS')
    smoke_cap = int(smoke_value) if smoke_value is not None else None
    coarse_epochs = 3 if smoke_cap is None else min(3, smoke_cap)
    refine_epochs = 6 if smoke_cap is None else min(6, smoke_cap)
    final_epochs = args.epochs if smoke_cap is None else min(args.epochs, smoke_cap)
    coarse_count = 24 if smoke_cap is None else 3
    refine_count = 18 if smoke_cap is None else 2
    rng = np.random.RandomState(args.seed + 7919)
    schedules = ['rapid_half', 'rapid_one', 'stair_1p5', 'two_drop']
    half_lives = [3.5, 7.0, 14.0]
    fractions = [0.20, 0.30, 0.40, 0.50]
    coarse_configs = []
    for i in range(coarse_count):
        frac = (i + rng.uniform(0.05, 0.95)) / float(coarse_count)
        coarse_configs.append({
            'dropout': float(rng.uniform(0.15, 0.40)),
            'weight_decay': float(10.0 ** (-4.522878745 + frac * 2.0)),
            'lr': float(10.0 ** rng.uniform(math.log10(4.5e-4), math.log10(1.4e-3))),
            'schedule': schedules[i % len(schedules)],
            'half_life': float(half_lives[(i * 2 + 1) % len(half_lives)]),
            'context_fraction': float(fractions[i % len(fractions)]),
        })
    rng.shuffle(coarse_configs)
    history = []
    best_probe = None
    weight_cache = {}
    for i, cfg in enumerate(coarse_configs):
        half = cfg['half_life']
        if half not in weight_cache:
            weight_cache[half] = torch.from_numpy(recency_weights(data['train_date'], half))
        primary, _, trace = train_package(cfg, args.seed + 1000 + i, coarse_epochs, Xt, yt, Xv, val_user, val_y, weight_cache[half], pair_data, total_dim, device, evaluator, False)
        record = {'stage': 'coarse', 'probe': i, 'seed': args.seed + 1000 + i, 'epochs': coarse_epochs, 'config': cfg, 'primary': round(primary, 7), 'trace': trace}
        history.append(record)
        append_progress(progress_path, {'stage': 'coarse', 'probe': i, 'config': cfg, 'primary': round(primary, 7)})
        if best_probe is None or primary > best_probe[0]:
            best_probe = (primary, copy.deepcopy(cfg))
    center = best_probe[1]
    refine_configs = []
    half_options = [3.5, 5.0, 7.0, 10.0, 14.0]
    for i in range(refine_count):
        cfg = {
            'dropout': float(np.clip(center['dropout'] + rng.normal(0.0, 0.045), 0.12, 0.44)),
            'weight_decay': float(np.clip(center['weight_decay'] * (10.0 ** rng.uniform(-0.38, 0.38)), 2e-5, 5e-3)),
            'lr': float(np.clip(center['lr'] * (10.0 ** rng.uniform(-0.20, 0.20)), 3.5e-4, 1.7e-3)),
            'schedule': schedules[(schedules.index(center['schedule']) + i) % len(schedules)],
            'half_life': float(half_options[(half_options.index(min(half_options, key=lambda x: abs(x - center['half_life']))) + i) % len(half_options)]),
            'context_fraction': float(np.clip(center['context_fraction'] + rng.choice([-0.10, 0.0, 0.10]), 0.15, 0.55)),
        }
        if i == 0:
            cfg = copy.deepcopy(center)
        refine_configs.append(cfg)
    best_refined = None
    for i, cfg in enumerate(refine_configs):
        half = cfg['half_life']
        if half not in weight_cache:
            weight_cache[half] = torch.from_numpy(recency_weights(data['train_date'], half))
        primary, _, trace = train_package(cfg, args.seed + 3000 + i, refine_epochs, Xt, yt, Xv, val_user, val_y, weight_cache[half], pair_data, total_dim, device, evaluator, False)
        record = {'stage': 'refine', 'probe': i, 'seed': args.seed + 3000 + i, 'epochs': refine_epochs, 'config': cfg, 'primary': round(primary, 7), 'trace': trace}
        history.append(record)
        append_progress(progress_path, {'stage': 'refine', 'probe': i, 'config': cfg, 'primary': round(primary, 7)})
        if best_refined is None or primary > best_refined[0]:
            best_refined = (primary, copy.deepcopy(cfg))
    winning_cfg = best_refined[1]
    winning_half = winning_cfg['half_life']
    if winning_half not in weight_cache:
        weight_cache[winning_half] = torch.from_numpy(recency_weights(data['train_date'], winning_half))
    member_count = 5 if smoke_cap is None else 1
    member_scores = []
    member_primaries = []
    for member in range(member_count):
        member_seed = args.seed + member
        primary, scores, trace = train_package(winning_cfg, member_seed, final_epochs, Xt, yt, Xv, val_user, val_y, weight_cache[winning_half], pair_data, total_dim, device, evaluator, True)
        member_scores.append(scores)
        member_primaries.append(primary)
        history.append({'stage': 'final_member', 'member': member, 'seed': member_seed, 'epochs': final_epochs, 'config': winning_cfg, 'primary': round(primary, 7), 'trace': trace})
        append_progress(progress_path, {'stage': 'final_member', 'member': member, 'seed': member_seed, 'primary': round(primary, 7)})
    if member_count > 1:
        for i in range(member_count):
            for j in range(i):
                assert not np.allclose(member_scores[i], member_scores[j]), 'ensemble members produced identical scores'
        parent_scores = train_parent_reference(args.seed + 9000, 12, Xt, yt, Xv, val_user, val_y, total_dim, device, evaluator)
        for scores in member_scores:
            assert not np.allclose(scores, parent_scores), 'context-stratified member is identical to parent predictions'
        final_scores = np.mean(np.stack([rank_transform(s) for s in member_scores]), axis=0)
    else:
        final_scores = member_scores[0]
    gauc, ndcg, primary = metric_values(evaluator(val_user, val_y, final_scores))
    history.append({'stage': 'selected_ensemble' if member_count > 1 else 'selected_single', 'members': member_count, 'winning_config': winning_cfg, 'member_primaries': [round(x, 7) for x in member_primaries], 'val_gauc': round(gauc, 7), 'val_ndcg5': round(ndcg, 7), 'val_primary': round(primary, 7)})
    with open(os.path.join(args.out_dir, 'metrics.json'), 'w') as fh:
        json.dump({'gauc': gauc, 'ndcg5': ndcg, 'primary': primary, 'winning_config': winning_cfg, 'history': history}, fh)
    with open(os.path.join(args.out_dir, 'predictions.csv'), 'w') as fh:
        fh.write('row_id,user_id,video_id,score\n')
        for i, score in enumerate(final_scores):
            fh.write(f'{i},{val_user[i]},{data["val_video"][i]},{score:.9g}\n')


if __name__ == '__main__':
    main()
