import argparse
import csv
import datetime
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from data.official.evaluate import evaluate as official_evaluate
except ImportError:
    official_evaluate = None
try:
    from harness.evaluate_provisional import evaluate as provisional_evaluate
except ImportError:
    provisional_evaluate = None


def metric_value(m, *names):
    for name in names:
        if name in m:
            return float(m[name])
    raise KeyError(names)


def run_evaluator(user, labels, scores, fast_path):
    evaluator = official_evaluate if fast_path and official_evaluate is not None else provisional_evaluate
    if evaluator is None:
        evaluator = official_evaluate
    return evaluator(user, labels.astype(int), scores)


def date_ordinals(values):
    out = np.zeros(len(values), dtype=np.float32)
    cache = {}
    for i, value in enumerate(values):
        s = str(value)
        if s.endswith('.0'):
            s = s[:-2]
        s = s.replace('-', '')
        if len(s) >= 8 and s[:8].isdigit():
            key = s[:8]
            if key not in cache:
                try:
                    cache[key] = datetime.date(int(key[:4]), int(key[4:6]), int(key[6:8])).toordinal()
                except ValueError:
                    cache[key] = 0
            out[i] = cache[key]
    return out


def read_csv_rows(path):
    rows = []
    with open(path, 'r', newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append({
                'user_id': row['user_id'],
                'video_id': row['video_id'],
                'tab': row.get('tab', '0'),
                'duration_ms': float(row.get('duration_ms', 0.0) or 0.0),
                'date': row.get('date', '0'),
                'long_view': float(row['long_view'])
            })
    return rows


def load_csv_data(data_dir):
    tr_rows = read_csv_rows(os.path.join(data_dir, 'train.csv'))
    va_rows = read_csv_rows(os.path.join(data_dir, 'val.csv'))
    durations = np.asarray([r['duration_ms'] for r in tr_rows], dtype=np.float64)
    quantiles = np.quantile(durations, np.linspace(0.1, 0.9, 9)) if len(durations) else np.zeros(9)

    def vocabulary(key):
        vals = sorted({r[key] for r in tr_rows})
        return {v: i + 1 for i, v in enumerate(vals)}

    user_map = vocabulary('user_id')
    video_map = vocabulary('video_id')
    tab_map = vocabulary('tab')
    field_dims = np.asarray([len(user_map) + 1, len(video_map) + 1, 1, len(tab_map) + 1, 10], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1]))

    def encode(rows):
        x = np.empty((len(rows), 5), dtype=np.int64)
        for i, r in enumerate(rows):
            x[i, 0] = user_map.get(r['user_id'], 0)
            x[i, 1] = video_map.get(r['video_id'], 0)
            x[i, 2] = 0
            x[i, 3] = tab_map.get(r['tab'], 0)
            x[i, 4] = int(np.searchsorted(quantiles, r['duration_ms'], side='right'))
        x += offsets
        return x

    train = {
        'X': encode(tr_rows),
        'y': np.asarray([r['long_view'] for r in tr_rows], dtype=np.float32),
        'user': np.asarray([r['user_id'] for r in tr_rows]),
        'date': np.asarray([r['date'] for r in tr_rows]),
        'field_dims': field_dims
    }
    val = {
        'X': encode(va_rows),
        'y': np.asarray([r['long_view'] for r in va_rows], dtype=np.float32),
        'user': np.asarray([r['user_id'] for r in va_rows]),
        'video': np.asarray([r['video_id'] for r in va_rows])
    }
    return train, val


def load_data(data_dir):
    train_npz = os.path.join(data_dir, 'train.npz')
    val_npz = os.path.join(data_dir, 'val.npz')
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        with np.load(train_npz) as z:
            train = {k: z[k] for k in z.files}
        with np.load(val_npz) as z:
            val = {k: z[k] for k in z.files}
        if 'video' not in val:
            val['video'] = np.zeros(len(val['y']), dtype=np.int64)
        return train, val, True
    train, val = load_csv_data(data_dir)
    return train, val, False


class DCNLite(torch.nn.Module):
    def __init__(self, total_dim, fields, k, dropout):
        super().__init__()
        width = fields * k
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.emb_drop = torch.nn.Dropout(dropout)
        self.cross_w = torch.nn.Parameter(torch.empty(width))
        self.cross_b = torch.nn.Parameter(torch.zeros(width))
        self.cross_head = torch.nn.Linear(width, 1, bias=False)
        self.deep = torch.nn.Sequential(
            torch.nn.Linear(width, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(64, 1)
        )
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)
        torch.nn.init.normal_(self.cross_w, std=0.01)
        torch.nn.init.zeros_(self.cross_head.weight)
        for layer in self.deep:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
        torch.nn.init.zeros_(self.deep[-1].weight)

    def forward(self, x):
        raw = self.emb(x)
        e = self.emb_drop(raw)
        summed = e.sum(dim=1)
        fm = 0.5 * (summed.square() - e.square().sum(dim=1)).sum(dim=1)
        linear = self.lin(x).sum(dim=(1, 2)) + self.bias
        flat = e.flatten(1)
        cross = flat + flat * torch.sum(flat * self.cross_w, dim=1, keepdim=True) + self.cross_b
        return linear + fm + self.cross_head(cross).squeeze(1) + self.deep(flat).squeeze(1)


def make_pairs(users, labels, seed):
    order = np.argsort(users, kind='mergesort')
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    rng = np.random.default_rng(seed)
    pos_parts = []
    neg_parts = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        idx = order[left:right]
        pos = idx[labels[idx] > 0.5]
        neg = idx[labels[idx] <= 0.5]
        if len(pos) and len(neg):
            pos_parts.append(pos)
            neg_parts.append(neg[rng.integers(0, len(neg), size=len(pos))])
    if not pos_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(pos_parts).astype(np.int64), np.concatenate(neg_parts).astype(np.int64)


def prediction(model, x_cpu, device):
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(x_cpu), 65536):
            xb = x_cpu[start:start + 65536].to(device, non_blocking=True)
            parts.append(model(xb).detach().cpu().numpy())
    return np.concatenate(parts).astype(np.float64)


def train_one(config, seed, epochs, arrays, evaluator, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    x_train, y_train, x_val, val_user, val_y, recency, pair_pos, pair_neg, total_dim = arrays
    model = DCNLite(total_dim, x_train.shape[1], 16, config['dropout']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'])
    n = len(y_train)
    pair_n = len(pair_pos)
    batch_size = 8192 if device.type == 'cuda' else 4096
    best_primary = -1.0
    best_scores = None
    best_metrics = None
    trace = []
    for epoch in range(epochs):
        permutation = torch.randperm(n)
        midpoint = (n + 1) // 2
        for half, (left, right) in enumerate(((0, midpoint), (midpoint, n))):
            model.train()
            losses = []
            segment = permutation[left:right]
            for start in range(0, len(segment), batch_size):
                idx = segment[start:start + batch_size]
                xb = x_train[idx].to(device, non_blocking=True)
                yb = y_train[idx].to(device, non_blocking=True)
                wb = recency[idx].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                point_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yb, reduction='none')
                point_loss = (point_loss * wb).sum() / wb.sum().clamp_min(1e-8)
                if pair_n:
                    psel = torch.randint(0, pair_n, (len(idx),))
                    pi = pair_pos[psel]
                    ni = pair_neg[psel]
                    xp = x_train[pi].to(device, non_blocking=True)
                    xn = x_train[ni].to(device, non_blocking=True)
                    pw = (recency[pi] + recency[ni]).mul(0.5).to(device, non_blocking=True)
                    rank_loss = torch.nn.functional.softplus(-(model(xp) - model(xn)))
                    rank_loss = (rank_loss * pw).sum() / pw.sum().clamp_min(1e-8)
                    loss = 0.5 * point_loss + 0.5 * rank_loss
                else:
                    loss = point_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            scores = prediction(model, x_val, device)
            metrics = evaluator(val_user, val_y, scores)
            primary = metric_value(metrics, 'primary')
            trace.append({
                'epoch': epoch + 0.5 * (half + 1),
                'train_loss': round(float(np.mean(losses)) if losses else 0.0, 6),
                'lr': float(optimizer.param_groups[0]['lr']),
                'val_gauc': round(metric_value(metrics, 'GAUC', 'gauc'), 6),
                'val_primary': round(primary, 6)
            })
            if primary > best_primary + 1e-8:
                best_primary = primary
                best_scores = scores.copy()
                best_metrics = metrics
        scheduler.step()
    return best_primary, best_scores, best_metrics, trace


def user_groups(users):
    order = np.argsort(users, kind='mergesort')
    sorted_users = users[order]
    bounds = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    return [order[left:right] for left, right in zip(bounds[:-1], bounds[1:])]


def rank_average(score_list, groups):
    result = np.zeros(len(score_list[0]), dtype=np.float64)
    for scores in score_list:
        ranks = np.empty(len(scores), dtype=np.float64)
        for idx in groups:
            local_order = np.argsort(scores[idx], kind='mergesort')
            local_rank = np.empty(len(idx), dtype=np.float64)
            local_rank[local_order] = np.arange(len(idx), dtype=np.float64)
            ranks[idx] = local_rank / max(1, len(idx) - 1)
        result += ranks
    return result / len(score_list)


def sigmoid_array(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


def probability_average(score_list):
    return np.mean(np.stack([sigmoid_array(s) for s in score_list], axis=0), axis=0)


def member_tie_rate(scores, groups):
    tied = 0
    total = 0
    for idx in groups:
        n = len(idx)
        if n < 2:
            continue
        values = scores[idx]
        for i in range(n):
            diffs = values[i] - values[i + 1:]
            tied += int(np.count_nonzero(diffs == 0.0))
            total += len(diffs)
    return float(tied / total) if total else 0.0


def margin_temperature(scores, groups):
    margins = []
    for idx in groups:
        values = scores[idx]
        n = len(values)
        for i in range(n):
            d = np.abs(values[i] - values[i + 1:])
            if len(d):
                margins.append(d[d > 0.0])
    nonempty = [x for x in margins if len(x)]
    if not nonempty:
        return 1.0
    values = np.concatenate(nonempty)
    return max(float(np.median(values)), 1e-4)


def anchored_soft_pairwise(score_list, anchor_pos, groups):
    count = len(score_list)
    if count == 1:
        return score_list[0].copy(), [1.0]
    weights = np.full(count, 0.4 / (count - 1), dtype=np.float64)
    weights[anchor_pos] = 0.6
    temperatures = [margin_temperature(s, groups) for s in score_list]
    result = np.zeros(len(score_list[0]), dtype=np.float64)
    for idx in groups:
        n = len(idx)
        if n == 1:
            result[idx] = 0.5
            continue
        local = np.zeros(n, dtype=np.float64)
        for member, scores in enumerate(score_list):
            values = scores[idx]
            diff = (values[:, None] - values[None, :]) / temperatures[member]
            votes = sigmoid_array(diff)
            np.fill_diagonal(votes, 0.0)
            local += weights[member] * votes.sum(axis=1) / (n - 1)
        result[idx] = local
    return result, temperatures


def rescue_harm_gate(anchor, ensemble, labels, groups):
    rescue = 0.0
    harm = 0.0
    informative_users = 0
    for idx in groups:
        n = len(idx)
        if n < 2:
            continue
        local_rescue = 0.0
        local_harm = 0.0
        pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                truth = labels[idx[i]] - labels[idx[j]]
                if truth == 0:
                    continue
                a_margin = (anchor[idx[i]] - anchor[idx[j]]) * truth
                e_margin = (ensemble[idx[i]] - ensemble[idx[j]]) * truth
                a_credit = 1.0 if a_margin > 0 else (0.5 if a_margin == 0 else 0.0)
                e_credit = 1.0 if e_margin > 0 else (0.5 if e_margin == 0 else 0.0)
                local_rescue += max(0.0, e_credit - a_credit)
                local_harm += max(0.0, a_credit - e_credit)
                pairs += 1
        if pairs:
            weight = float(n)
            rescue += weight * local_rescue / pairs
            harm += weight * local_harm / pairs
            informative_users += 1
    ratio = rescue / max(harm, 1e-12)
    passed = rescue > harm and rescue > 0.0 and ratio > 1.2
    return {
        'rescue': float(rescue),
        'harm': float(harm),
        'net_rescue': float(rescue - harm),
        'rescue_harm_ratio': float(ratio),
        'informative_users': int(informative_users),
        'passed': bool(passed)
    }


def serializable_config(config):
    return {k: (int(v) if k == 'step_size' else float(v)) for k, v in config.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=12)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    progress_path = os.path.join(args.out_dir, 'progress.log')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    train, val, fast_path = load_data(args.data_dir)
    evaluator = lambda u, y, s: run_evaluator(u, y, s, fast_path)
    x_train = torch.from_numpy(train['X'].astype(np.int64))
    y_train_np = train['y'].astype(np.float32)
    y_train = torch.from_numpy(y_train_np)
    x_val = torch.from_numpy(val['X'].astype(np.int64))
    val_user = np.asarray(val['user'])
    val_y = val['y'].astype(np.float32)
    groups = user_groups(val_user)
    total_dim = int(np.asarray(train['field_dims']).sum())
    ordinals = date_ordinals(train['date']) if 'date' in train else np.zeros(len(y_train_np), dtype=np.float32)
    max_ordinal = float(ordinals.max()) if len(ordinals) else 0.0
    ages = np.maximum(0.0, max_ordinal - ordinals)
    pair_pos_np, pair_neg_np = make_pairs(np.asarray(train['user']), y_train_np, args.seed)
    pair_pos = torch.from_numpy(pair_pos_np)
    pair_neg = torch.from_numpy(pair_neg_np)

    smoke = int(os.environ.get('SMOKE_EPOCHS', '0') or 0)
    coarse_epochs = min(3, smoke) if smoke > 0 else 3
    refine_epochs = min(5, smoke) if smoke > 0 else 5
    final_epochs = min(args.epochs, smoke) if smoke > 0 else args.epochs
    coarse_count = 2 if smoke > 0 else 40
    refine_count = 1 if smoke > 0 else 18
    final_seed_count = 1 if smoke > 0 else 7

    rng = np.random.default_rng(args.seed + 991)
    coarse_configs = []
    for _ in range(coarse_count):
        coarse_configs.append({
            'dropout': float(rng.uniform(0.12, 0.43)),
            'weight_decay': float(10.0 ** rng.uniform(math.log10(2e-5), math.log10(4e-3))),
            'lr': float(10.0 ** rng.uniform(math.log10(3.5e-4), math.log10(1.5e-3))),
            'gamma': float(rng.choice(np.asarray([0.32, 0.45, 0.58, 0.72, 0.84]))),
            'step_size': int(rng.choice(np.asarray([1, 1, 1, 2, 2, 3]))),
            'half_life': float(rng.choice(np.asarray([3.0, 4.5, 6.5, 9.0, 13.0, 17.0])))
        })

    history = []
    best_primary = -1.0
    best_config = None

    def execute_probe(stage, probe_index, config, epochs):
        nonlocal best_primary, best_config
        weights = np.exp(-math.log(2.0) * ages / config['half_life']).astype(np.float32)
        weights /= max(float(weights.mean()), 1e-8)
        recency = torch.from_numpy(weights)
        arrays = (x_train, y_train, x_val, val_user, val_y, recency, pair_pos, pair_neg, total_dim)
        probe_seed = args.seed + probe_index + (0 if stage == 'coarse' else 10000)
        primary, _, metrics, trace = train_one(config, probe_seed, epochs, arrays, evaluator, device)
        record = {
            'stage': stage,
            'probe': probe_index,
            'config': serializable_config(config),
            'epochs': epochs,
            'gauc': metric_value(metrics, 'GAUC', 'gauc'),
            'ndcg5': metric_value(metrics, 'nDCG@5', 'ndcg5'),
            'primary': primary,
            'best_checkpoint': max(trace, key=lambda z: z['val_primary'])['epoch']
        }
        history.append(record)
        with open(progress_path, 'a') as fh:
            fh.write(json.dumps(record, sort_keys=True) + '\n')
        if primary > best_primary:
            best_primary = primary
            best_config = dict(config)

    for i, config in enumerate(coarse_configs):
        execute_probe('coarse', i, config, coarse_epochs)

    center = dict(best_config)
    refine_configs = []
    for _ in range(refine_count):
        refine_configs.append({
            'dropout': float(np.clip(center['dropout'] + rng.normal(0.0, 0.045), 0.08, 0.48)),
            'weight_decay': float(np.clip(center['weight_decay'] * math.exp(rng.normal(0.0, 0.48)), 1e-5, 8e-3)),
            'lr': float(np.clip(center['lr'] * math.exp(rng.normal(0.0, 0.25)), 2.5e-4, 2e-3)),
            'gamma': float(np.clip(center['gamma'] + rng.normal(0.0, 0.075), 0.25, 0.92)),
            'step_size': int(np.clip(center['step_size'] + rng.choice(np.asarray([-1, 0, 0, 0, 1])), 1, 3)),
            'half_life': float(np.clip(center['half_life'] * math.exp(rng.normal(0.0, 0.22)), 2.5, 21.0))
        })
    for i, config in enumerate(refine_configs):
        execute_probe('refine', i, config, refine_epochs)

    final_weights = np.exp(-math.log(2.0) * ages / best_config['half_life']).astype(np.float32)
    final_weights /= max(float(final_weights.mean()), 1e-8)
    final_recency = torch.from_numpy(final_weights)
    final_arrays = (x_train, y_train, x_val, val_user, val_y, final_recency, pair_pos, pair_neg, total_dim)
    final_scores = []
    final_runs = []
    for offset in range(final_seed_count):
        run_seed = args.seed + offset
        primary, scores, metrics, trace = train_one(best_config, run_seed, final_epochs, final_arrays, evaluator, device)
        tie_rate = member_tie_rate(scores, groups)
        final_scores.append(scores)
        final_runs.append({
            'seed': run_seed,
            'best_primary': primary,
            'gauc': metric_value(metrics, 'GAUC', 'gauc'),
            'ndcg5': metric_value(metrics, 'nDCG@5', 'ndcg5'),
            'tie_rate': tie_rate,
            'trace': trace
        })
        progress_record = {'stage': 'final_member', 'seed': run_seed, 'primary': primary, 'tie_rate': tie_rate}
        with open(progress_path, 'a') as fh:
            fh.write(json.dumps(progress_record, sort_keys=True) + '\n')

    member_primary = np.asarray([r['best_primary'] for r in final_runs], dtype=np.float64)
    member_ties = np.asarray([r['tie_rate'] for r in final_runs], dtype=np.float64)
    median_primary = float(np.median(member_primary))
    median_tie = float(np.median(member_ties))
    tie_limit = max(0.01, median_tie + 0.01, median_tie * 5.0)
    eligible = [
        i for i in range(len(final_scores))
        if member_primary[i] >= median_primary - 0.0010 and member_ties[i] <= tie_limit
    ]
    if not eligible:
        eligible = [int(np.argmax(member_primary))]
    anchor_index = max(eligible, key=lambda i: member_primary[i])
    anchor_scores = final_scores[anchor_index]

    ensemble_history = []
    passing = []
    targets = [1] if smoke > 0 else [3, 5, 7]
    for target_count in targets:
        selected = eligible[:min(target_count, len(eligible))]
        if anchor_index not in selected:
            if len(selected) < target_count:
                selected.append(anchor_index)
            elif selected:
                selected[-1] = anchor_index
            else:
                selected = [anchor_index]
        selected = list(dict.fromkeys(selected))
        selected_scores = [final_scores[i] for i in selected]
        anchor_pos = selected.index(anchor_index)
        candidates = [
            ('rank_average', rank_average(selected_scores, groups), None),
            ('probability_average', probability_average(selected_scores), None)
        ]
        soft_scores, temperatures = anchored_soft_pairwise(selected_scores, anchor_pos, groups)
        candidates.append(('anchored_soft_pairwise', soft_scores, temperatures))
        for rule, candidate_scores, temps in candidates:
            metrics = evaluator(val_user, val_y, candidate_scores)
            gate = rescue_harm_gate(anchor_scores, candidate_scores, val_y, groups)
            if len(selected) == 1 and rule == 'anchored_soft_pairwise':
                gate['passed'] = True
            record = {
                'target_member_count': target_count,
                'actual_member_count': len(selected),
                'rule': rule,
                'member_seeds': [final_runs[i]['seed'] for i in selected],
                'anchor_seed': final_runs[anchor_index]['seed'],
                'temperatures': temps,
                'gauc': metric_value(metrics, 'GAUC', 'gauc'),
                'ndcg5': metric_value(metrics, 'nDCG@5', 'ndcg5'),
                'primary': metric_value(metrics, 'primary'),
                'gate': gate
            }
            ensemble_history.append(record)
            with open(progress_path, 'a') as fh:
                fh.write(json.dumps({'stage': 'ensemble_probe', **record}, sort_keys=True) + '\n')
            if gate['passed']:
                passing.append((record['primary'], rule == 'anchored_soft_pairwise', candidate_scores, record))

    if passing:
        passing.sort(key=lambda x: (x[0], x[1]), reverse=True)
        _, _, blended_scores, selected_ensemble = passing[0]
    else:
        blended_scores = anchor_scores
        anchor_metrics = evaluator(val_user, val_y, blended_scores)
        selected_ensemble = {
            'target_member_count': 1,
            'actual_member_count': 1,
            'rule': 'anchor_fallback',
            'member_seeds': [final_runs[anchor_index]['seed']],
            'anchor_seed': final_runs[anchor_index]['seed'],
            'temperatures': None,
            'gauc': metric_value(anchor_metrics, 'GAUC', 'gauc'),
            'ndcg5': metric_value(anchor_metrics, 'nDCG@5', 'ndcg5'),
            'primary': metric_value(anchor_metrics, 'primary'),
            'gate': {'passed': True, 'reason': 'no multi-member candidate passed rescue/harm gate'}
        }

    final_metrics = evaluator(val_user, val_y, blended_scores)
    output_metrics = {
        'gauc': metric_value(final_metrics, 'GAUC', 'gauc'),
        'ndcg5': metric_value(final_metrics, 'nDCG@5', 'ndcg5'),
        'primary': metric_value(final_metrics, 'primary'),
        'best_config': serializable_config(best_config),
        'history': history,
        'final_runs': final_runs,
        'ensemble_gate': {
            'median_member_primary': median_primary,
            'median_tie_rate': median_tie,
            'tie_limit': tie_limit,
            'eligible_member_seeds': [final_runs[i]['seed'] for i in eligible],
            'dropped_member_seeds': [final_runs[i]['seed'] for i in range(len(final_runs)) if i not in eligible]
        },
        'ensemble_history': ensemble_history,
        'selected_ensemble': selected_ensemble
    }
    with open(os.path.join(args.out_dir, 'metrics.json'), 'w') as fh:
        json.dump(output_metrics, fh)

    video = val.get('video', np.zeros(len(val_y), dtype=np.int64))
    with open(os.path.join(args.out_dir, 'predictions.csv'), 'w') as fh:
        fh.write('row_id,user_id,video_id,score\n')
        for i, score in enumerate(blended_scores):
            fh.write(f'{i},{val_user[i]},{video[i]},{score:.9g}\n')


if __name__ == '__main__':
    main()
