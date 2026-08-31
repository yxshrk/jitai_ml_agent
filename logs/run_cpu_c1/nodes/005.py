"""Gated ensemble-design sweep over a causal Sequence DeepFM champion.

The underlying model and training objective are unchanged from the parent. The
closing procedure trains competent seed and mild dial-jitter members, retains
validation-competent checkpoints, probes rank, probability, and best-anchored
soft-vote combinations, and accepts only ensembles whose within-user pair
rescue/harm ratio exceeds 1.2 with positive net rescue.
"""
import argparse
import csv
import datetime
import json
import os
import sys
from collections import defaultdict, deque

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONTEXT_DIMS = [24, 7, 2, 9, 8]


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def metric_values(m):
    return (float(m.get("GAUC", m.get("gauc"))),
            float(m.get("nDCG@5", m.get("ndcg5"))),
            float(m["primary"]))


def parse_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def load_csv_fallback(data_dir):
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")

    def read_rows(path, training):
        result = []
        with open(path, "r", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                item = {
                    "user": row.get("user_id", "0"),
                    "video": row.get("video_id", "0"),
                    "tab": row.get("tab", "0"),
                    "hourmin": parse_int(row.get("hourmin", 0)),
                    "date": parse_int(row.get("date", 19700101), 19700101),
                    "duration": max(0, parse_int(row.get("duration_ms", 0))),
                    "y": float(row.get("long_view", 0.0)),
                }
                if training:
                    item["play"] = max(0, parse_int(row.get("play_time_ms", 0)))
                result.append(item)
        return result

    tr_rows = read_rows(train_path, True)
    va_rows = read_rows(val_path, False)
    durations = np.asarray([r["duration"] for r in tr_rows], dtype=np.float64)
    edges = (np.unique(np.quantile(durations, np.linspace(0.1, 0.9, 9)))
             if len(durations) else np.asarray([], dtype=np.float64))
    user_map = {v: i + 1 for i, v in enumerate(sorted({r["user"] for r in tr_rows}))}
    video_map = {v: i + 1 for i, v in enumerate(sorted({r["video"] for r in tr_rows}))}
    tab_map = {v: i + 1 for i, v in enumerate(sorted({r["tab"] for r in tr_rows}))}
    field_dims = np.asarray([len(user_map) + 1, len(video_map) + 1, 1,
                             len(tab_map) + 1, len(edges) + 2], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1])).astype(np.int64)

    def make(rows, training):
        n = len(rows)
        x = np.zeros((n, 5), dtype=np.int64)
        raw_users = np.empty(n, dtype=object)
        raw_videos = np.empty(n, dtype=object)
        for i, r in enumerate(rows):
            local = [user_map.get(r["user"], 0), video_map.get(r["video"], 0), 0,
                     tab_map.get(r["tab"], 0),
                     int(np.searchsorted(edges, r["duration"], side="right")) + 1]
            x[i] = np.asarray(local, dtype=np.int64) + offsets
            raw_users[i] = r["user"]
            raw_videos[i] = r["video"]
        out = {
            "X": x,
            "y": np.asarray([r["y"] for r in rows], dtype=np.float32),
            "user": raw_users,
            "video_out": raw_videos,
            "duration_ms": np.asarray([r["duration"] for r in rows], dtype=np.float32),
            "hourmin": np.asarray([r["hourmin"] for r in rows], dtype=np.int64),
            "date": np.asarray([r["date"] for r in rows], dtype=np.int64),
            "field_dims": field_dims,
        }
        if training:
            out["play_time_ms"] = np.asarray([r["play"] for r in rows], dtype=np.float32)
        return out

    return make(tr_rows, True), make(va_rows, False), False


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        tr_file = np.load(train_npz)
        va_file = np.load(val_npz)
        tr = {k: np.asarray(tr_file[k]) for k in tr_file.files}
        va = {k: np.asarray(va_file[k]) for k in va_file.files}
        tr_file.close()
        va_file.close()
        tr["video_out"] = np.zeros(len(tr["y"]), dtype=np.int64)
        va["video_out"] = np.zeros(len(va["y"]), dtype=np.int64)
        return tr, va, True
    return load_csv_fallback(data_dir)


def date_to_day(values):
    vals = np.asarray(values)
    output = np.zeros(len(vals), dtype=np.int64)
    cache = {}
    for i, raw in enumerate(vals):
        key = str(raw)
        if key not in cache:
            text = key.split(".")[0].replace("-", "")
            try:
                if len(text) >= 8:
                    d = datetime.date(int(text[:4]), int(text[4:6]), int(text[6:8]))
                    cache[key] = d.toordinal()
                else:
                    cache[key] = parse_int(raw)
            except (ValueError, TypeError):
                cache[key] = parse_int(raw)
        output[i] = cache[key]
    return output


def make_time_arrays(data):
    hm = np.asarray(data["hourmin"], dtype=np.int64)
    hour = np.clip(hm // 100, 0, 23).astype(np.int64)
    day = date_to_day(data["date"])
    weekday = np.mod(day, 7).astype(np.int64)
    minute = day * 1440 + np.clip(hm // 100, 0, 23) * 60 + np.clip(hm % 100, 0, 59)
    return hour, weekday, minute


def build_causal_features(tr, va):
    field_dims = np.asarray(tr["field_dims"], dtype=np.int64)
    official_total = int(field_dims.sum())
    official_offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1])).astype(np.int64)
    tab_local_train = np.asarray(tr["X"][:, 3], dtype=np.int64) - official_offsets[3]
    tab_local_val = np.asarray(va["X"][:, 3], dtype=np.int64) - official_offsets[3]
    tr_hour, tr_weekday, tr_minute = make_time_arrays(tr)
    va_hour, va_weekday, va_minute = make_time_arrays(va)
    histories = defaultdict(lambda: deque(maxlen=12))
    previous_time = {}
    session_position = {}

    def process(data, hour, weekday, minute, tab_local):
        n = len(data["y"])
        hist = np.full((n, 12), -1, dtype=np.int32)
        gap = np.zeros(n, dtype=np.int64)
        position = np.zeros(n, dtype=np.int64)
        order = np.lexsort((np.arange(n, dtype=np.int64), minute))
        users = np.asarray(data["user"])
        authors = np.asarray(data["X"][:, 2], dtype=np.int64)
        gap_edges = np.asarray([1, 5, 15, 30, 60, 180, 720], dtype=np.int64)
        for idx in order:
            u = users[idx].item() if isinstance(users[idx], np.generic) else users[idx]
            h = histories[u]
            if h:
                seq = list(h)
                hist[idx, -len(seq):] = seq
            now = int(minute[idx])
            if u in previous_time:
                delta = max(0, now - previous_time[u])
                gap[idx] = 1 + int(np.searchsorted(gap_edges, delta, side="right"))
                pos = session_position[u] + 1 if delta <= 30 else 0
            else:
                gap[idx] = 0
                pos = 0
            position[idx] = min(pos, 7)
            previous_time[u] = now
            session_position[u] = pos
            h.append(int(authors[idx]))
        is_rand = (tab_local == 1).astype(np.int64)
        context = np.column_stack((hour, weekday, is_rand, gap, position)).astype(np.int64)
        context_offsets = official_total + np.concatenate(([0], np.cumsum(CONTEXT_DIMS)[:-1]))
        context += context_offsets
        return np.concatenate((np.asarray(data["X"], dtype=np.int64), context), axis=1), hist

    tr_x, tr_h = process(tr, tr_hour, tr_weekday, tr_minute, tab_local_train)
    va_x, va_h = process(va, va_hour, va_weekday, va_minute, tab_local_val)
    total_categories = official_total + int(sum(CONTEXT_DIMS)) + 1
    pad_id = total_categories - 1
    tr_h[tr_h < 0] = pad_id
    va_h[va_h < 0] = pad_id
    return tr_x, tr_h.astype(np.int64), va_x, va_h.astype(np.int64), total_categories, pad_id


class SequenceDeepFM(torch.nn.Module):
    def __init__(self, total_categories, pad_id, embedding_dim=16, dropout=0.2):
        super().__init__()
        self.pad_id = int(pad_id)
        self.emb = torch.nn.Embedding(total_categories, embedding_dim, padding_idx=pad_id)
        self.linear = torch.nn.Embedding(total_categories, 1, padding_idx=pad_id)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.deep = torch.nn.Sequential(
            torch.nn.Linear(11 * embedding_dim, 128), torch.nn.ReLU(),
            torch.nn.Dropout(dropout), torch.nn.Linear(128, 64),
            torch.nn.ReLU(), torch.nn.Dropout(dropout))
        self.deep_out = torch.nn.Linear(64, 1)
        self.watch_head = torch.nn.Linear(64, 1)
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.linear.weight)
        with torch.no_grad():
            self.emb.weight[self.pad_id].zero_()
            self.linear.weight[self.pad_id].zero_()

    def forward(self, x, history):
        regular = self.emb(x)
        hist_emb = self.emb(history)
        mask = (history != self.pad_id).unsqueeze(-1)
        count = mask.sum(1).clamp_min(1)
        pooled = (hist_emb * mask).sum(1) / count
        fields = torch.cat((regular, pooled.unsqueeze(1)), dim=1)
        summed = fields.sum(1)
        fm = 0.5 * (summed.square() - fields.square().sum(1)).sum(1)
        hist_linear = (self.linear(history) * mask).sum((1, 2)) / count.squeeze(-1)
        linear = self.linear(x).sum((1, 2)) + hist_linear
        hidden = self.deep(fields.flatten(1))
        main = self.bias + linear + fm + self.deep_out(hidden).squeeze(1)
        return main, self.watch_head(hidden).squeeze(1)


def predict(model, x, history, device):
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(x), 65536):
            end = min(start + 65536, len(x))
            xb = torch.from_numpy(x[start:end]).to(device=device, dtype=torch.long)
            hb = torch.from_numpy(history[start:end]).to(device=device, dtype=torch.long)
            logits, _ = model(xb, hb)
            chunks.append(logits.detach().cpu().numpy())
    return np.concatenate(chunks).astype(np.float32)


def tie_rate(scores):
    if len(scores) < 2:
        return 0.0
    ordered = np.sort(np.asarray(scores, dtype=np.float32))
    return float(np.mean(ordered[1:] == ordered[:-1]))


def train_member(config, epochs, tr, va, tr_x, tr_h, va_x, va_h,
                 total_categories, pad_id, evaluator, device):
    member_seed = int(config["seed"])
    seed_everything(member_seed)
    model = SequenceDeepFM(total_categories, pad_id, dropout=float(config["dropout"])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["lr"]))
    bce = torch.nn.BCEWithLogitsLoss()
    n = len(tr["y"])
    batch_size = 8192
    y = np.asarray(tr["y"], dtype=np.float32)
    play = np.maximum(np.asarray(tr["play_time_ms"], dtype=np.float32), 0.0)
    duration = np.maximum(np.asarray(tr["duration_ms"], dtype=np.float32), 0.0)
    censored = (duration > 0) & (play >= duration)
    observed_log = np.log1p(np.where(censored, duration, play).astype(np.float64))
    center = float(observed_log.mean())
    scale = max(float(observed_log.std()), 1e-6)
    watch_target = ((observed_log - center) / scale).astype(np.float32)
    censor_float = censored.astype(np.float32)
    rng = np.random.RandomState(member_seed)
    patience = 0
    best_primary = -1.0
    epoch_candidates = []
    history_rows = []

    for epoch in range(epochs):
        model.train()
        permutation = rng.permutation(n)
        loss_sum = 0.0
        seen = 0
        for start in range(0, n, batch_size):
            ids = permutation[start:start + batch_size]
            xb = torch.from_numpy(tr_x[ids]).to(device=device, dtype=torch.long)
            hb = torch.from_numpy(tr_h[ids]).to(device=device, dtype=torch.long)
            yb = torch.from_numpy(y[ids]).to(device=device, dtype=torch.float32)
            wb = torch.from_numpy(watch_target[ids]).to(device=device, dtype=torch.float32)
            cb = torch.from_numpy(censor_float[ids]).to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            logits, watch_pred = model(xb, hb)
            main_loss = bce(logits, yb)
            ordinary = (watch_pred - wb).square()
            lower_bound = torch.relu(wb - watch_pred).square()
            aux_loss = ((1.0 - cb) * ordinary + cb * lower_bound).mean()
            loss = main_loss + float(config["aux_weight"]) * aux_loss
            loss.backward()
            optimizer.step()
            count = len(ids)
            loss_sum += float(loss.detach().cpu()) * count
            seen += count

        scores = predict(model, va_x, va_h, device)
        metrics = evaluator(va["user"], np.asarray(va["y"]).astype(int), scores)
        gauc, _, primary = metric_values(metrics)
        history_rows.append({
            "member_id": config["id"], "seed": member_seed, "epoch": epoch + 1,
            "train_loss": round(loss_sum / max(seen, 1), 6),
            "val_gauc": round(gauc, 6), "val_primary": round(primary, 6)})
        epoch_candidates.append({
            "id": config["id"] + "_e" + str(epoch + 1),
            "train_id": config["id"], "source": config["source"],
            "seed": member_seed, "epoch": epoch + 1, "primary": primary,
            "tie_rate": tie_rate(scores), "scores": scores})
        if primary > best_primary + 1e-7:
            best_primary = primary
            patience = 0
        else:
            patience += 1
            if patience >= 3:
                break

    epoch_candidates.sort(key=lambda z: (-z["primary"], z["epoch"]))
    return epoch_candidates[:2], history_rows


def make_user_groups(users):
    grouped = defaultdict(list)
    for i, raw in enumerate(np.asarray(users)):
        value = raw.item() if isinstance(raw, np.generic) else raw
        grouped[value].append(i)
    return [np.asarray(v, dtype=np.int64) for v in grouped.values()]


def sigmoid_array(x):
    x = np.clip(np.asarray(x, dtype=np.float64), -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-x))


def rank_average(members, groups):
    output = np.zeros(len(members[0]["scores"]), dtype=np.float64)
    for ids in groups:
        if len(ids) == 1:
            output[ids] = 0.5
            continue
        local = np.zeros(len(ids), dtype=np.float64)
        for member in members:
            values = np.asarray(member["scores"])[ids]
            order = np.argsort(values, kind="mergesort")
            ranks = np.empty(len(ids), dtype=np.float64)
            ranks[order] = np.arange(len(ids), dtype=np.float64)
            local += ranks / float(len(ids) - 1)
        output[ids] = local / float(len(members))
    return output


def soft_vote_transform(scores, groups):
    scores = np.asarray(scores, dtype=np.float64)
    output = np.zeros(len(scores), dtype=np.float64)
    scales = []
    for ids in groups:
        if len(ids) > 1:
            s = float(np.std(scores[ids]))
            if s > 1e-6:
                scales.append(s)
    temperature = max(float(np.median(scales)) if scales else float(np.std(scores)), 1e-3)
    for ids in groups:
        values = scores[ids]
        if len(ids) == 1:
            output[ids] = 0.5
        else:
            diffs = (values[:, None] - values[None, :]) / temperature
            votes = sigmoid_array(diffs)
            np.fill_diagonal(votes, 0.5)
            output[ids] = votes.mean(axis=1)
    return output


def rescue_harm(anchor_scores, candidate_scores, labels, groups):
    labels = np.asarray(labels).astype(np.int8)
    rescue = 0.0
    harm = 0.0
    useful_users = 0
    for ids in groups:
        pos = ids[labels[ids] == 1]
        neg = ids[labels[ids] == 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        useful_users += 1
        ad = np.asarray(anchor_scores)[pos, None] - np.asarray(anchor_scores)[neg][None, :]
        cd = np.asarray(candidate_scores)[pos, None] - np.asarray(candidate_scores)[neg][None, :]
        anchor_correct = ad > 0
        candidate_correct = cd > 0
        weight = 1.0 / float(len(pos) * len(neg))
        rescue += float(np.sum((~anchor_correct) & candidate_correct)) * weight
        harm += float(np.sum(anchor_correct & (~candidate_correct))) * weight
    denom = float(max(useful_users, 1))
    rescue /= denom
    harm /= denom
    ratio = rescue / max(harm, 1e-12)
    return rescue, harm, ratio


def competent_pool(candidates):
    if not candidates:
        return []
    median_primary = float(np.median([m["primary"] for m in candidates]))
    median_ties = float(np.median([m["tie_rate"] for m in candidates]))
    kept = [m for m in candidates
            if m["primary"] >= median_primary - 0.0010
            and m["tie_rate"] <= max(0.001, median_ties + 0.001)]
    kept.sort(key=lambda z: (-z["primary"], z["tie_rate"], z["id"]))
    return kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = args.epochs
    smoke_raw = os.environ.get("SMOKE_EPOCHS")
    smoke_mode = smoke_raw is not None
    if smoke_mode:
        epochs = min(epochs, max(1, int(smoke_raw)))

    tr, va, fast_path = load_data(args.data_dir)
    if fast_path:
        from data.official.evaluate import evaluate as evaluator
    else:
        from harness.evaluate_provisional import evaluate as evaluator
    tr_x, tr_h, va_x, va_h, total_categories, pad_id = build_causal_features(tr, va)
    progress_path = os.path.join(args.out_dir, "progress.log")

    runs_per_source = 2 if smoke_mode else 28
    configs = []
    for i in range(runs_per_source):
        configs.append({"id": "base_%02d" % i, "source": "base",
                        "seed": args.seed + i, "dropout": 0.2,
                        "lr": 1e-3, "aux_weight": 0.1})
    jitter_drop = [0.15, 0.20, 0.25, 0.20, 0.18, 0.22, 0.20]
    jitter_lr = [8e-4, 9e-4, 1e-3, 1.1e-3, 1.2e-3, 1e-3, 9e-4]
    jitter_aux = [0.07, 0.10, 0.13, 0.10, 0.08, 0.12, 0.10]
    for i in range(runs_per_source):
        j = i % len(jitter_drop)
        configs.append({"id": "jitter_%02d" % i, "source": "jitter",
                        "seed": args.seed + 1000 + i,
                        "dropout": jitter_drop[j], "lr": jitter_lr[j],
                        "aux_weight": jitter_aux[j]})

    all_snapshots = []
    train_history = []
    for config in configs:
        snapshots, rows = train_member(
            config, epochs, tr, va, tr_x, tr_h, va_x, va_h,
            total_categories, pad_id, evaluator, device)
        all_snapshots.extend(snapshots)
        train_history.extend(rows)
        best = snapshots[0]
        with open(progress_path, "a") as fh:
            fh.write(json.dumps({"probe": config, "best_epoch": best["epoch"],
                                 "best_primary": best["primary"]}) + "\n")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    best_by_train = {}
    for member in all_snapshots:
        old = best_by_train.get(member["train_id"])
        if old is None or member["primary"] > old["primary"]:
            best_by_train[member["train_id"]] = member
    base_best = [m for m in best_by_train.values() if m["source"] == "base"]
    jitter_best = [m for m in best_by_train.values() if m["source"] == "jitter"]
    pools = {
        "consecutive": competent_pool(base_best),
        "dial_jitter": competent_pool(jitter_best),
        "mixed": competent_pool(base_best + jitter_best),
        "snapshots": competent_pool(all_snapshots),
    }
    global_competent = competent_pool(base_best + jitter_best)
    if global_competent:
        anchor = global_competent[0]
    else:
        anchor = max(base_best + jitter_best, key=lambda z: z["primary"])
    groups = make_user_groups(va["user"])
    labels = np.asarray(va["y"]).astype(int)
    soft_cache = {}

    def soft_for(member):
        key = member["id"]
        if key not in soft_cache:
            soft_cache[key] = soft_vote_transform(member["scores"], groups)
        return soft_cache[key]

    parent_members = []
    for wanted_seed in (args.seed, args.seed + 1):
        choices = [m for m in base_best if m["seed"] == wanted_seed]
        if choices:
            parent_members.append(max(choices, key=lambda z: z["primary"]))
    if parent_members:
        parent_scores = np.mean(np.stack([m["scores"] for m in parent_members]), axis=0)
    else:
        parent_scores = np.asarray(anchor["scores"], dtype=np.float64)
    parent_metrics = evaluator(va["user"], labels, parent_scores)
    _, _, parent_primary = metric_values(parent_metrics)

    design_history = []
    accepted = []
    counts = [3, 5, 7]
    for source_name, pool in pools.items():
        for count in counts:
            if len(pool) < count:
                continue
            selected = pool[:count]
            probability_scores = np.mean(
                np.stack([sigmoid_array(m["scores"]) for m in selected]), axis=0)
            rank_scores = rank_average(selected, groups)
            member_soft = np.mean(np.stack([soft_for(m) for m in selected]), axis=0)
            anchored_scores = 0.6 * soft_for(anchor) + 0.4 * member_soft
            rules = [("probability_average", probability_scores),
                     ("per_user_rank_average", rank_scores),
                     ("best_anchored_soft_vote", anchored_scores)]
            for rule, scores in rules:
                metrics = evaluator(va["user"], labels, scores)
                gauc, ndcg5, primary = metric_values(metrics)
                rescue, harm, ratio = rescue_harm(anchor["scores"], scores, labels, groups)
                gate = bool(rescue > harm and rescue > 0.0 and ratio > 1.2)
                record = {
                    "source": source_name, "member_count": count, "rule": rule,
                    "members": [m["id"] for m in selected],
                    "gauc": gauc, "ndcg5": ndcg5, "primary": primary,
                    "rescue": rescue, "harm": harm, "rescue_harm_ratio": ratio,
                    "gate_passed": gate}
                design_history.append(record)
                with open(progress_path, "a") as fh:
                    fh.write(json.dumps({"ensemble_probe": record}) + "\n")
                if gate:
                    accepted.append((primary, rule == "best_anchored_soft_vote",
                                     source_name, count, scores, record))

    eligible = [x for x in accepted if x[0] >= parent_primary]
    if eligible:
        eligible.sort(key=lambda x: (x[0], x[1]), reverse=True)
        chosen = eligible[0]
        final_scores = np.asarray(chosen[4], dtype=np.float64)
        chosen_record = chosen[5]
    else:
        final_scores = np.asarray(parent_scores, dtype=np.float64)
        chosen_record = {"source": "parent_fallback", "member_count": len(parent_members),
                         "rule": "mean_logit", "primary": parent_primary,
                         "gate_passed": True,
                         "members": [m["id"] for m in parent_members]}

    metrics = evaluator(va["user"], labels, final_scores)
    gauc, ndcg5, primary = metric_values(metrics)
    member_summaries = [{k: m[k] for k in
                         ("id", "train_id", "source", "seed", "epoch", "primary", "tie_rate")}
                        for m in all_snapshots]
    payload = {"gauc": gauc, "ndcg5": ndcg5, "primary": primary,
               "history": train_history, "members": member_summaries,
               "ensemble_history": design_history, "chosen_ensemble": chosen_record,
               "anchor": anchor["id"], "parent_close_primary": parent_primary,
               "competence_rule": "primary >= source median - 0.001 and non-anomalous ties",
               "rescue_gate": "positive net rescue and rescue/harm > 1.2"}
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(payload, fh)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(final_scores):
            user = va["user"][i]
            video = va["video_out"][i]
            if isinstance(user, np.generic):
                user = user.item()
            if isinstance(video, np.generic):
                video = video.item()
            writer.writerow([i, user, video, format(float(score), ".9g")])


if __name__ == "__main__":
    main()
