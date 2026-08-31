"""Gauge-fixed FM with a zero-initialized gated causal-session residual."""
import argparse
import csv
import datetime
import json
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FMWithSession(torch.nn.Module):
    def __init__(self, total_dim, context_dims, k=16, session=False,
                 context_k=4, hidden=32, gate_init=0.1):
        super().__init__()
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)
        self.session = bool(session)
        if self.session:
            self.context_embs = torch.nn.ModuleList([
                torch.nn.Embedding(int(dim), context_k) for dim in context_dims
            ])
            for emb in self.context_embs:
                torch.nn.init.normal_(emb.weight, std=0.01)
            input_dim = len(context_dims) * context_k
            self.session_tower = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden, 1),
            )
            torch.nn.init.zeros_(self.session_tower[-1].weight)
            torch.nn.init.zeros_(self.session_tower[-1].bias)
            p = min(max(float(gate_init), 1e-4), 1.0 - 1e-4)
            self.gate_logit = torch.nn.Parameter(torch.tensor(np.log(p / (1.0 - p)), dtype=torch.float32))

    def base_score(self, x):
        e = self.emb(x)
        s = e.sum(1)
        pair = 0.5 * (s * s - (e * e).sum(1)).sum(1)
        return self.bias + self.lin(x).sum((1, 2)) + pair

    def forward(self, x, context=None):
        base = self.base_score(x)
        if not self.session:
            return base, base
        z = torch.cat([emb(context[:, j]) for j, emb in enumerate(self.context_embs)], dim=1)
        residual = self.session_tower(z).squeeze(1)
        total = base + torch.sigmoid(self.gate_logit) * residual
        return total, base


def scalar_id(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def parse_hour(value):
    text = str(value).strip()
    if ":" in text:
        try:
            return max(0, min(23, int(text.split(":", 1)[0])))
        except ValueError:
            return 0
    try:
        number = int(float(text))
    except ValueError:
        return 0
    if 0 <= number <= 23:
        return number
    return max(0, min(23, number // 100))


def date_ordinal(value):
    text = str(value).strip()
    try:
        number = int(float(text))
        year = number // 10000
        month = (number // 100) % 100
        day = number % 100
        return datetime.date(year, month, day).toordinal()
    except (ValueError, OverflowError):
        try:
            return int(float(text))
        except ValueError:
            return 0


def load_csv_data(data_dir):
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    train_rows = []
    durations = []
    with open(train_path, "r", newline="") as fh:
        for row in csv.DictReader(fh):
            user = row["user_id"]
            video = row["video_id"]
            tab = row["tab"]
            duration = float(row["duration_ms"])
            train_rows.append((user, video, tab, duration, float(row["long_view"]),
                               row.get("hourmin", "0"), row.get("date", "0")))
            durations.append(duration)

    durations_np = np.asarray(durations, dtype=np.float64)
    if len(durations_np):
        edges = np.maximum.accumulate(np.quantile(durations_np, np.linspace(0.1, 0.9, 9)))
    else:
        edges = np.zeros(9, dtype=np.float64)
    user_values = sorted({r[0] for r in train_rows})
    video_values = sorted({r[1] for r in train_rows})
    tab_values = sorted({r[2] for r in train_rows})
    user_map = {v: i + 1 for i, v in enumerate(user_values)}
    video_map = {v: i + 1 for i, v in enumerate(video_values)}
    tab_map = {v: i + 1 for i, v in enumerate(tab_values)}
    field_dims = np.asarray([len(user_map) + 1, len(video_map) + 1, 1,
                             len(tab_map) + 1, 10], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1])).astype(np.int64)

    def encode(user, video, tab, duration):
        values = np.asarray([user_map.get(user, 0), video_map.get(video, 0), 0,
                             tab_map.get(tab, 0),
                             int(np.searchsorted(edges, duration, side="right"))], dtype=np.int64)
        return values + offsets

    Xt = np.empty((len(train_rows), 5), dtype=np.int64)
    yt = np.empty(len(train_rows), dtype=np.float32)
    train_users, train_hours, train_dates = [], [], []
    for i, row in enumerate(train_rows):
        user, video, tab, duration, label, hourmin, date = row
        Xt[i] = encode(user, video, tab, duration)
        yt[i] = label
        train_users.append(scalar_id(user))
        train_hours.append(hourmin)
        train_dates.append(date)

    val_features, val_labels, val_users, val_videos, val_hours, val_dates = [], [], [], [], [], []
    with open(val_path, "r", newline="") as fh:
        for row in csv.DictReader(fh):
            user = row["user_id"]
            video = row["video_id"]
            val_features.append(encode(user, video, row["tab"], float(row["duration_ms"])))
            val_labels.append(float(row["long_view"]))
            val_users.append(scalar_id(user))
            val_videos.append(scalar_id(video))
            val_hours.append(row.get("hourmin", "0"))
            val_dates.append(row.get("date", "0"))
    return {
        "Xt": Xt, "yt": yt, "train_users": np.asarray(train_users),
        "train_hourmin": np.asarray(train_hours), "train_date": np.asarray(train_dates),
        "Xv": np.asarray(val_features, dtype=np.int64).reshape(-1, 5),
        "yv": np.asarray(val_labels, dtype=np.float32), "val_users": np.asarray(val_users),
        "val_videos": np.asarray(val_videos), "val_hourmin": np.asarray(val_hours),
        "val_date": np.asarray(val_dates), "field_dims": field_dims, "npz": False,
    }


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        tr = np.load(train_npz)
        va = np.load(val_npz)
        field_dims = tr["field_dims"].astype(np.int64)
        val_videos = va["video"] if "video" in va.files else va["X"][:, 1].astype(np.int64) - int(field_dims[0])
        return {
            "Xt": tr["X"].astype(np.int64), "yt": tr["y"].astype(np.float32),
            "train_users": tr["user"], "train_hourmin": tr["hourmin"], "train_date": tr["date"],
            "Xv": va["X"].astype(np.int64), "yv": va["y"].astype(np.float32),
            "val_users": va["user"], "val_videos": val_videos,
            "val_hourmin": va["hourmin"], "val_date": va["date"],
            "field_dims": field_dims, "npz": True,
        }
    return load_csv_data(data_dir)


def gap_bucket(delta):
    delta = max(0, int(delta))
    if delta == 0:
        return 0
    if delta == 1:
        return 1
    if delta <= 3:
        return 2
    if delta <= 7:
        return 3
    if delta <= 23:
        return 4
    if delta <= 71:
        return 5
    if delta <= 167:
        return 6
    return 7


def position_bucket(position):
    if position <= 3:
        return int(position)
    if position <= 5:
        return 4
    if position <= 8:
        return 5
    return 6


def build_context(data):
    field_dims = data["field_dims"].astype(np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1])).astype(np.int64)
    tab_dim = int(field_dims[3])
    dur_dim = int(field_dims[4])
    state = {}

    def encode_split(users, hourmins, dates, x):
        out = np.empty((len(users), 5), dtype=np.int64)
        for i in range(len(users)):
            user = users[i].item() if isinstance(users[i], np.generic) else users[i]
            hour = parse_hour(hourmins[i])
            stamp = date_ordinal(dates[i]) * 24 + hour
            previous = state.get(user)
            if previous is None:
                gap = 7
                position = 0
            else:
                previous_stamp, previous_position = previous
                gap = gap_bucket(stamp - previous_stamp)
                position = previous_position + 1 if stamp == previous_stamp else 0
            state[user] = (stamp, position)
            pos = position_bucket(position)
            tab = int(x[i, 3] - offsets[3])
            dur = int(x[i, 4] - offsets[4])
            tab = min(max(tab, 0), tab_dim - 1)
            dur = min(max(dur, 0), dur_dim - 1)
            out[i, 0] = gap
            out[i, 1] = pos
            out[i, 2] = pos * tab_dim + tab
            out[i, 3] = pos * dur_dim + dur
            out[i, 4] = hour * tab_dim + tab
        return out

    ct = encode_split(data["train_users"], data["train_hourmin"], data["train_date"], data["Xt"])
    cv = encode_split(data["val_users"], data["val_hourmin"], data["val_date"], data["Xv"])
    dims = [8, 7, 7 * tab_dim, 7 * dur_dim, 24 * tab_dim]
    return ct, cv, dims


def make_user_groups(users):
    order = np.argsort(users, kind="stable")
    if len(order) == 0:
        return []
    sorted_users = users[order]
    boundaries = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
    return [x.astype(np.int64, copy=False) for x in np.split(order, boundaries)]


def centered_logits(logits, group_ids, group_count, global_bias):
    sums = torch.zeros(group_count, dtype=logits.dtype, device=logits.device)
    sums.scatter_add_(0, group_ids, logits)
    counts = torch.bincount(group_ids, minlength=group_count).to(logits.dtype)
    means = sums / counts.clamp_min(1.0)
    return logits - means[group_ids] + global_bias


def complete_slate_batches(groups, rng, target_size):
    pending, pending_size = [], 0
    for group_number in rng.permutation(len(groups)):
        group = groups[int(group_number)]
        if pending and pending_size + len(group) > target_size:
            yield pending
            pending, pending_size = [], 0
        pending.append(group)
        pending_size += len(group)
        if pending_size >= target_size:
            yield pending
            pending, pending_size = [], 0
    if pending:
        yield pending


def center_numpy_by_user(raw_scores, users, global_bias):
    order = np.argsort(users, kind="stable")
    result = np.empty_like(raw_scores)
    if len(order) == 0:
        return result
    sorted_users = users[order]
    boundaries = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1, len(order)]
    for j in range(len(boundaries) - 1):
        idx = order[boundaries[j]:boundaries[j + 1]]
        result[idx] = raw_scores[idx] - raw_scores[idx].mean() + global_bias
    return result


def pairwise_loss(base, labels, group_lengths, rng):
    differences = []
    start = 0
    for length in group_lengths:
        end = start + int(length)
        local_y = labels[start:end].detach().cpu().numpy()
        positives = np.flatnonzero(local_y > 0.5)
        negatives = np.flatnonzero(local_y <= 0.5)
        count = min(len(positives), len(negatives))
        if count:
            positives = rng.permutation(positives)[:count] + start
            negatives = rng.permutation(negatives)[:count] + start
            p = torch.as_tensor(positives, dtype=torch.long, device=base.device)
            n = torch.as_tensor(negatives, dtype=torch.long, device=base.device)
            differences.append(base[p] - base[n])
        start = end
    if not differences:
        return base.sum() * 0.0
    return torch.nn.functional.softplus(-torch.cat(differences)).mean()


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_run(config, run_seed, epochs, data, context_t, context_v, context_dims,
              groups, device, evaluate):
    seed_everything(run_seed)
    rng = np.random.default_rng(run_seed)
    model = FMWithSession(
        int(data["field_dims"].sum()), context_dims, k=16,
        session=config["session"], context_k=config["context_k"],
        hidden=config["hidden"], gate_init=config["gate_init"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = torch.nn.BCEWithLogitsLoss()
    Xt = torch.from_numpy(data["Xt"])
    yt = torch.from_numpy(data["yt"])
    Xv = torch.from_numpy(data["Xv"])
    Ct = torch.from_numpy(context_t)
    Cv = torch.from_numpy(context_v)
    best_primary = -1.0
    best_scores = None
    best_metrics = None
    best_epoch = 0
    patience = 0
    curve = []

    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        count_sum = 0
        for batch_groups in complete_slate_batches(groups, rng, 8192):
            idx_np = np.concatenate(batch_groups)
            lengths = [len(group) for group in batch_groups]
            gid_np = np.repeat(np.arange(len(batch_groups), dtype=np.int64), lengths)
            xb = Xt[idx_np].to(device)
            cb = Ct[idx_np].to(device)
            yb = yt[idx_np].to(device)
            gids = torch.from_numpy(gid_np).to(device)
            optimizer.zero_grad(set_to_none=True)
            total, base = model(xb, cb)
            fixed = centered_logits(total, gids, len(batch_groups), model.bias)
            loss = bce(fixed, yb)
            if config["pair_weight"] > 0.0:
                loss = loss + config["pair_weight"] * pairwise_loss(base, yb, lengths, rng)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(idx_np)
            count_sum += len(idx_np)

        model.eval()
        pieces = []
        with torch.no_grad():
            for start in range(0, len(Xv), 65536):
                xb = Xv[start:start + 65536].to(device)
                cb = Cv[start:start + 65536].to(device)
                total, _ = model(xb, cb)
                pieces.append(total.detach().cpu().numpy())
        raw = np.concatenate(pieces) if pieces else np.empty(0, dtype=np.float32)
        scores = center_numpy_by_user(raw, data["val_users"], float(model.bias.detach().cpu()))
        metrics = evaluate(data["val_users"], data["yv"].astype(int), scores)
        primary = float(metrics["primary"])
        curve.append({
            "epoch": epoch + 1,
            "train_loss": round(loss_sum / max(1, count_sum), 6),
            "gauc": round(float(metrics.get("GAUC", metrics.get("gauc", 0.0))), 7),
            "primary": round(primary, 7),
        })
        if primary > best_primary + 1e-6:
            best_primary = primary
            best_scores = scores.copy()
            best_metrics = metrics
            best_epoch = epoch + 1
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break
    return best_primary, best_scores, best_metrics, best_epoch, curve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()
    smoke_value = os.environ.get("SMOKE_EPOCHS")
    epochs = args.epochs if smoke_value is None else min(args.epochs, max(1, int(smoke_value)))

    seed_everything(args.seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_data(args.data_dir)
    if data["npz"]:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate

    context_t, context_v, context_dims = build_context(data)
    groups = make_user_groups(data["train_users"])
    os.makedirs(args.out_dir, exist_ok=True)
    progress_path = os.path.join(args.out_dir, "progress.log")

    configs = [
        {"name": "anchor", "session": False, "context_k": 4, "hidden": 16, "gate_init": 0.1, "pair_weight": 0.0},
        {"name": "session_g05_h16", "session": True, "context_k": 4, "hidden": 16, "gate_init": 0.05, "pair_weight": 0.0},
        {"name": "session_g10_h16", "session": True, "context_k": 4, "hidden": 16, "gate_init": 0.1, "pair_weight": 0.0},
        {"name": "session_g10_h32", "session": True, "context_k": 4, "hidden": 32, "gate_init": 0.1, "pair_weight": 0.0},
        {"name": "session_g20_h32", "session": True, "context_k": 4, "hidden": 32, "gate_init": 0.2, "pair_weight": 0.0},
        {"name": "session_pair10", "session": True, "context_k": 4, "hidden": 32, "gate_init": 0.1, "pair_weight": 0.1},
        {"name": "session_pair30", "session": True, "context_k": 4, "hidden": 32, "gate_init": 0.1, "pair_weight": 0.3},
        {"name": "session_pair50", "session": True, "context_k": 4, "hidden": 32, "gate_init": 0.1, "pair_weight": 0.5},
    ]
    if smoke_value is not None:
        configs = [configs[0], configs[3], configs[6]]
        probe_seeds = [args.seed + 100]
    else:
        probe_seeds = [args.seed + 100 + i for i in range(5)]

    history = []
    summaries = {}
    for config in configs:
        scores_for_config = []
        for probe_seed in probe_seeds:
            primary, _, metrics, best_epoch, curve = train_run(
                config, probe_seed, epochs, data, context_t, context_v,
                context_dims, groups, device, evaluate)
            entry = {
                "phase": "probe", "config": config, "seed": probe_seed,
                "best_epoch": best_epoch,
                "gauc": float(metrics.get("GAUC", metrics.get("gauc"))),
                "ndcg5": float(metrics.get("nDCG@5", metrics.get("ndcg5"))),
                "primary": float(primary), "curve": curve,
            }
            history.append(entry)
            scores_for_config.append(primary)
            with open(progress_path, "a") as fh:
                fh.write(json.dumps({"config": config["name"], "seed": probe_seed,
                                     "primary": float(primary)}) + "\n")
        summaries[config["name"]] = {
            "mean_primary": float(np.mean(scores_for_config)),
            "std_primary": float(np.std(scores_for_config)),
            "scores": [float(x) for x in scores_for_config],
        }

    winner = max(configs, key=lambda c: (summaries[c["name"]]["mean_primary"], -configs.index(c)))
    primary, best_scores, final_metrics, best_epoch, curve = train_run(
        winner, args.seed, epochs, data, context_t, context_v,
        context_dims, groups, device, evaluate)
    history.append({
        "phase": "final", "config": winner, "seed": args.seed,
        "best_epoch": best_epoch,
        "gauc": float(final_metrics.get("GAUC", final_metrics.get("gauc"))),
        "ndcg5": float(final_metrics.get("nDCG@5", final_metrics.get("ndcg5"))),
        "primary": float(primary), "curve": curve,
    })

    result = {
        "gauc": float(final_metrics.get("GAUC", final_metrics.get("gauc"))),
        "ndcg5": float(final_metrics.get("nDCG@5", final_metrics.get("ndcg5"))),
        "primary": float(final_metrics["primary"]),
        "selected_config": winner,
        "probe_summaries": summaries,
        "history": history,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(result, fh)
    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(best_scores):
            writer.writerow([i, data["val_users"][i], data["val_videos"][i], f"{float(score):.9g}"])


if __name__ == "__main__":
    main()
