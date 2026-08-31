import argparse
import csv
import datetime
import json
import math
import os
import sys

import numpy as np
import torch


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def date_ord(v):
    try:
        s = str(int(v))
        return datetime.date(int(s[:4]), int(s[4:6]), int(s[6:8])).toordinal()
    except Exception:
        return 0


def encode_csv(train_path, val_path):
    train_names = ["user_id", "video_id", "tab", "duration_ms", "date", "long_view"]
    val_names = ["user_id", "video_id", "tab", "duration_ms", "long_view"]

    def read(path, names):
        with open(path, "r", newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader)
            positions = {name: header.index(name) for name in names}
            result = {name: [] for name in names}
            for row in reader:
                for name in names:
                    result[name].append(row[positions[name]])
        return result

    tr = read(train_path, train_names)
    va = read(val_path, val_names)
    td = np.asarray([float(x or 0) for x in tr["duration_ms"]], dtype=np.float64)
    vd = np.asarray([float(x or 0) for x in va["duration_ms"]], dtype=np.float64)
    edges = np.unique(np.quantile(td, np.linspace(0.1, 0.9, 9)))
    tb = np.searchsorted(edges, td, side="right")
    vb = np.searchsorted(edges, vd, side="right")
    train_columns = [tr["user_id"], tr["video_id"], ["__author__"] * len(td), tr["tab"], tb]
    val_columns = [va["user_id"], va["video_id"], ["__author__"] * len(vd), va["tab"], vb]
    xt_parts, xv_parts, dims = [], [], []
    offset = 0
    for train_col, val_col in zip(train_columns, val_columns):
        mapping = {}
        for value in train_col:
            key = str(value)
            if key not in mapping:
                mapping[key] = len(mapping) + 1
        dim = len(mapping) + 1
        xt_parts.append(np.asarray([mapping[str(x)] + offset for x in train_col], dtype=np.int64))
        xv_parts.append(np.asarray([mapping.get(str(x), 0) + offset for x in val_col], dtype=np.int64))
        dims.append(dim)
        offset += dim
    return {
        "Xt": np.stack(xt_parts, axis=1),
        "yt": np.asarray(tr["long_view"], dtype=np.float32),
        "dates": np.asarray([int(float(x or 0)) for x in tr["date"]], dtype=np.int64),
        "Xv": np.stack(xv_parts, axis=1),
        "yv": np.asarray(va["long_view"], dtype=np.float32),
        "users": np.asarray(va["user_id"]),
        "videos": np.asarray(va["video_id"]),
        "train_users": np.asarray(tr["user_id"]),
        "field_dims": np.asarray(dims, dtype=np.int64),
        "fast": False,
    }


def load_data(data_dir):
    train_path = os.path.join(data_dir, "train.npz")
    val_path = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_path) and os.path.exists(val_path):
        with np.load(train_path) as tr, np.load(val_path) as va:
            xt = tr["X"].astype(np.int64)
            xv = va["X"].astype(np.int64)
            return {
                "Xt": xt,
                "yt": tr["y"].astype(np.float32),
                "dates": tr["date"].copy() if "date" in tr.files else np.zeros(len(xt), dtype=np.int64),
                "Xv": xv,
                "yv": va["y"].astype(np.float32),
                "users": va["user"].copy(),
                "videos": xv[:, 1].copy(),
                "train_users": tr["user"].copy(),
                "field_dims": tr["field_dims"].astype(np.int64),
                "fast": True,
            }
    return encode_csv(os.path.join(data_dir, "train.csv"), os.path.join(data_dir, "val.csv"))


class FM(torch.nn.Module):
    def __init__(self, total_dim, k=16):
        super().__init__()
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)

    def forward(self, x):
        e = self.emb(x)
        summed = e.sum(1)
        pair = 0.5 * (summed.square() - e.square().sum(1)).sum(1)
        return self.bias + self.lin(x).sum((1, 2)) + pair


class DCNLite(torch.nn.Module):
    def __init__(self, total_dim, fields=5, k=16, hidden=128, dropout=0.25):
        super().__init__()
        width = fields * k
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.emb_drop = torch.nn.Dropout(dropout)
        self.cross_w = torch.nn.Parameter(torch.empty(width))
        self.cross_b = torch.nn.Parameter(torch.zeros(width))
        self.cross_out = torch.nn.Linear(width, 1, bias=False)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(width, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, 1),
        )
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)
        torch.nn.init.normal_(self.cross_w, std=0.01)
        torch.nn.init.xavier_uniform_(self.cross_out.weight)
        for module in self.mlp:
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        e = self.emb_drop(self.emb(x))
        summed = e.sum(1)
        fm = 0.5 * (summed.square() - e.square().sum(1)).sum(1)
        x0 = e.reshape(e.shape[0], -1)
        cross = x0 * x0.matmul(self.cross_w).unsqueeze(1) + self.cross_b + x0
        return self.bias + self.lin(x).sum((1, 2)) + fm + self.cross_out(cross).squeeze(1) + self.mlp(x0).squeeze(1)


def metric_values(evaluate_fn, users, labels, scores):
    result = evaluate_fn(users, labels.astype(int), scores)
    return {
        "gauc": float(result.get("GAUC", result.get("gauc"))),
        "ndcg5": float(result.get("nDCG@5", result.get("ndcg5"))),
        "primary": float(result["primary"]),
    }


def predict(model, x, device, batch_size=65536):
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            parts.append(model(x[start:start + batch_size].to(device, non_blocking=True)).cpu().numpy())
    return np.concatenate(parts).astype(np.float64)


def recency_weights(dates, half_life):
    ords = np.asarray([date_ord(x) for x in dates], dtype=np.int64)
    valid = ords > 0
    if not np.any(valid):
        return np.ones(len(ords), dtype=np.float32)
    latest = int(ords[valid].max())
    ages = np.maximum(0, latest - ords)
    ages[~valid] = 0
    weights = np.power(0.5, ages.astype(np.float64) / float(half_life))
    weights /= max(float(weights.mean()), 1e-8)
    return weights.astype(np.float32)


def make_pairs(users, labels, seed, max_per_user=16):
    users = np.asarray(users)
    labels = np.asarray(labels)
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    boundaries = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1, len(order)]
    rng = np.random.RandomState(seed)
    positives, negatives = [], []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        idx = order[left:right]
        pos = idx[labels[idx] > 0.5]
        neg = idx[labels[idx] <= 0.5]
        if len(pos) == 0 or len(neg) == 0:
            continue
        count = min(max(len(pos), len(neg)), max_per_user)
        positives.append(rng.choice(pos, count, replace=len(pos) < count))
        negatives.append(rng.choice(neg, count, replace=len(neg) < count))
    if not positives:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(positives), np.concatenate(negatives)


def train_dcn(data, evaluate_fn, config, seed, epochs, device, checkpoint_half=False):
    seed_all(seed)
    xt = torch.from_numpy(data["Xt"])
    yt = torch.from_numpy(data["yt"])
    xv = torch.from_numpy(data["Xv"])
    weights = torch.from_numpy(recency_weights(data["dates"], config["half_life"]))
    pos, neg = make_pairs(data["train_users"], data["yt"], seed + 1701)
    pos, neg = torch.from_numpy(pos), torch.from_numpy(neg)
    model = DCNLite(int(data["field_dims"].sum()), data["Xt"].shape[1], 16,
                    int(config["hidden"]), float(config["dropout"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["lr"]),
                                  weight_decay=float(config["weight_decay"]))
    n = len(yt)
    best_primary, best_scores = -1.0, None
    checkpoints = []
    half_count = 0
    for epoch in range(int(epochs)):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed * 100003 + epoch)
        permutation = torch.randperm(n, generator=generator)
        midpoint = (n + 1) // 2
        for half_index, indices in enumerate((permutation[:midpoint], permutation[midpoint:])):
            model.train()
            losses = []
            for start in range(0, len(indices), 8192):
                idx = indices[start:start + 8192]
                xb = xt[idx].to(device, non_blocking=True)
                yb = yt[idx].to(device, non_blocking=True)
                wb = weights[idx].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                element = torch.nn.functional.binary_cross_entropy_with_logits(logits, yb, reduction="none")
                bce = (element * wb).sum() / wb.sum().clamp_min(1e-8)
                if len(pos):
                    pair_generator = torch.Generator(device="cpu")
                    pair_generator.manual_seed(seed * 1000003 + epoch * 10007 + half_index * 997 + start)
                    choice = torch.randint(0, len(pos), (len(idx),), generator=pair_generator)
                    pi, ni = pos[choice], neg[choice]
                    pl = model(xt[pi].to(device, non_blocking=True))
                    nl = model(xt[ni].to(device, non_blocking=True))
                    pw = (0.5 * (weights[pi] + weights[ni])).to(device, non_blocking=True)
                    pair_loss = (torch.nn.functional.softplus(-(pl - nl)) * pw).sum() / pw.sum().clamp_min(1e-8)
                    loss = (1.0 - float(config["bpr_mix"])) * bce + float(config["bpr_mix"]) * pair_loss
                else:
                    loss = bce
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            half_count += 1
            if half_count % int(config["decay_halves"]) == 0:
                for group in optimizer.param_groups:
                    group["lr"] *= float(config["gamma"])
            if checkpoint_half or half_index == 1:
                scores = predict(model, xv, device)
                metrics = metric_values(evaluate_fn, data["users"], data["yv"], scores)
                checkpoints.append({
                    "epoch": epoch + (half_index + 1) / 2.0,
                    "train_loss": round(float(np.mean(losses)) if losses else 0.0, 6),
                    "val_gauc": round(metrics["gauc"], 6),
                    "val_primary": round(metrics["primary"], 6),
                })
                if metrics["primary"] > best_primary + 1e-9:
                    best_primary = metrics["primary"]
                    best_scores = scores.copy()
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_primary, best_scores, checkpoints


def train_parent_reference(data, evaluate_fn, seed, epochs, device):
    seed_all(seed)
    xt, yt, xv = torch.from_numpy(data["Xt"]), torch.from_numpy(data["yt"]), torch.from_numpy(data["Xv"])
    model = FM(int(data["field_dims"].sum()), 16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best, best_scores, patience = -1.0, None, 0
    for epoch in range(int(epochs)):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed * 65537 + epoch)
        permutation = torch.randperm(len(yt), generator=generator)
        model.train()
        for start in range(0, len(yt), 8192):
            idx = permutation[start:start + 8192]
            xb, yb = xt[idx].to(device), yt[idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(xb), yb)
            loss.backward()
            optimizer.step()
        scores = predict(model, xv, device)
        primary = metric_values(evaluate_fn, data["users"], data["yv"], scores)["primary"]
        if primary > best + 1e-9:
            best, best_scores, patience = primary, scores.copy(), 0
        else:
            patience += 1
            if patience >= 2:
                break
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_scores


def average_rank(values):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n <= 1:
        return np.full(n, 0.5, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranked = np.empty(n, dtype=np.float64)
    left = 0
    while left < n:
        right = left + 1
        while right < n and sorted_values[right] == sorted_values[left]:
            right += 1
        ranked[order[left:right]] = 0.5 * (left + right - 1) / float(n - 1)
        left = right
    return ranked


def within_user_rank(users, scores):
    users = np.asarray(users)
    scores = np.asarray(scores, dtype=np.float64)
    result = np.empty(len(scores), dtype=np.float64)
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    boundaries = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1, len(order)]
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        idx = order[left:right]
        result[idx] = average_rank(scores[idx])
    return result


def signed_sketch(train_x, labels, dates, score_x, half_life, seed, dimension=64):
    train_users = train_x[:, 0].astype(np.int64)
    train_videos = train_x[:, 1].astype(np.int64)
    score_users = score_x[:, 0].astype(np.int64)
    score_videos = score_x[:, 1].astype(np.int64)
    all_users = np.unique(np.concatenate([train_users, score_users]))
    all_videos = np.unique(np.concatenate([train_videos, score_videos]))
    tu = np.searchsorted(all_users, train_users)
    tv = np.searchsorted(all_videos, train_videos)
    su = np.searchsorted(all_users, score_users)
    sv = np.searchsorted(all_videos, score_videos)
    weights = recency_weights(dates, half_life).astype(np.float64)
    user_weight = np.bincount(tu, weights=weights, minlength=len(all_users))
    user_sum = np.bincount(tu, weights=weights * labels.astype(np.float64), minlength=len(all_users))
    global_mean = float(np.sum(weights * labels) / max(np.sum(weights), 1e-12))
    means = np.full(len(all_users), global_mean, dtype=np.float64)
    valid = user_weight > 0
    means[valid] = user_sum[valid] / user_weight[valid]
    residual = np.sqrt(weights) * (labels.astype(np.float64) - means[tu])
    rng = np.random.RandomState(seed)
    hashes = rng.randint(0, 2, size=(len(all_users), dimension)).astype(np.float32)
    hashes = hashes * 2.0 - 1.0
    video_vectors = np.zeros((len(all_videos), dimension), dtype=np.float32)
    np.add.at(video_vectors, tv, residual[:, None].astype(np.float32) * hashes[tu])
    video_norm = np.linalg.norm(video_vectors, axis=1, keepdims=True)
    video_vectors /= np.maximum(video_norm, 1e-8)
    user_vectors = np.zeros((len(all_users), dimension), dtype=np.float32)
    np.add.at(user_vectors, tu, residual[:, None].astype(np.float32) * video_vectors[tv])
    user_norm = np.linalg.norm(user_vectors, axis=1, keepdims=True)
    user_vectors /= np.maximum(user_norm, 1e-8)
    return np.sum(user_vectors[su] * video_vectors[sv], axis=1).astype(np.float64)


def rolling_split(data):
    ords = np.asarray([date_ord(x) for x in data["dates"]], dtype=np.int64)
    valid_dates = np.unique(ords[ords > 0])
    if len(valid_dates) >= 5:
        cutoff = valid_dates[max(1, int(math.floor(0.8 * len(valid_dates))))]
        train_idx = np.flatnonzero((ords > 0) & (ords < cutoff))
        hold_idx = np.flatnonzero(ords >= cutoff)
    else:
        cut = max(1, int(0.85 * len(ords)))
        train_idx = np.arange(cut, dtype=np.int64)
        hold_idx = np.arange(cut, len(ords), dtype=np.int64)
    if len(train_idx) == 0 or len(hold_idx) == 0:
        cut = max(1, int(0.85 * len(ords)))
        train_idx = np.arange(cut, dtype=np.int64)
        hold_idx = np.arange(cut, len(ords), dtype=np.int64)
    rolling = {
        "Xt": data["Xt"][train_idx],
        "yt": data["yt"][train_idx],
        "dates": data["dates"][train_idx],
        "Xv": data["Xt"][hold_idx],
        "yv": data["yt"][hold_idx],
        "users": data["train_users"][hold_idx],
        "videos": data["Xt"][hold_idx, 1],
        "train_users": data["train_users"][train_idx],
        "field_dims": data["field_dims"],
        "fast": data["fast"],
    }
    return rolling, train_idx, hold_idx


def clean_config(config):
    return {
        "dropout": round(float(config["dropout"]), 6),
        "weight_decay": float(config["weight_decay"]),
        "lr": float(config["lr"]),
        "gamma": round(float(config["gamma"]), 6),
        "decay_halves": int(config["decay_halves"]),
        "half_life": round(float(config["half_life"]), 6),
        "hidden": int(config["hidden"]),
        "bpr_mix": round(float(config["bpr_mix"]), 6),
    }


def append_progress(path, entry):
    with open(path, "a") as fh:
        fh.write(json.dumps(entry, separators=(",", ":")) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    progress_path = os.path.join(args.out_dir, "progress.log")
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_data(args.data_dir)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    if data["fast"]:
        from data.official.evaluate import evaluate as evaluate_fn
    else:
        from harness.evaluate_provisional import evaluate as evaluate_fn

    smoke_value = os.environ.get("SMOKE_EPOCHS")
    smoke = int(smoke_value) if smoke_value is not None else None
    coarse_epochs = max(1, min(3, smoke) if smoke is not None else 3)
    refine_epochs = max(1, min(5, smoke) if smoke is not None else 5)
    final_epochs = max(1, min(args.epochs, smoke) if smoke is not None else args.epochs)
    coarse_count = 4 if smoke is not None else 42
    refine_count = 2 if smoke is not None else 18
    rng = np.random.RandomState(args.seed + 991)
    history, results = [], []
    half_choices = np.asarray([3.5, 5.0, 7.0, 10.0, 14.0])
    decay_choices = np.asarray([2, 3, 4, 5, 6])
    hidden_choices = np.asarray([64, 96, 128, 160])

    for probe in range(coarse_count):
        config = {
            "dropout": rng.uniform(0.14, 0.42),
            "weight_decay": 10.0 ** rng.uniform(math.log10(3e-5), math.log10(3e-3)),
            "lr": 10.0 ** rng.uniform(math.log10(3.5e-4), math.log10(1.8e-3)),
            "gamma": rng.uniform(0.22, 0.68),
            "decay_halves": int(rng.choice(decay_choices)),
            "half_life": float(rng.choice(half_choices)),
            "hidden": int(rng.choice(hidden_choices)),
            "bpr_mix": 0.5,
        }
        primary, _, checkpoints = train_dcn(data, evaluate_fn, config, args.seed + 1000 + probe,
                                             coarse_epochs, device)
        entry = {"stage": "coarse", "probe": probe + 1, "config": clean_config(config),
                 "primary": round(primary, 7), "checkpoints": checkpoints}
        history.append(entry)
        results.append((primary, config))
        append_progress(progress_path, entry)

    results.sort(key=lambda x: x[0], reverse=True)
    center = results[0][1]
    for probe in range(refine_count):
        config = {
            "dropout": float(np.clip(center["dropout"] + rng.normal(0, 0.035), 0.10, 0.48)),
            "weight_decay": 10.0 ** float(np.clip(np.log10(center["weight_decay"]) + rng.normal(0, 0.23), math.log10(2e-5), math.log10(5e-3))),
            "lr": 10.0 ** float(np.clip(np.log10(center["lr"]) + rng.normal(0, 0.13), math.log10(2.5e-4), math.log10(2.2e-3))),
            "gamma": float(np.clip(center["gamma"] + rng.normal(0, 0.07), 0.15, 0.78)),
            "decay_halves": int(np.clip(center["decay_halves"] + rng.choice([-1, 0, 0, 1]), 2, 7)),
            "half_life": float(np.clip(center["half_life"] * np.exp(rng.normal(0, 0.18)), 2.5, 18.0)),
            "hidden": int(center["hidden"]),
            "bpr_mix": 0.5,
        }
        primary, _, checkpoints = train_dcn(data, evaluate_fn, config, args.seed + 3000 + probe,
                                             refine_epochs, device)
        entry = {"stage": "refine", "probe": probe + 1, "config": clean_config(config),
                 "primary": round(primary, 7), "checkpoints": checkpoints}
        history.append(entry)
        results.append((primary, config))
        append_progress(progress_path, entry)

    results.sort(key=lambda x: x[0], reverse=True)
    winning_config = results[0][1]

    rolling, rolling_train_idx, rolling_hold_idx = rolling_split(data)
    _, rolling_champion, rolling_checkpoints = train_dcn(
        rolling, evaluate_fn, winning_config, args.seed + 7001, final_epochs, device, True
    )
    rolling_graph = signed_sketch(
        data["Xt"][rolling_train_idx], data["yt"][rolling_train_idx], data["dates"][rolling_train_idx],
        data["Xt"][rolling_hold_idx], winning_config["half_life"], args.seed + 8101, 64
    )
    champion_rank = within_user_rank(rolling["users"], rolling_champion)
    graph_rank = within_user_rank(rolling["users"], rolling_graph)
    alpha_results = []
    for alpha in (0.05, 0.1, 0.2):
        blended = champion_rank + alpha * graph_rank
        metrics = metric_values(evaluate_fn, rolling["users"], rolling["yv"], blended)
        alpha_results.append((metrics["primary"], alpha, metrics))
        entry = {"stage": "rolling_alpha", "alpha": alpha, "primary": round(metrics["primary"], 7)}
        history.append(entry)
        append_progress(progress_path, entry)
    alpha_results.sort(key=lambda x: (x[0], -x[1]), reverse=True)
    selected_alpha = float(alpha_results[0][1])
    history.append({"stage": "rolling_selection", "selected_alpha": selected_alpha,
                    "checkpoints": rolling_checkpoints})

    parent_scores = train_parent_reference(data, evaluate_fn, args.seed, final_epochs, device)
    member_scores, member_metrics = [], []
    for member in range(5):
        member_seed = args.seed + member
        _, scores, checkpoints = train_dcn(data, evaluate_fn, winning_config, member_seed,
                                            final_epochs, device, True)
        if np.allclose(scores, parent_scores, rtol=1e-7, atol=1e-8):
            raise AssertionError("ensemble member unexpectedly matches parent predictions")
        for previous in member_scores:
            if np.allclose(scores, previous, rtol=1e-7, atol=1e-8):
                raise AssertionError("distinct-seed ensemble members are identical")
        metrics = metric_values(evaluate_fn, data["users"], data["yv"], scores)
        member_scores.append(scores)
        member_metrics.append(metrics)
        entry = {"stage": "final_member", "member": member + 1, "seed": member_seed,
                 "primary": round(metrics["primary"], 7), "config": clean_config(winning_config),
                 "checkpoints": checkpoints}
        history.append(entry)
        append_progress(progress_path, entry)

    ensemble_scores = np.mean(np.stack([within_user_rank(data["users"], x) for x in member_scores]), axis=0)
    if np.allclose(ensemble_scores, parent_scores, rtol=1e-7, atol=1e-8):
        raise AssertionError("ensemble unexpectedly matches parent predictions")
    ensemble_metrics = metric_values(evaluate_fn, data["users"], data["yv"], ensemble_scores)
    best_member = int(np.argmax([x["primary"] for x in member_metrics]))
    if ensemble_metrics["primary"] >= member_metrics[best_member]["primary"]:
        champion_scores = ensemble_scores
        champion_name = "rank_average_5"
        champion_metrics = ensemble_metrics
    else:
        champion_scores = member_scores[best_member]
        champion_name = "best_member_%d" % (best_member + 1)
        champion_metrics = member_metrics[best_member]
    history.append({"stage": "champion_close", "selected": champion_name,
                    "champion_primary": round(champion_metrics["primary"], 7),
                    "ensemble_primary": round(ensemble_metrics["primary"], 7),
                    "member_primaries": [round(x["primary"], 7) for x in member_metrics]})

    graph_scores = signed_sketch(data["Xt"], data["yt"], data["dates"], data["Xv"],
                                 winning_config["half_life"], args.seed + 8101, 64)
    final_scores = within_user_rank(data["users"], champion_scores) + selected_alpha * within_user_rank(data["users"], graph_scores)
    if np.allclose(final_scores, champion_scores, rtol=1e-7, atol=1e-8):
        raise AssertionError("signed-sketch blend is a no-op")
    final_metrics = metric_values(evaluate_fn, data["users"], data["yv"], final_scores)
    history.append({"stage": "signed_sketch_final", "selected_alpha": selected_alpha,
                    "champion_primary": round(champion_metrics["primary"], 7),
                    "blended_primary": round(final_metrics["primary"], 7),
                    "graph_std": round(float(np.std(graph_scores)), 9)})

    metrics = {
        "gauc": final_metrics["gauc"],
        "ndcg5": final_metrics["ndcg5"],
        "primary": final_metrics["primary"],
        "selected": "signed_sketch_rank_blend",
        "selected_alpha": selected_alpha,
        "base_champion": champion_name,
        "winning_config": clean_config(winning_config),
        "history": history,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, separators=(",", ":"))
    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for index, score in enumerate(final_scores):
            writer.writerow([index, data["users"][index], data["videos"][index], "%.9g" % float(score)])


if __name__ == "__main__":
    main()
