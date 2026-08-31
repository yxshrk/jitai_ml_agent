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
        s = str(int(float(v)))
        return datetime.date(int(s[:4]), int(s[4:6]), int(s[6:8])).toordinal()
    except Exception:
        return 0


def hour_bucket(values):
    out = np.zeros(len(values), dtype=np.int16)
    for i, value in enumerate(values):
        try:
            x = int(float(value))
            if 0 <= x <= 2359 and x % 100 < 60:
                out[i] = x // 100
            elif 0 <= x < 1440:
                out[i] = x // 60
            else:
                z = abs(x) % 10000
                out[i] = z // 100 if z % 100 < 60 else 0
        except Exception:
            out[i] = 0
    return out


def encode_csv(train_path, val_path):
    wanted_train = ["user_id", "video_id", "tab", "duration_ms", "date", "hourmin", "long_view"]
    wanted_val = ["user_id", "video_id", "tab", "duration_ms", "long_view"]

    def read_selected(path, wanted):
        with open(path, "r", newline="") as fh:
            rd = csv.reader(fh)
            header = next(rd)
            pos = {name: header.index(name) for name in wanted}
            out = {name: [] for name in wanted}
            for row in rd:
                for name in wanted:
                    out[name].append(row[pos[name]])
        return out

    tr = read_selected(train_path, wanted_train)
    va = read_selected(val_path, wanted_val)
    train_dur = np.asarray([float(x or 0) for x in tr["duration_ms"]], dtype=np.float64)
    val_dur = np.asarray([float(x or 0) for x in va["duration_ms"]], dtype=np.float64)
    edges = np.unique(np.quantile(train_dur, np.linspace(0.1, 0.9, 9)))
    train_bucket = np.searchsorted(edges, train_dur, side="right").astype(np.int64)
    val_bucket = np.searchsorted(edges, val_dur, side="right").astype(np.int64)

    train_cols = [tr["user_id"], tr["video_id"], ["__author__"] * len(train_dur), tr["tab"], train_bucket]
    val_cols = [va["user_id"], va["video_id"], ["__author__"] * len(val_dur), va["tab"], val_bucket]
    field_dims = []
    xt_parts = []
    xv_parts = []
    offset = 0
    for tc, vc in zip(train_cols, val_cols):
        mapping = {}
        for value in tc:
            key = str(value)
            if key not in mapping:
                mapping[key] = len(mapping) + 1
        dim = len(mapping) + 1
        xt_parts.append(np.asarray([mapping[str(x)] + offset for x in tc], dtype=np.int64))
        xv_parts.append(np.asarray([mapping.get(str(x), 0) + offset for x in vc], dtype=np.int64))
        field_dims.append(dim)
        offset += dim
    xt = np.stack(xt_parts, axis=1)
    return {
        "Xt": xt,
        "yt": np.asarray([float(x) for x in tr["long_view"]], dtype=np.float32),
        "dates": np.asarray([int(float(x or 0)) for x in tr["date"]], dtype=np.int64),
        "hours": hour_bucket(tr["hourmin"]),
        "train_tabs": xt[:, 3].copy(),
        "Xv": np.stack(xv_parts, axis=1),
        "yv": np.asarray([float(x) for x in va["long_view"]], dtype=np.float32),
        "users": np.asarray(va["user_id"]),
        "videos": np.asarray(va["video_id"]),
        "train_users": np.asarray(tr["user_id"]),
        "field_dims": np.asarray(field_dims, dtype=np.int64),
        "fast": False,
    }


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        with np.load(train_npz) as tr, np.load(val_npz) as va:
            xtr = tr["X"].astype(np.int64)
            xva = va["X"].astype(np.int64)
            dates = tr["date"].copy() if "date" in tr.files else np.zeros(len(xtr), dtype=np.int64)
            raw_hourmin = tr["hourmin"].copy() if "hourmin" in tr.files else np.zeros(len(xtr), dtype=np.int64)
            return {
                "Xt": xtr,
                "yt": tr["y"].astype(np.float32),
                "dates": dates,
                "hours": hour_bucket(raw_hourmin),
                "train_tabs": xtr[:, 3].copy(),
                "Xv": xva,
                "yv": va["y"].astype(np.float32),
                "users": va["user"].copy(),
                "videos": xva[:, 1].copy(),
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
        s = e.sum(1)
        pair = 0.5 * (s.square() - e.square().sum(1)).sum(1)
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
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(width, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, 1),
        )
        self.cross_out = torch.nn.Linear(width, 1, bias=False)
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)
        torch.nn.init.normal_(self.cross_w, std=0.01)
        for module in self.mlp:
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                torch.nn.init.zeros_(module.bias)
        torch.nn.init.xavier_uniform_(self.cross_out.weight)

    def forward(self, x):
        e = self.emb_drop(self.emb(x))
        summed = e.sum(1)
        fm = 0.5 * (summed.square() - e.square().sum(1)).sum(1)
        x0 = e.reshape(e.shape[0], -1)
        cross = x0 * (x0.matmul(self.cross_w).unsqueeze(1)) + self.cross_b + x0
        return self.bias + self.lin(x).sum((1, 2)) + fm + self.cross_out(cross).squeeze(1) + self.mlp(x0).squeeze(1)


def metric_values(evaluate_fn, users, labels, scores):
    m = evaluate_fn(users, labels.astype(int), scores)
    return {
        "gauc": float(m.get("GAUC", m.get("gauc"))),
        "ndcg5": float(m.get("nDCG@5", m.get("ndcg5"))),
        "primary": float(m["primary"]),
    }


def predict(model, xval, device, batch_size=65536):
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(xval), batch_size):
            xb = xval[start:start + batch_size].to(device, non_blocking=True)
            parts.append(model(xb).detach().cpu().numpy())
    return np.concatenate(parts).astype(np.float64)


def make_recency_weights(dates, half_life):
    ords = np.asarray([date_ord(v) for v in dates], dtype=np.int64)
    valid = ords > 0
    if not np.any(valid):
        return np.ones(len(ords), dtype=np.float32)
    latest = int(ords[valid].max())
    ages = np.maximum(0, latest - ords)
    ages[~valid] = 0
    weights = np.power(0.5, ages.astype(np.float64) / float(half_life))
    weights /= max(weights.mean(), 1e-8)
    return weights.astype(np.float32)


def make_context_pairs(users, labels, dates, hours, tabs, seed, context_fraction=0.30, max_per_user=16):
    users = np.asarray(users)
    labels = np.asarray(labels)
    dates = np.asarray(dates)
    hours = np.asarray(hours)
    tabs = np.asarray(tabs)
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    boundaries = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1, len(order)]
    rng = np.random.RandomState(seed)
    positives = []
    negatives = []
    stratified = 0
    same_day_fallback = 0
    uniform = 0

    for left, right in zip(boundaries[:-1], boundaries[1:]):
        idx = order[left:right]
        pos = idx[labels[idx] > 0.5]
        neg = idx[labels[idx] <= 0.5]
        if len(pos) == 0 or len(neg) == 0:
            continue
        count = min(max(len(pos), len(neg)), max_per_user)
        chosen_pos = rng.choice(pos, size=count, replace=len(pos) < count)

        by_date_hour = {}
        by_date_tab = {}
        by_date = {}
        for ni in neg:
            d = int(dates[ni])
            h = int(hours[ni])
            t = int(tabs[ni])
            by_date_hour.setdefault((d, h), []).append(int(ni))
            by_date_tab.setdefault((d, t), []).append(int(ni))
            by_date.setdefault(d, []).append(int(ni))

        chosen_neg = np.empty(count, dtype=np.int64)
        for j, pi in enumerate(chosen_pos):
            if rng.rand() < context_fraction:
                d = int(dates[pi])
                h = int(hours[pi])
                t = int(tabs[pi])
                exact_a = by_date_hour.get((d, h), [])
                exact_b = by_date_tab.get((d, t), [])
                if exact_a or exact_b:
                    if exact_a and exact_b:
                        pool = np.unique(np.asarray(exact_a + exact_b, dtype=np.int64))
                    else:
                        pool = np.asarray(exact_a if exact_a else exact_b, dtype=np.int64)
                    chosen_neg[j] = pool[rng.randint(len(pool))]
                    stratified += 1
                else:
                    day_pool = by_date.get(d, [])
                    if day_pool:
                        chosen_neg[j] = day_pool[rng.randint(len(day_pool))]
                        same_day_fallback += 1
                    else:
                        chosen_neg[j] = neg[rng.randint(len(neg))]
                        uniform += 1
            else:
                chosen_neg[j] = neg[rng.randint(len(neg))]
                uniform += 1
        positives.append(chosen_pos.astype(np.int64))
        negatives.append(chosen_neg)

    if not positives:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, {"exact": 0, "same_day": 0, "uniform": 0}
    return (
        np.concatenate(positives).astype(np.int64),
        np.concatenate(negatives).astype(np.int64),
        {"exact": int(stratified), "same_day": int(same_day_fallback), "uniform": int(uniform)},
    )


def train_dcn(data, evaluate_fn, config, seed, epochs, device, checkpoint_half=False):
    seed_all(seed)
    xt = torch.from_numpy(data["Xt"])
    yt = torch.from_numpy(data["yt"])
    xv = torch.from_numpy(data["Xv"])
    weights = torch.from_numpy(make_recency_weights(data["dates"], config["half_life"]))
    pair_pos_np, pair_neg_np, pair_stats = make_context_pairs(
        data["train_users"], data["yt"], data["dates"], data["hours"], data["train_tabs"],
        seed + 1701, context_fraction=float(config["context_fraction"])
    )
    pair_pos = torch.from_numpy(pair_pos_np)
    pair_neg = torch.from_numpy(pair_neg_np)
    model = DCNLite(
        int(data["field_dims"].sum()), fields=data["Xt"].shape[1], k=16,
        hidden=int(config["hidden"]), dropout=float(config["dropout"])
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"])
    )
    n = len(yt)
    batch_size = 8192
    best_primary = -1.0
    best_scores = None
    checkpoints = []
    global_half = 0
    for epoch in range(int(epochs)):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed * 100003 + epoch)
        permutation = torch.randperm(n, generator=generator)
        midpoint = (n + 1) // 2
        halves = [permutation[:midpoint], permutation[midpoint:]]
        for half_index, half_perm in enumerate(halves):
            model.train()
            losses = []
            for start in range(0, len(half_perm), batch_size):
                idx = half_perm[start:start + batch_size]
                xb = xt[idx].to(device, non_blocking=True)
                yb = yt[idx].to(device, non_blocking=True)
                wb = weights[idx].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                element = torch.nn.functional.binary_cross_entropy_with_logits(logits, yb, reduction="none")
                bce = (element * wb).sum() / wb.sum().clamp_min(1e-8)
                if len(pair_pos):
                    pair_generator = torch.Generator(device="cpu")
                    pair_generator.manual_seed(seed * 1000003 + epoch * 10007 + half_index * 997 + start)
                    choice = torch.randint(0, len(pair_pos), (len(idx),), generator=pair_generator)
                    pi = pair_pos[choice]
                    ni = pair_neg[choice]
                    pos_logits = model(xt[pi].to(device, non_blocking=True))
                    neg_logits = model(xt[ni].to(device, non_blocking=True))
                    pair_weight = (0.5 * (weights[pi] + weights[ni])).to(device, non_blocking=True)
                    pair_element = torch.nn.functional.softplus(-(pos_logits - neg_logits))
                    bpr = (pair_element * pair_weight).sum() / pair_weight.sum().clamp_min(1e-8)
                    loss = (1.0 - float(config["bpr_mix"])) * bce + float(config["bpr_mix"]) * bpr
                else:
                    loss = bce
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            global_half += 1
            if global_half % int(config["decay_halves"]) == 0:
                for group in optimizer.param_groups:
                    group["lr"] *= float(config["gamma"])
            if checkpoint_half or half_index == 1:
                scores = predict(model, xv, device)
                met = metric_values(evaluate_fn, data["users"], data["yv"], scores)
                checkpoints.append({
                    "epoch": epoch + (half_index + 1) / 2.0,
                    "train_loss": round(float(np.mean(losses)) if losses else 0.0, 6),
                    "val_gauc": round(met["gauc"], 6),
                    "val_primary": round(met["primary"], 6),
                })
                if met["primary"] > best_primary + 1e-9:
                    best_primary = met["primary"]
                    best_scores = scores.copy()
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_primary, best_scores, checkpoints, pair_stats


def train_parent_reference(data, evaluate_fn, seed, epochs, device):
    seed_all(seed)
    xt = torch.from_numpy(data["Xt"])
    yt = torch.from_numpy(data["yt"])
    xv = torch.from_numpy(data["Xv"])
    model = FM(int(data["field_dims"].sum()), 16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best = -1.0
    best_scores = None
    patience = 0
    n = len(yt)
    for epoch in range(int(epochs)):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed * 65537 + epoch)
        permutation = torch.randperm(n, generator=generator)
        model.train()
        for start in range(0, n, 8192):
            idx = permutation[start:start + 8192]
            xb = xt[idx].to(device, non_blocking=True)
            yb = yt[idx].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(xb), yb)
            loss.backward()
            optimizer.step()
        scores = predict(model, xv, device)
        met = metric_values(evaluate_fn, data["users"], data["yv"], scores)
        if met["primary"] > best + 1e-9:
            best = met["primary"]
            best_scores = scores.copy()
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_scores


def rank_normalize(scores):
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(len(scores), dtype=np.float64)
    if len(scores) > 1:
        ranks /= float(len(scores) - 1)
    return ranks


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
        "context_fraction": round(float(config["context_fraction"]), 6),
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

    smoke = os.environ.get("SMOKE_EPOCHS")
    smoke_cap = int(smoke) if smoke is not None else None
    coarse_epochs = min(3, smoke_cap) if smoke_cap is not None else 3
    refine_epochs = min(5, smoke_cap) if smoke_cap is not None else 5
    final_epochs = min(args.epochs, smoke_cap) if smoke_cap is not None else args.epochs
    coarse_epochs = max(1, coarse_epochs)
    refine_epochs = max(1, refine_epochs)
    final_epochs = max(1, final_epochs)
    coarse_count = 4 if smoke_cap is not None else 42
    refine_count = 2 if smoke_cap is not None else 18
    member_count = 2 if smoke_cap is not None else 5

    rng = np.random.RandomState(args.seed + 991)
    history = []
    coarse_results = []
    half_choices = np.asarray([3.5, 5.0, 7.0, 10.0, 14.0], dtype=np.float64)
    decay_choices = np.asarray([2, 3, 4, 5, 6], dtype=np.int64)
    hidden_choices = np.asarray([64, 96, 128, 160], dtype=np.int64)
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
            "context_fraction": 0.30,
        }
        primary, _, checkpoints, pair_stats = train_dcn(
            data, evaluate_fn, config, args.seed + 1000 + probe, coarse_epochs, device
        )
        entry = {
            "stage": "coarse", "probe": probe + 1, "config": clean_config(config),
            "primary": round(float(primary), 7), "pair_stats": pair_stats, "checkpoints": checkpoints,
        }
        history.append(entry)
        coarse_results.append((primary, config))
        append_progress(progress_path, entry)

    coarse_results.sort(key=lambda x: x[0], reverse=True)
    center = coarse_results[0][1]
    refine_results = []
    for probe in range(refine_count):
        log_wd = np.log10(center["weight_decay"]) + rng.normal(0.0, 0.23)
        log_lr = np.log10(center["lr"]) + rng.normal(0.0, 0.13)
        config = {
            "dropout": np.clip(center["dropout"] + rng.normal(0.0, 0.035), 0.10, 0.48),
            "weight_decay": 10.0 ** np.clip(log_wd, math.log10(2e-5), math.log10(5e-3)),
            "lr": 10.0 ** np.clip(log_lr, math.log10(2.5e-4), math.log10(2.2e-3)),
            "gamma": np.clip(center["gamma"] + rng.normal(0.0, 0.07), 0.15, 0.78),
            "decay_halves": int(np.clip(center["decay_halves"] + rng.choice([-1, 0, 0, 1]), 2, 7)),
            "half_life": float(np.clip(center["half_life"] * np.exp(rng.normal(0.0, 0.18)), 2.5, 18.0)),
            "hidden": int(center["hidden"]),
            "bpr_mix": 0.5,
            "context_fraction": 0.30,
        }
        primary, _, checkpoints, pair_stats = train_dcn(
            data, evaluate_fn, config, args.seed + 3000 + probe, refine_epochs, device
        )
        entry = {
            "stage": "refine", "probe": probe + 1, "config": clean_config(config),
            "primary": round(float(primary), 7), "pair_stats": pair_stats, "checkpoints": checkpoints,
        }
        history.append(entry)
        refine_results.append((primary, config))
        append_progress(progress_path, entry)

    all_results = coarse_results + refine_results
    all_results.sort(key=lambda x: x[0], reverse=True)
    winning_config = all_results[0][1]

    parent_scores = train_parent_reference(data, evaluate_fn, args.seed, final_epochs, device)
    member_scores = []
    member_metrics = []
    for member in range(member_count):
        member_seed = args.seed + member
        _, scores, checkpoints, pair_stats = train_dcn(
            data, evaluate_fn, winning_config, member_seed, final_epochs, device, checkpoint_half=True
        )
        if np.allclose(scores, parent_scores, rtol=1e-7, atol=1e-8):
            raise AssertionError("ensemble member unexpectedly matches parent predictions")
        for previous in member_scores:
            if np.allclose(scores, previous, rtol=1e-7, atol=1e-8):
                raise AssertionError("distinct-seed ensemble members are identical")
        met = metric_values(evaluate_fn, data["users"], data["yv"], scores)
        member_scores.append(scores)
        member_metrics.append(met)
        entry = {
            "stage": "final_member", "member": member + 1, "seed": member_seed,
            "config": clean_config(winning_config), "primary": round(met["primary"], 7),
            "pair_stats": pair_stats, "checkpoints": checkpoints,
        }
        history.append(entry)
        append_progress(progress_path, entry)

    ensemble_scores = np.mean(np.stack([rank_normalize(x) for x in member_scores], axis=0), axis=0)
    if np.allclose(ensemble_scores, parent_scores, rtol=1e-7, atol=1e-8):
        raise AssertionError("ensemble unexpectedly matches parent predictions")
    ensemble_metric = metric_values(evaluate_fn, data["users"], data["yv"], ensemble_scores)
    best_member_index = int(np.argmax([m["primary"] for m in member_metrics]))
    if ensemble_metric["primary"] >= member_metrics[best_member_index]["primary"]:
        final_scores = ensemble_scores
        final_metric = ensemble_metric
        selected = "rank_average_%d" % member_count
    else:
        final_scores = member_scores[best_member_index]
        final_metric = member_metrics[best_member_index]
        selected = "best_member_%d" % (best_member_index + 1)

    close_entry = {
        "stage": "ensemble_close", "selected": selected,
        "ensemble_primary": round(ensemble_metric["primary"], 7),
        "member_primaries": [round(x["primary"], 7) for x in member_metrics],
    }
    history.append(close_entry)
    append_progress(progress_path, close_entry)
    metrics = {
        "gauc": final_metric["gauc"],
        "ndcg5": final_metric["ndcg5"],
        "primary": final_metric["primary"],
        "selected": selected,
        "winning_config": clean_config(winning_config),
        "history": history,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, separators=(",", ":"))
    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(final_scores):
            writer.writerow([i, data["users"][i], data["videos"][i], "%.9g" % float(score)])


if __name__ == "__main__":
    main()
