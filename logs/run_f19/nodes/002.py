"""Regularized DCN-lite with same-context stratified BPR negatives.

The accepted optimizer, architecture, recency weighting, hybrid objective, and
configuration search remain unchanged. For 30 percent of BPR pairs, negatives
are sampled from the positive's same user/day/hour or user/day/tab context,
falling back to same-day and then uniform within-user negatives.
"""
import argparse
import datetime
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.official.evaluate import evaluate


class DCNLite(torch.nn.Module):
    def __init__(self, total_dim, fields=5, k=16, hidden=96, cross_layers=1,
                 dropout=0.25):
        super().__init__()
        self.fields = fields
        self.k = k
        self.dropout = float(dropout)
        width = fields * k
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.cross_w = torch.nn.ParameterList([
            torch.nn.Parameter(torch.empty(width)) for _ in range(cross_layers)
        ])
        self.cross_b = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros(width)) for _ in range(cross_layers)
        ])
        self.deep = torch.nn.Sequential(
            torch.nn.Linear(width, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(hidden, hidden // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
        )
        self.out = torch.nn.Linear(width + hidden // 2, 1)
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)
        for w in self.cross_w:
            torch.nn.init.normal_(w, std=0.01)
        torch.nn.init.normal_(self.out.weight, std=0.01)
        torch.nn.init.zeros_(self.out.bias)

    def forward(self, x):
        raw = self.emb(x)
        e = F.dropout(raw, p=self.dropout, training=self.training)
        summed = e.sum(1)
        fm = 0.5 * (summed.square() - e.square().sum(1)).sum(1)
        linear = self.lin(x).sum((1, 2))
        x0 = e.reshape(e.shape[0], -1)
        xl = x0
        for w, b in zip(self.cross_w, self.cross_b):
            xl = x0 * (xl * w).sum(1, keepdim=True) + b + xl
        deep = self.deep(x0)
        nonlinear = self.out(torch.cat((xl, deep), dim=1)).squeeze(1)
        return self.bias + linear + fm + nonlinear


def seed_everything(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def date_ages(values):
    vals = np.asarray(values)
    unique = np.unique(vals)
    ordinal = {}
    try:
        for value in unique:
            text = str(int(value)).zfill(8)
            ordinal[value] = datetime.date(int(text[:4]), int(text[4:6]),
                                           int(text[6:8])).toordinal()
    except (TypeError, ValueError):
        ordinal = {value: rank for rank, value in enumerate(sorted(unique.tolist()))}
    newest = max(ordinal.values())
    return np.asarray([newest - ordinal[value] for value in vals], dtype=np.float32)


def build_pair_tables(users, labels, dates, hourmins, tabs):
    users = np.asarray(users)
    labels = np.asarray(labels) >= 0.5
    dates = np.asarray(dates)
    hourmins = np.asarray(hourmins).astype(np.int64, copy=False)
    hours = hourmins // 100
    tabs = np.asarray(tabs)

    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    cuts = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
    bounds = np.concatenate(([0], cuts, [len(order)]))

    positives = []
    uniform_chunks = []
    hour_chunks = []
    tab_chunks = []
    day_chunks = []
    uniform_starts = []
    uniform_counts = []
    hour_starts = []
    hour_counts = []
    tab_starts = []
    tab_counts = []
    day_starts = []
    day_counts = []
    uniform_cursor = 0
    hour_cursor = 0
    tab_cursor = 0
    day_cursor = 0

    def materialize(groups, chunks, cursor):
        metadata = {}
        for key, values in groups.items():
            arr = np.asarray(values, dtype=np.int64)
            metadata[key] = (cursor, len(arr))
            chunks.append(arr)
            cursor += len(arr)
        return metadata, cursor

    for left, right in zip(bounds[:-1], bounds[1:]):
        idx = order[left:right]
        pos = idx[labels[idx]]
        neg = idx[~labels[idx]]
        if not len(pos) or not len(neg):
            continue

        positives.append(pos.astype(np.int64, copy=False))
        uniform_chunks.append(neg.astype(np.int64, copy=False))
        uniform_starts.append(np.full(len(pos), uniform_cursor, dtype=np.int64))
        uniform_counts.append(np.full(len(pos), len(neg), dtype=np.int64))
        uniform_cursor += len(neg)

        hour_groups = {}
        tab_groups = {}
        day_groups = {}
        for row in neg:
            day = dates[row].item() if hasattr(dates[row], "item") else dates[row]
            hour = int(hours[row])
            tab = tabs[row].item() if hasattr(tabs[row], "item") else tabs[row]
            hour_groups.setdefault((day, hour), []).append(int(row))
            tab_groups.setdefault((day, tab), []).append(int(row))
            day_groups.setdefault(day, []).append(int(row))

        hour_meta, hour_cursor = materialize(
            hour_groups, hour_chunks, hour_cursor)
        tab_meta, tab_cursor = materialize(
            tab_groups, tab_chunks, tab_cursor)
        day_meta, day_cursor = materialize(
            day_groups, day_chunks, day_cursor)

        hs = np.zeros(len(pos), dtype=np.int64)
        hc = np.zeros(len(pos), dtype=np.int64)
        ts = np.zeros(len(pos), dtype=np.int64)
        tc = np.zeros(len(pos), dtype=np.int64)
        ds = np.zeros(len(pos), dtype=np.int64)
        dc = np.zeros(len(pos), dtype=np.int64)
        for j, row in enumerate(pos):
            day = dates[row].item() if hasattr(dates[row], "item") else dates[row]
            hour = int(hours[row])
            tab = tabs[row].item() if hasattr(tabs[row], "item") else tabs[row]
            if (day, hour) in hour_meta:
                hs[j], hc[j] = hour_meta[(day, hour)]
            if (day, tab) in tab_meta:
                ts[j], tc[j] = tab_meta[(day, tab)]
            if day in day_meta:
                ds[j], dc[j] = day_meta[day]
        hour_starts.append(hs)
        hour_counts.append(hc)
        tab_starts.append(ts)
        tab_counts.append(tc)
        day_starts.append(ds)
        day_counts.append(dc)

    names = (
        "positive", "uniform_pool", "uniform_start", "uniform_count",
        "hour_pool", "hour_start", "hour_count", "tab_pool", "tab_start",
        "tab_count", "day_pool", "day_start", "day_count")
    if not positives:
        return {name: np.empty(0, dtype=np.int64) for name in names}

    return {
        "positive": np.concatenate(positives),
        "uniform_pool": np.concatenate(uniform_chunks),
        "uniform_start": np.concatenate(uniform_starts),
        "uniform_count": np.concatenate(uniform_counts),
        "hour_pool": np.concatenate(hour_chunks),
        "hour_start": np.concatenate(hour_starts),
        "hour_count": np.concatenate(hour_counts),
        "tab_pool": np.concatenate(tab_chunks),
        "tab_start": np.concatenate(tab_starts),
        "tab_count": np.concatenate(tab_counts),
        "day_pool": np.concatenate(day_chunks),
        "day_start": np.concatenate(day_starts),
        "day_count": np.concatenate(day_counts),
    }


def metric_values(metric):
    return {
        "gauc": float(metric.get("GAUC", metric.get("gauc", 0.0))),
        "ndcg5": float(metric.get("nDCG@5", metric.get("ndcg5", 0.0))),
        "primary": float(metric["primary"]),
    }


def make_coarse_configs(seed):
    rng = np.random.default_rng(seed + 1701)
    count = 12
    dropouts = np.linspace(0.17, 0.39, count)[rng.permutation(count)]
    decays = np.geomspace(4.0e-5, 2.4e-3, count)[rng.permutation(count)]
    lrs = np.geomspace(4.8e-4, 1.35e-3, count)[rng.permutation(count)]
    gammas = np.linspace(0.36, 0.76, count)[rng.permutation(count)]
    half_lives = np.asarray([3.5, 7.0, 14.0] * 4)[rng.permutation(count)]
    steps = np.asarray([1, 2, 3, 2] * 3)[rng.permutation(count)]
    hidden = np.asarray([64, 96, 128, 96] * 3)[rng.permutation(count)]
    crosses = np.asarray([1, 1, 2, 1, 2, 1] * 2)[rng.permutation(count)]
    configs = []
    for i in range(count):
        configs.append({
            "dropout": float(dropouts[i]),
            "weight_decay": float(decays[i]),
            "lr": float(lrs[i]),
            "decay_gamma": float(gammas[i]),
            "decay_step": int(steps[i]),
            "half_life": float(half_lives[i]),
            "hidden": int(hidden[i]),
            "cross_layers": int(crosses[i]),
            "bpr_mix": 0.5,
            "context_rho": 0.3,
        })
    return configs


def make_refine_configs(base, seed):
    rng = np.random.default_rng(seed + 2903)
    configs = [dict(base)]
    hidden_choices = np.asarray([64, 80, 96, 112, 128])
    for _ in range(5):
        cfg = dict(base)
        cfg["dropout"] = float(np.clip(
            base["dropout"] + rng.normal(0.0, 0.035), 0.13, 0.43))
        cfg["weight_decay"] = float(np.clip(
            base["weight_decay"] * math.exp(rng.normal(0.0, 0.42)),
            2.5e-5, 3.2e-3))
        cfg["lr"] = float(np.clip(
            base["lr"] * math.exp(rng.normal(0.0, 0.18)), 3.5e-4, 1.7e-3))
        cfg["decay_gamma"] = float(np.clip(
            base["decay_gamma"] + rng.normal(0.0, 0.075), 0.28, 0.84))
        cfg["decay_step"] = int(np.clip(
            base["decay_step"] + rng.choice([-1, 0, 1]), 1, 4))
        cfg["half_life"] = float(np.clip(
            base["half_life"] * math.exp(rng.normal(0.0, 0.24)), 2.8, 18.0))
        nearest = int(np.argmin(np.abs(hidden_choices - base["hidden"])))
        shift = int(rng.choice([-1, 0, 1]))
        cfg["hidden"] = int(hidden_choices[np.clip(nearest + shift, 0,
                                                   len(hidden_choices) - 1)])
        cfg["cross_layers"] = int(np.clip(
            base["cross_layers"] + rng.choice([-1, 0, 1]), 1, 2))
        configs.append(cfg)
    return configs


def append_progress(path, record):
    with open(path, "a") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=14)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    progress_path = os.path.join(args.out_dir, "progress.log")
    if os.path.exists(progress_path):
        os.remove(progress_path)

    train_path = os.path.join(args.data_dir, "train.npz")
    val_path = os.path.join(args.data_dir, "val.npz")
    tr = np.load(train_path)
    va = np.load(val_path)

    field_dims = tr["field_dims"].astype(np.int64)
    total_dim = int(field_dims.sum())
    xt_np = tr["X"].astype(np.int64)
    yt_np = tr["y"].astype(np.float32)
    xv_np = va["X"].astype(np.int64)
    val_users = va["user"]
    val_labels = va["y"].astype(int)
    ages_np = date_ages(tr["date"])

    pair_np = build_pair_tables(
        tr["user"], yt_np, tr["date"], tr["hourmin"], xt_np[:, 3])

    Xt = torch.from_numpy(xt_np).to(device)
    yt = torch.from_numpy(yt_np).to(device)
    Xv = torch.from_numpy(xv_np).to(device)
    ages = torch.from_numpy(ages_np).to(device)
    pair = {key: torch.from_numpy(value).to(device)
            for key, value in pair_np.items()}

    smoke = os.environ.get("SMOKE_EPOCHS")
    smoke_cap = int(smoke) if smoke is not None else None
    coarse_epochs = min(3, smoke_cap) if smoke_cap is not None else 3
    refine_epochs = min(6, smoke_cap) if smoke_cap is not None else 6
    final_epochs = min(args.epochs, smoke_cap) if smoke_cap is not None else args.epochs
    coarse_epochs = max(1, coarse_epochs)
    refine_epochs = max(1, refine_epochs)
    final_epochs = max(1, final_epochs)

    n = len(yt_np)
    batch_size = 8192
    history = []

    pairing_stats = {
        "eligible_positives": int(len(pair_np["positive"])),
        "hour_context_available": float(
            np.mean(pair_np["hour_count"] > 0)) if len(pair_np["positive"]) else 0.0,
        "tab_context_available": float(
            np.mean(pair_np["tab_count"] > 0)) if len(pair_np["positive"]) else 0.0,
        "day_fallback_available": float(
            np.mean(pair_np["day_count"] > 0)) if len(pair_np["positive"]) else 0.0,
    }

    def predict(model):
        model.eval()
        chunks = []
        with torch.no_grad():
            for left in range(0, len(Xv), 65536):
                chunks.append(model(Xv[left:left + 65536]).detach().cpu().numpy())
        return np.concatenate(chunks)

    def overwrite_negative(neg_idx, mask, pool, starts, counts):
        positions = torch.nonzero(mask, as_tuple=False).flatten()
        if positions.numel() == 0:
            return
        local_counts = counts[positions]
        offsets = torch.floor(
            torch.rand(positions.numel(), device=device) *
            local_counts.to(torch.float32)).to(torch.long)
        neg_idx[positions] = pool[starts[positions] + offsets]

    def sample_pair_indices(sample_count, rho):
        selected = torch.randint(
            len(pair["positive"]), (sample_count,), device=device)
        pos_idx = pair["positive"][selected]

        uniform_counts = pair["uniform_count"][selected]
        uniform_offsets = torch.floor(
            torch.rand(sample_count, device=device) *
            uniform_counts.to(torch.float32)).to(torch.long)
        neg_idx = pair["uniform_pool"][
            pair["uniform_start"][selected] + uniform_offsets].clone()

        stratified = torch.rand(sample_count, device=device) < float(rho)
        hour_counts = pair["hour_count"][selected]
        tab_counts = pair["tab_count"][selected]
        day_counts = pair["day_count"][selected]
        hour_available = hour_counts > 0
        tab_available = tab_counts > 0
        context_coin = torch.rand(sample_count, device=device) < 0.5

        use_hour = stratified & hour_available & (
            ~tab_available | context_coin)
        use_tab = stratified & tab_available & (
            ~hour_available | ~context_coin)
        used_exact = use_hour | use_tab
        use_day = stratified & ~used_exact & (day_counts > 0)

        overwrite_negative(
            neg_idx, use_hour, pair["hour_pool"],
            pair["hour_start"][selected], hour_counts)
        overwrite_negative(
            neg_idx, use_tab, pair["tab_pool"],
            pair["tab_start"][selected], tab_counts)
        overwrite_negative(
            neg_idx, use_day, pair["day_pool"],
            pair["day_start"][selected], day_counts)
        return pos_idx, neg_idx

    def train_candidate(config, epochs, row_fraction, run_seed, stage,
                        probe_index, half_epoch_checks=False, keep_snapshot=False):
        seed_everything(run_seed)
        model = DCNLite(
            total_dim=total_dim,
            fields=xt_np.shape[1],
            k=16,
            hidden=int(config["hidden"]),
            cross_layers=int(config["cross_layers"]),
            dropout=float(config["dropout"]),
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(config["lr"]),
            weight_decay=float(config["weight_decay"]))
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(config["decay_step"]),
            gamma=float(config["decay_gamma"]))

        recency = torch.pow(torch.tensor(2.0, device=device),
                            -ages / float(config["half_life"]))
        recency = recency / recency.mean().clamp_min(1e-8)
        take = max(batch_size, min(n, int(round(n * row_fraction))))
        total_batches = int(math.ceil(take / batch_size))
        best_primary = -1.0
        best_scores = None
        best_metric = None
        best_event = 0.0
        best_state = None
        events = []

        for epoch in range(epochs):
            model.train()
            permutation = torch.randperm(n, device=device)[:take]
            midpoint = max(1, int(math.ceil(total_batches / 2.0)))
            checkpoints = {total_batches}
            if half_epoch_checks:
                checkpoints.add(midpoint)
            loss_sum = 0.0
            seen_batches = 0
            for batch_number, left in enumerate(range(0, take, batch_size), start=1):
                idx = permutation[left:left + batch_size]
                optimizer.zero_grad(set_to_none=True)
                logits = model(Xt[idx])
                point_loss = F.binary_cross_entropy_with_logits(
                    logits, yt[idx], reduction="none")
                bce_loss = (point_loss * recency[idx]).sum() / recency[idx].sum().clamp_min(1e-8)

                if len(pair["positive"]):
                    pos_idx, neg_idx = sample_pair_indices(
                        len(idx), float(config["context_rho"]))
                    pair_logits = model(torch.cat((Xt[pos_idx], Xt[neg_idx]), dim=0))
                    pos_logits = pair_logits[:len(idx)]
                    neg_logits = pair_logits[len(idx):]
                    pair_weights = 0.5 * (recency[pos_idx] + recency[neg_idx])
                    pair_loss = (F.softplus(-(pos_logits - neg_logits)) * pair_weights).sum()
                    pair_loss = pair_loss / pair_weights.sum().clamp_min(1e-8)
                    mix = float(config["bpr_mix"])
                    loss = (1.0 - mix) * bce_loss + mix * pair_loss
                else:
                    loss = bce_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                loss_sum += float(loss.detach().item())
                seen_batches += 1

                if batch_number in checkpoints:
                    scores = predict(model)
                    metric = metric_values(evaluate(val_users, val_labels, scores))
                    event_epoch = epoch + batch_number / total_batches
                    event = {
                        "checkpoint_epoch": round(float(event_epoch), 3),
                        "train_loss": round(loss_sum / max(1, seen_batches), 6),
                        "lr": float(optimizer.param_groups[0]["lr"]),
                        "gauc": metric["gauc"],
                        "ndcg5": metric["ndcg5"],
                        "primary": metric["primary"],
                    }
                    events.append(event)
                    if metric["primary"] > best_primary + 1e-9:
                        best_primary = metric["primary"]
                        best_scores = scores.copy()
                        best_metric = metric
                        best_event = event_epoch
                        if keep_snapshot:
                            best_state = {key: value.detach().cpu().clone()
                                          for key, value in model.state_dict().items()}
                    if batch_number != total_batches:
                        model.train()
            scheduler.step()

        record = {
            "stage": stage,
            "probe": int(probe_index),
            "seed": int(run_seed),
            "epochs": int(epochs),
            "row_fraction": float(row_fraction),
            "config": dict(config),
            "best_epoch": round(float(best_event), 3),
            "gauc": best_metric["gauc"],
            "ndcg5": best_metric["ndcg5"],
            "primary": best_metric["primary"],
            "checkpoints": events,
        }
        history.append(record)
        append_progress(progress_path, {
            "stage": stage,
            "probe": int(probe_index),
            "config": dict(config),
            "primary": best_metric["primary"],
        })
        del model, optimizer, scheduler, recency, best_state
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return best_primary, best_scores, best_metric, record

    coarse_configs = make_coarse_configs(args.seed)
    coarse_results = []
    for index, config in enumerate(coarse_configs):
        result = train_candidate(
            config=config,
            epochs=coarse_epochs,
            row_fraction=0.62,
            run_seed=args.seed + 101,
            stage="coarse",
            probe_index=index,
        )
        coarse_results.append((result[0], config))
    coarse_results.sort(key=lambda item: item[0], reverse=True)
    coarse_winner = dict(coarse_results[0][1])

    refine_configs = make_refine_configs(coarse_winner, args.seed)
    refine_results = []
    refine_seed = args.seed + 202
    for index, config in enumerate(refine_configs):
        result = train_candidate(
            config=config,
            epochs=refine_epochs,
            row_fraction=1.0,
            run_seed=refine_seed,
            stage="refine",
            probe_index=index,
        )
        refine_results.append((result[0], dict(config), result[3]))
    refine_results.sort(key=lambda item: item[0], reverse=True)
    winning_config = dict(refine_results[0][1])
    winning_refine_primary = float(refine_results[0][0])

    final_primary, final_scores, final_metric, final_record = train_candidate(
        config=winning_config,
        epochs=final_epochs,
        row_fraction=1.0,
        run_seed=refine_seed,
        stage="final",
        probe_index=0,
        half_epoch_checks=True,
        keep_snapshot=True,
    )

    metrics_payload = {
        "gauc": final_metric["gauc"],
        "ndcg5": final_metric["ndcg5"],
        "primary": final_metric["primary"],
        "winning_config": winning_config,
        "pairing_stats": pairing_stats,
        "coarse_winner_primary": float(coarse_results[0][0]),
        "winning_refine_primary": winning_refine_primary,
        "final_best_epoch": final_record["best_epoch"],
        "history": history,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics_payload, fh)

    video_offset = int(field_dims[0])
    video_ids = xv_np[:, 1] - video_offset
    with open(os.path.join(args.out_dir, "predictions.csv"), "w") as fh:
        fh.write("row_id,user_id,video_id,score\n")
        for row_id, score in enumerate(final_scores):
            fh.write(f"{row_id},{val_users[row_id]},{video_ids[row_id]},{score:.8g}\n")


if __name__ == "__main__":
    main()
