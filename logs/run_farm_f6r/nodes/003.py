"""Sequence DeepFM composite with complete-user, gauge-fixed BCE training."""
import argparse
import csv
import datetime
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def date_parts(values):
    out = np.zeros(len(values), dtype=np.int64)
    days = np.zeros(len(values), dtype=np.int64)
    cache = {}
    for i, value in enumerate(values):
        iv = int(value)
        if iv not in cache:
            text = str(iv)
            try:
                dt = datetime.datetime.strptime(text, "%Y%m%d").date()
                cache[iv] = (dt.weekday(), dt.toordinal())
            except ValueError:
                cache[iv] = (iv % 7, iv)
        out[i], days[i] = cache[iv]
    return out, days


def hour_and_minute(values):
    values = np.asarray(values, dtype=np.int64)
    if len(values) == 0:
        return values.copy(), values.copy()
    if int(np.max(values)) > 1439:
        hour = np.clip(values // 100, 0, 23)
        minute = hour * 60 + np.clip(values % 100, 0, 59)
    elif int(np.max(values)) > 23:
        minute = np.clip(values, 0, 1439)
        hour = minute // 60
    else:
        hour = np.clip(values, 0, 23)
        minute = hour * 60
    return hour.astype(np.int64), minute.astype(np.int64)


def encode_offsets(train_raw, val_raw):
    train_cols = []
    val_cols = []
    dims = []
    offset = 0
    for tr_col, va_col in zip(train_raw, val_raw):
        mapping = {}
        tr_enc = np.empty(len(tr_col), dtype=np.int64)
        for i, value in enumerate(tr_col):
            if value not in mapping:
                mapping[value] = len(mapping)
            tr_enc[i] = mapping[value]
        unknown = len(mapping)
        va_enc = np.empty(len(va_col), dtype=np.int64)
        for i, value in enumerate(va_col):
            va_enc[i] = mapping.get(value, unknown)
        dim = unknown + 1
        train_cols.append(tr_enc + offset)
        val_cols.append(va_enc + offset)
        dims.append(dim)
        offset += dim
    return (np.stack(train_cols, axis=1), np.stack(val_cols, axis=1),
            np.asarray(dims, dtype=np.int64))


def load_csv_data(data_dir):
    def read_file(path, training):
        user = []
        video = []
        tab = []
        hourmin = []
        date = []
        duration = []
        labels = []
        play = []
        with open(path, "r", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                user.append(row["user_id"])
                video.append(row["video_id"])
                tab.append(row["tab"])
                hourmin.append(int(float(row["hourmin"])))
                date.append(int(float(row["date"])))
                duration.append(float(row["duration_ms"]))
                labels.append(float(row["long_view"]))
                if training:
                    play.append(float(row["play_time_ms"]))
        return {
            "user_raw": np.asarray(user),
            "video_raw": np.asarray(video),
            "tab_raw": np.asarray(tab),
            "hourmin": np.asarray(hourmin, dtype=np.int64),
            "date": np.asarray(date, dtype=np.int64),
            "duration_ms": np.asarray(duration, dtype=np.float32),
            "y": np.asarray(labels, dtype=np.float32),
            "play_time_ms": np.asarray(play, dtype=np.float32) if training else None,
        }

    tr = read_file(os.path.join(data_dir, "train.csv"), True)
    va = read_file(os.path.join(data_dir, "val.csv"), False)
    quantiles = np.quantile(tr["duration_ms"], np.linspace(0.0, 1.0, 11)[1:-1])
    tr_bucket = np.searchsorted(quantiles, tr["duration_ms"], side="right").astype(str)
    va_bucket = np.searchsorted(quantiles, va["duration_ms"], side="right").astype(str)
    tr_fields = [tr["user_raw"], tr["video_raw"], tr["video_raw"],
                 tr["tab_raw"], tr_bucket]
    va_fields = [va["user_raw"], va["video_raw"], va["video_raw"],
                 va["tab_raw"], va_bucket]
    Xt, Xv, dims = encode_offsets(tr_fields, va_fields)
    tr["X"] = Xt.astype(np.int32)
    va["X"] = Xv.astype(np.int32)
    tr["field_dims"] = dims
    va["field_dims"] = dims
    tr["user"] = tr["user_raw"]
    va["user"] = va["user_raw"]
    return tr, va, False


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        trn = np.load(train_npz)
        van = np.load(val_npz)
        tr = {key: trn[key] for key in trn.files}
        va = {key: van[key] for key in van.files}
        tr["video_raw"] = np.zeros(len(tr["y"]), dtype=np.int64)
        va["video_raw"] = np.zeros(len(va["y"]), dtype=np.int64)
        return tr, va, True
    return load_csv_data(data_dir)


def evaluator(fast_path):
    if fast_path:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    return evaluate


def metric_dict(evaluate_fn, users, labels, scores):
    m = evaluate_fn(users, labels.astype(int), scores)
    return {
        "gauc": float(m.get("GAUC", m.get("gauc"))),
        "ndcg5": float(m.get("nDCG@5", m.get("ndcg5"))),
        "primary": float(m["primary"]),
    }


def local_tab(X, field_dims):
    return X[:, 3].astype(np.int64) - int(np.sum(field_dims[:3]))


def build_causal_features(tr, va):
    ntr = len(tr["y"])
    nva = len(va["y"])
    histories_tr = np.full((ntr, 12), -1, dtype=np.int32)
    histories_va = np.full((nva, 12), -1, dtype=np.int32)
    gap_tr = np.full(ntr, 8, dtype=np.int64)
    gap_va = np.full(nva, 8, dtype=np.int64)
    pos_tr = np.zeros(ntr, dtype=np.int64)
    pos_va = np.zeros(nva, dtype=np.int64)

    weekday_tr, day_tr = date_parts(tr["date"])
    weekday_va, day_va = date_parts(va["date"])
    hour_tr, minute_tr = hour_and_minute(tr["hourmin"])
    hour_va, minute_va = hour_and_minute(va["hourmin"])
    time_tr = day_tr * 1440 + minute_tr
    time_va = day_va * 1440 + minute_va
    tab_tr = local_tab(tr["X"], tr["field_dims"])
    tab_va = local_tab(va["X"], tr["field_dims"])
    rand_tr = (tab_tr != 0).astype(np.int64)
    rand_va = (tab_va != 0).astype(np.int64)

    users = np.concatenate([np.asarray(tr["user"]), np.asarray(va["user"])])
    times = np.concatenate([time_tr, time_va])
    split = np.concatenate([np.zeros(ntr, dtype=np.int8), np.ones(nva, dtype=np.int8)])
    row = np.concatenate([np.arange(ntr), np.arange(nva)])
    authors = np.concatenate([tr["X"][:, 2], va["X"][:, 2]]).astype(np.int32)
    original = np.arange(ntr + nva)
    order = np.lexsort((original, split, times, users.astype(str)))
    state = {}
    gap_edges = np.asarray([0, 1, 2, 5, 10, 30, 60, 180], dtype=np.int64)
    for combined_index in order:
        user = users[combined_index]
        current_time = int(times[combined_index])
        history, previous_time, previous_pos = state.get(user, ([], None, -1))
        if previous_time is None:
            gap_bucket = 8
            session_pos = 0
        else:
            gap_minutes = max(0, current_time - previous_time)
            gap_bucket = int(np.searchsorted(gap_edges, gap_minutes, side="right") - 1)
            gap_bucket = max(0, min(7, gap_bucket))
            session_pos = 0 if gap_minutes > 30 else min(15, previous_pos + 1)
        hist_values = history[-12:]
        if split[combined_index] == 0:
            r = int(row[combined_index])
            if hist_values:
                histories_tr[r, :len(hist_values)] = hist_values
            gap_tr[r] = gap_bucket
            pos_tr[r] = session_pos
        else:
            r = int(row[combined_index])
            if hist_values:
                histories_va[r, :len(hist_values)] = hist_values
            gap_va[r] = gap_bucket
            pos_va[r] = session_pos
        history = (history + [int(authors[combined_index])])[-12:]
        state[user] = (history, current_time, session_pos)

    base_dim = int(np.sum(tr["field_dims"]))
    offsets = [base_dim, base_dim + 24, base_dim + 31, base_dim + 33, base_dim + 42]
    context_tr = np.stack([
        hour_tr + offsets[0], weekday_tr + offsets[1], rand_tr + offsets[2],
        gap_tr + offsets[3], pos_tr + offsets[4]
    ], axis=1)
    context_va = np.stack([
        hour_va + offsets[0], weekday_va + offsets[1], rand_va + offsets[2],
        gap_va + offsets[3], pos_va + offsets[4]
    ], axis=1)
    total_dim = base_dim + 58
    return (context_tr.astype(np.int32), context_va.astype(np.int32), histories_tr,
            histories_va, total_dim)


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
        pair = 0.5 * (s * s - (e * e).sum(1)).sum(1)
        return self.bias + self.lin(x).sum((1, 2)) + pair


class SequenceDeepFM(torch.nn.Module):
    def __init__(self, total_dim, k=16, dropout=0.20):
        super().__init__()
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(11 * k, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.main_head = torch.nn.Linear(64, 1)
        self.watch_head = torch.nn.Linear(64, 1)

    def forward(self, x, context, history):
        ids = torch.cat([x, context], dim=1)
        current_e = self.emb(ids)
        mask = (history >= 0).float().unsqueeze(-1)
        safe_history = history.clamp_min(0)
        hist_e = (self.emb(safe_history) * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        fields = torch.cat([current_e, hist_e.unsqueeze(1)], dim=1)
        summed = fields.sum(1)
        pair = 0.5 * (summed.square() - fields.square().sum(1)).sum(1)
        linear = self.lin(ids).sum((1, 2))
        hist_linear = (self.lin(safe_history) * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        deep = self.mlp(fields.flatten(1))
        logit = self.bias + linear + hist_linear.squeeze(1) + pair
        logit = logit + self.main_head(deep).squeeze(1)
        watch = self.watch_head(deep).squeeze(1)
        return logit, watch


def gauge_center(logits, user_ids, global_bias):
    _, inverse = torch.unique(user_ids, sorted=False, return_inverse=True)
    count = torch.zeros(int(inverse.max().item()) + 1, device=logits.device,
                        dtype=logits.dtype)
    total = torch.zeros_like(count)
    count.scatter_add_(0, inverse, torch.ones_like(logits))
    total.scatter_add_(0, inverse, logits)
    means = total / count.clamp_min(1.0)
    return logits - means[inverse] + global_bias


def complete_user_groups(users):
    users = np.asarray(users)
    _, inverse = np.unique(users, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    sorted_inverse = inverse[order]
    boundaries = np.flatnonzero(np.diff(sorted_inverse)) + 1
    return [part.astype(np.int64, copy=False)
            for part in np.split(order.astype(np.int64, copy=False), boundaries)]


def complete_user_batches(groups, rng, target_rows=4096):
    group_order = rng.permutation(len(groups))
    pending = []
    pending_rows = 0
    for group_index in group_order:
        group = groups[int(group_index)]
        if pending and pending_rows + len(group) > target_rows:
            yield np.concatenate(pending)
            pending = []
            pending_rows = 0
        pending.append(group)
        pending_rows += len(group)
        if pending_rows >= target_rows:
            yield np.concatenate(pending)
            pending = []
            pending_rows = 0
    if pending:
        yield np.concatenate(pending)


def predict_parent(model, X, device):
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(X), 65536):
            xb = torch.from_numpy(X[start:start + 65536].astype(np.int64)).to(device)
            output.append(model(xb).detach().cpu().numpy())
    return np.concatenate(output)


def train_parent_reference(tr, va, device, epochs):
    seed_everything(42)
    total_dim = int(np.sum(tr["field_dims"]))
    model = ParentFM(total_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()
    n = len(tr["y"])
    rng = np.random.RandomState(42)
    X = tr["X"].astype(np.int64)
    y = tr["y"].astype(np.float32)
    for _ in range(epochs):
        model.train()
        perm = rng.permutation(n)
        for start in range(0, n, 8192):
            ids = perm[start:start + 8192]
            xb = torch.from_numpy(X[ids]).to(device)
            yb = torch.from_numpy(y[ids]).to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
    return predict_parent(model, va["X"], device)


def predict_composite(model, X, context, history, device):
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(X), 32768):
            end = start + 32768
            xb = torch.from_numpy(X[start:end].astype(np.int64)).to(device)
            cb = torch.from_numpy(context[start:end].astype(np.int64)).to(device)
            hb = torch.from_numpy(history[start:end].astype(np.int64)).to(device)
            logits, _ = model(xb, cb, hb)
            output.append(logits.detach().cpu().numpy())
    return np.concatenate(output)


def train_member(seed, tr, va, context_tr, context_va, history_tr, history_va,
                 total_dim, device, epochs, evaluate_fn):
    seed_everything(seed)
    model = SequenceDeepFM(total_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=2e-5)
    milestones = sorted(set([max(1, epochs // 3), max(2, (2 * epochs) // 3)]))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones,
                                                      gamma=0.35)
    bce = torch.nn.BCEWithLogitsLoss()
    X = tr["X"].astype(np.int64)
    y = tr["y"].astype(np.float32)
    play = np.maximum(tr["play_time_ms"].astype(np.float32), 0.0)
    duration = np.maximum(tr["duration_ms"].astype(np.float32), 1.0)
    watch_target = (np.log1p(np.minimum(play, duration)) / 10.0).astype(np.float32)
    censored = ((play >= duration) & (duration > 1.0)).astype(np.float32)
    groups = complete_user_groups(tr["user"])
    rng = np.random.RandomState(seed)
    best_primary = -1.0
    best_scores = None
    best_state = None
    patience = 0
    history_log = []

    for epoch in range(epochs):
        model.train()
        running = 0.0
        batches = 0
        for ids in complete_user_batches(groups, rng, target_rows=4096):
            xb = torch.from_numpy(X[ids]).to(device)
            cb = torch.from_numpy(context_tr[ids].astype(np.int64)).to(device)
            hb = torch.from_numpy(history_tr[ids].astype(np.int64)).to(device)
            yb = torch.from_numpy(y[ids]).to(device)
            tb = torch.from_numpy(watch_target[ids]).to(device)
            zb = torch.from_numpy(censored[ids]).to(device)
            ub = xb[:, 0]
            opt.zero_grad(set_to_none=True)
            raw_logits, watch_pred = model(xb, cb, hb)
            centered_logits = gauge_center(raw_logits, ub, model.bias)
            main_loss = bce(centered_logits, yb)
            uncensored_loss = torch.nn.functional.smooth_l1_loss(
                watch_pred, tb, reduction="none")
            censored_loss = torch.relu(tb - watch_pred).square()
            watch_loss = ((1.0 - zb) * uncensored_loss + zb * censored_loss).mean()
            loss = main_loss + 0.05 * watch_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            running += float(loss.detach().cpu())
            batches += 1
        scheduler.step()
        scores = predict_composite(model, va["X"], context_va, history_va, device)
        metrics = metric_dict(evaluate_fn, va["user"], va["y"], scores)
        history_log.append({
            "epoch": epoch + 1,
            "train_loss": round(running / max(1, batches), 6),
            "lr": float(opt.param_groups[0]["lr"]),
            "val_gauc": round(metrics["gauc"], 6),
            "val_primary": round(metrics["primary"], 6),
        })
        if metrics["primary"] > best_primary + 1e-6:
            best_primary = metrics["primary"]
            best_scores = scores.copy()
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 3:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_scores, best_primary, history_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=14)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, va, fast_path = load_data(args.data_dir)
    evaluate_fn = evaluator(fast_path)
    smoke = os.environ.get("SMOKE_EPOCHS")
    epochs = args.epochs if smoke is None else min(args.epochs, max(1, int(smoke)))
    context_tr, context_va, history_tr, history_va, total_dim = build_causal_features(tr, va)

    parent_epochs = min(8, epochs)
    parent_scores = train_parent_reference(tr, va, device, parent_epochs)
    member_scores = []
    member_history = []
    member_primaries = []
    progress_path = os.path.join(args.out_dir, "progress.log")
    member_seeds = [args.seed, args.seed + 1, args.seed + 2]
    for member_seed in member_seeds:
        scores, primary, history_log = train_member(
            member_seed, tr, va, context_tr, context_va, history_tr, history_va,
            total_dim, device, epochs, evaluate_fn)
        if np.allclose(scores, parent_scores, rtol=1e-7, atol=1e-8):
            raise AssertionError("Gauge-fixed member predictions equal parent predictions")
        for previous in member_scores:
            if np.allclose(scores, previous, rtol=1e-7, atol=1e-8):
                raise AssertionError("Distinct-seed members produced identical scores")
        member_scores.append(scores)
        member_primaries.append(float(primary))
        member_history.append({
            "seed": int(member_seed),
            "best_primary": float(primary),
            "epochs": history_log,
        })
        with open(progress_path, "a") as fh:
            fh.write(json.dumps({
                "seed": int(member_seed),
                "val_primary": float(primary),
                "model": "seq_deepfm_gauge_fixed_bce",
            }) + "\n")

    final_scores = np.mean(np.stack(member_scores, axis=0), axis=0)
    if np.allclose(final_scores, parent_scores, rtol=1e-7, atol=1e-8):
        raise AssertionError("Gauge-fixed ensemble predictions equal parent predictions")
    final_metrics = metric_dict(evaluate_fn, va["user"], va["y"], final_scores)
    metrics_output = {
        "gauc": final_metrics["gauc"],
        "ndcg5": final_metrics["ndcg5"],
        "primary": final_metrics["primary"],
        "history": member_history,
        "member_primaries": member_primaries,
        "ensemble": {"method": "mean_logit", "seeds": member_seeds},
        "objective": "complete_user_gauge_fixed_bce",
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics_output, fh)
    with open(os.path.join(args.out_dir, "predictions.csv"), "w") as fh:
        fh.write("row_id,user_id,video_id,score\n")
        videos = va.get("video_raw", np.zeros(len(final_scores), dtype=np.int64))
        for i, score in enumerate(final_scores):
            fh.write(f"{i},{va['user'][i]},{videos[i]},{float(score):.8g}\n")


if __name__ == "__main__":
    main()
