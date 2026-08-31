import argparse
import csv
import datetime
import json
import math
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class GatedSessionFM(torch.nn.Module):
    def __init__(self, total_dim, session_dims, k=16):
        super().__init__()
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)
        self.session_embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(int(d), 4) for d in session_dims
        ])
        for emb in self.session_embeddings:
            torch.nn.init.normal_(emb.weight, std=0.01)
        self.session_tower = torch.nn.Sequential(
            torch.nn.Linear(4 * len(session_dims), 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
        )
        torch.nn.init.zeros_(self.session_tower[-1].weight)
        torch.nn.init.zeros_(self.session_tower[-1].bias)
        self.gate_logit = torch.nn.Parameter(torch.tensor(math.log(0.1 / 0.9)))

    def base_score(self, x):
        e = self.emb(x)
        s = e.sum(1)
        pair = 0.5 * (s * s - (e * e).sum(1)).sum(1)
        return self.bias + self.lin(x).sum((1, 2)) + pair

    def forward(self, x, session_x):
        base = self.base_score(x)
        z = torch.cat([
            emb(session_x[:, j]) for j, emb in enumerate(self.session_embeddings)
        ], dim=1)
        residual = self.session_tower(z).squeeze(1)
        return base + torch.sigmoid(self.gate_logit) * residual


def date_to_ordinal(value):
    try:
        s = str(int(value))
        if len(s) == 8:
            return datetime.date(int(s[:4]), int(s[4:6]), int(s[6:8])).toordinal()
    except Exception:
        pass
    try:
        return int(value)
    except Exception:
        return 0


def decode_hours(hourmin):
    x = np.asarray(hourmin).astype(np.int64)
    if len(x) == 0:
        return x
    mx = int(np.max(x))
    if mx <= 23:
        return np.clip(x, 0, 23)
    if mx <= 2359 and np.all((x % 100) < 60):
        return np.clip(x // 100, 0, 23)
    if mx <= 1439:
        return np.clip(x // 60, 0, 23)
    return np.clip(x // 100, 0, 23)


def gap_bucket(delta):
    if delta is None:
        return 7
    if delta <= 0:
        return 0
    if delta == 1:
        return 1
    if delta <= 3:
        return 2
    if delta <= 7:
        return 3
    if delta <= 23:
        return 4
    if delta <= 72:
        return 5
    if delta <= 168:
        return 6
    return 7


def position_bucket(position):
    if position <= 3:
        return position
    if position <= 5:
        return 4
    if position <= 8:
        return 5
    return 6


def make_session_features(train, val, field_dims):
    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1]))).astype(np.int64)
    tab_dim = int(field_dims[3])
    dur_dim = int(field_dims[4])
    state_last_time = {}
    state_session = {}
    state_position = {}

    def transform(split):
        users = np.asarray(split["user"])
        dates_raw = np.asarray(split["date"])
        hours = decode_hours(split["hourmin"])
        x = np.asarray(split["X"], dtype=np.int64)
        tabs = np.clip(x[:, 3] - offsets[3], 0, tab_dim - 1)
        durs = np.clip(x[:, 4] - offsets[4], 0, dur_dim - 1)
        out = np.empty((len(x), 5), dtype=np.int64)
        for i in range(len(x)):
            user = int(users[i])
            day = date_to_ordinal(dates_raw[i])
            hour = int(hours[i])
            timestamp = day * 24 + hour
            previous = state_last_time.get(user)
            gap = gap_bucket(None if previous is None else timestamp - previous)
            session_key = (day, hour)
            if state_session.get(user) == session_key:
                position = state_position[user]
            else:
                position = 0
            pb = position_bucket(position)
            tab = int(tabs[i])
            dur = int(durs[i])
            out[i, 0] = gap
            out[i, 1] = pb
            out[i, 2] = pb * tab_dim + tab
            out[i, 3] = pb * dur_dim + dur
            out[i, 4] = hour * tab_dim + tab
            state_last_time[user] = timestamp
            state_session[user] = session_key
            state_position[user] = position + 1
        return out

    train_session = transform(train)
    val_session = transform(val)
    dims = [8, 7, 7 * tab_dim, 7 * dur_dim, 24 * tab_dim]
    return train_session, val_session, dims


def make_pair_groups(users, labels):
    users = np.asarray(users)
    labels = np.asarray(labels) > 0.5
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    groups = []
    for j in range(len(boundaries) - 1):
        idx = order[boundaries[j]:boundaries[j + 1]].astype(np.int64)
        pos = idx[labels[idx]]
        neg = idx[~labels[idx]]
        if len(pos) and len(neg):
            ideal_count = min(5, len(pos))
            idcg = sum(1.0 / math.log2(r + 2.0) for r in range(ideal_count))
            groups.append((idx, pos.astype(np.int64), neg.astype(np.int64), float(idcg)))
    return groups


def sample_pairs(groups, seed):
    rng = np.random.default_rng(seed)
    positives = []
    negatives = []
    idcgs = []
    for _, pos, neg, idcg in groups:
        positives.append(pos)
        negatives.append(neg[rng.integers(0, len(neg), size=len(pos))])
        idcgs.append(np.full(len(pos), idcg, dtype=np.float32))
    if not positives:
        empty_i = np.empty(0, dtype=np.int64)
        return empty_i, empty_i.copy(), np.empty(0, dtype=np.float32)
    return np.concatenate(positives), np.concatenate(negatives), np.concatenate(idcgs)


def metric_values(m):
    return {
        "gauc": float(m.get("GAUC", m.get("gauc", 0.0))),
        "ndcg5": float(m.get("nDCG@5", m.get("ndcg5", 0.0))),
        "primary": float(m["primary"]),
    }


def evaluate_scores(evaluator, users, labels, scores):
    return metric_values(evaluator(users, labels.astype(int), scores))


def load_csv_data(data_dir):
    with open(os.path.join(data_dir, "train.csv"), newline="") as fh:
        train_rows = list(csv.DictReader(fh))
    with open(os.path.join(data_dir, "val.csv"), newline="") as fh:
        val_rows = list(csv.DictReader(fh))

    def values(rows, key, default="0"):
        return [row.get(key, default) for row in rows]

    train_duration = np.asarray([float(v or 0) for v in values(train_rows, "duration_ms")])
    quantiles = np.quantile(train_duration, np.linspace(0.1, 0.9, 9)) if len(train_duration) else np.arange(9)
    maps = []
    for key in ("user_id", "video_id"):
        unique = sorted(set(values(train_rows, key)))
        maps.append({v: i + 1 for i, v in enumerate(unique)})
    tab_unique = sorted(set(values(train_rows, "tab")))
    tab_map = {v: i + 1 for i, v in enumerate(tab_unique)}
    field_dims = np.asarray([
        len(maps[0]) + 1, len(maps[1]) + 1, len(maps[1]) + 1,
        len(tab_map) + 1, 10
    ], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1])))

    def encode(rows, is_train):
        n = len(rows)
        x = np.zeros((n, 5), dtype=np.int64)
        raw_user = np.zeros(n, dtype=np.int64)
        y = np.zeros(n, dtype=np.float32)
        click = np.zeros(n, dtype=np.float32)
        play = np.zeros(n, dtype=np.float32)
        duration = np.zeros(n, dtype=np.float32)
        hourmin = np.zeros(n, dtype=np.int64)
        date = np.zeros(n, dtype=np.int64)
        video_text = []
        for i, row in enumerate(rows):
            uid = row.get("user_id", "0")
            vid = row.get("video_id", "0")
            tab = row.get("tab", "0")
            dur = float(row.get("duration_ms", 0) or 0)
            u = maps[0].get(uid, 0)
            v = maps[1].get(vid, 0)
            t = tab_map.get(tab, 0)
            d = int(np.searchsorted(quantiles, dur, side="right"))
            x[i] = [u + offsets[0], v + offsets[1], v + offsets[2], t + offsets[3], d + offsets[4]]
            try:
                raw_user[i] = int(uid)
            except Exception:
                raw_user[i] = u
            y[i] = float(row.get("long_view", 0) or 0)
            if is_train:
                click[i] = float(row.get("click", 0) or 0)
                play[i] = float(row.get("play_time_ms", 0) or 0)
            duration[i] = dur
            hourmin[i] = int(float(row.get("hourmin", 0) or 0))
            date[i] = int(float(row.get("date", 0) or 0))
            video_text.append(vid)
        return {
            "X": x.astype(np.int32), "y": y, "user": raw_user, "click": click,
            "play_time_ms": play, "duration_ms": duration, "hourmin": hourmin,
            "date": date, "video_text": np.asarray(video_text, dtype=object)
        }

    return encode(train_rows, True), encode(val_rows, False), field_dims


def predict(model, xv, sv, batch_size):
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(xv), batch_size):
            end = min(len(xv), start + batch_size)
            chunks.append(model(xv[start:end], sv[start:end]).detach().cpu().numpy())
    return np.concatenate(chunks).astype(np.float64)


def predict_base(model, x, batch_size):
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(len(x), start + batch_size)
            chunks.append(model.base_score(x[start:end]).detach().cpu().numpy())
    return np.concatenate(chunks).astype(np.float64)


def compute_lambda_weights(model, xt, groups, pos_np, neg_np, idcg_np):
    scores = predict_base(model, xt, 65536)
    ranks = np.empty(len(scores), dtype=np.int32)
    for idx, _, _, _ in groups:
        local_order = np.argsort(-scores[idx], kind="stable")
        ranks[idx[local_order]] = np.arange(1, len(idx) + 1, dtype=np.int32)
    pos_rank = ranks[pos_np]
    neg_rank = ranks[neg_np]
    pos_discount = np.where(pos_rank <= 5, 1.0 / np.log2(pos_rank + 1.0), 0.0)
    neg_discount = np.where(neg_rank <= 5, 1.0 / np.log2(neg_rank + 1.0), 0.0)
    weights = np.abs(pos_discount - neg_discount) / np.maximum(idcg_np, 1e-12)
    return weights.astype(np.float32)


def train_one(variant, seed, epochs, tensors, pair_groups, evaluator, val_users, val_labels):
    device = tensors["device"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = GatedSessionFM(tensors["total_dim"], tensors["session_dims"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = torch.nn.BCEWithLogitsLoss()
    xt, yt = tensors["xt"], tensors["yt"]
    xv, st, sv = tensors["xv"], tensors["st"], tensors["sv"]
    pos_np, neg_np, idcg_np = sample_pairs(pair_groups, seed + 7919)
    pos = torch.as_tensor(pos_np, dtype=torch.long, device=device)
    neg = torch.as_tensor(neg_np, dtype=torch.long, device=device)
    n = len(yt)
    bs = 8192
    best_primary = -1.0
    best_scores = None
    patience = 0
    curve = []
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 104729)

    for epoch in range(epochs):
        if len(pos_np) and variant == "lambda":
            weight_np = compute_lambda_weights(model, xt, pair_groups, pos_np, neg_np, idcg_np)
        else:
            weight_np = np.ones(len(pos_np), dtype=np.float32)
        pair_weights = torch.as_tensor(weight_np, dtype=torch.float32, device=device)
        model.train()
        perm = torch.randperm(n, generator=generator, device=device)
        loss_sum = 0.0
        batches = 0
        for start in range(0, n, bs):
            idx = perm[start:start + bs]
            logits = model(xt[idx], st[idx])
            point_loss = bce(logits, yt[idx])
            if len(pos):
                pair_choice = torch.randint(len(pos), (len(idx),), generator=generator, device=device)
                pidx = pos[pair_choice]
                nidx = neg[pair_choice]
                raw_pair_loss = torch.nn.functional.softplus(
                    -(model.base_score(xt[pidx]) - model.base_score(xt[nidx]))
                )
                chosen_weights = pair_weights[pair_choice]
                if variant == "lambda":
                    pair_loss = (raw_pair_loss * chosen_weights).sum() / chosen_weights.sum().clamp_min(1e-8)
                else:
                    pair_loss = raw_pair_loss.mean()
                loss = 0.5 * point_loss + 0.5 * pair_loss
            else:
                loss = point_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += float(loss.detach().cpu())
            batches += 1

        scores = predict(model, xv, sv, 65536)
        met = evaluate_scores(evaluator, val_users, val_labels, scores)
        nonzero = float(np.mean(weight_np > 0)) if len(weight_np) else 0.0
        mean_weight = float(np.mean(weight_np)) if len(weight_np) else 0.0
        curve.append({
            "epoch": epoch + 1,
            "train_loss": round(loss_sum / max(1, batches), 6),
            "val_gauc": round(met["gauc"], 6),
            "val_primary": round(met["primary"], 6),
            "pair_weight_nonzero_fraction": round(nonzero, 6),
            "pair_weight_mean": round(mean_weight, 6)
        })
        if met["primary"] > best_primary + 1e-6:
            best_primary = met["primary"]
            best_scores = scores.copy()
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break
    return best_scores, curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    smoke = os.environ.get("SMOKE_EPOCHS")
    epochs = min(args.epochs, max(1, int(smoke))) if smoke is not None else args.epochs
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fast = (
        os.path.exists(os.path.join(args.data_dir, "train.npz")) and
        os.path.exists(os.path.join(args.data_dir, "val.npz"))
    )
    if fast:
        tr_npz = np.load(os.path.join(args.data_dir, "train.npz"))
        va_npz = np.load(os.path.join(args.data_dir, "val.npz"))
        tr = {k: tr_npz[k] for k in tr_npz.files}
        va = {k: va_npz[k] for k in va_npz.files}
        field_dims = np.asarray(tr["field_dims"], dtype=np.int64)
        from data.official.evaluate import evaluate as evaluator
        video_output = (np.asarray(va["X"])[:, 1] - int(field_dims[0])).astype(np.int64)
    else:
        tr, va, field_dims = load_csv_data(args.data_dir)
        from harness.evaluate_provisional import evaluate as evaluator
        video_output = va["video_text"]

    session_train, session_val, session_dims = make_session_features(tr, va, field_dims)
    tensors = {
        "device": device,
        "total_dim": int(field_dims.sum()),
        "session_dims": session_dims,
        "xt": torch.as_tensor(np.asarray(tr["X"], dtype=np.int64), device=device),
        "yt": torch.as_tensor(np.asarray(tr["y"], dtype=np.float32), device=device),
        "xv": torch.as_tensor(np.asarray(va["X"], dtype=np.int64), device=device),
        "st": torch.as_tensor(session_train, dtype=torch.long, device=device),
        "sv": torch.as_tensor(session_val, dtype=torch.long, device=device),
    }
    val_users = np.asarray(va["user"])
    val_labels = np.asarray(va["y"])
    pair_groups = make_pair_groups(tr["user"], tr["y"])
    variants = ["uniform", "lambda"]
    if smoke is not None:
        probe_seeds = 1
        final_extra = 0
    elif device.type == "cuda":
        probe_seeds = 48
        final_extra = 16
    else:
        probe_seeds = 16
        final_extra = 8

    history = []
    aggregates = {}
    progress_path = os.path.join(args.out_dir, "progress.log")
    for variant in variants:
        score_sum = np.zeros(len(val_labels), dtype=np.float64)
        per_seed = []
        for j in range(probe_seeds):
            run_seed = args.seed + j * 1009
            scores, curve = train_one(
                variant, run_seed, epochs, tensors, pair_groups,
                evaluator, val_users, val_labels
            )
            met = evaluate_scores(evaluator, val_users, val_labels, scores)
            score_sum += scores
            per_seed.append(met["primary"])
            history.append({
                "phase": "paired_probe", "variant": variant, "seed": run_seed,
                "gauc": met["gauc"], "ndcg5": met["ndcg5"],
                "primary": met["primary"], "curve": curve
            })
            with open(progress_path, "a") as fh:
                fh.write(json.dumps({
                    "phase": "paired_probe", "variant": variant,
                    "seed": run_seed, "primary": met["primary"]
                }) + "\n")
        averaged = score_sum / probe_seeds
        ensemble_met = evaluate_scores(evaluator, val_users, val_labels, averaged)
        aggregates[variant] = {
            "scores": score_sum,
            "count": probe_seeds,
            "metrics": ensemble_met,
            "per_seed_primary": per_seed
        }
        history.append({
            "phase": "probe_ensemble", "variant": variant,
            "seeds": probe_seeds, "gauc": ensemble_met["gauc"],
            "ndcg5": ensemble_met["ndcg5"], "primary": ensemble_met["primary"],
            "mean_seed_primary": float(np.mean(per_seed)),
            "std_seed_primary": float(np.std(per_seed))
        })

    paired_delta = np.asarray(aggregates["lambda"]["per_seed_primary"]) - np.asarray(aggregates["uniform"]["per_seed_primary"])
    history.append({
        "phase": "paired_attribution",
        "lambda_minus_uniform_mean_primary": float(np.mean(paired_delta)),
        "lambda_minus_uniform_std": float(np.std(paired_delta)),
        "positive_seed_fraction": float(np.mean(paired_delta > 0))
    })

    winner = max(variants, key=lambda v: aggregates[v]["metrics"]["primary"])
    final_sum = aggregates[winner]["scores"].copy()
    final_count = aggregates[winner]["count"]
    for j in range(final_extra):
        run_seed = args.seed + 50021 + j * 1009
        scores, curve = train_one(
            winner, run_seed, epochs, tensors, pair_groups,
            evaluator, val_users, val_labels
        )
        met = evaluate_scores(evaluator, val_users, val_labels, scores)
        final_sum += scores
        final_count += 1
        history.append({
            "phase": "final", "variant": winner, "seed": run_seed,
            "gauc": met["gauc"], "ndcg5": met["ndcg5"],
            "primary": met["primary"], "curve": curve
        })
        with open(progress_path, "a") as fh:
            fh.write(json.dumps({
                "phase": "final", "variant": winner,
                "seed": run_seed, "primary": met["primary"]
            }) + "\n")

    final_scores = final_sum / final_count
    final_metrics = evaluate_scores(evaluator, val_users, val_labels, final_scores)
    payload = {
        "gauc": final_metrics["gauc"],
        "ndcg5": final_metrics["ndcg5"],
        "primary": final_metrics["primary"],
        "selected_variant": winner,
        "ensemble_members": final_count,
        "paired_delta_mean": float(np.mean(paired_delta)),
        "history": history
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(payload, fh)
    with open(os.path.join(args.out_dir, "predictions.csv"), "w") as fh:
        fh.write("row_id,user_id,video_id,score\n")
        for i, score in enumerate(final_scores):
            fh.write(f"{i},{va['user'][i]},{video_output[i]},{score:.8g}\n")


if __name__ == "__main__":
    main()
