"""Matched-seed ablation of a gated causal-session residual on user-centered FM."""
import argparse
import csv
import datetime
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class GatedSessionFM(torch.nn.Module):
    def __init__(self, total_dim, session_dims, k=16, session_k=8,
                 hidden=32, use_session=False):
        super().__init__()
        self.use_session = bool(use_session)
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)

        if self.use_session:
            dims = np.asarray(session_dims, dtype=np.int64)
            offsets = np.concatenate(([0], np.cumsum(dims[:-1])))
            self.register_buffer(
                "session_offsets",
                torch.tensor(offsets, dtype=torch.long),
            )
            self.session_emb = torch.nn.Embedding(int(dims.sum()), session_k)
            torch.nn.init.normal_(self.session_emb.weight, std=0.01)
            self.session_hidden = torch.nn.Linear(len(dims) * session_k, hidden)
            self.session_out = torch.nn.Linear(hidden, 1)
            torch.nn.init.zeros_(self.session_out.weight)
            torch.nn.init.zeros_(self.session_out.bias)
            self.gate_logit = torch.nn.Parameter(
                torch.tensor(-2.197224577, dtype=torch.float32)
            )

    def base_relative(self, x):
        e = self.emb(x)
        summed = e.sum(1)
        pair = 0.5 * (summed * summed - (e * e).sum(1)).sum(1)
        return self.lin(x).sum((1, 2)) + pair

    def session_residual(self, session_x):
        indexed = session_x + self.session_offsets.unsqueeze(0)
        e = self.session_emb(indexed).reshape(len(session_x), -1)
        hidden = torch.relu(self.session_hidden(e))
        residual = self.session_out(hidden).squeeze(1)
        return torch.sigmoid(self.gate_logit) * residual

    def relative_logit(self, x, session_x=None):
        score = self.base_relative(x)
        if self.use_session:
            score = score + self.session_residual(session_x)
        return score

    def forward(self, x, session_x=None):
        return self.relative_logit(x, session_x) + self.bias


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_hour(value):
    text = str(value).strip()
    if ":" in text:
        return max(0, min(23, int(text.split(":", 1)[0])))
    try:
        number = int(float(text))
    except ValueError:
        return 0
    if number <= 23:
        return max(0, number)
    return max(0, min(23, number // 100))


def date_ordinal(value):
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) >= 8:
        try:
            date = datetime.date(
                int(digits[:4]), int(digits[4:6]), int(digits[6:8])
            )
            return date.toordinal()
        except ValueError:
            pass
    try:
        return int(float(text))
    except ValueError:
        return 0


def scalar_key(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def gap_bucket(delta_hours, first):
    if first or delta_hours >= 168:
        return 7
    if delta_hours <= 0:
        return 0
    if delta_hours == 1:
        return 1
    if delta_hours <= 3:
        return 2
    if delta_hours <= 7:
        return 3
    if delta_hours <= 23:
        return 4
    if delta_hours <= 71:
        return 5
    return 6


def position_bucket(position):
    if position <= 3:
        return int(position)
    if position <= 5:
        return 4
    if position <= 8:
        return 5
    return 6


def build_session_features(train_users, train_dates, train_hourmin, train_x,
                           val_users, val_dates, val_hourmin, val_x, field_dims):
    tab_dim = int(field_dims[3])
    dur_dim = int(field_dims[4])
    tab_offset = int(np.sum(field_dims[:3]))
    dur_offset = int(np.sum(field_dims[:4]))
    state_time = {}
    state_session = {}

    def encode(users, dates, hourmins, x):
        result = np.zeros((len(x), 5), dtype=np.int64)
        for i in range(len(x)):
            user = scalar_key(users[i])
            day = date_ordinal(dates[i])
            hour = parse_hour(hourmins[i])
            timestamp = day * 24 + hour
            previous = state_time.get(user)
            delta = 10 ** 9 if previous is None else max(0, timestamp - previous)
            gap = gap_bucket(delta, previous is None)
            session_key = (day, hour)
            old_session = state_session.get(user)
            if old_session is not None and old_session[0] == session_key:
                position = old_session[1] + 1
            else:
                position = 0
            state_time[user] = timestamp
            state_session[user] = (session_key, position)

            pos = position_bucket(position)
            tab = int(x[i, 3]) - tab_offset
            dur = int(x[i, 4]) - dur_offset
            tab = max(0, min(tab_dim - 1, tab))
            dur = max(0, min(dur_dim - 1, dur))
            result[i, 0] = gap
            result[i, 1] = pos
            result[i, 2] = pos * tab_dim + tab
            result[i, 3] = pos * dur_dim + dur
            result[i, 4] = hour * tab_dim + tab
        return result

    train_session = encode(
        train_users, train_dates, train_hourmin, train_x
    )
    val_session = encode(val_users, val_dates, val_hourmin, val_x)
    session_dims = np.asarray(
        [8, 7, 7 * tab_dim, 7 * dur_dim, 24 * tab_dim], dtype=np.int64
    )
    return train_session, val_session, session_dims


def load_csv_data(data_dir):
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")

    train_rows = []
    with open(train_path, newline="") as fh:
        reader = csv.DictReader(fh)
        has_author = "author_id" in (reader.fieldnames or [])
        for row in reader:
            train_rows.append({
                "user": row["user_id"],
                "video": row["video_id"],
                "author": row["author_id"] if has_author else "__NO_AUTHOR__",
                "tab": row["tab"],
                "duration": float(row["duration_ms"]),
                "hourmin": row["hourmin"],
                "date": row["date"],
                "y": float(row["long_view"]),
            })

    val_rows = []
    with open(val_path, newline="") as fh:
        reader = csv.DictReader(fh)
        has_author = "author_id" in (reader.fieldnames or [])
        for row in reader:
            val_rows.append({
                "user": row["user_id"],
                "video": row["video_id"],
                "author": row["author_id"] if has_author else "__NO_AUTHOR__",
                "tab": row["tab"],
                "duration": float(row["duration_ms"]),
                "hourmin": row["hourmin"],
                "date": row["date"],
                "y": float(row["long_view"]),
            })

    duration_train = np.asarray(
        [row["duration"] for row in train_rows], dtype=np.float64
    )
    quantiles = np.quantile(duration_train, np.arange(1, 10) / 10.0)
    field_names = ("user", "video", "author", "tab")
    mappings = {}
    dims = []
    for field in field_names:
        values = sorted({row[field] for row in train_rows})
        mappings[field] = {value: i + 1 for i, value in enumerate(values)}
        dims.append(len(values) + 1)
    dims.append(10)
    field_dims = np.asarray(dims, dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1]))).astype(np.int64)

    def encode(rows):
        x = np.zeros((len(rows), 5), dtype=np.int64)
        for i, row in enumerate(rows):
            for j, field in enumerate(field_names):
                x[i, j] = mappings[field].get(row[field], 0) + offsets[j]
            bucket = int(np.searchsorted(
                quantiles, row["duration"], side="right"
            ))
            x[i, 4] = min(bucket, 9) + offsets[4]
        return x

    xt = encode(train_rows)
    xv = encode(val_rows)
    train_raw_user = np.asarray([row["user"] for row in train_rows])
    val_raw_user = np.asarray([row["user"] for row in val_rows])
    st, sv, session_dims = build_session_features(
        train_raw_user,
        np.asarray([row["date"] for row in train_rows]),
        np.asarray([row["hourmin"] for row in train_rows]),
        xt,
        val_raw_user,
        np.asarray([row["date"] for row in val_rows]),
        np.asarray([row["hourmin"] for row in val_rows]),
        xv,
        field_dims,
    )
    return {
        "Xt": xt,
        "St": st,
        "yt": np.asarray([row["y"] for row in train_rows], dtype=np.float32),
        "train_user": xt[:, 0].copy(),
        "Xv": xv,
        "Sv": sv,
        "yv": np.asarray([row["y"] for row in val_rows], dtype=np.int64),
        "val_user": val_raw_user,
        "val_video": np.asarray([row["video"] for row in val_rows]),
        "field_dims": field_dims,
        "session_dims": session_dims,
        "fast": False,
    }


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        with np.load(train_npz) as tr, np.load(val_npz) as va:
            field_dims = tr["field_dims"].astype(np.int64)
            xt = tr["X"].astype(np.int64)
            xv = va["X"].astype(np.int64)
            train_raw_user = np.asarray(tr["user"])
            val_raw_user = np.asarray(va["user"])
            st, sv, session_dims = build_session_features(
                train_raw_user,
                np.asarray(tr["date"]),
                np.asarray(tr["hourmin"]),
                xt,
                val_raw_user,
                np.asarray(va["date"]),
                np.asarray(va["hourmin"]),
                xv,
                field_dims,
            )
            video_offset = int(field_dims[0])
            return {
                "Xt": xt,
                "St": st,
                "yt": tr["y"].astype(np.float32),
                "train_user": xt[:, 0].copy(),
                "Xv": xv,
                "Sv": sv,
                "yv": va["y"].astype(np.int64),
                "val_user": val_raw_user,
                "val_video": xv[:, 1] - video_offset,
                "field_dims": field_dims,
                "session_dims": session_dims,
                "fast": True,
            }
    return load_csv_data(data_dir)


def get_evaluator(fast):
    if fast:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    return evaluate


def metric_values(metrics):
    return (
        float(metrics.get("GAUC", metrics.get("gauc"))),
        float(metrics.get("nDCG@5", metrics.get("ndcg5"))),
        float(metrics["primary"]),
    )


def make_user_groups(user_ids):
    order = np.argsort(user_ids, kind="stable")
    sorted_users = user_ids[order]
    boundaries = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(order)]))
    return order, starts, ends


def make_complete_user_batches(order, starts, ends, rng, batch_size):
    group_order = rng.permutation(len(starts))
    batches = []
    pieces = []
    count = 0
    for group_index in group_order:
        piece = order[starts[group_index]:ends[group_index]]
        size = len(piece)
        if pieces and count + size > batch_size:
            batches.append(np.concatenate(pieces))
            pieces = []
            count = 0
        pieces.append(piece)
        count += size
        if count >= batch_size:
            batches.append(np.concatenate(pieces))
            pieces = []
            count = 0
    if pieces:
        batches.append(np.concatenate(pieces))
    return batches


def centered_logits(model, x, session_x, users):
    relative = model.relative_logit(x, session_x)
    _, inverse, counts = torch.unique(
        users, sorted=False, return_inverse=True, return_counts=True
    )
    sums = torch.zeros(len(counts), dtype=relative.dtype, device=relative.device)
    sums.scatter_add_(0, inverse, relative)
    means = sums / counts.to(relative.dtype)
    return relative - means[inverse] + model.bias


def make_pair_indices(batch_users, batch_labels, rng, cap=8):
    if len(batch_users) == 0:
        return None, None
    boundaries = np.flatnonzero(batch_users[1:] != batch_users[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(batch_users)]))
    positive_parts = []
    negative_parts = []
    for start, end in zip(starts, ends):
        local = np.arange(start, end)
        positives = local[batch_labels[start:end] > 0.5]
        negatives = local[batch_labels[start:end] <= 0.5]
        if len(positives) == 0 or len(negatives) == 0:
            continue
        count = min(cap, max(len(positives), len(negatives)))
        positive_parts.append(rng.choice(positives, size=count, replace=True))
        negative_parts.append(rng.choice(negatives, size=count, replace=True))
    if not positive_parts:
        return None, None
    return np.concatenate(positive_parts), np.concatenate(negative_parts)


def predict(model, xv, sv, device):
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(xv), 65536):
            xb = xv[start:start + 65536].to(device, non_blocking=True)
            if model.use_session:
                sb = sv[start:start + 65536].to(device, non_blocking=True)
            else:
                sb = None
            chunks.append(model(xb, sb).detach().cpu().numpy())
    return np.concatenate(chunks).astype(np.float64)


def train_one(data, evaluate, device, seed, epochs, config):
    seed_everything(seed)
    xt = torch.from_numpy(data["Xt"])
    st = torch.from_numpy(data["St"])
    yt = torch.from_numpy(data["yt"])
    train_user = torch.from_numpy(data["train_user"])
    xv = torch.from_numpy(data["Xv"])
    sv = torch.from_numpy(data["Sv"])
    use_session = bool(config["use_session"])
    pair_weight = float(config["pair_weight"])
    model = GatedSessionFM(
        int(data["field_dims"].sum()),
        data["session_dims"],
        k=16,
        session_k=8,
        hidden=32,
        use_session=use_session,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = torch.nn.BCEWithLogitsLoss()
    order, starts, ends = make_user_groups(data["train_user"])
    rng = np.random.RandomState(seed)
    best_primary = -1.0
    best_scores = None
    best_epoch = 0
    patience = 0
    epoch_history = []

    for epoch in range(epochs):
        model.train()
        batches = make_complete_user_batches(order, starts, ends, rng, 8192)
        loss_value = 0.0
        for idx_np in batches:
            idx = torch.from_numpy(idx_np)
            xb = xt[idx].to(device, non_blocking=True)
            yb = yt[idx].to(device, non_blocking=True)
            ub = train_user[idx].to(device, non_blocking=True)
            sb = st[idx].to(device, non_blocking=True) if use_session else None
            optimizer.zero_grad(set_to_none=True)
            logits = centered_logits(model, xb, sb, ub)
            point_loss = bce(logits, yb)
            loss = point_loss
            pair_count = 0
            if pair_weight > 0.0:
                users_np = data["train_user"][idx_np]
                labels_np = data["yt"][idx_np]
                pos_np, neg_np = make_pair_indices(users_np, labels_np, rng)
                if pos_np is not None:
                    pos = torch.from_numpy(pos_np).to(device)
                    neg = torch.from_numpy(neg_np).to(device)
                    base_scores = model.base_relative(xb)
                    pair_loss = torch.nn.functional.softplus(
                        -(base_scores[pos] - base_scores[neg])
                    ).mean()
                    loss = (1.0 - pair_weight) * point_loss + pair_weight * pair_loss
                    pair_count = len(pos_np)
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach().cpu())

        scores = predict(model, xv, sv, device)
        metrics = evaluate(data["val_user"], data["yv"], scores)
        gauc, ndcg5, primary = metric_values(metrics)
        gate = float(torch.sigmoid(model.gate_logit).detach().cpu()) \
            if use_session else 0.0
        epoch_history.append({
            "epoch": epoch + 1,
            "train_loss": round(loss_value, 6),
            "val_gauc": round(gauc, 8),
            "val_ndcg5": round(ndcg5, 8),
            "val_primary": round(primary, 8),
            "gate": round(gate, 8),
        })
        if primary > best_primary + 1e-6:
            best_primary = primary
            best_scores = scores.copy()
            best_epoch = epoch + 1
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break

    final_metrics = evaluate(data["val_user"], data["yv"], best_scores)
    gauc, ndcg5, primary = metric_values(final_metrics)
    result = {
        "seed": int(seed),
        "config": config["name"],
        "use_session": use_session,
        "pair_weight": pair_weight,
        "gauc": gauc,
        "ndcg5": ndcg5,
        "primary": primary,
        "best_epoch": int(best_epoch),
        "epochs": epoch_history,
    }
    return result, best_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    smoke = os.environ.get("SMOKE_EPOCHS")
    epochs = args.epochs
    if smoke is not None:
        epochs = min(epochs, max(1, int(smoke)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_data(args.data_dir)
    evaluate = get_evaluator(data["fast"])
    configs = [
        {"name": "anchor_centered_bce", "use_session": False,
         "pair_weight": 0.0},
        {"name": "gated_session_centered_bce", "use_session": True,
         "pair_weight": 0.0},
        {"name": "gated_session_routed_hybrid", "use_session": True,
         "pair_weight": 0.5},
    ]

    if smoke is not None:
        probe_count = 1
    elif device.type == "cuda":
        probe_count = 40
    else:
        probe_count = 16
    seeds = [args.seed + 1009 * i for i in range(probe_count)]
    history = []
    config_scores = {config["name"]: [] for config in configs}
    progress_path = os.path.join(args.out_dir, "progress.log")

    with open(progress_path, "w") as progress:
        for seed in seeds:
            for config in configs:
                result, _ = train_one(
                    data, evaluate, device, seed, epochs, config
                )
                history.append(result)
                config_scores[config["name"]].append(result["primary"])
                progress.write(json.dumps({
                    "seed": seed,
                    "config": config["name"],
                    "primary": result["primary"],
                    "best_epoch": result["best_epoch"],
                }, sort_keys=True) + "\n")
                progress.flush()

        summaries = []
        for config in configs:
            values = np.asarray(config_scores[config["name"]], dtype=np.float64)
            summaries.append({
                "config": config["name"],
                "mean_primary": float(values.mean()),
                "std_primary": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "probe_count": int(len(values)),
            })
        winner_name = max(summaries, key=lambda item: item["mean_primary"])["config"]
        winner_config = next(
            config for config in configs if config["name"] == winner_name
        )
        final_result, final_scores = train_one(
            data, evaluate, device, args.seed, epochs, winner_config
        )
        final_result["phase"] = "selected_final_fit"
        history.append(final_result)
        progress.write(json.dumps({
            "phase": "selected_final_fit",
            "seed": args.seed,
            "config": winner_name,
            "primary": final_result["primary"],
            "best_epoch": final_result["best_epoch"],
        }, sort_keys=True) + "\n")
        progress.flush()

    anchor = np.asarray(config_scores["anchor_centered_bce"], dtype=np.float64)
    session = np.asarray(
        config_scores["gated_session_centered_bce"], dtype=np.float64
    )
    routed = np.asarray(
        config_scores["gated_session_routed_hybrid"], dtype=np.float64
    )
    final_metrics = evaluate(data["val_user"], data["yv"], final_scores)
    gauc, ndcg5, primary = metric_values(final_metrics)
    ablation = {
        "probe_count_per_config": int(probe_count),
        "summaries": summaries,
        "session_minus_anchor_mean": float((session - anchor).mean()),
        "routed_hybrid_minus_anchor_mean": float((routed - anchor).mean()),
        "routed_hybrid_minus_session_mean": float((routed - session).mean()),
        "selected_config": winner_name,
        "final_prediction_seed": int(args.seed),
    }

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump({
            "gauc": gauc,
            "ndcg5": ndcg5,
            "primary": primary,
            "ablation": ablation,
            "history": history,
        }, fh)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w") as fh:
        fh.write("row_id,user_id,video_id,score\n")
        for i, score in enumerate(final_scores):
            fh.write(
                f"{i},{data['val_user'][i]},{data['val_video'][i]},{score:.9g}\n"
            )


if __name__ == "__main__":
    main()
