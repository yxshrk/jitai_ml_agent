import argparse
import csv
import datetime
import json
import os
import sys
from collections import deque

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from data.official.evaluate import evaluate
except ImportError:
    from harness.evaluate_provisional import evaluate


def metric_values(m):
    return {
        "gauc": float(m.get("GAUC", m.get("gauc"))),
        "ndcg5": float(m.get("nDCG@5", m.get("ndcg5"))),
        "primary": float(m["primary"]),
    }


def parse_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def load_csv_data(data_dir):
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")

    def read_rows(path, training):
        rows = []
        with open(path, "r", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                item = {
                    "user_id": row.get("user_id", "0"),
                    "video_id": row.get("video_id", "0"),
                    "author_id": row.get("author_id", row.get("video_id", "0")),
                    "tab": row.get("tab", "0"),
                    "hourmin": parse_int(row.get("hourmin", 0)),
                    "date": parse_int(row.get("date", 19700101), 19700101),
                    "duration_ms": float(row.get("duration_ms", 0) or 0),
                    "long_view": float(row.get("long_view", 0) or 0),
                }
                if training:
                    item["play_time_ms"] = float(row.get("play_time_ms", 0) or 0)
                rows.append(item)
        return rows

    train_rows = read_rows(train_path, True)
    val_rows = read_rows(val_path, False)
    durations = np.asarray([r["duration_ms"] for r in train_rows], dtype=np.float64)
    edges = np.unique(np.quantile(durations, np.linspace(0.1, 0.9, 9))) if len(durations) else np.array([])

    field_names = ["user_id", "video_id", "author_id", "tab"]
    mappings = []
    for name in field_names:
        values = sorted({r[name] for r in train_rows})
        mappings.append({value: i + 1 for i, value in enumerate(values)})

    field_dims = [len(m) + 1 for m in mappings]
    field_dims.append(10)
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int64)

    def encode(rows, training):
        n = len(rows)
        X = np.zeros((n, 5), dtype=np.int32)
        for j, name in enumerate(field_names):
            mapping = mappings[j]
            X[:, j] = np.asarray([mapping.get(r[name], 0) for r in rows], dtype=np.int32) + offsets[j]
        local_bucket = np.searchsorted(edges, np.asarray([r["duration_ms"] for r in rows]), side="right")
        local_bucket = np.minimum(local_bucket, 9)
        X[:, 4] = local_bucket.astype(np.int32) + offsets[4]
        out = {
            "X": X,
            "y": np.asarray([r["long_view"] for r in rows], dtype=np.float32),
            "user": np.asarray([parse_int(r["user_id"]) for r in rows], dtype=np.int64),
            "hourmin": np.asarray([r["hourmin"] for r in rows], dtype=np.int32),
            "date": np.asarray([r["date"] for r in rows], dtype=np.int32),
            "duration_ms": np.asarray([r["duration_ms"] for r in rows], dtype=np.float32),
            "field_dims": np.asarray(field_dims, dtype=np.int64),
            "video_output": np.asarray([r["video_id"] for r in rows], dtype=object),
        }
        if training:
            out["play_time_ms"] = np.asarray([r["play_time_ms"] for r in rows], dtype=np.float32)
        return out

    return encode(train_rows, True), encode(val_rows, False)


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        tr_file = np.load(train_npz)
        va_file = np.load(val_npz)
        tr = {key: tr_file[key] for key in tr_file.files if key in {
            "X", "y", "user", "play_time_ms", "duration_ms", "hourmin", "date", "field_dims"
        }}
        va = {key: va_file[key] for key in va_file.files if key in {
            "X", "y", "user", "duration_ms", "hourmin", "date", "field_dims"
        }}
        tr_file.close()
        va_file.close()
        va["video_output"] = np.zeros(len(va["y"]), dtype=np.int64)
        return tr, va
    return load_csv_data(data_dir)


def date_ordinals(train_dates, val_dates):
    unique_dates = np.unique(np.concatenate([train_dates, val_dates]).astype(np.int64))
    mapping = {}
    weekdays = {}
    for value in unique_dates:
        text = str(int(value))
        try:
            day = datetime.datetime.strptime(text, "%Y%m%d").date()
        except ValueError:
            day = datetime.date(1970, 1, 1) + datetime.timedelta(days=int(value) % 20000)
        mapping[int(value)] = day.toordinal()
        weekdays[int(value)] = day.weekday()
    return mapping, weekdays


def build_causal_features(tr, va, history_len=12):
    base_dims = tr["field_dims"].astype(np.int64)
    base_total = int(base_dims.sum())
    context_dims = np.asarray([24, 7, 2, 10, 21], dtype=np.int64)
    context_offsets = base_total + np.cumsum(np.concatenate([[0], context_dims[:-1]])).astype(np.int64)
    total_categorical = base_total + int(context_dims.sum())
    padding_idx = total_categorical
    date_map, weekday_map = date_ordinals(tr["date"], va["date"])
    histories = {}
    previous_minute = {}
    previous_position = {}
    gap_edges = np.asarray([0, 1, 2, 5, 10, 30, 60, 180, 1440], dtype=np.int64)

    def process(split):
        n = len(split["X"])
        explicit = np.zeros((n, 10), dtype=np.int64)
        explicit[:, :5] = split["X"].astype(np.int64)
        hist = np.full((n, history_len), padding_idx, dtype=np.int64)
        hist_mask = np.zeros((n, history_len), dtype=np.float32)
        hourmin = split["hourmin"].astype(np.int64)
        dates = split["date"].astype(np.int64)
        hours = np.clip(hourmin // 100, 0, 23)
        minutes = np.clip(hourmin % 100, 0, 59)
        absolute_minute = np.asarray(
            [date_map[int(d)] * 1440 + int(h) * 60 + int(m) for d, h, m in zip(dates, hours, minutes)],
            dtype=np.int64,
        )
        order = np.lexsort((np.arange(n, dtype=np.int64), absolute_minute))
        for idx in order:
            user = int(split["user"][idx])
            current_minute = int(absolute_minute[idx])
            prior = histories.get(user)
            if prior:
                values = list(prior)
                start = history_len - len(values)
                hist[idx, start:] = values
                hist_mask[idx, start:] = 1.0
            if user in previous_minute:
                gap = max(0, current_minute - previous_minute[user])
                gap_bucket = int(np.searchsorted(gap_edges, gap, side="right"))
                if gap <= 30:
                    position = min(20, previous_position[user] + 1)
                else:
                    position = 0
            else:
                gap_bucket = 9
                position = 0
            local_tab = int(split["X"][idx, 3] - int(base_dims[:3].sum()))
            is_rand = 1 if local_tab == 1 else 0
            explicit[idx, 5] = int(hours[idx]) + context_offsets[0]
            explicit[idx, 6] = int(weekday_map[int(dates[idx])]) + context_offsets[1]
            explicit[idx, 7] = is_rand + context_offsets[2]
            explicit[idx, 8] = min(gap_bucket, 9) + context_offsets[3]
            explicit[idx, 9] = position + context_offsets[4]
            if prior is None:
                prior = deque(maxlen=history_len)
                histories[user] = prior
            prior.append(int(split["X"][idx, 2]))
            previous_minute[user] = current_minute
            previous_position[user] = position
        return explicit, hist, hist_mask

    train_features = process(tr)
    val_features = process(va)
    return train_features, val_features, total_categorical, padding_idx


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


class SequenceDeepFM(torch.nn.Module):
    def __init__(self, total_dim, padding_idx, fields=10, k=16):
        super().__init__()
        self.padding_idx = padding_idx
        self.emb = torch.nn.Embedding(total_dim + 1, k, padding_idx=padding_idx)
        self.lin = torch.nn.Embedding(total_dim + 1, 1, padding_idx=padding_idx)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.embedding_dropout = torch.nn.Dropout(0.10)
        self.deep = torch.nn.Sequential(
            torch.nn.Linear((fields + 1) * k, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.20),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.20),
        )
        self.deep_head = torch.nn.Linear(64, 1)
        self.aux_head = torch.nn.Linear(64, 1)
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        with torch.no_grad():
            self.emb.weight[padding_idx].zero_()
        torch.nn.init.zeros_(self.lin.weight)

    def forward(self, x, history, history_mask):
        explicit_e = self.emb(x)
        history_e = self.emb(history)
        denominator = history_mask.sum(1, keepdim=True).clamp_min(1.0)
        pooled_history = (history_e * history_mask.unsqueeze(-1)).sum(1) / denominator
        all_e = torch.cat([explicit_e, pooled_history.unsqueeze(1)], dim=1)
        dropped_e = self.embedding_dropout(all_e)
        summed = dropped_e.sum(1)
        fm = 0.5 * (summed.square() - dropped_e.square().sum(1)).sum(1)
        explicit_linear = self.lin(x).sum((1, 2))
        history_linear = self.lin(history).squeeze(-1)
        history_linear = (history_linear * history_mask).sum(1) / denominator.squeeze(1)
        deep_state = self.deep(dropped_e.flatten(1))
        logit = self.bias + explicit_linear + history_linear + fm + self.deep_head(deep_state).squeeze(1)
        aux = self.aux_head(deep_state).squeeze(1)
        return logit, aux


def predict_fm(model, X, device):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(X), 65536):
            xb = X[start:start + 65536].to(device)
            outputs.append(model(xb).detach().cpu().numpy())
    return np.concatenate(outputs)


def predict_sequence(model, X, H, M, device):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(X), 32768):
            end = start + 32768
            xb = X[start:end].to(device)
            hb = H[start:end].to(device)
            mb = M[start:end].to(device)
            logits, _ = model(xb, hb, mb)
            outputs.append(logits.detach().cpu().numpy())
    return np.concatenate(outputs)


def train_baseline(seed, epochs, tr, va, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    X_train = torch.from_numpy(tr["X"].astype(np.int64))
    y_train = torch.from_numpy(tr["y"].astype(np.float32))
    X_val = torch.from_numpy(va["X"].astype(np.int64))
    model = FM(int(tr["field_dims"].sum()), 16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(seed)
    best_primary = -1.0
    best_scores = None
    patience = 0
    curve = []
    n = len(y_train)
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(n, generator=generator)
        loss_sum = 0.0
        seen = 0
        for start in range(0, n, 8192):
            idx = permutation[start:start + 8192]
            xb = X_train[idx].to(device)
            yb = y_train[idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(idx)
            seen += len(idx)
        scores = predict_fm(model, X_val, device)
        values = metric_values(evaluate(va["user"], va["y"].astype(int), scores))
        curve.append({
            "epoch": epoch + 1,
            "train_loss": round(loss_sum / max(seen, 1), 6),
            "val_gauc": round(values["gauc"], 6),
            "val_primary": round(values["primary"], 6),
        })
        if values["primary"] > best_primary + 1e-6:
            best_primary = values["primary"]
            best_scores = scores.copy()
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break
    return best_scores, best_primary, curve


def train_composite(seed, epochs, tr, va, train_features, val_features, total_dim, padding_idx, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    X_train = torch.from_numpy(train_features[0])
    H_train = torch.from_numpy(train_features[1])
    M_train = torch.from_numpy(train_features[2])
    X_val = torch.from_numpy(val_features[0])
    H_val = torch.from_numpy(val_features[1])
    M_val = torch.from_numpy(val_features[2])
    y_train = torch.from_numpy(tr["y"].astype(np.float32))
    duration = np.maximum(tr["duration_ms"].astype(np.float32), 0.0)
    play = np.maximum(tr["play_time_ms"].astype(np.float32), 0.0)
    observed = np.minimum(play, duration)
    aux_target = torch.from_numpy((np.log1p(observed) / 10.0).astype(np.float32))
    censored = torch.from_numpy(((duration > 0) & (play >= duration)).astype(np.float32))
    model = SequenceDeepFM(total_dim, padding_idx).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-6)
    bce = torch.nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(seed)
    best_primary = -1.0
    best_scores = None
    patience = 0
    curve = []
    n = len(y_train)
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(n, generator=generator)
        total_loss_sum = 0.0
        main_loss_sum = 0.0
        aux_loss_sum = 0.0
        seen = 0
        for start in range(0, n, 8192):
            idx = permutation[start:start + 8192]
            xb = X_train[idx].to(device)
            hb = H_train[idx].to(device)
            mb = M_train[idx].to(device)
            yb = y_train[idx].to(device)
            tb = aux_target[idx].to(device)
            cb = censored[idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, aux_prediction = model(xb, hb, mb)
            main_loss = bce(logits, yb)
            exact_error = (aux_prediction - tb).square()
            censored_error = torch.relu(tb - aux_prediction).square()
            aux_loss = ((1.0 - cb) * exact_error + cb * censored_error).mean()
            loss = main_loss + 0.10 * aux_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            batch_size = len(idx)
            total_loss_sum += float(loss.detach().cpu()) * batch_size
            main_loss_sum += float(main_loss.detach().cpu()) * batch_size
            aux_loss_sum += float(aux_loss.detach().cpu()) * batch_size
            seen += batch_size
        scores = predict_sequence(model, X_val, H_val, M_val, device)
        values = metric_values(evaluate(va["user"], va["y"].astype(int), scores))
        curve.append({
            "epoch": epoch + 1,
            "train_loss": round(total_loss_sum / max(seen, 1), 6),
            "train_bce": round(main_loss_sum / max(seen, 1), 6),
            "train_aux": round(aux_loss_sum / max(seen, 1), 6),
            "val_gauc": round(values["gauc"], 6),
            "val_primary": round(values["primary"], 6),
        })
        if values["primary"] > best_primary + 1e-6:
            best_primary = values["primary"]
            best_scores = scores.copy()
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break
    return best_scores, best_primary, curve


def safe_correlation(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    smoke_epochs = os.environ.get("SMOKE_EPOCHS")
    epochs = args.epochs
    if smoke_epochs is not None:
        epochs = min(epochs, max(1, int(smoke_epochs)))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        device = torch.device("cpu")
    tr, va = load_data(args.data_dir)
    train_features, val_features, total_dim, padding_idx = build_causal_features(tr, va, 12)
    seeds = [args.seed, args.seed + 1, args.seed + 2]
    baseline_scores = []
    composite_scores = []
    run_history = []
    progress_path = os.path.join(args.out_dir, "progress.log")
    with open(progress_path, "w") as progress:
        for seed in seeds:
            base_score, base_primary, base_curve = train_baseline(seed, epochs, tr, va, device)
            baseline_scores.append(base_score)
            record = {
                "config": "matched_fm_parent",
                "seed": seed,
                "primary": float(base_primary),
                "selected_epoch": int(np.argmax([x["val_primary"] for x in base_curve]) + 1),
                "curve": base_curve,
            }
            run_history.append(record)
            progress.write(json.dumps({"config": "matched_fm_parent", "seed": seed, "primary": base_primary}) + "\n")
            progress.flush()
            comp_score, comp_primary, comp_curve = train_composite(
                seed, epochs, tr, va, train_features, val_features, total_dim, padding_idx, device
            )
            composite_scores.append(comp_score)
            assert not np.allclose(comp_score, base_score), "Composite member equals matched parent predictions"
            corr = safe_correlation(comp_score, base_score)
            record = {
                "config": "seq_deepfm_composite_full",
                "seed": seed,
                "primary": float(comp_primary),
                "matched_parent_primary": float(base_primary),
                "matched_delta": float(comp_primary - base_primary),
                "parent_prediction_correlation": corr,
                "selected_epoch": int(np.argmax([x["val_primary"] for x in comp_curve]) + 1),
                "curve": comp_curve,
            }
            run_history.append(record)
            progress.write(json.dumps({
                "config": "seq_deepfm_composite_full",
                "seed": seed,
                "primary": comp_primary,
                "matched_delta": comp_primary - base_primary,
            }) + "\n")
            progress.flush()
    for i in range(len(composite_scores)):
        for j in range(i + 1, len(composite_scores)):
            assert not np.allclose(composite_scores[i], composite_scores[j]), "Ensemble members are identical"
    final_scores = np.mean(np.stack(composite_scores, axis=0), axis=0)
    parent_close = np.mean(np.stack(baseline_scores, axis=0), axis=0)
    assert not np.allclose(final_scores, parent_close), "Final ensemble equals parent ensemble"
    final_metrics = metric_values(evaluate(va["user"], va["y"].astype(int), final_scores))
    parent_metrics = metric_values(evaluate(va["user"], va["y"].astype(int), parent_close))
    member_primaries = [
        metric_values(evaluate(va["user"], va["y"].astype(int), scores))["primary"]
        for scores in composite_scores
    ]
    parent_primaries = [
        metric_values(evaluate(va["user"], va["y"].astype(int), scores))["primary"]
        for scores in baseline_scores
    ]
    deltas = np.asarray(member_primaries) - np.asarray(parent_primaries)
    output_metrics = {
        "gauc": final_metrics["gauc"],
        "ndcg5": final_metrics["ndcg5"],
        "primary": final_metrics["primary"],
        "history": run_history,
        "summary": {
            "seeds": seeds,
            "member_primary_mean": float(np.mean(member_primaries)),
            "member_primary_std": float(np.std(member_primaries, ddof=1)),
            "matched_parent_primary_mean": float(np.mean(parent_primaries)),
            "matched_parent_primary_std": float(np.std(parent_primaries, ddof=1)),
            "matched_deltas": [float(x) for x in deltas],
            "matched_delta_mean": float(np.mean(deltas)),
            "matched_delta_std": float(np.std(deltas, ddof=1)),
            "parent_mean_logit_primary": parent_metrics["primary"],
            "ensemble_prediction_correlation_with_parent": safe_correlation(final_scores, parent_close),
            "ensemble_members": 3,
            "ensemble_rule": "predeclared consecutive-seed mean-logit",
        },
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(output_metrics, fh)
    with open(os.path.join(args.out_dir, "predictions.csv"), "w") as fh:
        fh.write("row_id,user_id,video_id,score\n")
        video_output = va["video_output"]
        for idx, score in enumerate(final_scores):
            fh.write(f"{idx},{va['user'][idx]},{video_output[idx]},{score:.8g}\n")


if __name__ == "__main__":
    main()
