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


def metric_values(result):
    return {
        "gauc": float(result.get("GAUC", result.get("gauc"))),
        "ndcg5": float(result.get("nDCG@5", result.get("ndcg5"))),
        "primary": float(result["primary"]),
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
        with open(path, "r", newline="") as handle:
            reader = csv.DictReader(handle)
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
    durations = np.asarray([row["duration_ms"] for row in train_rows], dtype=np.float64)
    if len(durations):
        edges = np.unique(np.quantile(durations, np.linspace(0.1, 0.9, 9)))
    else:
        edges = np.asarray([], dtype=np.float64)

    field_names = ["user_id", "video_id", "author_id", "tab"]
    mappings = []
    for name in field_names:
        values = sorted({row[name] for row in train_rows})
        mappings.append({value: index + 1 for index, value in enumerate(values)})

    field_dims = [len(mapping) + 1 for mapping in mappings] + [10]
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int64)

    def encode(rows, training):
        n = len(rows)
        x = np.zeros((n, 5), dtype=np.int32)
        for column, name in enumerate(field_names):
            mapping = mappings[column]
            x[:, column] = np.asarray(
                [mapping.get(row[name], 0) for row in rows], dtype=np.int32
            ) + offsets[column]
        buckets = np.searchsorted(
            edges,
            np.asarray([row["duration_ms"] for row in rows], dtype=np.float64),
            side="right",
        )
        x[:, 4] = np.minimum(buckets, 9).astype(np.int32) + offsets[4]
        output = {
            "X": x,
            "y": np.asarray([row["long_view"] for row in rows], dtype=np.float32),
            "user": np.asarray([parse_int(row["user_id"]) for row in rows], dtype=np.int64),
            "hourmin": np.asarray([row["hourmin"] for row in rows], dtype=np.int32),
            "date": np.asarray([row["date"] for row in rows], dtype=np.int32),
            "duration_ms": np.asarray([row["duration_ms"] for row in rows], dtype=np.float32),
            "field_dims": np.asarray(field_dims, dtype=np.int64),
            "video_output": np.asarray([row["video_id"] for row in rows], dtype=object),
        }
        if training:
            output["play_time_ms"] = np.asarray(
                [row["play_time_ms"] for row in rows], dtype=np.float32
            )
        return output

    return encode(train_rows, True), encode(val_rows, False)


def load_data(data_dir):
    train_path = os.path.join(data_dir, "train.npz")
    val_path = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_path) and os.path.exists(val_path):
        train_file = np.load(train_path)
        val_file = np.load(val_path)
        train = {
            key: train_file[key]
            for key in train_file.files
            if key in {
                "X", "y", "user", "play_time_ms", "duration_ms", "hourmin", "date", "field_dims"
            }
        }
        val = {
            key: val_file[key]
            for key in val_file.files
            if key in {"X", "y", "user", "duration_ms", "hourmin", "date", "field_dims"}
        }
        train_file.close()
        val_file.close()
        dims = train["field_dims"].astype(np.int64)
        video_offset = int(dims[0])
        val["video_output"] = (val["X"][:, 1].astype(np.int64) - video_offset).astype(object)
        return train, val
    return load_csv_data(data_dir)


def date_ordinals(train_dates, val_dates):
    values = np.unique(np.concatenate([train_dates, val_dates]).astype(np.int64))
    ordinal = {}
    weekday = {}
    for value in values:
        try:
            day = datetime.datetime.strptime(str(int(value)), "%Y%m%d").date()
        except ValueError:
            day = datetime.date(1970, 1, 1) + datetime.timedelta(days=int(value) % 20000)
        ordinal[int(value)] = day.toordinal()
        weekday[int(value)] = day.weekday()
    return ordinal, weekday


def build_causal_features(train, val, history_len=12):
    base_dims = train["field_dims"].astype(np.int64)
    base_total = int(base_dims.sum())
    context_dims = np.asarray([24, 7, 2, 10, 21], dtype=np.int64)
    context_offsets = base_total + np.cumsum(
        np.concatenate([[0], context_dims[:-1]])
    ).astype(np.int64)
    total_categorical = base_total + int(context_dims.sum())
    padding_idx = total_categorical
    date_map, weekday_map = date_ordinals(train["date"], val["date"])
    histories = {}
    previous_minute = {}
    previous_position = {}
    gap_edges = np.asarray([0, 1, 2, 5, 10, 30, 60, 180, 1440], dtype=np.int64)

    def process(split):
        n = len(split["X"])
        explicit = np.zeros((n, 10), dtype=np.int64)
        explicit[:, :5] = split["X"].astype(np.int64)
        history = np.full((n, history_len), padding_idx, dtype=np.int64)
        history_mask = np.zeros((n, history_len), dtype=np.float32)
        hourmin = split["hourmin"].astype(np.int64)
        dates = split["date"].astype(np.int64)
        hours = np.clip(hourmin // 100, 0, 23)
        minutes = np.clip(hourmin % 100, 0, 59)
        absolute_minute = np.asarray(
            [
                date_map[int(day)] * 1440 + int(hour) * 60 + int(minute)
                for day, hour, minute in zip(dates, hours, minutes)
            ],
            dtype=np.int64,
        )
        order = np.lexsort((np.arange(n, dtype=np.int64), absolute_minute))
        for index in order:
            user = int(split["user"][index])
            current_minute = int(absolute_minute[index])
            prior = histories.get(user)
            if prior:
                values = list(prior)
                start = history_len - len(values)
                history[index, start:] = values
                history_mask[index, start:] = 1.0
            if user in previous_minute:
                gap = max(0, current_minute - previous_minute[user])
                gap_bucket = int(np.searchsorted(gap_edges, gap, side="right"))
                position = min(20, previous_position[user] + 1) if gap <= 30 else 0
            else:
                gap_bucket = 9
                position = 0
            local_tab = int(split["X"][index, 3] - int(base_dims[:3].sum()))
            explicit[index, 5] = int(hours[index]) + context_offsets[0]
            explicit[index, 6] = int(weekday_map[int(dates[index])]) + context_offsets[1]
            explicit[index, 7] = (1 if local_tab == 1 else 0) + context_offsets[2]
            explicit[index, 8] = min(gap_bucket, 9) + context_offsets[3]
            explicit[index, 9] = position + context_offsets[4]
            if prior is None:
                prior = deque(maxlen=history_len)
                histories[user] = prior
            prior.append(int(split["X"][index, 2]))
            previous_minute[user] = current_minute
            previous_position[user] = position
        return explicit, history, history_mask

    return process(train), process(val), total_categorical, padding_idx


def build_sse_probabilities(train, total_dim, padding_idx, max_probability=0.12):
    probabilities = np.zeros(total_dim + 1, dtype=np.float32)
    dims = train["field_dims"].astype(np.int64)
    user_start = 0
    user_end = int(dims[0])
    author_start = int(dims[:2].sum())
    author_end = author_start + int(dims[2])
    user_ids = train["X"][:, 0].astype(np.int64)
    author_ids = train["X"][:, 2].astype(np.int64)
    user_counts = np.bincount(user_ids, minlength=total_dim + 1).astype(np.float64)
    author_counts = np.bincount(author_ids, minlength=total_dim + 1).astype(np.float64)
    user_rows = np.arange(user_start, user_end, dtype=np.int64)
    author_rows = np.arange(author_start, author_end, dtype=np.int64)
    user_nonzero = user_counts[user_rows] > 0
    author_nonzero = author_counts[author_rows] > 0
    probabilities[user_rows[user_nonzero]] = (
        max_probability / np.sqrt(user_counts[user_rows[user_nonzero]])
    ).astype(np.float32)
    probabilities[author_rows[author_nonzero]] = (
        max_probability / np.sqrt(author_counts[author_rows[author_nonzero]])
    ).astype(np.float32)
    probabilities[0] = 0.0
    probabilities[author_start] = 0.0
    probabilities[padding_idx] = 0.0
    return torch.from_numpy(probabilities), 0, author_start


class SequenceDeepFM(torch.nn.Module):
    def __init__(self, total_dim, padding_idx, fields=10, k=16):
        super().__init__()
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
        torch.nn.init.zeros_(self.lin.weight)
        with torch.no_grad():
            self.emb.weight[padding_idx].zero_()
            self.lin.weight[padding_idx].zero_()

    def forward(self, x, history, history_mask):
        explicit_embedding = self.emb(x)
        history_embedding = self.emb(history)
        denominator = history_mask.sum(1, keepdim=True).clamp_min(1.0)
        pooled_history = (
            history_embedding * history_mask.unsqueeze(-1)
        ).sum(1) / denominator
        all_embedding = torch.cat(
            [explicit_embedding, pooled_history.unsqueeze(1)], dim=1
        )
        dropped = self.embedding_dropout(all_embedding)
        summed = dropped.sum(1)
        fm = 0.5 * (summed.square() - dropped.square().sum(1)).sum(1)
        explicit_linear = self.lin(x).sum((1, 2))
        history_linear = self.lin(history).squeeze(-1)
        history_linear = (history_linear * history_mask).sum(1) / denominator.squeeze(1)
        deep_state = self.deep(dropped.flatten(1))
        logits = (
            self.bias
            + explicit_linear
            + history_linear
            + fm
            + self.deep_head(deep_state).squeeze(1)
        )
        auxiliary = self.aux_head(deep_state).squeeze(1)
        return logits, auxiliary


def apply_frequency_adaptive_sse(x, history, probabilities, user_shared, author_shared, generator):
    x_sse = x.clone()
    history_sse = history.clone()
    user_ids = x_sse[:, 0]
    author_ids = x_sse[:, 2]
    user_probability = probabilities[user_ids]
    author_probability = probabilities[author_ids]
    user_mask = torch.rand(len(x_sse), generator=generator) < user_probability
    author_mask = torch.rand(len(x_sse), generator=generator) < author_probability
    x_sse[user_mask, 0] = user_shared
    x_sse[author_mask, 2] = author_shared
    history_probability = probabilities[history_sse]
    history_mask = torch.rand(history_sse.shape, generator=generator) < history_probability
    history_sse[history_mask] = author_shared
    return x_sse, history_sse


def predict(model, x, history, history_mask, device):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), 32768):
            end = start + 32768
            logits, _ = model(
                x[start:end].to(device),
                history[start:end].to(device),
                history_mask[start:end].to(device),
            )
            outputs.append(logits.detach().cpu().numpy())
    return np.concatenate(outputs)


def train_model(seed, epochs, train, val, train_features, val_features, total_dim,
                padding_idx, device, use_sse, sse_probabilities, user_shared,
                author_shared):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    x_train = torch.from_numpy(train_features[0])
    h_train = torch.from_numpy(train_features[1])
    m_train = torch.from_numpy(train_features[2])
    x_val = torch.from_numpy(val_features[0])
    h_val = torch.from_numpy(val_features[1])
    m_val = torch.from_numpy(val_features[2])
    y_train = torch.from_numpy(train["y"].astype(np.float32))
    duration = np.maximum(train["duration_ms"].astype(np.float32), 0.0)
    play = np.maximum(train["play_time_ms"].astype(np.float32), 0.0)
    observed = np.minimum(play, duration)
    auxiliary_target = torch.from_numpy((np.log1p(observed) / 10.0).astype(np.float32))
    censored = torch.from_numpy(((duration > 0) & (play >= duration)).astype(np.float32))
    model = SequenceDeepFM(total_dim, padding_idx).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-6)
    bce = torch.nn.BCEWithLogitsLoss()
    permutation_generator = torch.Generator().manual_seed(seed)
    sse_generator = torch.Generator().manual_seed(seed + 1000003)
    best_primary = -1.0
    best_scores = None
    patience = 0
    curve = []
    n = len(y_train)
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(n, generator=permutation_generator)
        loss_sum = 0.0
        bce_sum = 0.0
        auxiliary_sum = 0.0
        seen = 0
        for start in range(0, n, 8192):
            indices = permutation[start:start + 8192]
            xb = x_train[indices]
            hb = h_train[indices]
            mb = m_train[indices]
            if use_sse:
                xb, hb = apply_frequency_adaptive_sse(
                    xb,
                    hb,
                    sse_probabilities,
                    user_shared,
                    author_shared,
                    sse_generator,
                )
            xb = xb.to(device)
            hb = hb.to(device)
            mb = mb.to(device)
            yb = y_train[indices].to(device)
            tb = auxiliary_target[indices].to(device)
            cb = censored[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, auxiliary_prediction = model(xb, hb, mb)
            main_loss = bce(logits, yb)
            exact_error = (auxiliary_prediction - tb).square()
            censored_error = torch.relu(tb - auxiliary_prediction).square()
            auxiliary_loss = ((1.0 - cb) * exact_error + cb * censored_error).mean()
            loss = main_loss + 0.10 * auxiliary_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            batch_size = len(indices)
            loss_sum += float(loss.detach().cpu()) * batch_size
            bce_sum += float(main_loss.detach().cpu()) * batch_size
            auxiliary_sum += float(auxiliary_loss.detach().cpu()) * batch_size
            seen += batch_size
        scores = predict(model, x_val, h_val, m_val, device)
        metrics = metric_values(evaluate(val["user"], val["y"].astype(int), scores))
        curve.append({
            "epoch": epoch + 1,
            "train_loss": round(loss_sum / max(seen, 1), 6),
            "train_bce": round(bce_sum / max(seen, 1), 6),
            "train_aux": round(auxiliary_sum / max(seen, 1), 6),
            "val_gauc": round(metrics["gauc"], 6),
            "val_primary": round(metrics["primary"], 6),
        })
        if metrics["primary"] > best_primary + 1e-6:
            best_primary = metrics["primary"]
            best_scores = scores.copy()
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break
    return best_scores, best_primary, curve


def safe_correlation(left, right):
    if np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    epochs = args.epochs
    smoke_epochs = os.environ.get("SMOKE_EPOCHS")
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

    train, val = load_data(args.data_dir)
    train_features, val_features, total_dim, padding_idx = build_causal_features(
        train, val, history_len=12
    )
    sse_probabilities, user_shared, author_shared = build_sse_probabilities(
        train, total_dim, padding_idx, max_probability=0.12
    )
    seeds = [args.seed, args.seed + 1, args.seed + 2]
    parent_scores = []
    candidate_scores = []
    parent_primaries = []
    candidate_primaries = []
    history = []
    progress_path = os.path.join(args.out_dir, "progress.log")

    with open(progress_path, "w") as progress:
        for seed in seeds:
            parent_score, parent_primary, parent_curve = train_model(
                seed, epochs, train, val, train_features, val_features, total_dim,
                padding_idx, device, False, sse_probabilities, user_shared, author_shared
            )
            parent_scores.append(parent_score)
            parent_primaries.append(parent_primary)
            parent_record = {
                "config": "seq_deepfm_composite_uniform_regularization_parent",
                "seed": seed,
                "primary": float(parent_primary),
                "selected_epoch": int(np.argmax([row["val_primary"] for row in parent_curve]) + 1),
                "curve": parent_curve,
            }
            history.append(parent_record)
            progress.write(json.dumps({
                "config": parent_record["config"],
                "seed": seed,
                "primary": float(parent_primary),
            }) + "\n")
            progress.flush()

            candidate_score, candidate_primary, candidate_curve = train_model(
                seed, epochs, train, val, train_features, val_features, total_dim,
                padding_idx, device, True, sse_probabilities, user_shared, author_shared
            )
            assert not np.allclose(candidate_score, parent_score), "SSE candidate equals matched parent"
            candidate_scores.append(candidate_score)
            candidate_primaries.append(candidate_primary)
            candidate_record = {
                "config": "seq_deepfm_frequency_adaptive_sse",
                "seed": seed,
                "sse_max_probability": 0.12,
                "sse_probability_rule": "0.12/sqrt(train_frequency), user_and_author_only",
                "primary": float(candidate_primary),
                "matched_parent_primary": float(parent_primary),
                "matched_delta": float(candidate_primary - parent_primary),
                "parent_prediction_correlation": safe_correlation(candidate_score, parent_score),
                "selected_epoch": int(np.argmax([row["val_primary"] for row in candidate_curve]) + 1),
                "curve": candidate_curve,
            }
            history.append(candidate_record)
            progress.write(json.dumps({
                "config": candidate_record["config"],
                "seed": seed,
                "primary": float(candidate_primary),
                "matched_delta": float(candidate_primary - parent_primary),
            }) + "\n")
            progress.flush()

    for i in range(len(candidate_scores)):
        for j in range(i + 1, len(candidate_scores)):
            assert not np.allclose(candidate_scores[i], candidate_scores[j]), "Candidate ensemble members are identical"
        assert not np.allclose(candidate_scores[i], parent_scores[i]), "Candidate member equals parent member"

    final_scores = np.mean(np.stack(candidate_scores, axis=0), axis=0)
    parent_close = np.mean(np.stack(parent_scores, axis=0), axis=0)
    assert not np.allclose(final_scores, parent_close), "Candidate ensemble equals parent ensemble"
    final_metrics = metric_values(evaluate(val["user"], val["y"].astype(int), final_scores))
    parent_metrics = metric_values(evaluate(val["user"], val["y"].astype(int), parent_close))
    deltas = np.asarray(candidate_primaries) - np.asarray(parent_primaries)
    output = {
        "gauc": final_metrics["gauc"],
        "ndcg5": final_metrics["ndcg5"],
        "primary": final_metrics["primary"],
        "history": history,
        "summary": {
            "diagnosis": "learning-curve telemetry missing; low-confidence rare-ID overfit hypothesis",
            "seeds": seeds,
            "candidate_primary_mean": float(np.mean(candidate_primaries)),
            "candidate_primary_std": float(np.std(candidate_primaries, ddof=1)),
            "parent_primary_mean": float(np.mean(parent_primaries)),
            "parent_primary_std": float(np.std(parent_primaries, ddof=1)),
            "matched_deltas": [float(value) for value in deltas],
            "matched_delta_mean": float(np.mean(deltas)),
            "matched_delta_std": float(np.std(deltas, ddof=1)),
            "acceptance_threshold": 0.002,
            "threshold_passed": bool(float(np.mean(deltas)) >= 0.002),
            "candidate_mean_logit_primary": final_metrics["primary"],
            "parent_mean_logit_primary": parent_metrics["primary"],
            "ensemble_prediction_correlation_with_parent": safe_correlation(final_scores, parent_close),
            "ensemble_members": 3,
            "ensemble_rule": "matched consecutive-seed mean-logit",
        },
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as handle:
        json.dump(output, handle)
    with open(os.path.join(args.out_dir, "predictions.csv"), "w") as handle:
        handle.write("row_id,user_id,video_id,score\n")
        for index, score in enumerate(final_scores):
            handle.write(
                f"{index},{val['user'][index]},{val['video_output'][index]},{score:.8g}\n"
            )


if __name__ == "__main__":
    main()
