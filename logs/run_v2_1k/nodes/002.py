import argparse
import csv
import datetime
import json
import math
import os
import random
import warnings
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

warnings.filterwarnings("ignore")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def date_ordinals(values):
    vals = np.asarray(values)
    unique = np.unique(vals)
    mapping = {}
    for value in unique:
        text = str(int(value))
        try:
            day = datetime.datetime.strptime(text, "%Y%m%d").date().toordinal()
        except ValueError:
            day = int(value)
        mapping[int(value)] = day
    return np.asarray([mapping[int(v)] for v in vals], dtype=np.int32)


def session_features(train_user, val_user, train_date, val_date, train_hourmin,
                     val_hourmin, train_x, val_x, field_dims):
    users = np.concatenate([train_user, val_user])
    dates_raw = np.concatenate([train_date, val_date]).astype(np.int64)
    hourmins = np.concatenate([train_hourmin, val_hourmin]).astype(np.int64)
    dates = date_ordinals(dates_raw)
    hours = np.clip(hourmins // 100, 0, 23).astype(np.int32)
    absolute_hour = dates.astype(np.int64) * 24 + hours.astype(np.int64)
    row = np.arange(len(users), dtype=np.int64)
    order = np.lexsort((row, absolute_hour, users))
    sorted_user = users[order]
    sorted_abs = absolute_hour[order]

    same_user = np.zeros(len(order), dtype=bool)
    same_user[1:] = sorted_user[1:] == sorted_user[:-1]
    gap = np.full(len(order), 10 ** 9, dtype=np.int64)
    gap[1:] = np.where(same_user[1:], sorted_abs[1:] - sorted_abs[:-1], 10 ** 9)
    gap = np.maximum(gap, 0)
    gap_bucket_sorted = np.searchsorted(
        np.asarray([1, 2, 4, 8, 24, 96, 192], dtype=np.int64), gap, side="right"
    ).astype(np.int64)

    same_session = np.zeros(len(order), dtype=bool)
    same_session[1:] = same_user[1:] & (sorted_abs[1:] == sorted_abs[:-1])
    starts = ~same_session
    indices = np.arange(len(order), dtype=np.int64)
    start_index = np.maximum.accumulate(np.where(starts, indices, 0))
    position = indices - start_index
    position_bucket_sorted = np.select(
        [position == 0, position == 1, position == 2, position == 3,
         position <= 5, position <= 8],
        [0, 1, 2, 3, 4, 5], default=6
    ).astype(np.int64)

    gap_bucket = np.empty(len(order), dtype=np.int64)
    position_bucket = np.empty(len(order), dtype=np.int64)
    gap_bucket[order] = gap_bucket_sorted
    position_bucket[order] = position_bucket_sorted

    all_x = np.concatenate([train_x, val_x], axis=0)
    offsets = np.cumsum(np.concatenate([[0], np.asarray(field_dims[:-1])])).astype(np.int64)
    tab = all_x[:, 3].astype(np.int64) - offsets[3]
    duration = all_x[:, 4].astype(np.int64) - offsets[4]
    tab_dim = int(field_dims[3])
    dur_dim = int(field_dims[4])
    pos_tab = position_bucket * tab_dim + tab
    pos_duration = position_bucket * dur_dim + duration
    hour_tab = hours.astype(np.int64) * tab_dim + tab
    features = np.stack(
        [gap_bucket, position_bucket, hours.astype(np.int64), pos_tab,
         pos_duration, hour_tab], axis=1
    ).astype(np.int64)
    dimensions = [8, 7, 24, 7 * tab_dim, 7 * dur_dim, 24 * tab_dim]
    split = len(train_user)
    return features[:split], features[split:], dimensions, dates[:split]


def encode_column(train_values, val_values):
    mapping = {}
    encoded_train = np.empty(len(train_values), dtype=np.int64)
    for i, value in enumerate(train_values):
        key = str(value)
        if key not in mapping:
            mapping[key] = len(mapping)
        encoded_train[i] = mapping[key]
    unknown = len(mapping)
    encoded_val = np.asarray([mapping.get(str(v), unknown) for v in val_values], dtype=np.int64)
    return encoded_train, encoded_val, unknown + 1


def load_csv(data_dir):
    def read(path, training):
        result = {k: [] for k in ["user_id", "video_id", "tab", "hourmin", "date",
                                    "duration_ms", "long_view"]}
        with open(path, "r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                result["user_id"].append(row["user_id"])
                result["video_id"].append(row["video_id"])
                result["tab"].append(row["tab"])
                result["hourmin"].append(int(float(row["hourmin"])))
                result["date"].append(int(float(row["date"])))
                result["duration_ms"].append(float(row["duration_ms"]))
                result["long_view"].append(float(row["long_view"]))
        return result

    tr = read(Path(data_dir) / "train.csv", True)
    va = read(Path(data_dir) / "val.csv", False)
    train_fields = []
    val_fields = []
    dimensions = []
    for name in ["user_id", "video_id"]:
        a, b, dim = encode_column(tr[name], va[name])
        train_fields.append(a)
        val_fields.append(b)
        dimensions.append(dim)
    author_train = np.zeros(len(tr["user_id"]), dtype=np.int64)
    author_val = np.zeros(len(va["user_id"]), dtype=np.int64)
    train_fields.append(author_train)
    val_fields.append(author_val)
    dimensions.append(1)
    tab_train, tab_val, tab_dim = encode_column(tr["tab"], va["tab"])
    train_fields.append(tab_train)
    val_fields.append(tab_val)
    dimensions.append(tab_dim)
    quantiles = np.quantile(np.asarray(tr["duration_ms"], dtype=np.float64), np.linspace(0.1, 0.9, 9))
    quantiles = np.unique(quantiles)
    dur_train = np.searchsorted(quantiles, np.asarray(tr["duration_ms"]), side="right")
    dur_val = np.searchsorted(quantiles, np.asarray(va["duration_ms"]), side="right")
    train_fields.append(dur_train)
    val_fields.append(dur_val)
    dimensions.append(len(quantiles) + 1)
    offsets = np.cumsum(np.concatenate([[0], np.asarray(dimensions[:-1])])).astype(np.int64)
    train_x = np.stack(train_fields, axis=1).astype(np.int64) + offsets
    val_x = np.stack(val_fields, axis=1).astype(np.int64) + offsets
    return {
        "train_x": train_x,
        "val_x": val_x,
        "train_y": np.asarray(tr["long_view"], dtype=np.float32),
        "val_y": np.asarray(va["long_view"], dtype=np.float32),
        "train_user": np.asarray(tr["user_id"]),
        "val_user": np.asarray(va["user_id"]),
        "val_video": np.asarray(va["video_id"]),
        "train_date": np.asarray(tr["date"], dtype=np.int64),
        "val_date": np.asarray(va["date"], dtype=np.int64),
        "train_hourmin": np.asarray(tr["hourmin"], dtype=np.int64),
        "val_hourmin": np.asarray(va["hourmin"], dtype=np.int64),
        "field_dims": np.asarray(dimensions, dtype=np.int64),
        "fast": False,
    }


def load_data(data_dir):
    train_npz = Path(data_dir) / "train.npz"
    val_npz = Path(data_dir) / "val.npz"
    if not (train_npz.exists() and val_npz.exists()):
        return load_csv(data_dir)
    with np.load(train_npz, allow_pickle=False) as tr, np.load(val_npz, allow_pickle=False) as va:
        field_dims = np.asarray(tr["field_dims"] if "field_dims" in tr else va["field_dims"], dtype=np.int64)
        offsets = np.cumsum(np.concatenate([[0], field_dims[:-1]])).astype(np.int64)
        val_video = np.asarray(va["X"][:, 1], dtype=np.int64) - offsets[1]
        return {
            "train_x": np.asarray(tr["X"], dtype=np.int64),
            "val_x": np.asarray(va["X"], dtype=np.int64),
            "train_y": np.asarray(tr["y"], dtype=np.float32),
            "val_y": np.asarray(va["y"], dtype=np.float32),
            "train_user": np.asarray(tr["user"]),
            "val_user": np.asarray(va["user"]),
            "val_video": val_video,
            "train_date": np.asarray(tr["date"], dtype=np.int64),
            "val_date": np.asarray(va["date"], dtype=np.int64),
            "train_hourmin": np.asarray(tr["hourmin"], dtype=np.int64),
            "val_hourmin": np.asarray(va["hourmin"], dtype=np.int64),
            "field_dims": field_dims,
            "fast": True,
        }


class ChampionWithSession(nn.Module):
    def __init__(self, field_dims, session_dims, variant, k=24, dropout=0.21):
        super().__init__()
        self.variant = variant
        total = int(np.sum(field_dims))
        self.embedding = nn.Embedding(total, k)
        self.linear = nn.Embedding(total, 1)
        input_dim = len(field_dims) * k
        self.cross_w = nn.ParameterList([nn.Parameter(torch.empty(input_dim)) for _ in range(2)])
        self.cross_b = nn.ParameterList([nn.Parameter(torch.zeros(input_dim)) for _ in range(2)])
        self.base_mlp = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1)
        )
        self.cross_out = nn.Linear(input_dim, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        if variant != "anchor":
            used = 3 if variant == "session" else 6
            self.session_embeddings = nn.ModuleList([
                nn.Embedding(int(session_dims[i]), 6) for i in range(used)
            ])
            session_input = used * 6
            self.session_tower = nn.Sequential(
                nn.Linear(session_input, 32), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(32, 16), nn.ReLU(), nn.Dropout(dropout), nn.Linear(16, 1)
            )
            nn.init.zeros_(self.session_tower[-1].weight)
            nn.init.zeros_(self.session_tower[-1].bias)
            self.gate_logit = nn.Parameter(torch.tensor(-2.1972246))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.zeros_(self.linear.weight)
        for weight in self.cross_w:
            nn.init.normal_(weight, std=0.01)
        for module in self.base_mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.cross_out.weight)
        nn.init.zeros_(self.cross_out.bias)
        if self.variant != "anchor":
            for emb in self.session_embeddings:
                nn.init.normal_(emb.weight, std=0.01)
            nn.init.zeros_(self.session_tower[-1].weight)
            nn.init.zeros_(self.session_tower[-1].bias)

    def forward(self, x, session):
        embedded = self.embedding(x).flatten(1)
        crossed = embedded
        for weight, bias in zip(self.cross_w, self.cross_b):
            scalar = torch.sum(crossed * weight, dim=1, keepdim=True)
            crossed = embedded * scalar + bias + crossed
        linear = self.linear(x).sum(dim=1).squeeze(1)
        base = linear + self.base_mlp(embedded).squeeze(1) + self.cross_out(crossed).squeeze(1) + self.bias
        if self.variant == "anchor":
            return base, base
        parts = [emb(session[:, i]) for i, emb in enumerate(self.session_embeddings)]
        residual = self.session_tower(torch.cat(parts, dim=1)).squeeze(1)
        residual = torch.tanh(residual)
        total = base + torch.sigmoid(self.gate_logit) * residual
        return base, total


def make_pairs(users, labels, rng):
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    positives = []
    negatives = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        idx = order[left:right]
        pos = idx[labels[idx] > 0.5]
        neg = idx[labels[idx] <= 0.5]
        if len(pos) and len(neg):
            positives.append(pos)
            negatives.append(rng.choice(neg, size=len(pos), replace=True))
    if not positives:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    p = np.concatenate(positives)
    n = np.concatenate(negatives)
    permutation = rng.permutation(len(p))
    return p[permutation], n[permutation]


def metric_eval(users, labels, scores, fast):
    if fast:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    result = evaluate(users, labels, scores)
    return {
        "gauc": float(result.get("GAUC", result.get("gauc"))),
        "ndcg5": float(result.get("nDCG@5", result.get("ndcg5"))),
        "primary": float(result.get("primary")),
    }


def predict(model, x, session, device, batch_size):
    model.eval()
    output = np.empty(len(x), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            xb = torch.as_tensor(x[start:end], dtype=torch.long, device=device)
            sb = torch.as_tensor(session[start:end], dtype=torch.long, device=device)
            _, score = model(xb, sb)
            output[start:end] = score.detach().cpu().numpy()
    return output


def train_one(data, train_session, val_session, session_dims, variant, seed, epochs, device):
    seed_all(seed)
    model = ChampionWithSession(data["field_dims"], session_dims, variant).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00168, weight_decay=0.000037)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    n = len(data["train_y"])
    batch_size = 8192 if device.type == "cuda" else 4096
    steps = int(math.ceil(n / batch_size))
    rng = np.random.default_rng(seed)
    pair_pos, pair_neg = make_pairs(data["train_user"], data["train_y"], rng)
    train_days = date_ordinals(data["train_date"])
    age = np.max(train_days) - train_days
    recency = np.exp(-math.log(2.0) * age / 7.0).astype(np.float32)
    recency /= max(float(recency.mean()), 1e-8)
    best_metric = None
    best_prediction = None
    best_epoch = 0.0

    for epoch in range(epochs):
        permutation = rng.permutation(n)
        if len(pair_pos):
            pair_permutation = rng.permutation(len(pair_pos))
        model.train()
        for step in range(steps):
            left = step * batch_size
            right = min(left + batch_size, n)
            idx = permutation[left:right]
            xb = torch.as_tensor(data["train_x"][idx], dtype=torch.long, device=device)
            sb = torch.as_tensor(train_session[idx], dtype=torch.long, device=device)
            yb = torch.as_tensor(data["train_y"][idx], dtype=torch.float32, device=device)
            wb = torch.as_tensor(recency[idx], dtype=torch.float32, device=device)
            _, total = model(xb, sb)
            point = F.binary_cross_entropy_with_logits(total, yb, reduction="none")
            point = torch.sum(point * wb) / torch.sum(wb)
            pair_loss = total.new_tensor(0.0)
            if len(pair_pos):
                pleft = (step * batch_size) % len(pair_pos)
                take = min(right - left, len(pair_pos))
                pair_idx = pair_permutation[np.arange(pleft, pleft + take) % len(pair_pos)]
                pi = pair_pos[pair_idx]
                ni = pair_neg[pair_idx]
                px = torch.as_tensor(data["train_x"][pi], dtype=torch.long, device=device)
                nx = torch.as_tensor(data["train_x"][ni], dtype=torch.long, device=device)
                ps = torch.as_tensor(train_session[pi], dtype=torch.long, device=device)
                ns = torch.as_tensor(train_session[ni], dtype=torch.long, device=device)
                positive_base, _ = model(px, ps)
                negative_base, _ = model(nx, ns)
                pair_loss = F.softplus(-(positive_base - negative_base)).mean()
            loss = 0.5 * point + 0.5 * pair_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if step + 1 == max(1, steps // 2) or step + 1 == steps:
                candidate = predict(model, data["val_x"], val_session, device, batch_size)
                metrics = metric_eval(data["val_user"], data["val_y"], candidate, data["fast"])
                fraction = epoch + (step + 1) / steps
                if best_metric is None or metrics["primary"] > best_metric["primary"]:
                    best_metric = metrics
                    best_prediction = candidate.copy()
                    best_epoch = fraction
                model.train()
        scheduler.step()
    best_metric = dict(best_metric)
    best_metric["best_epoch"] = float(best_epoch)
    return best_prediction, best_metric


def within_user_rank_average(users, predictions):
    result = np.zeros(len(users), dtype=np.float64)
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    for prediction in predictions:
        ranked = np.empty(len(users), dtype=np.float64)
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            idx = order[left:right]
            local_order = np.argsort(prediction[idx], kind="mergesort")
            local_rank = np.empty(len(idx), dtype=np.float64)
            local_rank[local_order] = np.arange(len(idx), dtype=np.float64)
            if len(idx) > 1:
                local_rank /= len(idx) - 1
            ranked[idx] = local_rank
        result += ranked
    return result / len(predictions)


def append_progress(path, record):
    with open(path, "a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.log"
    if progress_path.exists():
        progress_path.unlink()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(args.seed)
    data = load_data(args.data_dir)
    train_session, val_session, session_dims, _ = session_features(
        data["train_user"], data["val_user"], data["train_date"], data["val_date"],
        data["train_hourmin"], data["val_hourmin"], data["train_x"], data["val_x"],
        data["field_dims"]
    )
    smoke = os.environ.get("SMOKE_EPOCHS")
    epochs = 6
    if smoke is not None:
        epochs = max(1, min(epochs, int(smoke)))
    replicates = 1 if smoke is not None else 14
    final_members = 1 if smoke is not None else 9
    variants = ["anchor", "session", "session_pair"]
    history = []
    variant_scores = {name: [] for name in variants}

    for variant_index, variant in enumerate(variants):
        for replicate in range(replicates):
            run_seed = args.seed + 1009 * variant_index + 37 * replicate
            _, metrics = train_one(data, train_session, val_session, session_dims,
                                   variant, run_seed, epochs, device)
            record = {"phase": "ablation", "variant": variant, "seed": run_seed, **metrics}
            history.append(record)
            variant_scores[variant].append(metrics["primary"])
            append_progress(progress_path, record)

    winning_variant = max(variants, key=lambda name: float(np.mean(variant_scores[name])))
    final_predictions = []
    for member in range(final_members):
        run_seed = args.seed + 50021 + 7919 * member
        prediction, metrics = train_one(data, train_session, val_session, session_dims,
                                        winning_variant, run_seed, epochs, device)
        final_predictions.append(prediction)
        record = {"phase": "final_ensemble", "variant": winning_variant,
                  "seed": run_seed, **metrics}
        history.append(record)
        append_progress(progress_path, record)

    scores = within_user_rank_average(data["val_user"], final_predictions)
    final_metrics = metric_eval(data["val_user"], data["val_y"], scores, data["fast"])
    metrics_output = {
        "gauc": final_metrics["gauc"],
        "ndcg5": final_metrics["ndcg5"],
        "primary": final_metrics["primary"],
        "selected_variant": winning_variant,
        "ablation_means": {name: float(np.mean(values)) for name, values in variant_scores.items()},
        "history": history,
    }
    with open(out_dir / "predictions.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, (user, video, score) in enumerate(zip(data["val_user"], data["val_video"], scores)):
            writer.writerow([i, user, video, float(score)])
    with open(out_dir / "metrics.json", "w") as handle:
        json.dump(metrics_output, handle, sort_keys=True)


if __name__ == "__main__":
    with open(os.devnull, "w") as sink, redirect_stdout(sink), redirect_stderr(sink):
        main()
