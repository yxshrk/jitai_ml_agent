import argparse
import csv
import json
import os
import random
import warnings

import numpy as np
import torch
from torch import nn

warnings.filterwarnings("ignore")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_npz(data_dir):
    train_path = os.path.join(data_dir, "train.npz")
    val_path = os.path.join(data_dir, "val.npz")
    if not (os.path.exists(train_path) and os.path.exists(val_path)):
        return None
    with np.load(train_path, allow_pickle=False) as z:
        train = {
            "X": z["X"].astype(np.int64, copy=False),
            "y": z["y"].astype(np.float32, copy=False),
            "user": z["user"],
            "play_time_ms": z["play_time_ms"].astype(np.float32, copy=False),
            "duration_ms": z["duration_ms"].astype(np.float32, copy=False),
            "field_dims": z["field_dims"].astype(np.int64, copy=False),
        }
    with np.load(val_path, allow_pickle=False) as z:
        val = {
            "X": z["X"].astype(np.int64, copy=False),
            "y": z["y"].astype(np.float32, copy=False),
            "user": z["user"],
        }
    return train, val, True


def read_csv_columns(path, training):
    users = []
    videos = []
    tabs = []
    durations = []
    labels = []
    plays = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            users.append(row["user_id"])
            videos.append(row["video_id"])
            tabs.append(row["tab"])
            durations.append(float(row["duration_ms"]))
            labels.append(float(row["long_view"]))
            if training:
                plays.append(float(row["play_time_ms"]))
    result = {
        "user": np.asarray(users),
        "video": np.asarray(videos),
        "tab": np.asarray(tabs),
        "duration_ms": np.asarray(durations, dtype=np.float32),
        "y": np.asarray(labels, dtype=np.float32),
    }
    if training:
        result["play_time_ms"] = np.asarray(plays, dtype=np.float32)
    return result


def make_mapping(values):
    unique = sorted(set(values.tolist()))
    return {value: i + 1 for i, value in enumerate(unique)}


def encode_values(values, mapping):
    return np.fromiter((mapping.get(v, 0) for v in values), dtype=np.int64, count=len(values))


def load_csv(data_dir):
    train_raw = read_csv_columns(os.path.join(data_dir, "train.csv"), True)
    val_raw = read_csv_columns(os.path.join(data_dir, "val.csv"), False)
    user_map = make_mapping(train_raw["user"])
    video_map = make_mapping(train_raw["video"])
    tab_map = make_mapping(train_raw["tab"])
    quantiles = np.quantile(train_raw["duration_ms"], np.linspace(0.1, 0.9, 9))
    quantiles = np.maximum.accumulate(quantiles)

    def encode(raw):
        user_code = encode_values(raw["user"], user_map)
        video_code = encode_values(raw["video"], video_map)
        author_code = np.zeros(len(user_code), dtype=np.int64)
        tab_code = encode_values(raw["tab"], tab_map)
        dur_code = np.searchsorted(quantiles, raw["duration_ms"], side="right").astype(np.int64) + 1
        columns = [user_code, video_code, author_code, tab_code, dur_code]
        dims = np.asarray([
            len(user_map) + 1,
            len(video_map) + 1,
            1,
            len(tab_map) + 1,
            11,
        ], dtype=np.int64)
        offsets = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(dims)[:-1]])
        return np.stack([columns[i] + offsets[i] for i in range(5)], axis=1), dims

    train_x, dims = encode(train_raw)
    val_x, _ = encode(val_raw)
    train = {
        "X": train_x,
        "y": train_raw["y"],
        "user": train_raw["user"],
        "play_time_ms": train_raw["play_time_ms"],
        "duration_ms": train_raw["duration_ms"],
        "field_dims": dims,
    }
    val = {
        "X": val_x,
        "y": val_raw["y"],
        "user": val_raw["user"],
        "video": val_raw["video"],
    }
    return train, val, False


def make_ordinal_targets(play_time_ms, duration_ms):
    denominator = np.minimum(np.maximum(duration_ms, 1.0), 18000.0)
    ratio = np.maximum(play_time_ms, 0.0) / denominator
    thresholds = np.asarray([0.25, 0.50, 0.75, 1.00], dtype=np.float32)
    return (ratio[:, None] >= thresholds[None, :]).astype(np.float32)


def make_pairs(users, labels, seed):
    n = len(labels)
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    rng = np.random.default_rng(seed)
    pos_parts = []
    neg_parts = []
    for j in range(len(boundaries) - 1):
        idx = order[boundaries[j]:boundaries[j + 1]]
        pos = idx[labels[idx] > 0.5]
        neg = idx[labels[idx] <= 0.5]
        if len(pos) == 0 or len(neg) == 0:
            continue
        pos = pos[rng.permutation(len(pos))]
        neg = neg[rng.permutation(len(neg))]
        count = max(len(pos), len(neg))
        pos_parts.append(np.resize(pos, count))
        neg_parts.append(np.resize(neg, count))
    if not pos_parts:
        fallback = np.arange(min(n, 1), dtype=np.int64)
        return fallback, fallback
    return np.concatenate(pos_parts).astype(np.int64), np.concatenate(neg_parts).astype(np.int64)


class DCNOrdinal(nn.Module):
    def __init__(self, total_dim, embed_dim=16, hidden_dim=128, dropout=0.30):
        super().__init__()
        self.embedding = nn.Embedding(total_dim, embed_dim)
        input_dim = 5 * embed_dim
        self.cross_w = nn.ParameterList([nn.Parameter(torch.empty(input_dim)) for _ in range(2)])
        self.cross_b = nn.ParameterList([nn.Parameter(torch.zeros(input_dim)) for _ in range(2)])
        self.deep = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.main_head = nn.Linear(input_dim + hidden_dim, 1)
        self.ordinal_head = nn.Linear(input_dim + hidden_dim, 4)
        nn.init.xavier_uniform_(self.embedding.weight)
        for w in self.cross_w:
            nn.init.normal_(w, std=0.01)

    def forward(self, x):
        x0 = self.embedding(x).flatten(1)
        cross = x0
        for w, b in zip(self.cross_w, self.cross_b):
            cross = x0 * torch.sum(cross * w, dim=1, keepdim=True) + b + cross
        deep = self.deep(x0)
        representation = torch.cat([cross, deep], dim=1)
        return self.main_head(representation).squeeze(1), self.ordinal_head(representation)


def predict(model, x, device, batch_size):
    model.eval()
    output = np.empty(len(x), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            stop = min(start + batch_size, len(x))
            xb = torch.as_tensor(x[start:stop], dtype=torch.long, device=device)
            logits, _ = model(xb)
            output[start:stop] = torch.sigmoid(logits).cpu().numpy()
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seed_everything(args.seed)
    loaded = load_npz(args.data_dir)
    if loaded is None:
        train, val, fast_path = load_csv(args.data_dir)
    else:
        train, val, fast_path = loaded

    train_x = np.asarray(train["X"], dtype=np.int64)
    val_x = np.asarray(val["X"], dtype=np.int64)
    train_y = np.asarray(train["y"], dtype=np.float32)
    val_y = np.asarray(val["y"], dtype=np.float32)
    ordinal_y = make_ordinal_targets(train["play_time_ms"], train["duration_ms"])
    pair_pos, pair_neg = make_pairs(np.asarray(train["user"]), train_y, args.seed)

    total_dim = int(max(np.sum(train["field_dims"]), train_x.max() + 1, val_x.max() + 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DCNOrdinal(total_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    bce = nn.BCEWithLogitsLoss()

    epochs = 12
    smoke = os.environ.get("SMOKE_EPOCHS")
    if smoke is not None:
        epochs = min(epochs, max(1, int(smoke)))
    batch_size = 8192
    rng = np.random.default_rng(args.seed)
    best_gauc = -float("inf")
    best_state = None
    stale = 0

    if fast_path:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate

    for _ in range(epochs):
        model.train()
        obs_order = rng.permutation(len(train_y))
        pair_order = rng.permutation(len(pair_pos))
        obs_batches = (len(obs_order) + batch_size - 1) // batch_size
        pair_batches = (len(pair_order) + batch_size - 1) // batch_size
        steps = max(obs_batches, pair_batches)
        for step in range(steps):
            ostart = (step % obs_batches) * batch_size
            oidx = obs_order[ostart:min(ostart + batch_size, len(obs_order))]
            pstart = (step % pair_batches) * batch_size
            psel = pair_order[pstart:min(pstart + batch_size, len(pair_order))]

            xb = torch.as_tensor(train_x[oidx], dtype=torch.long, device=device)
            yb = torch.as_tensor(train_y[oidx], dtype=torch.float32, device=device)
            ob = torch.as_tensor(ordinal_y[oidx], dtype=torch.float32, device=device)
            pos_x = torch.as_tensor(train_x[pair_pos[psel]], dtype=torch.long, device=device)
            neg_x = torch.as_tensor(train_x[pair_neg[psel]], dtype=torch.long, device=device)

            optimizer.zero_grad(set_to_none=True)
            logits, ordinal_logits = model(xb)
            pos_logits, _ = model(pos_x)
            neg_logits, _ = model(neg_x)
            point_loss = bce(logits, yb)
            pair_loss = -torch.nn.functional.logsigmoid(pos_logits - neg_logits).mean()
            ordinal_loss = bce(ordinal_logits, ob)
            loss = 0.5 * point_loss + 0.5 * pair_loss + 0.3 * ordinal_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        scheduler.step()

        val_scores = predict(model, val_x, device, batch_size * 2)
        current = evaluate(np.asarray(val["user"]), val_y, val_scores)
        current_gauc = float(current["GAUC"])
        if current_gauc > best_gauc + 1.0e-7:
            best_gauc = current_gauc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 2:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    scores = predict(model, val_x, device, batch_size * 2)
    metrics_raw = evaluate(np.asarray(val["user"]), val_y, scores)
    metrics = {
        "gauc": float(metrics_raw["GAUC"]),
        "ndcg5": float(metrics_raw["nDCG@5"]),
        "primary": float(metrics_raw["primary"]),
    }

    if "video" in val:
        video_ids = val["video"]
    else:
        video_ids = val_x[:, 1]
    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, (user_id, video_id, score) in enumerate(zip(val["user"], video_ids, scores)):
            writer.writerow([i, user_id.item() if hasattr(user_id, "item") else user_id, video_id.item() if hasattr(video_id, "item") else video_id, "%.9g" % float(score)])
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
