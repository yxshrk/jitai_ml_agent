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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_npz(data_dir):
    train = np.load(os.path.join(data_dir, "train.npz"), allow_pickle=False)
    val = np.load(os.path.join(data_dir, "val.npz"), allow_pickle=False)
    x_train = np.asarray(train["X"], dtype=np.int64)
    y_train = np.asarray(train["y"], dtype=np.float32)
    train_user = np.asarray(train["user"])
    x_val = np.asarray(val["X"], dtype=np.int64)
    y_val = np.asarray(val["y"], dtype=np.float32)
    val_user = np.asarray(val["user"])
    field_dims = np.asarray(train["field_dims"], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1]))).astype(np.int64)
    val_video = x_val[:, 1] - offsets[1]
    return x_train, y_train, train_user, x_val, y_val, val_user, val_video, field_dims


def read_csv_rows(path, training):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = {
                "user_id": row["user_id"],
                "video_id": row["video_id"],
                "tab": row["tab"],
                "duration_ms": float(row["duration_ms"] or 0.0),
                "long_view": float(row["long_view"]),
            }
            if training and "author_id" in row:
                item["author_id"] = row["author_id"]
            else:
                item["author_id"] = "__unknown_author__"
            rows.append(item)
    return rows


def make_mapping(values):
    unique = sorted(set(values))
    return {value: i + 1 for i, value in enumerate(unique)}


def load_csv(data_dir):
    train_rows = read_csv_rows(os.path.join(data_dir, "train.csv"), True)
    val_rows = read_csv_rows(os.path.join(data_dir, "val.csv"), False)
    user_map = make_mapping([r["user_id"] for r in train_rows])
    video_map = make_mapping([r["video_id"] for r in train_rows])
    author_map = make_mapping([r["author_id"] for r in train_rows])
    tab_map = make_mapping([r["tab"] for r in train_rows])
    durations = np.asarray([r["duration_ms"] for r in train_rows], dtype=np.float64)
    quantiles = np.quantile(durations, np.linspace(0.1, 0.9, 9)) if len(durations) else np.zeros(9)
    quantiles = np.maximum.accumulate(quantiles)
    field_dims = np.asarray([
        len(user_map) + 1,
        len(video_map) + 1,
        len(author_map) + 1,
        len(tab_map) + 1,
        10,
    ], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1]))).astype(np.int64)

    def encode(rows):
        x = np.empty((len(rows), 5), dtype=np.int64)
        y = np.empty(len(rows), dtype=np.float32)
        users = []
        videos = []
        for i, row in enumerate(rows):
            raw = np.asarray([
                user_map.get(row["user_id"], 0),
                video_map.get(row["video_id"], 0),
                author_map.get(row["author_id"], 0),
                tab_map.get(row["tab"], 0),
                int(np.searchsorted(quantiles, row["duration_ms"], side="right")),
            ], dtype=np.int64)
            x[i] = raw + offsets
            y[i] = row["long_view"]
            users.append(row["user_id"])
            videos.append(row["video_id"])
        return x, y, np.asarray(users), np.asarray(videos)

    x_train, y_train, train_user, _ = encode(train_rows)
    x_val, y_val, val_user, val_video = encode(val_rows)
    return x_train, y_train, train_user, x_val, y_val, val_user, val_video, field_dims


def load_data(data_dir):
    if os.path.isfile(os.path.join(data_dir, "train.npz")) and os.path.isfile(os.path.join(data_dir, "val.npz")):
        return load_npz(data_dir), True
    return load_csv(data_dir), False


class DCNLite(nn.Module):
    def __init__(self, field_dims, embed_dim=16, hidden_dim=128, dropout=0.20):
        super().__init__()
        total = int(np.sum(field_dims))
        input_dim = len(field_dims) * embed_dim
        self.embedding = nn.Embedding(total, embed_dim)
        self.linear = nn.Embedding(total, 1)
        self.cross_weight = nn.Parameter(torch.empty(input_dim))
        self.cross_bias = nn.Parameter(torch.zeros(input_dim))
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.cross_out = nn.Linear(input_dim, 1)
        self.embedding_dropout = nn.Dropout(dropout)
        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.cross_weight, std=0.01)
        nn.init.xavier_uniform_(self.cross_out.weight)
        nn.init.zeros_(self.cross_out.bias)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        emb = self.embedding_dropout(self.embedding(x)).flatten(1)
        cross = emb * torch.sum(emb * self.cross_weight, dim=1, keepdim=True) + self.cross_bias + emb
        first_order = self.linear(x).sum(dim=1).squeeze(1)
        return first_order + self.cross_out(cross).squeeze(1) + self.mlp(emb).squeeze(1)


def build_pairs(users, labels, seed):
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    rng = np.random.default_rng(seed)
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
    return np.concatenate(positives), np.concatenate(negatives)


def predict(model, x, device, batch_size=16384):
    model.eval()
    result = np.empty(len(x), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            xb = torch.from_numpy(x[start:end]).to(device)
            result[start:end] = torch.sigmoid(model(xb)).cpu().numpy()
    return result


def official_metrics(user_ids, labels, scores, npz_mode):
    if npz_mode:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    values = evaluate(user_ids, labels, scores)
    return {
        "gauc": float(values["GAUC"]),
        "ndcg5": float(values["nDCG@5"]),
        "primary": float(values["primary"]),
    }


def clone_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def update_swa(swa_state, state, count):
    if swa_state is None:
        return {k: v.clone() for k, v in state.items()}
    alpha = 1.0 / float(count)
    for key in swa_state:
        if torch.is_floating_point(swa_state[key]):
            swa_state[key].mul_(1.0 - alpha).add_(state[key], alpha=alpha)
        else:
            swa_state[key].copy_(state[key])
    return swa_state


def main():
    args = parse_args()
    seed_everything(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    data, npz_mode = load_data(args.data_dir)
    x_train, y_train, train_user, x_val, y_val, val_user, val_video, field_dims = data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DCNLite(field_dims).to(device)
    embedding_params = list(model.embedding.parameters()) + list(model.linear.parameters())
    embedding_ids = {id(p) for p in embedding_params}
    dense_params = [p for p in model.parameters() if id(p) not in embedding_ids]
    optimizer = torch.optim.AdamW([
        {"params": embedding_params, "weight_decay": 1e-6},
        {"params": dense_params, "weight_decay": 1e-4},
    ], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    bce = nn.BCEWithLogitsLoss()
    pair_pos, pair_neg = build_pairs(train_user, y_train, args.seed)
    smoke = os.environ.get("SMOKE_EPOCHS")
    epochs = 6
    if smoke is not None:
        epochs = max(1, min(epochs, int(smoke)))
    batch_size = 4096
    rng = np.random.default_rng(args.seed)
    swa_state = None
    swa_count = 0
    best_state = None
    best_gauc = -np.inf

    for epoch in range(epochs):
        model.train()
        point_order = rng.permutation(len(x_train))
        pair_order = rng.permutation(len(pair_pos)) if len(pair_pos) else np.empty(0, dtype=np.int64)
        steps = (len(point_order) + batch_size - 1) // batch_size
        pair_batch = max(1, (len(pair_order) + steps - 1) // max(steps, 1))
        for step in range(steps):
            left = step * batch_size
            point_idx = point_order[left:min(left + batch_size, len(point_order))]
            xb = torch.from_numpy(x_train[point_idx]).to(device)
            yb = torch.from_numpy(y_train[point_idx]).to(device)
            point_loss = bce(model(xb), yb)
            if len(pair_order):
                pl = step * pair_batch
                pr = min(pl + pair_batch, len(pair_order))
                if pl < pr:
                    selected = pair_order[pl:pr]
                    pos_x = torch.from_numpy(x_train[pair_pos[selected]]).to(device)
                    neg_x = torch.from_numpy(x_train[pair_neg[selected]]).to(device)
                    pair_loss = torch.nn.functional.softplus(-(model(pos_x) - model(neg_x))).mean()
                    loss = 0.5 * point_loss + 0.5 * pair_loss
                else:
                    loss = point_loss
            else:
                loss = point_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        scheduler.step()

        state = clone_state(model)
        if epoch >= 1 and epoch <= 4:
            swa_count += 1
            swa_state = update_swa(swa_state, state, swa_count)
            if swa_count >= 2 or epochs == 1:
                raw_state = clone_state(model)
                model.load_state_dict(swa_state)
                scores = predict(model, x_val, device)
                metrics = official_metrics(val_user, y_val, scores, npz_mode)
                if metrics["gauc"] > best_gauc:
                    best_gauc = metrics["gauc"]
                    best_state = clone_state(model)
                model.load_state_dict(raw_state)

    if best_state is None:
        if swa_state is not None:
            best_state = swa_state
        else:
            best_state = clone_state(model)
    model.load_state_dict(best_state)
    final_scores = predict(model, x_val, device)
    final_metrics = official_metrics(val_user, y_val, final_scores, npz_mode)

    predictions_path = os.path.join(args.out_dir, "predictions.csv")
    with open(predictions_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, (user_id, video_id, score) in enumerate(zip(val_user, val_video, final_scores)):
            writer.writerow([i, user_id.item() if isinstance(user_id, np.generic) else user_id,
                             video_id.item() if isinstance(video_id, np.generic) else video_id,
                             float(score)])
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
