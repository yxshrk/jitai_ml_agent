import argparse
import csv
import json
import os
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def load_npz(data_dir):
    train = np.load(os.path.join(data_dir, "train.npz"), allow_pickle=True)
    val = np.load(os.path.join(data_dir, "val.npz"), allow_pickle=True)
    field_dims = np.asarray(train["field_dims"], dtype=np.int64)
    train_data = {
        "X": np.asarray(train["X"], dtype=np.int64),
        "y": np.asarray(train["y"], dtype=np.float32),
        "user": np.asarray(train["user"]),
    }
    val_data = {
        "X": np.asarray(val["X"], dtype=np.int64),
        "y": np.asarray(val["y"], dtype=np.float32),
        "user": np.asarray(val["user"]),
    }
    if val_data["X"].shape[1] > 1:
        video_codes = val_data["X"][:, 1] - int(field_dims[0])
    else:
        video_codes = np.zeros(len(val_data["y"]), dtype=np.int64)
    val_data["video_out"] = video_codes
    val_data["user_out"] = val_data["user"]
    return train_data, val_data, field_dims, True


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
            }
            if training:
                item["long_view"] = float(row["long_view"])
            else:
                item["long_view"] = float(row["long_view"])
            rows.append(item)
    return rows


def make_mapping(values):
    unique = sorted(set(values))
    return {value: index + 1 for index, value in enumerate(unique)}


def load_csv(data_dir):
    train_rows = read_csv_rows(os.path.join(data_dir, "train.csv"), True)
    val_rows = read_csv_rows(os.path.join(data_dir, "val.csv"), False)

    user_map = make_mapping([r["user_id"] for r in train_rows])
    video_map = make_mapping([r["video_id"] for r in train_rows])
    tab_map = make_mapping([r["tab"] for r in train_rows])

    durations = np.asarray([r["duration_ms"] for r in train_rows], dtype=np.float64)
    quantiles = np.quantile(durations, np.linspace(0.1, 0.9, 9)) if len(durations) else np.zeros(9)
    quantiles = np.maximum.accumulate(quantiles)

    field_dims = np.asarray([
        len(user_map) + 1,
        len(video_map) + 1,
        1,
        len(tab_map) + 1,
        10,
    ], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1])).astype(np.int64)

    def encode(rows):
        x = np.empty((len(rows), 5), dtype=np.int64)
        y = np.empty(len(rows), dtype=np.float32)
        users = np.empty(len(rows), dtype=object)
        videos = np.empty(len(rows), dtype=object)
        for i, row in enumerate(rows):
            duration_bucket = int(np.searchsorted(quantiles, row["duration_ms"], side="right"))
            x[i, 0] = user_map.get(row["user_id"], 0)
            x[i, 1] = video_map.get(row["video_id"], 0)
            x[i, 2] = 0
            x[i, 3] = tab_map.get(row["tab"], 0)
            x[i, 4] = duration_bucket
            x[i] += offsets
            y[i] = row["long_view"]
            users[i] = row["user_id"]
            videos[i] = row["video_id"]
        return x, y, users, videos

    train_x, train_y, train_user, _ = encode(train_rows)
    val_x, val_y, val_user, val_video = encode(val_rows)
    train_data = {"X": train_x, "y": train_y, "user": train_user}
    val_data = {
        "X": val_x,
        "y": val_y,
        "user": val_user,
        "user_out": val_user,
        "video_out": val_video,
    }
    return train_data, val_data, field_dims, False


def build_pairs(users, labels, seed):
    users = np.asarray(users)
    labels = np.asarray(labels)
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    rng = np.random.RandomState(seed)
    positive_parts = []
    negative_parts = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        group = order[left:right]
        positives = group[labels[group] > 0.5]
        negatives = group[labels[group] <= 0.5]
        if len(positives) == 0 or len(negatives) == 0:
            continue
        chosen_negatives = negatives[rng.randint(0, len(negatives), size=len(positives))]
        positive_parts.append(positives.astype(np.int64, copy=False))
        negative_parts.append(chosen_negatives.astype(np.int64, copy=False))
    if not positive_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(positive_parts), np.concatenate(negative_parts)


class DCNLite(nn.Module):
    def __init__(self, field_dims, embed_dim=16, dropout=0.30):
        super().__init__()
        self.num_fields = len(field_dims)
        self.embed_dim = embed_dim
        total_features = int(np.sum(field_dims))
        input_dim = self.num_fields * embed_dim
        self.embedding = nn.Embedding(total_features, embed_dim)
        self.linear_embedding = nn.Embedding(total_features, 1)
        self.cross_weight = nn.ModuleList([nn.Linear(input_dim, 1, bias=False) for _ in range(2)])
        self.cross_bias = nn.ParameterList([nn.Parameter(torch.zeros(input_dim)) for _ in range(2)])
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.output = nn.Linear(input_dim + 128, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.zeros_(self.linear_embedding.weight)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        embeddings = self.embedding(x)
        x0 = embeddings.reshape(x.shape[0], -1)
        crossed = x0
        for weight, bias in zip(self.cross_weight, self.cross_bias):
            crossed = x0 * weight(crossed) + bias + crossed
        deep = self.mlp(x0)
        first_order = self.linear_embedding(x).sum(dim=1).squeeze(-1)
        summed = embeddings.sum(dim=1)
        fm = 0.5 * (summed.square() - embeddings.square().sum(dim=1)).sum(dim=1)
        return self.output(torch.cat([crossed, deep], dim=1)).squeeze(1) + first_order + fm + self.global_bias


def predict(model, x, device, batch_size=32768):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.as_tensor(x[start:start + batch_size], dtype=torch.long, device=device)
            outputs.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(outputs).astype(np.float64, copy=False)


def official_metrics(user_ids, labels, scores, npz_mode):
    if npz_mode:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    result = evaluate(user_ids, labels, scores)
    return {
        "gauc": float(result.get("GAUC", result.get("gauc"))),
        "ndcg5": float(result.get("nDCG@5", result.get("ndcg5"))),
        "primary": float(result.get("primary")),
    }


def train_model(train_data, val_data, field_dims, seed, device, npz_mode):
    x_train = train_data["X"]
    y_train = train_data["y"]
    x_val = val_data["X"]
    positive_indices, negative_indices = build_pairs(train_data["user"], y_train, seed)

    model = DCNLite(field_dims, embed_dim=16, dropout=0.30).to(device)
    embedding_parameters = list(model.embedding.parameters()) + list(model.linear_embedding.parameters())
    embedding_ids = {id(p) for p in embedding_parameters}
    dense_parameters = [p for p in model.parameters() if id(p) not in embedding_ids]
    optimizer = torch.optim.AdamW([
        {"params": embedding_parameters, "weight_decay": 0.0},
        {"params": dense_parameters, "weight_decay": 1e-3},
    ], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)

    epochs = 16
    smoke_epochs = os.environ.get("SMOKE_EPOCHS")
    if smoke_epochs is not None:
        epochs = min(epochs, max(1, int(smoke_epochs)))

    batch_size = 8192
    rng = np.random.RandomState(seed)
    best_gauc = -np.inf
    best_state = None
    stale_epochs = 0

    for _ in range(epochs):
        model.train()
        permutation = rng.permutation(len(x_train))
        for start in range(0, len(x_train), batch_size):
            indices = permutation[start:start + batch_size]
            xb = torch.as_tensor(x_train[indices], dtype=torch.long, device=device)
            yb = torch.as_tensor(y_train[indices], dtype=torch.float32, device=device)
            logits = model(xb)
            point_loss = F.binary_cross_entropy_with_logits(logits, yb)

            if len(positive_indices):
                pair_choice = rng.randint(0, len(positive_indices), size=len(indices))
                pos_x = torch.as_tensor(x_train[positive_indices[pair_choice]], dtype=torch.long, device=device)
                neg_x = torch.as_tensor(x_train[negative_indices[pair_choice]], dtype=torch.long, device=device)
                pos_logits = model(pos_x)
                neg_logits = model(neg_x)
                pair_loss = F.softplus(-(pos_logits - neg_logits)).mean()
                accessed = torch.cat([xb.reshape(-1), pos_x.reshape(-1), neg_x.reshape(-1)])
            else:
                pair_loss = point_loss
                accessed = xb.reshape(-1)

            unique_rows = torch.unique(accessed)
            row_l2 = model.embedding(unique_rows).square().sum() / float(max(1, len(indices)))
            loss = 0.5 * point_loss + 0.5 * pair_loss + 1e-4 * row_l2
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        scores = predict(model, x_val, device)
        metrics = official_metrics(val_data["user"], val_data["y"], scores, npz_mode)
        if metrics["gauc"] > best_gauc + 1e-7:
            best_gauc = metrics["gauc"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        scheduler.step()
        if stale_epochs >= 4:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def write_outputs(out_dir, val_data, scores, metrics):
    os.makedirs(out_dir, exist_ok=True)
    prediction_path = os.path.join(out_dir, "predictions.csv")
    with open(prediction_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, (user_id, video_id, score) in enumerate(zip(val_data["user_out"], val_data["video_out"], scores)):
            if isinstance(user_id, np.generic):
                user_id = user_id.item()
            if isinstance(video_id, np.generic):
                video_id = video_id.item()
            writer.writerow([i, user_id, video_id, "{:.10f}".format(float(score))])
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, separators=(",", ":"))


def main():
    args = parse_args()
    seed_everything(args.seed)
    npz_available = os.path.exists(os.path.join(args.data_dir, "train.npz")) and os.path.exists(os.path.join(args.data_dir, "val.npz"))
    if npz_available:
        train_data, val_data, field_dims, npz_mode = load_npz(args.data_dir)
    else:
        train_data, val_data, field_dims, npz_mode = load_csv(args.data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_model(train_data, val_data, field_dims, args.seed, device, npz_mode)
    scores = predict(model, val_data["X"], device)
    metrics = official_metrics(val_data["user"], val_data["y"], scores, npz_mode)
    write_outputs(args.out_dir, val_data, scores, metrics)


if __name__ == "__main__":
    main()
