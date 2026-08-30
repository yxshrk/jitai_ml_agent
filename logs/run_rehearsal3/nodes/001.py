import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_csv_data(data_dir):
    def read_file(path, is_train):
        users, videos, authors, tabs, durations, labels = [], [], [], [], [], []
        with open(path, "r", newline="") as fh:
            reader = csv.DictReader(fh)
            has_author = "author_id" in (reader.fieldnames or [])
            for row in reader:
                users.append(row["user_id"])
                videos.append(row["video_id"])
                authors.append(row["author_id"] if has_author else "0")
                tabs.append(row["tab"])
                durations.append(float(row["duration_ms"]))
                labels.append(float(row["long_view"]))
        return {
            "user_raw": np.asarray(users),
            "video_raw": np.asarray(videos),
            "author_raw": np.asarray(authors),
            "tab_raw": np.asarray(tabs),
            "duration": np.asarray(durations, dtype=np.float64),
            "y": np.asarray(labels, dtype=np.float32),
        }

    train = read_file(os.path.join(data_dir, "train.csv"), True)
    val = read_file(os.path.join(data_dir, "val.csv"), False)

    def make_mapping(values):
        return {v: i + 1 for i, v in enumerate(sorted(set(values.tolist())))}

    mappings = [
        make_mapping(train["user_raw"]),
        make_mapping(train["video_raw"]),
        make_mapping(train["author_raw"]),
        make_mapping(train["tab_raw"]),
    ]
    quantiles = np.quantile(train["duration"], np.linspace(0.1, 0.9, 9))
    field_dims = np.asarray([len(m) + 1 for m in mappings] + [10], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1])).astype(np.int64)

    def encode(split):
        raw_fields = [split["user_raw"], split["video_raw"], split["author_raw"], split["tab_raw"]]
        cols = []
        for values, mapping in zip(raw_fields, mappings):
            cols.append(np.fromiter((mapping.get(v, 0) for v in values), dtype=np.int64, count=len(values)))
        cols.append(np.searchsorted(quantiles, split["duration"], side="right").astype(np.int64))
        x = np.column_stack(cols).astype(np.int64)
        x += offsets[None, :]
        return x

    return {
        "X": encode(train),
        "y": train["y"],
        "user": train["user_raw"],
        "field_dims": field_dims,
    }, {
        "X": encode(val),
        "y": val["y"],
        "user": val["user_raw"],
        "video": val["video_raw"],
        "field_dims": field_dims,
    }


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        tr_file = np.load(train_npz)
        va_file = np.load(val_npz)
        tr = {k: tr_file[k] for k in tr_file.files}
        va = {k: va_file[k] for k in va_file.files}
        field_dims = tr["field_dims"].astype(np.int64)
        if "video" not in va:
            va["video"] = va["X"][:, 1].astype(np.int64) - int(field_dims[0])
        return tr, va, True
    tr, va = load_csv_data(data_dir)
    return tr, va, False


class RegularizedDCNLite(torch.nn.Module):
    def __init__(self, total_dim, fields=5, embedding_dim=16, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.embedding = torch.nn.Embedding(total_dim, embedding_dim)
        self.linear = torch.nn.Embedding(total_dim, 1)
        input_dim = fields * embedding_dim
        self.cross_weight = torch.nn.Parameter(torch.empty(input_dim))
        self.cross_bias = torch.nn.Parameter(torch.zeros(input_dim))
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.output = torch.nn.Linear(input_dim + hidden_dim, 1)
        self.global_bias = torch.nn.Parameter(torch.zeros(1))
        torch.nn.init.normal_(self.embedding.weight, std=0.01)
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.normal_(self.cross_weight, std=0.01)
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        torch.nn.init.zeros_(self.mlp[0].bias)
        torch.nn.init.xavier_uniform_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)

    def forward(self, x, return_embeddings=False):
        embeddings = self.embedding(x)
        x0 = embeddings.flatten(1)
        cross = x0 * torch.sum(x0 * self.cross_weight, dim=1, keepdim=True) + self.cross_bias + x0
        hidden = self.mlp(x0)
        score = self.global_bias + self.linear(x).sum((1, 2)) + self.output(torch.cat((cross, hidden), dim=1)).squeeze(1)
        if return_embeddings:
            return score, embeddings
        return score


def build_pair_index(users, labels):
    users = np.asarray(users)
    labels = np.asarray(labels) >= 0.5
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    positive_parts = []
    negative_lists = []
    pair_offsets = []
    pair_counts = []
    cursor = 0

    for left, right in zip(boundaries[:-1], boundaries[1:]):
        group_indices = order[left:right]
        group_labels = labels[group_indices]
        positives = group_indices[group_labels]
        negatives = group_indices[~group_labels]
        if len(positives) and len(negatives):
            positive_parts.append(positives.astype(np.int64, copy=False))
            negative_lists.append(negatives.astype(np.int64, copy=False))
            pair_offsets.append(np.full(len(positives), cursor, dtype=np.int64))
            pair_counts.append(np.full(len(positives), len(negatives), dtype=np.int64))
            cursor += len(negatives)

    if not positive_parts:
        return None

    pair_positive = np.concatenate(positive_parts)
    pair_offsets = np.concatenate(pair_offsets)
    pair_counts = np.concatenate(pair_counts)
    negative_pool = np.concatenate(negative_lists)
    return (
        torch.from_numpy(pair_positive),
        torch.from_numpy(pair_offsets),
        torch.from_numpy(pair_counts),
        torch.from_numpy(negative_pool),
    )


def predict(model, x, batch_size=65536):
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            parts.append(model(x[start:start + batch_size]).cpu().numpy())
    return np.concatenate(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)

    tr, va, fast_path = load_data(args.data_dir)
    if fast_path:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate

    train_x_np = tr["X"].astype(np.int64, copy=False)
    train_y_np = tr["y"].astype(np.float32, copy=False)
    val_x_np = va["X"].astype(np.int64, copy=False)
    val_y_np = va["y"].astype(np.int64, copy=False)
    train_x = torch.from_numpy(train_x_np)
    train_y = torch.from_numpy(train_y_np)
    val_x = torch.from_numpy(val_x_np)

    total_dim = int(np.asarray(tr["field_dims"]).sum())
    model = RegularizedDCNLite(total_dim=total_dim, fields=5, embedding_dim=16, hidden_dim=128, dropout=0.3)

    embedding_parameters = [model.embedding.weight, model.linear.weight]
    embedding_ids = {id(p) for p in embedding_parameters}
    dense_parameters = [p for p in model.parameters() if id(p) not in embedding_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": embedding_parameters, "weight_decay": 0.0},
            {"params": dense_parameters, "weight_decay": 1e-3},
        ],
        lr=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    bce = torch.nn.BCEWithLogitsLoss()
    pair_index = build_pair_index(train_x_np[:, 0], train_y_np)

    n = len(train_y)
    batch_size = 8192
    pair_batch_size = 2048
    best_gauc = -1.0
    best_scores = None
    patience = 0

    for _ in range(args.epochs):
        model.train()
        permutation = torch.randperm(n)
        for start in range(0, n, batch_size):
            point_indices = permutation[start:start + batch_size]
            point_x = train_x[point_indices]
            point_y = train_y[point_indices]

            optimizer.zero_grad(set_to_none=True)
            if pair_index is not None:
                pair_positive, pair_offsets, pair_counts, negative_pool = pair_index
                pair_slots = torch.randint(0, len(pair_positive), (min(pair_batch_size, len(point_indices)),))
                positive_indices = pair_positive[pair_slots]
                offsets = pair_offsets[pair_slots]
                counts = pair_counts[pair_slots]
                random_offsets = torch.floor(torch.rand(len(pair_slots)) * counts.to(torch.float32)).to(torch.int64)
                negative_indices = negative_pool[offsets + random_offsets]
                positive_x = train_x[positive_indices]
                negative_x = train_x[negative_indices]
                combined_x = torch.cat((point_x, positive_x, negative_x), dim=0)
                combined_scores, accessed_embeddings = model(combined_x, return_embeddings=True)
                point_count = len(point_x)
                pair_count = len(positive_x)
                point_scores = combined_scores[:point_count]
                positive_scores = combined_scores[point_count:point_count + pair_count]
                negative_scores = combined_scores[point_count + pair_count:]
                point_loss = bce(point_scores, point_y)
                pair_loss = torch.nn.functional.softplus(-(positive_scores - negative_scores)).mean()
                ranking_loss = 0.5 * point_loss + 0.5 * pair_loss
            else:
                point_scores, accessed_embeddings = model(point_x, return_embeddings=True)
                ranking_loss = bce(point_scores, point_y)

            row_l2 = accessed_embeddings.pow(2).sum(dim=2).mean()
            loss = ranking_loss + 1e-3 * row_l2
            loss.backward()
            optimizer.step()

        scores = predict(model, val_x)
        metrics = evaluate(va["user"], val_y_np, scores)
        gauc = float(metrics.get("GAUC", metrics.get("gauc")))
        if gauc > best_gauc + 1e-6:
            best_gauc = gauc
            best_scores = scores.copy()
            patience = 0
        else:
            patience += 1
        scheduler.step()
        if patience >= 4:
            break

    final_metrics = evaluate(va["user"], val_y_np, best_scores)
    gauc = float(final_metrics.get("GAUC", final_metrics.get("gauc")))
    ndcg5 = float(final_metrics.get("nDCG@5", final_metrics.get("ndcg5")))
    primary = float(final_metrics["primary"])

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump({"gauc": gauc, "ndcg5": ndcg5, "primary": primary}, fh)

    val_users = va["user"]
    val_videos = va["video"]
    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(best_scores):
            writer.writerow([i, val_users[i], val_videos[i], format(float(score), ".7g")])


if __name__ == "__main__":
    main()
