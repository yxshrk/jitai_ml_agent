import argparse
import csv
import json
import os
import random
import warnings

import numpy as np
import torch
from torch import nn


class FactorizationMachine(nn.Module):
    def __init__(self, total_features, embedding_dim=16):
        super().__init__()
        self.linear = nn.Embedding(total_features, 1)
        self.embedding = nn.Embedding(total_features, embedding_dim)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

    def forward(self, x):
        linear_term = self.linear(x).sum(dim=1).squeeze(-1)
        embeddings = self.embedding(x)
        summed = embeddings.sum(dim=1)
        interaction = 0.5 * (
            summed.square().sum(dim=1)
            - embeddings.square().sum(dim=(1, 2))
        )
        return self.bias + linear_term + interaction


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def make_mapping(values):
    unique = sorted(set(values))
    return {value: index + 1 for index, value in enumerate(unique)}


def load_csv_data(data_dir):
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")

    train_rows = []
    with open(train_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            train_rows.append(row)

    val_rows = []
    with open(val_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            val_rows.append(row)

    train_users = [row["user_id"] for row in train_rows]
    train_videos = [row["video_id"] for row in train_rows]
    train_tabs = [row["tab"] for row in train_rows]

    user_map = make_mapping(train_users)
    video_map = make_mapping(train_videos)
    tab_map = make_mapping(train_tabs)

    train_durations = np.asarray(
        [float(row.get("duration_ms", 0.0) or 0.0) for row in train_rows],
        dtype=np.float64,
    )
    quantiles = np.linspace(0.1, 0.9, 9)
    duration_edges = np.quantile(train_durations, quantiles).astype(np.float64)

    field_dims = np.asarray(
        [len(user_map) + 1, len(video_map) + 1, 1, len(tab_map) + 1, 10],
        dtype=np.int64,
    )
    offsets = np.concatenate(
        [np.zeros(1, dtype=np.int64), np.cumsum(field_dims[:-1], dtype=np.int64)]
    )

    def encode(rows):
        x = np.empty((len(rows), 5), dtype=np.int64)
        for i, row in enumerate(rows):
            duration = float(row.get("duration_ms", 0.0) or 0.0)
            x[i, 0] = user_map.get(row["user_id"], 0) + offsets[0]
            x[i, 1] = video_map.get(row["video_id"], 0) + offsets[1]
            x[i, 2] = offsets[2]
            x[i, 3] = tab_map.get(row["tab"], 0) + offsets[3]
            x[i, 4] = int(np.searchsorted(duration_edges, duration, side="right")) + offsets[4]
        return x

    train_x = encode(train_rows)
    val_x = encode(val_rows)
    train_y = np.asarray([float(row["long_view"]) for row in train_rows], dtype=np.float32)
    val_y = np.asarray([float(row["long_view"]) for row in val_rows], dtype=np.float32)
    val_users = np.asarray([row["user_id"] for row in val_rows])
    val_videos = np.asarray([row["video_id"] for row in val_rows])
    return train_x, train_y, val_x, val_y, val_users, val_videos, field_dims, False


def load_npz_data(data_dir):
    train_file = np.load(os.path.join(data_dir, "train.npz"), allow_pickle=False)
    val_file = np.load(os.path.join(data_dir, "val.npz"), allow_pickle=False)

    train_x = np.asarray(train_file["X"], dtype=np.int64)
    train_y = np.asarray(train_file["y"], dtype=np.float32)
    val_x = np.asarray(val_file["X"], dtype=np.int64)
    val_y = np.asarray(val_file["y"], dtype=np.float32)
    field_dims = np.asarray(train_file["field_dims"], dtype=np.int64).reshape(-1)
    val_users = np.asarray(val_file["user"])

    if "video" in val_file.files:
        val_videos = np.asarray(val_file["video"])
    else:
        video_offset = int(field_dims[0])
        val_videos = val_x[:, 1].astype(np.int64) - video_offset

    train_file.close()
    val_file.close()
    return train_x, train_y, val_x, val_y, val_users, val_videos, field_dims, True


def predict(model, x, device, batch_size):
    model.eval()
    output = np.empty(len(x), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            batch_x = torch.from_numpy(x[start:end]).to(device=device, dtype=torch.long)
            output[start:end] = torch.sigmoid(model(batch_x)).cpu().numpy().astype(np.float32)
    return output


def copy_parameters(target, source_parameters):
    with torch.no_grad():
        for target_parameter, source_parameter in zip(target.parameters(), source_parameters):
            target_parameter.copy_(source_parameter)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    seed_everything(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    fast_path = os.path.exists(os.path.join(args.data_dir, "train.npz")) and os.path.exists(
        os.path.join(args.data_dir, "val.npz")
    )
    if fast_path:
        train_x, train_y, val_x, val_y, val_users, val_videos, field_dims, use_official = load_npz_data(args.data_dir)
    else:
        train_x, train_y, val_x, val_y, val_users, val_videos, field_dims, use_official = load_csv_data(args.data_dir)

    if use_official:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_features = int(field_dims.sum())
    model = FactorizationMachine(total_features, embedding_dim=16).to(device)
    averaged_model = FactorizationMachine(total_features, embedding_dim=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.BCEWithLogitsLoss()

    epochs = 6
    smoke_epochs = os.environ.get("SMOKE_EPOCHS")
    if smoke_epochs is not None:
        epochs = min(epochs, max(1, int(smoke_epochs)))

    batch_size = 16384
    rng = np.random.default_rng(args.seed)
    averaged_parameters = None
    best_primary = -float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(train_x))
        for start in range(0, len(train_x), batch_size):
            indices = order[start:start + batch_size]
            batch_x = torch.from_numpy(train_x[indices]).to(device=device, dtype=torch.long)
            batch_y = torch.from_numpy(train_y[indices]).to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        current_parameters = [parameter.detach().clone() for parameter in model.parameters()]
        if averaged_parameters is None:
            averaged_parameters = current_parameters
        else:
            with torch.no_grad():
                for averaged, current in zip(averaged_parameters, current_parameters):
                    averaged.mul_(0.5).add_(current, alpha=0.5)

        copy_parameters(averaged_model, averaged_parameters)
        epoch_scores = predict(averaged_model, val_x, device, batch_size)
        epoch_metrics = evaluate(val_users, val_y, epoch_scores)
        epoch_primary = float(epoch_metrics["primary"])
        if epoch_primary > best_primary:
            best_primary = epoch_primary
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in averaged_model.state_dict().items()
            }

    averaged_model.load_state_dict(best_state)
    averaged_model.to(device)
    scores = predict(averaged_model, val_x, device, batch_size)
    metrics = evaluate(val_users, val_y, scores)

    predictions_path = os.path.join(args.out_dir, "predictions.csv")
    with open(predictions_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for row_id, (user_id, video_id, score) in enumerate(zip(val_users, val_videos, scores)):
            writer.writerow([row_id, user_id, video_id, format(float(score), ".10g")])

    metrics_output = {
        "gauc": float(metrics["GAUC"]),
        "ndcg5": float(metrics["nDCG@5"]),
        "primary": float(metrics["primary"]),
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics_output, handle, separators=(",", ":"))


if __name__ == "__main__":
    main()
