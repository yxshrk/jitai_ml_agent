"""Deterministic hybrid pointwise and within-user BPR Factorization Machine."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.evaluate_provisional import evaluate


FEATURE_COLUMNS = ("user_id", "video_id", "tab", "hourmin", "duration_ms")


def read_csv(path: Path) -> dict[str, np.ndarray]:
    columns = FEATURE_COLUMNS + ("long_view",)
    values: dict[str, list[int]] = {name: [] for name in columns}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for name in columns:
                values[name].append(int(row[name]))
    if not values["user_id"]:
        raise ValueError(f"empty split: {path}")
    return {name: np.asarray(column, dtype=np.int64) for name, column in values.items()}


def raw_features(data: dict[str, np.ndarray]) -> np.ndarray:
    hour = data["hourmin"] // 100
    # Five coarse duration bands in seconds; clipping reserves stable known IDs.
    duration_bucket = np.clip(data["duration_ms"] // 20_000, 0, 5)
    return np.column_stack((data["user_id"], data["video_id"], data["tab"], hour, duration_bucket))


def encode(train_raw: np.ndarray, val_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    train_encoded = np.empty_like(train_raw)
    val_encoded = np.empty_like(val_raw)
    offset = 0
    for column in range(train_raw.shape[1]):
        values = np.unique(train_raw[:, column])
        mapping = {int(value): offset + i for i, value in enumerate(values)}
        unknown = offset + len(values)
        train_encoded[:, column] = [mapping[int(value)] for value in train_raw[:, column]]
        val_encoded[:, column] = [mapping.get(int(value), unknown) for value in val_raw[:, column]]
        offset = unknown + 1
    return train_encoded, val_encoded, offset


def build_within_user_pairs(
    user_ids: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = np.argsort(user_ids, kind="stable")
    sorted_users = user_ids[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    positive_parts: list[np.ndarray] = []
    negative_parts: list[np.ndarray] = []

    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        indices = order[start:stop]
        positives = indices[labels[indices] == 1]
        negatives = indices[labels[indices] == 0]
        if len(positives) == 0 or len(negatives) == 0:
            continue
        pair_count = len(indices)
        positive_parts.append(rng.choice(positives, size=pair_count, replace=True))
        negative_parts.append(rng.choice(negatives, size=pair_count, replace=True))

    if not positive_parts:
        raise ValueError("training split has no users containing both label classes")
    return np.concatenate(positive_parts), np.concatenate(negative_parts)


class FactorizationMachine(nn.Module):
    def __init__(self, n_features: int, k: int) -> None:
        super().__init__()
        self.linear = nn.Embedding(n_features, 1)
        self.embedding = nn.Embedding(n_features, k)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, std=0.05)

    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        linear = self.linear(fields).sum(dim=1).squeeze(1)
        embedded = self.embedding(fields)
        summed = embedded.sum(dim=1)
        interaction = 0.5 * (summed.square() - embedded.square().sum(dim=1)).sum(dim=1)
        return self.bias + linear + interaction


def run(
    data_dir: Path,
    out_dir: Path,
    seed: int = 42,
    k: int = 16,
    epochs: int = 60,
    batch_size: int = 256,
    learning_rate: float = 0.02,
) -> dict[str, float]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

    train = read_csv(data_dir / "train.csv")
    val = read_csv(data_dir / "val.csv")
    train_x, val_x, feature_count = encode(raw_features(train), raw_features(val))
    x = torch.as_tensor(train_x, dtype=torch.long)
    y = torch.as_tensor(train["long_view"], dtype=torch.float32)

    positive_rows, negative_rows = build_within_user_pairs(
        train["user_id"], train["long_view"], seed
    )
    positive_rows_t = torch.as_tensor(positive_rows, dtype=torch.long)
    negative_rows_t = torch.as_tensor(negative_rows, dtype=torch.long)

    model = FactorizationMachine(feature_count, k)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(seed)

    for _ in range(epochs):
        point_order = torch.randperm(len(x), generator=generator)
        pair_samples = torch.randint(
            len(positive_rows_t), (len(x),), generator=generator
        )
        for point_indices, pair_indices in zip(
            point_order.split(batch_size), pair_samples.split(batch_size), strict=True
        ):
            optimizer.zero_grad(set_to_none=True)
            pointwise_loss = loss_fn(model(x[point_indices]), y[point_indices])

            positive_x = x[positive_rows_t[pair_indices]]
            negative_x = x[negative_rows_t[pair_indices]]
            pair_fields = torch.cat((positive_x, negative_x), dim=0)
            pair_logits = model(pair_fields)
            positive_logits, negative_logits = pair_logits.chunk(2)
            pairwise_loss = -F.logsigmoid(positive_logits - negative_logits).mean()

            loss = 0.5 * pointwise_loss + 0.5 * pairwise_loss
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(torch.as_tensor(val_x, dtype=torch.long))).numpy()
    evaluated = evaluate(val["user_id"], val["long_view"], scores)
    metrics = {
        "gauc": float(evaluated["gauc"]),
        "ndcg5": float(evaluated["ndcg5"]),
        "primary": float(evaluated["primary"]),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for row_id, (user_id, video_id, score) in enumerate(
            zip(val["user_id"], val["video_id"], scores, strict=True)
        ):
            writer.writerow([row_id, int(user_id), int(video_id), f"{float(score):.10f}"])
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, sort_keys=True)
        handle.write("\n")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    args = parser.parse_args()
    run(
        args.data_dir,
        args.out_dir,
        args.seed,
        args.k,
        args.epochs,
        args.batch_size,
        args.learning_rate,
    )


if __name__ == "__main__":
    main()
