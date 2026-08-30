"""Deterministic pointwise PyTorch Factorization Machine with 32-dimensional embeddings."""

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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.evaluate_provisional import evaluate


def read_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty split: {path}")
    columns = rows[0].keys()
    return {name: np.asarray([int(row[name]) for row in rows], dtype=np.int64) for name in columns}


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
    k: int = 32,
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

    model = FactorizationMachine(feature_count, k)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for indices in torch.randperm(len(x), generator=generator).split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x[indices]), y[indices])
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(torch.as_tensor(val_x, dtype=torch.long))).numpy()
    metrics = evaluate(val["user_id"], val["long_view"], scores)

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
    parser.add_argument("--k", type=int, default=32)
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
