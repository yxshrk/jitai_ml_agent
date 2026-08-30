"""Atomic capacity test: run the parent polish stack with embedding dimension k=8."""
import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_path = out_dir / "predictions.csv"
    metrics_path = out_dir / "metrics.json"

    spec = importlib.util.find_spec("zoo.polish_stack")
    if spec is None or not spec.origin:
        raise SystemExit(2)

    if pred_path.exists() and metrics_path.exists():
        return

    smoke_epochs = os.environ.get("SMOKE_EPOCHS")
    epochs = 1
    if smoke_epochs is not None:
        epochs = min(epochs, max(1, int(smoke_epochs)))

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    cmd = [
        sys.executable,
        spec.origin,
        "--lr", "0.00168",
        "--dropout", "0.21",
        "--weight-decay", "0.000037",
        "--k", "8",
        "--recency-half-life", "7.0",
        "--epochs", str(epochs),
        "--data-dir", args.data_dir,
        "--out-dir", args.out_dir,
        "--seed", str(args.seed),
    ]
    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        env=env,
    )
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    if not pred_path.exists() or not metrics_path.exists():
        raise SystemExit(3)


if __name__ == "__main__":
    main()
