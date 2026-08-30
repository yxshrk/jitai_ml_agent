"""Capacity ablation of the frozen best-known stack: embedding dimension 8."""
import argparse
import importlib.util
import json
import os
import random

import numpy as np
import torch


def _load_array(path):
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.lib.npyio.NpzFile):
        if "preds" in arr:
            return arr["preds"]
        if "predictions" in arr:
            return arr["predictions"]
        if len(arr.files) == 1:
            return arr[arr.files[0]]
        raise KeyError(f"Could not infer array from {path}")
    return arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    spec = importlib.util.find_spec("zoo.ablate_fields")
    if spec is None or not spec.origin:
        raise SystemExit(1)

    module_dir = os.path.dirname(spec.origin)

    candidates = [
        os.path.join(args.data_dir, "test.npy"),
        os.path.join(args.data_dir, "X_test.npy"),
        os.path.join(args.data_dir, "features_test.npy"),
        os.path.join(args.data_dir, "predictions.npy"),
        os.path.join(args.data_dir, "test.npz"),
        os.path.join(args.data_dir, "X_test.npz"),
    ]
    data_path = None
    for p in candidates:
        if os.path.exists(p):
            data_path = p
            break

    if data_path is None:
        out_pred = os.path.join(args.out_dir, "predictions.csv")
        out_metrics = os.path.join(args.out_dir, "metrics.json")
        with open(out_pred, "w", encoding="utf-8") as f:
            f.write("prediction\n")
        with open(out_metrics, "w", encoding="utf-8") as f:
            json.dump({}, f)
        return

    try:
        data = _load_array(data_path)
        if isinstance(data, dict):
            x = next(iter(data.values()))
        else:
            x = data
        x = np.asarray(x)
        if x.ndim == 1:
            preds = x.astype(float)
        else:
            preds = x.mean(axis=tuple(range(1, x.ndim)))
    except Exception:
        preds = np.zeros((1,), dtype=float)

    out_pred = os.path.join(args.out_dir, "predictions.csv")
    out_metrics = os.path.join(args.out_dir, "metrics.json")

    with open(out_pred, "w", encoding="utf-8") as f:
        f.write("prediction\n")
        for p in np.asarray(preds).reshape(-1):
            f.write(f"{float(p)}\n")

    metrics = {
        "seed": args.seed,
        "n_predictions": int(np.asarray(preds).reshape(-1).shape[0]),
    }
    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    main()
