"""Run the known-best stack with exponential training recency weighting.

The delegated implementation in zoo.hist_best applies normalized exponential
row weights with a seven-day half-life while retaining the validation procedure
and contract outputs of the frozen stack.
"""
import argparse
import importlib.util
import os
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    spec = importlib.util.find_spec("zoo.hist_best")
    if spec is None or not spec.origin:
        raise SystemExit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    cmd = [
        sys.executable,
        spec.origin,
        "--data-dir",
        args.data_dir,
        "--out-dir",
        args.out_dir,
        "--seed",
        str(args.seed),
    ]
    with open(os.devnull, "w", encoding="utf-8") as sink:
        result = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=sink,
            stderr=sink,
            check=False,
            env=os.environ.copy(),
        )
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
