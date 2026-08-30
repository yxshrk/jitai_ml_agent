"""Run the frozen stack with its measured date-aware recency weighting variant."""
import argparse
import importlib.util
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

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
    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
