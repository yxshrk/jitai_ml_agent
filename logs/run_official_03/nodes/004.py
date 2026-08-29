"""Run the frozen regularized five-field stack with embedding dimension k=8."""
import argparse
import importlib.util
import os
import re
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    spec = importlib.util.find_spec("zoo.ablate_fields")
    if spec is None or not spec.origin:
        raise SystemExit(1)

    help_run = subprocess.run(
        [sys.executable, spec.origin, "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),
        check=False,
    )
    help_text = help_run.stdout or ""
    dim_flag = None
    for candidate in ("--embedding-dim", "--embed-dim", "--emb-dim", "--dim", "--k"):
        if re.search(r"(?<![A-Za-z0-9_-])" + re.escape(candidate) + r"(?![A-Za-z0-9_-])", help_text):
            dim_flag = candidate
            break
    if dim_flag is None:
        raise SystemExit(1)

    cmd = [
        sys.executable,
        spec.origin,
        "--field-level", "0",
        "--regularized",
        dim_flag, "8",
        "--data-dir", args.data_dir,
        "--out-dir", args.out_dir,
        "--seed", str(args.seed),
    ]
    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=os.environ.copy(),
        check=False,
    )
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
