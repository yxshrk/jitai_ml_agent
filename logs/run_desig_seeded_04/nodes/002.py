"""Two-member seed ensemble of the unchanged frozen champion configuration."""
import argparse
import importlib.util
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    spec = importlib.util.find_spec("zoo.ensemble_node")
    if spec is None or spec.origin is None:
        raise SystemExit(1)

    cmd = [
        sys.executable,
        spec.origin,
        "--data-dir",
        args.data_dir,
        "--out-dir",
        args.out_dir,
        "--seed",
        str(args.seed),
        "--member-script",
        "zoo/frozen_stack.py",
        "--member-args",
        "",
        "--n-members",
        "2",
    ]
    with open("/dev/null", "w", encoding="utf-8") as sink:
        result = subprocess.run(cmd, stdout=sink, stderr=sink, check=False)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
