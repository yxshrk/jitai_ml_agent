"""Five-seed per-user rank ensemble of the frozen champion configuration."""
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

    ensemble_spec = importlib.util.find_spec("zoo.ensemble_node")
    member_spec = importlib.util.find_spec("zoo.ablate_fields")
    if ensemble_spec is None or not ensemble_spec.origin:
        raise SystemExit(1)
    if member_spec is None or not member_spec.origin:
        raise SystemExit(1)

    epochs = 7
    smoke = os.environ.get("SMOKE_EPOCHS")
    if smoke is not None:
        epochs = min(epochs, max(1, int(smoke)))

    os.makedirs(args.out_dir, exist_ok=True)
    cmd = [
        sys.executable,
        ensemble_spec.origin,
        "--data-dir", args.data_dir,
        "--out-dir", args.out_dir,
        "--seed", str(args.seed),
        "--member-script", member_spec.origin,
        "--member-args", "--field-level 0 --regularized",
        "--n-members", "5",
        "--member-epochs", str(epochs),
    ]
    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        env=os.environ.copy(),
    )
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
