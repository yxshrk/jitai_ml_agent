"""Frozen best-known stack with embedding dimension reduced from 16 to 8.

This is a clean capacity test: all fields, losses, regularization, training
schedule, and validation-based model selection remain unchanged.
"""
import importlib.util
import subprocess
import sys


def main():
    args = sys.argv[1:]
    spec = importlib.util.find_spec("zoo.ablate_fields")
    if spec is None or not spec.origin:
        raise SystemExit("cannot locate zoo.ablate_fields on PYTHONPATH")
    cmd = [
        sys.executable,
        spec.origin,
        "--field-level",
        "0",
        "--regularized",
        *args,
    ]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
