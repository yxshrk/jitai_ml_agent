"""Test lower embedding capacity on the frozen best-known stack.

This keeps field level, regularization, loss, head, and training schedule fixed,
changing only the embedding dimension from 16 to 8.
"""
import importlib.util
import subprocess
import sys


def main():
    args = sys.argv[1:]
    spec = importlib.util.find_spec("zoo.ablate_fields")
    if spec is None or not spec.origin:
        raise SystemExit(1)
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
