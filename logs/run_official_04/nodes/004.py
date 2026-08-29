"""Test reduced embedding capacity on the frozen best-known stack."""
import subprocess
import sys


def main():
    args = sys.argv[1:]
    # Location-independent: the harness copies node code elsewhere, so resolve
    # the real repo zoo/ via PYTHONPATH (set by the harness) or this file's repo.
    import importlib.util

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
