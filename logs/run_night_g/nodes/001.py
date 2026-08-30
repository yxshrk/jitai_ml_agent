"""The frozen best-known stack as a plain contract-CLI script (CONTRACTS.md 3).

Thin wrapper over zoo/ablate_fields.py at field-level 0 with strong
regularization — the configuration confirmed at valid primary 0.6047 +/- 0.0003
(EXPERIMENTS_ABLATION.md) and re-confirmed 0.6048 +/- 0.0005 (EXPERIMENTS_DIMS.md).
Used as the disclosed seed/reference node for agent runs."""
import subprocess, sys, os

def main():
    args = sys.argv[1:]
    # Location-independent: the harness copies node code elsewhere, so resolve
    # the real repo zoo/ via PYTHONPATH (set by the harness) or this file's repo.
    import importlib.util
    spec = importlib.util.find_spec("zoo.ablate_fields")
    if spec is None or not spec.origin:
        raise SystemExit("cannot locate zoo.ablate_fields on PYTHONPATH")
    cmd = [sys.executable, spec.origin, "--field-level", "0", "--regularized", *args]
    raise SystemExit(subprocess.call(cmd))

if __name__ == "__main__":
    main()
