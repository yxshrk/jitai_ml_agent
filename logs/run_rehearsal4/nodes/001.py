"""The frozen best-known stack as a plain contract-CLI script (CONTRACTS.md 3).

Thin wrapper over zoo/ablate_fields.py at field-level 0 with strong
regularization — the configuration confirmed at valid primary 0.6047 +/- 0.0003
(EXPERIMENTS_ABLATION.md) and re-confirmed 0.6048 +/- 0.0005 (EXPERIMENTS_DIMS.md).
Used as the disclosed seed/reference node for agent runs."""
import subprocess, sys, os

def main():
    args = sys.argv[1:]
    here = os.path.dirname(os.path.abspath(__file__))
    cmd = [sys.executable, os.path.join(here, "ablate_fields.py"),
           "--field-level", "0", "--regularized", *args]
    raise SystemExit(subprocess.call(cmd))

if __name__ == "__main__":
    main()
