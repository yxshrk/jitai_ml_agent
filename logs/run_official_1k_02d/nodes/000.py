"""Best-known 1K configuration (farm phase-3 best, 0.6214 seed-42 valid primary).

Wrapper over zoo/polish_stack.py; used as the disclosed seed/reference node for
1K agent runs and the bonus submission recipe."""
import subprocess, sys
import importlib.util

def main():
    spec = importlib.util.find_spec("zoo.polish_stack")
    if spec is None or not spec.origin:
        raise SystemExit("cannot locate zoo.polish_stack on PYTHONPATH")
    cmd = [sys.executable, spec.origin, "--lr", "0.00168", "--dropout", "0.21",
           "--weight-decay", "0.000037", "--k", "24", "--recency-half-life", "7.0",
           "--epochs", "6", *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd))

if __name__ == "__main__":
    main()
