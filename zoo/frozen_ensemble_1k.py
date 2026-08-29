"""Team reference: 5-seed rank-average ensemble of the frozen stack (disclosed).

Direct, readable invocation so any reader (human or agent) sees the exact
configuration: members = zoo/frozen_stack_1k.py (the frozen 1K-tuned champion config),
seeds {seed..seed+4}, per-user rank average via zoo/ensemble_node.py."""
import subprocess, sys, importlib.util

def main():
    spec = importlib.util.find_spec("zoo.ensemble_node")
    cmd = [sys.executable, spec.origin, "--member-script", "zoo/frozen_stack_1k.py",
           "--member-args", "", "--n-members", "5", *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd))

if __name__ == "__main__":
    main()
