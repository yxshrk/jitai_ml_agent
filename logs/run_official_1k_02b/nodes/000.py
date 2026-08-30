"""FM reference baseline sized for KuaiRand-1K (fewer epochs; no official 1K
baseline exists — this is our disclosed internal reference for calibration)."""
import subprocess, sys, importlib.util

def main():
    spec = importlib.util.find_spec("zoo.baseline_ws")
    cmd = [sys.executable, spec.origin, "--epochs", "5", *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd))

if __name__ == "__main__":
    main()
