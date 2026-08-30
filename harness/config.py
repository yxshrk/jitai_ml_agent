"""Harness configuration. Absolute paths so the loop can be started from anywhere."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / 'kuairand-starter-kit'
RAW = KIT / 'KuaiRand-Pure' / 'data'          # raw download -- never exposed to agent scripts
WORKSPACE = ROOT / 'workspace'                 # everything an agent script may read (cwd of every run)
WS_DATA = WORKSPACE / 'data'
PRIVATE = ROOT / 'private'                     # harness-only: test features (no labels)
RUNS = ROOT / 'runs'
SEEDS = ROOT / 'harness' / 'seeds'
KB = ROOT / 'kb'

SPLITS = {'train': (20220408, 20220421), 'valid': (20220422, 20220428), 'test': (20220429, 20220508)}
BASELINE_VALID_PRIMARY = 0.6016   # official FM, 5-seed mean
EPS = 0.002                       # official convergence epsilon
N_CONVERGE = 3                    # official N
MAX_ITERS = 50                    # official cap
WALL_CLOCK_S = 6 * 3600           # official backstop
SMOKE_TIMEOUT_S = 120
RUN_TIMEOUT_S = 1800
DEFAULT_SEED = 0                  # same seed for every node in a run, so deltas are not seed noise

# Static firewall (ADR-0005): an agent script containing any of these strings is rejected before it runs.
FORBIDDEN_PATTERNS = ['KuaiRand-Pure', 'log_standard_4_22', 'log_standard_4_08', 'log_random',
                      'private/', 'private\\', 'test_features', 'video_features_statistic', '../']
