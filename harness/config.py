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
EPS = 0.002                       # official convergence epsilon, applied to the champion's seed-mean (ADR-0012)
CONFIRM_SEEDS = 2                 # extra seeds run for every candidate with a positive single-seed delta
MIN_EFFECT = 0.0005               # acceptance: the seed-mean improvement must be at least this (ADR-0011: was 0.001) ...
T_CRIT = 2.5                      # ... and at least T_CRIT standard errors (guards against the winner's curse)
STD_FLOOR = 0.0002                # floor on a per-node seed std estimate (3 seeds is a small sample)
N_CONVERGE = 3                    # official N
MAX_ITERS = 50                    # official cap
WALL_CLOCK_S = 6 * 3600           # official backstop
SMOKE_TIMEOUT_S = 120
RUN_TIMEOUT_S = 1800
DEFAULT_SEED = 0                  # same seed for every node in a run, so single-seed deltas are not seed noise
MAX_DIFF_LINES = 200              # an 'improve' node changing more lines than this is bounced back once (edit, don't rewrite)
SEED_SD = 0.0003                  # measured seed-to-seed SD of the validation primary (baseline seeds 0/1/2)

# Static firewall (ADR-0005): an agent script containing any of these strings is rejected before it runs.
FORBIDDEN_PATTERNS = ['KuaiRand-Pure', 'log_standard_4_22', 'log_standard_4_08', 'log_random',
                      'private/', 'private\\', 'test_features', 'video_features_statistic', '../']

def rules_text():
    """The acceptance and convergence rules as one sentence each, generated from the constants above so the text
    the LLM roles read can never drift from what the code does (ADR-0012)."""
    return (f"ACCEPTANCE (code, ADR-0010/0011): a candidate whose single-seed delta is positive is re-run with {CONFIRM_SEEDS} more "
            f"seeds (seeds {DEFAULT_SEED}..{DEFAULT_SEED + CONFIRM_SEEDS}); it is accepted iff its seed-mean gain over the champion "
            f"is >= {MIN_EFFECT} AND >= {T_CRIT} standard errors of the difference (seed-to-seed SD ~ {SEED_SD}); a node whose "
            f"predictions are byte-identical to its parent's is a no-op and is rejected without seeds. "
            f"CONVERGENCE (code, ADR-0012): the champion's seed-mean validation primary must rise by more than eps = {EPS} "
            f"(cumulative since the last such rise) within {N_CONVERGE} consecutive generations, else the run stops; "
            f"the cap is {MAX_ITERS} iterations and {WALL_CLOCK_S // 3600} h.")
