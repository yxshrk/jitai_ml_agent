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
CONFIRM_SEEDS = 3                 # FRESH seeds (1..3) run for every candidate whose seed-0 delta is positive; seed 0 is the
                                  # screen (a selected maximum) and is reported but not used in the decision (ADR-0012)
MAX_CONFIRM_SEEDS = 5             # adaptive: two more fresh seeds when the z-score is borderline (Z_BORDER <= z < Z_CRIT)
MIN_EFFECT = 0.0005               # acceptance: the fresh-seed mean gain over the champion must be at least this ...
Z_CRIT = 3.0                      # ... and at least Z_CRIT standard errors with the POOLED seed SD (a z-test, not a 3-vs-3 t-test)
Z_BORDER = 2.0                    # below this the candidate is rejected outright; between Z_BORDER and Z_CRIT more seeds decide
SEED_SD_PRIOR = 0.0003            # prior on the seed-to-seed SD of the primary (live_01/02), blended with the run's pooled estimate
SEED_SD_PRIOR_DF = 4              # weight of that prior in degrees of freedom
N_CONVERGE = 3                    # official N
RESET_MIN_GAIN = 0.001            # the convergence streak resets when the champion's fresh-seed mean has risen by at least this since
                                  # the last reset (cumulative, like min_delta against best-seen): eps/2 on a statistic with ~1/3 the
                                  # single-seed noise, so one false acceptance cannot buy 3 more generations but a staircase of real gains counts
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

# ADR-0014: the organizers allow any open-source library; numpy-only was our own contract. These are the libraries
# installed in the venv that agent scripts may import (the Critic checks the contract's determinism rules).
AVAILABLE_LIBS = ['numpy', 'pandas 2.3', 'scikit-learn 1.6', 'lightgbm 4.6', 'torch 2.8 (CPU only)']
HARD_GROUP_REJECTS = 2            # rejected deepens on one breakdown group before it is marked hard (ADR-0014)
FREE_SLOT_FROM_GENERATION = 2     # from this generation on, one Selector slot goes to an untried / not-yet-stacked card
# ADR-0016: family campaigns — from this generation on, the Selector's slots belong to ONE card family per generation
# (chosen in code; k different mechanisms of that family); a family closes after CAMPAIGN_FLAT_GENERATIONS generations
# without an accepted node from it, and the next family opens. Composition (ensembling) comes last.
CAMPAIGNS_FROM_GENERATION = 2
CAMPAIGN_FLAT_GENERATIONS = 2
CAMPAIGN_LAST_FAMILIES = ('ensembling',)
# ADR-0012 amendment (after live_07): who may be designated for submission. 'strict' = accepted nodes only (the champion
# lineage), re-ranked by fresh-seed mean among themselves; 'adaptive' = an unaccepted node that leads on fresh-seed mean
# may be designated iff, with MAX_CONFIRM_SEEDS fresh seeds, its gain over the champion is >= MIN_EFFECT at z >= Z_BORDER.
DESIGNATION_DEFAULT = 'strict'

def libs_text():
    """The library rule as one sentence, generated from AVAILABLE_LIBS so prompts cannot drift from the contract."""
    return ('LIBRARIES (ADR-0014): ' + ', '.join(AVAILABLE_LIBS) + ' and the standard library are available to scripts; '
            'CPU only (never MPS/CUDA); thread count from the OMP_NUM_THREADS environment variable; every library seeded '
            'from --seed (numpy Generator, torch.manual_seed, LightGBM seed + deterministic=True); SMOKE_EPOCHS caps epochs '
            'AND boosting rounds.')

def rules_text():
    """The acceptance and convergence rules as one sentence each, generated from the constants above so the text
    the LLM roles read can never drift from what the code does (ADR-0012)."""
    return (f"ACCEPTANCE (code, ADR-0010/0011/0012): a candidate whose seed-{DEFAULT_SEED} delta is positive is re-run with {CONFIRM_SEEDS} "
            f"FRESH seeds; it is accepted iff its fresh-seed mean gain over the champion's fresh-seed mean is >= {MIN_EFFECT} AND "
            f"z >= {Z_CRIT}, where z uses the seed SD pooled over every seed run of this run (prior {SEED_SD_PRIOR}); a borderline "
            f"z in [{Z_BORDER}, {Z_CRIT}) gets {MAX_CONFIRM_SEEDS - CONFIRM_SEEDS} more seeds before the decision; a node whose "
            f"predictions are byte-identical to its parent's is a no-op and is rejected without seeds. "
            f"CONVERGENCE (code, ADR-0012): the streak resets when the champion's fresh-seed mean has risen by >= {RESET_MIN_GAIN} since "
            f"the last reset (cumulative); {N_CONVERGE} consecutive generations without such a rise stop the run. Smaller confirmed "
            f"gains still move the champion; this is the organizers' eps = {EPS} rescaled to the seed-mean's noise, and the literal "
            f"single-seed eps rule is tracked and reported alongside; the cap is {MAX_ITERS} iterations and {WALL_CLOCK_S // 3600} h.")

# ADR-0015: the feature screen. A candidate whose target_component is in SCREEN_COMPONENTS (or a wildcard naming a
# new_signal) is probed first: a small script computes the proposed feature(s) on valid (label-stripped data dir) and
# the screen measures within-user discrimination against the champion's predictions. Below SCREEN_MIN_GAIN the slot is
# dropped without spending an Implementer, a Critic, a run and three seeds. Measured 2026-08-31: every item-statistic /
# session / exposure-context feature scored <= +0.0005 on the FM champion (kb/data/screens).
SCREEN_COMPONENTS = ('features', 'encoding', 'history')
SCREEN_MIN_GAIN = 0.0003          # best of (additive on the champion, lambdarank stack gain) on valid; MIN_EFFECT is 0.0005
SCREEN_TIMEOUT_S = 180
SCREEN_FROM_GENERATION = 1
