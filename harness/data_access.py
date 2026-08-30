"""Builds the agent workspace (train + valid, no test) and the harness-private test features.

This is the structural firewall of ADR-0005: agent scripts run with cwd = workspace/ and are handed
--data-dir workspace/data; the raw download (which contains the test labels) is never passed to them,
and the referee's static check rejects any script that names it.

  workspace/data/train.csv                 every log column (outcome columns are legal as targets / history)
  workspace/data/valid.csv                 row_id + show-time features + long_view (needed to self-score)
  workspace/data/video_features_basic.csv  side table (no labels)
  workspace/data/user_features.csv         side table (no labels)
  workspace/evaluate.py                    the official scorer, unmodified
  private/test_features.csv                row_id + show-time features, NO labels; used once by submit.py
"""
import csv, json, shutil
from . import config as C

LOG_COLS = ['user_id', 'video_id', 'date', 'hourmin', 'time_ms', 'is_click', 'is_like', 'is_follow', 'is_comment',
            'is_forward', 'is_hate', 'long_view', 'play_time_ms', 'duration_ms', 'profile_stay_time',
            'comment_stay_time', 'is_profile_enter', 'is_rand', 'tab']
FEATURE_COLS = ['user_id', 'video_id', 'date', 'hourmin', 'time_ms', 'tab', 'duration_ms', 'is_rand']
EXPECTED = {'train': 1_141_112, 'valid': 124_909, 'test': 170_588}

def build(force=False):
    C.WS_DATA.mkdir(parents=True, exist_ok=True); C.PRIVATE.mkdir(exist_ok=True)
    manifest = C.WS_DATA / 'manifest.json'
    if manifest.exists() and not force:
        return json.loads(manifest.read_text())
    train_p, valid_p, test_p = C.WS_DATA / 'train.csv', C.WS_DATA / 'valid.csv', C.PRIVATE / 'test_features.csv'
    counts = {'train': 0, 'valid': 0, 'test': 0}
    with open(train_p, 'w', newline='') as ftr, open(valid_p, 'w', newline='') as fva, open(test_p, 'w', newline='') as fte:
        wtr, wva, wte = csv.writer(ftr), csv.writer(fva), csv.writer(fte)
        wtr.writerow(LOG_COLS)
        wva.writerow(['row_id'] + FEATURE_COLS + ['long_view'])
        wte.writerow(['row_id'] + FEATURE_COLS)
        # Same file order and date filter as data.load() in the starter kit, so row_id lines up with submit.py.
        for f in ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv'):
            with open(C.RAW / f, newline='') as fh:
                for row in csv.DictReader(fh):
                    d = int(row['date'])
                    if C.SPLITS['train'][0] <= d <= C.SPLITS['train'][1]:
                        wtr.writerow([row[c] for c in LOG_COLS]); counts['train'] += 1
                    elif C.SPLITS['valid'][0] <= d <= C.SPLITS['valid'][1]:
                        wva.writerow([counts['valid']] + [row[c] for c in FEATURE_COLS] + [row['long_view']]); counts['valid'] += 1
                    elif C.SPLITS['test'][0] <= d <= C.SPLITS['test'][1]:
                        wte.writerow([counts['test']] + [row[c] for c in FEATURE_COLS]); counts['test'] += 1
    assert counts == EXPECTED, f'split sizes {counts} != expected {EXPECTED}'
    shutil.copy(C.RAW / 'video_features_basic_pure.csv', C.WS_DATA / 'video_features_basic.csv')
    shutil.copy(C.RAW / 'user_features_pure.csv', C.WS_DATA / 'user_features.csv')
    shutil.copy(C.KIT / 'evaluate.py', C.WORKSPACE / 'evaluate.py')
    manifest.write_text(json.dumps(counts, indent=1))
    return counts

if __name__ == '__main__':
    import sys
    print(build(force='--force' in sys.argv))
