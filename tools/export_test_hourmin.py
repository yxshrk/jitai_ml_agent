"""Emit hourmin for the 1K test rows, aligned to data/test_features_1k/test.npz.

Streams the same two standard-log files in the same order with the same
test-date filter as data/export_test_features.py, so row order is identical.
Label columns are never read. Output: data/test_features_1k/test_hourmin.npz.
"""
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT.parent / "KuaiRand-1K" / "data"
TEST_MIN_DATE, TEST_MAX_DATE = 20220429, 20220508


def main():
    out = ROOT / "data/test_features_1k/test_hourmin.npz"
    ref = np.load(ROOT / "data/test_features_1k/test.npz")
    hourmins, users, dates = [], [], []
    for name in ("log_standard_4_08_to_4_21_1k.csv", "log_standard_4_22_to_5_08_1k.csv"):
        with (RAW / name).open(newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            header = next(reader)
            iu, id_, ih = header.index("user_id"), header.index("date"), header.index("hourmin")
            for row in reader:
                if not row:
                    continue
                date = int(row[id_])
                if TEST_MIN_DATE <= date <= TEST_MAX_DATE:
                    hourmins.append(int(row[ih]))
                    users.append(int(row[iu]))
                    dates.append(date)
    hourmin = np.asarray(hourmins, dtype=np.int64)
    if len(hourmin) != len(ref["user_id"]):
        sys.exit(f"row count mismatch: {len(hourmin)} vs {len(ref['user_id'])}")
    if not (np.asarray(users) == ref["user_id"]).all() or not (np.asarray(dates) == ref["date"]).all():
        sys.exit("alignment check FAILED: user/date sequences differ from test.npz")
    np.savez_compressed(out, hourmin=hourmin)
    print(f"wrote {out}: {len(hourmin):,} rows, alignment verified against test.npz")


if __name__ == "__main__":
    main()
