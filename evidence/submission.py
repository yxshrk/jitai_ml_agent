"""Build and validate the final submission CSV (row_id,user_id,video_id,score).

Mirrors the official starter-kit submit.py --check validations
(../starter-kit/kuairand-starter-kit/submit.py), but takes the evaluation
split as a plain CSV (columns include user_id,video_id in row order) instead
of needing the real KuaiRand data dir.

Checks performed by `check`:
  - header is exactly row_id,user_id,video_id,score
  - every record has exactly 4 fields
  - row_id is 0-based, strictly increasing, no gaps
  - row count matches the split exactly (no extra, no missing rows)
  - user_id/video_id on each row align with the split file's row order
  - score parses as a finite float (no NaN/Inf)

Usage:
  uv run python evidence/submission.py --make  predictions.csv split.csv out.csv
  uv run python evidence/submission.py --check submission.csv  split.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

HEADER = ["row_id", "user_id", "video_id", "score"]


class SubmissionError(ValueError):
    pass


def load_split(split_path: str | Path) -> list[tuple[str, str]]:
    """Read the split file; return [(user_id, video_id), ...] in row order."""
    with open(split_path, newline="") as fh:
        r = csv.reader(fh)
        head = next(r, None)
        if head is None:
            raise SubmissionError(f"split file {split_path} is empty")
        try:
            iu, iv = head.index("user_id"), head.index("video_id")
        except ValueError:
            raise SubmissionError(
                f"split file {split_path} must have user_id and video_id columns, got {head}"
            )
        return [(rec[iu], rec[iv]) for rec in r if rec]


def build(predictions_path: str | Path, split_path: str | Path, out_path: str | Path) -> int:
    """Build the submission CSV from a predictions file (contract section 3:
    row_id,user_id,video_id,score; a header-less single 'score' column is
    also accepted). Validates the result before returning."""
    rows = load_split(split_path)
    scores: list[float] = []
    with open(predictions_path, newline="") as fh:
        r = csv.reader(fh)
        first = next(r, None)
        if first is None:
            raise SubmissionError(f"predictions file {predictions_path} is empty")
        if first == HEADER:
            score_idx = 3
        elif len(first) == 1:
            score_idx = 0
            scores.append(float(first[0]))
        else:
            raise SubmissionError(
                f"unrecognized predictions format: header {first}"
            )
        for rec in r:
            if rec:
                scores.append(float(rec[score_idx]))
    if len(scores) != len(rows):
        raise SubmissionError(
            f"predictions have {len(scores)} rows but split has {len(rows)}"
        )
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        for i, ((uid, vid), s) in enumerate(zip(rows, scores)):
            w.writerow([i, uid, vid, f"{float(s):.6g}"])
    check(out_path, split_path)
    return len(rows)


def check(submission_path: str | Path, split_path: str | Path) -> list[float]:
    """Validate a submission CSV against the split. Raises SubmissionError on
    any malformation; returns the scores on success."""
    rows = load_split(split_path)
    with open(submission_path, newline="") as fh:
        r = csv.reader(fh)
        head = next(r, None)
        if head != HEADER:
            raise SubmissionError(f"header must be {','.join(HEADER)}, got {head}")
        scores: list[float] = []
        n = 0
        for ln, rec in enumerate(r, start=2):
            if len(rec) != 4:
                raise SubmissionError(f"line {ln}: {len(rec)} fields, expected 4")
            rid, uid, vid, sc = rec
            try:
                rid_i = int(rid)
            except ValueError:
                raise SubmissionError(f"line {ln}: row_id {rid!r} is not an integer")
            if rid_i != n:
                raise SubmissionError(
                    f"line {ln}: row_id={rid}, expected {n} "
                    f"(row_id must be 0-based, strictly increasing, no gaps)"
                )
            if n >= len(rows):
                raise SubmissionError(
                    f"submission has more rows than the split ({len(rows)} rows)"
                )
            if uid != rows[n][0] or vid != rows[n][1]:
                raise SubmissionError(
                    f"line {ln}: alignment error - submission ({uid},{vid}) vs "
                    f"split row {n} ({rows[n][0]},{rows[n][1]})"
                )
            try:
                v = float(sc)
            except ValueError:
                raise SubmissionError(f"line {ln}: score {sc!r} is not a number")
            if math.isnan(v) or math.isinf(v):
                raise SubmissionError(f"line {ln}: score is NaN/Inf, not allowed")
            scores.append(v)
            n += 1
    if n != len(rows):
        raise SubmissionError(
            f"submission has {n} rows, split has {len(rows)} - count mismatch"
        )
    return scores


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--make", nargs=3, metavar=("PREDICTIONS", "SPLIT", "OUT"))
    g.add_argument("--check", nargs=2, metavar=("SUBMISSION", "SPLIT"))
    args = ap.parse_args()
    try:
        if args.make:
            n = build(*args.make)
            print(f"wrote {args.make[2]}: {n:,d} rows (validated)")
        else:
            scores = check(*args.check)
            print(f"OK: format and alignment valid, {len(scores):,d} rows")
    except SubmissionError as e:
        print(f"INVALID: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
