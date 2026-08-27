import csv
from pathlib import Path

from data.synthetic.generate import FIELDS, SPLITS, generate


def test_generator_is_deterministic_and_respects_windows(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    counts = generate(first, seed=17)
    assert counts == generate(second, seed=17)
    assert sum(counts.values()) == 5_000

    for split, (lower, upper) in SPLITS.items():
        assert (first / f"{split}.csv").read_bytes() == (second / f"{split}.csv").read_bytes()
        with (first / f"{split}.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert rows and list(rows[0]) == FIELDS
        assert all(lower <= int(row["date"]) <= upper for row in rows)
        assert all(int(row["long_view"]) in (0, 1) for row in rows)
        assert all(int(row["tab"]) in range(5) for row in rows)
