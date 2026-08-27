"""Materialize the real KuaiRand-Pure train/val splits as workspace CSVs
(same schema as data/synthetic + author_id). The harness copies these into the
agent workspace; the test split is deliberately never exported here."""
import csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RAW = os.path.join(os.path.dirname(__file__), "..", "..", "KuaiRand-Pure", "data")
OUT = os.path.join(os.path.dirname(__file__), "real_ws")
SPLITS = {"train": (20220408, 20220421), "val": (20220422, 20220428)}  # NO test
COLS = ["user_id", "video_id", "author_id", "tab", "hourmin", "date",
        "duration_ms", "long_view", "click", "like", "play_time_ms"]

def main():
    os.makedirs(OUT, exist_ok=True)
    vid2author = {}
    with open(os.path.join(RAW, "video_features_basic_pure.csv")) as fh:
        for r in csv.DictReader(fh):
            vid2author[r["video_id"]] = r["author_id"]
    writers, files = {}, {}
    for name in SPLITS:
        files[name] = open(os.path.join(OUT, f"{name}.csv"), "w", newline="")
        writers[name] = csv.writer(files[name]); writers[name].writerow(COLS)
    counts = {k: 0 for k in SPLITS}
    for f in ("log_standard_4_08_to_4_21_pure.csv", "log_standard_4_22_to_5_08_pure.csv"):
        with open(os.path.join(RAW, f)) as fh:
            for r in csv.DictReader(fh):
                d = int(r["date"])
                for name, (lo, hi) in SPLITS.items():
                    if lo <= d <= hi:
                        writers[name].writerow([r["user_id"], r["video_id"],
                            vid2author.get(r["video_id"], "UNK"), r["tab"], r["hourmin"], r["date"],
                            r["duration_ms"], r["long_view"], r["is_click"], r["is_like"], r["play_time_ms"]])
                        counts[name] += 1
    for fh in files.values(): fh.close()
    print(counts)
    assert counts["train"] == 1141112 and counts["val"] == 124909, counts

if __name__ == "__main__" and "--npz" not in sys.argv:
    main()

def write_npz():
    """Pre-encode train/val to int32 arrays (official-style vocab from train only)."""
    import numpy as np
    rows = {}
    for name in SPLITS:
        with open(os.path.join(OUT, f"{name}.csv")) as fh:
            rows[name] = list(csv.DictReader(fh))
    tr = rows["train"]
    durs = np.array([float(r["duration_ms"]) for r in tr])
    edges = np.quantile(durs, np.linspace(0, 1, 11)[1:-1])
    FIELDS = ["user_id", "video_id", "author_id", "tab", "dur_bucket"]
    def raw(r):
        return [r["user_id"], r["video_id"], r["author_id"], r["tab"],
                str(int(np.searchsorted(edges, float(r["duration_ms"]))))]
    vocabs = [dict() for _ in FIELDS]
    for r in tr:
        for i, v in enumerate(raw(r)):
            vocabs[i].setdefault(v, len(vocabs[i]))
    unk = [len(v) for v in vocabs]
    field_dims = np.array([len(v) + 1 for v in vocabs], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(field_dims[:-1])]).astype(np.int32)
    for name in SPLITS:
        rs = rows[name]
        X = np.empty((len(rs), len(FIELDS)), dtype=np.int32)
        for n, r in enumerate(rs):
            for i, v in enumerate(raw(r)):
                X[n, i] = vocabs[i].get(v, unk[i]) + offsets[i]
        np.savez_compressed(os.path.join(OUT, f"{name}.npz"),
            X=X,
            y=np.array([int(r["long_view"]) for r in rs], dtype=np.float32),
            user=np.array([int(r["user_id"]) for r in rs], dtype=np.int64),
            click=np.array([int(r["click"]) for r in rs], dtype=np.float32),
            play_time_ms=np.array([float(r["play_time_ms"]) for r in rs], dtype=np.float32),
            duration_ms=np.array([float(r["duration_ms"]) for r in rs], dtype=np.float32),
            hourmin=np.array([int(r["hourmin"]) for r in rs], dtype=np.int32),
            date=np.array([int(r["date"]) for r in rs], dtype=np.int32),
            field_dims=field_dims)
        print(name, X.shape)

if __name__ == "__main__" and "--npz" in sys.argv:
    write_npz()
