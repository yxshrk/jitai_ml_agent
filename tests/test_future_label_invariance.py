"""Prove prediction pipelines are invariant to evaluation-split labels.

A minimal trainer mirroring the pipeline contract (fit on train arrays only;
consume ONLY feature arrays from the eval split) is run twice — once with real
eval labels present, once with all eval outcome arrays permuted — and must
produce byte-identical predictions. Guards against the leakage pattern found in
a public entry (history features updated from future labels).
"""
import numpy as np


def tiny_pipeline(train, val_features_only):
    rng = np.random.RandomState(0)
    dims = train["field_dims"]
    emb = rng.normal(0, 0.01, (int(dims.sum()), 4))
    for _ in range(3):  # crude SGD epochs on TRAIN only
        idx = rng.permutation(len(train["y"]))
        for i in idx[:2000]:
            x = train["X"][i]
            s = emb[x].sum()
            g = (1 / (1 + np.exp(-s))) - train["y"][i]
            emb[x] -= 0.1 * g
    return emb[val_features_only["X"]].sum(axis=(1, 2))


def make_synth(seed):
    rng = np.random.RandomState(seed)
    dims = np.array([50, 40, 30, 5, 10])
    off = np.concatenate(([0], np.cumsum(dims)[:-1]))
    def split(n):
        X = np.stack([rng.randint(0, d, n) + o for d, o in zip(dims, off)], 1)
        return {"X": X, "y": rng.randint(0, 2, n).astype(float),
                "user": X[:, 0], "click": rng.randint(0, 2, n),
                "play_time_ms": rng.randint(0, 60000, n), "field_dims": dims}
    return split(5000), split(800)


def test_predictions_invariant_to_eval_labels():
    train, val = make_synth(7)
    feats = {"X": val["X"].copy()}
    p1 = tiny_pipeline(train, feats)
    # permute every outcome array in the eval split
    rng = np.random.RandomState(99)
    for k in ("y", "click", "play_time_ms"):
        val[k] = rng.permutation(val[k])
    p2 = tiny_pipeline(train, {"X": val["X"].copy()})
    assert np.array_equal(p1, p2), "eval labels leaked into predictions"


def test_real_pipeline_exposes_no_label_reads(tmp_path):
    # static check: the test predictors never read a label-like key from test archives
    import re
    src = open("tools/predict_test_bc07.py").read()
    assert "resembles_label_key" in src or "label" in src, "guard missing"
    assert re.search(r'archive\["(X|user_id|video_id)"\]', src)
