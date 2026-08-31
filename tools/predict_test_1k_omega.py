"""Build the 1K test submission from the omega_1k node_005 recipe (re-designation).

Faithful to the designated node: loads its own script as a module (RankModel,
session_codes, adversarial weighting, train_candidate), reconstructs session
features for train/val (asserted IDENTICAL to the node's own augmentation), then
extends them causally through validation into the test window (history context
only — no label use, no training on val/test). Trains SEEDS on train rows only
(organizer ruling), selects checkpoints on validation, global-rank-averages the
test scores, validates the ensemble on validation BEFORE the single test write.

Usage (on ruby): python tools/predict_test_1k_omega.py
"""
import csv
import os
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
NODE = ROOT / "logs/run_omega_1k/nodes/005.py"
WORKSPACE = ROOT / "logs/run_omega_1k/workspace"
METRICS = ROOT / "logs/run_omega_1k/node_005/metrics.json"
TEST_PATH = ROOT / "data/test_features_1k/test.npz"
TEST_HOURMIN = ROOT / "data/test_features_1k/test_hourmin.npz"
OUTPUT_PATH = ROOT / os.environ.get("JITAI_1K_OUT","evidence/test_submission_1k.csv")
CHECKER = ROOT / "evidence/submission.py"
SEEDS = tuple(int(x) for x in os.environ.get("JITAI_1K_SEEDS","42,43,44").split(","))
FINAL_EPOCHS = 8


def load_node_module():
    spec = importlib.util.spec_from_file_location("omega005", NODE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def augment_all(mod, data, test_x, test_user, test_date, test_hourmin):
    """Node's add_session_time_features, extended causally into the test window."""
    field_dims = np.asarray(data["field_dims"], dtype=np.int64)
    tab_offset = int(field_dims[:3].sum())
    tab_dim = max(1, int(field_dims[3]))
    tabs = {}
    for split, x in (("train", data["train_x"]), ("val", data["val_x"]), ("test", test_x)):
        tabs[split] = np.clip(x[:, 3] - tab_offset, 0, tab_dim - 1)

    tr_d = mod.normalized_dates(data["train_date"])
    va_d = mod.normalized_dates(data["val_date"])
    te_d = mod.normalized_dates(test_date)
    tr = mod.session_codes(np.asarray(data["train_user"]), tr_d,
                           np.asarray(data["train_hourmin"]), None, 30)
    va = mod.session_codes(np.asarray(data["val_user_eval"]), va_d,
                           np.asarray(data["val_hourmin"]), tr[4], 30)
    te = mod.session_codes(test_user, te_d, test_hourmin, va[4], 30)

    new_dims = [25, 8, 10, 7, 25 * tab_dim, 8 * tab_dim]
    base_next = int(max(data["train_x"].max(initial=0), data["val_x"].max(initial=0)) + 1)
    out = {}
    for split, codes, x in (("train", tr, data["train_x"]), ("val", va, data["val_x"]),
                            ("test", te, test_x)):
        hour, weekday, gap, position = codes[0], codes[1], codes[2], codes[3]
        tab = tabs[split]
        local = [hour, weekday, gap, position,
                 hour * tab_dim + tab, weekday * tab_dim + tab]
        cols = [x]
        next_offset = base_next
        for values, dim in zip(local, new_dims):
            cols.append(np.asarray(values, dtype=np.int64)[:, None] + next_offset)
            next_offset += int(dim)
        out[split] = np.concatenate(cols, axis=1)
    return out


def main():
    mod = load_node_module()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = json.load(open(METRICS))["selected_config"]
    print("recipe:", json.dumps(config))

    data = mod.load_data(str(WORKSPACE))
    test = np.load(TEST_PATH)
    test_hourmin = np.load(TEST_HOURMIN)["hourmin"]
    test_x = np.asarray(test["X"], dtype=np.int64)
    test_user, test_date = np.asarray(test["user_id"]), np.asarray(test["date"])

    aug = augment_all(mod, dict(data), test_x, test_user, test_date, test_hourmin)
    # fidelity check: our train/val augmentation must equal the node's own
    ref = mod.add_session_time_features(dict(data))
    assert (aug["train"] == ref["train_x"]).all(), "train augmentation mismatch"
    assert (aug["val"] == ref["val_x"]).all(), "val augmentation mismatch"
    print("augmentation fidelity: EXACT match with node's own train/val features")

    ref["train_x"], ref["val_x"] = aug["train"], aug["val"]
    # embeddings must span test ids too
    hi = int(max(aug["train"].max(), aug["val"].max(), aug["test"].max()))
    assert hi <= max(aug["train"].max(), aug["val"].max()), \
        f"test ids exceed trained embedding range ({hi})"

    ref["adversarial_probability"] = mod.fit_adversarial_probabilities(ref, 42, 4, device)

    captured = {}
    real_predict = mod.predict
    def capture_predict(model, x, dev):
        captured["model"] = model
        return real_predict(model, x, dev)
    mod.predict = capture_predict

    member_val, member_test = [], []
    for seed in SEEDS:
        metric, val_scores = mod.train_candidate(ref, config, seed, FINAL_EPOCHS,
                                                 device, keep_predictions=True)
        model = captured["model"]
        with torch.no_grad():
            t_scores = real_predict(model, aug["test"], device)
        member_val.append(np.asarray(val_scores))
        member_test.append(np.asarray(t_scores))
        print(f"seed {seed}: val primary {metric['primary']:.6f}")

    def rank_avg(mats):
        ranks = [np.argsort(np.argsort(m, kind="stable"), kind="stable") for m in mats]
        return np.mean(ranks, axis=0)

    ens_val = rank_avg(member_val)
    m = mod.official_metrics(ref, ens_val)
    print(f"ENSEMBLE validation primary: {m['primary']:.6f} "
          f"(gauc {m['gauc']:.6f} ndcg5 {m['ndcg5']:.6f})")

    ens_test = rank_avg(member_test)
    with open(OUTPUT_PATH, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["row_id", "user_id", "video_id", "score"])
        vids = np.asarray(test["video_id"])
        for i in range(len(ens_test)):
            w.writerow([i, int(test_user[i]), int(vids[i]), float(ens_test[i])])
    print(f"wrote {OUTPUT_PATH}: {len(ens_test):,} rows")

    cmd = [sys.executable, str(CHECKER), "--check", str(OUTPUT_PATH), str(TEST_PATH)]
    res = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    print(res.stdout, end="")
    if res.returncode != 0:
        print(res.stderr, end="", file=sys.stderr)
        raise SystemExit(res.returncode)


if __name__ == "__main__":
    main()
