"""Heterogeneous blend audit over EXISTING champion predictions (evidence only).

Components (all validation predictions, different objectives):
  champ  = run_bigclock_07 node_006 (designated, 0.605575)
  tkern  = run_novel_l1 node_004 (temporal pair kernel, 0.60524)
  dsamp  = run_qb_b node_001 (decayed-positive sampling, 0.60466)
  gbce   = run_novel_r1 node_003 (gauge-fixed BCE, 0.60447)
PREDECLARED evaluations (no weight search): (A) equal-weight per-user midrank
blend of all four; (B) champion-anchored 0.6*champ + 0.4*mean(others).
Diagnostics: per-user Spearman vs champ; GAUC-weighted rescue/harm vs champ.
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.official.evaluate import evaluate

COMPONENTS = {
    "champ": "logs/run_bigclock_07/node_006/predictions.csv",
    "tkern": "logs/run_novel_l1/node_004/predictions.csv",
    "dsamp": "logs/run_qb_b/node_001/predictions.csv",
    "gbce":  "logs/run_novel_r1/node_003/predictions.csv",
}


def load(path):
    return np.loadtxt(ROOT / path, delimiter=",", skiprows=1, usecols=3)


def user_groups(users):
    order = np.argsort(users, kind="stable")
    u = users[order]
    cuts = np.r_[0, 1 + np.flatnonzero(u[1:] != u[:-1]), len(u)]
    return order, cuts


def midranks(scores, order, cuts):
    out = np.empty(len(scores))
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        idx = order[lo:hi]
        v = scores[idx]
        srt = np.argsort(v, kind="stable")
        r = np.empty(len(v))
        i = 0
        sv = v[srt]
        while i < len(v):
            j = i + 1
            while j < len(v) and sv[j] == sv[i]:
                j += 1
            r[srt[i:j]] = 0.5 * (i + j - 1)
            i = j
        out[idx] = r / max(1, len(v) - 1)
    return out


def rescue_harm(y, users, champ, cand, order, cuts):
    R = H = W = 0.0
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        idx = order[lo:hi]
        yi = y[idx]
        pos, neg = idx[yi == 1], idx[yi == 0]
        if not len(pos) or not len(neg):
            continue
        w = 1.0 / len(neg)
        cp, cn = champ[pos][:, None], champ[neg][None, :]
        dp, dn = cand[pos][:, None], cand[neg][None, :]
        c_ok = 0.5 * (np.sign(cp - cn) + 1)
        d_ok = 0.5 * (np.sign(dp - dn) + 1)
        diff = d_ok - c_ok
        R += w * np.clip(diff, 0, None).sum()
        H += w * np.clip(-diff, 0, None).sum()
        W += w * diff.size
    return R / W, H / W


def main():
    val = np.load(ROOT / "data/real_ws/val.npz")
    y, users = val["y"].astype(int), val["user"]
    order, cuts = user_groups(users)
    S = {k: load(p) for k, p in COMPONENTS.items()}
    for k, s in S.items():
        assert len(s) == len(y), k
    MR = {k: midranks(s, order, cuts) for k, s in S.items()}
    print("component primaries:")
    for k, s in S.items():
        print(f"  {k}: {evaluate(users, y, s)['primary']:.6f}")
    print("\ndiagnostics vs champion:")
    for k in ("tkern", "dsamp", "gbce"):
        rho = np.corrcoef(MR["champ"], MR[k])[0, 1]
        r, h = rescue_harm(y, users, S["champ"], S[k], order, cuts)
        print(f"  {k}: midrank-corr {rho:.3f}  rescue {r:.4f} harm {h:.4f} "
              f"net {r-h:+.4f}  ratio {r/max(h,1e-9):.2f}")
    others = (MR["tkern"] + MR["dsamp"] + MR["gbce"]) / 3
    blendA = (MR["champ"] + MR["tkern"] + MR["dsamp"] + MR["gbce"]) / 4
    blendB = 0.6 * MR["champ"] + 0.4 * others
    print("\nPREDECLARED blends (evaluated once each):")
    print(f"  A equal-weight 4-way : {evaluate(users, y, blendA)['primary']:.6f}")
    print(f"  B champ-anchored 60/40: {evaluate(users, y, blendB)['primary']:.6f}")
    print(f"  champion reference    : {evaluate(users, y, S['champ'])['primary']:.6f}")


if __name__ == "__main__":
    main()
