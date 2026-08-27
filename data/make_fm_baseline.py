"""Generate official FM-baseline validation predictions (starter kit, seed 0).

Runs the starter kit's own FM (imported read-only from ../starter-kit) on the real
data and caches its validation predictions to data/real/fm_baseline_valid.npz, so
tests can verify the vendored evaluate.py reproduces the official FM score.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT.parent / "starter-kit" / "kuairand-starter-kit"
OUT = ROOT / "data" / "real" / "fm_baseline_valid.npz"

sys.path.insert(0, str(KIT))
from baseline import FM  # noqa: E402
from data import encode, load  # noqa: E402
from evaluate import evaluate  # noqa: E402


def main(seed: int = 0) -> None:
    splits = load(str(ROOT.parent / "KuaiRand-Pure" / "data"))
    enc, dim = encode(splits)
    Xtr, ytr, _ = enc["train"]
    Xva, yva, uva = enc["valid"]
    m = FM(dim, k=16, lr=0.001, seed=seed)
    rng = np.random.default_rng(seed)
    bs, best, best_state, bad = 8192, -1.0, None, 0
    for ep in range(1, 41):
        idx = rng.permutation(len(ytr))
        t0 = time.time()
        losses = [m.step(Xtr[idx[i:i + bs]], ytr[idx[i:i + bs]]) for i in range(0, len(idx), bs)]
        va = evaluate(uva, yva, m.predict(Xva))
        print(f"epoch {ep:2d} loss {np.mean(losses):.4f} valid primary {va['primary']:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if va["primary"] > best + 1e-5:
            best, bad = va["primary"], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= 4:
                break
    m.V, m.W, m.b = best_state
    scores = m.predict(Xva)
    va = evaluate(uva, yva, scores)
    print("final valid:", va)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "wb") as fh:
        np.savez(fh, scores=scores.astype(np.float64), labels=yva,
                 users=np.asarray([int(u) for u in uva], dtype=np.int64),
                 gauc=va["GAUC"], ndcg5=va["nDCG@5"], primary=va["primary"])


if __name__ == "__main__":
    main()
