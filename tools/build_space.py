"""Visualization instrumentation (NOT a scored run): retrain the baseline FM and
the champion package config (same recipes as tools/instrument_weights.py), dump
each model's VIDEO embedding table, project both through a shared PCA basis
(fit on the treated space), and write site/space.js for the Learned Space site.

Per video: 3D position in both spaces, train-frequency decile, impressions,
positive (long_view) rate. Coordinates quantized to keep the file ~1-2 MB.
"""
import json
from pathlib import Path
import sys
import numpy as np, torch
sys.path.insert(0, '.')
from data.official.evaluate import evaluate

tr = np.load('data/real_ws/train.npz'); va = np.load('data/real_ws/val.npz')
Xt, yt = torch.as_tensor(tr['X'].astype(np.int64)), torch.as_tensor(tr['y'].astype(np.float32))
Xv = torch.as_tensor(va['X'].astype(np.int64)); yv = va['y'].astype(int); uv = va['user']
dims = tr['field_dims']; total = int(dims.sum())

vid_col = Xt[:, 1].numpy()
lo, hi = int(vid_col.min()), int(vid_col.max())
counts = np.bincount(vid_col - lo, minlength=hi - lo + 1)
pos = np.bincount(vid_col - lo, weights=tr['y'], minlength=hi - lo + 1)
vids = np.flatnonzero(counts > 0)                    # embedding rows lo+vids
dec_edges = np.quantile(counts[vids], np.linspace(0, 1, 11))
decile = np.clip(np.searchsorted(dec_edges, counts[vids], side='right') - 1, 0, 9)


class Net(torch.nn.Module):
    def __init__(self, k=16, mlp=False, dropout=0.0):
        super().__init__()
        self.emb = torch.nn.Embedding(total, k); torch.nn.init.normal_(self.emb.weight, std=0.01)
        self.lin = torch.nn.Embedding(total, 1); torch.nn.init.zeros_(self.lin.weight)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.mlp = torch.nn.Sequential(torch.nn.Linear(5*k,128), torch.nn.ReLU(),
                    torch.nn.Dropout(dropout), torch.nn.Linear(128,1)) if mlp else None
        self.drop = torch.nn.Dropout(dropout)
    def forward(self, x):
        e = self.emb(x); s = e.sum(1)
        fm = self.bias + self.lin(x).sum((1,2)) + 0.5*(s*s-(e*e).sum(1)).sum(1)
        if self.mlp is not None:
            fm = fm + self.mlp(self.drop(e).flatten(1)).squeeze(1)
        return fm


def run(tag, mlp, dropout, wd, lr, gamma, step, halflife, epochs=8):
    torch.manual_seed(42)
    net = Net(mlp=mlp, dropout=dropout)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1,int(step)), gamma=gamma)
    if halflife:
        w = torch.as_tensor((0.5 ** ((tr['date'].max()-tr['date'])/halflife)).astype(np.float32))
    else:
        w = torch.ones(len(yt))
    bs, n = 8192, len(yt)
    g = torch.Generator().manual_seed(0)
    for ep in range(epochs):
        order = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            idx = order[i:i+bs]
            opt.zero_grad()
            out = net(Xt[idx])
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(
                out, yt[idx], reduction='none') * w[idx]).mean()
            loss.backward(); opt.step()
        sch.step()
        net.eval()
        with torch.no_grad():
            m = evaluate(uv, yv, net(Xv).numpy())
        print(tag, ep, round(m['primary'], 5)); net.train()
    net.eval()
    with torch.no_grad():
        E = net.emb.weight[torch.as_tensor(lo + vids)].numpy()
        primary = evaluate(uv, yv, net(Xv).numpy())['primary']
    return E, round(primary, 5)


E_base, p_base = run('base', mlp=False, dropout=0.0, wd=1e-6, lr=1e-3, gamma=1.0, step=1, halflife=None)
E_treat, p_treat = run('treat', mlp=True, dropout=0.18, wd=9e-5, lr=1e-3, gamma=0.57, step=2, halflife=7.0)

# shared PCA basis fit on the treated space; both clouds projected through it
mu = E_treat.mean(0)
U, S, Vt = np.linalg.svd(E_treat - mu, full_matrices=False)
P = Vt[:3].T
proj = lambda E: (E - mu) @ P


def norm01(A):
    lo_, hi_ = np.percentile(A, 1, axis=0), np.percentile(A, 99, axis=0)
    return np.clip((A - lo_) / (hi_ - lo_ + 1e-9), 0, 1)


B, T = norm01(proj(E_base)), norm01(proj(E_treat))
nb = np.linalg.norm(E_base, axis=1); nt = np.linalg.norm(E_treat, axis=1)
q = lambda A: (A * 1000).astype(int).tolist()
rate = (pos[vids] / np.maximum(1, counts[vids]))

payload = {
    'meta': {'p_base': p_base, 'p_treat': p_treat, 'n': int(len(vids)),
             'note': 'instrumented retrains for visualization only — not scored runs'},
    'base': q(B), 'treat': q(T),
    'decile': decile.tolist(),
    'impressions': counts[vids].astype(int).tolist(),
    'rate': (rate * 100).round(1).tolist(),
    'norm_base': np.round(nb, 3).tolist(), 'norm_treat': np.round(nt, 3).tolist(),
}
Path('site/space.js').write_text('window.SPACE=' + json.dumps(payload) + ';')
print(f'wrote site/space.js: {len(vids)} videos, base {p_base} -> treat {p_treat}')
