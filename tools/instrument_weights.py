"""Visualization instrumentation (NOT a scored run): retrain baseline FM and the
champion package config on the real train split, logging every half-epoch:
 - median embedding norm of VIDEO ids by train-frequency decile (rare..common)
 - validation primary
Output: site/weights.json  — powers the 'inside the network' animation.
"""
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, '.')
from data.official.evaluate import evaluate

tr = np.load('data/real_ws/train.npz'); va = np.load('data/real_ws/val.npz')
Xt, yt = torch.as_tensor(tr['X'].astype(np.int64)), torch.as_tensor(tr['y'].astype(np.float32))
Xv = torch.as_tensor(va['X'].astype(np.int64)); yv = va['y'].astype(int); uv = va['user']
dims = tr['field_dims']; total = int(dims.sum())
# video field = column 1; frequency deciles from train counts
vid_col = Xt[:, 1].numpy()
lo, hi = vid_col.min(), vid_col.max()
counts = np.bincount(vid_col - lo)
dec = np.quantile(counts[counts > 0], np.linspace(0, 1, 11))
def decile_masks():
    ms = []
    for i in range(10):
        m = (counts >= dec[i]) & (counts <= dec[i+1]) & (counts > 0)
        ms.append(np.flatnonzero(m) + lo)
    return ms
DM = decile_masks()

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
    snaps = []
    def snap(ck):
        with torch.no_grad():
            W = net.emb.weight
            norms = [float(torch.linalg.norm(W[torch.as_tensor(ix)],dim=1).median()) for ix in DM]
            sc = net(Xv).numpy()
        m = evaluate(uv, yv, sc)
        snaps.append({'ck': ck, 'primary': round(m['primary'],5), 'norms': [round(x,5) for x in norms]})
        print(tag, ck, round(m['primary'],5))
    g = torch.Generator().manual_seed(0)
    for ep in range(epochs):
        order = torch.randperm(n, generator=g)
        half = n//2
        for part,(a,b) in enumerate(((0,half),(half,n))):
            net.train()
            for i in range(a,b,bs):
                idx = order[i:i+bs]
                opt.zero_grad()
                out = net(Xt[idx])
                loss = (torch.nn.functional.binary_cross_entropy_with_logits(
                    out, yt[idx], reduction='none') * w[idx]).mean()
                loss.backward(); opt.step()
            net.eval(); snap(ep + 0.5*(part+1))
        sch.step()
    return snaps

out = {}
out['baseline'] = run('base', mlp=False, dropout=0.0, wd=1e-6, lr=1e-3, gamma=1.0, step=1, halflife=None, epochs=8)
out['treated']  = run('treat', mlp=True, dropout=0.18, wd=9e-5, lr=1e-3, gamma=0.57, step=2, halflife=7.0, epochs=8)
out['decile_counts'] = [int(counts[(counts>=dec[i])&(counts<=dec[i+1])&(counts>0)].mean()) for i in range(10)]
json.dump(out, open('site/weights.json','w'))
print('wrote site/weights.json')
