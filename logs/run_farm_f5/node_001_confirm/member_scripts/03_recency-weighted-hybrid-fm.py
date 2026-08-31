import argparse, json, math, os, sys
from collections import defaultdict
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.official.evaluate import evaluate


class FM(torch.nn.Module):
    def __init__(self, total, k=16):
        super().__init__(); self.emb = torch.nn.Embedding(total, k); self.lin = torch.nn.Embedding(total, 1); self.bias = torch.nn.Parameter(torch.zeros(1)); self.drop = torch.nn.Dropout(0.12)
        torch.nn.init.normal_(self.emb.weight, std=0.01); torch.nn.init.zeros_(self.lin.weight)
    def forward(self, x):
        e = self.drop(self.emb(x)); s = e.sum(1); pair = 0.5 * (s.square() - e.square().sum(1)).sum(1)
        return self.bias + self.lin(x).sum((1, 2)) + pair


def recency(date):
    d = np.asarray(date).astype(np.int64); uniq = sorted(np.unique(d).tolist()); rank = {v: i for i, v in enumerate(uniq)}; age = np.asarray([len(uniq)-1-rank[int(v)] for v in d], np.float32)
    w = np.exp(-math.log(2.0) * age / 7.0); return (w / w.mean()).astype(np.float32)


def make_pairs(users, labels, seed):
    groups = defaultdict(lambda: [[], []])
    for i, (u, y) in enumerate(zip(users, labels.astype(np.int64))): groups[int(u)][int(y)].append(i)
    rng = np.random.RandomState(seed); pos = []; neg = []
    for p, n in groups.values():
        if p and n:
            pp = np.asarray(p, np.int64); nn = np.asarray(n, np.int64); pos.extend(pp.tolist()); neg.extend(nn[rng.randint(0, len(nn), len(pp))].tolist())
    return np.asarray(pos, np.int64), np.asarray(neg, np.int64)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data-dir', required=True); ap.add_argument('--out-dir', required=True); ap.add_argument('--seed', type=int, default=911); ap.add_argument('--epochs', type=int, default=15); a = ap.parse_args()
    smoke = os.environ.get('SMOKE_EPOCHS'); epochs = min(a.epochs, int(smoke)) if smoke else a.epochs
    np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tr = np.load(os.path.join(a.data_dir, 'train.npz')); va = np.load(os.path.join(a.data_dir, 'val.npz'))
    xnp = tr['X'].astype(np.int64); ynp = tr['y'].astype(np.float32); xt = torch.from_numpy(xnp); yt = torch.from_numpy(ynp); xv = torch.from_numpy(va['X'].astype(np.int64)); wt = torch.from_numpy(recency(tr['date']))
    pi, ni = make_pairs(tr['user'], ynp, a.seed); pi = torch.from_numpy(pi); ni = torch.from_numpy(ni)
    dims = tr['field_dims'].astype(np.int64); users = va['user']; yv = va['y'].astype(np.int64); videos = va['X'][:, 1].astype(np.int64) - int(dims[0])
    model = FM(int(dims.sum())).to(dev); opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=5e-5); sched = torch.optim.lr_scheduler.StepLR(opt, 2, gamma=0.6)
    gen = torch.Generator().manual_seed(a.seed); best = -1.; best_scores = None; wait = 0; hist = []; bs = 8192
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(yt), generator=gen); pperm = torch.randperm(len(pi), generator=gen) if len(pi) else None; last = 0.
        for b, st in enumerate(range(0, len(yt), bs)):
            ii = perm[st:st+bs]; xb = xt[ii].to(dev); yb = yt[ii].to(dev); wb = wt[ii].to(dev); logits = model(xb)
            bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, yb, reduction='none'); loss_bce = (bce * wb).sum() / wb.sum().clamp_min(1.)
            if len(pi):
                q = (b * bs) % len(pi); jj = pperm[q:min(q+bs, len(pi))]
                if len(jj) == 0: jj = pperm[:min(bs, len(pi))]
                sp = model(xt[pi[jj]].to(dev)); sn = model(xt[ni[jj]].to(dev)); loss_pair = torch.nn.functional.softplus(-(sp-sn)).mean(); loss = 0.5 * loss_bce + 0.5 * loss_pair
            else: loss = loss_bce
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); last = float(loss.detach().cpu())
        sched.step(); model.eval(); parts = []
        with torch.no_grad():
            for p in range(0, len(xv), 65536): parts.append(model(xv[p:p+65536].to(dev)).cpu().numpy())
        scores = np.concatenate(parts); m = evaluate(users, yv, scores); primary = float(m['primary']); hist.append({'epoch': ep+1, 'train_loss': round(last, 6), 'val_primary': round(primary, 6)})
        if primary > best + 1e-6: best, best_scores, wait = primary, scores.copy(), 0
        else:
            wait += 1
            if wait >= 3: break
    os.makedirs(a.out_dir, exist_ok=True); m = evaluate(users, yv, best_scores)
    with open(os.path.join(a.out_dir, 'metrics.json'), 'w') as f: json.dump({'gauc': m.get('GAUC', m.get('gauc')), 'ndcg5': m.get('nDCG@5', m.get('ndcg5')), 'primary': m['primary'], 'history': hist}, f)
    with open(os.path.join(a.out_dir, 'predictions.csv'), 'w') as f:
        f.write('row_id,user_id,video_id,score\n')
        for i, s in enumerate(best_scores): f.write(f'{i},{users[i]},{videos[i]},{s:.9g}\n')


if __name__ == '__main__': main()
