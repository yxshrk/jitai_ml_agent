import argparse, json, os, sys
from collections import defaultdict, deque
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.official.evaluate import evaluate


def histories(xtr, xval, length, pad):
    state = defaultdict(lambda: deque(maxlen=length))
    ht = np.full((len(xtr), length), pad, np.int64)
    for i, (u, v) in enumerate(zip(xtr[:, 0], xtr[:, 1])):
        h = list(state[int(u)])
        if h: ht[i, -len(h):] = h
        state[int(u)].append(int(v))
    hv = np.full((len(xval), length), pad, np.int64)
    for i, u in enumerate(xval[:, 0]):
        h = list(state[int(u)])
        if h: hv[i, -len(h):] = h
    return ht, hv


class SeqDeepFM(torch.nn.Module):
    def __init__(self, total, pad, k=16):
        super().__init__(); self.pad = pad
        self.emb = torch.nn.Embedding(total + 1, k, padding_idx=pad); self.lin = torch.nn.Embedding(total, 1)
        self.att = torch.nn.Linear(2*k, 1)
        self.deep = torch.nn.Sequential(torch.nn.Linear(6*k, 128), torch.nn.ReLU(), torch.nn.Dropout(0.24), torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Dropout(0.14), torch.nn.Linear(64, 1))
        self.gate = torch.nn.Sequential(torch.nn.Linear(2*k, k), torch.nn.Sigmoid())
        self.bias = torch.nn.Parameter(torch.zeros(1)); torch.nn.init.normal_(self.emb.weight, std=0.01); torch.nn.init.zeros_(self.lin.weight)
    def forward(self, x, hist):
        e = self.emb(x); he = self.emb(hist); q = e[:, 1].unsqueeze(1).expand_as(he)
        logits = self.att(torch.cat((q, he), 2)).squeeze(2).masked_fill(hist.eq(self.pad), -1e4)
        alpha = torch.softmax(logits, 1); valid = hist.ne(self.pad).any(1, keepdim=True).float(); pooled = (alpha.unsqueeze(2) * he).sum(1) * valid
        gated = pooled * self.gate(torch.cat((e[:, 1], pooled), 1))
        s = e.sum(1); fm = 0.5 * (s.square() - e.square().sum(1)).sum(1)
        return self.bias + self.lin(x).sum((1, 2)) + fm + self.deep(torch.cat((e.flatten(1), gated), 1)).squeeze(1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data-dir', required=True); ap.add_argument('--out-dir', required=True); ap.add_argument('--seed', type=int, default=271); ap.add_argument('--epochs', type=int, default=13); a = ap.parse_args()
    smoke = os.environ.get('SMOKE_EPOCHS'); epochs = min(a.epochs, int(smoke)) if smoke else a.epochs
    np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tr = np.load(os.path.join(a.data_dir, 'train.npz')); va = np.load(os.path.join(a.data_dir, 'val.npz'))
    xtr = tr['X'].astype(np.int64); xval = va['X'].astype(np.int64); dims = tr['field_dims'].astype(np.int64); total = int(dims.sum()); pad = total
    htr, hval = histories(xtr, xval, 5, pad)
    xt = torch.from_numpy(xtr); yt = torch.from_numpy(tr['y'].astype(np.float32)); xv = torch.from_numpy(xval); ht = torch.from_numpy(htr); hv = torch.from_numpy(hval)
    users = va['user']; yv = va['y'].astype(np.int64); videos = xval[:, 1] - int(dims[0])
    model = SeqDeepFM(total, pad).to(dev); opt = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=4e-5); sched = torch.optim.lr_scheduler.StepLR(opt, 2, gamma=0.58); bce = torch.nn.BCEWithLogitsLoss(label_smoothing=0.02)
    gen = torch.Generator().manual_seed(a.seed); best = -1.; best_scores = None; wait = 0; histlog = []; bs = 6144
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(yt), generator=gen); last = 0.
        for p in range(0, len(yt), bs):
            ii = perm[p:p+bs]; opt.zero_grad(set_to_none=True); loss = bce(model(xt[ii].to(dev), ht[ii].to(dev)), yt[ii].to(dev)); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); last = float(loss.detach().cpu())
        sched.step(); model.eval(); parts = []
        with torch.no_grad():
            for p in range(0, len(xv), 32768): parts.append(model(xv[p:p+32768].to(dev), hv[p:p+32768].to(dev)).cpu().numpy())
        scores = np.concatenate(parts); m = evaluate(users, yv, scores); primary = float(m['primary']); histlog.append({'epoch': ep+1, 'train_loss': round(last, 6), 'val_primary': round(primary, 6)})
        if primary > best + 1e-6: best, best_scores, wait = primary, scores.copy(), 0
        else:
            wait += 1
            if wait >= 3: break
    os.makedirs(a.out_dir, exist_ok=True); m = evaluate(users, yv, best_scores)
    with open(os.path.join(a.out_dir, 'metrics.json'), 'w') as f: json.dump({'gauc': m.get('GAUC', m.get('gauc')), 'ndcg5': m.get('nDCG@5', m.get('ndcg5')), 'primary': m['primary'], 'history': histlog}, f)
    with open(os.path.join(a.out_dir, 'predictions.csv'), 'w') as f:
        f.write('row_id,user_id,video_id,score\n')
        for i, s in enumerate(best_scores): f.write(f'{i},{users[i]},{videos[i]},{s:.9g}\n')


if __name__ == '__main__': main()
