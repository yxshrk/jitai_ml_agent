import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.official.evaluate import evaluate


class PairKernel(torch.nn.Module):
    def __init__(self, total, k=16):
        super().__init__()
        self.emb = torch.nn.Embedding(total, k)
        self.lin = torch.nn.Embedding(total, 1)
        self.hour = torch.nn.Embedding(24, 8)
        self.day = torch.nn.Embedding(8, 4)
        self.kernel = torch.nn.Sequential(torch.nn.Linear(12 + 2*k, 64), torch.nn.ReLU(), torch.nn.Dropout(0.18), torch.nn.Linear(64, 10))
        self.bias = torch.nn.Parameter(torch.zeros(1))
        torch.nn.init.normal_(self.emb.weight, std=0.01); torch.nn.init.zeros_(self.lin.weight)
    def forward(self, x, hour, day):
        e = self.emb(x)
        ctx = torch.cat((self.hour(hour), self.day(day), e[:, 3], e[:, 4]), 1)
        w = 0.5 + torch.sigmoid(self.kernel(ctx))
        vals = []; q = 0
        for i in range(5):
            for j in range(i+1, 5): vals.append((e[:, i] * e[:, j]).sum(1)); q += 1
        pair = (torch.stack(vals, 1) * w).sum(1)
        return self.bias + self.lin(x).sum((1, 2)) + pair


def temporal(npz):
    hm = np.asarray(npz['hourmin']).astype(np.int64)
    hour = np.where(hm >= 100, hm // 100, hm // 60) % 24
    raw = np.asarray(npz['date']).astype(np.int64)
    uniq = {v: i % 7 for i, v in enumerate(sorted(np.unique(raw).tolist()))}
    day = np.asarray([uniq[int(v)] for v in raw], np.int64)
    return hour.astype(np.int64), day


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data-dir', required=True); ap.add_argument('--out-dir', required=True); ap.add_argument('--seed', type=int, default=137); ap.add_argument('--epochs', type=int, default=15); a = ap.parse_args()
    smoke = os.environ.get('SMOKE_EPOCHS'); epochs = min(a.epochs, int(smoke)) if smoke else a.epochs
    np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tr = np.load(os.path.join(a.data_dir, 'train.npz')); va = np.load(os.path.join(a.data_dir, 'val.npz'))
    xt = torch.from_numpy(tr['X'].astype(np.int64)); yt = torch.from_numpy(tr['y'].astype(np.float32)); xv = torch.from_numpy(va['X'].astype(np.int64))
    ht, dt = temporal(tr); hv, dv = temporal(va); ht = torch.from_numpy(ht); dt = torch.from_numpy(dt); hv = torch.from_numpy(hv); dv = torch.from_numpy(dv)
    users = va['user']; yv = va['y'].astype(np.int64); dims = tr['field_dims'].astype(np.int64); videos = va['X'][:, 1].astype(np.int64) - int(dims[0])
    model = PairKernel(int(dims.sum())).to(dev); opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=4e-5); sched = torch.optim.lr_scheduler.StepLR(opt, 2, gamma=0.6); bce = torch.nn.BCEWithLogitsLoss(label_smoothing=0.015)
    gen = torch.Generator().manual_seed(a.seed); best = -1.; best_scores = None; wait = 0; hist = []; bs = 8192
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(yt), generator=gen); last = 0.
        for p in range(0, len(yt), bs):
            ii = perm[p:p+bs]; opt.zero_grad(set_to_none=True)
            loss = bce(model(xt[ii].to(dev), ht[ii].to(dev), dt[ii].to(dev)), yt[ii].to(dev)); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); last = float(loss.detach().cpu())
        sched.step(); model.eval(); parts = []
        with torch.no_grad():
            for p in range(0, len(xv), 65536): parts.append(model(xv[p:p+65536].to(dev), hv[p:p+65536].to(dev), dv[p:p+65536].to(dev)).cpu().numpy())
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
