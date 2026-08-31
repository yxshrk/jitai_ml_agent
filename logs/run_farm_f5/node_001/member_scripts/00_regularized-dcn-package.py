import argparse, csv, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_data(data_dir):
    tp = os.path.join(data_dir, 'train.npz')
    vp = os.path.join(data_dir, 'val.npz')
    if os.path.exists(tp) and os.path.exists(vp):
        from data.official.evaluate import evaluate
        tr, va = np.load(tp), np.load(vp)
        dims = tr['field_dims'].astype(np.int64)
        video = va['X'][:, 1].astype(np.int64) - int(dims[0])
        return tr['X'].astype(np.int64), tr['y'].astype(np.float32), va['X'].astype(np.int64), va['y'].astype(np.int64), va['user'], video, dims, evaluate
    from harness.evaluate_provisional import evaluate
    def rows(path, train):
        with open(path, newline='') as f:
            out = []
            for r in csv.DictReader(f):
                out.append((r['user_id'], r['video_id'], r['video_id'], r['tab'], float(r['duration_ms']), int(float(r['long_view']))))
        return out
    rt, rv = rows(os.path.join(data_dir, 'train.csv'), True), rows(os.path.join(data_dir, 'val.csv'), False)
    cuts = np.quantile(np.asarray([r[4] for r in rt]), np.linspace(0.1, 0.9, 9))
    maps = []
    for j in range(4):
        vals = sorted(set(r[j] for r in rt))
        maps.append({v: i + 1 for i, v in enumerate(vals)})
    dims = np.asarray([len(m) + 2 for m in maps] + [11], dtype=np.int64)
    off = np.concatenate(([0], np.cumsum(dims[:-1])))
    def enc(rr):
        x = np.empty((len(rr), 5), np.int64)
        for i, r in enumerate(rr):
            for j in range(4): x[i, j] = off[j] + maps[j].get(r[j], 0)
            x[i, 4] = off[4] + np.searchsorted(cuts, r[4], side='right')
        return x
    return enc(rt), np.asarray([r[5] for r in rt], np.float32), enc(rv), np.asarray([r[5] for r in rv], np.int64), np.asarray([r[0] for r in rv]), np.asarray([r[1] for r in rv]), dims, evaluate


class DCN(torch.nn.Module):
    def __init__(self, total, fields=5, k=16):
        super().__init__()
        self.emb = torch.nn.Embedding(total, k)
        self.lin = torch.nn.Embedding(total, 1)
        d = fields * k
        self.cw = torch.nn.Parameter(torch.empty(2, d))
        self.cb = torch.nn.Parameter(torch.zeros(2, d))
        self.deep = torch.nn.Sequential(torch.nn.Linear(d, 128), torch.nn.ReLU(), torch.nn.Dropout(0.22), torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Dropout(0.12), torch.nn.Linear(64, 1))
        self.cross_out = torch.nn.Linear(d, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)
        torch.nn.init.normal_(self.cw, std=0.01)
    def forward(self, x):
        e = self.emb(x)
        s = e.sum(1)
        fm = 0.5 * (s.square() - e.square().sum(1)).sum(1)
        x0 = e.flatten(1)
        z = x0
        for i in range(2): z = x0 * (z * self.cw[i]).sum(1, keepdim=True) + self.cb[i] + z
        return self.bias + self.lin(x).sum((1, 2)) + fm + self.cross_out(z).squeeze(1) + self.deep(x0).squeeze(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=14)
    a = ap.parse_args()
    smoke = os.environ.get('SMOKE_EPOCHS')
    epochs = min(a.epochs, int(smoke)) if smoke else a.epochs
    np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xt, yt, xv, yv, users, videos, dims, evaluate = load_data(a.data_dir)
    xt = torch.from_numpy(xt); yt = torch.from_numpy(yt); xv = torch.from_numpy(xv)
    model = DCN(int(dims.sum())).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=3e-5)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.55)
    lossfn = torch.nn.BCEWithLogitsLoss(label_smoothing=0.02)
    gen = torch.Generator().manual_seed(a.seed)
    best = -1.; best_scores = None; wait = 0; hist = []; bs = 8192
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(yt), generator=gen); last = 0.
        for p in range(0, len(yt), bs):
            ii = perm[p:p+bs]; xb = xt[ii].to(dev); yb = yt[ii].to(dev)
            opt.zero_grad(set_to_none=True); loss = lossfn(model(xb), yb); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); last = float(loss.detach().cpu())
        sched.step(); model.eval(); parts = []
        with torch.no_grad():
            for p in range(0, len(xv), 65536): parts.append(model(xv[p:p+65536].to(dev)).cpu().numpy())
        scores = np.concatenate(parts); m = evaluate(users, yv, scores); primary = float(m['primary'])
        hist.append({'epoch': ep+1, 'train_loss': round(last, 6), 'val_primary': round(primary, 6)})
        if primary > best + 1e-6: best, best_scores, wait = primary, scores.copy(), 0
        else:
            wait += 1
            if wait >= 3: break
    os.makedirs(a.out_dir, exist_ok=True)
    m = evaluate(users, yv, best_scores)
    with open(os.path.join(a.out_dir, 'metrics.json'), 'w') as f: json.dump({'gauc': m.get('GAUC', m.get('gauc')), 'ndcg5': m.get('nDCG@5', m.get('ndcg5')), 'primary': m['primary'], 'history': hist}, f)
    with open(os.path.join(a.out_dir, 'predictions.csv'), 'w') as f:
        f.write('row_id,user_id,video_id,score\n')
        for i, s in enumerate(best_scores): f.write(f'{i},{users[i]},{videos[i]},{s:.9g}\n')


if __name__ == '__main__': main()
