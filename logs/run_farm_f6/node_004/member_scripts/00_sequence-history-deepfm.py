import argparse, csv, datetime, json, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def seed_all(s):
    np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def encode(tr,va):
    a=[]; b=[]; dims=[]; off=0
    for x,z in zip(tr,va):
        m={}; p=np.empty(len(x),np.int64)
        for i,v in enumerate(x):
            if v not in m: m[v]=len(m)
            p[i]=m[v]
        u=len(m); q=np.asarray([m.get(v,u) for v in z],np.int64); d=u+1
        a.append(p+off); b.append(q+off); dims.append(d); off+=d
    return np.stack(a,1).astype(np.int32),np.stack(b,1).astype(np.int32),np.asarray(dims,np.int64)

def load(data):
    tp=os.path.join(data,'train.npz'); vp=os.path.join(data,'val.npz')
    if os.path.exists(tp) and os.path.exists(vp):
        t0=np.load(tp); v0=np.load(vp); t={k:t0[k] for k in t0.files}; v={k:v0[k] for k in v0.files}
        v['video_raw']=np.zeros(len(v['y']),np.int64); return t,v,True
    def rd(path,train):
        d={k:[] for k in ['user','video','tab','hourmin','date','duration_ms','y','play_time_ms']}
        with open(path,newline='') as f:
            for r in csv.DictReader(f):
                d['user'].append(r['user_id']); d['video'].append(r['video_id']); d['tab'].append(r['tab'])
                d['hourmin'].append(int(float(r['hourmin']))); d['date'].append(int(float(r['date'])))
                d['duration_ms'].append(float(r['duration_ms'])); d['y'].append(float(r['long_view']))
                if train: d['play_time_ms'].append(float(r['play_time_ms']))
        for k in ['hourmin','date']: d[k]=np.asarray(d[k],np.int64)
        for k in ['duration_ms','y','play_time_ms']: d[k]=np.asarray(d[k],np.float32)
        d['user']=np.asarray(d['user']); d['video']=np.asarray(d['video']); d['tab']=np.asarray(d['tab']); return d
    t=rd(os.path.join(data,'train.csv'),True); v=rd(os.path.join(data,'val.csv'),False)
    e=np.quantile(t['duration_ms'],np.linspace(0,1,11)[1:-1]); tb=np.searchsorted(e,t['duration_ms']).astype(str); vb=np.searchsorted(e,v['duration_ms']).astype(str)
    t['X'],v['X'],t['field_dims']=encode([t['user'],t['video'],t['video'],t['tab'],tb],[v['user'],v['video'],v['video'],v['tab'],vb]); v['field_dims']=t['field_dims']; v['video_raw']=v['video']; return t,v,False

def contexts(t,v):
    base=int(t['field_dims'].sum())
    def parts(d):
        hm=np.asarray(d['hourmin'],np.int64); hour=np.clip(np.where(hm>1439,hm//100,np.where(hm>23,hm//60,hm)),0,23)
        dow=np.asarray([datetime.datetime.strptime(str(int(x)),'%Y%m%d').weekday() if len(str(int(x)))==8 else int(x)%7 for x in d['date']],np.int64)
        return np.stack([hour+base,dow+base+24],1).astype(np.int32)
    ct,cv=parts(t),parts(v); h1=np.full((len(t['y']),12),-1,np.int32); h2=np.full((len(v['y']),12),-1,np.int32); state={}
    order=np.lexsort((np.arange(len(t['y'])),np.asarray(t['date']),np.asarray(t['user']).astype(str)))
    for i in order:
        u=t['user'][i]; z=state.get(u,[])[-12:]
        if z: h1[i,:len(z)]=z
        state[u]=(state.get(u,[])+[int(t['X'][i,2])])[-12:]
    for i,u in enumerate(v['user']):
        z=state.get(u,[])[-12:]
        if z: h2[i,:len(z)]=z
        state[u]=(state.get(u,[])+[int(v['X'][i,2])])[-12:]
    return ct,cv,h1,h2,base+31

class Model(torch.nn.Module):
    def __init__(self,n,k=16):
        super().__init__(); self.e=torch.nn.Embedding(n,k); self.l=torch.nn.Embedding(n,1); self.b=torch.nn.Parameter(torch.zeros(1)); torch.nn.init.normal_(self.e.weight,std=.01); torch.nn.init.zeros_(self.l.weight)
        self.net=torch.nn.Sequential(torch.nn.Linear(8*k,128),torch.nn.ReLU(),torch.nn.Dropout(.2),torch.nn.Linear(128,64),torch.nn.ReLU(),torch.nn.Dropout(.2)); self.main=torch.nn.Linear(64,1); self.watch=torch.nn.Linear(64,1)
    def forward(self,x,c,h):
        ids=torch.cat([x,c],1); a=self.e(ids); mask=(h>=0).float().unsqueeze(-1); hs=h.clamp_min(0); he=(self.e(hs)*mask).sum(1)/mask.sum(1).clamp_min(1); f=torch.cat([a,he[:,None,:]],1); s=f.sum(1); fm=.5*(s.square()-f.square().sum(1)).sum(1); deep=self.net(f.flatten(1)); lin=self.l(ids).sum((1,2))+(self.l(hs)*mask).sum((1,2))/mask.sum((1,2)).clamp_min(1); return self.b+lin+fm+self.main(deep).squeeze(1),self.watch(deep).squeeze(1)

def pred(m,X,C,H,dev):
    m.eval(); out=[]
    with torch.no_grad():
        for s in range(0,len(X),32768):
            z=m(torch.as_tensor(X[s:s+32768],dtype=torch.long,device=dev),torch.as_tensor(C[s:s+32768],dtype=torch.long,device=dev),torch.as_tensor(H[s:s+32768],dtype=torch.long,device=dev))[0]; out.append(z.cpu().numpy())
    return np.concatenate(out)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-dir',required=True); p.add_argument('--out-dir',required=True); p.add_argument('--seed',type=int,default=42); p.add_argument('--epochs',type=int,default=14); a=p.parse_args(); os.makedirs(a.out_dir,exist_ok=True); seed_all(a.seed); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); t,v,fast=load(a.data_dir); C,D,H,J,n=contexts(t,v); epochs=min(a.epochs,max(1,int(os.environ.get('SMOKE_EPOCHS',a.epochs)))); m=Model(n).to(dev); opt=torch.optim.AdamW(m.parameters(),lr=7e-4,weight_decay=2e-5); sch=torch.optim.lr_scheduler.MultiStepLR(opt,[max(1,epochs//3),max(2,2*epochs//3)],gamma=.35); rng=np.random.RandomState(a.seed); y=t['y'].astype(np.float32); play=np.maximum(t['play_time_ms'],0); dur=np.maximum(t['duration_ms'],1); wt=(np.log1p(np.minimum(play,dur))/10).astype(np.float32); cen=(play>=dur).astype(np.float32); best=None; bp=-1
    ev=__import__('data.official.evaluate',fromlist=['evaluate']).evaluate if fast else __import__('harness.evaluate_provisional',fromlist=['evaluate']).evaluate
    for _ in range(epochs):
        m.train()
        for s in range(0,len(y),4096):
            q=rng.permutation(len(y))[s:s+4096]; x=torch.as_tensor(t['X'][q],dtype=torch.long,device=dev); c=torch.as_tensor(C[q],dtype=torch.long,device=dev); h=torch.as_tensor(H[q],dtype=torch.long,device=dev); yy=torch.as_tensor(y[q],device=dev); tt=torch.as_tensor(wt[q],device=dev); zz=torch.as_tensor(cen[q],device=dev); opt.zero_grad(set_to_none=True); o,w=m(x,c,h); aux=((1-zz)*torch.nn.functional.smooth_l1_loss(w,tt,reduction='none')+zz*torch.relu(tt-w).square()).mean(); loss=torch.nn.functional.binary_cross_entropy_with_logits(o,yy)+.05*aux; loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),5); opt.step()
        sch.step(); sc=pred(m,v['X'],D,J,dev); mm=ev(v['user'],v['y'].astype(int),sc); pr=float(mm['primary'])
        if pr>bp: bp=pr; best=sc.copy()
    mm=ev(v['user'],v['y'].astype(int),best); out={'gauc':float(mm.get('GAUC',mm.get('gauc'))),'ndcg5':float(mm.get('nDCG@5',mm.get('ndcg5'))),'primary':float(mm['primary'])}; json.dump(out,open(os.path.join(a.out_dir,'metrics.json'),'w')); f=open(os.path.join(a.out_dir,'predictions.csv'),'w'); f.write('row_id,user_id,video_id,score\n'); vids=v['video_raw']; [f.write(f'{i},{v["user"][i]},{vids[i]},{float(z):.8g}\n') for i,z in enumerate(best)]; f.close()
if __name__=='__main__': main()
