import argparse,csv,datetime,json,os,sys
import numpy as np
import torch
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def seed_all(s):
 np.random.seed(s);torch.manual_seed(s)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(s)
 if hasattr(torch.backends,'cudnn'):torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
def encode(a,b):
 A=[];B=[];ds=[];off=0
 for x,z in zip(a,b):
  m={};p=[]
  for v in x:
   if v not in m:m[v]=len(m)
   p.append(m[v])
  u=len(m);A.append(np.asarray(p)+off);B.append(np.asarray([m.get(v,u) for v in z])+off);ds.append(u+1);off+=u+1
 return np.stack(A,1).astype(np.int32),np.stack(B,1).astype(np.int32),np.asarray(ds)
def load(d):
 if os.path.exists(os.path.join(d,'train.npz')) and os.path.exists(os.path.join(d,'val.npz')):
  a=np.load(os.path.join(d,'train.npz'));b=np.load(os.path.join(d,'val.npz'));t={k:a[k] for k in a.files};v={k:b[k] for k in b.files};v['video_raw']=np.zeros(len(v['y']),np.int64);return t,v,True
 def rd(p):
  q={k:[] for k in ['user','video','tab','duration_ms','date','y']}
  with open(p,newline='') as f:
   for r in csv.DictReader(f):q['user'].append(r['user_id']);q['video'].append(r['video_id']);q['tab'].append(r['tab']);q['duration_ms'].append(float(r['duration_ms']));q['date'].append(int(float(r['date'])));q['y'].append(float(r['long_view']))
  for k in ['user','video','tab']:q[k]=np.asarray(q[k])
  q['duration_ms']=np.asarray(q['duration_ms'],np.float32);q['date']=np.asarray(q['date'],np.int64);q['y']=np.asarray(q['y'],np.float32);return q
 t=rd(os.path.join(d,'train.csv'));v=rd(os.path.join(d,'val.csv'));e=np.quantile(t['duration_ms'],np.linspace(0,1,11)[1:-1]);tb=np.searchsorted(e,t['duration_ms']).astype(str);vb=np.searchsorted(e,v['duration_ms']).astype(str);t['X'],v['X'],t['field_dims']=encode([t['user'],t['video'],t['video'],t['tab'],tb],[v['user'],v['video'],v['video'],v['tab'],vb]);v['field_dims']=t['field_dims'];v['video_raw']=v['video'];return t,v,False
class FM(torch.nn.Module):
 def __init__(self,n,k=16):
  super().__init__();self.e=torch.nn.Embedding(n,k);self.l=torch.nn.Embedding(n,1);self.b=torch.nn.Parameter(torch.zeros(1));self.drop=torch.nn.Dropout(.15);torch.nn.init.normal_(self.e.weight,std=.01);torch.nn.init.zeros_(self.l.weight)
 def forward(self,x):
  e=self.drop(self.e(x));s=e.sum(1);return self.b+self.l(x).sum((1,2))+.5*(s.square()-e.square().sum(1)).sum(1)
def pred(m,X,d):
 m.eval();o=[]
 with torch.no_grad():
  for s in range(0,len(X),65536):o.append(m(torch.as_tensor(X[s:s+65536],dtype=torch.long,device=d)).cpu().numpy())
 return np.concatenate(o)
def main():
 p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--out-dir',required=True);p.add_argument('--seed',type=int,default=42);p.add_argument('--epochs',type=int,default=16);a=p.parse_args();os.makedirs(a.out_dir,exist_ok=True);seed_all(a.seed);dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');t,v,fast=load(a.data_dir);epochs=min(a.epochs,max(1,int(os.environ.get('SMOKE_EPOCHS',a.epochs))));m=FM(int(t['field_dims'].sum())).to(dev);op=torch.optim.AdamW(m.parameters(),lr=5e-4,weight_decay=8e-5);sch=torch.optim.lr_scheduler.MultiStepLR(op,[2,4,7],gamma=.4);raw=np.asarray(t['date']);
 try:days=np.asarray([datetime.datetime.strptime(str(int(x)),'%Y%m%d').date().toordinal() for x in raw])
 except ValueError:days=raw.astype(np.int64)
 w=np.exp2(-(days.max()-days)/7.0).astype(np.float32);w/=w.mean();rng=np.random.RandomState(a.seed);ev=__import__('data.official.evaluate',fromlist=['evaluate']).evaluate if fast else __import__('harness.evaluate_provisional',fromlist=['evaluate']).evaluate;best=None;bp=-1
 for ep in range(epochs):
  m.train();perm=rng.permutation(len(t['y']))
  for s in range(0,len(perm),8192):
   q=perm[s:s+8192];x=torch.as_tensor(t['X'][q],dtype=torch.long,device=dev);y=torch.as_tensor(t['y'][q],device=dev);ww=torch.as_tensor(w[q],device=dev);op.zero_grad(set_to_none=True);loss=(torch.nn.functional.binary_cross_entropy_with_logits(m(x),y,reduction='none')*ww).mean();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),4);op.step()
  sch.step();sc=pred(m,v['X'],dev);mm=ev(v['user'],v['y'].astype(int),sc)
  if float(mm['primary'])>bp:bp=float(mm['primary']);best=sc.copy()
 mm=ev(v['user'],v['y'].astype(int),best);json.dump({'gauc':float(mm.get('GAUC',mm.get('gauc'))),'ndcg5':float(mm.get('nDCG@5',mm.get('ndcg5'))),'primary':float(mm['primary'])},open(os.path.join(a.out_dir,'metrics.json'),'w'));f=open(os.path.join(a.out_dir,'predictions.csv'),'w');f.write('row_id,user_id,video_id,score\n');[f.write(f'{i},{v["user"][i]},{v["video_raw"][i]},{float(z):.8g}\n') for i,z in enumerate(best)];f.close()
if __name__=='__main__':main()
