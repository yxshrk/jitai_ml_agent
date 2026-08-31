import argparse,csv,json,os,random
import numpy as np
import torch

def seed_all(s):
 random.seed(s);np.random.seed(s);torch.manual_seed(s)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(s)
 if hasattr(torch.backends,'cudnn'):torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False

def load(d):
 p=os.path.join(d,'train.npz');q=os.path.join(d,'val.npz')
 if os.path.exists(p) and os.path.exists(q):
  a=np.load(p);b=np.load(q);tr={k:a[k] for k in a.files};va={k:b[k] for k in b.files};va['video_raw']=np.zeros(len(va['y']),dtype=np.int64);return tr,va,True
 def rd(path):
  z={k:[] for k in ['user_id','video_id','tab','hourmin','date','duration_ms','long_view']}
  with open(path,newline='') as f:
   for r in csv.DictReader(f):
    for k in z:z[k].append(r[k])
  for k in ['hourmin','date']:z[k]=np.asarray(z[k],dtype=np.int64)
  for k in ['duration_ms','long_view']:z[k]=np.asarray(z[k],dtype=np.float32)
  return z
 tr,va=rd(os.path.join(d,'train.csv')),rd(os.path.join(d,'val.csv'));cuts=np.quantile(tr['duration_ms'],np.arange(1,10)/10);rawtr=[tr['user_id'],tr['video_id'],tr['video_id'],tr['tab'],np.searchsorted(cuts,tr['duration_ms']).astype(str)];rawva=[va['user_id'],va['video_id'],va['video_id'],va['tab'],np.searchsorted(cuts,va['duration_ms']).astype(str)];xs=[];xv=[];dims=[];off=0
 for a,b in zip(rawtr,rawva):
  m={};u=np.empty(len(a),np.int64)
  for i,v in enumerate(a):m.setdefault(v,len(m));u[i]=m[v]
  w=np.asarray([m.get(v,len(m)) for v in b]);dim=len(m)+1;xs.append(u+off);xv.append(w+off);dims.append(dim);off+=dim
 return {'X':np.stack(xs,1).astype(np.int32),'y':tr['long_view'],'user':np.asarray(tr['user_id']),'date':tr['date'],'field_dims':np.asarray(dims)}, {'X':np.stack(xv,1).astype(np.int32),'y':va['long_view'],'user':np.asarray(va['user_id']),'video_raw':np.asarray(va['video_id'])},False

def metrics(fast,u,y,s):
 if fast:
  from data.official.evaluate import evaluate
 else:
  from harness.evaluate_provisional import evaluate
 m=evaluate(u,y.astype(int),s);return {'gauc':float(m.get('GAUC',m.get('gauc'))),'ndcg5':float(m.get('nDCG@5',m.get('ndcg5'))),'primary':float(m['primary'])}

class DCN(torch.nn.Module):
 def __init__(self,n):
  super().__init__();self.e=torch.nn.Embedding(n,16);self.l=torch.nn.Embedding(n,1);torch.nn.init.normal_(self.e.weight,std=.01);torch.nn.init.zeros_(self.l.weight);self.w=torch.nn.Parameter(torch.randn(80)*.01);self.b=torch.nn.Parameter(torch.zeros(80));self.deep=torch.nn.Sequential(torch.nn.Linear(80,128),torch.nn.ReLU(),torch.nn.Dropout(.30),torch.nn.Linear(128,64),torch.nn.ReLU(),torch.nn.Dropout(.25));self.out=torch.nn.Linear(144,1);self.bias=torch.nn.Parameter(torch.zeros(1))
 def forward(self,x):
  z=self.e(x).flatten(1);cross=z*(z@self.w).unsqueeze(1)+self.b+z;h=self.deep(z);return self.bias+self.l(x).sum((1,2))+self.out(torch.cat([cross,h],1)).squeeze(1)

def pred(m,x,dev):
 m.eval();r=[]
 with torch.no_grad():
  for i in range(0,len(x),32768):r.append(m(torch.as_tensor(x[i:i+32768],dtype=torch.long,device=dev)).cpu().numpy())
 return np.concatenate(r)

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--out-dir',required=True);p.add_argument('--seed',type=int,default=42);p.add_argument('--epochs',type=int,default=14);a=p.parse_args();os.makedirs(a.out_dir,exist_ok=True);seed_all(a.seed);dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');tr,va,fast=load(a.data_dir);ep=a.epochs;sm=os.environ.get('SMOKE_EPOCHS');ep=min(ep,max(1,int(sm))) if sm else ep;m=DCN(int(np.sum(tr['field_dims']))).to(dev);opt=torch.optim.AdamW(m.parameters(),lr=6e-4,weight_decay=8e-5);sch=torch.optim.lr_scheduler.MultiStepLR(opt,[max(1,ep//3),max(2,2*ep//3)],gamma=.3);x=tr['X'];y=tr['y'].astype(np.float32);dates=np.asarray(tr.get('date',np.zeros(len(y))));uniq={v:i for i,v in enumerate(sorted(set(dates.tolist())))};age=np.asarray([len(uniq)-1-uniq[v] for v in dates]);wt=np.exp(-np.log(2)*age/7).astype(np.float32);rng=np.random.RandomState(a.seed);best=None;bm=None;hist=[];pat=0
 for e in range(ep):
  m.train();perm=rng.permutation(len(y));losses=[]
  for j in range(0,len(y),4096):
   ix=perm[j:j+4096];xb=torch.as_tensor(x[ix],dtype=torch.long,device=dev);yb=torch.as_tensor(y[ix],device=dev);wb=torch.as_tensor(wt[ix],device=dev);opt.zero_grad(set_to_none=True);z=m(xb);loss=(torch.nn.functional.binary_cross_entropy_with_logits(z,yb,reduction='none')*wb).sum()/wb.sum();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step();losses.append(float(loss.detach().cpu()))
  sch.step();s=pred(m,va['X'],dev);mm=metrics(fast,va['user'],va['y'],s);hist.append({'epoch':e+1,'train_loss':float(np.mean(losses)),'primary':mm['primary']})
  if bm is None or mm['primary']>bm:bm=mm['primary'];best=s.copy();pat=0
  else:pat+=1
  if pat>=3:break
 mm=metrics(fast,va['user'],va['y'],best);json.dump({**mm,'history':hist},open(os.path.join(a.out_dir,'metrics.json'),'w'));f=open(os.path.join(a.out_dir,'predictions.csv'),'w');f.write('row_id,user_id,video_id,score\n');v=va['video_raw']
 for i,s in enumerate(best):f.write(f'{i},{va["user"][i]},{v[i]},{float(s):.8g}\n')
 f.close()
if __name__=='__main__':main()
