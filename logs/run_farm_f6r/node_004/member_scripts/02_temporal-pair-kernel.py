import argparse,csv,json,os,random,datetime
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
 tr,va=rd(os.path.join(d,'train.csv')),rd(os.path.join(d,'val.csv'));cuts=np.quantile(tr['duration_ms'],np.arange(1,10)/10);rt=[tr['user_id'],tr['video_id'],tr['video_id'],tr['tab'],np.searchsorted(cuts,tr['duration_ms']).astype(str)];rv=[va['user_id'],va['video_id'],va['video_id'],va['tab'],np.searchsorted(cuts,va['duration_ms']).astype(str)];xs=[];xv=[];dims=[];off=0
 for c,dv in zip(rt,rv):
  m={};u=[]
  for v in c:m.setdefault(v,len(m));u.append(m[v])
  w=[m.get(v,len(m)) for v in dv];dim=len(m)+1;xs.append(np.asarray(u)+off);xv.append(np.asarray(w)+off);dims.append(dim);off+=dim
 return {'X':np.stack(xs,1).astype(np.int32),'y':tr['long_view'],'user':np.asarray(tr['user_id']),'date':tr['date'],'hourmin':tr['hourmin'],'field_dims':np.asarray(dims)}, {'X':np.stack(xv,1).astype(np.int32),'y':va['long_view'],'user':np.asarray(va['user_id']),'date':va['date'],'hourmin':va['hourmin'],'video_raw':np.asarray(va['video_id'])},False

def aug(z,base):
 hm=np.asarray(z['hourmin'],dtype=np.int64);hour=np.where(hm>1439,np.clip(hm//100,0,23),np.where(hm>23,np.clip(hm//60,0,23),np.clip(hm,0,23)));day=np.asarray(z['date'],dtype=np.int64)%7;return np.concatenate([z['X'],(base+hour)[:,None],(base+24+day)[:,None]],1).astype(np.int32)

def metric(fast,u,y,s):
 if fast:
  from data.official.evaluate import evaluate
 else:
  from harness.evaluate_provisional import evaluate
 m=evaluate(u,y.astype(int),s);return {'gauc':float(m.get('GAUC',m.get('gauc'))),'ndcg5':float(m.get('nDCG@5',m.get('ndcg5'))),'primary':float(m['primary'])}

class Kernel(torch.nn.Module):
 def __init__(self,n):
  super().__init__();self.e=torch.nn.Embedding(n,16);self.l=torch.nn.Embedding(n,1);torch.nn.init.normal_(self.e.weight,std=.01);torch.nn.init.zeros_(self.l.weight);self.g=torch.nn.Sequential(torch.nn.Linear(32,32),torch.nn.Tanh(),torch.nn.Dropout(.2),torch.nn.Linear(32,1));self.bias=torch.nn.Parameter(torch.zeros(1))
 def forward(self,x):
  e=self.e(x);s=e.sum(1);fm=.5*(s.square()-e.square().sum(1)).sum(1);tem=e[:,-2:].flatten(1);item=e[:,1]*e[:,-2]+e[:,2]*e[:,-1];return self.bias+self.l(x).sum((1,2))+fm+self.g(tem).squeeze(1)+item.sum(1)

def pred(m,x,d):
 m.eval();r=[]
 with torch.no_grad():
  for i in range(0,len(x),32768):r.append(m(torch.as_tensor(x[i:i+32768],dtype=torch.long,device=d)).cpu().numpy())
 return np.concatenate(r)

def pairs(users,y,rng):
 pos={};neg={}
 for i,(u,t) in enumerate(zip(users,y)):(pos if t>.5 else neg).setdefault(str(u),[]).append(i)
 pp=[];nn=[]
 for u,a in pos.items():
  if u in neg:
   b=neg[u]
   for i in a:pp.append(i);nn.append(b[rng.randint(len(b))])
 return np.asarray(pp),np.asarray(nn)

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--out-dir',required=True);p.add_argument('--seed',type=int,default=42);p.add_argument('--epochs',type=int,default=12);a=p.parse_args();os.makedirs(a.out_dir,exist_ok=True);seed_all(a.seed);dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');tr,va,fast=load(a.data_dir);base=int(np.sum(tr['field_dims']));x=aug(tr,base);xv=aug(va,base);y=tr['y'].astype(np.float32);ep=a.epochs;sm=os.environ.get('SMOKE_EPOCHS');ep=min(ep,max(1,int(sm))) if sm else ep;m=Kernel(base+31).to(dev);opt=torch.optim.AdamW(m.parameters(),lr=7e-4,weight_decay=4e-5);sch=torch.optim.lr_scheduler.MultiStepLR(opt,[max(1,ep//3),max(2,2*ep//3)],gamma=.35);rng=np.random.RandomState(a.seed);best=None;bm=-1;hist=[];pat=0
 for e in range(ep):
  pp,nn=pairs(tr['user'],y,rng);order=rng.permutation(len(pp));m.train();ls=[]
  for j in range(0,len(order),4096):
   k=order[j:j+4096];pi=pp[k];ni=nn[k];xp=torch.as_tensor(x[pi],dtype=torch.long,device=dev);xn=torch.as_tensor(x[ni],dtype=torch.long,device=dev);opt.zero_grad(set_to_none=True);sp=m(xp);sn=m(xn);rank=torch.nn.functional.softplus(-(sp-sn)).mean();point=.5*(torch.nn.functional.softplus(-sp).mean()+torch.nn.functional.softplus(sn).mean());loss=.5*rank+.5*point;loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step();ls.append(float(loss.detach().cpu()))
  sch.step();s=pred(m,xv,dev);mm=metric(fast,va['user'],va['y'],s);hist.append({'epoch':e+1,'train_loss':float(np.mean(ls)) if ls else 0.0,'primary':mm['primary']})
  if mm['primary']>bm:bm=mm['primary'];best=s.copy();pat=0
  else:pat+=1
  if pat>=3:break
 mm=metric(fast,va['user'],va['y'],best);json.dump({**mm,'history':hist},open(os.path.join(a.out_dir,'metrics.json'),'w'));f=open(os.path.join(a.out_dir,'predictions.csv'),'w');f.write('row_id,user_id,video_id,score\n')
 for i,s in enumerate(best):f.write(f'{i},{va["user"][i]},{va["video_raw"][i]},{float(s):.8g}\n')
 f.close()
if __name__=='__main__':main()
