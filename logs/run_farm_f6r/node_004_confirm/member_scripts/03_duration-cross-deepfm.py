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
  for k in ['date','hourmin']:z[k]=np.asarray(z[k],dtype=np.int64)
  for k in ['duration_ms','long_view']:z[k]=np.asarray(z[k],dtype=np.float32)
  return z
 tr,va=rd(os.path.join(d,'train.csv')),rd(os.path.join(d,'val.csv'));cuts=np.quantile(tr['duration_ms'],np.arange(1,10)/10);rt=[tr['user_id'],tr['video_id'],tr['video_id'],tr['tab'],np.searchsorted(cuts,tr['duration_ms']).astype(str)];rv=[va['user_id'],va['video_id'],va['video_id'],va['tab'],np.searchsorted(cuts,va['duration_ms']).astype(str)];xs=[];xv=[];dims=[];off=0
 for c,dv in zip(rt,rv):
  m={};u=[]
  for v in c:m.setdefault(v,len(m));u.append(m[v])
  w=[m.get(v,len(m)) for v in dv];dim=len(m)+1;xs.append(np.asarray(u)+off);xv.append(np.asarray(w)+off);dims.append(dim);off+=dim
 return {'X':np.stack(xs,1).astype(np.int32),'y':tr['long_view'],'user':np.asarray(tr['user_id']),'duration_ms':tr['duration_ms'],'date':tr['date'],'field_dims':np.asarray(dims)}, {'X':np.stack(xv,1).astype(np.int32),'y':va['long_view'],'user':np.asarray(va['user_id']),'duration_ms':va['duration_ms'],'video_raw':np.asarray(va['video_id'])},False

def metric(fast,u,y,s):
 if fast:
  from data.official.evaluate import evaluate
 else:
  from harness.evaluate_provisional import evaluate
 m=evaluate(u,y.astype(int),s);return {'gauc':float(m.get('GAUC',m.get('gauc'))),'ndcg5':float(m.get('nDCG@5',m.get('ndcg5'))),'primary':float(m['primary'])}

def enhance(tr,va):
 base=int(np.sum(tr['field_dims']));cuts=np.quantile(tr['duration_ms'],np.arange(1,50)/50);dt=np.searchsorted(cuts,tr['duration_ms']);dv=np.searchsorted(cuts,va['duration_ms']);st=(tr['duration_ms']<=18000).astype(np.int64);sv=(va['duration_ms']<=18000).astype(np.int64);tabt=tr['X'][:,3]-int(np.sum(tr['field_dims'][:3]));tabv=va['X'][:,3]-int(np.sum(tr['field_dims'][:3]));nt=int(tr['field_dims'][3]);ct=np.clip(tabt,0,nt-1)*50+dt;cv=np.clip(tabv,0,nt-1)*50+dv;xt=np.concatenate([tr['X'],(base+dt)[:,None],(base+50+st)[:,None],(base+52+ct)[:,None]],1);xv=np.concatenate([va['X'],(base+dv)[:,None],(base+50+sv)[:,None],(base+52+cv)[:,None]],1);return xt.astype(np.int32),xv.astype(np.int32),base+52+nt*50

class DeepFM(torch.nn.Module):
 def __init__(self,n):
  super().__init__();self.e=torch.nn.Embedding(n,16);self.l=torch.nn.Embedding(n,1);torch.nn.init.normal_(self.e.weight,std=.01);torch.nn.init.zeros_(self.l.weight);self.m=torch.nn.Sequential(torch.nn.Linear(128,128),torch.nn.ReLU(),torch.nn.Dropout(.3),torch.nn.Linear(128,64),torch.nn.ReLU(),torch.nn.Dropout(.25),torch.nn.Linear(64,1));self.b=torch.nn.Parameter(torch.zeros(1))
 def forward(self,x):
  e=self.e(x);s=e.sum(1);fm=.5*(s.square()-e.square().sum(1)).sum(1);return self.b+self.l(x).sum((1,2))+fm+self.m(e.flatten(1)).squeeze(1)

def pred(m,x,d):
 m.eval();r=[]
 with torch.no_grad():
  for i in range(0,len(x),32768):r.append(m(torch.as_tensor(x[i:i+32768],dtype=torch.long,device=d)).cpu().numpy())
 return np.concatenate(r)

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--out-dir',required=True);p.add_argument('--seed',type=int,default=42);p.add_argument('--epochs',type=int,default=14);a=p.parse_args();os.makedirs(a.out_dir,exist_ok=True);seed_all(a.seed);dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');tr,va,fast=load(a.data_dir);x,xv,n=enhance(tr,va);y=tr['y'].astype(np.float32);ep=a.epochs;sm=os.environ.get('SMOKE_EPOCHS');ep=min(ep,max(1,int(sm))) if sm else ep;m=DeepFM(n).to(dev);opt=torch.optim.AdamW(m.parameters(),lr=6e-4,weight_decay=7e-5);sch=torch.optim.lr_scheduler.MultiStepLR(opt,[max(1,ep//3),max(2,2*ep//3)],gamma=.3);rng=np.random.RandomState(a.seed);best=None;bm=-1;hist=[];pat=0
 for e in range(ep):
  m.train();perm=rng.permutation(len(y));ls=[]
  for j in range(0,len(y),4096):
   ix=perm[j:j+4096];xb=torch.as_tensor(x[ix],dtype=torch.long,device=dev);yb=torch.as_tensor(y[ix],device=dev);opt.zero_grad(set_to_none=True);loss=torch.nn.functional.binary_cross_entropy_with_logits(m(xb),yb,label_smoothing=.02);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step();ls.append(float(loss.detach().cpu()))
  sch.step();s=pred(m,xv,dev);mm=metric(fast,va['user'],va['y'],s);hist.append({'epoch':e+1,'train_loss':float(np.mean(ls)),'primary':mm['primary']})
  if mm['primary']>bm:bm=mm['primary'];best=s.copy();pat=0
  else:pat+=1
  if pat>=3:break
 mm=metric(fast,va['user'],va['y'],best);json.dump({**mm,'history':hist},open(os.path.join(a.out_dir,'metrics.json'),'w'));f=open(os.path.join(a.out_dir,'predictions.csv'),'w');f.write('row_id,user_id,video_id,score\n')
 for i,s in enumerate(best):f.write(f'{i},{va["user"][i]},{va["video_raw"][i]},{float(s):.8g}\n')
 f.close()
if __name__=='__main__':main()
