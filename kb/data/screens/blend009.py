"""Heterogeneous-ensemble headroom above the seed-averaged champion: lambdarank GBDT on OOF target stats + session/context
features + tab/dur, rank-blended with node_009 (and with node_003 for comparison)."""
import sys, os, importlib.util, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0,'workspace'); from evaluate import evaluate
tr=pd.read_csv('workspace/data/train.csv'); va=pd.read_csv('workspace/data/valid.csv'); vf=pd.read_csv('workspace/data/video_features_basic.csv')
tr=tr.merge(vf,on='video_id',how='left'); va=va.merge(vf,on='video_id',how='left')
tr['ptr']=np.clip(tr.play_time_ms/np.maximum(tr.duration_ms,1),0,3)
edges=tr.duration_ms.quantile(np.linspace(0,1,11)[1:-1]).values
for d in (tr,va): d['dur_b']=np.searchsorted(edges,d.duration_ms.values)
# session features from node_002's own function
spec=importlib.util.spec_from_file_location('n2','runs/live_07/nodes/002.py'); n2=importlib.util.module_from_spec(spec); spec.loader.exec_module(n2)
trr=n2.read_rows('workspace/data/train.csv',['user_id','video_id','tab','duration_ms','long_view','time_ms']); var=n2.read_rows('workspace/data/valid.csv',['row_id','user_id','video_id','tab','duration_ms','long_view','time_ms'])
Str,st=n2.exposure_features(trr,0,5); Sva,_=n2.exposure_features(var,1,6,st)
rng=np.random.RandomState(0); fold=rng.randint(0,5,len(tr)); M=20
def stat(keys,label,fit,apply):
    g=fit.groupby(keys)[label].agg(['sum','count']); mu=fit[label].mean(); r=((g['sum']+M*mu)/(g['count']+M)).rename('r'); return apply.join(r,on=keys)['r'].fillna(mu).values
Ftr,Fva=[],[]
for keys,label in [(['video_id'],'long_view'),(['author_id'],'long_view'),(['tag'],'long_view'),(['video_id'],'ptr'),(['tab','video_id'],'long_view'),(['tab','author_id'],'long_view')]:
    col=np.empty(len(tr))
    for f in range(5): col[fold==f]=stat(keys,label,tr[fold!=f],tr[fold==f])
    Ftr.append(col); Fva.append(stat(keys,label,tr,va))
base=['tab','dur_b','duration_ms','video_duration','hourmin']
Xtr=np.column_stack(Ftr+[tr[c].fillna(0).values for c in base]+[Str]); Xva=np.column_stack(Fva+[va[c].fillna(0).values for c in base]+[Sva])
u=va.user_id.values; y=va.long_view.values
def score(s): m=evaluate(u.tolist(),y.tolist(),np.asarray(s,float).tolist()); return m['primary']
trs=tr.sort_values(['user_id','time_ms'],kind='stable'); idx=trs.index.values; hold=(trs.user_id.values%10==0)
def gsizes(mask): return trs[mask].groupby('user_id',sort=False).size().values
ytr=tr.long_view.values
params=dict(objective='lambdarank',learning_rate=0.05,num_leaves=63,min_data_in_leaf=200,feature_fraction=0.8,lambda_l2=10,num_threads=2,seed=0,deterministic=True,force_row_wise=True,verbose=-1,eval_at=[5],lambdarank_truncation_level=10)
ds=lgb.Dataset(Xtr[idx][~hold],ytr[idx][~hold],group=gsizes(~hold)); dv=lgb.Dataset(Xtr[idx][hold],ytr[idx][hold],group=gsizes(hold),reference=ds)
m=lgb.train(params,ds,num_boost_round=800,valid_sets=[dv],callbacks=[lgb.early_stopping(50,verbose=False)])
p=m.predict(Xva,num_iteration=m.best_iteration); print('GBDT lambdarank alone: %.4f (iters %d)'%(score(p),m.best_iteration))
def rank_in_user(s):
    r=np.zeros(len(s)); df=pd.DataFrame({'u':u,'s':s}); return df.groupby('u').s.rank(pct=True).values
def z(f): return (f-f.mean())/f.std()
for node in ('003','009'):
    c=pd.read_csv(f'runs/live_07/outputs/{node}/predictions.csv').set_index('row_id').loc[va.row_id,'score'].values
    p0=score(c); print(f'node_{node}: {p0:.4f}')
    for w in (0.25,0.5,0.75,1.0):
        print(f'   z-blend w={w}: {score(z(c)+w*z(p)):.4f} ({score(z(c)+w*z(p))-p0:+.4f})   rank-blend w={w}: {score(rank_in_user(c)+w*rank_in_user(p)):.4f}')
