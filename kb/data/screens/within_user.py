import sys, importlib.util, numpy as np
sys.path.insert(0,'workspace'); sys.path.insert(0,'kuairand-starter-kit')
spec=importlib.util.spec_from_file_location('n2','runs/live_07/nodes/002.py'); n2=importlib.util.module_from_spec(spec); spec.loader.exec_module(n2)
from evaluate import evaluate
tr=n2.read_rows('workspace/data/train.csv',['user_id','video_id','tab','duration_ms','long_view','time_ms'])
va=n2.read_rows('workspace/data/valid.csv',['row_id','user_id','video_id','tab','duration_ms','long_view','time_ms'])
Str,st=n2.exposure_features(tr,0,5); Sva,_=n2.exposure_features(va,1,6,st)
u=np.array([int(x[1]) for x in va]); y=np.array([int(x[5]!='0') for x in va]); tab=np.array([int(x[3]) for x in va]); dur=np.array([float(x[4]) for x in va]); t=np.array([int(x[6]) for x in va])
names=['session_pos','recent_10m','previous_gap']
order=np.argsort(u,kind='stable'); us=u[order]; starts=np.r_[0,np.flatnonzero(np.diff(us))+1,len(us)]
groups=[order[a:b] for a,b in zip(starts[:-1],starts[1:])]
def within_user_var(f):
    return np.mean([len(np.unique(f[g]))>1 for g in groups])
print('n valid',len(va),'users',len(groups))
for i,nm in enumerate(names):
    f=Sva[:,i].astype(int)
    print(f'{nm:14s} within-user-varies {within_user_var(f):.3f}  GAUC(-f) {evaluate(u.tolist(),y.tolist(),(-f).tolist())["GAUC"]:.4f} GAUC(+f) {evaluate(u.tolist(),y.tolist(),f.tolist())["GAUC"]:.4f}  marginal', [round(float(y[f==v].mean()),3) for v in sorted(set(f.tolist()))])
print('tab            within-user-varies %.3f GAUC(-tab) %.4f'%(within_user_var(tab),evaluate(u.tolist(),y.tolist(),(-tab).tolist())['GAUC']))
print('dur            GAUC(dur) %.4f'%evaluate(u.tolist(),y.tolist(),dur.tolist())['GAUC'])
spread=[(t[g].max()-t[g].min())/60000 for g in groups]
print('valid rows per user span (minutes): median %.1f p25 %.1f p75 %.1f p90 %.1f'%tuple(np.percentile(spread,[50,25,75,90])))
