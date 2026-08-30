"""Screen candidate signals by what the metric rewards: within-user discrimination on valid.
For each feature f: fraction of users whose valid rows differ on f, standalone GAUC (best sign), and the gain when
added on top of a tab x duration-bucket prior (train rates) -- 'adds beyond tab/dur'. Train labels only; valid labels
are used only to score."""
import sys, numpy as np, pandas as pd
sys.path.insert(0,'workspace')
from evaluate import evaluate
tr=pd.read_csv('workspace/data/train.csv'); va=pd.read_csv('workspace/data/valid.csv')
vf=pd.read_csv('workspace/data/video_features_basic.csv'); uf=pd.read_csv('workspace/data/user_features.csv')
print('train dates',tr.date.min(),tr.date.max(),' valid dates',va.date.min(),va.date.max())
tr=tr.merge(vf,on='video_id',how='left'); va=va.merge(vf,on='video_id',how='left')
tr['dur_b']=pd.qcut(tr.duration_ms,10,labels=False,duplicates='drop'); edges=tr.duration_ms.quantile(np.linspace(0,1,11)[1:-1]).values
va['dur_b']=np.searchsorted(edges,va.duration_ms.values)
prior=tr.long_view.mean()
def rate(keys,label='long_view',m=20,name=None):
    g=tr.groupby(keys)[label].agg(['sum','count']); r=(g['sum']+m*tr[label].mean())/(g['count']+m)
    r.name=name or ('r_'+'_'.join(keys)+'_'+label); return va.join(r,on=keys)[r.name].fillna(tr[label].mean()).values
tr['ptr']=np.clip(tr.play_time_ms/np.maximum(tr.duration_ms,1),0,3)
u=va.user_id.values; y=va.long_view.values
order=np.argsort(u,kind='stable'); us=u[order]; starts=np.r_[0,np.flatnonzero(np.diff(us))+1,len(us)]; groups=[order[a:b] for a,b in zip(starts[:-1],starts[1:])]
def gauc(s): return evaluate(u.tolist(),y.tolist(),np.asarray(s,dtype=float).tolist())['GAUC']
base=rate(['tab','dur_b'],name='base'); g0=gauc(base)
print(f'baseline tab x dur_b train-rate: GAUC {g0:.4f}')
def z(f): f=np.asarray(f,float); return (f-f.mean())/(f.std()+1e-9)
feats={
 'video_rate_lv': rate(['video_id']),
 'video_rate_click': rate(['video_id'],'is_click'),
 'video_rate_like': rate(['video_id'],'is_like'),
 'video_rate_hate': rate(['video_id'],'is_hate'),
 'video_mean_playthrough': rate(['video_id'],'ptr'),
 'video_exposures_train': np.log1p(va.join(tr.groupby('video_id').size().rename('c'),on='video_id')['c'].fillna(0).values),
 'author_rate_lv': rate(['author_id']),
 'tag_rate_lv': rate(['tag']),
 'music_rate_lv': rate(['music_id']),
 'upload_type_rate': rate(['upload_type']),
 'video_type_rate': rate(['video_type']),
 'video_duration_side': va.video_duration.fillna(0).values,
 'duration_ms': va.duration_ms.values,
 'dur_minus_video_duration': (va.duration_ms-va.video_duration.fillna(va.duration_ms)).values,
 'upload_age_days': (pd.to_datetime(va.date.astype(str))-pd.to_datetime(va.upload_dt.astype(str),errors='coerce')).dt.days.fillna(0).values,
 'server_aspect': (va.server_height/np.maximum(va.server_width,1)).fillna(1).values,
 'hourmin': va.hourmin.values,
 'is_rand': va.is_rand.values,
 'user_author_rate_lv': rate(['user_id','author_id'],m=5),
 'user_tag_rate_lv': rate(['user_id','tag'],m=5),
 'user_tab_rate_lv': rate(['user_id','tab'],m=5),
 'user_durb_rate_lv': rate(['user_id','dur_b'],m=5),
 'user_video_seen_before': va.join(tr.groupby(['user_id','video_id']).size().rename('c'),on=['user_id','video_id'])['c'].fillna(0).values,
 'user_author_exposures': va.join(tr.groupby(['user_id','author_id']).size().rename('c'),on=['user_id','author_id'])['c'].fillna(0).values,
 'tab_x_video_rate': rate(['tab','video_id'],m=10),
 'tab_x_author_rate': rate(['tab','author_id'],m=10),
 'video_rate_lv_recent7d': None,
}
d=tr.date.max(); rec=tr[tr.date>d-7]; g=rec.groupby('video_id').long_view.agg(['sum','count']); r=((g['sum']+20*prior)/(g['count']+20)).rename('rr')
feats['video_rate_lv_recent7d']=va.join(r,on='video_id')['rr'].fillna(prior).values
rows=[]
for k,f in feats.items():
    f=np.asarray(f,float); varies=np.mean([len(np.unique(f[g]))>1 for g in groups])
    gp,gm=gauc(f),gauc(-f); sign=1 if gp>=gm else -1
    add=gauc(z(base)+0.5*sign*z(f))-g0
    rows.append((k,varies,max(gp,gm),sign,add))
rows.sort(key=lambda r:-r[4])
print(f'{"feature":28s} {"varies":>7s} {"GAUC":>7s} sign {"+beyond tab/dur":>16s}')
for k,v,g_,s,a in rows: print(f'{k:28s} {v:7.3f} {g_:7.4f} {s:+d} {a:+16.4f}')
