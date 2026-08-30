"""Exposure-context features computed from the user's OTHER rows in the same split (no labels): backward (strictly
prior) and forward (later exposures). Within-user GAUC standalone and additive on champion node_003."""
import sys, numpy as np, pandas as pd
sys.path.insert(0,'workspace'); from evaluate import evaluate
va=pd.read_csv('workspace/data/valid.csv'); vf=pd.read_csv('workspace/data/video_features_basic.csv'); va=va.merge(vf[['video_id','author_id','tag','music_id']],on='video_id',how='left')
va=va.sort_values(['user_id','time_ms','row_id'],kind='stable').reset_index(drop=True)
g=va.groupby('user_id',sort=False)
same_user_next=g.user_id.shift(-1).eq(va.user_id); same_user_prev=g.user_id.shift(1).eq(va.user_id)
va['gap_next']=np.where(same_user_next,(g.time_ms.shift(-1)-va.time_ms)/1000.0,np.nan)          # seconds until the next exposure
va['gap_prev']=np.where(same_user_prev,(va.time_ms-g.time_ms.shift(1))/1000.0,np.nan)
va['next_same_author']=np.where(same_user_next,g.author_id.shift(-1).eq(va.author_id),False).astype(float)
va['prev_same_author']=np.where(same_user_prev,g.author_id.shift(1).eq(va.author_id),False).astype(float)
va['next_same_video']=np.where(same_user_next,g.video_id.shift(-1).eq(va.video_id),False).astype(float)
va['next_same_tag']=np.where(same_user_next,g.tag.shift(-1).eq(va.tag),False).astype(float)
va['gap_next_over_dur']=va.gap_next*1000/np.maximum(va.duration_ms,1)                             # >=1 means the user could have watched it all
va['gap_next_capped']=np.minimum(va.gap_next.fillna(1e6),600)
va['n_after_10m']=[0]*len(va)
t=va.time_ms.values; u=va.user_id.values; n_after=np.zeros(len(va)); n_before=np.zeros(len(va))
i=0
while i<len(va):
    j=i
    while j<len(va) and u[j]==u[i]: j+=1
    tt=t[i:j]; n_after[i:j]=np.searchsorted(tt,tt+600_000,'right')-np.arange(j-i)-1; n_before[i:j]=np.arange(j-i)-np.searchsorted(tt,tt-600_000,'left'); i=j
va['n_after_10m']=n_after; va['n_before_10m']=n_before
va['is_last_of_user']=(~same_user_next).astype(float); va['is_first_of_user']=(~same_user_prev).astype(float)
uu=va.user_id.values; y=va.long_view.values
def score(s): m=evaluate(uu.tolist(),y.tolist(),np.asarray(s,float).tolist()); return m['GAUC'],m['nDCG@5'],m['primary']
champ=pd.read_csv('runs/live_07/outputs/003/predictions.csv').set_index('row_id').loc[va.row_id,'score'].values
c0=score(champ); print('champion: GAUC %.4f nDCG %.4f primary %.4f'%c0)
def z(f): f=np.asarray(f,float); f=np.where(np.isnan(f),np.nanmedian(f),f); return (f-f.mean())/(f.std()+1e-9)
order=np.argsort(uu,kind='stable'); starts=np.r_[0,np.flatnonzero(np.diff(uu[order]))+1,len(uu)]; groups=[order[a:b] for a,b in zip(starts[:-1],starts[1:])]
print(f'{"feature":22s} {"varies":>6s} {"GAUC":>7s} sign  Δprimary on champion (best w)')
for nm in ['gap_prev','prev_same_author','n_before_10m','is_first_of_user','gap_next','gap_next_capped','gap_next_over_dur','next_same_author','next_same_video','next_same_tag','n_after_10m','is_last_of_user']:
    f=va[nm].values.astype(float); fz=z(f); varies=np.mean([len(np.unique(np.round(fz[g_],6)))>1 for g_ in groups])
    gp,gm=score(fz)[0],score(-fz)[0]; sign=1 if gp>=gm else -1
    best=max(((w,score(z(champ)+w*sign*fz)[2]-c0[2]) for w in (0.1,0.25,0.5,1,2)),key=lambda t:t[1])
    print(f'{nm:22s} {varies:6.3f} {max(gp,gm):7.4f} {sign:+d}   {best[1]:+.4f} (w={best[0]})')
# label rate by gap_next bucket (the mechanism)
b=pd.cut(va.gap_next,[-1,5,15,30,60,120,300,900,3600,1e9]); print('\nlong_view rate by seconds until next exposure:'); print(va.groupby(b,observed=True).long_view.agg(['mean','size']).round(3).to_string())
print('\nlong_view rate by gap_next/duration:'); print(va.groupby(pd.cut(va.gap_next_over_dur,[-1,0.25,0.5,0.8,1.0,1.5,3,1e9]),observed=True).long_view.agg(['mean','size']).round(3).to_string())
