"""Oracle bounds: how much can a whole signal FAMILY add on top of the champion, measured with information the model
could never have (valid labels, leave-one-out or split-half; the random-exposure log; the leaky statistics file)?
A family whose oracle adds ~0 cannot be rescued by feature engineering. Additive z-blend on the champion, best weight."""
from _common import *
tr,va,vf,uf=load(); mu=tr.long_view.mean()
def stat(df,keys,m=10):
    g=df.groupby(keys).long_view.agg(['sum','count']); r=((g['sum']+m*mu)/(g['count']+m)).rename('r'); return va.join(r,on=keys)['r'].fillna(mu).values
def loo(keys,m=10):   # video/author-level only: a user-level LOO is a mechanical leak of the row's own label
    g=va.groupby(keys).long_view.agg(['sum','count']); j=va.join(g,on=keys); return ((j['sum']-va.long_view+m*mu)/(j['count']-1+m)).values
def rep(name,f): print('%-46s standalone %.4f  add %+.4f (w %+.2f)'%((name,score(va,f)[2])+tuple(add_test(va,f)[1][1::-1])))
print('champion primary %.4f'%score(va,va.champ.values)[2])
print('--- item side, valid-week labels (LOO) ---')
for k in (['video_id'],['video_id','tab'],['author_id']): rep('ORACLE '+'+'.join(k)+' valid-week LOO rate',loo(k))
st=pd.read_csv(ROOT+'/kuairand-starter-kit/KuaiRand-Pure/data/video_features_statistic_pure.csv'); j=va.join(st.set_index('video_id'),on='video_id')
rep('LEAKY whole-period long_time_play_cnt/show_cnt',(j.long_time_play_cnt/np.maximum(j.show_cnt,1)).fillna(0).values)
print('--- user state / taste, split-half: score half 0 with rates from half 1 of the same user (no self-leak) ---')
edges=tr.duration_ms.quantile(np.linspace(0,1,11)[1:-1]).values; tr['dur_b']=np.searchsorted(edges,tr.duration_ms.values); va['dur_b']=np.searchsorted(edges,va.duration_ms.values)
rng=np.random.default_rng(0); half=rng.integers(0,2,len(va)); e=va[half==0].copy(); src=va[half==1]
def sc(df,s): return evaluate(df.user_id.tolist(),df.long_view.tolist(),np.asarray(s,float).tolist())['primary']
b0=sc(e,e.champ.values)
for keys in (['user_id','date'],['user_id','tab'],['user_id','dur_b'],['user_id','tag'],['user_id','author_id'],['user_id','music_id'],['user_id','video_type']):
    g=src.groupby(keys).long_view.agg(['sum','count']); r=((g['sum']+mu)/(g['count']+1)).rename('r'); jj=e.join(r,on=keys); f=jj['r'].fillna(mu).values
    best=max(((w*s,sc(e,z(e.champ.values)+w*s*z(f))-b0) for w in (0.1,0.2,0.35,0.5,1.0) for s in (1,-1)),key=lambda t:t[1])
    print('  other-half %-22s coverage %.3f  add %+.4f (w %+.2f)'%('+'.join(keys),jj['r'].notna().mean(),best[1],best[0]))
print('--- random-exposure log, same week as valid (KuaiRand-Pure file, not in the workspace) ---')
rl=pd.read_csv(ROOT+'/kuairand-starter-kit/KuaiRand-Pure/data/log_random_4_22_to_5_08_pure.csv'); rv=rl[(rl.date>=20220422)&(rl.date<=20220428)]
rep('rand user-week rate',stat(rv,['user_id'],5)); rep('rand user-day rate',stat(rv,['user_id','date'],3)); rep('rand video-week rate',stat(rv,['video_id'],10))
print('--- train history the FM could already learn from ---')
for keys in (['user_id','tab'],['user_id','dur_b'],['user_id','tag'],['user_id','author_id'],['user_id','music_id'],['user_id','video_type']): rep('train '+'+'.join(keys)+' rate',stat(tr,keys,2))
print('--- label-free day-level context in valid ---')
v=va.sort_values(['user_id','time_ms','row_id']).copy(); g=v.groupby('user_id',sort=False)
v['n_day']=v.groupby(['user_id','date']).row_id.transform('size'); v['n_day_before']=v.groupby(['user_id','date']).cumcount(); v['day_idx']=g.date.rank(method='dense')
v['first_of_day']=(v.n_day_before==0).astype(float); v['n_same_hour']=v.groupby(['user_id','date','hourmin']).row_id.transform('size'); v=v.sort_values('row_id')
for c in ['n_day','n_day_before','day_idx','first_of_day','n_same_hour']: rep(c,v[c].values)
tr['wd']=pd.to_datetime(tr.date.astype(str)).dt.weekday; va['wd']=pd.to_datetime(va.date.astype(str)).dt.weekday; tr['hb']=tr.hourmin//600; va['hb']=va.hourmin//600
rep('train user x weekday rate',stat(tr,['user_id','wd'],3)); rep('train user x hour-bucket rate',stat(tr,['user_id','hb'],3))
