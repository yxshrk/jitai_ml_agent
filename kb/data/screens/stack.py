"""Upper bound: lambdarank GBDT stacked on the FM score + every feature family, 5-fold CV over users on valid."""
import sys, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0,'workspace'); from evaluate import evaluate
tr=pd.read_csv('workspace/data/train.csv'); va=pd.read_csv('workspace/data/valid.csv'); vf=pd.read_csv('workspace/data/video_features_basic.csv'); uf=pd.read_csv('workspace/data/user_features.csv')
tr=tr.merge(vf,on='video_id',how='left'); va=va.merge(vf,on='video_id',how='left').merge(uf,on='user_id',how='left')
tr['ptr']=np.clip(tr.play_time_ms/np.maximum(tr.duration_ms,1),0,3)
va['fm']=pd.read_csv('runs/live_07/outputs/003/predictions.csv').set_index('row_id').loc[va.row_id,'score'].values
M=20
def stat(keys,label):
    g=tr.groupby(keys)[label].agg(['sum','count']); mu=tr[label].mean(); r=((g['sum']+M*mu)/(g['count']+M)).rename('r'); return va.join(r,on=keys)['r'].fillna(mu).values
F={}
for keys,label in [(['video_id'],'long_view'),(['author_id'],'long_view'),(['music_id'],'long_view'),(['tag'],'long_view'),(['video_id'],'is_click'),(['video_id'],'ptr'),(['video_id'],'is_like'),
                   (['tab','video_id'],'long_view'),(['tab','author_id'],'long_view'),(['user_id','tab'],'long_view'),(['user_id','author_id'],'long_view'),(['user_id','tag'],'long_view')]:
    F['ts_'+'_'.join(keys)+'_'+label]=stat(keys,label)
F['user_rate']=stat(['user_id'],'long_view'); F['user_ptr']=stat(['user_id'],'ptr'); F['user_n']=np.log1p(va.join(tr.groupby('user_id').size().rename('c'),on='user_id')['c'].fillna(0).values)
F['video_n']=np.log1p(va.join(tr.groupby('video_id').size().rename('c'),on='video_id')['c'].fillna(0).values)
F['user_author_n']=va.join(tr.groupby(['user_id','author_id']).size().rename('c'),on=['user_id','author_id'])['c'].fillna(0).values
for c in ['tab','duration_ms','video_duration','hourmin','video_type','upload_type','music_type','server_width','server_height','is_rand']:
    F[c]=pd.to_numeric(va[c],errors='coerce').fillna(-1).values
F['upload_age']=(pd.to_datetime(va.date.astype(str))-pd.to_datetime(va.upload_dt.astype(str),errors='coerce')).dt.days.fillna(-1).values
for c in ['user_active_degree','is_lowactive_period','is_live_streamer','is_video_author','follow_user_num','fans_user_num','friend_user_num','register_days']+[f'onehot_feat{i}' for i in range(18)]:
    if c in va: F['u_'+c]=pd.to_numeric(va[c],errors='coerce').fillna(-1).values if va[c].dtype!=object else va[c].astype('category').cat.codes.values
# exposure context in the split
va=va.sort_values(['user_id','time_ms','row_id'],kind='stable'); F={k:v[va.index.values] for k,v in F.items()}; va=va.reset_index(drop=True)
g=va.groupby('user_id',sort=False); sn=g.user_id.shift(-1).eq(va.user_id); sp=g.user_id.shift(1).eq(va.user_id)
F['gap_next']=np.where(sn,np.minimum((g.time_ms.shift(-1)-va.time_ms)/1000,1e5),-1); F['gap_prev']=np.where(sp,np.minimum((va.time_ms-g.time_ms.shift(1))/1000,1e5),-1)
F['prev_same_author']=np.where(sp,g.author_id.shift(1).eq(va.author_id),0).astype(float); F['next_same_author']=np.where(sn,g.author_id.shift(-1).eq(va.author_id),0).astype(float)
F['pos_in_user']=g.cumcount().values; F['n_user_rows']=g.user_id.transform('size').values
F['dur_rel_user']=va.duration_ms.values/np.maximum(g.duration_ms.transform('mean').values,1)
F['fm']=va.fm.values; F['fm_rank_in_user']=g.fm.rank().values
names=list(F); X=np.column_stack([F[k] for k in names]); y=va.long_view.values; u=va.user_id.values
def score(s): m=evaluate(u.tolist(),y.tolist(),np.asarray(s,float).tolist()); return m['GAUC'],m['nDCG@5'],m['primary']
print('FM champion on valid: GAUC %.4f nDCG %.4f primary %.4f'%score(va.fm.values), '| features',len(names))
users=np.unique(u); rng=np.random.RandomState(0); fold_of_user=dict(zip(users,rng.randint(0,5,len(users)))); fold=np.array([fold_of_user[x] for x in u])
sizes=lambda m: pd.Series(u[m]).groupby(u[m],sort=False).size().values
params=dict(objective='lambdarank',learning_rate=0.03,num_leaves=31,min_data_in_leaf=100,feature_fraction=0.7,lambda_l2=10,num_threads=2,seed=0,deterministic=True,force_row_wise=True,verbose=-1,eval_at=[5])
oof=np.zeros(len(y)); imps=np.zeros(len(names)); its=[]
for f in range(5):
    trm=fold!=f; tem=fold==f
    inner=(fold==(f+1)%5); fitm=trm&~inner
    ds=lgb.Dataset(X[fitm],y[fitm],group=sizes(fitm),feature_name=names); dv=lgb.Dataset(X[inner],y[inner],group=sizes(inner),reference=ds)
    m=lgb.train(params,ds,num_boost_round=800,valid_sets=[dv],callbacks=[lgb.early_stopping(50,verbose=False)])
    oof[tem]=m.predict(X[tem],num_iteration=m.best_iteration); imps+=m.feature_importance('gain'); its.append(m.best_iteration)
print('stacked lambdarank OOF over users: GAUC %.4f nDCG %.4f primary %.4f  (best iters %s)'%(score(oof)+(its,)))
for w in (0.5,1,2): 
    zz=lambda f:(f-f.mean())/f.std(); print('   blend z(fm)+%.1f z(stack): primary %.4f'%(w,score(zz(va.fm.values)+w*zz(oof))[2]))
print('top gains:',[(n,int(g_)) for g_,n in sorted(zip(imps,names),reverse=True)[:12]])
# same, without the FM score (does the feature table alone reach the FM?)
keep=[i for i,n in enumerate(names) if not n.startswith('fm')]
oof2=np.zeros(len(y))
for f in range(5):
    trm=fold!=f; tem=fold==f; inner=(fold==(f+1)%5); fitm=trm&~inner
    ds=lgb.Dataset(X[fitm][:,keep],y[fitm],group=sizes(fitm)); dv=lgb.Dataset(X[inner][:,keep],y[inner],group=sizes(inner),reference=ds)
    m=lgb.train(params,ds,num_boost_round=800,valid_sets=[dv],callbacks=[lgb.early_stopping(50,verbose=False)]); oof2[tem]=m.predict(X[tem][:,keep],num_iteration=m.best_iteration)
print('feature table alone (no FM), OOF: GAUC %.4f nDCG %.4f primary %.4f'%score(oof2))
