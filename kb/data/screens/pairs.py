"""Pairwise error attribution on the champion: which (pos,neg) pairs within a user does it misorder?"""
from _common import *
tr,va,vf,uf=load()
print('champion valid: GAUC %.4f nDCG %.4f primary %.4f'%score(va,va.champ.values))
v=va.copy()
g=v.groupby('user_id').long_view.agg(['sum','count']); disc=g[(g['sum']>0)&(g['sum']<g['count'])].index
d=v[v.user_id.isin(disc)]
pos=d[d.long_view==1][['user_id','row_id','video_id','author_id','tab','duration_ms','date','time_ms','champ','tag']]
neg=d[d.long_view==0][['user_id','row_id','video_id','author_id','tab','duration_ms','date','time_ms','champ','tag']]
P=pos.merge(neg,on='user_id',suffixes=('_p','_n'))
npos=d.groupby('user_id').long_view.sum(); nneg=d.groupby('user_id').long_view.apply(lambda s:(s==0).sum())
P['w']=1.0/nneg.loc[P.user_id].values   # each user's pairs weighted so the user counts npos (GAUC weight)
P['err']=np.where(P.champ_p>P.champ_n,0.0,np.where(P.champ_p==P.champ_n,0.5,1.0))
print('pairs %d, weighted error (=1-GAUC) %.4f'%(len(P),(P.w*P.err).sum()/P.w.sum()))
def cut(name,mask):
    m=mask.values; wl=P.w[m].sum()/P.w.sum(); e=(P.w[m]*P.err[m]).sum()/max(P.w[m].sum(),1e-9)
    print(f'  {name:42s} share {wl:6.3f}  err {e:.3f}  contrib {wl*e:.4f}')
print('by pair type:')
cut('same tab',P.tab_p==P.tab_n); cut('diff tab',P.tab_p!=P.tab_n)
for t in (0,1,2,4,6): cut(f'  both tab {t}',(P.tab_p==t)&(P.tab_n==t))
cut('  pos tab1, neg tab0',(P.tab_p==1)&(P.tab_n==0)); cut('  pos tab0, neg tab1',(P.tab_p==0)&(P.tab_n==1))
cut('same date',P.date_p==P.date_n); cut('diff date',P.date_p!=P.date_n)
cut('  pos earlier day',P.date_p<P.date_n); cut('  pos later day',P.date_p>P.date_n)
cut('same date, pos earlier (time)',(P.date_p==P.date_n)&(P.time_ms_p<P.time_ms_n)); cut('same date, pos later',(P.date_p==P.date_n)&(P.time_ms_p>P.time_ms_n))
cut('same video',P.video_id_p==P.video_id_n); cut('same author',(P.author_id_p==P.author_id_n)&(P.video_id_p!=P.video_id_n))
cut('neg dur=0',P.duration_ms_n==0); cut('pos shorter',(P.duration_ms_p<P.duration_ms_n)&(P.duration_ms_n>0)); cut('pos longer',P.duration_ms_p>P.duration_ms_n)
cut('both <18s',(P.duration_ms_p<18000)&(P.duration_ms_n<18000)&(P.duration_ms_n>0)); cut('both >=18s',(P.duration_ms_p>=18000)&(P.duration_ms_n>=18000))
cut('|gap|<10min',(P.time_ms_p-P.time_ms_n).abs()<600e3); cut('|gap|>1day',(P.time_ms_p-P.time_ms_n).abs()>86400e3)
cut('same tag',P.tag_p==P.tag_n)
# time-of-day / session-level: does the model already know date-level drift within user?
print('within-user time signals on all valid users (standalone primary, additive on champion):')
for name,f in [('-time_ms',-va.time_ms.values),('-date',-va.date.values.astype(float)),('hourmin',va.hourmin.values.astype(float)),('champ',va.champ.values)]:
    print('  %-12s standalone %.4f  add %s'%(name,score(va,f)[2],add_test(va,f)[1]))
