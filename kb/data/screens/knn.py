"""Is there collaborative taste the FM misses? Item-item kNN over train positives, scored on valid, additive on champion."""
from _common import *
import scipy.sparse as sp
tr,va,vf,uf=load(); mu=tr.long_view.mean()
vids=np.unique(np.r_[tr.video_id.values,va.video_id.values]); vid_ix={v:i for i,v in enumerate(vids)}; V=len(vids)
users=np.unique(np.r_[tr.user_id.values,va.user_id.values]); u_ix={u:i for i,u in enumerate(users)}; U=len(users)
ui=tr.user_id.map(u_ix).values; vi=tr.video_id.map(vid_ix).values
def build(w):
    M=sp.csr_matrix((w.astype(np.float32),(ui,vi)),shape=(U,V)); M.sum_duplicates(); return M
def knn_scores(M, shrink=10.0, center=False):
    # cosine item-item similarity from user x item matrix M (users as the sample axis)
    n=np.sqrt(np.asarray(M.multiply(M).sum(0)).ravel())+1e-6
    S=(M.T@M).toarray(); S=S/(n[:,None]*n[None,:]); np.fill_diagonal(S,0)
    S=S*(np.asarray((M.T@M).todense())/(np.asarray((M.T@M).todense())+shrink))  # shrink by co-occurrence count
    prof=M@S            # user x item: sum over history of sim; sparse@dense -> dense (U x V)
    cnt=np.asarray(M.sum(1)).ravel()
    return prof, cnt
va_u=va.user_id.map(u_ix).values; va_v=va.video_id.map(vid_ix).values
pos=(tr.long_view.values==1).astype(float)
res={}
for name,w in (('positives',pos),('negatives',(1-pos)),('watch fraction',np.clip(tr.play_time_ms.values/np.maximum(tr.duration_ms.values,1),0,1)),('centered label',pos-mu)):
    M=build(w); prof,cnt=knn_scores(M); f=prof[va_u,va_v]/np.maximum(cnt[va_u],1)
    st=score(va,f)[2]; print('  kNN(%s): standalone %.4f add %s'%(name,st,add_test(va,f)[1])); res[name]=f
f=res['positives']-res['negatives']*0.5; print('  kNN(pos - 0.5 neg): standalone %.4f add %s'%(score(va,f)[2],add_test(va,f)[1]))
# also: co-visitation regardless of label (what videos co-occur in the same feed) as a 'they were shown together' proxy
M=build(np.ones(len(tr))); prof,cnt=knn_scores(M); f=prof[va_u,va_v]/np.maximum(cnt[va_u],1); print('  kNN(all exposures): standalone %.4f add %s'%(score(va,f)[2],add_test(va,f)[1]))
