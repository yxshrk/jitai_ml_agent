import sys, numpy as np, pandas as pd
import os; ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..','..'))
sys.path.insert(0, ROOT+'/workspace')
from evaluate import evaluate
def load(champ='009'):
    tr=pd.read_csv(f'{ROOT}/workspace/data/train.csv'); va=pd.read_csv(f'{ROOT}/workspace/data/valid.csv')
    vf=pd.read_csv(f'{ROOT}/workspace/data/video_features_basic.csv'); uf=pd.read_csv(f'{ROOT}/workspace/data/user_features.csv')
    tr=tr.merge(vf[['video_id','author_id','tag','music_id','video_type','upload_dt']],on='video_id',how='left')
    va=va.merge(vf[['video_id','author_id','tag','music_id','video_type','upload_dt']],on='video_id',how='left')
    p=pd.read_csv(f'{ROOT}/runs/live_07/outputs/{champ}/predictions.csv')
    va['champ']=p.set_index('row_id').loc[va.row_id,'score'].values
    return tr,va,vf,uf
def score(va,s):
    m=evaluate(va.user_id.tolist(),va.long_view.tolist(),np.asarray(s,float).tolist()); return m['GAUC'],m['nDCG@5'],m['primary']
def z(f): f=np.asarray(f,float); return (f-f.mean())/(f.std()+1e-9)
def add_test(va,f,ws=(0.1,0.2,0.35,0.5,1.0),base=None):
    """best additive gain of z(f) on top of the champion (or base) over a small weight grid, both signs"""
    b=va.champ.values if base is None else base; b0=score(va,b)[2]; best=(0,0,b0)
    for w in ws:
        for sg in (1,-1):
            p=score(va,z(b)+sg*w*z(f))[2]
            if p>best[2]: best=(sg*w,p-b0,p)
    return b0,best
