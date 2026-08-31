"""Decision replay: the selector chooses at each evolving state; each chosen
recipe is executed INSTANTLY from cached tuned artifacts (tonight's runs).
Measures end-to-end judgment quality without training or code generation."""
import sys, csv
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from agent.brain import Brain
from data.official.evaluate import evaluate

val = np.load(ROOT/'data/real_ws/val.npz'); Y=val['y'].astype(int); U=val['user']
def load(p):
    s=np.empty(len(Y))
    with open(p) as fh:
        r=csv.reader(fh); next(r)
        for row in r: s[int(row[0])]=float(row[3])
    return s
# cached tuned artifact per method (agent-authored, from tonight's runs)
ART = {
 'seq-deepfm-composite':      'logs/probe_composite_close/seed_42/predictions.csv',
 'seq-deepfm-author-history': 'logs/probe_composite_close/seed_43/predictions.csv',
 'package-dial-sweep':        'logs/run_bigclock_07/node_003/predictions.csv',
 'stage-matrix-sweep':        'logs/run_qb_b/node_001/predictions.csv',
 'temporal-pair-kernel':      'logs/run_novel_l1/node_004/predictions.csv',
 'context-stratified-pairs':  'logs/run_final_s4/node_003/predictions.csv',
 'gauge-fixed-bce':           'logs/run_final_s2/node_001/predictions.csv',
 'recency-weighting':         'logs/run_night_e/node_004/predictions.csv',
}
CLOSES = {'ensemble-design-sweep','heterogeneous-ensemble-design','diverse-family-farm-close'}
CURVE=[{"epoch":e,"train_loss":0.55-0.01*e,"val_gauc":0.665+0.002*min(e,4)-0.001*max(0,e-4),
        "val_primary":0.600+0.001*min(e,4)-0.0006*max(0,e-4)} for e in range(1,11)]

brain=Brain((ROOT/'MENU.md').read_text(),provider='openai',knowledge_mode='full')
journal=['node_000 [baseline] draft "baseline FM" primary=0.6018 ACCEPTED (sigma=0.0001)']
members={}; champ=0.6018; streak=0
for it in range(1,7):
    sel=brain.select_method(journal,CURVE,{"no_improve_streak":streak,"iterations_done":it,"max_iters":16},
                            excluded_families=[],dataset='pure',prior_runs=None)
    pick=sel.get('chosen_method_id'); print(f"iter {it}: selector -> {pick} (diag {sel.get('diagnosis')})")
    if pick in CLOSES:
        pool=members if len(members)>=2 else {**members,'seq-deepfm-composite':load(ART['seq-deepfm-composite'])}
        ranks=[np.argsort(np.argsort(s,kind='stable'),kind='stable') for s in pool.values()]
        ens=np.mean(ranks,axis=0); p=evaluate(U,Y,ens)['primary']
        print(f"  CLOSE over {list(pool)} -> {p:.6f}")
    elif pick in ART:
        s=load(ART[pick]); p=evaluate(U,Y,s)['primary']; members[pick]=s
        print(f"  executed from cache -> {p:.6f}")
    else:
        print("  (no cached artifact — treated as flat)"); p=champ
    delta=p-champ
    accepted = delta>0.0005
    if accepted and delta>0.002: streak=0
    else: streak+=1
    if accepted: champ=max(champ,p)
    journal.append(f'node_{it:03d} [draft] "{pick}" primary={p:.4f} '+('ACCEPTED' if accepted else 'REJECTED'))
    if pick in CLOSES: break
    if streak>=3: print("  (converged by rule)"); break
print(f"\nREPLAY FINAL: {champ:.6f} after {it} decisions")
