"""Campaign post-mortem: mechanical aggregation over every completed run journal."""
import json, glob, collections
rows=[]; fail=collections.Counter(); accepts=collections.Counter(); rejects=collections.Counter()
near=[]; closes=[]
for sp in sorted(glob.glob('logs/run_*/summary.json')):
    run=sp.split('/')[1]
    try: s=json.load(open(sp))
    except Exception: continue
    best=s.get('best_metrics',{}).get('primary',0)
    nodes=[]
    try:
        for l in open(sp.replace('summary.json','journal.jsonl')):
            nodes.append(json.loads(l))
    except Exception: pass
    champ=0
    for n in nodes:
        nid=n.get('node_id'); p=(n.get('metrics') or {}).get('primary',0) or 0
        err=(n.get('error') or ''); summ=(n.get('change_summary') or '')
        if nid=='node_000': champ=p; continue
        if n.get('accepted'):
            champ=max(champ,p); accepts[summ[:40]]+=1
        elif err:
            if 'parse' in err or 'JSON' in err: fail['llm_parse']+=1
            elif 'slice' in err: fail['dict_history_bug']+=1
            elif 'timeout' in err: fail['timeout']+=1
            elif 'anthropic' in err: fail['anthropic_empty']+=1
            elif 'budget' in err: fail['budget']+=1
            else: fail['other_exec']+=1
        else:
            rejects['rejected']+=1
            if p>champ and p-champ<0.0009 and p>0.603: near.append((run,nid,round(p,5),round(p-champ,5)))
        low=summ.lower()
        if ('ensemble' in low or 'member' in low or 'committee' in low) and nid!='node_000' and p>0.55:
            closes.append((run,nid,round(p,5),bool(n.get('accepted'))))
    rows.append((run,s.get('dataset','pure'),s.get('stop_reason'),s.get('iterations'),round(best,5)))
print('=== completed runs:',len(rows))
print('=== failure causes:',dict(fail))
print('=== total accepts:',sum(accepts.values()),'rejects:',sum(rejects.values()))
print('=== sub-floor near-misses ABOVE champion (>0.603):')
for t in sorted(near,key=lambda x:-x[2])[:10]: print('   ',t)
print('=== ensemble/member close attempts (top 12 by score):')
for t in sorted(closes,key=lambda x:-x[2])[:12]: print('   ',t)
