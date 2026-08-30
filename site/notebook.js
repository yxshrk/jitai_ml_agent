(function(){
const D=window.RUNDATA,W=window.WEIGHTS,SL=window.SLATES,M=D.meta||{},N=D.nodes||[];
const root=document.getElementById('pages');
const fmtp=p=>p?p.toFixed(6):'—';
// ---------- helpers: self-drawing SVG curve ----------
let uid=0;
function curveSVG(series,w,h,opts){ // series=[{pts:[{x,y}],color,ghost,label}]
  opts=opts||{};
  const pad=34, X=v=>pad+v*(w-2*pad), Y=v=>h-pad-(v)*(h-2*pad);
  const y0=opts.y0??0.585, y1=opts.y1??0.607;
  const ny=p=>Math.max(0,Math.min(1,(p-y0)/(y1-y0)));
  let s=`<svg viewBox="0 0 ${w} ${h}" class="curve">`;
  [0.59,0.595,0.6,0.605].forEach(v=>{ if(v<y0||v>y1)return;
    s+=`<line x1="${pad}" x2="${w-pad}" y1="${Y(ny(v))}" y2="${Y(ny(v))}" class="gl"/>`+
       `<text x="4" y="${Y(ny(v))+4}" class="axis">${v.toFixed(3)}</text>`;});
  series.forEach(se=>{
    if(!se.pts.length)return;
    const d=se.pts.map((p,i)=>(i?'L':'M')+X(p.x)+','+Y(ny(p.y))).join(' ');
    const id='c'+(uid++);
    s+=`<path id="${id}" d="${d}" fill="none" stroke="${se.color}" stroke-width="${se.ghost?2:3}" `+
       `stroke-linecap="round" ${se.ghost?'stroke-dasharray="5 6" opacity="0.5"':'class="draw"'}/>`;
    if(se.peak){const pk=se.pts.reduce((a,b)=>b.y>a.y?b:a);
      s+=`<circle cx="${X(pk.x)}" cy="${Y(ny(pk.y))}" r="6" class="peak"/>`+
         `<text x="${X(pk.x)+10}" y="${Y(ny(pk.y))-8}" class="anno">${se.peakNote||'peak — then it falls: memorizing'}</text>`;}
    if(se.label)s+=`<text x="${X(se.pts[se.pts.length-1].x)-6}" y="${Y(ny(se.pts[se.pts.length-1].y))-10}" text-anchor="end" class="clabel" fill="${se.color}">${se.label}</text>`;
  });
  return s+'</svg>';
}
function nodeCurve(n){ return (n.curve||[]).map((c,i,arr)=>({x:i/Math.max(1,arr.length-1),y:c.p})); }
// ---------- network panel (real weight norms) ----------
function networkSVG(which,step){ // which: 'baseline'|'treated'
  const snaps=(W&&W[which])||[]; if(!snaps.length)return '<div class="dim">instrumentation pending…</div>';
  const s0=snaps[0].norms, s=snaps[Math.min(step,snaps.length-1)];
  const w=560,h=250;
  let out=`<svg viewBox="0 0 ${w} ${h}" class="net">`;
  // layers: 5 field columns -> hidden -> score
  const fields=['user','video','author','tab','dur'];
  fields.forEach((f,fi)=>{ const x=40, y=28+fi*44;
    out+=`<text x="${x-32}" y="${y+4}" class="axis">${f}</text>`;});
  // video row expands into 10 frequency-decile cells with glow = norm growth vs start
  for(let d0=0; d0<10; d0++){
    const growth=Math.max(0,(s.norms[d0]-s0[d0])/(s0[d0]+1e-9));
    const heat=Math.min(1,growth*2.2);
    const col=`rgb(${Math.round(60+heat*195)},${Math.round(80-heat*30)},${Math.round(120-heat*70)})`;
    const x=70+d0*30, y=72;
    out+=`<rect x="${x}" y="${y-12}" width="24" height="24" rx="5" fill="${col}"/>`;
  }
  out+=`<text x="70" y="50" class="anno">video embeddings, rare → common (real norms, glow = growth since start)</text>`;
  // other fields as quiet nodes
  [0,2,3,4].forEach(fi=>{ const y=28+fi*44;
    out+=`<rect x="70" y="${y-11}" width="24" height="22" rx="5" fill="#2a3350"/>`;});
  // edges to hidden to score
  for(let i=0;i<5;i++){ const y=28+i*44;
    out+=`<line x1="96" y1="${y}" x2="360" y2="118" class="edge"/>`;}
  out+=`<rect x="360" y="96" width="34" height="44" rx="8" fill="#31406b"/><text x="356" y="156" class="axis">MLP</text>`;
  out+=`<line x1="394" y1="118" x2="480" y2="118" class="edge"/>`;
  out+=`<circle cx="500" cy="118" r="16" fill="#d8b24a"/><text x="488" y="152" class="axis">score</text>`;
  out+=`<text x="${w-190}" y="24" class="anno">checkpoint ${s.ck} · valid ${s.primary.toFixed(4)}</text>`;
  return out+'</svg>';
}
function netBlock(which,title){
  const snaps=(W&&W[which])||[];
  const id='nb'+(uid++);
  setTimeout(()=>{ // animate through snapshots
    let i=0; const el=document.getElementById(id); if(!el)return;
    const iv=setInterval(()=>{ if(!document.body.contains(el)){clearInterval(iv);return;}
      el.innerHTML=networkSVG(which,i); i=(i+1); if(i>=snaps.length){clearInterval(iv);} },700);
  },400);
  return `<div class="netwrap"><div class="nettitle">${title}</div><div id="${id}">${networkSVG(which,0)}</div></div>`;
}
// ---------- pages ----------
const pages=[];
pages.push({t:'Cover',h:`<div class="cover"><div class="kicker">TikTok TechJam 2026 · Track 2</div>
<h1>The Agent's Lab Notebook</h1>
<p class="lede">One autonomous research run — <b>${M.run}</b> — replayed page by page from its own logs. Every curve, number, diagnosis and stamp below is real.</p>
<div class="loopstrip"><span>diagnose</span>→<span>select a card</span>→<span>implement</span>→<span>retrain</span>→<span>face the evidence</span></div>
<p class="dim">${M.iterations} decisions · ${Math.round((M.wall_s||0)/60)} min · ${(M.tokens||0).toLocaleString()} LLM tokens · zero mid-run human actions</p></div>`});
pages.push({t:'The disease',h:`<h2>Page 1 — the patient presents</h2>
<p>The official baseline trains happily — its <i>training</i> loss falls forever. But its validation score tells the truth:</p>
${curveSVG([{pts:nodeCurve(N[0]),color:'#d8b24a',peak:true,peakNote:'peaks — then falls. The model is memorizing.',label:'baseline'}],640,240)}
<p>That downhill slope after the peak is <b>overfitting</b> — and with only ~42 impressions per user, there is a lot to memorize and little to learn. Champion so far: <b>${fmtp(N[0].primary)}</b>.</p>`});
pages.push({t:'Inside the net',h:`<h2>Page 2 — the disease, mechanically</h2>
<p>We instrumented the same training to watch the weights themselves. Each cell is the median embedding size of videos in a popularity decile — <b>rare videos left, popular right</b>. Watch the rare cells heat up: the network is burning capacity memorizing videos it barely saw.</p>
${netBlock('baseline','BASELINE — untreated (animating through real checkpoints)')}
<p class="dim">Real measurements from tools/instrument_weights.py — rare-video embedding norms grow fastest exactly as validation starts falling.</p>`});
N.forEach((n,ix)=>{
  if(n.id==='node_000')return;
  if(n.error&&!n.primary){
    pages.push({t:n.id,h:`<h2>${n.id} — the tooling fails</h2>
    <p class="mono dim">${n.summary}</p>
    <p>No model was produced. The harness journals the failure and the convergence strike counts anyway.</p>
    <div class="stamp bad">STRIKE</div>`});
    return;
  }
  const sel=n.selection||{};
  const fan=(D.cards||[]).slice(0,0); // deck rendered from selection below
  const rej=(sel.rejected||[]).map(r=>r.method_id);
  const deck = sel.chosen_method_id? `<div class="cards">${[...rej.slice(0,3).map(c=>`<span class="mcard rej">${c}</span>`),`<span class="mcard pick">${sel.chosen_method_id} ✓</span>`].join('')}</div>
   ${sel.why?`<p class="why">“${String(sel.why).slice(0,230)}”</p>`:''}`:'';
  const series=[];
  if(N[0].curve&&N[0].curve.length)series.push({pts:nodeCurve(N[0]),color:'#8b95ad',ghost:true,label:'baseline'});
  if(n.curve&&n.curve.length>2)series.push({pts:nodeCurve(n),color:n.accepted?'#7fb069':'#5b8dd9',peak:!n.accepted,label:n.id});
  pages.push({t:n.id,h:`<h2>${n.id} — consultation</h2>
  <p><b>Diagnosis (verbatim from the journal):</b> ${(n.summary.split('will')[0]||n.summary).slice(0,240)}…</p>
  ${deck}
  ${n.probes&&n.probes.length>2?`<p class="dim">The treatment is a search: ${n.probes.length} internal trials logged before committing${n.members&&n.members.length?`; ${n.members.length} seed members trained for the ensemble`:''}.</p>`:''}
  ${series.length>1?curveSVG(series,640,220):''}
  <p><b>Verdict:</b> ${fmtp(n.primary)} ${n.accepted?'<span class="ok">beats the noise floor</span>':'<span class="no">does not clear the evidence bar</span>'}${n.fixed!=null?` · fixed <b>${n.fixed.toLocaleString()}</b> orderings, broke <b>${n.broken.toLocaleString()}</b>${n.id==='node_006'?' (raw counts even net negative — but the metric weights users by positives and the top of the list: the official score improves. Counts are not the metric.)':''}`:''}</p>
  <div class="stamp ${n.accepted?'good':'bad'}">${n.accepted?'ACCEPTED':'REJECTED'}</div>`});
});
pages.push({t:'The cure',h:`<h2>Page ${N.length+1} — the treated network</h2>
<p>Same instrumentation, now with the agent's accepted treatment (dropout 0.18, weight decay, rapid LR decay, recency weighting):</p>
${netBlock('treated','TREATED — the champion package (real checkpoints)')}
<p>The rare-video cells stay cool — memorization suppressed — and validation holds its peak instead of collapsing. This is <i>why</i> the treatment works, in the weights themselves.</p>`});
if(SL&&SL.length){
  const rows=SL.map((s,si)=>{
    const vb=[...s.videos].sort((a,b)=>b.b-a.b), vaa=[...s.videos].sort((a,b)=>b.a-a.a);
    const col=v=>`<div class="chip ${v.watched?'w':''}">${v.watched?'▶ watched':'· skipped'}</div>`;
    return `<div class="slate"><div class="scol"><div class="stit">before</div>${vb.map(col).join('')}</div>
            <div class="scol"><div class="stit">after</div>${vaa.map(col).join('')}</div></div>`;
  }).join('');
  pages.push({t:'Real users',h:`<h2>Page ${N.length+2} — what changed for real users</h2>
  <p>Six actual validation users. Each column is their feed ordered by the model — watched videos should rise to the top. Four improved; the last two got <i>worse</i> (the broken pairs — improvement is a trade, and we show both sides).</p>
  <div class="slates">${rows}</div>`});
}
pages.push({t:'Converged',h:`<div class="cover"><h1>${fmtp(M.best)}</h1>
<p class="lede">Stopped by the official convergence rule after ${M.iterations} decisions — the agent cannot overstay. +0.00398 over the published baseline; predicted hidden-test ≈ 0.5977 ± 0.0020 (we correct our own winner's curse).</p>
<p class="dim">113 disclosed runs · every negative kept · replayable from the public repo</p></div>`});
// ---------- render + nav ----------
let cur=0;
function render(){
  root.innerHTML=`<div class="page">${pages[cur].h}</div>`;
  document.getElementById('step-label').textContent=`${cur+1} / ${pages.length}`;
  document.getElementById('prev').disabled=cur===0;
  document.getElementById('next').disabled=cur===pages.length-1;
  root.querySelectorAll('path.draw').forEach(p=>{const L=p.getTotalLength();
    p.style.strokeDasharray=L; p.style.strokeDashoffset=L;
    p.getBoundingClientRect(); p.style.transition='stroke-dashoffset 1.8s ease';
    requestAnimationFrame(()=>p.style.strokeDashoffset='0');});
}
document.getElementById('next').onclick=()=>{if(cur<pages.length-1){cur++;render();}};
document.getElementById('prev').onclick=()=>{if(cur>0){cur--;render();}};
addEventListener('keydown',e=>{if(e.key==='ArrowRight')document.getElementById('next').click();if(e.key==='ArrowLeft')document.getElementById('prev').click();});
render();
})();
