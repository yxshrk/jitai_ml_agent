/* Flight Recorder: mle-agent scrollytelling site.
   2D mission-log replay of the designated run (rundata.js) plus the instrumented
   rare-video-memorization evidence charts (weights.js). No frameworks. */
(function(){
'use strict';
const D=window.RUNDATA||{}, N=D.nodes||[], M=D.meta||{};
const BASELINE=0.6016, BEST=M.best||0.605575, EPS=0.002;
const $=s=>document.querySelector(s);
const esc=s=>{const d=document.createElement('div');d.textContent=String(s??'');return d.innerHTML;};
const nice=id=>String(id||'').replace(/-/g,' ');
const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
/* presenter mode (present.html) reuses every renderer below; scroll-driven
   wiring is skipped there and present.js drives the same machinery by key */
const PRESENT=document.body.classList.contains('present');

/* ---------- scroll progress + bar score --------------------------------- */
if(!PRESENT)addEventListener('scroll',()=>{
  const h=document.documentElement;
  $('#prog').style.width=(h.scrollTop/(h.scrollHeight-h.clientHeight)*100)+'%';
  const hint=document.querySelector('.scrollhint');
  if(hint)hint.classList.toggle('gone',h.scrollTop>innerHeight*0.5);
},{passive:true});

/* ---------- hero chart: the whole run, self-drawing ---------------------- */
(function hero(){
  const svg=$('#herochart'), W=1120, Hh=480, P=64, PT=44, PB=70;
  const S0=0.6005,S1=0.6065;
  const xs=i=>P+i/(N.length-1||1)*(W-2*P);
  const ys=v=>Hh-PB-(Math.min(S1,Math.max(S0,v))-S0)/(S1-S0)*(Hh-PB-PT);
  let h='';
  for(let gx=P,step=(W-2*P)/((N.length-1||1)*2);gx<=W-P+1;gx+=step)
    h+='<line class="axis" x1="'+gx+'" y1="'+ys(S1)+'" x2="'+gx+'" y2="'+(Hh-PB)+'"/>';
  for(let v=S0;v<=S1+1e-9;v+=0.0005){
    const major=Math.round(v*10000)%10===0;
    h+='<line class="axis" x1="'+P+'" y1="'+ys(v)+'" x2="'+(W-P)+'" y2="'+ys(v)+'"'+(major?'':' opacity="0.45"')+'/>';
    if(major)h+='<text class="axistext" x="'+(P-10)+'" y="'+(ys(v)+4)+'" text-anchor="end">'+v.toFixed(3)+'</text>';
  }
  h+='<line class="axis" x1="'+P+'" y1="'+(Hh-PB)+'" x2="'+(W-P)+'" y2="'+(Hh-PB)+'"/>';
  N.forEach((n,i)=>{h+='<text class="axistext" x="'+xs(i)+'" y="'+(Hh-PB+24)+'" text-anchor="middle">'
    +String(i).padStart(2,'0')+'</text>';});
  h+='<text class="axistext" x="'+P+'" y="'+(ys(S1)-16)+'">validation primary ↑</text>';
  h+='<text class="axistext" x="'+(W-P)+'" y="'+(Hh-PB+48)+'" text-anchor="end">iteration →</text>';
  h+='<line class="baseline-l" x1="'+P+'" y1="'+ys(BASELINE)+'" x2="'+(W-P)+'" y2="'+ys(BASELINE)+'"/>';
  h+='<text class="axistext" x="'+(W-P)+'" y="'+(ys(BASELINE)-6)+'" text-anchor="end">baseline 0.6016</text>';
  const acc=N.map((n,i)=>({n,i})).filter(o=>o.n.accepted&&o.n.primary!=null);
  const pts=acc.map(o=>xs(o.i)+','+ys(o.n.primary)).join(' ');
  h+='<polyline id="heroline" class="trace" points="'+pts+'"/>';
  N.forEach((n,i)=>{if(n.primary==null){
      const x=xs(i),y=Hh-PB-7;
      h+='<g stroke="'+css('--amber')+'" stroke-width="1.8" opacity=".85"><line x1="'+(x-6)+'" y1="'+(y-6)+'" x2="'+(x+6)+'" y2="'+(y+6)+'"/>'
        +'<line x1="'+(x-6)+'" y1="'+(y+6)+'" x2="'+(x+6)+'" y2="'+(y-6)+'"/></g>';
      return;}
    h+='<circle r="6" cx="'+xs(i)+'" cy="'+ys(n.primary)+'" fill="'
      +(n.accepted?css('--go'):css('--no'))+'" opacity="'+(n.accepted?1:.65)+'"/>';});
  h+='<text class="axistext" x="'+xs(acc[acc.length-1].i)+'" y="'
    +(ys(BEST)-10)+'" text-anchor="end" fill="'+css('--go')+'">'+BEST.toFixed(4)+'</text>';
  svg.innerHTML=h;
  // chart spans the full text-column width; aspect-ratio CSS sets its height
  const line=$('#heroline'), len=line.getTotalLength();
  const play=()=>{
    line.style.transition='none';line.style.strokeDasharray=len;line.style.strokeDashoffset=len;
    void line.getBoundingClientRect();
    requestAnimationFrame(()=>{line.style.transition='stroke-dashoffset 2.4s ease .4s';
      line.style.strokeWidth=3;line.style.strokeDashoffset=0;});
  };
  line.style.strokeDasharray=len;line.style.strokeDashoffset=len;
  if(!PRESENT)play();
  window.FR_heroPlay=play;
})();

/* ---------- loop: cycle the stage highlight ------------------------------ */
const setStage=k=>document.querySelectorAll('.stage').forEach((c,i)=>c.classList.toggle('live',i===k));
(function loop(){
  const cards=document.querySelectorAll('.stage');if(!cards.length||PRESENT)return;
  let t=0;setStage(0);
  setInterval(()=>{t=(t+1)%cards.length;setStage(t);},2200);
})();

/* ---------- mission log --------------------------------------------------- */
const CH={W:720,H:560,P:60,S0:0.6005,S1:0.6065};
const cx=i=>CH.P+i/(N.length-1||1)*(CH.W-2*CH.P);
const cy=v=>CH.H-CH.P-(Math.min(CH.S1,Math.max(CH.S0,v))-CH.S0)/(CH.S1-CH.S0)*(CH.H-2*CH.P);
(function bigChart(){
  const svg=$('#bigchart');let h='';
  for(let gx=CH.P,step=(CH.W-2*CH.P)/((N.length-1||1)*2);gx<=CH.W-CH.P+1;gx+=step)
    h+='<line class="axis" x1="'+gx+'" y1="'+cy(CH.S1)+'" x2="'+gx+'" y2="'+(CH.H-CH.P)+'"/>';
  for(let v=CH.S0;v<=CH.S1+1e-9;v+=0.0005){
    const major=Math.round(v*10000)%10===0;
    h+='<line class="axis" x1="'+CH.P+'" y1="'+cy(v)+'" x2="'+(CH.W-CH.P)+'" y2="'+cy(v)+'"'+(major?'':' opacity="0.45"')+'/>';
    if(major)h+='<text class="axistext" x="'+(CH.P-10)+'" y="'+(cy(v)+4)+'" text-anchor="end">'+v.toFixed(3)+'</text>';
  }
  h+='<text class="axistext" x="'+CH.P+'" y="'+(cy(CH.S1)-12)+'">validation primary ↑</text>';
  h+='<text class="axistext" x="'+(CH.W-CH.P)+'" y="'+(CH.H-CH.P+38)+'" text-anchor="end">iteration →</text>';
  h+='<line class="baseline-l" x1="'+CH.P+'" y1="'+cy(BASELINE)+'" x2="'+(CH.W-CH.P)+'" y2="'+cy(BASELINE)+'"/>';
  h+='<text class="axistext" x="'+(CH.W-CH.P)+'" y="'+(cy(BASELINE)+16)+'" text-anchor="end" fill="'
    +css('--no')+'">baseline</text>';
  h+='<rect id="epsband" class="epsband" x="'+CH.P+'" y="'+cy(BEST)+'" width="'
    +(CH.W-2*CH.P)+'" height="'+(cy(BEST-EPS)-cy(BEST))+'"/>';
  h+='<polyline id="bigtrace" class="trace" points=""/>';
  N.forEach((n,i)=>{
    if(n.primary==null){
      h+='<g id="nd'+i+'" class="dead void"><line x1="'+(cx(i)-6)+'" y1="'+(cy(CH.S0)+0)+'" x2="'
        +(cx(i)+6)+'" y2="'+(cy(CH.S0)-12)+'" /><line x1="'+(cx(i)+6)+'" y1="'+cy(CH.S0)
        +'" x2="'+(cx(i)-6)+'" y2="'+(cy(CH.S0)-12)+'"/></g>';
      h+='<text id="ilbl'+i+'" class="axistext iterlbl" x="'+cx(i)+'" y="'+(CH.H-CH.P+20)+'" text-anchor="middle">'
        +String(i).padStart(2,'0')+'</text>';
      return;
    }
    if(n.accepted)h+='<circle id="nd'+i+'" class="nodedot" r="7" cx="'+cx(i)+'" cy="'+cy(n.primary)+'"/>';
    else h+='<g id="nd'+i+'" class="dead"><circle r="7" cx="'+cx(i)+'" cy="'+cy(n.primary)+'"/>'
      +'<line x1="'+(cx(i)-9)+'" y1="'+(cy(n.primary)+9)+'" x2="'+(cx(i)+9)+'" y2="'+(cy(n.primary)-9)+'"/></g>';
    h+='<text id="ilbl'+i+'" class="axistext iterlbl" x="'+cx(i)+'" y="'+(CH.H-CH.P+20)+'" text-anchor="middle">'
      +String(i).padStart(2,'0')+'</text>';
  });
  h+='<circle id="curpt" class="curpt" r="12"/>';
  h+='<text id="curlbl" class="axistext curlbl" text-anchor="middle" opacity="0"></text>';
  svg.innerHTML=h;
})();
/* full trace drawn once; reveal is a smooth dashoffset so the line grows and
   shrinks with scroll direction */
let traceLens=null,traceTotal=0;
function initTrace(){
  const pts=[],idxs=[];
  N.forEach((n,i)=>{if(n.accepted&&n.primary!=null){pts.push([cx(i),cy(n.primary)]);idxs.push(i);}});
  const el=$('#bigtrace');
  el.setAttribute('points',pts.map(p=>p.join(',')).join(' '));
  traceTotal=el.getTotalLength();
  traceLens={};let acc=0;
  for(let j=0;j<pts.length;j++){
    if(j>0)acc+=Math.hypot(pts[j][0]-pts[j-1][0],pts[j][1]-pts[j-1][1]);
    traceLens[idxs[j]]=acc;
  }
  // hide instantly (no transition) so the first reveal grows from node 0
  // instead of visibly undrawing the full line
  el.style.transition='none';
  el.style.strokeDasharray=traceTotal;el.style.strokeDashoffset=traceTotal;
  void el.getBoundingClientRect();
  el.style.transition='';
}
initTrace();
function traceUpTo(k){ // reveal the accepted line through iteration k
  if(traceLens==null)initTrace();
  let len=0;
  Object.keys(traceLens).forEach(i=>{if(+i<=k)len=Math.max(len,traceLens[i]);});
  $('#bigtrace').style.strokeDashoffset=traceTotal-len;
}

/* pin the chart so it locks vertically centered in the viewport, while its
   resting position stays snug under the section heading */
(function pinCenter(){
  const pin=document.getElementById('chartpin');if(!pin||PRESENT)return;
  const set=()=>{
    const h=pin.getBoundingClientRect().height;
    pin.style.top=Math.max(64,Math.round((innerHeight-h)/2))+'px';
  };
  set();addEventListener('resize',set);
})();

/* entries */
function headline(n,i){
  const sel=n.selection||{};
  if(i===0)return 'Reproduce the baseline. <span class="go">0.6018</span>. Base camp.';
  if(n.primary==null)return 'The training script crashed. Logged as <span class="amber">VOID</span>, loop continues.';
  const m='tries <b>'+esc(nice(sel.chosen_method_id||'a new package'))+'</b>';
  return n.accepted
    ?m+' → <span class="go">'+n.primary.toFixed(4)+'</span>. Cleared the bar.'
    :m+' → <span class="no">'+n.primary.toFixed(4)+'</span>. Below the gate. Dead end.';
}
(function entries(){
  const box=$('#entries');
  N.forEach((n,i)=>{
    const sel=n.selection||{};
    const stamp=n.primary==null?'<span class="stamp void">VOID</span>'
      :n.accepted?'<span class="stamp acc">ACCEPTED</span>':'<span class="stamp rej">DEAD END</span>';
    let h='<div class="entry'+(n.primary==null?' isvoid':'')+'" data-i="'+i+'">'+stamp+'<h3>ITER '+String(i).padStart(2,'0')+'</h3>'
      +'<div>'+headline(n,i)+'</div>';
    if(sel.diagnosis||sel.why){
      // trim the journal excerpt at a sentence boundary, never mid-word
      let why=sel.why||'';
      if(why.length>420){
        const cut=why.slice(0,420),stop=cut.lastIndexOf('. ');
        why=(stop>200?cut.slice(0,stop+1):cut.slice(0,cut.lastIndexOf(' ')))+' …';
      }
      const q=(sel.diagnosis?('['+esc(sel.diagnosis)+'] '):'')+esc(why);
      h+='<div class="whylbl">why the agent chose this · from the journal</div>'
        +'<div class="quote"><span class="typed" data-full="'+q.replace(/"/g,'&quot;')+'"></span></div>';
    }else if(i>0&&n.primary!=null&&n.summary){
      // iterations without a selection record still journal their hypothesis
      h+='<div class="whylbl">the hypothesis · from the journal</div>'
        +'<div class="quote"><span class="typed" data-full="'+esc(n.summary).replace(/"/g,'&quot;')+'"></span></div>';
    }
    const rej=sel.rejected||[];
    if(sel.chosen_method_id||rej.length){
      h+='<div class="whylbl" style="margin-top:14px">treatments considered · <s>rejected</s> · chosen ✓</div>';
      h+='<div class="chips">';
      rej.forEach(r=>h+='<span class="chip rej">'+esc(nice(r.method_id||r))+'</span>');
      if(sel.chosen_method_id)h+='<span class="chip pick">'+esc(nice(sel.chosen_method_id))+'</span>';
      h+='</div>';
    }
    if(n.primary!=null)h+='<div class="meas dim">measured '+n.primary.toFixed(6)
      +(n.accepted&&i>0?' · Δ +'+(n.primary-BASELINE).toFixed(4)+' vs baseline':'')+'</div>';
    if((sel.why||'').length>420||sel.citation)
      h+='<details><summary>full journal entry</summary><div class="full">'
        +esc(sel.why||n.summary||'')+(sel.citation?'\n\ncitations: '+esc(sel.citation):'')+'</div></details>';
    h+='</div>';
    box.insertAdjacentHTML('beforeend',h);
  });
  // final entry: convergence
  box.insertAdjacentHTML('beforeend','<div class="entry" data-i="conv">'
    +'<span class="stamp acc">CONVERGED</span><h3>STOP</h3>'
    +'<div>Three iterations in a row failed to beat the best score by <b>0.002</b>, so the competition\'s stopping rule (epsilon) fires. '
    +'Final: <span class="go">'+BEST.toFixed(6)+'</span>, +'+(BEST-BASELINE).toFixed(4)
    +' over the baseline. No human said stop.</div></div>');
})();

/* journal quotes render immediately */
function typeIn(el){
  if(el.dataset.done)return;el.dataset.done=1;
  el.innerHTML=el.dataset.full||'';
}

/* scroll activation: draws on the way down, undraws on the way back up */
const active=new Set();
function repaint(){
  let maxI=-1,conv=active.has('conv');
  active.forEach(i=>{if(i!=='conv')maxI=Math.max(maxI,+i);});
  N.forEach((n,k)=>{
    const nd=document.getElementById('nd'+k);
    if(nd)nd.classList.toggle('onn',k<=maxI||conv);
  });
  $('#epsband').classList.toggle('onn',conv);
  traceUpTo(conv?N.length:maxI);
  let best=BASELINE;
  N.forEach((n,k)=>{if((k<=maxI||conv)&&n.accepted&&n.primary!=null)best=Math.max(best,n.primary);});
  if(conv)best=BEST;
  const _bs=$('#barscore');if(_bs)_bs.textContent=best.toFixed(4);
  // label the iteration the reader is currently on
  const pt=$('#curpt'),lb=$('#curlbl');
  let cur=conv?-1:maxI;
  if(cur>=0&&N[cur]){
    const n=N[cur],y=n.primary!=null?cy(n.primary):cy(CH.S0)-6; // VOID cross is drawn 12px tall above the axis; ring centers on it
    pt.setAttribute('cx',cx(cur));pt.setAttribute('cy',y);pt.style.opacity=.9;
    lb.setAttribute('x',cx(cur));lb.setAttribute('y',y-24);
    lb.textContent='ITER '+String(cur).padStart(2,'0')+(n.primary!=null?' · '+n.primary.toFixed(4):' · VOID');
    lb.setAttribute('opacity',1);
    lb.classList.toggle('void',n.primary==null);
  }else{pt.style.opacity=0;lb.setAttribute('opacity',0);}
}
/* hysteresis: activate at 60% visible, deactivate only below 20% (scrolling up),
   so a card hovering near one threshold cannot flicker on and off */
const io=new IntersectionObserver(es=>es.forEach(e=>{
  const el=e.target,i=el.dataset.i,r=e.intersectionRatio;
  if(r>=0.5&&!active.has(i)){
    active.add(i);el.classList.add('onn');
    el.querySelectorAll('.typed').forEach(typeIn);
  }else if(r<=0.05&&active.has(i)&&e.boundingClientRect.top>0){
    // left the viewport downward: the reader scrolled back up past this entry
    active.delete(i);el.classList.remove('onn');
  }else return;
  repaint();
}),{rootMargin:'0px 0px -28% 0px',threshold:[0.05,0.5]});
if(!PRESENT)document.querySelectorAll('.entry').forEach(el=>io.observe(el));

/* hover an entry: highlight its point on the chart */
document.querySelectorAll('.entry').forEach(el=>{
  const i=el.dataset.i;if(i==null||i==='conv')return;
  const marks=()=>[document.getElementById('nd'+i),document.getElementById('ilbl'+i)].filter(Boolean);
  el.addEventListener('mouseenter',()=>marks().forEach(m=>m.classList.add('hl')));
  el.addEventListener('mouseleave',()=>marks().forEach(m=>m.classList.remove('hl')));
});

/* ---------- receipts ------------------------------------------------------ */
(function receipts(){
  const wall=M.wall_s?(M.wall_s/60).toFixed(1)+' min':'n/a';
  // per-metric finals come from the designated run's scored artifact
  const fin=[...N].reverse().find(n=>n.selected_ensemble)?.selected_ensemble||{};
  const cells=[
    [BEST.toFixed(4),'final validation score (primary)'],
    [fin.gauc?fin.gauc.toFixed(4):'n/a','GAUC: per-user ranking quality'],
    [fin.ndcg5?fin.ndcg5.toFixed(4):'n/a','nDCG@5: top-5 ranking quality'],
    ['+'+(BEST-BASELINE).toFixed(4),'gain vs validation baseline (0.6016)'],
    [String(M.iterations||6)+' of 50','iterations used before the rule stopped it'],
    [wall,'wall-clock, start to stop'],
    [(M.tokens||0).toLocaleString('en-US'),'LLM tokens, in + out'],
    [M.stop==='converged'?'ε rule':'cap','what ended the run: the competition\'s stopping rule'],
    ['0','mid-run interventions'],
  ];
  $('#rgrid').innerHTML=cells.map(c=>'<div class="cell"><div class="v">'+c[0]
    +'</div><div class="k">'+c[1]+'</div></div>').join('');
})();


/* ---------- method-card library ------------------------------------------ */
(function methods(){
  const lib=document.getElementById('methodslib');if(!lib||!window.METHODS)return;
  const cls=st=>/measured-win|external-win/.test(st)?'win':/dead|superseded/.test(st)?'mdead':'try';
  const short=st=>/measured-win|external-win/.test(st)?'measured win'
    :/measured-alive/.test(st)?'measured alive'
    :/measured-dead/.test(st)?'measured dead':/superseded/.test(st)?'superseded'
    :/conditional/i.test(st)?'conditional':/untried/.test(st)?'untried':(st||'untried').split('(')[0].slice(0,24);
  const esc2=t=>{const d=document.createElement('div');d.textContent=t;return d.innerHTML;};
  function render(q){
    q=(q||'').toLowerCase();
    const rows=window.METHODS.filter(c=>!q||JSON.stringify(c).toLowerCase().includes(q));
    lib.innerHTML=rows.map(c=>'<div class="mrow">'
      +'<div><div class="mid">'+esc2(c.title||c.id)+'</div><div class="mcite">'+esc2((c.citation||'').replace(/`/g,''))+'</div></div>'
      +'<div class="mmech">'+esc2((c.mechanism||'').slice(0,240))+((c.mechanism||'').length>240?'…':'')+'</div>'
      +'<div class="mstat"><span class="'+cls(c.status)+'">'+esc2(short(c.status))+'</span>'
      +(c.evidence&&c.evidence!=='none'?'<div class="mcite">'+esc2(c.evidence.split(' ')[0])+'</div>':'')+'</div>'
      +'</div>').join('')||'<div class="mrow"><div class="mmech">no cards match</div></div>';
  }
  render('');
  const inp=document.getElementById('msearch');
  if(inp)inp.addEventListener('input',()=>render(inp.value));
})();

/* ---------- evidence: rare-video memorization, baseline vs treated -------- */
(function evidence(){
  const Wt=window.WEIGHTS;const grid=document.getElementById('evgrid');
  if(!grid)return;
  if(!Wt){grid.innerHTML='<p class="lede">(instrumented data missing: run tools/instrument_weights.py)</p>';return;}
  const panels=[['baseline','BASELINE','rare-video embeddings balloon'],
                ['treated','AGENT-TREATED','rare-video embeddings stay bounded']];
  const PW=560,PH=300,P=40;
  grid.innerHTML=panels.map(([key,title,subtitle])=>{
    const snaps=Wt[key];
    const maxN=Math.max(...Wt.baseline.concat(Wt.treated).flatMap(s=>s.norms));
    const PR=84; const xs=i=>P+i/(snaps.length-1)*(PW-P-PR-P);
    const ys=v=>PH-P-(v/maxN)*(PH-2*P);
    let h='<div class="evpanel"><div class="t"><b>'+title+'</b> · '+subtitle+'</div>'
      +'<svg viewBox="0 0 '+PW+' '+PH+'">';
    for(let gx=P;gx<=PW-P-PR+1;gx+=(PW-P-PR-P)/10)
      h+='<line class="axis" x1="'+gx+'" y1="'+P+'" x2="'+gx+'" y2="'+(PH-P)+'"/>';
    for(let gy=PH-P;gy>=P-1;gy-=(PH-2*P)/6)
      h+='<line class="axis" x1="'+P+'" y1="'+gy+'" x2="'+(PW-P-PR)+'" y2="'+gy+'"/>';
    h+='<line class="axis" x1="'+P+'" y1="'+(PH-P)+'" x2="'+(PW-P-PR)+'" y2="'+(PH-P)+'"/>';
    h+='<text class="axistext" x="'+P+'" y="'+(PH-P+18)+'">training epochs →</text>';
    h+='<text class="axistext" x="'+P+'" y="'+(P-10)+'">average embedding size</text>';
    // three lines only: rarest decile (the story), median, most seen
    const series=[[0,'var(--no)',2.5,'rarest 10%'],
                  [4,'var(--faint)',1.6,'median'],
                  [9,'#8A8A84',1.6,'most seen']];
    series.forEach(([d,col,wid,name])=>{
      const pts=snaps.map((s,i)=>xs(i)+','+ys(s.norms[d])).join(' ');
      h+='<polyline class="evline" data-key="'+key+'" points="'+pts+'" stroke="'+col
        +'" stroke-width="'+wid+'"/>';
      const endy=ys(snaps[snaps.length-1].norms[d]);
      h+='<text class="evlbl" x="'+(PW-P-PR+8)+'" y="'+(endy+4)+'" style="fill:'
        +(d===0?'var(--no)':'var(--dim)')+'">'+name+'</text>';
    });
    const last=snaps[snaps.length-1];
    const midi=Math.floor(snaps.length*0.55);
    if(key==='baseline'){
      h+='<text class="evnote" x="'+xs(midi)+'" y="'+(ys(snaps[midi].norms[0])+24)
        +'" text-anchor="middle">never stops growing ↗</text>';
    }else{
      h+='<text class="evnote" x="'+xs(midi)+'" y="'+(ys(snaps[midi].norms[0])-16)
        +'" text-anchor="middle" style="fill:var(--go)">flattens out ✓</text>';
    }
    h+='<text class="axistext" x="'+(PW-P-PR)+'" y="'+(PH-P+18)+'" text-anchor="end">final valid '
      +last.primary.toFixed(4)+'</text>';
    h+='</svg></div>';
    return h;
  }).join('');
  // charts render fully drawn; no entrance animation

})();

/* ---------- architecture diagram (shared: index + presenter) --------------
   Drawn in four parts so it can be walked: agent row, harness row, journal
   loop-back, stop + exclusion. setArch(k) keeps parts 0..k lit and dims the
   rest; setArch(-1) lights everything (site default). ---------------------- */
const ARCH_PARTS=['agent','harness','journal','stop'];
function setArch(k){
  const svg=document.getElementById('archdiagram');if(!svg)return;
  svg.querySelectorAll('[data-part]').forEach(g=>{
    const i=ARCH_PARTS.indexOf(g.dataset.part);
    g.classList.toggle('off',k>=0&&i>k);
    g.classList.toggle('lit',k>=0&&i===k);
  });
}
(function arch(){
  const svg=document.getElementById('archdiagram');if(!svg)return;
  const parts={agent:'',harness:'',journal:'',stop:''};
  let cur='agent';
  const add=s=>{parts[cur]+=s;};
  const box=(x,y,w,hh,cls,title,sub)=>{
    add('<rect class="abox '+cls+'" x="'+x+'" y="'+y+'" width="'+w+'" height="'+hh+'" rx="8"/>');
    add('<text class="at" x="'+(x+18)+'" y="'+(y+30)+'">'+title+'</text>');
    (sub||[]).forEach((t,i)=>{add('<text class="as" x="'+(x+18)+'" y="'+(y+52+i*18)+'">'+t+'</text>');});
  };
  // labels sit centered on their line, on a small chip so they stay readable
  const arrow=(x1,y1,x2,y2,cls,lbl)=>{
    add('<line class="aflow '+(cls||'')+'" x1="'+x1+'" y1="'+y1+'" x2="'+x2+'" y2="'+y2
      +'" marker-end="url(#ah'+(cls||'k')+')"/>');
    if(lbl){
      const lx=(x1+x2)/2, ly=(y1+y2)/2+4, w=lbl.length*7.4+22;
      add('<rect class="achip" x="'+(lx-w/2)+'" y="'+(ly-15)+'" width="'+w+'" height="22" rx="11"/>');
      add('<text class="albl '+(cls||'')+'" x="'+lx+'" y="'+ly+'" text-anchor="middle">'+lbl+'</text>');
    }
  };
  const defs='<defs>'+['k','go','no'].map(c=>'<marker id="ah'+c+'" viewBox="0 0 10 10" refX="9" refY="5" '
    +'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
    +'<path d="M0,0L10,5L0,10z" fill="'+(c==='go'?css('--go'):c==='no'?css('--no'):css('--ink'))+'"/></marker>').join('')+'</defs>';
  // grid: four columns; harness row vertically centered between the other two
  const X=[60,340,620,900],W=220,BH=100,Y1=96,Y2=366,Y3=636;
  const my1=Y1+BH/2,my2=Y2+BH/2;
  // part 1: the agent row, left to right
  cur='agent';
  add('<text class="alane" x="'+(X[3]+W)+'" y="60" text-anchor="end">AGENT · THE LLM PROPOSES</text>');
  box(X[1],Y1,W,BH,'agent','DIAGNOSE',['reads the last training','curves, names the bottleneck']);
  box(X[2],Y1,W,BH,'agent','TREAT',['picks a treatment from the','42-card method library']);
  box(X[3],Y1,W,BH,'agent','WRITE SCRIPT',['a whole script, or a patch','against a frozen contract']);
  arrow(X[1]+W,my1,X[2],my1,'','');
  arrow(X[2]+W,my1,X[3],my1,'','');
  // part 2: the harness row, right to left, inside the workspace boundary
  cur='harness';
  add('<text class="alane" x="'+X[1]+'" y="'+(Y2-36)+'">HARNESS · FIXED CODE, NO LLM · NEVER TRUSTS THE AGENT</text>');
  arrow(X[3]+W/2,Y1+BH,X[3]+W/2,Y2-22,'','hands the script off');
  add('<rect class="abound" x="'+(X[2]-20)+'" y="'+(Y2-22)+'" width="'+(X[3]+W-X[2]+40)+'" height="'+(BH+44)+'" rx="10"/>');
  add('<text class="as" x="'+(X[2]-6)+'" y="'+(Y2+BH+38)+'">agent workspace boundary</text>');
  box(X[3],Y2,W,BH,'','SCREEN + SMOKE',['code screened for test','access · smoke + sanity gate']);
  box(X[2],Y2,W,BH,'','TRAIN',['same time-based split:','train + validation only']);
  box(X[1],Y2,W,BH,'','EVALUATE',['per-user ranking quality','(official GAUC + nDCG@5)']);
  box(X[0],Y2,W,BH,'go','GATE',['keeps a change only if the','gain beats measured noise']);
  arrow(X[3],my2,X[2]+W,my2,'','');
  arrow(X[2],my2,X[1]+W,my2,'','scores');
  arrow(X[1],my2,X[0]+W,my2,'','Δ');
  // part 3: the journal closes the loop; the run starts from the published baseline
  cur='journal';
  {const sx=X[0]+W/2, lbl='START · iteration 0: the published baseline, 3 seeds', w=lbl.length*7.4+22;
   add('<line class="aflow go" x1="'+sx+'" y1="30" x2="'+sx+'" y2="'+(Y1-2)+'" marker-end="url(#ahgo)"/>');
   add('<rect class="achip" x="'+(sx+10)+'" y="44" width="'+w+'" height="22" rx="11"/>');
   add('<text class="albl go" x="'+(sx+21)+'" y="59">'+lbl+'</text>');}
  box(X[0],Y1,W,BH,'','JOURNAL',['the run’s memory: every','hypothesis, score, failure']);
  arrow(X[0]+W,my1,X[1],my1,'','');
  arrow(X[0]+W/2,Y2,X[0]+W/2,Y1+BH,'go','accept / dead end, journaled');
  // part 4: what ends the run, and what can never enter it
  cur='stop';
  box(X[0],Y3,W+60,90,'go','STOPPING RULE (EPSILON)',['3 iterations without a 0.002','gain end the run · no human stops it']);
  box(X[3],Y3,W,90,'no','HIDDEN TEST SET',['never copied into the workspace:','no path can reach it']);
  arrow(X[0]+W/2,Y2+BH,X[0]+W/2,Y3,'go','');
  arrow(X[3]+W/2,Y2+BH+22,X[3]+W/2,Y3,'no','no path exists');
  svg.innerHTML=defs+ARCH_PARTS.map(p=>'<g class="apart" data-part="'+p+'">'+parts[p]+'</g>').join('');
  // site: a numbered legend under the diagram lights each part on hover
  const legend=document.getElementById('archlegend');
  if(legend){
    legend.querySelectorAll('[data-part]').forEach(li=>{
      li.addEventListener('mouseenter',()=>setArch(ARCH_PARTS.indexOf(li.dataset.part)));
      li.addEventListener('mouseleave',()=>setArch(-1));
    });
  }
})();

/* hooks for presenter mode: present.js drives the same replay machinery */
window.FR={active,repaint,typeIn,setStage,setArch,iters:N.length,heroPlay:window.FR_heroPlay};
delete window.FR_heroPlay;
})();
