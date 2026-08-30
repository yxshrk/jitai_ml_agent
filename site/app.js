/* Flight Recorder — mle-agent scrollytelling site.
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

/* ---------- scroll progress + bar score --------------------------------- */
addEventListener('scroll',()=>{
  const h=document.documentElement;
  $('#prog').style.width=(h.scrollTop/(h.scrollHeight-h.clientHeight)*100)+'%';
},{passive:true});

/* ---------- hero chart: the whole run, self-drawing ---------------------- */
(function hero(){
  const svg=$('#herochart'), W=640, Hh=420, P=52;
  const S0=0.6005,S1=0.6062;
  const xs=i=>P+i/(N.length-1||1)*(W-2*P);
  const ys=v=>Hh-P-(Math.min(S1,Math.max(S0,v))-S0)/(S1-S0)*(Hh-2*P);
  let h='<line class="axis" x1="'+P+'" y1="'+(Hh-P)+'" x2="'+(W-P)+'" y2="'+(Hh-P)+'"/>';
  h+='<line class="baseline-l" x1="'+P+'" y1="'+ys(BASELINE)+'" x2="'+(W-P)+'" y2="'+ys(BASELINE)+'"/>';
  h+='<text class="axistext" x="'+(W-P)+'" y="'+(ys(BASELINE)-6)+'" text-anchor="end">baseline 0.6016</text>';
  const acc=N.map((n,i)=>({n,i})).filter(o=>o.n.accepted&&o.n.primary!=null);
  const pts=acc.map(o=>xs(o.i)+','+ys(o.n.primary)).join(' ');
  h+='<polyline id="heroline" class="trace" points="'+pts+'"/>';
  N.forEach((n,i)=>{if(n.primary==null)return;
    h+='<circle r="6" cx="'+xs(i)+'" cy="'+ys(n.primary)+'" fill="'
      +(n.accepted?css('--go'):css('--no'))+'" opacity="'+(n.accepted?1:.65)+'"/>';});
  h+='<text class="axistext" x="'+xs(acc[acc.length-1].i)+'" y="'
    +(ys(BEST)-10)+'" text-anchor="end" fill="'+css('--go')+'">'+BEST.toFixed(4)+'</text>';
  svg.innerHTML=h;
  const line=$('#heroline'), len=line.getTotalLength();
  line.style.strokeDasharray=len;line.style.strokeDashoffset=len;
  requestAnimationFrame(()=>{line.style.transition='stroke-dashoffset 2.4s ease .4s';line.style.strokeWidth=3;
    line.style.strokeDashoffset=0;});
})();

/* ---------- loop diagram -------------------------------------------------- */
(function loop(){
  const stages=['DIAGNOSE','TREAT','RETRAIN','MEASURE'];
  const g=$('#loopg');let h='';
  const bw=150,bh=44,y=40,gap=(820-4*bw-40)/3;
  stages.forEach((s,i)=>{
    const x=20+i*(bw+gap);
    h+='<rect class="boxr" x="'+x+'" y="'+y+'" width="'+bw+'" height="'+bh+'" rx="3"/>';
    h+='<text x="'+(x+bw/2)+'" y="'+(y+bh/2+5)+'" text-anchor="middle">'+s+'</text>';
    if(i<3)h+='<path class="arrow" d="M'+(x+bw+4)+' '+(y+bh/2)+' L'+(x+bw+gap-6)+' '+(y+bh/2)+'"/>';
  });
  // loopback arrow measure -> diagnose
  h+='<path class="arrow" d="M'+(20+3*(bw+gap)+bw/2)+' '+(y+bh+6)
    +' C '+(20+3*(bw+gap)+bw/2)+' 135, '+(20+bw/2)+' 135, '+(20+bw/2)+' '+(y+bh+6)+'"/>';
  h+='<text class="axistext" x="410" y="128" text-anchor="middle" fill="'+css('--dim')
    +'">accept only if Δ ≥ ε on validation · stop on 3 quiet iterations</text>';
  h+='<circle id="looppulse" class="pulse" r="4" cx="20" cy="'+(y+bh/2)+'"/>';
  g.innerHTML=h;
  // pulse orbiting the loop
  const pulse=$('#looppulse');const xsP=[95,95+bw+gap,95+2*(bw+gap),95+3*(bw+gap)];
  let t=0;setInterval(()=>{t=(t+1)%4;
    pulse.setAttribute('cx',xsP[t]);},1200);
})();

/* ---------- mission log --------------------------------------------------- */
const CH={W:560,H:420,P:46,S0:0.6005,S1:0.6062};
const cx=i=>CH.P+i/(N.length-1||1)*(CH.W-2*CH.P);
const cy=v=>CH.H-CH.P-(Math.min(CH.S1,Math.max(CH.S0,v))-CH.S0)/(CH.S1-CH.S0)*(CH.H-2*CH.P);
(function bigChart(){
  const svg=$('#bigchart');let h='';
  for(let v=0.601;v<=CH.S1;v+=0.001){
    h+='<line class="axis" x1="'+CH.P+'" y1="'+cy(v)+'" x2="'+(CH.W-CH.P)+'" y2="'+cy(v)+'" opacity="0.5"/>';
    h+='<text class="axistext" x="'+(CH.P-8)+'" y="'+(cy(v)+4)+'" text-anchor="end">'+v.toFixed(3)+'</text>';
  }
  h+='<line class="baseline-l" x1="'+CH.P+'" y1="'+cy(BASELINE)+'" x2="'+(CH.W-CH.P)+'" y2="'+cy(BASELINE)+'"/>';
  h+='<text class="axistext" x="'+(CH.W-CH.P)+'" y="'+(cy(BASELINE)+16)+'" text-anchor="end" fill="'
    +css('--no')+'">baseline</text>';
  h+='<rect id="epsband" class="epsband" x="'+CH.P+'" y="'+cy(BEST)+'" width="'
    +(CH.W-2*CH.P)+'" height="'+(cy(BEST-EPS)-cy(BEST))+'"/>';
  h+='<polyline id="bigtrace" class="trace" points=""/>';
  N.forEach((n,i)=>{
    if(n.primary==null){
      h+='<g id="nd'+i+'" class="dead"><line x1="'+(cx(i)-6)+'" y1="'+(cy(CH.S0)+0)+'" x2="'
        +(cx(i)+6)+'" y2="'+(cy(CH.S0)-12)+'" /><line x1="'+(cx(i)+6)+'" y1="'+cy(CH.S0)
        +'" x2="'+(cx(i)-6)+'" y2="'+(cy(CH.S0)-12)+'"/></g>';
      return;
    }
    if(n.accepted)h+='<circle id="nd'+i+'" class="nodedot" r="5" cx="'+cx(i)+'" cy="'+cy(n.primary)+'"/>';
    else h+='<g id="nd'+i+'" class="dead"><circle r="5" cx="'+cx(i)+'" cy="'+cy(n.primary)+'"/>'
      +'<line x1="'+(cx(i)-7)+'" y1="'+(cy(n.primary)+7)+'" x2="'+(cx(i)+7)+'" y2="'+(cy(n.primary)-7)+'"/></g>';
    h+='<text class="axistext" x="'+cx(i)+'" y="'+(CH.H-CH.P+18)+'" text-anchor="middle">'
      +String(i).padStart(2,'0')+'</text>';
  });
  svg.innerHTML=h;
})();
function traceUpTo(k){ // accepted polyline through iteration k
  const pts=[];
  N.forEach((n,i)=>{if(i<=k&&n.accepted&&n.primary!=null)pts.push(cx(i)+','+cy(n.primary));});
  $('#bigtrace').setAttribute('points',pts.join(' '));
}

/* entries */
function headline(n,i){
  const sel=n.selection||{};
  if(i===0)return 'Reproduce the baseline. <span class="go">0.6018</span> — base camp.';
  if(n.primary==null)return 'Node crashed mid-run. Logged as <span class="amber">VOID</span>, loop continues.';
  const m='tries <b>'+esc(nice(sel.chosen_method_id||'a new package'))+'</b>';
  return n.accepted
    ?m+' → <span class="go">'+n.primary.toFixed(4)+'</span>. Cleared the bar.'
    :m+' → <span class="no">'+n.primary.toFixed(4)+'</span>. Below ε — dead end.';
}
(function entries(){
  const box=$('#entries');
  N.forEach((n,i)=>{
    const sel=n.selection||{};
    const stamp=n.primary==null?'<span class="stamp void">VOID</span>'
      :n.accepted?'<span class="stamp acc">ACCEPTED</span>':'<span class="stamp rej">DEAD END</span>';
    let h='<div class="entry" data-i="'+i+'">'+stamp+'<h3>ITER '+String(i).padStart(2,'0')+'</h3>'
      +'<div>'+headline(n,i)+'</div>';
    if(sel.diagnosis||sel.why){
      const q=(sel.diagnosis?('['+esc(sel.diagnosis)+'] '):'')+esc((sel.why||'').slice(0,260))
        +((sel.why||'').length>260?'…':'');
      h+='<div class="quote"><span class="typed" data-full="'+q.replace(/"/g,'&quot;')+'"></span></div>';
    }
    const rej=sel.rejected||[];
    if(sel.chosen_method_id||rej.length){
      h+='<div class="chips">';
      rej.forEach(r=>h+='<span class="chip rej">'+esc(nice(r.method_id||r))+'</span>');
      if(sel.chosen_method_id)h+='<span class="chip pick">'+esc(nice(sel.chosen_method_id))+'</span>';
      h+='</div>';
    }
    if(n.primary!=null)h+='<div class="meas dim">measured '+n.primary.toFixed(6)
      +(n.accepted&&i>0?' · Δ +'+(n.primary-BASELINE).toFixed(4)+' vs baseline':'')+'</div>';
    if((sel.why||'').length>260||sel.citation)
      h+='<details><summary>full journal entry</summary><div class="full">'
        +esc(sel.why||n.summary||'')+(sel.citation?'\n\ncitations: '+esc(sel.citation):'')+'</div></details>';
    h+='</div>';
    box.insertAdjacentHTML('beforeend',h);
  });
  // final entry: convergence
  box.insertAdjacentHTML('beforeend','<div class="entry" data-i="conv">'
    +'<span class="stamp acc">CONVERGED</span><h3>STOP</h3>'
    +'<div>Three consecutive iterations inside <b>ε = 0.002</b> — the official rule fires. '
    +'Final: <span class="go">'+BEST.toFixed(6)+'</span>, +'+(BEST-BASELINE).toFixed(4)
    +' over the baseline. No human said stop.</div></div>');
})();

/* typewriter */
function typeIn(el){
  const full=el.dataset.full||'';if(el.dataset.done)return;el.dataset.done=1;
  let k=0;const t=setInterval(()=>{
    k+=3;el.innerHTML=full.slice(0,k);
    if(k>=full.length){clearInterval(t);el.closest('.quote').classList.add('done');}
  },12);
}

/* scroll activation */
let bestSoFar=BASELINE;
const io=new IntersectionObserver(es=>es.forEach(e=>{
  if(!e.isIntersecting)return;
  const el=e.target;el.classList.add('onn');
  const i=el.dataset.i;
  el.querySelectorAll('.typed').forEach(typeIn);
  if(i==='conv'){$('#epsband').classList.add('onn');traceUpTo(N.length);$('#barscore').textContent=BEST.toFixed(4);return;}
  const k=+i,n=N[k];
  const nd=document.getElementById('nd'+k);if(nd)nd.classList.add('onn');
  traceUpTo(k);
  if(n&&n.accepted&&n.primary!=null)bestSoFar=Math.max(bestSoFar,n.primary);
  $('#barscore').textContent=bestSoFar.toFixed(4);
}),{threshold:0.45});
document.querySelectorAll('.entry').forEach(el=>io.observe(el));

/* ---------- receipts ------------------------------------------------------ */
(function receipts(){
  const wall=M.wall_s?Math.round(M.wall_s/60)+' min':'—';
  const cells=[
    [BEST.toFixed(4),'final valid primary'],
    ['+'+(BEST-BASELINE).toFixed(4),'gain vs published baseline'],
    [String(M.iterations||6),'iterations to convergence'],
    [wall,'agent wall-clock'],
    [((M.tokens||0)/1000).toFixed(0)+'k','LLM tokens'],
    ['0','manual interventions'],
  ];
  $('#rgrid').innerHTML=cells.map(c=>'<div class="cell"><div class="v">'+c[0]
    +'</div><div class="k">'+c[1]+'</div></div>').join('');
})();


/* ---------- evidence: rare-video memorization, baseline vs treated -------- */
(function evidence(){
  const Wt=window.WEIGHTS;const grid=document.getElementById('evgrid');
  if(!grid)return;
  if(!Wt){grid.innerHTML='<p class="lede">(instrumented data missing — run tools/instrument_weights.py)</p>';return;}
  const panels=[['baseline','BASELINE','memorizes what it barely saw'],
                ['treated','AGENT-TREATED','rare videos stay grounded']];
  const PW=520,PH=300,P=40;
  grid.innerHTML=panels.map(([key,title,subtitle])=>{
    const snaps=Wt[key];
    const maxN=Math.max(...Wt.baseline.concat(Wt.treated).flatMap(s=>s.norms));
    const xs=i=>P+i/(snaps.length-1)*(PW-2*P);
    const ys=v=>PH-P-(v/maxN)*(PH-2*P);
    let h='<div class="evpanel"><div class="t"><b>'+title+'</b> · '+subtitle+'</div>'
      +'<svg viewBox="0 0 '+PW+' '+PH+'">';
    h+='<line class="axis" x1="'+P+'" y1="'+(PH-P)+'" x2="'+(PW-P)+'" y2="'+(PH-P)+'"/>';
    h+='<text class="axistext" x="'+P+'" y="'+(PH-P+18)+'">training →</text>';
    h+='<text class="axistext" x="'+P+'" y="'+(P-10)+'">embedding size</text>';
    for(let d=9;d>=0;d--){
      const pts=snaps.map((s,i)=>xs(i)+','+ys(s.norms[d])).join(' ');
      const rare=d<=2;
      h+='<polyline class="evline" data-key="'+key+'" points="'+pts+'" stroke="'
        +(rare?'var(--no)':'var(--faint)')+'" stroke-width="'+(rare?2:1.2)+'" opacity="'+(rare?0.95:0.8)+'"/>';
    }
    const last=snaps[snaps.length-1];
    if(key==='baseline'){
      h+='<text class="evnote" x="'+(PW-P-6)+'" y="'+(ys(Math.max(...last.norms.slice(0,3)))-14)
        +'" text-anchor="end">rare videos ballooning ↑</text>';
    }else{
      h+='<text class="evnote" x="'+(PW-P)+'" y="'+(ys(Math.max(...last.norms.slice(0,3)))-10)
        +'" text-anchor="end" fill="var(--go)">held down by the treatments</text>';
    }
    h+='<text class="axistext" x="'+(PW-P)+'" y="'+(PH-P+18)+'" text-anchor="end">final valid '
      +last.primary.toFixed(4)+'</text>';
    h+='</svg></div>';
    return h;
  }).join('');
  // draw-in on first view
  const io2=new IntersectionObserver(es=>es.forEach(e=>{
    if(!e.isIntersecting)return;io2.disconnect();
    grid.querySelectorAll('.evline').forEach((ln,i)=>{
      const len=ln.getTotalLength();
      ln.style.strokeDasharray=len;ln.style.strokeDashoffset=len;
      setTimeout(()=>{ln.style.strokeDashoffset=0;},60*(i%12));
    });
  }),{threshold:0.3});
  io2.observe(grid);
})();
})();
