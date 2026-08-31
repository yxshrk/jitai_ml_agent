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
  N.forEach((n,i)=>{if(n.primary==null)return;
    h+='<circle r="6" cx="'+xs(i)+'" cy="'+ys(n.primary)+'" fill="'
      +(n.accepted?css('--go'):css('--no'))+'" opacity="'+(n.accepted?1:.65)+'"/>';});
  h+='<text class="axistext" x="'+xs(acc[acc.length-1].i)+'" y="'
    +(ys(BEST)-10)+'" text-anchor="end" fill="'+css('--go')+'">'+BEST.toFixed(4)+'</text>';
  svg.innerHTML=h;
  // size the chart to exactly fill the viewport space left under the text,
  // so nothing below the hero peeks and nothing gets clipped
  const fitHero=()=>{
    svg.style.maxWidth='';
    const hint=document.querySelector('.scrollhint');
    const avail=innerHeight-svg.getBoundingClientRect().top
      -(hint?hint.getBoundingClientRect().height:60)-10;
    const w=Math.min(svg.parentElement.clientWidth,avail*(W/Hh));
    svg.style.maxWidth=Math.max(420,Math.floor(w))+'px';
  };
  fitHero();addEventListener('resize',fitHero);
  const line=$('#heroline'), len=line.getTotalLength();
  line.style.strokeDasharray=len;line.style.strokeDashoffset=len;
  requestAnimationFrame(()=>{line.style.transition='stroke-dashoffset 2.4s ease .4s';line.style.strokeWidth=3;
    line.style.strokeDashoffset=0;});
})();

/* ---------- loop: cycle the stage highlight ------------------------------ */
(function loop(){
  const cards=document.querySelectorAll('.stage');if(!cards.length)return;
  let t=0;cards[0].classList.add('live');
  setInterval(()=>{cards[t].classList.remove('live');t=(t+1)%cards.length;
    cards[t].classList.add('live');},2200);
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
      h+='<g id="nd'+i+'" class="dead"><line x1="'+(cx(i)-6)+'" y1="'+(cy(CH.S0)+0)+'" x2="'
        +(cx(i)+6)+'" y2="'+(cy(CH.S0)-12)+'" /><line x1="'+(cx(i)+6)+'" y1="'+cy(CH.S0)
        +'" x2="'+(cx(i)-6)+'" y2="'+(cy(CH.S0)-12)+'"/></g>';
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
  el.style.strokeDasharray=traceTotal;el.style.strokeDashoffset=traceTotal;
}
function traceUpTo(k){ // reveal the accepted line through iteration k
  if(traceLens==null)initTrace();
  let len=0;
  Object.keys(traceLens).forEach(i=>{if(+i<=k)len=Math.max(len,traceLens[i]);});
  $('#bigtrace').style.strokeDashoffset=traceTotal-len;
}

/* pin the chart so it locks vertically centered in the viewport, while its
   resting position stays snug under the section heading */
(function pinCenter(){
  const pin=document.getElementById('chartpin');if(!pin)return;
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
  if(n.primary==null)return 'Node crashed mid-run. Logged as <span class="amber">VOID</span>, loop continues.';
  const m='tries <b>'+esc(nice(sel.chosen_method_id||'a new package'))+'</b>';
  return n.accepted
    ?m+' → <span class="go">'+n.primary.toFixed(4)+'</span>. Cleared the bar.'
    :m+' → <span class="no">'+n.primary.toFixed(4)+'</span>. Below ε. Dead end.';
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
    +'<div>Three consecutive iterations inside <b>ε = 0.002</b>, so the official rule fires. '
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
  $('#barscore').textContent=best.toFixed(4);
  // label the iteration the reader is currently on
  const pt=$('#curpt'),lb=$('#curlbl');
  let cur=conv?-1:maxI;
  if(cur>=0&&N[cur]){
    const n=N[cur],y=n.primary!=null?cy(n.primary):cy(CH.S0)-6; // VOID cross is drawn 12px tall above the axis; ring centers on it
    pt.setAttribute('cx',cx(cur));pt.setAttribute('cy',y);pt.style.opacity=.9;
    lb.setAttribute('x',cx(cur));lb.setAttribute('y',y-24);
    lb.textContent='ITER '+String(cur).padStart(2,'0')+(n.primary!=null?' · '+n.primary.toFixed(4):' · VOID');
    lb.setAttribute('opacity',1);
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
}),{rootMargin:'0px 0px -45% 0px',threshold:[0.05,0.5]});
document.querySelectorAll('.entry').forEach(el=>io.observe(el));

/* hover an entry: highlight its point on the chart */
document.querySelectorAll('.entry').forEach(el=>{
  const i=el.dataset.i;if(i==null||i==='conv')return;
  const marks=()=>[document.getElementById('nd'+i),document.getElementById('ilbl'+i)].filter(Boolean);
  el.addEventListener('mouseenter',()=>marks().forEach(m=>m.classList.add('hl')));
  el.addEventListener('mouseleave',()=>marks().forEach(m=>m.classList.remove('hl')));
});

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
  // draw-in on first view
  const io2=new IntersectionObserver(es=>es.forEach(e=>{
    if(!e.isIntersecting)return;io2.disconnect();
    grid.querySelectorAll('.evline').forEach((ln,i)=>{
      const len=ln.getTotalLength();
      ln.style.strokeDasharray=len;ln.style.strokeDashoffset=len;
      setTimeout(()=>{ln.style.strokeDashoffset=0;},60*(i%12));
    });
  }),{threshold:0.65});
  io2.observe(grid);
})();
})();
