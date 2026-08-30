/* Flight Recorder — mle-agent scrollytelling site.
   2D mission-log replay of the designated run (rundata.js) with one 3D garnish:
   the scroll-morphed embedding starscape (space.js). No frameworks. */
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
  const svg=$('#herochart'), W=760, Hh=190, P=34;
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
    h+='<circle r="4" cx="'+xs(i)+'" cy="'+ys(n.primary)+'" fill="'
      +(n.accepted?css('--go'):css('--no'))+'" opacity="'+(n.accepted?1:.65)+'"/>';});
  h+='<text class="axistext" x="'+xs(acc[acc.length-1].i)+'" y="'
    +(ys(BEST)-10)+'" text-anchor="end" fill="'+css('--go')+'">'+BEST.toFixed(4)+'</text>';
  svg.innerHTML=h;
  const line=$('#heroline'), len=line.getTotalLength();
  line.style.strokeDasharray=len;line.style.strokeDashoffset=len;
  requestAnimationFrame(()=>{line.style.transition='stroke-dashoffset 2.4s ease .4s';
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

/* ---------- starscape (scroll-morphed) ----------------------------------- */
(function space(){
  const SP=window.SPACE;const canvas=$('#spacecanvas');
  if(!SP||!window.THREE){$('#spacetall').style.height='auto';
    $('#spacecaption').textContent='(embedding visual unavailable — run tools/build_space.py)';return;}
  const n=SP.meta.n,dec=SP.decile;
  const renderer=new THREE.WebGLRenderer({canvas,antialias:true});
  renderer.setPixelRatio(Math.min(2,window.devicePixelRatio||1));
  const scene=new THREE.Scene();scene.background=new THREE.Color(css('--bg'));
  const camera=new THREE.PerspectiveCamera(50,2,0.1,400);
  const R=30;
  const A=new Float32Array(n*3),B=new Float32Array(n*3),pos=new Float32Array(n*3);
  for(let i=0;i<n;i++)for(let a=0;a<3;a++){
    A[i*3+a]=(SP.base[i][a]/1000-0.5)*R;B[i*3+a]=(SP.treat[i][a]/1000-0.5)*R;pos[i*3+a]=A[i*3+a];
  }
  const geo=new THREE.BufferGeometry();
  geo.setAttribute('position',new THREE.BufferAttribute(pos,3));
  const col=new Float32Array(n*3);
  const cGo=new THREE.Color(css('--go')),cNo=new THREE.Color(css('--no')),cDim=new THREE.Color(css('--dim'));
  for(let i=0;i<n;i++){
    const c=dec[i]<=2?cNo.clone().lerp(cDim,0.25):cDim.clone().lerp(cGo,dec[i]/9*0.7);
    col[i*3]=c.r;col[i*3+1]=c.g;col[i*3+2]=c.b;
  }
  geo.setAttribute('color',new THREE.BufferAttribute(col,3));
  const dotTex=(function(){const cv=document.createElement('canvas');cv.width=cv.height=64;
    const g=cv.getContext('2d');g.beginPath();g.arc(32,32,26,0,7);g.fillStyle='#fff';g.fill();
    return new THREE.CanvasTexture(cv);})();
  scene.add(new THREE.Points(geo,new THREE.PointsMaterial({size:0.32,map:dotTex,
    vertexColors:true,transparent:true,opacity:0.85,alphaTest:0.4})));
  let morph=-1;
  const tall=$('#spacetall'),cap=$('#spacecaption'),lbl=$('#morphlbl');
  const CAPS=[[0,'the baseline\'s space — memorized red outliers everywhere (score 0.6015)'],
              [0.45,'the agent\'s treatments applying: regularization + recency + ensemble…'],
              [0.85,'the champion\'s space — tighter, structured, rare videos reined in (score 0.6045)']];
  function setMorph(m){
    if(Math.abs(m-morph)<0.004)return;morph=m;
    for(let i=0;i<n*3;i++)pos[i]=A[i]+(B[i]-A[i])*m;
    geo.attributes.position.needsUpdate=true;
    lbl.textContent='morph '+(m*100).toFixed(0)+'%';
    cap.textContent=CAPS.filter(c=>m>=c[0]).pop()[1];
  }
  function resize(){const r=canvas.parentElement.getBoundingClientRect();
    renderer.setSize(r.width,r.height,false);camera.aspect=r.width/r.height;
    camera.updateProjectionMatrix();}
  addEventListener('resize',resize);resize();
  const reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;
  let theta=0;
  function frame(){
    const r=tall.getBoundingClientRect();
    const vis=r.top<innerHeight&&r.bottom>0;
    if(vis){
      const p=Math.min(1,Math.max(0,-r.top/(r.height-innerHeight)));
      setMorph(p);
      if(!reduced)theta+=0.0012;
      camera.position.set(Math.sin(theta)*54,10,Math.cos(theta)*54);
      camera.lookAt(0,0,0);
      renderer.render(scene,camera);
    }
    requestAnimationFrame(frame);
  }
  setMorph(0);requestAnimationFrame(frame);
})();
})();
