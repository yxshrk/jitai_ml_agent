(function(){
const D=window.RUNDATA, M=D.meta||{}, N=D.nodes||[];
const panel=document.getElementById('panel-body'), stepLabel=document.getElementById('step-label');
const ticker=document.getElementById('score'), strikeEl=document.getElementById('strikes');
const deckEl=document.getElementById('deck');
function showDeck(sel){
  if(!sel||!sel.chosen_method_id){deckEl.classList.remove('open');return;}
  const all=D.cards||[],chosen=sel.chosen_method_id,rej=(sel.rejected||[]).map(r=>r.method_id);
  const sample=all.filter(c=>c!==chosen&&!rej.includes(c)).slice(0,6);
  const ordr=[...rej.slice(0,3),chosen,...sample].slice(0,10);
  deckEl.innerHTML='<div class="deck-title">Treatment selection — from '+all.length+' cited method cards</div><div class="fan">'
    +ordr.map(c=>'<div class="'+(c===chosen?'card chosen':(rej.includes(c)?'card rejected':'card'))+'">'+c
    +(c===chosen?'<span class="pick">✓ chosen</span>':(rej.includes(c)?'<span class="rej">considered</span>':''))+'</div>').join('')
    +'</div>'+(sel.why?'<div class="deck-why">“'+String(sel.why).slice(0,220)+'”</div>':'');
  deckEl.classList.add('open');
}
// ---------- scene ----------
const canvas=document.getElementById('c');
const renderer=new THREE.WebGLRenderer({canvas,antialias:true});
const scene=new THREE.Scene(); scene.background=new THREE.Color(0x0b0e14);
const camera=new THREE.PerspectiveCamera(46,2,0.1,300);
scene.add(new THREE.AmbientLight(0x99a5c5,0.8));
const sun=new THREE.DirectionalLight(0xfff2dd,0.9); sun.position.set(8,18,10); scene.add(sun);
const camT=new THREE.Vector3(4,4.5,0);
// axes/floor of the "curve theater"
const XW=20, Y0=0.588, Y1=0.608, YS=9, ZGAP=2.6;
const yof=p=>Math.max(0,(p-Y0)/(Y1-Y0))*YS;
const grid=new THREE.GridHelper(46,23,0x1c2438,0x151b2c); grid.position.y=0; scene.add(grid);
// score gridlines + labels via sprites
function label(txt,x,y,z,size=1.1,color='#8b95ad',parent){
  const cv=document.createElement('canvas');cv.width=256;cv.height=64;
  const g=cv.getContext('2d');g.font='600 30px ui-monospace,monospace';g.fillStyle=color;g.fillText(txt,6,40);
  const t=new THREE.CanvasTexture(cv);
  const sp=new THREE.Sprite(new THREE.SpriteMaterial({map:t,transparent:true}));
  sp.scale.set(4*size,1*size,1); sp.position.set(x,y,z); (parent||scene).add(sp); return sp;
}
[0.59,0.595,0.60,0.605].forEach(v=>{
  const y=yof(v);
  const ln=new THREE.Line(new THREE.BufferGeometry().setFromPoints(
    [new THREE.Vector3(-XW/2,y,-14),new THREE.Vector3(-XW/2,y,6)]),
    new THREE.LineBasicMaterial({color:0x222b44}));
  scene.add(ln); label(v.toFixed(3),-XW/2-2.6,y,5,0.9);
});
// evidence bar (right side)
const barGroup=new THREE.Group(); scene.add(barGroup);
let barMesh=null, ball=null;
function setBar(champ,floor){
  while(barGroup.children.length)barGroup.remove(barGroup.children[0]);
  const y=yof(champ+floor);
  barMesh=new THREE.Mesh(new THREE.BoxGeometry(3.4,0.12,1.6),
    new THREE.MeshStandardMaterial({color:0xd8b24a,emissive:0xd8b24a,emissiveIntensity:0.25}));
  barMesh.position.set(XW/2+4,y,0); barGroup.add(barMesh);
  label('evidence bar',XW/2+4,y+0.9,0,0.85,'#d8b24a',barGroup);
}
const ribbons=new THREE.Group(); scene.add(ribbons);
let anims=[];
function animate(a){anims.push(a);}
function tick(dt){anims=anims.filter(a=>{a.t+=dt;const k=Math.min(1,a.t/a.dur);a.fn(k);return k<1;});}
function ribbonFrom(curve,z,color,ghost){
  const pts=curve.map((c,i)=>new THREE.Vector3(-XW/2+ (i/(Math.max(1,curve.length-1)))*XW, yof(c.p), z));
  const g=new THREE.BufferGeometry().setFromPoints(pts);
  const mat=new THREE.LineBasicMaterial({color,transparent:true,opacity:ghost?0.35:1,linewidth:2});
  const line=new THREE.Line(g,mat);
  line.geometry.setDrawRange(0,ghost?pts.length:0);
  ribbons.add(line);
  if(!ghost) animate({t:0,dur:2.2,fn:k=>line.geometry.setDrawRange(0,Math.floor(k*pts.length))});
  // sag marker: peak point
  let mi=0; curve.forEach((c,i)=>{if(c.p>curve[mi].p)mi=i;});
  if(!ghost&&curve.length>2){
    const peak=pts[mi];
    const dot=new THREE.Mesh(new THREE.SphereGeometry(0.22,10,10),
      new THREE.MeshStandardMaterial({color:0xffffff,emissive:0xffffff,emissiveIntensity:0.6}));
    dot.position.copy(peak); dot.visible=false; ribbons.add(dot);
    animate({t:0,dur:2.6,fn:k=>{if(k>0.85){dot.visible=true; dot.scale.setScalar(1+0.4*Math.sin(k*40));}}});
  }
  return {line,pts,peakIdx:mi};
}
function leapBall(fromY,toP,pass){
  if(ball)barGroup.remove(ball);
  ball=new THREE.Mesh(new THREE.SphereGeometry(0.45,14,14),
    new THREE.MeshStandardMaterial({color:pass?0x7fb069:0x8a5560,emissive:pass?0x7fb069:0x8a5560,emissiveIntensity:0.4}));
  const x=XW/2+4, y1=yof(toP);
  ball.position.set(x,0.4,2.6); barGroup.add(ball);
  animate({t:0,dur:1.6,fn:k=>{
    const up=Math.sin(Math.min(1,k*1.15)*Math.PI)* (y1+1.2);
    ball.position.set(x, 0.4+ (k<0.87? up : (pass? y1 : Math.max(0.4,(1-k)*8*y1/ y1))), 2.6-2.6*Math.min(1,k*1.3));
    if(k>=0.87){ ball.position.y = pass? y1+0.3 : Math.max(0.4, y1*(1-(k-0.87)/0.13)); }
  }});
}
// ---------- build consultation steps ----------
const fmtp=p=>p?p.toFixed(6):'—';
let strikes=0, score=(N[0]&&N[0].primary)||0.6018, champ=score, floor=0.0009, zi=0;
function setScore(v){score=v;}
const steps=[];
let champCurve=(N[0]&&N[0].curve)||[];
steps.push({label:'The patient',html:`<h3>The consultation loop</h3><p>Each iteration is one consultation: read the vitals (real learning curves), diagnose, choose a treatment from the card library, retrain, and face the evidence bar. Everything below is drawn from <b>${M.run}</b>'s actual logs.</p><p class="dim">Ribbons = validation score across training checkpoints. Depth = iterations. The gold ribbon is the reigning champion.</p>`,
  act:()=>{while(ribbons.children.length)ribbons.remove(ribbons.children[0]); zi=0; strikes=0; champ=N[0].primary; setScore(champ); setBar(champ,floor);
    if(champCurve.length)ribbonFrom(champCurve,0,0xd8b24a,false);}});
steps.push({label:'Diagnosis 0',html:`<h3>First vitals — the disease</h3><p>The baseline's own curve shows it: validation climbs, <b>peaks at epoch ~7-8, then falls</b> while training loss keeps improving — the model starts memorizing. The white pulse marks the peak the agent will keep diagnosing all run.</p><p><b>Champion: ${fmtp(champ)}</b>. The evidence bar (right) sits a noise-floor above it: to become champion, a candidate must clear it.</p>`,
  act:()=>{}});
N.forEach((n,ix)=>{
  if(n.id==='node_000')return;
  zi+=1; const z=-zi*ZGAP;
  if(n.error&&!n.primary){
    steps.push({label:`${n.id}: failure`,html:`<h3>${n.id} — the tooling fails (real, journaled)</h3><p>${n.summary}</p><p class="dim">No model was produced; the strike counter advances anyway. Robustness = surviving your own instruments.</p>`,
      act:()=>{strikes=Math.min(3,strikes+1);}});
    return;
  }
  // diagnosis + treatment
  const diag=(n.summary.split('will')[0]||n.summary).slice(0,260);
  steps.push({selection:n.selection,label:`${n.id}: diagnose`,html:`<h3>${n.id} — diagnose & prescribe</h3><p><b>Diagnosis (verbatim):</b> ${diag}…</p>${n.selection?`<p><b>Treatment:</b> <span style="color:var(--gold)">${n.selection.chosen_method_id}</span> — see the card fan (left) for what it weighed.</p>`:''}${n.probes&&n.probes.length>2?`<p class="dim">This treatment is a search: ${n.probes.length} internal trials logged before committing.</p>`:''}`,
    act:()=>{}});
  // retrain
  steps.push({label:`${n.id}: retrain`,html:`<h3>${n.id} — retrain</h3><p>The new ribbon draws against the champion's ghost. Watch the shape: ${n.accepted?'it holds its peak higher':'does it hold its peak, or sag the same way?'}${n.members&&n.members.length>1?` (${n.members.length} seed members trained; validation selects the combination)`:''}.</p>`,
    act:()=>{ if(champCurve.length)ribbonFrom(champCurve,z+1.1,0xd8b24a,true);
      const cv=(n.curve&&n.curve.length>2)?n.curve:(n.members&&n.members.length?null:null);
      if(cv)ribbonFrom(cv,z,0x5b8dd9,false);
      else label((n.members&&n.members.length? n.members.length+' members trained':'(curve not logged)'),0,yof(n.primary||champ)+1.2,z,1.0,'#8b95ad',ribbons);}});
  // verdict + evidence bar + fixed/broken
  const passed=n.accepted;
  steps.push({label:`${n.id}: verdict`,html:`<h3>${n.id} — the evidence bar</h3><p>Scored <b>${fmtp(n.primary)}</b> vs champion ${fmtp(champ)} + floor.</p>${(n.fixed!=null)?`<p>Real per-pair audit: <span class="good">fixed ${n.fixed.toLocaleString()}</span> orderings, <span class="bad">broke ${n.broken.toLocaleString()}</span> — improvement is a narrow trade, not magic${(n.id==='node_006')?'; note the raw counts even net negative here, but GAUC weights users by positives and nDCG weights the top — the official metric improves. Counts are not the metric.':''}.</p>`:''}<p>${passed?'<span class="good">CLEARS — new champion.</span>':'<span class="bad">Falls short — rejected; strike recorded.</span>'}</p>`,
    act:()=>{ leapBall(0,n.primary||champ,passed);
      if(passed){ const gain=(n.primary-champ); strikes=gain>0.002?0:Math.min(3,strikes+1);
        champ=n.primary; setScore(champ); if(n.curve&&n.curve.length>2)champCurve=n.curve; setBar(champ,floor);
      } else strikes=Math.min(3,strikes+1);}});
});
steps.push({label:'Converged',html:`<h3>The rule stops the run</h3><p>Third consultation without a ≥0.002 gain — the official convergence rule ends it. Final champion <b>${fmtp(M.best)}</b>: ${M.iterations} decisions, ${Math.round((M.wall_s||0)/60)} minutes, ${(M.tokens||0).toLocaleString()} tokens, zero mid-run human actions.</p><p class="dim">Replayable end-to-end from ${M.run}/ in the public repo.</p>`,
  act:()=>{setScore(M.best||champ);}});
let cur=-1;
function go(i){ if(i<0||i>=steps.length)return; cur=i; const st=steps[i];
  panel.innerHTML=st.html; showDeck(st.selection||null); st.act();
  stepLabel.textContent=`${i+1} / ${steps.length}`;
  document.getElementById('prev').disabled=i===0;
  document.getElementById('next').disabled=i===steps.length-1;
}
document.getElementById('next').onclick=()=>go(cur+1);
document.getElementById('prev').onclick=()=>go(cur-1);
addEventListener('keydown',e=>{if(e.key==='ArrowRight')go(cur+1);if(e.key==='ArrowLeft')go(cur-1);});
// camera orbit
let theta=0.35,phi=1.15,radius=30,drag=false,px=0,py=0,auto=true;
canvas.addEventListener('pointerdown',e=>{drag=true;auto=false;px=e.clientX;py=e.clientY;});
addEventListener('pointerup',()=>drag=false);
addEventListener('pointermove',e=>{if(!drag)return;theta-=(e.clientX-px)*0.005;phi=Math.max(0.35,Math.min(1.5,phi-(e.clientY-py)*0.004));px=e.clientX;py=e.clientY;});
canvas.addEventListener('wheel',e=>{radius=Math.max(14,Math.min(60,radius+e.deltaY*0.02));e.preventDefault();},{passive:false});
let last=performance.now(),shown=score;
function loop(now){const dt=(now-last)/1000;last=now;
  if(auto)theta+=dt*0.03;
  camera.position.set(camT.x+radius*Math.sin(phi)*Math.cos(theta),camT.y+radius*Math.cos(phi),camT.z+radius*Math.sin(phi)*Math.sin(theta));
  camera.lookAt(camT); tick(dt);
  shown+=(score-shown)*Math.min(1,dt*3); ticker.textContent=shown.toFixed(6);
  strikeEl.textContent='✕'.repeat(strikes)+'·'.repeat(Math.max(0,3-strikes));
  const w=canvas.clientWidth,h=canvas.clientHeight;
  if(canvas.width!==w||canvas.height!==h){renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix();}
  renderer.render(scene,camera); requestAnimationFrame(loop);
}
go(0); requestAnimationFrame(loop);
})();
