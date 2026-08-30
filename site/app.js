(function(){
const D = window.RUNDATA;
const panel = document.getElementById('panel-body');
const stepLabel = document.getElementById('step-label');
const ticker = document.getElementById('score');
const strikeEl = document.getElementById('strikes');

// ---------- three.js scene ----------
const canvas = document.getElementById('c');
const renderer = new THREE.WebGLRenderer({canvas, antialias:true});
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0e14);
scene.fog = new THREE.Fog(0x0b0e14, 30, 90);
const camera = new THREE.PerspectiveCamera(50, 2, 0.1, 200);
camera.position.set(14, 12, 20);
let camTarget = new THREE.Vector3(0, 2, 0);
scene.add(new THREE.AmbientLight(0x8899bb, 0.7));
const sun = new THREE.DirectionalLight(0xffeecc, 1.1); sun.position.set(10, 20, 8); scene.add(sun);

// terrain from corpus probes: x=dropout, z=gamma, y=primary
const X0=0.05, X1=0.45, Z0=0.3, Z1=1.0, Y0=0.585, Y1=0.606;
const W=22, DEP=22, GRID=56;
const nx = v => ( (v-X0)/(X1-X0) - 0.5) * W;
const nz = v => ( (v-Z0)/(Z1-Z0) - 0.5) * DEP;
const ny = p => Math.max(0, (p-Y0)/(Y1-Y0)) * 7.5;
const pts = D.corpus.filter(c=>c.g!=null && c.d>=X0 && c.d<=X1 && c.g>=Z0 && c.g<=Z1);
function idw(x,z){ // inverse-distance height from real probes
  let num=0, den=0;
  for(const c of pts){
    const dx=nx(c.d)-x, dz=nz(c.g)-z; const d2=dx*dx+dz*dz+0.35;
    const w=1/(d2*d2); num+=w*c.p; den+=w;
  }
  return den? num/den : Y0;
}
const geo = new THREE.PlaneGeometry(W, DEP, GRID, GRID);
geo.rotateX(-Math.PI/2);
const pos = geo.attributes.position;
const colors = [];
const cLow = new THREE.Color(0x233150), cMid = new THREE.Color(0x2f6f5f), cHigh = new THREE.Color(0xd8b24a);
for(let i=0;i<pos.count;i++){
  const p = idw(pos.getX(i), pos.getZ(i));
  pos.setY(i, ny(p));
  const t = Math.min(1, Math.max(0,(p-0.594)/(0.605-0.594)));
  const col = t<0.6? cLow.clone().lerp(cMid, t/0.6) : cMid.clone().lerp(cHigh,(t-0.6)/0.4);
  colors.push(col.r,col.g,col.b);
}
geo.setAttribute('color', new THREE.Float32BufferAttribute(colors,3));
geo.computeVertexNormals();
const terrain = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({vertexColors:true, roughness:0.85, metalness:0.05}));
scene.add(terrain);
const wire = new THREE.Mesh(geo.clone(), new THREE.MeshBasicMaterial({wireframe:true, color:0x36415e, transparent:true, opacity:0.25}));
wire.position.y += 0.02; scene.add(wire);

// groups for dynamic objects
const probeGroup = new THREE.Group(); scene.add(probeGroup);
const memberGroup = new THREE.Group(); memberGroup.visible=false; scene.add(memberGroup);
let flag=null;
function makeFlag(x,z,y,color){
  const g=new THREE.Group();
  const pole=new THREE.Mesh(new THREE.CylinderGeometry(0.05,0.05,2.4),new THREE.MeshStandardMaterial({color:0xffffff}));
  pole.position.y=1.2; g.add(pole);
  const cloth=new THREE.Mesh(new THREE.ConeGeometry(0.45,0.9,4),new THREE.MeshStandardMaterial({color}));
  cloth.rotation.z=-Math.PI/2; cloth.position.set(0.45,2.0,0); g.add(cloth);
  g.position.set(x,y,z); return g;
}
// animation queue
let anims=[];
function animate(obj){ anims.push(obj); }
function tick(dt){
  anims = anims.filter(a=>{ a.t+=dt; const k=Math.min(1,a.t/a.dur); a.fn(k); return k<1; });
}
function dropProbe(c, color, delay, big){
  const r = big?0.22:0.13;
  const m = new THREE.Mesh(new THREE.SphereGeometry(r, 12, 12),
    new THREE.MeshStandardMaterial({color, emissive:color, emissiveIntensity:0.35}));
  const x=nx(c.dropout??c.d), z=nz(c.gamma??c.g??0.6), y=ny(c.primary??c.p);
  m.position.set(x, 14, z); m.visible=false; probeGroup.add(m);
  animate({t:-delay, dur:0.7, fn:k=>{ if(k<=0)return; m.visible=true; m.position.y = 14 - (14-y-0.15)*(k*k); }});
  return m;
}
function clearGroup(g){ while(g.children.length) g.remove(g.children[0]); }

// ---------- steps generated from ANY run's data ----------
const M = D.meta||{}; const N = D.nodes||D.steps||[];
const deckEl = document.getElementById('deck');
function showDeck(sel){
  if(!sel||!sel.chosen_method_id){ deckEl.classList.remove('open'); return; }
  const all = D.cards||[]; const chosen = sel.chosen_method_id;
  const rej = (sel.rejected||[]).map(r=>r.method_id);
  const sample = all.filter(c=>c!==chosen && !rej.includes(c)).slice(0,7);
  const order = [...rej.slice(0,3), chosen, ...sample].slice(0,10);
  deckEl.innerHTML = '<div class="deck-title">The agent\'s selection — from a library of '+all.length+' cited method cards</div>'+
    '<div class="fan">'+order.map(c=>{
      const cls = c===chosen?'card chosen':(rej.includes(c)?'card rejected':'card');
      return '<div class="'+cls+'">'+c+(c===chosen?'<span class="pick">✓ chosen</span>':(rej.includes(c)?'<span class="rej">considered</span>':''))+'</div>';
    }).join('')+'</div>'+
    (sel.why?'<div class="deck-why">“'+String(sel.why).slice(0,220)+'”</div>':'');
  deckEl.classList.add('open');
}
const fmtp = p => p? p.toFixed(6) : '—';
let strikes = 0, score = (N[0]&&N[0].primary)||0.6018;
function setScore(v){ score=v; }
const steps=[];
steps.push({label:'The landscape', html:`<h3>The search landscape (real data)</h3><p>Terrain built from <b>${pts.length} logged probe trainings</b> across the whole campaign: dropout (x) × LR decay (z), height = validation score. The gold ridge is where winning configs live. The agent cannot see this map — it must feel for it.</p><p class="dim">Run shown: <b>${M.run||'—'}</b> · ${M.iterations||'?'} decisions · ${(M.wall_s?Math.round(M.wall_s/60):'?')} min · drag to orbit.</p>`,
  act:()=>{ clearGroup(probeGroup); clearGroup(memberGroup); memberGroup.visible=false; if(flag){scene.remove(flag);flag=null;} strikes=0; if(N[0]&&N[0].primary) setScore(N[0].primary);} });
let champ = (N[0]&&N[0].primary)||0.6018;
N.forEach((n,ix)=>{
  if(n.id==='node_000'){
    steps.push({label:'Baseline', html:`<h3>Reproduce the baseline</h3><p>${n.summary}</p><p>Champion starts at <b>${fmtp(n.primary)}</b>.</p>`,
      act:()=>{ if(flag)scene.remove(flag); flag=makeFlag(nx(0.2),nz(0.9),ny(n.primary),0x8899aa); scene.add(flag); setScore(n.primary);} });
    return;
  }
  const sweep = n.probes && n.probes.length>2;
  const ens = n.members && n.members.length>1;
  if(n.error && !n.primary){
    steps.push({label:`${n.id}: failed`, html:`<h3>${n.id} — execution failure (real, journaled)</h3><p>${n.summary}</p><p class="dim">Failures count toward the convergence streak — the run must survive its own tooling.</p>`,
      act:()=>{ strikes=Math.min(3,strikes+1);} });
    return;
  }
  if(ens){
    const sel = n.selected_ensemble||{};
    steps.push({selection:n.selection, label:`${n.id}: ensemble`, html:`<h3>${n.id} — designing an ensemble</h3><p>${n.summary.slice(0,260)}…</p><p>Trains <b>${n.members.length} seed variants</b> (columns = real member scores)${sel.member_count?`; validation selects <b>${sel.member_count}</b>, combined by ${String(sel.combination_rule||'').replace(/_/g,' ')}`:''}.</p>`,
      act:()=>{ memberGroup.visible=true; clearGroup(memberGroup);
        const selSeeds=(n.selected_ensemble&&n.selected_ensemble.seeds)||[];
        n.members.forEach((m,i)=>{ const h=(m.primary-0.5988)*220; const on=selSeeds.includes(m.seed);
          const col=new THREE.Mesh(new THREE.BoxGeometry(0.8,0.1,0.8),
            new THREE.MeshStandardMaterial({color:on?0xd8b24a:0x51608a,transparent:!on,opacity:on?1:0.55}));
          col.position.set(-6+i*2,6.2,-13); memberGroup.add(col);
          animate({t:-i*0.2,dur:0.9,fn:k=>{if(k<=0)return; col.scale.y=1+k*(h*10); col.position.y=6.2+k*h*0.05;}});
        }); } });
  } else if(sweep){
    const coarse=n.probes.filter(p=>/coarse/.test(p.stage));
    const refine=n.probes.filter(p=>/refine/.test(p.stage));
    const fin=n.probes.find(p=>/final/.test(p.stage))||n.probes[n.probes.length-1];
    steps.push({selection:n.selection, label:`${n.id}: sweep`, html:`<h3>${n.id} — a search inside one decision</h3><p><b>Hypothesis:</b> ${n.summary.slice(0,260)}…</p><p><b>${coarse.length||n.probes.length} coarse</b> probes scatter${refine.length?`, then <b>${refine.length} refinement</b> probes climb the ridge`:''} — every dot a real training from this node's probe log.</p>`,
      act:()=>{ (coarse.length?coarse:n.probes).forEach((p,i)=>dropProbe({dropout:p.d,gamma:p.g,primary:p.p},0x5b8dd9,i*0.22));
                refine.forEach((p,i)=>dropProbe({dropout:p.d,gamma:p.g,primary:p.p},0x7fb069,1.6+i*0.28)); } });
  }
  const verdict = n.accepted
    ? `<span class="good">ACCEPTED</span> — new champion <b>${fmtp(n.primary)}</b>`
    : `<span class="bad">rejected</span> at ${fmtp(n.primary)} (evidence bar held)`;
  steps.push({label:`${n.id}: ${n.accepted?'accepted':'rejected'}`,
    html:`<h3>${n.id} — verdict</h3><p>${sweep||ens?'The committed candidate is scored by the official evaluator and judged against the noise floor.':n.summary.slice(0,220)+'…'}</p><p>${verdict}</p>`,
    act:()=>{ if(n.accepted){ strikes=(n.primary-champ)>0.002?0:Math.min(3,strikes+1); champ=n.primary; setScore(n.primary);
        const fp=(n.probes&&n.probes.length)?(n.probes.find(p=>/final/.test(p.stage))||n.probes[n.probes.length-1]):null;
        if(fp){ if(flag)scene.remove(flag); flag=makeFlag(nx(fp.d),nz(fp.g),ny(n.primary),0xd8b24a); scene.add(flag); }
        else if(flag){ animate({t:0,dur:1.0,fn:k=>{flag.position.y=ny(champ)+k*0.4;}}); }
      } else { strikes=Math.min(3,strikes+1); if(n.probes&&n.probes[0]) dropProbe({dropout:n.probes[0].d,gamma:n.probes[0].g,primary:n.primary},0x8a5560,0); } } });
});
steps.push({label:'Converged', html:`<h3>Stopped by the rule</h3><p>Final champion <b>${fmtp(M.best)}</b> after ${M.iterations} decisions, ${(M.wall_s?Math.round(M.wall_s/60):'?')} minutes, ${(M.tokens||0).toLocaleString()} LLM tokens, zero mid-run human actions.</p><p class="dim">Every frame of this walkthrough is replayable from ${M.run}/ in the public repo.</p>`,
  act:()=>{ setScore(M.best||champ); } });
let cur=-1;
function go(i){
  if(i<0||i>=steps.length) return;
  cur=i; const st=steps[i];
  panel.innerHTML = st.html; showDeck(st.selection||null); st.act();
  stepLabel.textContent = `${i+1} / ${steps.length}`;
  document.getElementById('prev').disabled = i===0;
  document.getElementById('next').disabled = i===steps.length-1;
}
document.getElementById('next').onclick=()=>go(cur+1);
document.getElementById('prev').onclick=()=>go(cur-1);
addEventListener('keydown',e=>{ if(e.key==='ArrowRight')go(cur+1); if(e.key==='ArrowLeft')go(cur-1); });

// orbit (drag) + slow auto-rotate
let theta=0.6, phi=1.05, radius=26, drag=false, px=0, pyy=0, auto=true;
canvas.addEventListener('pointerdown',e=>{drag=true;auto=false;px=e.clientX;pyy=e.clientY;});
addEventListener('pointerup',()=>drag=false);
addEventListener('pointermove',e=>{ if(!drag)return; theta-=(e.clientX-px)*0.005; phi=Math.max(0.3,Math.min(1.45,phi-(e.clientY-pyy)*0.004)); px=e.clientX;pyy=e.clientY;});
canvas.addEventListener('wheel',e=>{ radius=Math.max(12,Math.min(50,radius+e.deltaY*0.02)); e.preventDefault();},{passive:false});

let last=performance.now(), shown=0.601838;
function loop(now){
  const dt=(now-last)/1000; last=now;
  if(auto) theta+=dt*0.05;
  camera.position.set(camTarget.x+radius*Math.sin(phi)*Math.cos(theta),
                      camTarget.y+radius*Math.cos(phi),
                      camTarget.z+radius*Math.sin(phi)*Math.sin(theta));
  camera.lookAt(camTarget);
  tick(dt);
  shown += (score-shown)*Math.min(1,dt*3);
  ticker.textContent = shown.toFixed(6);
  strikeEl.textContent = '✕'.repeat(strikes) + '·'.repeat(Math.max(0,3-strikes));
  const w=canvas.clientWidth, h=canvas.clientHeight;
  if(canvas.width!==w||canvas.height!==h){ renderer.setSize(w,h,false); camera.aspect=w/h; camera.updateProjectionMatrix(); }
  renderer.render(scene,camera);
  requestAnimationFrame(loop);
}
go(0); requestAnimationFrame(loop);
})();
