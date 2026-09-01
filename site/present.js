/* Presenter mode driver. All rendering comes from app.js (window.FR hooks);
   this file only sequences scenes and drives the log replay by keypress. */
(function(){
'use strict';
const FR=window.FR;
const scenes=[...document.querySelectorAll('.scene')];
const titles=scenes.map(s=>s.dataset.title||s.id);
const pcap=document.getElementById('pcap'),pnum=document.getElementById('pnum');
const overview=document.getElementById('overview');
const logScene=scenes.findIndex(s=>s.id==='s-log');
const runScene=scenes.findIndex(s=>s.id==='s-run');
const loopScene=scenes.findIndex(s=>s.id==='s-loop');
const archScene=scenes.findIndex(s=>s.id==='s-arch');
const ARCH_STEPS=4;
let arch=0; // lit part of the architecture diagram while inside that scene
const methodsScene=scenes.findIndex(s=>s.id==='s-methods');
const METHOD_FILTER='ensemble'; // pre-typed glimpse: one slice of the 42 cards, no table scroll on video
const STAGES=4;
let stage=0; // lit stage card while inside the loop scene
/* log substeps: iterations 0..N-1 then convergence */
const steps=[...Array(FR.iters).keys()].map(String).concat('conv');
let cur=0, step=-1; // step = index into steps while inside the log scene

function showEntry(){
  document.querySelectorAll('#s-log .entry').forEach(el=>{
    const on=el.dataset.i===steps[step];
    el.classList.toggle('pcur',on);
    if(on)el.querySelectorAll('.typed').forEach(FR.typeIn);
  });
}
function setLog(k){ // sync app.js replay state to substep k
  step=k;
  FR.active.clear();
  for(let j=0;j<=k;j++)FR.active.add(steps[j]);
  FR.repaint();
  showEntry();
}
function show(i,fromLeft){
  cur=Math.max(0,Math.min(scenes.length-1,i));
  scenes.forEach((s,j)=>s.classList.toggle('cur',j===cur));
  pcap.innerHTML=scenes[cur].dataset.cap||'';
  pnum.textContent=(cur+1)+' / '+scenes.length;
  if(cur===runScene)FR.heroPlay();
  if(cur===logScene)setLog(fromLeft?steps.length-1:0);
  if(cur===loopScene)setStage(fromLeft?STAGES-1:0);
  if(cur===archScene)setArch(fromLeft?ARCH_STEPS-1:0);
  if(cur===methodsScene)presetFilter();
}
function presetFilter(){
  const inp=document.getElementById('msearch');if(!inp||inp.value)return;
  inp.value=METHOD_FILTER;inp.dispatchEvent(new Event('input'));
}
function setStage(k){stage=k;FR.setStage(k);}
function setArch(k){arch=k;FR.setArch(k);}
function next(){
  if(cur===logScene&&step<steps.length-1)return setLog(step+1);
  if(cur===loopScene&&stage<STAGES-1)return setStage(stage+1);
  if(cur===archScene&&arch<ARCH_STEPS-1)return setArch(arch+1);
  if(cur<scenes.length-1)show(cur+1);
}
function prev(){
  if(cur===logScene&&step>0)return setLog(step-1);
  if(cur===loopScene&&stage>0)return setStage(stage-1);
  if(cur===archScene&&arch>0)return setArch(arch-1);
  if(cur>0)show(cur-1,true);
}
/* overview */
overview.querySelector('.grid').innerHTML=titles.map((t,i)=>
  '<button data-i="'+i+'"><span class="n">'+String(i+1).padStart(2,'0')+'</span><b>'+t+'</b></button>').join('');
function toggleOverview(on){
  overview.classList.toggle('on',on??!overview.classList.contains('on'));
  overview.querySelectorAll('button').forEach(b=>b.classList.toggle('cur',+b.dataset.i===cur));
}
overview.addEventListener('click',e=>{
  const b=e.target.closest('button');
  if(b){show(+b.dataset.i);toggleOverview(false);}
  else toggleOverview(false);
});
/* keyboard: reliable on stage; never steal keys from the search field */
addEventListener('keydown',e=>{
  if(e.key==='Escape'){toggleOverview();e.preventDefault();return;}
  if(e.target.tagName==='INPUT')return;
  if(overview.classList.contains('on'))return;
  if(['ArrowRight',' ','PageDown','Enter'].includes(e.key)){next();e.preventDefault();}
  else if(['ArrowLeft','PageUp','Backspace'].includes(e.key)){prev();e.preventDefault();}
  else if(e.key==='Home')show(0);
  else if(e.key==='End')show(scenes.length-1);
  else if(/^[1-9]$/.test(e.key)&&+e.key<=scenes.length)show(+e.key-1);
  else if(e.key==='0'&&scenes.length>=10)show(9);
});
/* click advances, except on interactive elements */
addEventListener('click',e=>{
  if(overview.classList.contains('on'))return;
  if(e.target.closest('input,a,button,details,summary'))return;
  next();
});
show(0);
})();
