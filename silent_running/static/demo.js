// ?demo=1: replays a scripted bedside conversation as fake server events (same contract as /ws), with a synthetic
// landmark "camera" on a canvas. Lets the UI be designed and presented with no backend, camera or API keys.
// Serve without the backend:  python -m http.server -d silent_running 8765  ->  http://localhost:8765/static/index.html?demo=1
// Keys in demo: space = mouth a phrase (manual), → / ← = next / previous scene, P = pause, Y / N = answer "sounds like".
(()=>{
if(!DEMO)return;
const now=()=>Date.now()/1000;
const SKIP={};
let gen=0,paused=false,idx=0,uid=0,manual=false,answer=null;
const history=[];let prompt_='';

// ---------------------------------------------------------------- synthetic camera
const cv=$('#democam'),g=cv.getContext('2d'),W=640,H=480;
$('#camwrap').classList.add('demo');
const F={talk:false,nod:-9,shake:-9,blinks:[],fingers:null,handT:0,grimace:0,listening:false,listenT:0};
let nextBlink=performance.now()/1000+2;
function faceGeom(t){
  const nd=t-F.nod,sk=t-F.shake;
  let dx=Math.sin(t*.7)*3+Math.sin(t*2.3)*1.2,dy=Math.sin(t*.9+1)*2.5;  // head-mounted camera sway
  if(nd<2.2)dy+=18*Math.sin(nd*2*Math.PI*1.6)*Math.exp(-nd*.9);
  if(sk<2.2)dx+=24*Math.sin(sk*2*Math.PI*1.6)*Math.exp(-sk*.9);
  let open=.06;
  if(F.talk)open=.2+.8*Math.abs(Math.sin(t*6.1)*Math.sin(t*2.7+1.3));
  let eye=1;
  if(t>nextBlink){F.blinks.push(t);nextBlink=t+3+Math.random()*3;}
  for(const b of F.blinks){const d=t-b;if(d>=0&&d<.16)eye=Math.min(eye,Math.abs(d-.08)/.08);}
  F.blinks=F.blinks.filter(b=>t-b<1);
  const gr=F.grimace;eye*=1-.55*gr;
  return {cx:320+dx,cy:220+dy,open,eye,gr};
}
function dot(x,y,r,c){g.fillStyle=c;g.beginPath();g.arc(x,y,r,0,7);g.fill();}
function ring(pts,c,close=true){g.strokeStyle=c;g.beginPath();pts.forEach(([x,y],i)=>i?g.lineTo(x,y):g.moveTo(x,y));if(close)g.closePath();g.stroke();}
function ell(cx,cy,rx,ry,n,a0=0,a1=Math.PI*2){const o=[];for(let i=0;i<n;i++){const a=a0+(a1-a0)*i/(a1-a0>=6.28?n:n-1);o.push([cx+rx*Math.cos(a),cy+ry*Math.sin(a)]);}return o;}
function drawHand(x,y,n,rise){
  y+=(1-rise)*220;const C='rgba(177,140,255,.9)';g.lineWidth=3;g.lineCap='round';
  g.fillStyle='rgba(40,34,60,.85)';g.beginPath();g.roundRect(x-34,y-10,68,78,18);g.fill();
  const base=[[-24,-8],[-8,-12],[8,-12],[24,-8]];
  base.forEach(([bx,by],i)=>{const ext=i<Math.min(n,4);const L=ext?58:14;const tx=x+bx+(i-1.5)*4*(ext?1:0),ty=y+by-L;
    g.strokeStyle=ext?C:'rgba(177,140,255,.35)';g.beginPath();g.moveTo(x+bx,y+by);g.lineTo(tx,ty);g.stroke();dot(tx,ty,4,ext?C:'rgba(177,140,255,.4)');dot(x+bx,y+by,3,'rgba(177,140,255,.6)');});
  const thumb=n>=5;const [tx,ty]=thumb?[x-66,y-20]:[x-44,y+14];
  g.strokeStyle=thumb?C:'rgba(177,140,255,.35)';g.beginPath();g.moveTo(x-30,y+30);g.lineTo(tx,ty);g.stroke();dot(tx,ty,4,C);dot(x,y+56,4,C);
}
function draw(){
  const t=performance.now()/1000,{cx,cy,open,eye,gr}=faceGeom(t);
  const bg=g.createRadialGradient(320,200,40,320,240,420);bg.addColorStop(0,'#14212e');bg.addColorStop(1,'#05080c');g.fillStyle=bg;g.fillRect(0,0,W,H);
  g.strokeStyle='rgba(90,169,255,.05)';g.lineWidth=1;for(let x=0;x<W;x+=32){g.beginPath();g.moveTo(x,0);g.lineTo(x,H);g.stroke();}for(let y=0;y<H;y+=32){g.beginPath();g.moveTo(0,y);g.lineTo(W,y);g.stroke();}
  // shoulders + head silhouette
  g.fillStyle='#101a25';g.beginPath();g.ellipse(cx,cy+300,230,130,0,Math.PI,0);g.fill();
  g.fillStyle='#1a2a3a';g.beginPath();g.ellipse(cx,cy,108,138,0,0,7);g.fill();
  g.fillStyle='#16242f';g.fillRect(cx-40,cy+120,80,60);
  const M='rgba(160,215,255,.55)',S='rgba(160,215,255,.22)';g.lineWidth=1;
  const oval=ell(cx,cy,104,134,40);ring(oval,S);oval.forEach(([x,y])=>dot(x,y,1.8,M));
  // brows (pull down + in with grimace)
  [-1,1].forEach(s=>{const b=ell(cx+s*42-s*gr*4,cy-52+gr*9,26,8,6,Math.PI*1.1,Math.PI*1.9).map(([x,y],i)=>[x,y+(s*(i-2.5))*gr*1.8]);ring(b,S,false);b.forEach(([x,y])=>dot(x,y,1.8,M));
    const e=ell(cx+s*42,cy-24,20,Math.max(1,9*eye),12);ring(e,S);e.forEach(([x,y])=>dot(x,y,1.6,M));if(eye>.3)dot(cx+s*42,cy-24,3.5,'rgba(160,215,255,.8)');});
  // nose
  [[0,-18],[0,-6],[0,6],[0,18],[-12,26],[0,30],[12,26]].forEach(([x,y])=>dot(cx+x,cy+y,1.8,M));
  // lips
  const my=cy+66,mw=46-open*6,oh=7+open*22,L=F.listening?'rgba(90,169,255,.95)':'rgba(61,220,151,.9)';
  const outer=[];for(let i=0;i<20;i++){const a=Math.PI*2*i/20;outer.push([cx+mw*Math.cos(a),my+(Math.sin(a)>0?oh:(oh*.55+3)*1.0)*Math.sin(a)-(Math.abs(Math.cos(a))<.2&&Math.sin(a)<0?-3:0)]);}
  const inner=ell(cx,my+2,mw*.7,Math.max(1,open*17),14);
  g.fillStyle='rgba(0,0,0,.55)';g.beginPath();inner.forEach(([x,y],i)=>i?g.lineTo(x,y):g.moveTo(x,y));g.fill();
  g.lineWidth=1.5;ring(outer,L);ring(inner,L);outer.forEach(([x,y])=>dot(x,y,2.2,L));inner.forEach(([x,y])=>dot(x,y,1.6,L));
  // hands
  const hr=Math.min(1,(t-F.handT)/.6);
  if(F.fingers!=null){const n=F.fingers;if(n>5){drawHand(480,330,5,hr);drawHand(160,330,n-5,hr);}else drawHand(480,330,n,hr);}
  cv.mouthBox=[cx-mw-6,my-oh-8,cx+mw+6,my+oh+8];
  requestAnimationFrame(draw);
}
requestAnimationFrame(draw);
setInterval(()=>{const [a,b,c,d]=cv.mouthBox||[0,0,0,0];
  meta({face:true,fps:30,frame_w:W,frame_h:H,bbox:[W-c,b,W-a,d],face_frac:.16,listening:F.listening,n_frames:F.listening?Math.round((performance.now()/1000-F.listenT)*25):0,
        auto:false,mouth_active:F.talk,expression:{angry:0,warm:F.grimace?0:.1,sad:F.grimace*.7}});},100);

// ---------------------------------------------------------------- event helpers
const ev=m=>handle(m);
async function sleep(ms){  // wall-clock time, so a busy main thread doesn't stretch scenes; holds while paused or during a manual take
  const g0=gen;let left=ms,t=performance.now();
  while(left>0){await new Promise(r=>setTimeout(r,Math.min(50,left)));if(g0!==gen)throw SKIP;const n=performance.now();if(!paused&&!manual)left-=n-t;t=n;}}
const wait=ms=>new Promise(r=>setTimeout(r,ms));
function nurse(text){prompt_=text;ev({type:'log',entry:{ts:now(),who:'nurse',text}});ctx();}
function ctx(){ev({type:'context',context:{notes:'Trach day 3, alert, mouths words',category:null,last_prompt:prompt_,history:history.slice(-6)}});}
function patientSays(text,conf,extra={}){history.push(text);ev({type:'log',entry:{ts:now(),who:'patient',text,confidence:conf,...extra}});ctx();}
function signal(kind,value,confidence=.9){ev({type:'signal',kind,value,confidence,ts:now()});}
function status(s,stage){ev({type:'status',status:s,...(stage?{stage}:{})});}
async function mouth(ms){F.listening=true;F.listenT=performance.now()/1000;status('listening');await sleep(250);F.talk=true;signal('mouthing',true,.95);await sleep(ms);F.talk=false;signal('mouthing',false,.95);await sleep(200);F.listening=false;status('processing','crop');}
function nv({head=null,fingers=null,blink=null,pain=0,emotion='neutral',intensity=0}={}){
  return {head:{value:head,confidence:head?.9:0},fingers:{value:fingers,confidence:fingers!=null?.9:0},blink_code:{value:blink,confidence:blink?.85:0},pain:{value:pain,confidence:.7},emotion:{label:emotion,intensity}};}
function result(rows,{nonverbal,emotion='neutral',intensity=0,changed=false,dur=1.6}={}){
  uid++;const words=rows[0][0].split(' '),step=dur/words.length;
  const ranking=rows.map(([phrase,v,f,reasons=[]])=>({phrase,vsr_prob:v,final_prob:f,prior:reasons.length?1.5:0,reasons,vsr_score:Math.log(v)*3,att:Math.log(v)*2.6,ctc:Math.log(v)*4,prefiltered_out:false}));
  const lat={crop:.04+Math.random()*.02,encode:.11+Math.random()*.03,phrase:.24+Math.random()*.08};lat.total=lat.crop+lat.encode+lat.phrase;
  ev({type:'raw',utt_id:uid,stage:'greedy',text:rows[0][0].toUpperCase().replace(/[^A-Z' ]/g,''),n_frames:Math.round(dur*25),duration:dur,latency:{crop:lat.crop,encode:lat.encode,greedy:.004}});
  status('idle');  // the server goes idle right before broadcasting the result
  ev({type:'result',utt_id:uid,mode:'phrase',raw_greedy:rows[0][0].toUpperCase(),selected:rows[0][0],confidence:rows[0][2],margin:.4,visual_top:[...rows].sort((a,b)=>b[1]-a[1])[0][0],
      context_changed_choice:changed,ranking,in_inventory:true,phrase_gap:-1.1,greedy_score:-4.2,best_phrase_score:-3.1,n_frames:Math.round(dur*25),duration:dur,source:'demo',label:null,
      expression:{emotion,intensity},timing:{duration:dur,rate:1,pauses:[],words:words.map((w,i)=>({word:w,start:i*step,end:i*step+step*.85}))},
      context:{last_prompt:prompt_},latency:lat,nonverbal:nonverbal||nv()});
  return uid;
}
function decision(u,text,confidence,source,reason,alternatives=[],action='speak',provider='grok-4 · xAI'){
  ev({type:'decision',utt_id:u,text,confidence,source,reason,provider,alternatives,action});}
function audioSoon(){setTimeout(()=>latMark('audio'),180+Math.random()*120);}  // the demo may have no audible voice until the page is clicked
async function ask(u,candidate,attempt,scripted){
  ev({type:'confirm',utt_id:u,candidate,attempt,state:'asking'});answer=null;
  for(let t=0;t<2000&&answer===null;t+=50)await sleep(50);
  const yes=answer!==null?answer:scripted;
  if(answer===null){if(yes){F.nod=performance.now()/1000;signal('nod','yes',.94);}else{F.shake=performance.now()/1000;signal('shake','no',.91);}await sleep(700);}
  ev({type:'confirm',utt_id:u,candidate,attempt,state:yes?'confirmed':'rejected'});return yes;
}

// ---------------------------------------------------------------- scenes
const SCENES=[
 {name:'Nurse asks · patient mouths · Grok fuses',async run(){
   nurse('How is your pain right now?');await sleep(1600);
   await mouth(1700);F.grimace=.65;signal('pain',.64,.72);await sleep(380);
   const u=result([['I am in pain',.46,.71,['keywords:pain']],['I am in bed',.21,.09],['I am tired',.12,.07],['I need a blanket',.08,.05],['The pain is getting worse',.06,.05]],
     {nonverbal:nv({pain:.64,emotion:'sad',intensity:.5}),emotion:'sad',intensity:.5});audioSoon();
   await sleep(420);
   decision(u,'I am in pain',.9,'fused','Lips favour “I am in pain” (46% visual). The nurse asked about pain and the face shows a 6/10 grimace, so fused confidence is high.',[{text:'The pain is getting worse',confidence:.06},{text:'I am tired',confidence:.03}]);
   patientSays('I am in pain',.9,{source:'fused',emotion:'sad'});await sleep(4200);F.grimace=0;}},
 {name:'Pain score on fingers · fast path, no LLM',async run(){
   nurse('Show me your pain on your fingers, zero to ten.');await sleep(1500);
   F.fingers=5;F.handT=performance.now()/1000;await sleep(700);signal('fingers',5,.8);await sleep(500);F.fingers=7;await sleep(600);signal('fingers',7,.93);await sleep(250);
   uid++;decision(uid,'My pain is a seven out of ten',.93,'gesture','Seven fingers held up (both hands) in answer to a pain-scale question. Clear gesture, so the LLM was skipped.',[{text:'My pain is a six out of ten',confidence:.04}],'speak','fast path · no LLM');
   patientSays('My pain is a seven out of ten',.93,{source:'gesture'});await sleep(3800);F.fingers=null;}},
 {name:'Low confidence · “Sounds like…?” · nod / shake',async run(){
   nurse('Are you warm enough?');await sleep(1500);
   await mouth(1400);await sleep(380);
   const u=result([['I am hot',.41,.41],['I am cold',.38,.38],['I need a blanket',.09,.09],['I am OK',.06,.06],['I am tired',.04,.04]]);
   await sleep(420);
   decision(u,'I am hot',.44,'fused','“Hot” and “cold” look almost identical on the lips (41% vs 38%), and the question fits both. Asking the patient to confirm.',[{text:'I am cold',confidence:.4},{text:'I need a blanket',confidence:.1}],'confirm');
   audioSoon();
   if(!await ask(u,'I am hot',1,false)){await sleep(600);if(await ask(u,'I am cold',2,true))patientSays('I am cold',.95,{source:'fused'});}
   else patientSays('I am hot',.95,{source:'fused'});
   await sleep(3500);}},
 {name:'Yes / no by blink code',async run(){
   nurse('Do you want me to call your family?');await sleep(1600);
   const t=performance.now()/1000;F.blinks.push(t,t+.02);nextBlink=t+4;await sleep(500);
   signal('blink_code','yes',.86);await sleep(250);
   uid++;decision(uid,'Yes',.86,'gesture','One deliberate blink (1 = yes) in answer to a yes/no question.',[{text:'No',confidence:.06}],'speak','fast path · no LLM');
   patientSays('Yes',.86,{source:'gesture'});await sleep(3500);}},
 {name:'Critical phrase · full-screen escalation',async run(){
   nurse('Okay, I will call them now.');await sleep(1800);
   await mouth(1500);F.grimace=1;signal('pain',.82,.8);await sleep(360);
   const u=result([["I can't breathe",.62,.83],['I can breathe better now',.14,.08],['I need to cough',.1,.05],['I need suction',.06,.03]],{nonverbal:nv({pain:.82,emotion:'scared',intensity:.8}),emotion:'sad',intensity:.8});
   ev({type:'alert',utt_id:u,text:"I can't breathe",confidence:.83,ts:now()});
   await sleep(300);decision(u,"I can't breathe",.92,'fused','Critical phrase. Lips 62% and a strong distress face; escalating.',[{text:'I need suction',confidence:.04}]);
   patientSays("I can't breathe",.92,{source:'fused',critical:true});
   await sleep(5500);$('#alert').classList.remove('on');speechSynthesis.cancel();F.grimace=0;await sleep(1200);}},
];

// ---------------------------------------------------------------- transport: the buttons and keys work in demo too
transport.send=async o=>{
  if(o.cmd==='start'){manual=true;F.listening=true;F.listenT=performance.now()/1000;status('listening');F.talk=true;signal('mouthing',true);}
  else if(o.cmd==='stop'){if(!manual)return;F.talk=false;signal('mouthing',false);F.listening=false;status('processing','crop');await wait(380);
    const u=result([['I need water',.52,.68,['keywords:water']],['I need a blanket',.18,.12],['I need ice chips',.11,.08],['I need to cough',.07,.05]]);audioSoon();
    await wait(400);decision(u,'I need water',.86,'fused','Lips favour “I need water”; no conflicting context.',[{text:'I need ice chips',confidence:.08}]);
    patientSays('I need water',.86,{source:'fused'});await wait(2500);manual=false;}
  else if(o.cmd==='nurse_text'){nurse(o.text);}
  else if(o.cmd==='nurse'){status('nurse_listening');await wait(2200);status('idle');nurse('Are you comfortable?');}
  else if(o.cmd==='mode'){ev({type:'state',state:{mode:o.mode}});}
  else if(o.cmd==='context'){ctx();}
  else if(o.cmd==='confirm_answer'){answer=o.answer==='yes';}
};

// ---------------------------------------------------------------- demo bar + runner
const bar=$('#demobar');bar.classList.add('on');
function drawBar(){bar.innerHTML=`<b>DEMO</b><span>${idx+1}/${SCENES.length} · ${SCENES[idx].name}</span><button id="dprev">⏮</button><button id="dplay">${paused?'▶':'⏸'}</button><button id="dnext">⏭</button>`;
  $('#dprev').onclick=()=>go(-1);$('#dnext').onclick=()=>go(1);$('#dplay').onclick=()=>{paused=!paused;drawBar();};}
function go(d){idx=(idx+d+SCENES.length)%SCENES.length;gen++;drawBar();}
window.addEventListener('keydown',e=>{if(typing())return;if(e.key==='ArrowRight')go(1);else if(e.key==='ArrowLeft')go(-1);else if(e.key==='p'){paused=!paused;drawBar();}});
function reset(){Object.assign(F,{talk:false,fingers:null,grimace:0,listening:false});$('#alert').classList.remove('on');status('idle');}
async function runner(){
  for(;;){const g0=gen;reset();drawBar();
    try{await SCENES[idx].run();if(g0===gen)idx=(idx+1)%SCENES.length;}catch(e){if(e!==SKIP)throw e;}
    if(idx===0&&g0===gen){await wait(800);$('#log').innerHTML='';history.length=0;}}
}
const DEMO_PHRASES={urgent:["I can't breathe","I am choking","My chest hurts","I need suction","I need help right now"],
  pain:['I am in pain','The pain is getting worse','The pain is better now','My head hurts','My throat hurts','I need pain medication'],
  breathing:['I need to cough','I need oxygen','I can breathe better now'],
  comfort:['I am cold','I am hot','I need a blanket','Turn me over','I want to sit up','Adjust my pillow','My mouth is dry'],
  needs:['I need water','I need ice chips','I need the bathroom','I am tired','I am in bed'],
  people:['Call my family','Where is my family','Call the nurse'],
  feelings:['I am scared','I am OK','Thank you'],answers:['Yes','No','I don\'t know']};
const table=Object.entries(DEMO_PHRASES).flatMap(([category,ps])=>ps.map(phrase=>({phrase,category,critical:category==='urgent'})));
ev({type:'hello',state:{mode:'phrase',expressive:true},phrases:table.map(r=>r.phrase),phrase_table:table,context:{notes:'Trach day 3, alert, mouths words',category:null,last_prompt:'',history:[]},log:[]});
$('#engine').textContent='DEMO · simulated events, no backend';status('idle');runner();
})();
