// Silent Running bedside UI. Events arrive over /ws (event contract in .conductor/AGENTS.md) and 10 Hz tracking over /ws_meta.
// ?demo=1 swaps the transport for a scripted replay (demo.js), so everything here renders without the backend or a camera.
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const DEMO=new URLSearchParams(location.search).get('demo')==='1';
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const fmtPct=p=>(p*100).toFixed(0)+'%';
const ms=s=>(s*1000).toFixed(0)+' ms';
let ws,wsm,mode='phrase',last=null,phrases=[],phraseTable=[],curCategory='';
const voices={list:[]};
let spoken={};  // utt_id -> text already said for it, so decision + confirm + alert never double-speak (utt ids restart with the server)

// ---------- transport (demo.js replaces transport.send)
const transport={send(o){if(ws&&ws.readyState===1)ws.send(JSON.stringify(o))}};
function send(o){transport.send(o)}
function connect(){
  ws=new WebSocket(`ws://${location.host}/ws`);
  ws.onmessage=e=>handle(JSON.parse(e.data));
  ws.onclose=()=>{wsm.close();lostServer();setTimeout(connect,1000);};
  wsm=new WebSocket(`ws://${location.host}/ws_meta`);
  wsm.onmessage=e=>meta(JSON.parse(e.data));
}
// the server is gone: its utt ids restart, an open question can no longer be answered here, preview and tracking are dead
function lostServer(){spoken={};mouthingSig=false;last=null;lat.uid=null;stopCountdown();cf=null;
  if(['confirm','processing'].includes($('#hero').dataset.state))heroIdle('Reconnecting…','Connection lost · the question was closed');
  meta({});setState('offline');$('#track').textContent='server offline · reconnecting…';$('#cam').removeAttribute('src');}

// ---------- live tracking (10 Hz)
let mouthingSig=false;
function meterUpdate(m){const ex=m.expression||{};['angry','warm','sad'].forEach(e=>{const el=document.getElementById('m_'+e);if(el)el.style.width=Math.min(100,(ex[e]||0)*100)+'%';});}
function meta(m){
  meterUpdate(m);
  const d=$('#dot'),t=$('#track');
  if(m.listening){d.className='dot rec';}else d.className='dot'+(m.face?' ok':'');
  if(!m.face)t.textContent='no face — look at the camera';
  else{
    // mouth width in camera pixels vs the width the model's crop reads (camera_proc.MouthPixels): below it the crop is upsampled (blurred)
    const low=m.mouth_px!=null&&m.mouth_px<m.mouth_px_need,q=low?'move closer / zoom in':(m.face_frac>0.3?'a bit further':'good distance');
    t.innerHTML=esc(m.listening?`listening · ${m.n_frames} frames`:`tracking mouth · ${q}`)
      +(m.mouth_px!=null?` · <b class="mouthpx ${low?'low':'ok'}">mouth ${m.mouth_px}/${m.mouth_px_need} px</b>`:'')
      +esc(` · ${(m.fps||0).toFixed(0)} fps`+(m.auto?` · motion ${m.energy.toFixed(1)}/${m.noise.toFixed(1)}`:''));}
  $('#camwrap').classList.toggle('listening',!!m.listening);
  setMouthing(mouthingSig||!!m.mouth_active,false);
  showHands(m.hands);
  showZoom(m.source);showRotate(m.source);
  drawOverlay(m);
}
// ESP32-CAM sensor zoom (its sources report zoom): the board re-windows its sensor in place, ~1 s without frames
function showZoom(src){const z=$('#zoom');z.hidden=!src||src.zoom==null;if(z.hidden||z.classList.contains('busy'))return;
  $$('#zoom button').forEach(b=>b.classList.toggle('on',+b.dataset.z===src.zoom));}
// rotation (every source reports it): turned on the laptop, so it is instant
function showRotate(src){const b=$('#rotate');b.hidden=!src||src.rotate==null;if(!b.hidden&&!b.classList.contains('busy'))b.textContent=`↻ ${src.rotate}°`;}
$('#rotate').onclick=async()=>{const b=$('#rotate'),cur=parseInt(b.textContent.replace(/\D/g,''))||0;b.classList.add('busy');
  try{await fetch('/api/rotate?degrees='+((cur+90)%360),{method:'POST'});}finally{b.classList.remove('busy');}};  // a refusal arrives as an error event
$$('#zoom button').forEach(b=>b.onclick=async()=>{const z=$('#zoom');z.classList.add('busy');
  try{await fetch('/api/zoom?zoom='+b.dataset.z,{method:'POST'});}finally{z.classList.remove('busy');}});  // a refusal arrives as an error event
function showHands(st){  // "on" | "off: <reason>" (e.g. a missing model), so a disabled hand tracker is visible to the operator
  const b=$('#hands');b.hidden=st==null;if(st==null)return;const on=st==='on';
  b.className='badge handsbadge'+(on?'':' off');b.textContent=`hand signals ${st}`;
  $$('#sig-fingers,#sig-hand').forEach(e=>{e.classList.toggle('off',!on);e.title=on?'':`hand signals ${st}`;});}
function drawOverlay(m){
  const g=$('#ovbox');
  if(!m.face||!m.bbox){g.innerHTML='';return;}
  const W=m.frame_w||640,H=m.frame_h||480;$('#overlay').setAttribute('viewBox',`0 0 ${W} ${H}`);
  let [x1,y1,x2,y2]=m.bbox;if(m.source&&m.source.mirror)[x1,x2]=[W-x2,W-x1];  // a mirrored preview (webcam): the bbox is not
  const L=Math.min(x2-x1,y2-y1)*.28,c=m.listening||m.mouth_active?'#5aa9ff':'#3ddc97';
  const p=`M${x1},${y1+L}V${y1}H${x1+L}M${x2-L},${y1}H${x2}V${y1+L}M${x2},${y2-L}V${y2}H${x2-L}M${x1+L},${y2}H${x1}V${y2-L}`;
  g.innerHTML=`<path d="${p}" stroke="${c}" style="vector-effect:non-scaling-stroke;stroke-width:3.5"/><text class="lbl" x="${x1}" y="${y1-W/80}" fill="${c}" font-size="${W/42}">${m.listening?'READING LIPS':'MOUTH'}</text>`;
}

// ---------- status
function setWarm(w){$('#warmup').hidden=w!==false;}  // shown only while the server says it is still warming (demo mode never does)
let heroBefore={state:'idle',eyebrow:''};  // restored when processing ends without anything replacing the hero (a quiet answer, an error)
let enrolling=false;  // the hero shows the enrollment prompt; cleared when the session ends
function setState(s,extra){
  if(enrolling&&s!=='enrolling'){enrolling=false;heroIdle('Hold Listen and mouth a phrase','Waiting for the patient');}
  const el=$('#state');el.className='state '+s;el.textContent=(s==='nurse_listening'?'nurse speaking':s)+(extra?' · '+extra:'');
  if(s!=='nurse_listening'&&$('#nurse').classList.contains('on')){$('#nurse').classList.remove('on');$('#nurselbl').textContent=NURSE_LBL;}
  const p=$('#statuspill');p.className='statuspill '+s;p.querySelector('span').textContent=s==='nurse_listening'?'nurse speaking':s;
  const h=$('#hero');
  if(s==='listening'&&h.dataset.state!=='confirm'){$('#eyebrow').textContent='Listening · mouth the phrase';}
  else if(s==='processing'){if(h.dataset.state!=='processing')heroBefore={state:h.dataset.state,eyebrow:$('#eyebrow').textContent};h.dataset.state='processing';$('#eyebrow').textContent='Reading lips…';}
  else if(s==='idle'&&h.dataset.state==='processing'){h.dataset.state=heroBefore.state;$('#eyebrow').textContent=heroBefore.eyebrow;}
  else if(s==='enrolling'&&extra){enrolling=true;heroIdle(extra,'Enrolling this patient · hold Listen, mouth the phrase, release');}  // the prompt must be readable at the bedside
}

// ---------- event dispatch
function handle(m){
  logEvent(m);
  if(m.type==='hello'){setState(m.state.status);setWarm(m.state.warm);phrases=m.phrases;phraseTable=m.phrase_table||[];applyCtx(m.context);mode=m.state.mode;syncMode();if(m.state.expressive!=null)$('#expressive').checked=m.state.expressive;$('#pace').checked=!!m.state.pace;
    $('#nphr').textContent=phrases.length+' phrases';$('#phrlist').innerHTML=phrases.map(p=>`<span>${esc(p)}</span>`).join('');$('#phrasedl').innerHTML=phrases.map(p=>`<option value="${esc(p)}">`).join('');buildChips();$('#log').innerHTML=LOG_EMPTY;(m.log||[]).forEach(addLog);
    if(!DEMO){if(!$('#cam').getAttribute('src'))$('#cam').src='/stream?session='+Date.now();fetch('/api/state').then(r=>r.json()).then(s=>{$('#engine').textContent=`${s.engine.model} · ${phrases.length} phrases`;});
      if($('#voice').value.startsWith('clone:'))fillBank($('#voice').value.slice(6));}}  // the voice may be listed before the phrases
  else if(m.type==='status'){setState(m.status,m.stage);if(m.status==='processing'&&m.stage==='crop')toast('');}
  else if(m.type==='raw'){$('#raw').innerHTML=`<span class="lbl">raw (CTC greedy)</span>${esc(m.text)||'<span class="small">(nothing)</span>'}`;$('#nbest').innerHTML='';$('#lat').textContent=`crop ${ms(m.latency.crop)} · encode ${ms(m.latency.encode)} · ${m.n_frames} frames (${m.duration.toFixed(1)} s)`;}
  else if(m.type==='result'){last=m;render(m,isQuiet(m));labelFor(m.utt_id,m.selected);if(m.mode==='open'&&!$('#llm').checked)speakOnce(m.utt_id,m.selected);}  // Phrase Mode speaks on its decision; Open Mode has none
  else if(m.type==='decision'){onDecision(m);labelFor(m.utt_id,m.text);}
  else if(m.type==='labeled'){if(lbl.utt===m.utt_id)$('#lblstat').textContent=`saved ✓ “${m.text}”`;}
  else if(m.type==='confirm'){onConfirm(m);}
  else if(m.type==='signal'){onSignal(m);}
  else if(m.type==='alert'){showAlert(m);}
  else if(m.type==='nbest'){if(m.error)$('#nbest').textContent='beam n-best failed: '+m.error;else renderNbest(m.nbest);}
  else if(m.type==='llm'){renderLLM(m);}
  else if(m.type==='llm_reason'){if(last&&m.utt_id===last.utt_id&&$('#llmreason'))$('#llmreason').textContent=m.error?`(reason unavailable: ${m.error})`:m.reason;}  // follows the verdict
  else if(m.type==='delivery'){const d=$('#delivrep');if(d){d.innerHTML=`delivered <b>${esc(m.emotion)}</b>${m.intensity?` ${(m.intensity*100).toFixed(0)}%`:''} · ${esc(m.model)}${m.tag?` · tag <code>${esc(m.tag)}</code>`:''} · stability ${esc(m.stability)} · speed ${m.rate.toFixed(2)}x · synth ${m.cached?'cached':m.streamed?`streamed, first audio ${esc(m.t_synth)} s`:esc(m.t_synth)+' s'}${m.retime&&m.retime.applied?` · retimed (global ${esc(m.retime.global)}x)`:(m.retime&&m.retime.reason?` · no retime: ${esc(m.retime.reason)}`:'')} · total ${esc(m.total)} s`;}renderAudioStrip(m.retime);}
  else if(m.type==='log'){addLog(m.entry);}
  else if(m.type==='error'){toast(m.message);}  // a failed decode comes with its own idle status
  else if(m.type==='context'){applyCtx(m.context);}
  else if(m.type==='state'){mode=m.state.mode;syncMode();setWarm(m.state.warm);$('#pace').checked=!!m.state.pace;}
  else if(m.type==='prewarmed'){$('#clonestat').textContent=`voice “${m.speaker}” ready · ${m.n} phrase variants cached`;if($('#voice').value==='clone:'+m.speaker)fillBank(m.speaker,true);}
  else if(m.type==='saved'){toast('saved '+m.file+' as "'+m.phrase+'"',true);}
}
function logEvent(m){
  const E=$('#events'),d=document.createElement('div');const {type,...rest}=m;
  let s=JSON.stringify(rest);if(s.length>180)s=s.slice(0,180)+'…';
  d.className=type;d.innerHTML=`<b>${esc(type)}</b>${esc(s)}`;E.prepend(d);while(E.children.length>80)E.lastChild.remove();
}
function syncMode(){$$('#mode button').forEach(b=>b.classList.toggle('on',b.dataset.mode===mode));}
function buildChips(){const cats=[...new Set(phraseTable.map(r=>r.category))];const c=$('#chips');c.innerHTML='';const sel=$('#category');sel.innerHTML='<option value="">— none —</option>';
  cats.forEach(cat=>{const b=document.createElement('button');b.textContent=cat;if(cat==='urgent')b.classList.add('urgent');b.onclick=()=>{curCategory=(curCategory===cat?'':cat);$$('#chips button').forEach(x=>x.classList.toggle('on',x.textContent===curCategory));$('#category').value=curCategory;pushCtx();};c.appendChild(b);
    const o=document.createElement('option');o.value=cat;o.textContent=cat;sel.appendChild(o);});}
const LOG_EMPTY=$('#log').innerHTML;
function addLog(e){const L=$('#log');if(L.querySelector('.small'))L.innerHTML='';const d=document.createElement('div');d.className='msg '+e.who+(e.critical?' critical':'');const t=new Date(e.ts*1000).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'});
  d.innerHTML=`<span class="t">${esc(e.text)}</span><span class="meta">${esc(e.who)} · ${t}${e.confidence?` · ${fmtPct(e.confidence)}`:''}${e.source?` · ${esc(e.source)}`:''}${e.emotion&&e.emotion!=='neutral'?` · ${esc(e.emotion)}`:''}</span>`;L.appendChild(d);$('#logscroll').scrollTop=0;  // 0 = bottom of the reversed container: bring the newest into view
  if(e.who==='nurse')showAsked(e.text);}
function showAsked(t){const q=$('#asked');q.className='q'+(t?'':' empty');q.textContent=t||'No question yet';if(t){void q.offsetWidth;q.classList.add('fresh');}}
function showAlert(m){$('#alerttext').textContent=m.text.toUpperCase();const a=$('#alert');a.classList.add('on');
  spoken[m.utt_id]=m.text;  // the urgent double announcement owns this utterance; a later decision must not cut it off
  speak(m.text,{emotion:'urgent',onstart:()=>latMark(m.utt_id,'audio'),onend:()=>speak(m.text,{emotion:'urgent'})});
  clearTimeout(showAlert.t);showAlert.t=setTimeout(()=>a.classList.remove('on'),15000);}
$('#alertok').onclick=()=>{$('#alert').classList.remove('on');};

// ---------- hero
const SRC_LABEL={lips:'Lip reading',gesture:'Gesture',fused:'Lips + context'};
function setBig(text,{question=false}={}){
  const bt=$('#bigtext'),T=(text||'').toUpperCase();
  bt.className='text'+(T.length>48?' long':(T.length>20?' mid':''));
  bt.innerHTML=question?`<span class="sl">Sounds like…</span>${esc(T)}?`:esc(T);
  void bt.offsetWidth;bt.classList.add('in');
}
function setRing(c){$('#ringarc').style.strokeDashoffset=c==null?326.7:326.7*(1-Math.max(0,Math.min(1,c)));$('#ringpct').textContent=c==null?'—':fmtPct(c);}
function heroSet(state,src){const h=$('#hero');h.dataset.state=state;h.classList.remove('weak');if(src)h.dataset.src=src;$('#src').textContent=src||h.dataset.src;}
function render(m,quiet){  // quiet: a yes/no answering the open question, or nothing to say: Dev ranking only, the hero keeps the question
  if(!quiet)renderHero(m);
  const cd=$('#cands');cd.innerHTML='';
  if(m.mode==='phrase'){
    const rows=m.ranking.filter(r=>!r.prefiltered_out);const mx=Math.max(...rows.map(r=>r.final_prob));
    rows.forEach((r,i)=>{const div=document.createElement('div');div.className='cand';const tag=r.reasons.length?`<b class="tag">+${r.prior.toFixed(1)} ${esc(r.reasons.join(', '))}</b>`:'';
      div.innerHTML=`<div class="idx">${i+1}</div><div class="bar"><i class="vis" style="width:${(r.vsr_prob/mx*100).toFixed(1)}%"></i><i style="width:${(r.final_prob/mx*100).toFixed(1)}%;opacity:.8"></i><span>${esc(r.phrase)}</span>${tag}</div><div class="pct">${fmtPct(r.final_prob)}</div>`;
      div.title=`VSR log-lik ${r.vsr_score.toFixed(2)} (att ${(r.att||0).toFixed(1)}, ctc ${r.ctc.toFixed(1)}) · visual-only ${fmtPct(r.vsr_prob)} · prior +${r.prior.toFixed(2)}`;cd.appendChild(div);});
    $('#llmpanel').style.display='none';
  }else{
    renderNbest(m.nbest,true);
    $('#llmpanel').style.display=$('#llm').checked?'':'none';$('#llmout').innerHTML='<span class="small">LLM proposing a correction from the visual hypotheses + context; the visual model will verify it…</span>';
  }
  $('#lat').textContent=Object.entries(m.latency).map(([k,v])=>`${k} ${ms(v)}`).join(' · ')+` · ${m.n_frames} frames (${m.duration.toFixed(1)} s)`;
}
function renderHero(m){
  heroSet('result','lips');setBig(m.selected);setRing(m.confidence);$('#hero').classList.toggle('weak',m.mode==='phrase'&&m.in_inventory===false);
  $('#eyebrow').textContent=`${SRC_LABEL.lips} · ${m.mode==='open'?'open vocabulary':'phrase match'} · ${ms(m.latency.total)}`;
  $('#reason').textContent=m.mode==='phrase'?(m.context_changed_choice?`Context changed the choice: the visual top was “${m.visual_top}”.`:''):'';
  $('#alts').innerHTML='';
  const conf=$('#conf');let pills='';const c=m.confidence;
  pills+=`<span class="pill ${c>0.7?'ok':'warn'}">lips ${fmtPct(c)}</span>`;
  if(m.critical)pills+=`<span class="pill bad">CRITICAL</span>`;
  if(m.mode==='phrase'){
    if(m.in_inventory===false)pills+=`<span class="pill warn">weak match: free transcript fits ${m.phrase_gap.toFixed(1)} nats better</span>`;
    if(m.context_changed_choice)pills+=`<span class="pill ctx">context changed choice</span>`;
    if(m.category)pills+=`<span class="pill">${esc(m.category)}</span>`;
  }else{pills+=`<span class="pill">open vocabulary · beam search</span>`;}
  pills+=nonverbalPills(m.nonverbal);
  conf.innerHTML=pills;
  applyNonverbal(m.nonverbal);
  renderDelivery(m);
  const dym=$('#dym');dym.innerHTML='';
  if(m.mode==='phrase'){
    const top=m.ranking.filter(r=>!r.prefiltered_out).slice(0,5),mb=Math.max(...top.map(r=>Math.max(r.final_prob,r.vsr_prob)));
    top.forEach((r,i)=>{const b=document.createElement('button');if(i===0)b.className='top';
      b.innerHTML=`<span>${esc(r.phrase)}${r.reasons.length?`<b class="why">+${r.prior.toFixed(1)} ${esc(r.reasons.join(', '))}</b>`:''}</span><small>${fmtPct(r.final_prob)}</small><span class="bars"><i class="bv" style="width:${(r.vsr_prob/mb*100).toFixed(1)}%"></i><i class="bc" style="width:${(r.final_prob/mb*100).toFixed(1)}%"></i></span>`;
      b.title=`visual-only ${fmtPct(r.vsr_prob)} → with context ${fmtPct(r.final_prob)}`;
      b.onclick=()=>pick(r.phrase,m.utt_id);dym.appendChild(b);});
  }else{
    m.nbest.slice(0,4).forEach((h,i)=>{const b=document.createElement('button');if(i===0)b.className='top';b.innerHTML=`<span>${esc(pretty(h.text))||'(empty)'}</span><small>${fmtPct(h.prob)}</small>`;b.onclick=()=>speak(pretty(h.text),{utt_id:m.utt_id});dym.appendChild(b);});
  }
  latResult(m);
  const L=m.latency;$('#vitals').innerHTML=`<span>server <b>${ms(L.total)}</b></span><span>face <b>${esc((m.expression||{}).emotion||'neutral')}</b></span><span>mouthed <b>${m.timing?m.timing.duration.toFixed(2)+' s':'—'}</b></span><span>mode <b>${esc(m.mode)}</b></span><span>utt <b>#${m.utt_id}</b></span>`;
}
function pick(text,uid){speak(text,{utt_id:uid});send({cmd:'confirm',text});setBig(text);}  // the nurse taps a candidate: say it, record it
function onDecision(d){
  latMark(d.utt_id,'decision');
  $('#decision').innerHTML=`<div class="dec"><div class="t">${esc(d.text)} <span class="pill">${esc(d.source)}</span> <span class="pill">${esc(d.action)}</span></div><div class="small">${esc(d.reason)}</div>
    <table><tr><td>utt</td><td>#${d.utt_id}</td></tr><tr><td>confidence</td><td>${fmtPct(d.confidence)}</td></tr><tr><td>provider</td><td>${esc(d.provider)}</td></tr>${(d.alternatives||[]).map(a=>`<tr><td>alt</td><td>${esc(a.text)} · ${fmtPct(a.confidence)}</td></tr>`).join('')}</table></div>`;
  if(d.action==='none')return;  // nothing to say (e.g. a yes/no that answered the open question): the hero keeps what it shows
  if(d.action==='speak'&&spoken[d.utt_id]===d.text)return;  // records a confirmation: the confirm event (or the critical alert) says it
  const alts=(d.alternatives||[]).filter(a=>a.text!==d.text);  // the server's list starts with the pick itself
  heroSet('result',d.source);setBig(d.text);setRing(d.confidence);
  $('#eyebrow').textContent=`${SRC_LABEL[d.source]||d.source}${d.provider?' · '+d.provider:''}`;
  $('#reason').textContent=d.reason||'';
  if(!last||d.utt_id!==last.utt_id){  // no lip result behind this decision (gesture fast path): drop the previous utterance's evidence
    $('#conf').innerHTML='';const dym=$('#dym');dym.innerHTML='';
    [{text:d.text,confidence:d.confidence},...alts].forEach((a,i)=>{const b=document.createElement('button');if(i===0)b.className='top';
      b.innerHTML=`<span>${esc(a.text)}</span><small>${fmtPct(a.confidence)}</small>`;b.onclick=()=>pick(a.text,d.utt_id);dym.appendChild(b);});
  }
  $('#alts').innerHTML=alts.length?'Also possible: '+alts.slice(0,3).map((a,i)=>`<span data-i="${i}" style="cursor:pointer">${esc(a.text)} ${fmtPct(a.confidence)}</span>`).join(' · '):'';
  $$('#alts span').forEach(s=>s.onclick=()=>pick(alts[+s.dataset.i].text,d.utt_id));
  if(d.action==='speak'&&!(last&&last.utt_id===d.utt_id&&last.critical))speakOnce(d.utt_id,d.text);  // a critical phrase is announced by its alert
}

// ---------- confirmation loop, owned by the server: "Sounds like X?", nod / Y = yes, shake / N = the next guess
let cf=null;  // the open question {key,utt_id,anims}
const isQuiet=m=>m.action==='answer'||m.action==='none';
function onConfirm(c){
  if(c.state!=='asking'&&(!cf||cf.utt_id!==c.utt_id))return;  // about a question this page is not showing; the server has one question at a time
  if(c.state==='asking'){
    heroSet('confirm');setBig(c.candidate,{question:true});if(c.confidence!=null)setRing(c.confidence);
    $('#eyebrow').textContent=`Checking with the patient · guess ${c.attempt}`;
    $('#confirmhint').textContent='Nod to confirm · shake for the next guess';
    const key=`${c.utt_id}:${c.attempt}`;if(cf&&cf.key===key)return;
    stopCountdown();cf={key,utt_id:c.utt_id};
    if(c.say&&$('#autospeak').checked)sayConfirm(c,{onstart:()=>latMark(c.utt_id,'audio'),  // the answer window (and its countdown) starts when the prompt ends
      onend:()=>{send({cmd:'prompt_played',utt_id:c.utt_id,attempt:c.attempt});if(cf&&cf.key===key)startCountdown(c.timeout);}});
  }else if(c.state==='rejected'){
    if(c.reason){stopCountdown();cf=null;if($('#hero').dataset.state==='confirm')heroSet('result');}  // closed by the server, e.g. the nurse tapped a phrase
    else if(c.say){endQuestion('Please mouth it again',`Not “${c.candidate}”`);if($('#autospeak').checked)sayConfirm(c);}  // the last guess
    else $('#confirmhint').textContent=`✗ Not “${c.candidate}” · trying the next guess`;
  }else if(c.state==='confirmed'){
    stopCountdown();cf=null;heroSet('confirmed');setBig(c.candidate);setRing(1);
    // the reason, alternatives and evidence chips described the guess being asked about, not the confirmed phrase
    $('#reason').textContent='';$('#alts').innerHTML='';$('#conf').innerHTML='';
    $('#eyebrow').textContent=`Confirmed by ${c.by==='nurse'?'the nurse':`the patient (${c.by})`} ✓${c.latency?` · answered in ${c.latency.answer.toFixed(1)} s`:''}`;
    spoken[c.utt_id]=c.candidate;  // said here; say is null for a critical phrase, which its alert announces
    if(c.say&&$('#autospeak').checked)sayConfirm(c,{onstart:()=>latMark(c.utt_id,'audio')});
  }else if(c.state==='timeout'){
    endQuestion('No answer',`Nothing spoken · “${c.candidate}” was not confirmed`);
  }
}
function sayConfirm(c,opts){if(c.say_voice==='system')speakSystem(c.say,opts||{});else speak(c.say,{...opts,utt_id:c.utt_id});}
function endQuestion(text,why){stopCountdown();cf=null;if($('#hero').dataset.state==='confirm')heroIdle(text,why);}  // a newer result may own the hero
function heroIdle(text,why){heroSet('idle');setRing(null);const bt=$('#bigtext');bt.className='text placeholder';bt.textContent=text;$('#eyebrow').textContent=why;$('#reason').textContent='';$('#alts').innerHTML='';}
function startCountdown(sec){stopCountdown();$('#hero').classList.add('counting');  // hidden until then: unplayed, the server's window is longer
  cf.anims=[$('#cdfill').animate([{transform:'scaleX(1)'},{transform:'scaleX(0)'}],{duration:sec*1000,fill:'forwards'}),
    $('#ringcd').animate([{strokeDashoffset:0},{strokeDashoffset:276.5}],{duration:sec*1000,fill:'forwards'})];}
function stopCountdown(){if(cf&&cf.anims)cf.anims.forEach(a=>a.cancel());$('#hero').classList.remove('counting');}
function answerQuestion(yes){send({cmd:'answer',value:yes?'yes':'no'});}  // the nurse answers for the patient; the server ignores it when nothing is asked
$('#cyes').onclick=()=>answerQuestion(true);$('#cno').onclick=()=>answerQuestion(false);

// ---------- nonverbal chips (live `signal` events + the per-utterance `nonverbal` summary)
const sigT={};
function setChip(id,html,fresh=true){const el=$('#sig-'+id);if(!el)return;const v=el.querySelector('.v span');if(v)v.innerHTML=html;
  el.classList.add('on');el.classList.remove('stale');if(fresh){el.classList.remove('fresh');void el.offsetWidth;el.classList.add('fresh');}
  clearTimeout(sigT[id]);clearTimeout(sigT[id+'_r']);sigT[id]=setTimeout(()=>el.classList.add('stale'),4000);
  sigT[id+'_r']=setTimeout(()=>{el.classList.remove('on','stale');if(v)v.textContent='—';if(id==='pain'){$('#painfill').style.width='0';$('#painval').textContent='';}},15000);}
function setPain(v,fresh){$('#painfill').style.width=(Math.max(0,Math.min(1,v))*100).toFixed(0)+'%';$('#painval').textContent=`${Math.round(v*10)}/10`;setChip('pain','',fresh);}
function setMouthing(on,fromSignal=true){if(fromSignal)mouthingSig=on;const el=$('#sig-mouthing');el.classList.toggle('on',on);el.querySelector('.v span').textContent=on?'moving':'still';}
function onSignal(s){
  const c=s.confidence!=null?` <small>${fmtPct(s.confidence)}</small>`:'';
  if(s.kind==='nod')setChip('head',`YES · nod${c}`);
  else if(s.kind==='shake')setChip('head',`NO · shake${c}`);
  else if(s.kind==='blink_code')setChip('blink',`${esc(String(s.value).toUpperCase())}${c}`);
  else if(s.kind==='fingers')setChip('fingers',`${esc(s.value)}${c}`);
  else if(s.kind==='thumb'||s.kind==='point')setChip('hand',esc(handText(s.kind,s.value))+c);
  else if(s.kind==='pain')setPain(+s.value||0,true);
  else if(s.kind==='mouthing')setMouthing(!!s.value);
}
function handText(kind,v){  // plain text (escape before innerHTML); pointing directions are from the patient's side
  return kind==='thumb'?`Thumb ${v==='down'?'down':'up'}`:`Pointing ${/^(left|right)$/.test(v)?"patient's "+v:v}`;}
function nonverbalPills(nv){
  if(!nv)return '';const p=[];
  if(nv.head&&nv.head.value)p.push(`head: ${nv.head.value} ${fmtPct(nv.head.confidence||0)}`);
  if(nv.blink_code&&nv.blink_code.value)p.push(`blink: ${nv.blink_code.value}`);
  if(nv.fingers&&nv.fingers.value!=null)p.push(`fingers: ${nv.fingers.value}`);
  ['thumb','point'].forEach(k=>{if(nv[k]&&nv[k].value)p.push(handText(k,nv[k].value));});
  if(nv.pain&&nv.pain.value>0.15)p.push(`pain ${Math.round(nv.pain.value*10)}/10`);
  if(nv.emotion&&nv.emotion.label&&nv.emotion.label!=='neutral')p.push(`face: ${nv.emotion.label} ${fmtPct(nv.emotion.intensity||0)}`);
  return p.map(x=>`<span class="pill nv">${esc(x)}</span>`).join('');
}
function applyNonverbal(nv){
  if(!nv)return;
  if(nv.head&&nv.head.value)setChip('head',`${nv.head.value==='yes'?'YES · nod':'NO · shake'}`,false);
  if(nv.blink_code&&nv.blink_code.value)setChip('blink',esc(nv.blink_code.value.toUpperCase()),false);
  if(nv.fingers&&nv.fingers.value!=null)setChip('fingers',esc(nv.fingers.value),false);
  ['thumb','point'].forEach(k=>{if(nv[k]&&nv[k].value)setChip('hand',esc(handText(k,nv[k].value)),false);});
  if(nv.pain&&nv.pain.value>0.15)setPain(nv.pain.value,false);
}

// ---------- latency: server stages + decision + time to first audio
const lat={uid:null,t:{},server:null};  // one utterance: only its own decision and first audio are counted
const LAT_COLORS={crop:'#56677b',encode:'#5aa9ff',phrase:'#3ddc97',beam:'#3ddc97',read:'#5aa9ff',decide:'#b18cff',voice:'#ffb547'};  // read: what was left of a reading that ran during the hang
function latResult(m){lat.uid=m.utt_id;lat.t={result:performance.now()};lat.server=m.latency;renderLatency();}
function latMark(uid,k){if(uid===lat.uid&&!lat.t[k]){lat.t[k]=performance.now();renderLatency();}}
function renderLatency(){
  if(!lat.server)return;
  const segs=Object.entries(lat.server).filter(([k])=>k!=='total');
  const dec=lat.t.decision&&!(lat.t.audio<lat.t.decision)?lat.t.decision:null;  // a decision after the first audio was not on its path
  if(dec)segs.push(['decide',(dec-lat.t.result)/1000]);
  if(lat.t.audio)segs.push(['voice',(lat.t.audio-(dec||lat.t.result))/1000]);
  const tot=segs.reduce((a,[,v])=>a+v,0),bar=$('#latbar'),span=Math.max(3,tot);  // the bar spans 3 s, or the whole total when slower
  bar.querySelector('.target').style.left=(2/span*100)+'%';
  bar.querySelectorAll('.seg').forEach(e=>e.remove());
  segs.forEach(([k,v])=>{const e=document.createElement('div');e.className='seg';e.style.width=(v/span*100)+'%';e.style.background=LAT_COLORS[k]||'#8b9cb0';e.title=`${k} ${ms(v)}`;bar.insertBefore(e,bar.querySelector('.target'));});
  $('#latlegend').innerHTML=segs.map(([k,v])=>`<span><i style="background:${LAT_COLORS[k]||'#8b9cb0'}"></i>${k} ${ms(v)}</span>`).join('')+(lat.t.audio?'':'<span>voice …</span>');
  const T=$('#lattotal');T.textContent=`${tot.toFixed(2)} s${lat.t.audio?'':' +'}`;T.className='lattotal '+(tot<2?'ok':'slow');
}

// ---------- dev panels
function renderDelivery(m){
  const ex=m.expression||{emotion:'neutral',intensity:0},tm=m.timing;
  let h=`<span class="pill ${ex.emotion==='neutral'?'':'ctx'}">face: ${esc(ex.emotion)}${ex.intensity?` ${(ex.intensity*100).toFixed(0)}%`:''}</span>`;
  if(ex.scores)h+=` <span class="small">${Object.entries(ex.scores).map(([k,v])=>`${esc(k)} ${v.toFixed(2)}`).join(' · ')}</span>`;
  if(tm)h+=` <span class="pill">mouthed ${tm.duration.toFixed(2)} s · rate ${tm.rate.toFixed(2)}x${tm.pauses.length?` · ${tm.pauses.length} pause${tm.pauses.length>1?'s':''}`:''}</span>`;else h+=` <span class="small">no word timing</span>`;
  h+=`<div id="delivrep" class="small" style="margin-top:4px">synthesis pending…</div>`;$('#deliv').innerHTML=h;
  const st=$('#strip');st.innerHTML='';
  if(tm&&tm.words.length){const t0=tm.words[0].start,T=Math.max(tm.words[tm.words.length-1].end-t0,0.1);
    st.innerHTML=`<div class="lbl">mouthed (video)</div><div class="row2" id="rowv"></div><div class="lbl">delivered (audio)</div><div class="row2" id="rowa"></div>`;
    const rv=$('#rowv');tm.words.forEach((w,i)=>{const e=document.createElement('span');e.style.left=((w.start-t0)/T*100)+'%';e.style.width=((w.end-w.start)/T*100)+'%';e.style.background=i%2?'#3ddc97':'#7fe7b3';e.textContent=w.word;rv.appendChild(e);});}
  const rp=$('#replay');rp.innerHTML='<span class="small" style="line-height:28px">replay as:</span>';
  [['neutral',0],['angry',0.9],['warm',0.9],['sad',0.8]].forEach(([e,i])=>{const b=document.createElement('button');b.textContent=e;b.onclick=()=>speak(m.selected,{emotion:e,intensity:i,utt_id:m.utt_id});rp.appendChild(b);});
}
function renderAudioStrip(rep){const ra=$('#rowa');if(!ra||!rep||!rep.audio_words||!rep.audio_words.length)return;const aw=rep.audio_words,t0=aw[0].start,T=Math.max(aw[aw.length-1].end-t0,0.1);ra.innerHTML='';aw.forEach((w,i)=>{const e=document.createElement('span');e.style.left=((w.start-t0)/T*100)+'%';e.style.width=((w.end-w.start)/T*100)+'%';e.style.background=i%2?'#5aa9ff':'#8fc4ff';e.textContent=w.word+(rep.ratios&&Math.abs(rep.ratios[i]-1)>0.05?` ×${rep.ratios[i]}`:'');ra.appendChild(e);});}
function renderNbest(nb,intoCands){
  if(!nb)return;
  if(intoCands){const cd=$('#cands');cd.innerHTML='';const mx=Math.max(...nb.map(h=>h.prob)),uid=last.utt_id;
    nb.forEach((h,i)=>{const div=document.createElement('div');div.className='cand';div.innerHTML=`<div class="idx">${i+1}</div><div class="bar"><i style="width:${(h.prob/mx*100).toFixed(1)}%"></i><span>${esc(pretty(h.text))||'(empty)'}</span><b class="tag">${h.score.toFixed(1)}</b></div><div class="pct">${fmtPct(h.prob)}</div>`;div.onclick=()=>speak(pretty(h.text),{utt_id:uid});cd.appendChild(div);});
  }else{$('#nbest').innerHTML=`<span class="lbl">beam n-best</span>`+nb.map(h=>`<div>${h.score.toFixed(2)}&nbsp; ${esc(h.text)||'(empty)'}</div>`).join('');}
}
function renderLLM(m){
  if(!last||m.utt_id!==last.utt_id)return;
  if(m.error){$('#llmout').innerHTML=`<span class="err">LLM unavailable: ${esc(m.error)}</span>`;speakOnce(last.utt_id,last.selected);return;}
  const raw=pretty(last.nbest[0].text);let verdict;
  if(!m.changed)verdict=`<span class="pill ok">LLM agrees with the visual model</span>`;
  else if(m.accepted)verdict=`<span class="pill ok">accepted · video supports it (${m.gap.toFixed(1)} nats vs raw)</span>`;
  else verdict=`<span class="pill warn">rejected · video does not support it (${m.gap.toFixed(1)} nats vs raw) · kept raw</span>`;
  $('#llmout').innerHTML=`<div><b>${esc(m.corrected)}</b> ${verdict} <span class="small">· ${esc(m.model)} · ${ms(m.latency)}</span></div>`+`<div class="why">${m.changed?`LLM proposed “${esc(m.proposal)}” instead of “${esc(raw)}”. `:''}<span id="llmreason">${esc(m.reason||'')}</span></div>`;  // the reason arrives after the verdict (llm_reason)
  if(m.alternatives&&m.alternatives.length>1)$('#llmout').insertAdjacentHTML('beforeend',`<div class="why">considered: ${m.alternatives.map(a=>`${esc(a.text)} <span class="small">(${a.gap.toFixed(1)} nats${a.fits?'':' · video: no'})</span>`).join(' · ')}</div>`);
  if(m.changed&&m.accepted){setBig(m.corrected);heroSet('result','fused');$('#conf').insertAdjacentHTML('beforeend',`<span class="pill ctx">context-corrected: ${esc(wordDiff(raw,m.corrected))} · verified by the visual model</span>`);}
  speakOnce(last.utt_id,(m.changed&&m.accepted)?m.corrected:last.selected);
}
function wordDiff(a,b){const A=a.split(' '),B=b.split(' ');let i=0;while(i<A.length&&i<B.length&&A[i].toLowerCase()===B[i].toLowerCase())i++;let j=0;while(j<A.length-i&&j<B.length-i&&A[A.length-1-j].toLowerCase()===B[B.length-1-j].toLowerCase())j++;const x=A.slice(i,A.length-j).join(' ')||'∅',y=B.slice(i,B.length-j).join(' ')||'∅';return `“${x}” → “${y}”`;}
function pretty(s){s=(s||'').toLowerCase().replace(/\bi\b/g,'I').replace(/\bi'/g,"I'");return s.charAt(0).toUpperCase()+s.slice(1)}

// ---------- toast
function toast(msg,good){const box=$('#err');if(!msg){box.innerHTML='';return;}[...box.children].forEach(t=>{if(t.textContent===msg)t.remove();});  // a repeat replaces its twin
  const t=document.createElement('div');t.className='toast'+(good?' good':'');t.textContent=msg;box.appendChild(t);setTimeout(()=>t.remove(),good?3000:6000);}

// ---------- voice
function loadVoices(){voices.list=speechSynthesis.getVoices().filter(v=>v.lang.startsWith('en'));}
speechSynthesis.onvoiceschanged=()=>{loadVoices();if(!DEMO)loadCloned();};loadVoices();if(!DEMO)setTimeout(loadCloned,500);
let cloned=[],curU=null,curSrc=null,bankVoice=null;const player=new Audio(),actx=new AudioContext({latencyHint:'interactive'}),bank=new Map();
// the automatic speech of an utterance: said once, and its start is that utterance's "first audio"
function speakOnce(uid,text,opts){if(!text||!$('#autospeak').checked||spoken[uid]===text)return;spoken[uid]=text;speak(text,{...opts,utt_id:uid,onstart:()=>latMark(uid,'audio')});}
// every browser utterance goes through here: a cancelled one must not fire its onend (a stale prompt_played, a repeated alert)
// stop whatever is playing: a superseded clip must not fall back later, a cancelled utterance must not fire its onend / onerror
function hush(){player.onerror=player.onplaying=player.onended=null;player.pause();if(curU)curU.onend=curU.onerror=null;speechSynthesis.cancel();
  if(curSrc){curSrc.onended=null;curSrc.stop();curSrc=null;}}
function utter(text,v,opts){hush();const u=curU=new SpeechSynthesisUtterance(text);if(v)u.voice=v;u.onstart=opts.onstart||null;u.onend=opts.onend||null;
  u.onerror=e=>toast('Browser speech failed: '+e.error);speechSynthesis.speak(u);}
const patientVoice=()=>voices.list.find(v=>v.name==='Samantha')||voices.list[0];
function speakBrowser(text,opts,fallback){utter(text,patientVoice(),opts);if(fallback)toast('ElevenLabs unavailable, used browser voice');}
// the system's own prompts ("Sounds like: X?"): a voice that is never the patient's (ElevenLabs / clone, or the browser patient voice)
function speakSystem(text,opts){utter(text,voices.list.find(v=>v.name==='Daniel')||voices.list.find(v=>v!==patientVoice()),opts);}
function speak(text,opts){if(!text)return;const sel=$('#voice').value||'';opts=opts||{};
  if(sel.startsWith('clone:')){const sp=sel.slice(6);hush();
    // the utterance's face sets the emotion whatever is said for it (lip phrase, LLM/Grok pick, a tapped candidate);
    // the mouthed word timing (rate + server retime) only fits the lip result's own phrase
    const cur=last&&opts.utt_id===last.utt_id,own=cur&&text===last.selected;
    const ex=(cur&&last.expression)||{emotion:'neutral',intensity:0};const pace=$('#pace').checked,tm=(pace&&own&&last.timing)||{};
    let emo=opts.emotion||ex.emotion||'neutral',inten=opts.intensity!=null?opts.intensity:(ex.intensity||0),rate=tm.rate||1;
    if(emo==='urgent'){emo='angry';inten=0.7;rate=1.1;}
    const face=$('#expressive').checked,ov=$('#override').value;  // the server applies the face switch and a forced emotion
    const plain=rate===1&&(!face||(ov?ov==='neutral':(emo==='neutral'||!(inten>0))));  // what /api/tts says: neutral, natural pace
    const buf=plain&&actx.state==='running'&&bank.get(bankKey(sp,text));
    if(buf){const src=curSrc=actx.createBufferSource();src.buffer=buf;src.connect(actx.destination);
      src.onended=()=>{curSrc=null;if(opts.onend)opts.onend();};src.start();if(opts.onstart)opts.onstart();
      const d=$('#delivrep');if(d)d.textContent='played from the voice bank (neutral, natural pace)';}
    else{const q=`text=${encodeURIComponent(text)}&voice=${encodeURIComponent(sp)}&_=${Date.now()}`;  // not banked yet: streamed, then banked
      const url=plain?`/api/tts_stream?${q}`:`/api/say?${q}&emotion=${emo}&intensity=${inten}&rate=${rate}&utt_id=${own?last.utt_id:0}&retime=${pace?1:0}`;
      let started=false;const d=$('#delivrep');if(d&&plain)d.textContent='streamed from ElevenLabs (not in the voice bank yet)';
      player.onended=()=>{player.onended=null;if(plain)bankLoad(sp,text).catch(e=>toast('Voice bank: could not add a phrase: '+e.message));if(opts.onend)opts.onend();};
      player.onplaying=()=>{started=true;if(opts.onstart)opts.onstart();};
      // onerror is the single fallback path: it fires only for the current src (a superseded load is aborted, not errored);
      // once audio has played, repeating the whole phrase in the browser voice would say it twice
      player.src=url;player.onerror=()=>{if(started)toast('The voice stopped mid-phrase (ElevenLabs failed)');else speakBrowser(text,opts,true);};
      player.play().catch(e=>{  // AbortError: a newer speak() replaced this src; NotSupportedError: the source failed and onerror fell back
        if(e.name==='NotAllowedError')toast('Click anywhere on the page once to enable audio, then try again.');else if(e.name!=='AbortError'&&e.name!=='NotSupportedError')toast('Audio playback failed: '+e.message);});}
    if(!$('#conf').querySelector('.voicepill'))$('#conf').insertAdjacentHTML('beforeend',`<span class="pill ctx voicepill">voice: ${esc(sp)}</span>`);}
  else speakBrowser(text,opts);}
// Neutral speech at the natural pace in an ElevenLabs voice: every phrase the server has cached is decoded into a Web Audio
// buffer when the voice is picked, so it starts at once instead of after a fetch and decode. Other text joins once played.
const bankKey=(sp,text)=>sp+'|'+text.trim().toLowerCase();
['pointerdown','keydown'].forEach(e=>window.addEventListener(e,()=>{if(actx.state!=='running')actx.resume();},{capture:true}));  // autoplay policy
async function bankLoad(sp,text){const r=await fetch(`/api/tts?text=${encodeURIComponent(text)}&voice=${encodeURIComponent(sp)}&cached_only=1`);
  if(r.ok&&bankVoice===sp)bank.set(bankKey(sp,text),await actx.decodeAudioData(await r.arrayBuffer()));}
async function fillBank(sp,reload){  // cached phrases only: synthesizing the rest here would spend ElevenLabs credits on every page load
  if(!sp||!phrases.length||(bankVoice===sp&&!reload))return;bankVoice=sp;bank.clear();const todo=phrases.slice(),t0=performance.now();let failed=0;
  await Promise.all(Array.from({length:6},async()=>{while(todo.length&&bankVoice===sp)await bankLoad(sp,todo.shift()).catch(()=>failed++);}));
  if(bankVoice!==sp)return;
  $('#bankstat').textContent=`${bank.size}/${phrases.length} phrases play instantly`+(failed?` · ${failed} failed to load`:'');$('#bankstat').title=`voice bank loaded in ${((performance.now()-t0)/1000).toFixed(1)} s`;
  if(failed)toast(`Voice bank: ${failed} phrases failed to load; they will stream from ElevenLabs instead`);}
function loadCloned(){fetch('/api/voices').then(r=>r.json()).then(j=>{cloned=j.voices||[];const sel=$('#voice');const add=(val,label)=>{if(![...sel.options].some(o=>o.value===val)){const o=document.createElement('option');o.value=val;o.textContent=label;sel.insertBefore(o,sel.firstChild);}};(j.stock||[]).slice().reverse().forEach(v=>add('clone:'+v.name,`☁ ${v.name} (ElevenLabs)`));cloned.forEach(v=>add('clone:'+v.speaker,`🎙 ${v.speaker} (my voice)`));if(j.selected&&j.available){sel.value='clone:'+j.selected;}if(sel.value.startsWith('clone:'))fillBank(sel.value.slice(6));});}
$('#voice').addEventListener('change',e=>{const v=e.target.value;send({cmd:'settings',voice:v.startsWith('clone:')?v.slice(6):null});if(v.startsWith('clone:'))fillBank(v.slice(6));});

// ---------- listen button + keyboard (space = listen, Y/N = answer "sounds like", Esc = dismiss alert)
const lb=$('#listen');let held=false;
function down(e){if(held)return;held=true;lb.classList.add('on');send({cmd:'start'});e&&e.preventDefault&&e.preventDefault();}
function up(e){if(!held)return;held=false;lb.classList.remove('on');send({cmd:'stop'});e&&e.preventDefault&&e.preventDefault();}
lb.addEventListener('mousedown',down);lb.addEventListener('touchstart',down,{passive:false});
window.addEventListener('mouseup',up);lb.addEventListener('touchend',up);
const typing=()=>/INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName);
window.addEventListener('keydown',e=>{if(typing())return;
  if(e.code==='Space'&&!e.repeat)down(e);
  else if(e.key==='Escape')$('#alert').classList.remove('on');
  else if(e.key==='y'||e.key==='n')answerQuestion(e.key==='y');});
window.addEventListener('keyup',e=>{if(e.code==='Space'&&!typing())up(e);});

// ---------- nurse
const NURSE_LBL='Nurse';
$('#nurse').onclick=()=>{$('#nurse').classList.add('on');$('#nurselbl').textContent='Listening…';send({cmd:'nurse',seconds:5});};  // until the nurse pauses (5 s at most): setState resets it
$('#nursesend').onclick=()=>{const t=$('#nursetext').value.trim();if(t){send({cmd:'nurse_text',text:t});$('#nursetext').value='';}};
$('#nursetext').addEventListener('keydown',e=>{if(e.key==='Enter')$('#nursesend').click();});
// ---------- corrections for data/captures: "what was actually said" for the latest utterance (scripts/captures.py)
let lbl={utt:null,text:''};
function labelFor(uid,text){if(DEMO||!uid)return;if(lbl.utt!==uid){$('#lbltext').value='';$('#lblstat').textContent='';}lbl={utt:uid,text:text||lbl.text};$('#labelrow').hidden=false;}
function sendLabel(text){text=(text||'').trim();if(!text||lbl.utt==null)return;send({cmd:'label',utt_id:lbl.utt,text});$('#lblstat').textContent='saving…';}
$('#lblok').onclick=()=>sendLabel(lbl.text);
$('#lblsave').onclick=()=>sendLabel($('#lbltext').value);
$('#lbltext').addEventListener('keydown',e=>{if(e.key==='Enter')sendLabel($('#lbltext').value);});

// ---------- tabs / mode / settings / context
$$('#tabs button').forEach(b=>b.onclick=()=>{$$('#tabs button').forEach(x=>x.classList.toggle('on',x===b));$$('.tab').forEach(t=>t.classList.toggle('on',t.id==='tab-'+b.dataset.tab));});
$$('#mode button').forEach(b=>b.onclick=()=>{mode=b.dataset.mode;send({cmd:'mode',mode});syncMode();});
$('#expressive').onchange=e=>send({cmd:'settings',expressive:e.target.checked});
$('#llm').onchange=e=>send({cmd:'settings',llm_enabled:e.target.checked});
$('#override').onchange=e=>send({cmd:'settings',emotion_override:e.target.value||null});
$('#pace').onchange=e=>send({cmd:'settings',pace:e.target.checked});  // the server loads the voice aligner it needs
$('#autolisten').onchange=e=>{send({cmd:'settings',auto_listen:e.target.checked});lb.style.opacity=e.target.checked?.45:1;lb.innerHTML=e.target.checked?'HANDS-FREE · just mouth a phrase<small>auto-detects mouth movement · or hold to force</small>':'HOLD TO LISTEN<small>or hold the space bar · hands-free available in Dev</small>';};
let ctxT;function pushCtx(){clearTimeout(ctxT);ctxT=setTimeout(()=>send({cmd:'context',notes:$('#notes').value,category:$('#category').value,last_prompt:$('#prompt').value}),250)}
['#notes','#prompt'].forEach(s=>$(s).addEventListener('input',pushCtx));$('#category').addEventListener('change',()=>{curCategory=$('#category').value;$$('#chips button').forEach(x=>x.classList.toggle('on',x.textContent===curCategory));pushCtx();});
function applyCtx(c){if(document.activeElement!==$('#notes'))$('#notes').value=c.notes||'';if(document.activeElement!==$('#prompt'))$('#prompt').value=c.last_prompt||'';$('#category').value=c.category||'';curCategory=c.category||'';$$('#chips button').forEach(x=>x.classList.toggle('on',x.textContent===curCategory));$('#hist').innerHTML=(c.history||[]).map(h=>`<span>${esc(h)}</span>`).join('')||'<span class="small">no history yet</span>';
  if(c.last_prompt!=null&&c.last_prompt!==$('#asked').textContent)showAsked(c.last_prompt);}
$('#nbestbtn').onclick=()=>send({cmd:'nbest'});
$('#save').onclick=()=>{const ph=prompt('Intended phrase for this sample:',last?last.selected:'');if(ph)send({cmd:'save_sample',phrase:ph,speaker:$('#speaker').value||'unknown'})};

// ---------- clone
let rec=null,recChunks=[],recTimer=null;
const CLONE_SCRIPT=["I need water. I am in pain. Please call the nurse.","The soft cushion broke the man's fall. Rice is often served in round bowls.","I did not sleep well last night, but I feel better this morning.","Can you please raise the head of the bed? The light is too bright in here.","Thank you for taking care of me today. When can I go home?","The juice of lemons makes fine punch. Four hours of steady work faced us."];
$('#clonebtn').onclick=async()=>{
  if(rec){rec.stop();return;}
  const name=prompt('Your name (used to label the voice):',$('#speaker').value||'');if(!name)return;$('#speaker').value=name;
  try{const stream=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true,noiseSuppression:true}});
    rec=new MediaRecorder(stream,{mimeType:'audio/webm;codecs=opus'});recChunks=[];rec.ondataavailable=e=>recChunks.push(e.data);
    let t=0,i=0;const st=$('#clonestat');$('#clonebtn').textContent='■ Stop (≥60 s)';
    const tick=()=>{st.innerHTML=`<b>read aloud:</b> “${CLONE_SCRIPT[i%CLONE_SCRIPT.length]}” · ${t}s`;if(t%12===11)i++;t++;};tick();recTimer=setInterval(tick,1000);
    rec.onstop=async()=>{clearInterval(recTimer);stream.getTracks().forEach(x=>x.stop());rec=null;$('#clonebtn').textContent='🎙 Clone my voice';
      const blob=new Blob(recChunks,{type:'audio/webm'});st.textContent=`uploading ${(blob.size/1024).toFixed(0)} KB, cloning…`;
      const r=await fetch(`/api/voice/clone_upload?speaker=${encodeURIComponent(name)}`,{method:'POST',body:blob});const j=await r.json();
      if(j.error){st.textContent='clone failed: '+j.error;return;}
      st.textContent=`cloned ✓ (${j.audio_seconds}s audio) · pre-synthesizing phrase bank…`;loadCloned();setTimeout(()=>{$('#voice').value='clone:'+name;fillBank(name,true);},800);};  // a new clone: the old one's audio must go
    rec.start(1000);
  }catch(e){$('#clonestat').textContent='mic error: '+e;}
};

if(!DEMO){$('#cam').src='/stream';connect();}
