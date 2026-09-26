// Silent Running bedside UI. Events arrive over /ws (event contract in .conductor/AGENTS.md) and 10 Hz tracking over /ws_meta.
// ?demo=1 swaps the transport for a scripted replay (demo.js), so everything here renders without the backend or a camera.
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const DEMO=new URLSearchParams(location.search).has('demo');
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const fmtPct=p=>(p*100).toFixed(0)+'%';
const ms=s=>(s*1000).toFixed(0)+' ms';
let ws,wsm,mode='phrase',last=null,phrases=[],phraseTable=[],curCategory='',decisionSeen=false;
const voices={list:[]};
let spoken={};  // utt_id -> text already spoken, so result + decision + confirm never double-speak (reset per server connection)

// ---------- transport (demo.js replaces transport.send)
const transport={send(o){if(ws&&ws.readyState===1)ws.send(JSON.stringify(o))}};
function send(o){transport.send(o)}
function connect(){
  ws=new WebSocket(`ws://${location.host}/ws`);
  ws.onmessage=e=>handle(JSON.parse(e.data));
  ws.onclose=()=>setTimeout(connect,1000);
  wsm=new WebSocket(`ws://${location.host}/ws_meta`);
  wsm.onmessage=e=>meta(JSON.parse(e.data));
}

// ---------- live tracking (10 Hz)
let mouthingSig=false;
function meterUpdate(m){const ex=m.expression||{};['angry','warm','sad'].forEach(e=>{const el=document.getElementById('m_'+e);if(el)el.style.width=Math.min(100,(ex[e]||0)*100)+'%';});}
function meta(m){
  meterUpdate(m);
  const d=$('#dot'),t=$('#track');
  if(m.listening){d.className='dot rec';}else d.className='dot'+(m.face?' ok':'');
  if(!m.face)t.textContent='no face — look at the camera';
  else{let q=m.face_frac<0.09?'move closer':(m.face_frac>0.3?'a bit further':'good distance');
    t.textContent=(m.listening?`listening · ${m.n_frames} frames`:`tracking mouth · ${q}`)+` · ${(m.fps||0).toFixed(0)} fps`+(m.auto?` · motion ${m.energy.toFixed(1)}/${m.noise.toFixed(1)}`:'');}
  $('#camwrap').classList.toggle('listening',!!m.listening);
  setMouthing(mouthingSig||!!m.mouth_active,false);
  drawOverlay(m);
}
function drawOverlay(m){
  const g=$('#ovbox');
  if(!m.face||!m.bbox){g.innerHTML='';return;}
  const W=m.frame_w||640,H=m.frame_h||480;$('#overlay').setAttribute('viewBox',`0 0 ${W} ${H}`);
  let [x1,y1,x2,y2]=m.bbox;[x1,x2]=[W-x2,W-x1];  // the preview is mirrored, the bbox is not
  const L=Math.min(x2-x1,y2-y1)*.28,c=m.listening||m.mouth_active?'#5aa9ff':'#3ddc97';
  const p=`M${x1},${y1+L}V${y1}H${x1+L}M${x2-L},${y1}H${x2}V${y1+L}M${x2},${y2-L}V${y2}H${x2-L}M${x1+L},${y2}H${x1}V${y2-L}`;
  g.innerHTML=`<path d="${p}" stroke="${c}" style="vector-effect:non-scaling-stroke;stroke-width:3.5"/><text class="lbl" x="${x1}" y="${y1-W/80}" fill="${c}" font-size="${W/42}">${m.listening?'READING LIPS':'MOUTH'}</text>`;
}

// ---------- status
function setState(s,extra){
  const el=$('#state');el.className='state '+s;el.textContent=(s==='nurse_listening'?'nurse speaking':s)+(extra?' · '+extra:'');
  const p=$('#statuspill');p.className='statuspill '+s;p.querySelector('span').textContent=s==='nurse_listening'?'nurse speaking':s;
  const h=$('#hero');
  if(s==='listening'&&h.dataset.state!=='confirm'){$('#eyebrow').textContent='Listening · mouth the phrase';}
  else if(s==='processing'){stopCountdown();cf=null;h.dataset.state='processing';$('#eyebrow').textContent='Reading lips…';}
  else if(s==='idle'&&h.dataset.state==='processing'){h.dataset.state=last?'result':'idle';}
}

// ---------- event dispatch
function handle(m){
  logEvent(m);
  if(m.type==='hello'){spoken={};decisionSeen=false;phrases=m.phrases;phraseTable=m.phrase_table||[];applyCtx(m.context);mode=m.state.mode;syncMode();if(m.state.expressive!=null)$('#expressive').checked=m.state.expressive;
    $('#nphr').textContent=phrases.length+' phrases';$('#phrlist').innerHTML=phrases.map(p=>`<span>${esc(p)}</span>`).join('');buildChips();(m.log||[]).forEach(addLog);
    if(!DEMO)fetch('/api/state').then(r=>r.json()).then(s=>{$('#engine').textContent=`${s.engine.model} · ${phrases.length} phrases`;});}
  else if(m.type==='status'){setState(m.status,m.stage);if(m.status==='processing'&&m.stage==='crop')toast('');}
  else if(m.type==='raw'){$('#raw').innerHTML=`<span class="lbl">raw (CTC greedy)</span>${esc(m.text)||'<span class="small">(nothing)</span>'}`;$('#nbest').innerHTML='';$('#lat').textContent=`crop ${ms(m.latency.crop)} · encode ${ms(m.latency.encode)} · ${m.n_frames} frames (${m.duration.toFixed(1)} s)`;}
  else if(m.type==='result'){last=m;render(m);if(!decisionSeen&&!(m.mode==='open'&&$('#llm').checked)&&!m.critical)speakOnce(m.utt_id,m.selected);}
  else if(m.type==='decision'){onDecision(m);}
  else if(m.type==='confirm'){onConfirm(m);}
  else if(m.type==='signal'){onSignal(m);}
  else if(m.type==='alert'){showAlert(m);}
  else if(m.type==='nbest'){renderNbest(m.nbest);}
  else if(m.type==='llm'){renderLLM(m);}
  else if(m.type==='delivery'){const d=$('#delivrep');if(d){d.innerHTML=`delivered <b>${m.emotion}</b>${m.intensity?` ${(m.intensity*100).toFixed(0)}%`:''} · ${m.model}${m.tag?` · tag <code>${esc(m.tag)}</code>`:''} · stability ${m.stability} · speed ${m.rate.toFixed(2)}x · synth ${m.cached?'cached':m.t_synth+' s'}${m.retime&&m.retime.applied?` · retimed (global ${m.retime.global}x)`:(m.retime&&m.retime.reason?` · no retime: ${esc(m.retime.reason)}`:'')} · total ${m.total} s`;}renderAudioStrip(m.retime);}
  else if(m.type==='log'){addLog(m.entry);}
  else if(m.type==='error'){toast(m.message);setState('idle');}
  else if(m.type==='context'){applyCtx(m.context);}
  else if(m.type==='state'){mode=m.state.mode;syncMode();}
  else if(m.type==='prewarmed'){$('#clonestat').textContent=`voice “${m.speaker}” ready · ${m.n} phrase variants cached`;}
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
function addLog(e){const L=$('#log');if(L.querySelector('.small'))L.innerHTML='';const d=document.createElement('div');d.className='msg '+e.who+(e.critical?' critical':'');const t=new Date(e.ts*1000).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'});
  d.innerHTML=`<span class="t">${esc(e.text)}</span><span class="meta">${e.who} · ${t}${e.confidence?` · ${fmtPct(e.confidence)}`:''}${e.source?` · ${esc(e.source)}`:''}${e.emotion&&e.emotion!=='neutral'?` · ${esc(e.emotion)}`:''}</span>`;L.appendChild(d);L.scrollTop=L.scrollHeight;
  if(e.who==='nurse')showAsked(e.text);}
function showAsked(t){const q=$('#asked');q.className='q'+(t?'':' empty');q.textContent=t||'No question yet';if(t){void q.offsetWidth;q.classList.add('fresh');}}
function showAlert(m){$('#alerttext').textContent=m.text.toUpperCase();const a=$('#alert');a.classList.add('on');
  spoken[m.utt_id]=m.text;  // the urgent double announcement owns this utterance; a later decision must not cut it off
  speak(m.text,{emotion:'urgent',onend:()=>speak(m.text,{emotion:'urgent'})});
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
function heroSet(state,src){const h=$('#hero');h.dataset.state=state;if(src)h.dataset.src=src;$('#src').textContent=src||h.dataset.src;}
function render(m){
  heroSet('result','lips');setBig(m.selected);setRing(m.confidence);
  $('#eyebrow').textContent=`${SRC_LABEL.lips} · ${m.mode==='open'?'open vocabulary':'phrase match'} · ${ms(m.latency.total)}`;
  $('#reason').textContent=m.mode==='phrase'?(m.context_changed_choice?`Context changed the choice: the visual top was “${m.visual_top}”.`:''):'';
  $('#alts').innerHTML='';stopCountdown();cf=null;
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
  const cd=$('#cands');cd.innerHTML='';
  if(m.mode==='phrase'){
    const rows=m.ranking.filter(r=>!r.prefiltered_out);const mx=Math.max(...rows.map(r=>r.final_prob));
    const mb=Math.max(...rows.slice(0,5).map(r=>Math.max(r.final_prob,r.vsr_prob)));
    rows.slice(0,5).forEach((r,i)=>{const b=document.createElement('button');if(i===0)b.className='top';
      b.innerHTML=`<span>${esc(r.phrase)}${r.reasons.length?`<b class="why">+${r.prior.toFixed(1)} ${esc(r.reasons.join(', '))}</b>`:''}</span><small>${fmtPct(r.final_prob)}</small><span class="bars"><i class="bv" style="width:${(r.vsr_prob/mb*100).toFixed(1)}%"></i><i class="bc" style="width:${(r.final_prob/mb*100).toFixed(1)}%"></i></span>`;
      b.title=`visual-only ${fmtPct(r.vsr_prob)} → with context ${fmtPct(r.final_prob)}`;
      b.onclick=()=>{speak(r.phrase);send({cmd:'confirm',text:r.phrase});setBig(r.phrase);};dym.appendChild(b);});
    rows.forEach((r,i)=>{const div=document.createElement('div');div.className='cand';const tag=r.reasons.length?`<b class="tag">+${r.prior.toFixed(1)} ${esc(r.reasons.join(', '))}</b>`:'';
      div.innerHTML=`<div class="idx">${i+1}</div><div class="bar"><i class="vis" style="width:${(r.vsr_prob/mx*100).toFixed(1)}%"></i><i style="width:${(r.final_prob/mx*100).toFixed(1)}%;opacity:.8"></i><span>${esc(r.phrase)}</span>${tag}</div><div class="pct">${fmtPct(r.final_prob)}</div>`;
      div.title=`VSR log-lik ${r.vsr_score.toFixed(2)} (att ${(r.att||0).toFixed(1)}, ctc ${r.ctc.toFixed(1)}) · visual-only ${fmtPct(r.vsr_prob)} · prior +${r.prior.toFixed(2)}`;cd.appendChild(div);});
    $('#llmpanel').style.display='none';
  }else{
    m.nbest.slice(0,4).forEach((h,i)=>{const b=document.createElement('button');if(i===0)b.className='top';b.innerHTML=`<span>${esc(pretty(h.text))||'(empty)'}</span><small>${fmtPct(h.prob)}</small>`;b.onclick=()=>speak(pretty(h.text));dym.appendChild(b);});
    renderNbest(m.nbest,true);
    $('#llmpanel').style.display=$('#llm').checked?'':'none';$('#llmout').innerHTML='<span class="small">LLM proposing a correction from the visual hypotheses + context; the visual model will verify it…</span>';
  }
  const L=m.latency;$('#lat').textContent=Object.entries(L).map(([k,v])=>`${k} ${ms(v)}`).join(' · ')+` · ${m.n_frames} frames (${m.duration.toFixed(1)} s)`;
  latResult(m);
  $('#vitals').innerHTML=`<span>server <b>${ms(L.total)}</b></span><span>face <b>${esc((m.expression||{}).emotion||'neutral')}</b></span><span>mouthed <b>${m.timing?m.timing.duration.toFixed(2)+' s':'—'}</b></span><span>mode <b>${m.mode}</b></span><span>utt <b>#${m.utt_id}</b></span>`;
}
function onDecision(d){
  decisionSeen=true;if(last&&d.utt_id===last.utt_id)latMark('decision');
  const confirmingThis=cf&&cf.utt_id===d.utt_id&&d.action!=='confirm';
  if(!confirmingThis){if(cf&&cf.utt_id!==d.utt_id){stopCountdown();cf=null;}
    heroSet('result',d.source||'fused');setBig(d.text);setRing(d.confidence);
    $('#eyebrow').textContent=`${SRC_LABEL[d.source]||d.source}${d.provider?' · '+d.provider:''}`;
    $('#reason').textContent=d.reason||'';
  }
  if(!last||d.utt_id!==last.utt_id){  // no lip result behind this decision (gesture fast path): drop the previous utterance's evidence
    $('#conf').innerHTML='';const dym=$('#dym');dym.innerHTML='';
    [{text:d.text,confidence:d.confidence},...(d.alternatives||[])].forEach((a,i)=>{const b=document.createElement('button');if(i===0)b.className='top';
      b.innerHTML=`<span>${esc(a.text)}</span><small>${fmtPct(a.confidence)}</small>`;b.onclick=()=>{speak(a.text);send({cmd:'confirm',text:a.text});setBig(a.text);};dym.appendChild(b);});
  }
  $('#alts').innerHTML=(d.alternatives||[]).length?'Also possible: '+d.alternatives.slice(0,3).map((a,i)=>`<span data-i="${i}" style="cursor:pointer">${esc(a.text)} ${fmtPct(a.confidence)}</span>`).join(' · '):'';
  $$('#alts span').forEach(s=>s.onclick=()=>{const a=d.alternatives[+s.dataset.i];speak(a.text);send({cmd:'confirm',text:a.text});setBig(a.text);});
  $('#decision').innerHTML=`<div class="dec"><div class="t">${esc(d.text)} <span class="pill">${esc(d.source)}</span> <span class="pill">${esc(d.action)}</span></div><div class="small">${esc(d.reason)}</div>
    <table><tr><td>utt</td><td>#${d.utt_id}</td></tr><tr><td>confidence</td><td>${fmtPct(d.confidence)}</td></tr><tr><td>provider</td><td>${esc(d.provider)}</td></tr>${(d.alternatives||[]).map(a=>`<tr><td>alt</td><td>${esc(a.text)} · ${fmtPct(a.confidence)}</td></tr>`).join('')}</table></div>`;
  if(d.action==='confirm')onConfirm({utt_id:d.utt_id,candidate:d.text,attempt:1,state:'asking'});
  else if(d.action==='speak')speakOnce(d.utt_id,d.text);
  else $('#eyebrow').textContent='Not sure · no answer spoken';
}

// ---------- confirmation loop ("Sounds like X?", nod = yes, shake = next)
let cf=null;const CONFIRM_MS=4000;
function onConfirm(c){
  if(c.state==='asking'?(last&&c.utt_id<last.utt_id):(!cf||cf.utt_id!==c.utt_id))return;  // stale: a newer utterance owns the hero
  if(c.state==='asking'){
    heroSet('confirm');setBig(c.candidate,{question:true});
    $('#eyebrow').textContent=`Checking with the patient · guess ${c.attempt}`;
    $('#confirmhint').textContent='Nod to confirm · shake for the next guess';
    const key=`${c.utt_id}:${c.attempt}`;
    if(!cf||cf.key!==key){cf={key,utt_id:c.utt_id,candidate:c.candidate};startCountdown();if($('#autospeak').checked)speak(`Sounds like: ${c.candidate}?`);}
  }else if(c.state==='rejected'){
    $('#confirmhint').textContent=`✗ Not “${c.candidate}” · trying the next guess`;
  }else if(c.state==='confirmed'){
    stopCountdown();heroSet('confirmed');setBig(c.candidate);$('#eyebrow').textContent='Confirmed by the patient ✓';setRing(1);cf=null;
    spoken[c.utt_id]=null;speakOnce(c.utt_id,c.candidate);
  }else if(c.state==='timeout'){
    stopCountdown();heroSet('result');setBig(c.candidate);$('#eyebrow').textContent='No answer · best guess, unconfirmed';cf=null;
  }
}
function startCountdown(){stopCountdown();
  cf.anims=[$('#cdfill').animate([{transform:'scaleX(1)'},{transform:'scaleX(0)'}],{duration:CONFIRM_MS,fill:'forwards'}),
    $('#ringcd').animate([{strokeDashoffset:0},{strokeDashoffset:276.5}],{duration:CONFIRM_MS,fill:'forwards'})];}
function stopCountdown(){if(cf&&cf.anims)cf.anims.forEach(a=>a.cancel());}
function answerConfirm(yes){if(!cf)return;send({cmd:'confirm_answer',utt_id:cf.utt_id,answer:yes?'yes':'no'});if(yes)send({cmd:'confirm',text:cf.candidate});}
$('#cyes').onclick=()=>answerConfirm(true);$('#cno').onclick=()=>answerConfirm(false);

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
  else if(s.kind==='thumb')setChip('hand',`Thumb ${s.value==='down'?'down':'up'}${c}`);
  else if(s.kind==='point')setChip('hand',`Pointing${s.value&&s.value!==true?' · '+esc(s.value):''}${c}`);
  else if(s.kind==='pain')setPain(+s.value||0,true);
  else if(s.kind==='mouthing')setMouthing(!!s.value);
}
function nonverbalPills(nv){
  if(!nv)return '';const p=[];
  if(nv.head&&nv.head.value)p.push(`head: ${nv.head.value} ${fmtPct(nv.head.confidence||0)}`);
  if(nv.blink_code&&nv.blink_code.value)p.push(`blink: ${nv.blink_code.value}`);
  if(nv.fingers&&nv.fingers.value!=null)p.push(`fingers: ${nv.fingers.value}`);
  if(nv.pain&&nv.pain.value>0.15)p.push(`pain ${Math.round(nv.pain.value*10)}/10`);
  if(nv.emotion&&nv.emotion.label&&nv.emotion.label!=='neutral')p.push(`face: ${nv.emotion.label} ${fmtPct(nv.emotion.intensity||0)}`);
  return p.map(x=>`<span class="pill nv">${esc(x)}</span>`).join('');
}
function applyNonverbal(nv){
  if(!nv)return;
  if(nv.head&&nv.head.value)setChip('head',`${nv.head.value==='yes'?'YES · nod':'NO · shake'}`,false);
  if(nv.blink_code&&nv.blink_code.value)setChip('blink',esc(nv.blink_code.value.toUpperCase()),false);
  if(nv.fingers&&nv.fingers.value!=null)setChip('fingers',esc(nv.fingers.value),false);
  if(nv.pain&&nv.pain.value>0.15)setPain(nv.pain.value,false);
}

// ---------- latency: server stages + decision + time to first audio
const lat={t:{},server:null};
const LAT_COLORS={crop:'#56677b',encode:'#5aa9ff',phrase:'#3ddc97',beam:'#3ddc97',decide:'#b18cff',voice:'#ffb547'};
function latResult(m){lat.t={result:performance.now()};lat.server=m.latency;renderLatency();}
function latMark(k){if(lat.t.result&&!lat.t[k]){lat.t[k]=performance.now();renderLatency();}}
function renderLatency(){
  if(!lat.server)return;
  const segs=Object.entries(lat.server).filter(([k])=>k!=='total');
  if(lat.t.decision)segs.push(['decide',(lat.t.decision-lat.t.result)/1000]);
  if(lat.t.audio)segs.push(['voice',(lat.t.audio-(lat.t.decision&&lat.t.decision<lat.t.audio?lat.t.decision:lat.t.result))/1000]);
  const tot=segs.reduce((a,[,v])=>a+v,0),bar=$('#latbar');
  bar.querySelectorAll('.seg').forEach(e=>e.remove());
  segs.forEach(([k,v])=>{const e=document.createElement('div');e.className='seg';e.style.width=Math.min(100,v/3*100)+'%';e.style.background=LAT_COLORS[k]||'#8b9cb0';e.title=`${k} ${ms(v)}`;bar.insertBefore(e,bar.querySelector('.target'));});
  $('#latlegend').innerHTML=segs.map(([k,v])=>`<span><i style="background:${LAT_COLORS[k]||'#8b9cb0'}"></i>${k} ${ms(v)}</span>`).join('')+(lat.t.audio?'':'<span>voice …</span>');
  const T=$('#lattotal');T.textContent=`${tot.toFixed(2)} s${lat.t.audio?'':' +'}`;T.className='lattotal '+(tot<2?'ok':'slow');
}

// ---------- dev panels
function renderDelivery(m){
  const ex=m.expression||{emotion:'neutral',intensity:0},tm=m.timing;
  let h=`<span class="pill ${ex.emotion==='neutral'?'':'ctx'}">face: ${esc(ex.emotion)}${ex.intensity?` ${(ex.intensity*100).toFixed(0)}%`:''}</span>`;
  if(ex.scores)h+=` <span class="small">${Object.entries(ex.scores).map(([k,v])=>`${k} ${v.toFixed(2)}`).join(' · ')}</span>`;
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
  if(intoCands){const cd=$('#cands');cd.innerHTML='';const mx=Math.max(...nb.map(h=>h.prob));
    nb.forEach((h,i)=>{const div=document.createElement('div');div.className='cand';div.innerHTML=`<div class="idx">${i+1}</div><div class="bar"><i style="width:${(h.prob/mx*100).toFixed(1)}%"></i><span>${esc(pretty(h.text))||'(empty)'}</span><b class="tag">${h.score.toFixed(1)}</b></div><div class="pct">${fmtPct(h.prob)}</div>`;div.onclick=()=>speak(pretty(h.text));cd.appendChild(div);});
  }else{$('#nbest').innerHTML=`<span class="lbl">beam n-best</span>`+nb.map(h=>`<div>${h.score.toFixed(2)}&nbsp; ${esc(h.text)||'(empty)'}</div>`).join('');}
}
function renderLLM(m){
  if(!last||m.utt_id!==last.utt_id)return;
  if(m.error){$('#llmout').innerHTML=`<span class="err">LLM unavailable: ${esc(m.error)}</span>`;speakOnce(last.utt_id,last.selected);return;}
  const raw=pretty(last.nbest[0].text);let verdict;
  if(!m.changed)verdict=`<span class="pill ok">LLM agrees with the visual model</span>`;
  else if(m.accepted)verdict=`<span class="pill ok">accepted · video supports it (${m.gap.toFixed(1)} nats vs raw)</span>`;
  else verdict=`<span class="pill warn">rejected · video does not support it (${m.gap.toFixed(1)} nats vs raw) · kept raw</span>`;
  $('#llmout').innerHTML=`<div><b>${esc(m.corrected)}</b> ${verdict} <span class="small">· ${esc(m.model)} · ${ms(m.latency)}</span></div>`+(m.changed?`<div class="why">LLM proposed “${esc(m.proposal)}” instead of “${esc(raw)}”. ${esc(m.reason)}</div>`:`<div class="why">${esc(m.reason)}</div>`);
  if(m.changed&&m.accepted){setBig(m.corrected);heroSet('result','fused');$('#conf').insertAdjacentHTML('beforeend',`<span class="pill ctx">context-corrected: ${esc(wordDiff(raw,m.corrected))} · verified by the visual model</span>`);}
  speakOnce(last.utt_id,(m.changed&&m.accepted)?m.corrected:last.selected);
}
function wordDiff(a,b){const A=a.split(' '),B=b.split(' ');let i=0;while(i<A.length&&i<B.length&&A[i].toLowerCase()===B[i].toLowerCase())i++;let j=0;while(j<A.length-i&&j<B.length-i&&A[A.length-1-j].toLowerCase()===B[B.length-1-j].toLowerCase())j++;const x=A.slice(i,A.length-j).join(' ')||'∅',y=B.slice(i,B.length-j).join(' ')||'∅';return `“${x}” → “${y}”`;}
function pretty(s){s=(s||'').toLowerCase().replace(/\bi\b/g,'I').replace(/\bi'/g,"I'");return s.charAt(0).toUpperCase()+s.slice(1)}

// ---------- toast
function toast(msg,good){const t=$('#err');clearTimeout(toast.t);if(!msg){t.classList.remove('on');return;}t.textContent=msg;t.className='toast on'+(good?' good':'');toast.t=setTimeout(()=>t.classList.remove('on'),good?3000:6000);}

// ---------- voice
function loadVoices(){voices.list=speechSynthesis.getVoices().filter(v=>v.lang.startsWith('en'));}
speechSynthesis.onvoiceschanged=()=>{loadVoices();if(!DEMO)loadCloned();};loadVoices();if(!DEMO)setTimeout(loadCloned,500);
let cloned=[];const player=new Audio();
player.addEventListener('playing',()=>latMark('audio'));
function speakOnce(uid,text,opts){if(!text||!$('#autospeak').checked||spoken[uid]===text)return;spoken[uid]=text;speak(text,{...opts,utt_id:uid});}
function speakBrowser(text,opts,fallback){speechSynthesis.cancel();const u=new SpeechSynthesisUtterance(text);const v=voices.list.find(v=>v.name==='Samantha')||voices.list[0];if(v)u.voice=v;
  u.onstart=()=>latMark('audio');if(opts&&opts.onend)u.onend=opts.onend;speechSynthesis.speak(u);if(fallback)toast('ElevenLabs unavailable, used browser voice');}
function speak(text,opts){if(!text)return;const sel=$('#voice').value||'';opts=opts||{};
  if(sel.startsWith('clone:')){const sp=sel.slice(6);player.pause();
    // the mouthed face + word timing only apply when speaking the lip result's own phrase (not a gesture answer or a "Sounds like" prompt)
    const own=last&&opts.utt_id===last.utt_id&&text===last.selected;
    const ex=(opts.expression)||(own&&last.expression)||{emotion:'neutral',intensity:0};const tm=(own&&last.timing)||{};
    let emo=opts.emotion||ex.emotion||'neutral',inten=opts.intensity!=null?opts.intensity:(ex.intensity||0),rate=tm.rate||1;
    if(emo==='urgent'){emo='angry';inten=0.7;rate=1.1;}
    const expressive=$('#expressive').checked;
    const url=expressive?`/api/say?text=${encodeURIComponent(text)}&emotion=${emo}&intensity=${inten}&rate=${rate}&voice=${encodeURIComponent(sp)}&utt_id=${own?last.utt_id:0}&_=${Date.now()}`:`/api/tts?text=${encodeURIComponent(text)}&voice=${encodeURIComponent(sp)}&_=${Date.now()}`;
    player.onended=()=>{player.onended=null;if(opts.onend)opts.onend();};
    player.src=url;player.onerror=()=>{speakBrowser(text,opts,true);};player.play().catch(e=>{if(e&&e.name==='NotAllowedError'){toast('Click anywhere on the page once to enable audio, then try again.');}else speakBrowser(text,opts,true);});
    if(!$('#conf').querySelector('.voicepill'))$('#conf').insertAdjacentHTML('beforeend',`<span class="pill ctx voicepill">voice: ${esc(sp)}</span>`);}
  else speakBrowser(text,opts);}
function loadCloned(){fetch('/api/voices').then(r=>r.json()).then(j=>{cloned=j.voices||[];const sel=$('#voice');const add=(val,label)=>{if(![...sel.options].some(o=>o.value===val)){const o=document.createElement('option');o.value=val;o.textContent=label;sel.insertBefore(o,sel.firstChild);}};(j.stock||[]).slice().reverse().forEach(v=>add('clone:'+v.name,`☁ ${v.name} (ElevenLabs)`));cloned.forEach(v=>add('clone:'+v.speaker,`🎙 ${v.speaker} (my voice)`));if(j.selected&&j.available){sel.value='clone:'+j.selected;}});}
$('#voice').addEventListener('change',e=>{const v=e.target.value;send({cmd:'settings',voice:v.startsWith('clone:')?v.slice(6):null});});

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
  else if($('#hero').dataset.state==='confirm'&&(e.key==='y'||e.key==='n'))answerConfirm(e.key==='y');});
window.addEventListener('keyup',e=>{if(e.code==='Space'&&!typing())up(e);});

// ---------- nurse
const NURSE_LBL='Nurse';
$('#nurse').onclick=()=>{const b=$('#nurse');b.classList.add('on');$('#nurselbl').textContent='Listening… 5 s';send({cmd:'nurse',seconds:5});setTimeout(()=>{b.classList.remove('on');$('#nurselbl').textContent=NURSE_LBL;},6500);};
$('#nursesend').onclick=()=>{const t=$('#nursetext').value.trim();if(t){send({cmd:'nurse_text',text:t});$('#nursetext').value='';}};
$('#nursetext').addEventListener('keydown',e=>{if(e.key==='Enter')$('#nursesend').click();});

// ---------- tabs / mode / settings / context
$$('#tabs button').forEach(b=>b.onclick=()=>{$$('#tabs button').forEach(x=>x.classList.toggle('on',x===b));$$('.tab').forEach(t=>t.classList.toggle('on',t.id==='tab-'+b.dataset.tab));});
$$('#mode button').forEach(b=>b.onclick=()=>{mode=b.dataset.mode;send({cmd:'mode',mode});syncMode();});
$('#autospeak').onchange=e=>send({cmd:'settings',auto_speak:e.target.checked,tts:'browser'});
$('#expressive').onchange=e=>send({cmd:'settings',expressive:e.target.checked});
$('#llm').onchange=e=>send({cmd:'settings',llm_enabled:e.target.checked});
$('#override').onchange=e=>send({cmd:'settings',emotion_override:e.target.value||null});
$('#autolisten').onchange=e=>{send({cmd:'settings',auto_listen:e.target.checked});lb.style.opacity=e.target.checked?.45:1;lb.innerHTML=e.target.checked?'HANDS-FREE · just mouth a phrase<small>auto-detects mouth movement · or hold to force</small>':'HOLD TO LISTEN<small>or hold the space bar · hands-free available in Dev</small>';};
let ctxT;function pushCtx(){clearTimeout(ctxT);ctxT=setTimeout(()=>send({cmd:'context',notes:$('#notes').value,category:$('#category').value,last_prompt:$('#prompt').value}),250)}
['#notes','#prompt'].forEach(s=>$(s).addEventListener('input',pushCtx));$('#category').addEventListener('change',()=>{curCategory=$('#category').value;$$('#chips button').forEach(x=>x.classList.toggle('on',x.textContent===curCategory));pushCtx();});
function applyCtx(c){if(document.activeElement!==$('#notes'))$('#notes').value=c.notes||'';if(document.activeElement!==$('#prompt'))$('#prompt').value=c.last_prompt||'';$('#category').value=c.category||'';curCategory=c.category||'';$$('#chips button').forEach(x=>x.classList.toggle('on',x.textContent===curCategory));$('#hist').innerHTML=(c.history||[]).map(h=>`<span>${esc(h)}</span>`).join('')||'<span class="small">no history yet</span>';
  if(c.last_prompt!=null&&c.last_prompt!==$('#asked').textContent)showAsked(c.last_prompt);}
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
      st.textContent=`cloned ✓ (${j.audio_seconds}s audio) · pre-synthesizing phrase bank…`;loadCloned();setTimeout(()=>{$('#voice').value='clone:'+name;},800);};
    rec.start(1000);
  }catch(e){$('#clonestat').textContent='mic error: '+e;}
};

if(!DEMO){$('#cam').src='/stream';connect();}
