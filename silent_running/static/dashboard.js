// Charge nurse unit board. State comes from GET /api/unit once, then every change from the server's websocket
// (unit, ascend and feeds events); times on screen tick locally every second. The two camera feeds and their labels follow
// the real sources: the glasses when they are connected, whatever stands in for them otherwise.
'use strict';
const $ = s => document.querySelector(s);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const LEVEL_WORD = {calm: 'Calm', request: 'Request', urgent: 'Urgent'};
const EMBED = new URLSearchParams(location.search).has('embed');
if (EMBED) document.body.classList.add('embed');

let U = {beds: [], alerts: [], metrics: {}, real_bed: null, ambient: null, feeds: null};
let A = {summary: {}, events: []};   // the Impiricus Ascend seam (simulated)
let L = [];                          // today's flat log
let logFilter = 'all';
let bedsById = {};
let openBed = null;
let detailLines = [];
let skew = 0;

// ---------------------------------------------------------------- time
const now = () => Date.now() / 1000 + skew;
function ago(ts) {
  const s = Math.max(0, now() - ts);
  if (s < 45) return 'just now';
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ${Math.round((s % 3600) / 60)} min ago`;
  return new Date(ts * 1000).toLocaleDateString();
}
function dur(s) {
  s = Math.max(0, Math.round(s));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60), r = s % 60;
  if (m < 60) return `${m}:${String(r).padStart(2, '0')}`;
  return `${Math.floor(m / 60)}h ${m % 60}m`;
}
const clock = ts => new Date(ts * 1000).toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'});
const clockS = ts => new Date(ts * 1000).toLocaleTimeString([], {hour: '2-digit', minute: '2-digit', second: '2-digit'});

// ---------------------------------------------------------------- data
async function load() {
  const r = await fetch('/api/unit');
  if (!r.ok) throw new Error('unit not available');
  U = await r.json();
  skew = U.now - Date.now() / 1000;
  index();
  try { A = await (await fetch('/api/ascend')).json(); } catch (e) {}
  try { L = (await (await fetch('/api/unit/log')).json()).rows; } catch (e) {}
  renderAll();
  if (openBed) await openDetail(openBed, true);
}
function index() { bedsById = Object.fromEntries(U.beds.map(b => [b.bed, b])); }

function applyEvent(m) {
  skew = m.now - Date.now() / 1000;
  if (m.bed) {
    const i = U.beds.findIndex(b => b.bed === m.bed.bed);
    if (i >= 0) U.beds[i] = m.bed; else U.beds.push(m.bed);
  }
  if (m.alert) {
    const j = U.alerts.findIndex(a => a.id === m.alert.id);
    if (j >= 0) U.alerts[j] = m.alert; else U.alerts.unshift(m.alert);
  }
  U.metrics = m.metrics;
  index();
  // the flat log grows from the same events
  if (m.event === 'patient' || m.event === 'nurse') logAdd({ts: m.line.ts, kind: m.line.who, bed: m.bed.bed, text: m.line.text, level: m.line.level, category: m.line.category, confidence: m.line.confidence, by: m.line.by});
  else if (m.event === 'ack') logAdd({ts: m.alert.ack_ts, kind: 'ack', bed: m.alert.bed, text: m.alert.text, level: m.alert.level, by: m.alert.ack_by, time_to_ack_s: Math.round((m.alert.ack_ts - m.alert.ts) * 10) / 10});
  else if (m.event === 'system') logAdd({ts: m.line.ts, kind: 'system', bed: null, text: m.line.text, detail: Object.fromEntries(Object.entries(m.line).filter(([k]) => !['ts', 'who', 'text'].includes(k)))});
  renderMetrics(); renderTiles(); renderAlerts();
  if (m.bed && openBed === m.bed.bed) {
    if (m.line) { detailLines.push(m.line); renderLines(); }
    renderDetailHead();
    if (m.event === 'patient' || m.event === 'ack') loadGuidance(openBed);
  }
}
function applyAscend(m) {
  const j = A.events.findIndex(e => e.id === m.event.id);
  if (j >= 0) A.events[j] = m.event; else A.events.push(m.event);
  A.events = A.events.slice(-80);
  A.summary = m.summary;
  const p = m.event.payload || {};
  if (j < 0) logAdd({ts: m.event.ts, kind: 'ascend', bed: m.event.bed, text: m.event.kind + (p.trigger ? ` · ${p.trigger}` : p.resource ? ` · ${p.resource}` : ''), detail: {direction: m.event.direction, status: m.event.status, ...p}});
  renderAscend(); renderMetrics();
}
function applyFeeds(m) {
  const before = JSON.stringify(U.feeds);
  U.feeds = m.feeds;
  if (JSON.stringify(m.feeds) !== before) { renderFeedLabels(); if (openBed) renderDetailFeeds(); }
}
function logAdd(row) { L.push(row); if (L.length > 3000) L = L.slice(-3000); renderLog(true); }

// ---------------------------------------------------------------- render: top bar
function renderAll() { renderMetrics(); renderTiles(); renderAlerts(); renderAscend(); renderLog(); }

function renderMetrics() {
  const m = U.metrics || {};
  const open = $('#m-open');
  open.querySelector('.v').innerHTML = `${m.open ?? '–'}${m.open_urgent ? `<small>${m.open_urgent} urgent</small>` : ''}`;
  open.classList.toggle('hot', !!m.open_urgent);
  $('#m-resp .v').textContent = m.avg_response_s == null ? '–' : dur(m.avg_response_s);
  $('#m-perbed .v').innerHTML = m.requests_per_bed == null ? '–' : `${m.requests_per_bed.toFixed(1)}<small>${m.requests_today} across ${m.beds} beds</small>`;
  const s = A.summary || {};
  const eng = (s.resources_opened || 0) + (s.msl_asks || 0) + (s.wallet_sent || 0);
  $('#m-ascend .v').innerHTML = `${eng}<small>${s.triggers || 0} triggers</small>`;
  const live = U.beds.find(b => b.real);
  const sim = U.beds.length - (live ? 1 : 0);
  $('#boardnote').textContent = live
    ? `Bed ${live.bed} is the live camera. The other ${sim === 6 ? 'six' : sim} beds are simulated so the unit looks like a unit.`
    : 'No live bed connected.';
}

// ---------------------------------------------------------------- render: feeds (synced to the real cameras)
function bedsideLabel() {
  const f = U.feeds && U.feeds.bedside;
  if (!f) return {text: 'Camera', live: false, sub: ''};
  const name = f.glasses ? 'Glasses camera' : `${f.label} (stand-in)`;
  const fps = f.fps ? ` · ${f.fps} fps` : '';
  return {text: f.opened ? name + fps : `${f.label} · no signal`, live: !!f.opened, sub: f.detail || ''};
}
function ambientLabel() {
  const a = U.feeds && U.feeds.ambient;
  if (!a) return {text: 'Ambient camera off', live: false, sub: ''};
  if (!a.opened) return {text: 'Ambient · unavailable', live: false, sub: a.error || 'starting the laptop camera'};
  return {text: `Ambient · laptop camera · ${a.fps} fps`, live: true, sub: a.same_as_bedside ? 'Same camera as the bedside feed until the glasses connect' : `index ${a.index}`};
}
function renderFeedLabels() {
  const b = bedsideLabel(), a = ambientLabel();
  const tile = $('.tile.real .video');
  if (tile) {
    tile.querySelector('.camlabel').textContent = b.text;
    let ns = tile.querySelector('.nosignal');
    if (!b.live && !ns) { ns = document.createElement('div'); ns.className = 'nosignal'; ns.textContent = 'No signal'; tile.appendChild(ns); }
    if (b.live && ns) ns.remove();
  }
  const amb = $('.tile.ambient');
  if (amb) {
    const img = amb.querySelector('img');
    if (a.live && !img) amb.querySelector('.video').innerHTML = `<img src="/ambient" alt="Ambient overview">`;
    if (!a.live && img) amb.querySelector('.video').innerHTML = `<div class="nosignal">${esc(a.sub || 'No signal')}</div>`;
    amb.querySelector('.cap .t').textContent = a.live ? 'Ambient overview' : 'Ambient overview · unavailable';
    amb.querySelector('.cap .s').textContent = a.live ? `${a.text.replace('Ambient · ', '')}${a.sub ? ' · ' + a.sub : ''}` : (a.sub || '');
  }
}

// ---------------------------------------------------------------- render: tiles
function tileHTML(b) {
  const text = b.alert ? b.alert.text : b.patient_text, ts = b.alert ? b.alert.ts : b.patient_ts;
  const said = text ? `<div class="said">${esc(text)}</div>` : `<div class="said quiet">Nothing said yet today</div>`;
  const tag = b.real ? `<span class="label tag livetag">Live</span>` : `<span class="label tag">Simulated</span>`;
  const video = b.real ? `<div class="video"><img src="/stream" alt="Bed ${esc(b.bed)} live camera"><span class="camlabel">${esc(bedsideLabel().text)}</span></div>` : '';
  return `<button class="tile ${b.status}${b.real ? ' real' : ''}" data-bed="${esc(b.bed)}" aria-label="Bed ${esc(b.bed)}, ${b.initials}, ${LEVEL_WORD[b.status]}">
    ${video}
    <div class="row"><span class="bed">Bed ${esc(b.bed)}</span><span class="ini">${esc(b.initials)}</span>${tag}</div>
    ${said}
    <div class="foot"><span class="status ${b.status}"><i></i>${LEVEL_WORD[b.status]}</span><span class="ago" data-ts="${ts || ''}">${ts ? ago(ts) : ''}</span></div>
  </button>`;
}
function ambientHTML() {
  const a = ambientLabel();
  const body = a.live ? `<img src="/ambient" alt="Ambient overview">` : `<div class="nosignal">${esc(a.sub || 'No signal')}</div>`;
  return `<div class="tile ambient"><div class="video">${body}</div>
    <div class="cap"><span class="t">${a.live ? 'Ambient overview' : 'Ambient overview · unavailable'}</span><span class="s">${esc(a.live ? a.text.replace('Ambient · ', '') + (a.sub ? ' · ' + a.sub : '') : a.sub)}</span></div></div>`;
}
function renderTiles() {
  const el = $('#tiles');
  // Rewriting a live <img src="/stream"> reopens the MJPEG connection; keep tiles by bed and patch their text instead.
  const want = U.beds.map(b => b.bed);
  const have = [...el.querySelectorAll('.tile[data-bed]')].map(t => t.dataset.bed);
  if (want.join() !== have.join() || !el.querySelector('.tile.ambient')) {
    el.innerHTML = U.beds.map(tileHTML).join('') + ambientHTML();
    renderFeedLabels();
    return;
  }
  for (const b of U.beds) {
    const t = el.querySelector(`.tile[data-bed="${CSS.escape(b.bed)}"]`);
    const fresh = document.createElement('div'); fresh.innerHTML = tileHTML(b);
    const nt = fresh.firstElementChild;
    t.className = nt.className; t.setAttribute('aria-label', nt.getAttribute('aria-label'));
    t.querySelector('.row').replaceWith(nt.querySelector('.row'));
    t.querySelector('.said').replaceWith(nt.querySelector('.said'));
    t.querySelector('.foot').replaceWith(nt.querySelector('.foot'));
  }
}

// ---------------------------------------------------------------- render: requests
function alertHTML(a) {
  const b = bedsById[a.bed] || {initials: ''};
  const open = a.ack_ts == null;
  const wait = open
    ? `<div class="wait" data-since="${a.ts}">${dur(now() - a.ts)}<small>waiting</small></div>`
    : `<div class="wait">${dur(a.ack_ts - a.ts)}<small>to acknowledge</small></div>`;
  return `<li class="alert ${a.level}${open ? '' : ' done'}" data-id="${a.id}">
    <div class="head"><b>Bed ${esc(a.bed)}</b><span>${esc(b.initials)}</span><span class="lvl">${LEVEL_WORD[a.level]}</span><span>${clock(a.ts)}</span></div>
    <div class="text">${esc(a.text)}</div>${wait}
    ${open ? `<button class="ack" data-id="${a.id}">Acknowledge</button>` : `<div class="by">Acknowledged by ${esc(a.ack_by || 'staff')} at ${clock(a.ack_ts)}</div>`}
  </li>`;
}
function renderAlerts() {
  const Lv = ['calm', 'request', 'urgent'];
  const open = U.alerts.filter(a => a.ack_ts == null).sort((x, y) => Lv.indexOf(y.level) - Lv.indexOf(x.level) || x.ts - y.ts);
  const done = U.alerts.filter(a => a.ack_ts != null).sort((x, y) => y.ack_ts - x.ack_ts).slice(0, 12);
  $('#feedcount').textContent = open.length ? `${open.length} waiting` : '';
  $('#feedempty').hidden = open.length > 0;
  $('#alerts').innerHTML = open.map(alertHTML).join('') + (done.length ? `<li class="divider">Acknowledged</li>` + done.map(alertHTML).join('') : '');
}

// ---------------------------------------------------------------- render: Ascend journey
const EVENT_WORD = {'spark.trigger': 'Spark trigger', 'engagement.opened': 'Resource opened', 'engagement.msl': 'MSL asked', 'engagement.wallet': 'Wallet card sent'};
function eventDetail(e) {
  const p = e.payload || {};
  if (e.kind === 'spark.trigger') return p.trigger === 'bedside_request'
    ? `${p.trigger} · ${p.category || '–'} · ${p.urgency}`
    : `${p.trigger} · ${dur(p.time_to_acknowledge_s || 0)} to acknowledge`;
  return p.resource || '';
}
function renderAscend() {
  const s = A.summary || {};
  $('#ascendsum').innerHTML = [
    `<span><b>${s.triggers ?? 0}</b>Spark triggers today</span>`,
    `<span><b>${s.resources_opened ?? 0}</b>resources opened</span>`,
    `<span><b>${s.msl_asks ?? 0}</b>MSL asks</span>`,
    `<span><b>${s.wallet_sent ?? 0}</b>Wallet cards sent</span>`,
    `<span>${esc(s.delivery || '')}</span>`].join('');
  const rows = [...A.events].sort((x, y) => y.ts - x.ts).slice(0, 10);
  $('#journey').innerHTML = rows.length ? rows.map(e => `<tr class="${e.direction}">
      <td class="faint">${clockS(e.ts)}</td><td class="dim">${e.direction === 'out' ? 'Board → Ascend' : 'Nurse → Ascend'}</td>
      <td class="ev">${EVENT_WORD[e.kind] || esc(e.kind)}</td><td>${e.bed ? 'Bed ' + esc(e.bed) : ''}</td>
      <td class="dim">${esc(eventDetail(e))}</td><td class="faint">${esc(e.status)}</td></tr>`).join('')
    : `<tr><td colspan="6" class="none">No requests yet today. The first request becomes the first Spark trigger.</td></tr>`;
}

// ---------------------------------------------------------------- render: log
const KIND_WORD = {patient: 'Patient', nurse: 'Nurse', ack: 'Acknowledged', system: 'System', ascend: 'Ascend'};
function logDetail(r) {
  if (r.kind === 'patient') return [r.level && r.level !== 'calm' ? r.level : null, r.category, r.confidence != null ? `${Math.round(r.confidence * 100)}%` : null].filter(Boolean).join(' · ');
  if (r.kind === 'nurse') return r.by || '';
  if (r.kind === 'ack') return `${dur(r.time_to_ack_s || 0)} to acknowledge · ${r.by || ''}`;
  if (r.kind === 'ascend') return `${r.detail && r.detail.direction === 'in' ? 'nurse → ascend' : 'board → ascend'} · ${(r.detail && r.detail.status) || ''}`;
  if (r.kind === 'system') return Object.entries(r.detail || {}).filter(([k]) => k !== 'kind').map(([k, v]) => `${k}=${v}`).join(' ');
  return '';
}
function renderLog(keepScroll) {
  const wrap = $('.logwrap');
  const atBottom = wrap.scrollHeight - wrap.scrollTop - wrap.clientHeight < 40;
  const live = U.real_bed;
  const rows = L.filter(r => logFilter === 'all' || (logFilter === 'live' ? r.bed === live : r.kind === logFilter));
  $('#logcount').textContent = `${rows.length} of ${L.length} events today`;
  $('#logrows').innerHTML = rows.length ? rows.map(r => `<tr class="${r.kind}${r.level && r.level !== 'calm' ? ' ' + r.level : ''}">
      <td class="faint">${clockS(r.ts)}</td><td class="k">${KIND_WORD[r.kind] || esc(r.kind)}</td><td class="b">${r.bed ? 'Bed ' + esc(r.bed) : ''}</td>
      <td class="text">${esc(r.text)}</td><td class="dim">${esc(logDetail(r))}</td></tr>`).join('')
    : `<tr><td colspan="5" class="none">Nothing logged yet today.</td></tr>`;
  if (!keepScroll || atBottom) wrap.scrollTop = wrap.scrollHeight;
}

function tick() {
  $('#clock').textContent = clock(now());
  document.querySelectorAll('.wait[data-since]').forEach(w => { w.firstChild.nodeValue = dur(now() - +w.dataset.since); });
  document.querySelectorAll('.ago[data-ts]').forEach(e => { if (e.dataset.ts) e.textContent = ago(+e.dataset.ts); });
}

// ---------------------------------------------------------------- bed detail
async function openDetail(bed, silent) {
  const r = await fetch(`/api/unit/bed?bed=${encodeURIComponent(bed)}`);
  if (!r.ok) return;
  const v = await r.json();
  openBed = bed;
  detailLines = v.transcript || [];
  bedsById[bed] = Object.assign(bedsById[bed] || {}, v);
  if (!silent) location.hash = `bed=${bed}`;
  $('#board').hidden = true; $('#detail').hidden = false;
  const real = !!v.real;
  $('#d-img').src = real ? '/stream' : '';
  $('#d-img').hidden = !real;
  $('#d-nocam').hidden = real;
  renderDetailFeeds();
  renderDetailHead(); renderLines();
  loadGuidance(bed);
}
function renderDetailFeeds() {
  const b = bedsById[openBed]; if (!b) return;
  const bl = bedsideLabel(), al = ambientLabel();
  $('#d-camlabel').textContent = b.real ? bl.text : '';
  $('#d-ambient').hidden = !al.live;
  if (al.live && !$('#d-amb').getAttribute('src')) $('#d-amb').src = '/ambient';
  $('#d-amblabel').textContent = al.text + (al.sub ? ' · ' + al.sub : '');
}
async function loadGuidance(bed) {
  const g = $('#guidance');
  let d;
  try { d = await (await fetch(`/api/ascend/bed?bed=${encodeURIComponent(bed)}`)).json(); } catch (e) { g.hidden = true; return; }
  const r = d.resource;
  if (!r) { g.hidden = true; return; }
  g.hidden = false;
  g.dataset.resource = r.title; g.dataset.kind = r.kind;
  $('#g-kind').textContent = r.kind;
  $('#g-title').textContent = r.title;
  $('#g-body').textContent = r.body;
  $('#g-reason').textContent = `Why this: ${r.reason}`;
  const cta = $('#g-cta'); cta.textContent = r.cta; cta.className = 'primary'; cta.disabled = false;
  const msl = $('#g-msl'); msl.textContent = 'Ask a medical science liaison'; msl.className = ''; msl.disabled = false;
  const n = d.next_best_action;
  $('#nba').hidden = !n;
  if (n) $('#nba-text').textContent = n.text;
}
async function engage(action, btn, doneText) {
  const g = $('#guidance');
  btn.disabled = true;
  const r = await fetch(`/api/ascend/engage?bed=${encodeURIComponent(openBed)}&action=${action}&resource=${encodeURIComponent(g.dataset.resource)}`, {method: 'POST'});
  if (r.ok) { btn.textContent = doneText; btn.classList.add('done'); } else btn.disabled = false;
}
function closeDetail() {
  openBed = null; detailLines = [];
  $('#d-img').removeAttribute('src'); $('#d-amb').removeAttribute('src');
  $('#detail').hidden = true; $('#board').hidden = false;
  if (location.hash) history.replaceState(null, '', location.pathname + location.search);
}
function renderDetailHead() {
  const b = bedsById[openBed]; if (!b) return;
  $('#d-title').textContent = `Bed ${b.bed} · ${b.initials}`;
  const st = $('#d-status'); st.className = `status ${b.status}`; st.querySelector('span').textContent = LEVEL_WORD[b.status];
  $('#d-note').textContent = (b.note || '') + (b.real ? '' : ' · simulated bed');
}
function renderLines() {
  const ol = $('#d-lines');
  if (!detailLines.length) { ol.innerHTML = `<li class="none">Nothing recorded at this bed yet today.</li>`; return; }
  ol.innerHTML = detailLines.map(l => `<li class="${l.who}${l.level && l.level !== 'calm' ? ' ' + l.level : ''}">
      <time>${clockS(l.ts)}</time><span class="who">${l.who === 'patient' ? 'Patient' : esc(l.by || 'Nurse')}</span>
      <span class="txt" data-level="${LEVEL_WORD[l.level] || ''}">${esc(l.text)}</span></li>`).join('');
  ol.scrollTop = ol.scrollHeight;
}

// ---------------------------------------------------------------- actions
document.addEventListener('click', async e => {
  const ack = e.target.closest('.ack');
  if (ack) {
    ack.disabled = true; ack.textContent = 'Acknowledging';
    const r = await fetch(`/api/unit/ack?alert=${ack.dataset.id}`, {method: 'POST'});
    if (!r.ok) { ack.disabled = false; ack.textContent = 'Acknowledge'; }
    return;
  }
  const tile = e.target.closest('.tile[data-bed]');
  if (tile) { openDetail(tile.dataset.bed); return; }
  if (e.target.closest('#back')) { e.preventDefault(); closeDetail(); return; }
  if (e.target.closest('#g-cta')) { const g = $('#guidance'); engage(g.dataset.kind === 'Wallet card' ? 'wallet' : 'opened', e.target, g.dataset.kind === 'Wallet card' ? 'Wallet card sent' : 'Opened'); return; }
  if (e.target.closest('#g-msl')) { engage('msl', e.target, 'Asked · MSL will follow up'); return; }
  const f = e.target.closest('#logfilters button');
  if (f) { logFilter = f.dataset.k; document.querySelectorAll('#logfilters button').forEach(b => b.classList.toggle('on', b === f)); renderLog(); }
});
$('#noteform').addEventListener('submit', async e => {
  e.preventDefault();
  const inp = $('#noteinput'); const text = inp.value.trim();
  if (!text || !openBed) return;
  inp.value = '';
  await fetch(`/api/unit/nurse?bed=${encodeURIComponent(openBed)}&text=${encodeURIComponent(text)}`, {method: 'POST'});
});

// ---------------------------------------------------------------- live
function connect() {
  const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
  ws.onopen = () => { $('#conn').className = 'live on'; $('#conn').lastElementChild.textContent = 'live'; load().catch(console.error); };
  ws.onmessage = ev => {
    const m = JSON.parse(ev.data);
    if (m.type === 'unit') applyEvent(m); else if (m.type === 'ascend') applyAscend(m); else if (m.type === 'feeds') applyFeeds(m);
  };
  ws.onclose = () => { $('#conn').className = 'live off'; $('#conn').lastElementChild.textContent = 'reconnecting'; setTimeout(connect, 1500); };
}
const h = location.hash.match(/bed=([^&]+)/);
if (h) openBed = decodeURIComponent(h[1]);
connect();
setInterval(tick, 1000); tick();
