// Charge nurse unit board. State comes from GET /api/unit once, then every change from the server's websocket
// ("unit" events); times on screen tick locally every second.
'use strict';
const $ = s => document.querySelector(s);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const LEVEL_WORD = {calm: 'Calm', request: 'Request', urgent: 'Urgent'};

let U = {beds: [], alerts: [], metrics: {}, real_bed: null, ambient: null};
let A = {summary: {}, events: []};   // the Impiricus Ascend seam (simulated)
let bedsById = {};
let openBed = null;        // bed id shown in detail, or null
let detailLines = [];      // transcript of the open bed
let skew = 0;              // server clock - local clock

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
  renderAll();
  if (openBed) await openDetail(openBed, true);
}
function index() { bedsById = Object.fromEntries(U.beds.map(b => [b.bed, b])); }

function applyEvent(m) {
  skew = m.now - Date.now() / 1000;
  const i = U.beds.findIndex(b => b.bed === m.bed.bed);
  if (i >= 0) U.beds[i] = m.bed; else U.beds.push(m.bed);
  if (m.alert) {
    const j = U.alerts.findIndex(a => a.id === m.alert.id);
    if (j >= 0) U.alerts[j] = m.alert; else U.alerts.unshift(m.alert);
  }
  U.metrics = m.metrics;
  index();
  renderMetrics(); renderTiles(); renderAlerts();
  if (openBed === m.bed.bed) {
    if (m.line) { detailLines.push(m.line); renderLines(); }
    renderDetailHead();
    if (m.event === 'patient' || m.event === 'ack') loadGuidance(openBed);   // a new request may change the resource
  }
}

// ---------------------------------------------------------------- render
function renderAll() { renderMetrics(); renderTiles(); renderAlerts(); renderAscend(); }

function applyAscend(m) {
  const j = A.events.findIndex(e => e.id === m.event.id);
  if (j >= 0) A.events[j] = m.event; else A.events.push(m.event);
  A.events = A.events.slice(-60);
  A.summary = m.summary;
  renderAscend(); renderMetrics();
}
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
    `<span><b>${s.triggers ?? 0}</b>Spark triggers sent today</span>`,
    `<span><b>${s.resources_opened ?? 0}</b>resources opened</span>`,
    `<span><b>${s.msl_asks ?? 0}</b>MSL asks</span>`,
    `<span><b>${s.wallet_sent ?? 0}</b>Wallet cards sent</span>`,
    `<span>${esc(s.delivery || '')}</span>`].join('');
  const rows = [...A.events].sort((x, y) => y.ts - x.ts).slice(0, 10);
  $('#journey').innerHTML = rows.length ? rows.map(e => `<tr class="${e.direction}">
      <td>${clockS(e.ts)}</td><td class="dir">${e.direction === 'out' ? 'Board → Ascend' : 'Nurse → Ascend'}</td>
      <td class="ev">${EVENT_WORD[e.kind] || esc(e.kind)}</td><td>${e.bed ? 'Bed ' + esc(e.bed) : ''}</td>
      <td class="detail">${esc(eventDetail(e))}</td><td class="st">${esc(e.status)}</td></tr>`).join('')
    : `<tr><td colspan="6" class="none">No requests yet today. The first request becomes the first Spark trigger.</td></tr>`;
}

function renderMetrics() {
  const m = U.metrics || {};
  const open = $('#m-open');
  open.querySelector('dd').innerHTML = `${m.open ?? '–'}${m.open_urgent ? `<small>${m.open_urgent} urgent</small>` : ''}`;
  open.classList.toggle('hot', !!m.open_urgent);
  $('#m-resp dd').textContent = m.avg_response_s == null ? '–' : dur(m.avg_response_s);
  $('#m-perbed dd').innerHTML = m.requests_per_bed == null ? '–' : `${m.requests_per_bed.toFixed(1)}<small>${m.requests_today} across ${m.beds} beds</small>`;
  const s = A.summary || {};
  const eng = (s.resources_opened || 0) + (s.msl_asks || 0) + (s.wallet_sent || 0);
  $('#m-ascend dd').innerHTML = `${eng}<small>${s.triggers || 0} triggers</small>`;
  const live = U.beds.find(b => b.real);
  const sim = U.beds.length - (live ? 1 : 0);
  $('#boardnote').textContent = live
    ? `Bed ${live.bed} is the live camera. The other ${sim === 6 ? 'six' : sim} beds are simulated so the unit looks like a unit.`
    : 'No live bed connected.';
}

function tileHTML(b) {
  // the words that opened the bed's alert while it waits; otherwise the last thing the patient mouthed
  const text = b.alert ? b.alert.text : b.patient_text, ts = b.alert ? b.alert.ts : b.patient_ts;
  const said = text ? `<div class="said">${esc(text)}</div>` : `<div class="said quiet">Nothing said yet today</div>`;
  const tag = b.real ? `<span class="tag live">Live</span>` : `<span class="tag">Simulated</span>`;
  const video = b.real ? `<div class="video"><img src="/stream" alt="Bed ${esc(b.bed)} live camera"><span class="camlabel">Glasses camera</span></div>` : '';
  return `<button class="tile ${b.status}${b.real ? ' real' : ''}" data-bed="${esc(b.bed)}" aria-label="Bed ${esc(b.bed)}, ${b.initials}, ${LEVEL_WORD[b.status]}">
    ${video}
    <div class="row"><span class="bed">Bed ${esc(b.bed)}</span><span class="ini">${esc(b.initials)}</span>${tag}</div>
    ${said}
    <div class="foot"><span class="status ${b.status}"><i></i>${LEVEL_WORD[b.status]}</span><span class="ago" data-ts="${ts || ''}">${ts ? ago(ts) : ''}</span></div>
  </button>`;
}
function ambientHTML() {
  const a = U.ambient;
  const body = a && a.opened
    ? `<img src="/ambient" alt="Ambient overview">`
    : `<div class="unavail">${a ? esc(a.error || 'Starting the laptop camera') : 'Ambient camera is off'}</div>`;
  return `<div class="tile ambient"><div class="video">${body}</div>
    <div class="cap"><b>Ambient overview</b><span>Laptop camera</span><span class="fps">${a && a.opened ? `${a.fps} fps` : ''}</span></div></div>`;
}
function renderTiles() {
  const el = $('#tiles');
  // Rewriting a live <img src="/stream"> reopens the MJPEG connection; keep tiles by bed and patch their text instead.
  const want = U.beds.map(b => b.bed);
  const have = [...el.querySelectorAll('.tile[data-bed]')].map(t => t.dataset.bed);
  if (want.join() !== have.join() || !el.querySelector('.tile.ambient')) {
    el.innerHTML = U.beds.map(tileHTML).join('') + ambientHTML();
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
  const L = ['calm', 'request', 'urgent'];
  const open = U.alerts.filter(a => a.ack_ts == null).sort((x, y) => L.indexOf(y.level) - L.indexOf(x.level) || x.ts - y.ts);
  const done = U.alerts.filter(a => a.ack_ts != null).sort((x, y) => y.ack_ts - x.ack_ts).slice(0, 12);
  $('#feedcount').textContent = open.length ? `${open.length} waiting` : '';
  $('#feedempty').hidden = open.length > 0;
  $('#alerts').innerHTML = open.map(alertHTML).join('') + (done.length ? `<li class="divider">Acknowledged</li>` + done.map(alertHTML).join('') : '');
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
  $('#d-camlabel').textContent = real ? 'Glasses camera · live' : '';
  const a = U.ambient;
  $('#d-ambient').hidden = !(a && a.opened);
  if (a && a.opened) $('#d-amb').src = '/ambient';
  renderDetailHead(); renderLines();
  loadGuidance(bed);
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
  $('#d-img').src = ''; $('#d-amb').src = '';   // stop the MJPEG connections
  $('#detail').hidden = true; $('#board').hidden = false;
  if (location.hash) history.replaceState(null, '', location.pathname);
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
    return;   // the websocket event redraws the feed and the bed
  }
  const tile = e.target.closest('.tile[data-bed]');
  if (tile) { openDetail(tile.dataset.bed); return; }
  if (e.target.closest('#back')) { e.preventDefault(); closeDetail(); return; }
  if (e.target.closest('#g-cta')) { const g = $('#guidance'); engage(g.dataset.kind === 'Wallet card' ? 'wallet' : 'opened', e.target, g.dataset.kind === 'Wallet card' ? 'Wallet card sent' : 'Opened'); return; }
  if (e.target.closest('#g-msl')) { engage('msl', e.target, 'Asked · MSL will follow up'); }
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
  ws.onopen = () => { $('#conn').className = 'live on'; $('#conn').lastChild.nodeValue = 'live'; load().catch(console.error); };
  ws.onmessage = ev => { const m = JSON.parse(ev.data); if (m.type === 'unit') applyEvent(m); else if (m.type === 'ascend') applyAscend(m); };
  ws.onclose = () => { $('#conn').className = 'live off'; $('#conn').lastChild.nodeValue = 'reconnecting'; setTimeout(connect, 1500); };
}
const h = location.hash.match(/bed=([^&]+)/);
if (h) openBed = decodeURIComponent(h[1]);
connect();
setInterval(tick, 1000); tick();
setInterval(() => { if (U.ambient && !U.ambient.opened) load().catch(() => {}); }, 6000);   // the ambient camera may open late
