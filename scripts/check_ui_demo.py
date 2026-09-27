"""Check the bedside UI's ?demo=1 replay renders every scripted scene (no backend, no camera needed).

    python scripts/check_ui_demo.py [--chrome /path/to/chrome]

Serves silent_running/ on a free port, opens /static/index.html?demo=1 in headless Chrome driven over the DevTools
protocol, and waits (real time) for each expected UI state in order: hero state / text / source badge (lips, or fused once
the LLM corrected the reading), nonverbal chips, the critical alert, then a manual space-bar take. Then, with the scenes
paused: a 40-message conversation at 1920x1080 and 1280x720 (the page must not grow and the newest message must be in view),
an injection fuzz over every string field the UI renders, the hand-signal chips and hand-tracker state, and speak(): the
face emotion for non-lip text, and a superseded playback not falling back to the browser voice (the mouthed pace only with
"match my pace").
Prints expected vs actual per step and any page JS exceptions.
Exit code 0 = every step reached and no exceptions.
"""
import argparse, glob, json, os, shutil, socket, subprocess, sys, tempfile, threading, time, urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from websockets.sync.client import connect

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Expected states in the order silent_running/static/demo.js plays them (SCENES), then the manual take.
STEPS = [
    ("scene 1: nurse asks, Grok corrects the lips' reading", {"state": "result", "src": "fused", "text": "I AM IN PAIN", "asked": "How is your pain right now?", "chips": ["pain"]}),
    ("scene 2: pain score on fingers", {"asked": "Show me your pain on your fingers, zero to ten.", "chips": ["fingers"]}),
    ("scene 3: a clear reading, Grok keeps it", {"state": "result", "src": "lips", "text": "I AM COLD"}),
    ("scene 4: blink code yes", {"asked": "Do you want me to call your family?", "chips": ["blink"]}),
    ("scene 5: critical phrase alert", {"alert": "I CAN'T BREATHE"}),
]
MANUAL = ("manual take: hold space, release", {"state": "result", "src": "lips", "text": "I NEED WATER"})

SNAPSHOT_JS = """(()=>{const q=s=>document.querySelector(s),h=q('#hero');return {
  state:h.dataset.state, src:h.dataset.src, text:q('#bigtext').textContent, ring:q('#ringpct').textContent, asked:q('#asked').textContent,
  chips:[...document.querySelectorAll('.sig.on')].map(e=>e.id.slice(4)),
  alert:q('#alert').classList.contains('on')?q('#alerttext').textContent:null}})()"""


def find_chrome(arg):
    cands = [arg, os.environ.get("CHROME")]  # Chrome for Testing (Playwright's cache) first: built for automation
    cands += sorted(glob.glob(os.path.expanduser("~/Library/Caches/ms-playwright/chromium-*/chrome-mac*/*.app/Contents/MacOS/*")))
    cands += [shutil.which("google-chrome"), shutil.which("chromium"), "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"]
    for c in cands:
        if c and os.path.isfile(c):
            return c
    sys.exit("no Chrome/Chromium found; pass --chrome or set CHROME")


class QuietHandler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=os.path.join(ROOT, "silent_running"), **kw)

    def log_message(self, *a):
        pass


def serve():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), QuietHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{httpd.server_address[1]}/static/index.html?demo=1"


class Page:
    """Minimal DevTools-protocol client: one page, synchronous calls, JS exceptions collected."""
    def __init__(self, chrome):
        self.profile = tempfile.mkdtemp(prefix="sr_ui_check_")
        s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
        self.proc = subprocess.Popen([chrome, "--headless=new", "--disable-gpu", "--no-first-run", "--mute-audio", "--autoplay-policy=no-user-gesture-required", "--window-size=1920,1080",
                                      f"--remote-debugging-port={port}", f"--user-data-dir={self.profile}", "about:blank"],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        pages, t_end = [], time.time() + 60  # Chrome can take >10 s to start on a loaded machine; the port opens before the tab exists
        while not pages and time.time() < t_end:
            try:
                pages = [t for t in json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=2)) if t["type"] == "page"]
            except OSError:
                pass
            time.sleep(0.2)
        if not pages:
            self.proc.kill(); sys.exit(f"Chrome did not expose a page over DevTools within 60 s: {chrome}")
        target = pages[0]
        self.ws = connect(target["webSocketDebuggerUrl"], max_size=None, legacy=True)
        self.n, self.exceptions = 0, []
        self.call("Runtime.enable")

    def call(self, method, **params):
        self.n += 1
        self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            m = json.loads(self.ws.recv())
            if m.get("method") == "Runtime.exceptionThrown":
                d = m["params"]["exceptionDetails"]
                self.exceptions.append(d.get("exception", {}).get("description") or d.get("text"))
            if m.get("id") == self.n:
                return m.get("result", {})

    def eval(self, js):
        return self.call("Runtime.evaluate", expression=js, returnByValue=True, awaitPromise=True)["result"].get("value")

    def key(self, kind, code=" "):
        self.call("Input.dispatchKeyEvent", type=kind, key=code, code="Space", windowsVirtualKeyCode=32, text=code if kind == "keyDown" else "")

    def close(self):
        self.ws.close(); self.proc.terminate(); self.proc.wait(); shutil.rmtree(self.profile, ignore_errors=True)


# Long conversation: the page must stay at the viewport height, the log scrolls inside its panel, newest message in view.
LONG_LOG_JS = """(()=>{for(let i=1;i<=40;i++)handle({type:'log',entry:{ts:Date.now()/1000,who:i%2?'nurse':'patient',text:'message '+i}});
  return new Promise(r=>setTimeout(()=>{const m=document.querySelector('main'),S=document.querySelector('#logscroll'),s=S.getBoundingClientRect(),
  n=document.querySelector('#log').lastElementChild.getBoundingClientRect();
  r({page_fits:m.scrollHeight<=m.clientHeight,log_scrolls:S.scrollHeight>S.clientHeight,newest_in_view:n.top>=s.top-1&&n.bottom<=s.bottom+1})},600))})()"""
LONG_LOG_EXP = {"page_fits": True, "log_scrolls": True, "newest_in_view": True}

# Every string field of every event the UI renders carries an HTML payload; none may become markup.
FUZZ_JS = """(()=>{const P='<img src=x onerror="window.__xss=(window.__xss||0)+1">',u=900,lat={crop:.1,encode:.1,beam:.1,total:.3};
  const tm={duration:1,rate:1,pauses:[],words:[{word:P,start:0,end:.5}]},ex={emotion:P,intensity:.5,scores:{[P]:.5}};
  const nv={head:{value:P,confidence:.9},fingers:{value:P,confidence:.9},blink_code:{value:P,confidence:.9},pain:{value:.5,confidence:.9},emotion:{label:P,intensity:.5},
    thumb:{value:P,confidence:.9},point:{value:P,confidence:.9}};meta({hands:P});
  [{type:'raw',utt_id:u,text:P,n_frames:10,duration:1,latency:lat},
   {type:'result',utt_id:u,selected:P,confidence:.5,nbest:[{text:P,score:1,prob:.5}],n_frames:10,duration:1,expression:ex,timing:tm,latency:lat,nonverbal:nv},
   {type:'llm',utt_id:u,changed:true,accepted:true,corrected:P,proposal:P,model:P,gap:1,latency:.3,alternatives:[{text:P,gap:-1,fits:true}]},{type:'llm_reason',utt_id:u,reason:P},
   {type:'llm',utt_id:u,error:P},{type:'llm_reason',utt_id:u,error:P},
   {type:'delivery',utt_id:u,emotion:P,intensity:.5,model:P,tag:P,stability:P,rate:1,cached:false,t_synth:P,total:P,retime:{applied:true,global:P,audio_words:[{word:P,start:0,end:.5}],ratios:[1]}},
   ...['blink_code','fingers','thumb','point'].map(kind=>({type:'signal',kind,value:P,confidence:.9,ts:0})),
   {type:'log',entry:{ts:0,who:P,text:P,confidence:.5,source:P,emotion:P}},{type:'context',context:{notes:P,category:P,last_prompt:P,history:[P]}},
   {type:'alert',utt_id:u,text:P},{type:'error',message:P},{type:'saved',file:P,phrase:P},{type:'prewarmed',speaker:P,n:P}].forEach(handle);
  document.querySelector('#alertok').click();
  return new Promise(r=>setTimeout(()=>r({xss_fired:window.__xss||0,injected_imgs:document.querySelectorAll('img:not(#cam)').length}),800))})()"""
FUZZ_EXP = {"xss_fired": 0, "injected_imgs": 0}

# speak() with an ElevenLabs voice selected (the static server 404s /api/say, so a source failure is exercised too).
# Hand signals: live thumb / point chips (directions from the patient's side), and a disabled hand tracker shown on the camera.
HANDS_JS = """(()=>{const t=s=>document.querySelector(s).textContent,r={};
  handle({type:'signal',kind:'thumb',value:'down',confidence:.8,ts:0});r.thumb=t('#sig-hand .v');
  handle({type:'signal',kind:'point',value:'left',confidence:.8,ts:0});r.point=t('#sig-hand .v');
  meta({hands:'off: FileNotFoundError: models/hand_landmarker.task'});r.badge=t('#hands');r.chips_off=document.querySelectorAll('.sig.off').length;
  meta({hands:'on'});r.chips_off_after_on=document.querySelectorAll('.sig.off').length;return r})()"""
HANDS_EXP = {"thumb": "Thumb down 80%", "point": "Pointing patient's left 80%", "badge": "hand signals off: FileNotFoundError: models/hand_landmarker.task",
             "chips_off": 2, "chips_off_after_on": 0}

SPEAK_JS = """(()=>{const s=document.querySelector('#voice'),o=document.createElement('option');o.value='clone:T';s.appendChild(o);s.value='clone:T';
  const spoken=[],sp=speechSynthesis.speak.bind(speechSynthesis);speechSynthesis.speak=x=>{spoken.push(x.text);sp(x)};
  handle({type:'result',utt_id:7,selected:'I am cold',confidence:.8,n_frames:40,duration:1.6,latency:{crop:.05,encode:.1,beam:.5,total:.65},
    nbest:['I AM COLD','I AM HOT'].map((text,i)=>({text,score:-1-i,prob:.8-i*.6})),
    expression:{emotion:'sad',intensity:.8},timing:{duration:1.6,rate:1.3,pauses:[],words:[]}});
  const q=()=>{const u=new URL(player.src).searchParams;return ['emotion','intensity','rate','utt_id'].map(k=>k+'='+u.get(k)).join('&')};
  const r={};speak('I am cold',{utt_id:7});r.lip_phrase=q();document.querySelector('#pace').checked=true;speak('I am cold',{utt_id:7});r.lip_phrase_pace=q();
  document.querySelector('#pace').checked=false;speak('I am freezing',{utt_id:7});r.llm_text=q();
  document.querySelectorAll('#dym button')[1].click();r.candidate_tap=q();
  const after=(ms,f)=>new Promise(res=>setTimeout(()=>res(f()),ms));
  return after(800,()=>{spoken.length=0;speak('FIRST');speak('SECOND')}).then(()=>after(1500,()=>{r.browser_fallback=[...spoken];spoken.length=0;speak('THIRD');speakBrowser('PROMPT',{})}))
    .then(()=>after(1500,()=>{r.clip_then_prompt=[...spoken];return r}))})()"""
SPEAK_EXP = {"lip_phrase": "emotion=sad&intensity=0.8&rate=1&utt_id=7", "lip_phrase_pace": "emotion=sad&intensity=0.8&rate=1.3&utt_id=7", "llm_text": "emotion=sad&intensity=0.8&rate=1&utt_id=0",
             "candidate_tap": "emotion=sad&intensity=0.8&rate=1&utt_id=0", "browser_fallback": ["SECOND"], "clip_then_prompt": ["PROMPT"]}


def wait_for(page, exp, timeout):
    """Poll until the UI shows `exp` (chips: all listed chips lit). Returns (ok, last snapshot restricted to exp's keys)."""
    t_end, snap = time.time() + timeout, {}
    while time.time() < t_end:
        snap = page.eval(SNAPSHOT_JS) or {}  # None until the page has loaded
        if snap and all(set(v) <= set(snap["chips"]) if k == "chips" else snap[k] == v for k, v in exp.items()):
            return True, {k: exp[k] if k == "chips" else snap[k] for k in exp}
        time.sleep(0.1)
    return False, {k: snap.get(k) for k in exp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chrome", default=None)
    args = ap.parse_args()
    chrome, url = find_chrome(args.chrome), serve()
    print(f"chrome: {chrome}\nurl:    {url}")
    page = Page(chrome)
    fails, checks = 0, []  # checks: (name, expected, actual) after the scenes
    try:
        page.call("Page.navigate", url=url)
        t0 = time.time()
        for name, exp in STEPS:
            ok, got = wait_for(page, exp, timeout=15)
            fails += not ok
            print(f"{'PASS' if ok else 'FAIL'} {time.time() - t0:5.1f}s {name}\n      expected {exp}" + ("" if ok else f"\n      actual   {got}"))
        page.eval("document.querySelector('#alertok').click()")
        page.key("keyDown"); time.sleep(1.2); page.key("keyUp")
        name, exp = MANUAL
        ok, got = wait_for(page, exp, timeout=5)
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'} {time.time() - t0:5.1f}s {name}\n      expected {exp}" + ("" if ok else f"\n      actual   {got}"))
        for _ in range(50):  # the take ends with the LLM's verdict and its log entry: wait for it, so it lands before the checks below
            if page.eval("[...document.querySelectorAll('#log .msg.patient .t')].some(e=>e.textContent==='I need water')"):
                break
            time.sleep(0.1)
        page.eval("document.querySelector('#dplay').click()")  # pause the scenes: the rest injects events directly
        for w, h in ((1920, 1080), (1280, 720)):
            page.call("Emulation.setDeviceMetricsOverride", width=w, height=h, deviceScaleFactor=1, mobile=False)
            page.eval("document.querySelector('#log').innerHTML=''")
            checks.append((f"long conversation (40 messages) at {w}x{h}", LONG_LOG_EXP, page.eval(LONG_LOG_JS)))
        checks.append(("HTML payload in every rendered string field", FUZZ_EXP, page.eval(FUZZ_JS)))
        checks.append(("hand signals: thumb / point chips, hand tracker off", HANDS_EXP, page.eval(HANDS_JS)))
        checks.append(("speak(): face emotion for non-lip text, no fallback for a superseded clip (by a clip or browser speech)", SPEAK_EXP, page.eval(SPEAK_JS)))
        for name, exp, got in checks:
            ok = got == exp
            fails += not ok
            print(f"{'PASS' if ok else 'FAIL'} {time.time() - t0:5.1f}s {name}\n      expected {exp}\n      actual   {got}")
    finally:
        exc = list(page.exceptions)
        page.close()
    print(f"page JS exceptions: {exc or 'none'}")
    n = len(STEPS) + 1 + len(checks)
    print(f"{n - fails}/{n} steps reached")
    sys.exit(1 if fails or exc else 0)


if __name__ == "__main__":
    main()
