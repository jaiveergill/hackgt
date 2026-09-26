"""Check the bedside UI's ?demo=1 replay renders every scripted scene (no backend, no camera needed).

    python scripts/check_ui_demo.py [--chrome /path/to/chrome]

Serves silent_running/ on a free port, opens /static/index.html?demo=1 in headless Chrome driven over the DevTools
protocol, and waits (real time) for each expected UI state in order: hero state / text / source badge, nonverbal chips,
the critical alert, then a manual space-bar take. Prints expected vs actual per step and any page JS exceptions.
Exit code 0 = every step reached and no exceptions.
"""
import argparse, glob, json, os, shutil, socket, subprocess, sys, tempfile, threading, time, urllib.request
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from websockets.sync.client import connect

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Expected states in the order silent_running/static/demo.js plays them (SCENES), then the manual take.
STEPS = [
    ("scene 1: nurse asks, lips + context fused", {"state": "result", "src": "fused", "text": "I AM IN PAIN", "ring": "90%", "asked": "How is your pain right now?", "chips": ["pain"]}),
    ("scene 2: pain score on fingers, gesture fast path", {"state": "result", "src": "gesture", "text": "MY PAIN IS A SEVEN OUT OF TEN", "chips": ["fingers"]}),
    ("scene 3: low confidence asks", {"state": "confirm", "text": "Sounds like…I AM HOT?"}),
    ("scene 3: shake -> next guess", {"state": "confirm", "text": "Sounds like…I AM COLD?", "chips": ["head"]}),
    ("scene 3: nod confirms", {"state": "confirmed", "text": "I AM COLD"}),
    ("scene 4: blink code yes", {"state": "result", "src": "gesture", "text": "YES", "chips": ["blink"]}),
    ("scene 5: critical phrase alert", {"alert": "I CAN'T BREATHE"}),
]
MANUAL = ("manual take: hold space, release", {"state": "result", "src": "fused", "text": "I NEED WATER"})

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


def serve():
    handler = partial(SimpleHTTPRequestHandler, directory=os.path.join(ROOT, "silent_running"))
    handler.log_message = lambda *a: None
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{httpd.server_address[1]}/static/index.html?demo=1"


class Page:
    """Minimal DevTools-protocol client: one page, synchronous calls, JS exceptions collected."""
    def __init__(self, chrome):
        self.profile = tempfile.mkdtemp(prefix="sr_ui_check_")
        s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
        self.proc = subprocess.Popen([chrome, "--headless=new", "--disable-gpu", "--no-first-run", "--mute-audio", "--window-size=1920,1080",
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
        return self.call("Runtime.evaluate", expression=js, returnByValue=True)["result"].get("value")

    def key(self, kind, code=" "):
        self.call("Input.dispatchKeyEvent", type=kind, key=code, code="Space", windowsVirtualKeyCode=32, text=code if kind == "keyDown" else "")

    def close(self):
        self.ws.close(); self.proc.terminate(); self.proc.wait(); shutil.rmtree(self.profile, ignore_errors=True)


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
    fails = 0
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
    finally:
        exc = list(page.exceptions)
        page.close()
    print(f"page JS exceptions: {exc or 'none'}")
    n = len(STEPS) + 1
    print(f"{n - fails}/{n} steps reached")
    sys.exit(1 if fails or exc else 0)


if __name__ == "__main__":
    main()
