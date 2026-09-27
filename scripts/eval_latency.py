"""End-to-end latency on the live path: end of mouthing -> result, decision, and first audio in the real UI. Holds the lock.

  python scripts/eval_latency.py --miracl ~/.cache/silent_running/miracl/full --speakers F01 M01 --takes 1 2 \
      [--voice Bella] [--mode open] [--root <worktree>] [--out results/latency.json]

Builds one video of the clips (each resampled to 25 fps between still stretches of face: 1.5 s before, 3 s after), boots
the server of --root (default: this checkout) headless, opens its UI in headless Chrome, turns hands-free listening on and
plays the video as the live source (POST /api/source file:...?loop=0), so it goes through the real capture process,
auto-listen, decode, decision and the UI's voice. End of mouthing = the wall time the source delivers a clip's last frame
(t_play + index / 25). Measured per clip:
  result / decision   the server event's arrival at a WebSocket client (--mode open: the decision is the LLM's verdict, `llm`)
  audio               the UI's own first-audio mark (latMark(utt, 'audio'): the <audio>/Web Audio playback started)
The UI speaks with --voice (an ElevenLabs voice); without it, the browser voice, whose start headless Chrome does not report.
Prints per-clip rows and medians, with the load average; waits for the server's warm flag first.
"""
import argparse, glob, json, os, socket, statistics as st, subprocess, sys, tempfile, threading, time, urllib.request
import numpy as np, av, cv2
from websockets.sync.client import connect

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.check_ui_demo import Page, find_chrome

FPS, LEAD, PRE, POST = 25, 4.0, 1.5, 3.0
MIRACL = ["Stop navigation", "Excuse me", "I am sorry", "Thank you", "Good bye", "I love this game", "Nice to meet you",
          "You are welcome", "How are you", "Have a good time"]
MARK_JS = """(()=>{window.__marks=[];const f=latMark;latMark=function(uid,k){window.__marks.push({uid,k,t:Date.now()/1000});return f(uid,k)};
  const g=toast;toast=function(m,good){if(m)window.__marks.push({k:'toast',msg:m,t:Date.now()/1000});return g(m,good)};return true})()"""


def frames_25(path):
    with av.open(path) as c:
        fps = float(c.streams.video[0].average_rate)
        fr = [f.to_ndarray(format="bgr24") for f in c.decode(video=0)]
    idx = np.round(np.arange(int(round(len(fr) * FPS / fps))) * fps / FPS).astype(int).clip(0, len(fr) - 1)
    return [fr[i] for i in idx]


def build_video(clips, out):
    """-> [{path, label, k_first, k_last}] with the frame index of each clip's first and last frame in `out`."""
    h, w = frames_25(clips[0]["path"])[0].shape[:2]
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
                           "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    k, rows = 0, []
    for i, c in enumerate(clips):
        fr = [cv2.resize(f, (w, h)) if f.shape[:2] != (h, w) else f for f in frames_25(c["path"])]
        pad = [fr[0]] * int((LEAD if i == 0 else PRE) * FPS)
        for f in pad + fr + [fr[-1]] * int(POST * FPS):
            ff.stdin.write(f.tobytes())
        rows.append({**c, "k_first": k + len(pad), "k_last": k + len(pad) + len(fr) - 1})
        k += len(pad) + len(fr) + int(POST * FPS)
    ff.stdin.close(); ff.wait()
    return rows, k


def http(url, method="GET", timeout=120):
    with urllib.request.urlopen(urllib.request.Request(url, method=method), timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--miracl", required=True)
    ap.add_argument("--speakers", nargs="+", default=["F01", "M01"])
    ap.add_argument("--takes", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--voice", default=None, help="ElevenLabs voice for the UI (e.g. Bella); default: the browser voice")
    ap.add_argument("--pace", choices=("on", "off"), default=None, help="set the UI's 'match my pace' option (default: as the UI starts)")
    ap.add_argument("--mode", choices=("phrase", "open"), default="phrase", help="Phrase Mode, or Open Mode (beam search + the LLM's reading)")
    ap.add_argument("--root", default=ROOT, help="worktree whose server is measured")
    ap.add_argument("--chrome", default=None)
    ap.add_argument("--out", default=None, help="write per-clip rows here (JSON)")
    a = ap.parse_args()
    d = os.path.expanduser(a.miracl)
    clips = [{"path": p, "label": MIRACL[int(p.split("/")[-2]) - 1]} for p in sorted(glob.glob(os.path.join(d, "*/*/*.mp4")))
             if p.split("/")[-3] in a.speakers and int(p.split("/")[-1][:-4]) in a.takes]
    video = os.path.join(tempfile.mkdtemp(prefix="sr_latency_"), "playlist.mp4")
    clips, n_frames = build_video(clips, video)
    print(f"{len(clips)} clips, {n_frames / FPS:.0f} s of video", flush=True)

    import fcntl
    lock = open(os.path.expanduser("~/.cache/silent_running/smoke.lock"), "w")  # same machine-wide lock as smoke.py (8 GB RAM)
    print("waiting for the machine-wide lock ...", flush=True)
    fcntl.flock(lock, fcntl.LOCK_EX)
    loads = [os.getloadavg()[0]]
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    base = f"http://127.0.0.1:{port}"
    log = open(os.path.join(os.path.dirname(video), "server.log"), "w")
    srv = subprocess.Popen([os.path.join(a.root, ".venv/bin/python"), "-m", "silent_running.server", "--no-camera", "--port", str(port)],
                           cwd=a.root, stdout=log, stderr=subprocess.STDOUT)
    page, events = None, []
    try:
        for _ in range(240):
            try:
                if http(base + "/api/state", timeout=2)["state"].get("warm"):
                    break
            except OSError:
                pass
            time.sleep(0.5)
        else:
            sys.exit("server did not report warm within 120 s")
        ws = connect(f"ws://127.0.0.1:{port}/ws", max_size=None)
        stop = threading.Event()

        def record():
            while not stop.is_set():
                try:
                    m = json.loads(ws.recv(timeout=0.5))
                except TimeoutError:
                    continue
                m["_t"] = time.time()
                events.append(m)
        threading.Thread(target=record, daemon=True).start()
        metas = []  # /ws_meta at 10 Hz: the capture process's state (auto-listen, motion energy, fps), for diagnosis

        def record_meta():
            with connect(f"ws://127.0.0.1:{port}/ws_meta", max_size=None) as wm:
                while not stop.is_set():
                    m = json.loads(wm.recv())
                    metas.append({"_t": time.time(), **{k: m.get(k) for k in ("status", "auto", "listening", "mouth_active", "energy", "noise", "fps", "face", "n_frames")}})
        threading.Thread(target=record_meta, daemon=True).start()
        page = Page(a.chrome or find_chrome(None))
        page.call("Page.enable")
        page.call("Page.navigate", url=base + "/")
        for _ in range(100):
            if page.eval("typeof latMark==='function'&&!!document.querySelector('#voice')"):
                break
            time.sleep(0.2)
        page.eval(MARK_JS)
        if a.pace:
            if not page.eval(f"(()=>{{const c=document.querySelector('#pace');if(c)c.checked={'true' if a.pace == 'on' else 'false'};return !!c}})()"):
                sys.exit("this UI has no 'match my pace' option (#pace)")
        if a.voice:
            for _ in range(100):  # the UI lists ElevenLabs voices once /api/voices answers
                if page.eval(f"[...document.querySelector('#voice').options].some(o=>o.value==='clone:{a.voice}')"):
                    break
                time.sleep(0.2)
            page.eval(f"(()=>{{const s=document.querySelector('#voice');s.value='clone:{a.voice}';s.dispatchEvent(new Event('change'));return s.value}})()")
            time.sleep(3)  # the UI's own work on a voice change (e.g. loading audio) finishes before the clips play
        info = http(base + "/api/source?spec=" + urllib.request.quote(f"file:{video}?loop=0"), method="POST")["source"]
        ws.send(json.dumps({"cmd": "mode", "mode": a.mode}))
        ws.send(json.dumps({"cmd": "settings", "auto_listen": True}))  # needs the camera; the video starts with LEAD s of still face
        t_play = info["t_play"]
        print(f"playing from {time.strftime('%H:%M:%S', time.localtime(t_play))}, load {os.getloadavg()[0]:.1f}", flush=True)
        t_end = t_play + n_frames / FPS + 6
        while time.time() < t_end:
            time.sleep(1)
            loads.append(os.getloadavg()[0])
        marks = page.eval("window.__marks") or []
        stop.set()
    finally:
        if page:
            page.close()
        srv.terminate()
        try:
            srv.wait(15)
        except subprocess.TimeoutExpired:
            srv.kill(); srv.wait()
    leaked = subprocess.run(["pgrep", "-f", "silent_running.server"], capture_output=True, text=True).stdout.split()
    json.dump({"clips": clips, "t_play": t_play, "events": events, "marks": marks, "loads": loads, "metas": metas}, open(os.path.join(os.path.dirname(video), "raw.json"), "w"))
    print(f"server exit {srv.returncode}; silent_running.server processes left: {leaked or 'none'}")

    rows = []
    for i, c in enumerate(clips):
        t_last = t_play + c["k_last"] / FPS
        lo = t_play + c["k_first"] / FPS - PRE + 0.1
        hi = t_play + clips[i + 1]["k_first"] / FPS - PRE + 0.1 if i + 1 < len(clips) else 1e18
        ev = [e for e in events if lo <= e["_t"] < hi]
        res = [e for e in ev if e["type"] == "result"]
        dec = [e for e in ev if e["type"] == ("llm" if a.mode == "open" else "decision")]
        r = {"clip": "/".join(c["path"].split("/")[-3:]), "label": c["label"], "n_results": len(res),
             "errors": [e["message"] for e in ev if e["type"] == "error"]}
        if res:
            uid = res[0]["utt_id"]
            d0 = next((e for e in dec if e["utt_id"] == uid), None)
            audio = next((m["t"] for m in marks if m.get("uid") == uid and m["k"] == "audio"), None)
            r.update(selected=res[0].get("selected"), correct=(res[0].get("selected") or "").lower().replace(" ", "") == c["label"].lower().replace(" ", ""),  # Goodbye
                     confidence=res[0].get("confidence"), action=d0 and d0.get("action", "speak"), stages=res[0].get("latency"),
                     result=res[0]["_t"] - t_last, decision=d0 and d0["_t"] - t_last, audio=audio and audio - t_last,
                     delivery=next(({k: e.get(k) for k in ("cached", "t_synth", "total", "emotion", "rate")} for e in ev
                                    if e["type"] == "delivery" and e.get("utt_id") in (uid, 0)), None))
        r["toasts"] = [m["msg"] for m in marks if m["k"] == "toast" and lo <= m["t"] < hi]
        rows.append(r)
        print(f"{r['clip']:14s} {c['label']:17s} -> {str(r.get('selected')):22s} {r.get('action') or '':8s} "
              + " ".join(f"{k} {r[k]:.2f}s" for k in ("result", "decision", "audio") if r.get(k) is not None)
              + (f"  results={len(res)}" if len(res) != 1 else "") + (f"  errors={r['errors']}" if r["errors"] else "")
              + (f"  UI toasts={r['toasts']}" if r["toasts"] else "") + (f"  delivery={r['delivery']}" if r.get("delivery") else ""), flush=True)

    def med(k, sel=lambda r: True):
        v = sorted(r[k] for r in rows if r.get(k) is not None and sel(r))
        return f"{st.median(v):.2f} s (p90 {v[int(0.9 * len(v))]:.2f}, n={len(v)})" if v else "n/a"
    stage = {k: st.median(v) * 1000 for k in ("lock", "crop", "encode", "phrase", "beam", "read", "total")
             if (v := [r["stages"][k] for r in rows if r.get("stages") and k in r["stages"]])}
    print(f"\nroot {a.root} ({subprocess.run(['git', '-C', a.root, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()}), "
          f"{a.mode} mode, voice {a.voice or 'browser'}, pace {a.pace or 'UI default'}, load {min(loads):.1f}-{max(loads):.1f}")
    print(f"clips {len(rows)}, with a result {sum(r['n_results'] > 0 for r in rows)}, more than one result {sum(r['n_results'] > 1 for r in rows)}, "
          f"top-1 {sum(bool(r.get('correct')) for r in rows)}")
    print(f"end of mouthing -> result   {med('result')}")
    print(f"end of mouthing -> decision {med('decision')}")
    print(f"end of mouthing -> audio    {med('audio')}  (decisions to speak: {sum(r.get('action') == 'speak' for r in rows)})")
    print("server stages, median: " + ", ".join(f"{k} {v:.0f} ms" for k, v in stage.items()))
    if a.out:
        json.dump({"root": a.root, "voice": a.voice, "loads": loads, "rows": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
