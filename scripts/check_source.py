"""Check that a recorded file drives the exact same live pipeline as a camera (Agent 1, milestone 1).

    .venv/bin/python scripts/check_source.py [--clip data/samples/ted1_short.mp4] [--expect "HAND GESTURES"]

Boots the real server with `--source file:<clip>` (looping, real time) and compares three paths on the same clip:
  1. reference: POST /api/decode_file (offline read of the whole file)
  2. manual:    ws start at the top of a loop, ws stop after one play-through (live camera process, preview, crop)
  3. auto:      auto-listen on for a few loops (mouth-motion segmentation in the camera process)
Also checks POST /api/source rejects a bad spec / missing file with a visible error and recovers.
Prints expected vs actual and exits 1 on failure. Shares smoke.py's machine-wide lock (8 GB RAM).
"""
import argparse, fcntl, json, os, socket, subprocess, sys, threading, time, urllib.parse, urllib.request
from websockets.sync.client import connect

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
failures = []


def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg, flush=True)
    if not ok:
        failures.append(msg)


def http(method, url, timeout=60):
    req = urllib.request.Request(url, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"null")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="data/samples/ted1_short.mp4")
    ap.add_argument("--expect", default="HAND GESTURES", help="substring the greedy transcript must contain")
    ap.add_argument("--auto-loops", type=int, default=3)
    ap.add_argument("--gap", type=float, default=3.0, help="frozen seconds between plays. The TED clip is continuous speech, "
                    "so auto-listen needs a pause to learn its noise floor; a frozen frame has zero motion, so this only proves "
                    "the pipeline is wired, not segmentation quality (milestone 2 uses real recorded pauses)")
    args = ap.parse_args()
    clip = os.path.relpath(os.path.join(ROOT, args.clip), ROOT)

    lock = open(os.path.expanduser("~/.cache/silent_running/smoke.lock"), "w")
    print("waiting for the machine-wide smoke lock ...", flush=True)
    fcntl.flock(lock, fcntl.LOCK_EX)
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    base = f"http://127.0.0.1:{port}"
    log_path = os.path.join(ROOT, ".context", "check_source_server.log")
    srv = subprocess.Popen([sys.executable, "-u", "-m", "silent_running.server", "--port", str(port), "--source", f"file:{clip}?gap={args.gap}"],
                           cwd=ROOT, stdout=open(log_path, "w"), stderr=subprocess.STDOUT)
    events, lock_ev = [], threading.Lock()
    try:
        for _ in range(180):
            try:
                if http("GET", base + "/api/state", timeout=2)[0] == 200:
                    break
            except Exception:
                time.sleep(1)
        state = lambda: http("GET", base + "/api/state")[1]["camera"]
        cam = state()
        check(cam["source"]["kind"] == "file" and cam["source"]["path"] == clip, f"server opened file source: {cam['source']}")
        check(not cam["source"]["mirror"], "file preview is not mirrored")

        ws = connect(f"ws://127.0.0.1:{port}/ws", max_size=None)
        def reader():
            for m in ws:
                e = json.loads(m)
                with lock_ev:
                    events.append((time.time(), e))
        threading.Thread(target=reader, daemon=True).start()

        def wait_result(after, timeout):
            t_end = time.time() + timeout
            while time.time() < t_end:
                with lock_ev:
                    hit = [e for t, e in events if t >= after and e["type"] in ("result", "error")]
                if hit:
                    return hit[0]
                time.sleep(0.05)
            return None

        # 1. reference
        code, ref = http("POST", f"{base}/api/decode_file?path={urllib.parse.quote(clip)}", timeout=120)
        ref_text = (ref or {}).get("raw_greedy", "")
        print(f"reference decode_file: {ref_text!r}")
        check(code == 200 and args.expect in ref_text.upper(), f"reference contains {args.expect!r}")

        # 2. manual listen over exactly one play-through (start in the still gap before the clip restarts)
        fps, plays = cam["source"]["fps"], cam["source"]["plays"]
        n_frames = int(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v", "-show_entries",
                                       "stream=nb_read_frames", "-of", "csv=p=0", os.path.join(ROOT, clip)], capture_output=True, text=True).stdout.strip())
        while (src := state()["source"])["plays"] == plays:
            time.sleep(0.05)
        t0 = time.time()
        ws.send(json.dumps({"cmd": "start"}))  # we are in the still gap; the clip's frame 0 is due at t_play
        time.sleep(max(src["t_play"] + n_frames / fps + 0.05 - time.time(), 0))
        ws.send(json.dumps({"cmd": "stop"}))
        t_stop = time.time()
        r = wait_result(t0, 30)
        live = (r or {}).get("raw_greedy", "")
        print(f"live manual: {live!r} (n_frames={r and r.get('n_frames')}, stop->result {time.time() - t_stop:.2f}s, "
              f"latency={r and r.get('latency')}, camera fps={state()['fps']}, load={os.getloadavg()[0]:.1f})")
        check(r is not None and r["type"] == "result" and args.expect in live.upper(), f"live manual listen contains {args.expect!r}")

        # 3. auto-listen for a few loops
        ws.send(json.dumps({"cmd": "settings", "auto_listen": True}))
        t_auto = time.time()
        time.sleep(args.auto_loops * (args.gap + n_frames / fps) + 3.0)
        ws.send(json.dumps({"cmd": "settings", "auto_listen": False}))
        time.sleep(2.0)
        with lock_ev:
            autos = [e for t, e in events if t >= t_auto and e["type"] in ("result", "error")]
        for e in autos:
            print(f"  auto {e['type']}: {e.get('raw_greedy', e.get('message'))!r} dur={e.get('duration')}")
        hits = sum(1 for e in autos if e["type"] == "result" and args.expect in e.get("raw_greedy", "").upper())
        check(hits >= 1, f"auto-listen segmented + decoded the clip: {hits} correct of {len(autos)} utterances over {args.auto_loops} loops")

        # 4. source switching: bad spec, missing file, then back
        code, body = http("POST", f"{base}/api/source?spec=bogus")
        check(code == 400, f"bad spec rejected: {code} {body}")
        t_err = time.time()
        code, body = http("POST", f"{base}/api/source?spec=" + urllib.parse.quote("file:data/nope.mp4"))
        time.sleep(0.5)
        with lock_ev:
            errs = [e["message"] for t, e in events if t >= t_err and e["type"] == "error"]
        cam = state()
        check(code == 400 and any("not found" in m for m in errs) and cam["source"]["path"] == clip,
              f"missing file -> 400 + error event, current source kept: {code} {errs} {cam['source']['path']}")
        t_sw = time.time()
        code, body = http("POST", f"{base}/api/source?spec=" + urllib.parse.quote(f"file:{clip}?loop=0"))
        print(f"switch to a new source took {time.time() - t_sw:.1f}s")
        time.sleep(2.0)
        cam = state()
        check(code == 200 and not cam["source"]["loop"] and cam["fps"] > 0,
              f"live switch to a new file source: loop={cam['source']['loop']} preview fps={cam['fps']} face={cam['face']}")
        ws.close()
    finally:
        srv.terminate()
        try:
            srv.wait(10)
        except subprocess.TimeoutExpired:
            srv.kill()
    print(f"\nserver log: {os.path.relpath(log_path, ROOT)}")
    print("CHECK_SOURCE " + ("PASSED" if not failures else f"FAILED ({len(failures)}): " + "; ".join(failures)))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
