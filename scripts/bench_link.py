"""Benchmark the ESP32-CAM link: which stream settings survive the WiFi link the glasses actually get.

Freezes happen when the stream needs more than the link can carry (a head shadowing the board's antenna drops its capacity),
so each setting is judged on what arrives: frame rate, freezes (gaps over 0.3 s), KB per frame and Mbit/s the stream needs,
pings to the board, and the mouth's width in pixels (camera_proc.MouthPixels, the crop reads 45). Settings run in
alternating rounds (A B C A B C ...) so a change in the room (someone walks by, the head turns) hits every setting alike.
Wear the glasses and look at a face (or a photo of one) at the nurse's distance the whole time.

    python scripts/bench_link.py 172.20.10.2                        # current / window=2x / window=2x + quality=20
    python scripts/bench_link.py 192.168.4.1 --seconds 30 --rounds 3
    python scripts/bench_link.py 172.20.10.2 --configs "quality=16" "quality=20"
    python scripts/bench_link.py --fake    # SIMULATION: checks this script against scripts/fake_esp32.py, not the real link
"""
import argparse, os, re, subprocess, sys, threading, time
import numpy as np, cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from silent_running.sources import make_source
from silent_running.camera_proc import MouthPixels, VideoProcess
from silent_running.expression import FaceLandmarker

CONFIGS = ["framesize=9&quality=16", "framesize=9&quality=16&window=2x", "framesize=9&quality=20&window=2x"]
GAP_S = 0.3


def pinger(host, stop, out):
    """One ping a second: RTT in ms, or None for a lost probe."""
    while not stop.is_set():
        r = subprocess.run(["ping", "-c", "1", "-W", "1000", host], capture_output=True, text=True)
        m = re.search(r"time=([\d.]+)", r.stdout)
        out.append(float(m.group(1)) if m else None)
        stop.wait(1.0)


def run(spec, seconds, board, fl, mp, clock):
    """Frames of one setting for `seconds`: arrival times, JPEG sizes, mouth widths."""
    src = make_source(spec)
    if board:
        src.board = board
    src.open()
    ts, kb, mouth = [], [], []
    t_end = time.time() + seconds
    try:
        while time.time() < t_end:
            ok, bgr, t = src.read()
            if not ok:
                continue
            ts.append(t); kb.append(src.frame_bytes / 1024)
            if len(ts) % 3 == 0:  # MediaPipe on every frame would compete with the reader for CPU
                clock[0] += 1
                lm, _ = fl(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), clock[0] * 40)
                if lm is not None:
                    mouth.append(mp(lm))
    finally:
        src.release()
    return ts, kb, mouth


def summary(runs, seconds, pings):
    ts_all, kb, mouth, iv = [], [], [], []
    for ts, k, m in runs:
        kb += k; mouth += m; ts_all.append(len(ts))
        iv += list(np.diff(ts))
    iv = np.array(iv) if iv else np.array([seconds])
    gaps = iv[iv > GAP_S]
    total = seconds * len(runs)
    fps = sum(ts_all) / total
    lost = sum(p is None for p in pings)
    rtt = [p for p in pings if p is not None]
    return {"fps": f"{fps:.1f}", "freezes/min": f"{len(gaps) / total * 60:.1f}", "longest": f"{iv.max():.2f} s",
            "frozen": f"{gaps.sum() / total:.0%}", "KB/frame": f"{np.median(kb):.1f}" if kb else "-",
            "Mbit/s": f"{np.mean(kb) * 8 * 1024 * fps / 1e6:.2f}" if kb else "-",
            "ping p50/p95": f"{np.percentile(rtt, 50):.0f}/{np.percentile(rtt, 95):.0f} ms" if rtt else "-",
            "ping lost": f"{lost}/{len(pings)}" if pings else "-",
            "mouth px": f"{np.median(mouth):.0f}" if mouth else "no face"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("host", nargs="?", default="172.20.10.2")
    ap.add_argument("--configs", nargs="+", default=CONFIGS, help="query strings for stream:<host>?<config>")
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--fake", action="store_true", help="SIMULATION against scripts/fake_esp32.py (checks this script, not the link)")
    a = ap.parse_args()
    board = None
    if a.fake:
        from fake_esp32 import FakeBoard
        fake = FakeBoard(link=lambda t: 0.02 if t % 6 < 0.6 else (1.0 if t % 6 < 2.5 else 4.0))
        port = fake.serve()
        host, board = f"127.0.0.1:{port}", f"http://127.0.0.1:{port}"
        print("SIMULATION: fake board; link 4 Mbit/s, every 6 s nearly dead for 0.6 s (a retransmit stall) then 1 Mbit/s for 1.9 s."
              " Capacity only (no packet loss, no TCP backoff), JPEG quality not modelled: checks the script, not the link")
    else:
        host = a.host
    fl, mp, clock = FaceLandmarker(), MouthPixels(VideoProcess(convert_gray=True)), [0]
    runs = {c: [] for c in a.configs}
    pings = {c: [] for c in a.configs}
    for r in range(a.rounds):
        for c in a.configs:
            spec = f"stream:http://{host}/stream?{c}" if a.fake else f"stream:{host}?{c}"
            stop, got = threading.Event(), []
            if not a.fake:
                threading.Thread(target=pinger, args=(host, stop, got), daemon=True).start()
            print(f"round {r + 1}/{a.rounds}  {c} ...", flush=True)
            try:
                runs[c].append(run(spec, a.seconds, board, fl, mp, clock))
            except RuntimeError as e:
                print(f"   could not open: {e}")
            stop.set()
            pings[c] += got
    rows = [(c, summary(runs[c], a.seconds, pings[c])) for c in a.configs if runs[c]]
    if not rows:
        sys.exit("no setting delivered frames")
    cols = list(rows[0][1])
    print(f"\n{a.rounds} rounds x {a.seconds:g} s per setting, host {host}\n")
    print("| setting | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
    for c, s in rows:
        print(f"| {c} | " + " | ".join(s[k] for k in cols) + " |")


if __name__ == "__main__":
    main()
