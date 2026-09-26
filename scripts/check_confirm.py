"""Check the "Sounds like: X?" confirmation loop end to end and measure its latency. Run from the workspace root:

    .venv/bin/python scripts/check_confirm.py [--clip data/samples/ted1_short.mp4] [--voice Bella]

The "new utterance" scenario needs a second clip that does not read as yes/no (ted1_short reads as a weak "Yes", which
would count as an unclear mouthed answer and leave the question open): 2.6 s cut from ted1_12s.mp4 at 6 s, which reads
as a weak "Sorry".

Boots the real server headless with MOCK_SIGNALS=1 (so POST /api/signal can stand in for the nod/shake detector), decodes
a clip whose phrase-mode match is weak, and runs one scenario per decode. Prints expected vs actual for each scenario plus
latencies. It plays the UI's part: report "prompt_played", wait a human reaction time, send the gesture, and (with --voice)
fetch the confirmed phrase's audio through /api/tts, as the UI does for an ElevenLabs voice. Waits for the server's
warm flag first, so latencies are steady state.

Latencies (end of mouthing = the moment decode_file is called; the clip is already recorded):
  prompt_event_s   end of mouthing -> "asking" event at the client. The UI starts "Sounds like: X?" in the system voice
                   on this event (no ElevenLabs call), so this is the time to its first audio minus the voice's start-up.
  nod_to_audio_s   nod -> the confirmed phrase's audio fully received from /api/tts (`cached` says whether the ElevenLabs
                   cache already had it); nod_to_audio_cached_s is the same with the phrase certainly cached.
"""
import argparse, asyncio, fcntl, json, os, socket, subprocess, sys, time
import httpx, websockets

ROOT = os.getcwd()
sys.path.insert(0, ROOT)
from silent_running.confirm import TIMEOUT

REACT = 0.5  # s between the prompt ending and the simulated gesture
# name, gestures sent one per question, expected (allowed final states, index of the final candidate; None = the last one asked)
SCENARIOS = [
    ("nod on the first guess", ["nod"], (("confirmed",), 0)),
    ("shake, then nod", ["shake", "nod"], (("confirmed",), 1)),
    ("shake every guess", ["shake", "shake", "shake"], (("rejected",), None)),
    # sent before prompt_played: the question is still in its pre-play window, so only the supersede can end it
    ("an unrelated new utterance never confirms the open question", ["new_utterance"], (("rejected",), 0)),
    ("low-confidence nod is ignored, then silence", ["weak_nod"], (("timeout",), None)),
]
SUPERSEDED = "superseded by a new utterance"


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


class Events:
    def __init__(self, ws):
        self.ws, self.seen = ws, []
        self.open = {}  # utt_id -> latest "asking" event of a question that has not ended yet

    async def wait(self, pred, timeout):
        t_end = time.time() + timeout
        while True:
            for e in self.seen:
                if pred(e):
                    self.seen.remove(e); return e
            left = t_end - time.time()
            if left <= 0:
                return None
            try:
                e = json.loads(await asyncio.wait_for(self.ws.recv(), timeout=left))
            except asyncio.TimeoutError:
                return None
            e["_t"] = time.time(); self.seen.append(e)
            if e["type"] == "confirm":
                if e["state"] == "asking":
                    self.open[e["utt_id"]] = e
                else:
                    self.open.pop(e["utt_id"], None)

    async def settle(self):
        """Start each scenario with no open question: report the open prompt as played and let it time out."""
        await self.wait(lambda e: False, 0.3)
        for uid, ask in list(self.open.items()):
            await self.ws.send(json.dumps({"cmd": "prompt_played", "utt_id": uid, "attempt": ask["attempt"]}))
            await self.wait(lambda e: e["type"] == "confirm" and e["utt_id"] == uid and e["state"] != "asking", TIMEOUT + 2)
        self.seen.clear()


async def scenario(http, ev, ws, clip, other_clip, gestures, voice):
    """Returns (final confirm event or None, alternatives offered, measurements)."""
    await ev.settle()
    t_end = time.time()  # end of mouthing
    res = (await http.post("/api/decode_file", params={"path": clip})).json()
    uid = res["utt_id"]
    dec = await ev.wait(lambda e: e["type"] == "decision" and e["utt_id"] == uid, 5)
    lat = (await ev.wait(lambda e: e["type"] == "result" and e["utt_id"] == uid, 1) or {}).get("latency", {})
    m = {"decode_latency_s": {k: round(v, 3) for k, v in lat.items()}, "confidence": round(res["confidence"], 3), "action": dec and dec["action"]}
    if not dec or dec["action"] != "confirm":
        return None, [], m
    alts = [a["text"] for a in dec["alternatives"]]
    final = None
    for k, g in enumerate(gestures):
        ask = await ev.wait(lambda e: e["type"] == "confirm" and e["utt_id"] == uid and e["state"] == "asking" and e["attempt"] == k + 1, 5)
        if ask is None:
            break
        m.setdefault("prompt_event_s", round(ask["_t"] - t_end, 3))
        m.setdefault("prompt_voice", ask.get("say_voice"))
        if g == "new_utterance":
            m["new_utterance_read_as"] = (await http.post("/api/decode_file", params={"path": other_clip})).json()["selected"]
            final = await ev.wait(lambda e: e["type"] == "confirm" and e["utt_id"] == uid and e["state"] != "asking", 5)
            m["reason"] = final and final.get("reason")
            break
        await ws.send(json.dumps({"cmd": "prompt_played", "utt_id": uid, "attempt": k + 1}))
        t_played = time.time()
        await asyncio.sleep(REACT)
        if g == "weak_nod":
            await http.post("/api/signal", params={"kind": "nod", "confidence": 0.3})
            final = await ev.wait(lambda e: e["type"] == "confirm" and e["utt_id"] == uid and e["state"] != "asking", TIMEOUT + 2)
            m["timeout_after_played_s"] = final and round(final["_t"] - t_played, 2)
            break
        t_gesture = time.time()
        await http.post("/api/signal", params={"kind": g, "confidence": 0.9})
        final = await ev.wait(lambda e: e["type"] == "confirm" and e["utt_id"] == uid and e["state"] in ("confirmed", "rejected") and e["attempt"] == k + 1, 5)
        if final:
            m.setdefault("gesture_to_event_ms", []).append(round((final["_t"] - t_gesture) * 1000, 1))
        if final is None or final["state"] == "confirmed":
            break
    if final and final["state"] == "confirmed":
        m["confirmed_voice"] = final.get("say_voice")
        if voice:  # the UI fetches the phrase as soon as the confirmed event arrives
            a = await http.get("/api/tts", params={"text": final["say"], "voice": voice})
            m["nod_to_audio_s"] = {"s": round(time.time() - t_gesture, 3), "cached": a.headers.get("X-TTS-Cached"), "status": a.status_code}
            t = time.time(); a = await http.get("/api/tts", params={"text": final["say"], "voice": voice})  # the same phrase, now cached
            m["nod_to_audio_cached_s"] = {"s": round(final["_t"] - t_gesture + time.time() - t, 3), "cached": a.headers.get("X-TTS-Cached"), "status": a.status_code}
        d = await ev.wait(lambda e: e["type"] == "decision" and e["utt_id"] == uid and e["action"] == "speak", 2)
        m["spoken_decision"] = d and d["text"]
    return final, alts, m


async def run(base, clip, other_clip, voice):
    ok = True
    async with httpx.AsyncClient(base_url=base, timeout=120) as http, websockets.connect(base.replace("http", "ws") + "/ws", max_size=None) as ws:
        await ws.recv()  # hello
        await http.post("/api/decode_file", params={"path": clip})  # warm-up; its question is closed by the first settle()
        if voice:  # the UI loads /api/voices at page load; the first lookup of a stock voice is a one-time network call
            t = time.time(); await http.get("/api/voices")
            print(f"voice list warm-up (once per server process, as the UI does on page load): {time.time() - t:.1f} s")
        ev = Events(ws)
        for name, gestures, (want_states, want_idx) in SCENARIOS:
            final, alts, m = await scenario(http, ev, ws, clip, other_clip, gestures, voice)
            asked = min(len(gestures), len(alts))
            want_cand = alts[want_idx if want_idx is not None else asked - 1] if alts else None
            got_state, got_cand = (final["state"], final["candidate"]) if final else (None, None)
            passed = got_state in want_states and got_cand == want_cand \
                and (got_state != "confirmed" or m.get("spoken_decision") == want_cand) \
                and ("new_utterance" not in gestures or m.get("reason") == SUPERSEDED) \
                and m.get("prompt_voice") == "system" and m.get("confirmed_voice") in (None, "patient") \
                and ("timeout_after_played_s" not in m or abs(m["timeout_after_played_s"] - TIMEOUT) < 0.5)
            ok &= passed
            print(f"{'PASS' if passed else 'FAIL'} {name}: expected {'/'.join(want_states)} {want_cand!r}, got {got_state} {got_cand!r}  alts={alts}")
            print("     " + json.dumps(m))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="data/samples/ted1_short.mp4")
    ap.add_argument("--voice", default=None, help="ElevenLabs voice to time the spoken prompts (e.g. Bella)")
    args = ap.parse_args()
    other_clip = os.path.join(ROOT, ".context", "check_confirm_other.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "6", "-t", "2.6", "-i", os.path.join(ROOT, "data/samples/ted1_12s.mp4"),
                    "-an", "-r", "25", "-c:v", "libx264", "-pix_fmt", "yuv420p", other_clip], check=True)
    lock = open(os.path.expanduser("~/.cache/silent_running/smoke.lock"), "w")  # same machine-wide lock as smoke.py (8 GB RAM)
    print("waiting for the machine-wide server lock ...", flush=True)
    fcntl.flock(lock, fcntl.LOCK_EX)
    port = free_port()
    log = open(os.path.join(ROOT, ".context", "check_confirm_server.log"), "w")
    srv = subprocess.Popen([sys.executable, "-m", "silent_running.server", "--no-camera", "--port", str(port)], cwd=ROOT,
                           stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "MOCK_SIGNALS": "1"})
    try:
        base = f"http://127.0.0.1:{port}"
        t = time.time()
        while time.time() - t < 300:  # engine warmup, then the aligner prewarm (both compete with decodes for CPU)
            try:
                if httpx.get(base + "/api/state", timeout=2).json()["state"]["warm"]:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(1)
        else:
            sys.exit(f"server never reported warm within 300 s; see {log.name}")
        print(f"server warm after {time.time() - t:.0f} s (load {os.getloadavg()[0]:.1f})", flush=True)
        ok = asyncio.run(run(base, os.path.abspath(args.clip), other_clip, args.voice))
    finally:
        srv.terminate(); srv.wait(10)
    print("\nCHECK_CONFIRM " + ("PASSED" if ok else "FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
