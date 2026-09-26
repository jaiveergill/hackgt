"""Check the "Sounds like: X?" confirmation loop end to end and measure its latency. Run from the workspace root:

    .venv/bin/python scripts/check_confirm.py [--clip data/samples/ted1_short.mp4] [--voice Bella]

Boots the real server headless with MOCK_SIGNALS=1 (so POST /api/signal can stand in for the nod/shake detector), decodes
a clip whose phrase-mode match is weak, and runs one scenario per decode. Prints expected vs actual for each scenario plus
latencies. It plays the UI's part: fetch the prompt audio (only timed with --voice, through /api/tts), report
"prompt_played", wait a human reaction time, then send the gesture.
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
    # superseded if the new decode lands inside the 4 s answer window, timeout if the machine is slower; either way nothing is spoken
    ("an unrelated new utterance never confirms the open question", ["new_utterance"], (("rejected", "timeout"), 0)),
    ("low-confidence nod is ignored, then silence", ["weak_nod"], (("timeout",), None)),
]


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


async def scenario(http, ev, ws, clip, gestures, voice):
    """Returns (final confirm event or None, alternatives offered, measurements)."""
    await ev.settle()
    res = (await http.post("/api/decode_file", params={"path": clip})).json()
    uid = res["utt_id"]
    dec = await ev.wait(lambda e: e["type"] == "decision" and e["utt_id"] == uid, 5)
    m = {"decode_total_s": round(res["latency_total"], 3), "confidence": round(res["confidence"], 3), "action": dec and dec["action"]}
    if not dec or dec["action"] != "confirm":
        return None, [], m
    alts = [a["text"] for a in dec["alternatives"]]
    final = None
    for k, g in enumerate(gestures):
        ask = await ev.wait(lambda e: e["type"] == "confirm" and e["utt_id"] == uid and e["state"] == "asking" and e["attempt"] == k + 1, 5)
        if ask is None:
            break
        m.setdefault("ask_since_utterance_s", ask["latency"]["since_utterance"])
        if voice:
            t = time.time(); a = await http.get("/api/tts", params={"text": ask["say"], "voice": voice})
            m.setdefault("prompt_tts", []).append({"s": round(time.time() - t, 3), "cached": a.headers.get("X-TTS-Cached"), "status": a.status_code})
        await ws.send(json.dumps({"cmd": "prompt_played", "utt_id": uid, "attempt": k + 1}))
        t_played = time.time()
        await asyncio.sleep(REACT)
        if g == "new_utterance":
            await http.post("/api/decode_file", params={"path": clip})
            final = await ev.wait(lambda e: e["type"] == "confirm" and e["utt_id"] == uid and e["state"] != "asking", 5)
            m["reason"] = final and final.get("reason")
            break
        if g == "weak_nod":
            await http.post("/api/signal", params={"kind": "nod", "confidence": 0.3})
            final = await ev.wait(lambda e: e["type"] == "confirm" and e["utt_id"] == uid and e["state"] != "asking", TIMEOUT + 2)
            m["timeout_after_played_s"] = final and round(final["_t"] - t_played, 2)
            break
        t = time.time()
        await http.post("/api/signal", params={"kind": g, "confidence": 0.9})
        final = await ev.wait(lambda e: e["type"] == "confirm" and e["utt_id"] == uid and e["state"] in ("confirmed", "rejected") and e["attempt"] == k + 1, 5)
        if final:
            m.setdefault("gesture_to_event_ms", []).append(round((final["_t"] - t) * 1000, 1))
        if final is None or final["state"] == "confirmed":
            break
    if final and final["state"] == "confirmed":
        d = await ev.wait(lambda e: e["type"] == "decision" and e["utt_id"] == uid and e["action"] == "speak", 2)
        m["spoken_decision"] = d and d["text"]
        if voice:
            t = time.time(); a = await http.get("/api/tts", params={"text": final["say"], "voice": voice})
            m["confirmed_tts"] = {"s": round(time.time() - t, 3), "cached": a.headers.get("X-TTS-Cached"), "status": a.status_code}
    return final, alts, m


async def run(base, clip, voice):
    ok = True
    async with httpx.AsyncClient(base_url=base, timeout=120) as http, websockets.connect(base.replace("http", "ws") + "/ws", max_size=None) as ws:
        await ws.recv()  # hello
        await http.post("/api/decode_file", params={"path": clip})  # warm-up; its question is closed by the first settle()
        if voice:  # the UI loads /api/voices at page load; the first lookup of a stock voice is a one-time network call
            t = time.time(); await http.get("/api/voices")
            print(f"voice list warm-up (once per server process, as the UI does on page load): {time.time() - t:.1f} s")
        ev = Events(ws)
        for name, gestures, (want_states, want_idx) in SCENARIOS:
            final, alts, m = await scenario(http, ev, ws, clip, gestures, voice)
            asked = min(len(gestures), len(alts))
            want_cand = alts[want_idx if want_idx is not None else asked - 1] if alts else None
            got_state, got_cand = (final["state"], final["candidate"]) if final else (None, None)
            passed = got_state in want_states and got_cand == want_cand \
                and (got_state != "confirmed" or m.get("spoken_decision") == want_cand) \
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
    lock = open(os.path.expanduser("~/.cache/silent_running/smoke.lock"), "w")  # same machine-wide lock as smoke.py (8 GB RAM)
    print("waiting for the machine-wide server lock ...", flush=True)
    fcntl.flock(lock, fcntl.LOCK_EX)
    port = free_port()
    log = open(os.path.join(ROOT, ".context", "check_confirm_server.log"), "w")
    srv = subprocess.Popen([sys.executable, "-m", "silent_running.server", "--no-camera", "--port", str(port)], cwd=ROOT,
                           stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "MOCK_SIGNALS": "1"})
    try:
        base = f"http://127.0.0.1:{port}"
        for _ in range(180):
            try:
                if httpx.get(base + "/api/state", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(1)
        ok = asyncio.run(run(base, os.path.abspath(args.clip), args.voice))
    finally:
        srv.terminate(); srv.wait(10)
    print("\nCHECK_CONFIRM " + ("PASSED" if ok else "FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
