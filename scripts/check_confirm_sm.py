"""No-model checks of the confirmation loop's server wiring (seconds, no server, no lock needed). Run from the workspace root:

    .venv/bin/python scripts/check_confirm_sm.py

1. Camera signals: a malformed ("signal", ...) message sent down the camera pipe gives an `error` event naming the bad
   field, the camera reader thread stays alive and previews keep updating, even while the confirm loop is busy
   (CameraProcess with a fake pipe, no worker process).
2. Mouthed yes/no: a confident "yes" whose utterance ended before "Sounds like: X?" was asked (it queued behind
   work_lock) neither answers nor supersedes X; one that ended after it answers it. Runs server.run_decode with a
   stub engine that reads "Yes".
3. played(): only the first report per attempt starts the answer window (every open UI tab reports it).
4. Voices: prompts are marked for the system voice, the confirmed phrase for the patient's voice.
Prints expected vs actual per check; exit code 0 = all passed.
"""
import os, sys, time, threading, multiprocessing as mp
import numpy as np

sys.path.insert(0, os.getcwd())
from silent_running import server, confirm, camera_proc
from silent_running.context import ContextStore

EVENTS = []
server.broadcast = EVENTS.append  # capture what would go to the UI over /ws
server._safe_timing = lambda enc, text: None  # word timing needs the real model
server.context = ContextStore()
server.confirm_loop = confirm.ConfirmLoop(server._confirm_emit, server._on_confirmed, timeout=0.5)
ok = True


def check(name, expected, actual):
    global ok
    ok &= expected == actual
    print(f"{'PASS' if expected == actual else 'FAIL'} {name}: expected {expected!r}, got {actual!r}")


def wait_for(pred, timeout=2.0):
    t = time.time()
    while time.time() - t < timeout:
        if pred():
            return True
        time.sleep(0.01)
    return False


def events(kind, **match):
    return [e for e in EVENTS if e["type"] == kind and all(e.get(k) == v for k, v in match.items())]


def question(uid, *alts):
    return {"type": "decision", "utt_id": uid, "alternatives": [{"text": a, "confidence": 0.3} for a in alts]}


# ---------------------------------------------------------------- 1. camera signals never stop the reader
class FakePipeCamera(camera_proc.CameraProcess):
    def _spawn(self):
        self.conn, self.worker = mp.Pipe()


cam = FakePipeCamera()
cam.on_signal = server._on_camera_signal  # wired as server.main() does
threading.Thread(target=server._signal_worker, daemon=True).start()
reader_alive = lambda: any(t.name.endswith("(_reader)") and t.is_alive() for t in threading.enumerate())
frame = 0


def preview_updates():
    global frame
    frame += 1
    jpg = f"jpg{frame}".encode()
    cam.worker.send(("preview", jpg, {"fps": 25.0}))
    return wait_for(lambda: cam.preview_jpeg == jpg)


for bad, field in [({"kind": "nod", "value": "yes", "confidence": None}, "signal.confidence"),
                   ({"kind": "wink", "confidence": 0.9}, "signal.kind"),
                   ({"kind": "nod", "confidence": np.float32(0.9)}, "signal.confidence"),
                   ({"kind": "nod", "confidence": 0.9, "ts": "now"}, "signal.ts"),
                   ({"kind": "nod", "confidence": 0.9, "ts": time.monotonic()}, "signal.ts"),
                   ({"kind": "thumb", "value": ["up"], "confidence": 0.9}, "signal.value"),
                   ("nod", "signal must be a dict")]:
    EVENTS.clear()
    cam.worker.send(("signal", bad))
    wait_for(lambda: events("error"))
    err = [e["message"] for e in events("error")]
    check(f"malformed signal {bad!r} -> error event naming {field}", True, bool(err) and field in err[0])
    check("  reader thread alive, preview still updates", (True, True), (reader_alive(), preview_updates()))

EVENTS.clear()
server.confirm_loop.start(question(1, "Water"))
with server.confirm_loop.lock:  # the loop is busy: the reader must still deliver previews while the nod waits
    cam.worker.send(("signal", {"kind": "nod", "value": "yes", "confidence": 0.9, "ts": time.time()}))
    check("preview updates while the confirm loop is busy with the nod", True, preview_updates())
wait_for(lambda: events("confirm", state="confirmed"))
check("the nod then confirms the question", ["Water"], [e["candidate"] for e in events("confirm", state="confirmed")])
wait_for(lambda: events("decision", utt_id=1))
check("confirmed decision is labelled a gesture", ["gesture"], [e["source"] for e in events("decision", utt_id=1)])
check("prompt / confirmed phrase voices", ("system", "patient"),
      (events("confirm", state="asking")[0].get("say_voice"), events("confirm", state="confirmed")[0].get("say_voice")))


# ---------------------------------------------------------------- 2. mouthed yes: the utterance end time is its timestamp
class StubEngine:
    def to_model_input(self, rois): return np.zeros((1, len(rois), 88, 88), np.float32)
    def encode(self, x): return "enc"
    def ctc_greedy(self, enc): return "YES"
    def score_phrases(self, enc, phrases): return [{"phrase": p, "score": -1.0} for p in phrases]


class StubPhraseDecoder:  # a confident, in-inventory "Yes"
    def decode(self, enc):
        rows = [{"phrase": "Yes", "vsr_score": -1.0, "final_prob": 0.9}, {"phrase": "No", "vsr_score": -4.0, "final_prob": 0.1}]
        return {"ranking": rows, "selected": "Yes", "confidence": 0.9, "margin": 3.0, "visual_top": "Yes", "context_changed_choice": False}


server.engine, server.phrase_decoder = StubEngine(), StubPhraseDecoder()
rois = np.zeros((20, 96, 96), np.uint8)

EVENTS.clear()
server.work_lock.acquire()  # the previous utterance is still decoding ...
th = threading.Thread(target=server.run_decode, args=(rois, 0.8)); th.start()  # ... when "yes" ends and queues
time.sleep(0.2)
server.confirm_loop.start(question(100, "Water", "Cold"))  # the previous decode then asks about its own guess
server.work_lock.release(); th.join()
res = events("result")[0]
fin = [(e["state"], e.get("reason"), e.get("by")) for e in events("confirm", utt_id=100) if e["state"] != "asking"]
check("yes mouthed before the question was asked neither answers nor supersedes it", ([], True), (fin, server.confirm_loop.pending()))
check("  it is spoken as the patient's words", "speak", res["action"])
check("  queueing behind work_lock is reported and counted in the total", (True, True),
      (res["latency"]["lock"] >= 0.2, res["latency"]["total"] >= res["latency"]["lock"] and res["latency_total"] >= 0.2))
server.confirm_loop.cancel()

EVENTS.clear()
server.confirm_loop.start(question(200, "Water", "Cold"))
time.sleep(0.05)
server.run_decode(rois, 0.8)  # "yes" mouthed after the question
fin = [(e["state"], e["candidate"], e.get("by")) for e in events("confirm", utt_id=200) if e["state"] != "asking"]
check("yes mouthed after the question answers it", [("confirmed", "Water", "lips")], fin)
wait_for(lambda: events("decision", utt_id=200, action="speak"))
check("  result action / confirmed decision source", ("answer", ["lips"]),
      (events("result")[0]["action"], [e["source"] for e in events("decision", utt_id=200, action="speak")]))

# ---------------------------------------------------------------- 3. played(): first report per attempt only
EVENTS.clear()
loop = server.confirm_loop
loop.start(question(300, "Water", "Cold"))
t = time.time()
first = loop.played(300, 1)
time.sleep(0.3)
second = loop.played(300, 1)  # a second UI tab finishes the same prompt later
wait_for(lambda: events("confirm", utt_id=300, state="timeout"), 3)
dt = events("confirm", utt_id=300, state="timeout")[0]["ts"] - t
check("first played() starts the window, a second report is ignored", (True, False), (first, second))
check(f"  timeout {dt:.2f} s after the first report (window 0.5 s)", True, abs(dt - 0.5) < 0.1)
loop.start(question(301, "Water", "Cold"))
loop.played(301, 1); loop.answer("no", source="nurse")
check("the next attempt's played() counts again", True, loop.played(301, 2))
loop.cancel()

cam.stop()
print("\nCHECK_CONFIRM_SM " + ("PASSED" if ok else "FAILED"))
sys.exit(0 if ok else 1)
