"""No-model checks of hands-free utterances sent ahead of their hang (camera_proc `ahead`, server.run_decode's `ahead`).
Seconds, no server, no lock needed. Run from the workspace root:

    .venv/bin/python scripts/check_ahead.py

The capture process's messages go down a fake camera pipe to the real CameraProcess reader, as in check_confirm_sm.py; the
engine is a stub that reads "Yes" and takes `ENCODE_S` to encode. Cases:
  1. confirmed after the reading finished: one result, published only at the confirmation (nothing before it, not even a
     status), latency counted from the confirmation with the reading under "ahead"
  2. dropped (the mouth moved again): no event at all, no utterance number used
  3. confirmed while the reading still runs: the result follows the reading; latency.read = what ran after the confirmation
  4. an utterance that failed in the capture process (face not tracked): its error only once confirmed, nothing if dropped
  5. an utterance sent the old way (no `ahead`) still decodes at once
Prints expected vs actual per check; exit code 0 = all passed.
"""
import os, sys, time, multiprocessing as mp
import numpy as np

sys.path.insert(0, os.getcwd())
from silent_running import server, confirm, camera_proc
from silent_running.context import ContextStore

EVENTS = []
server.broadcast = EVENTS.append  # capture what would go to the UI over /ws
server._safe_timing = lambda enc, text: None  # word timing needs the real model
server.context = ContextStore()
server.confirm_loop = confirm.ConfirmLoop(server._confirm_emit, server._on_confirmed, timeout=0.5)
ENCODE_S = 0.1
ok = True


def check(name, expected, actual):
    global ok
    ok &= expected == actual
    print(f"{'PASS' if expected == actual else 'FAIL'} {name}: expected {expected!r}, got {actual!r}")


def wait_for(pred, timeout=3.0):
    t = time.time()
    while time.time() - t < timeout:
        if pred():
            return True
        time.sleep(0.005)
    return False


def kinds():
    return [e["type"] + (f":{e['status']}" if e["type"] == "status" else "") for e in EVENTS]


class StubEngine:
    def to_model_input(self, rois): return np.zeros((1, len(rois), 88, 88), np.float32)
    def encode(self, x): time.sleep(ENCODE_S); return "enc"
    def ctc_greedy(self, enc): return "YES"


class StubPhraseDecoder:  # a confident, in-inventory "Yes"
    def decode(self, enc, free=None):
        rows = [{"phrase": "Yes", "vsr_score": -1.0, "final_prob": 0.9}, {"phrase": "No", "vsr_score": -4.0, "final_prob": 0.1}]
        return {"ranking": rows, "selected": "Yes", "confidence": 0.9, "margin": 3.0, "visual_top": "Yes", "context_changed_choice": False,
                "free_score": -1.0}


class FakePipeCamera(camera_proc.CameraProcess):
    def _spawn(self):
        self.conn, self.worker = mp.Pipe()


server.engine, server.phrase_decoder = StubEngine(), StubPhraseDecoder()
cam = server.camera = FakePipeCamera()
cam.on_auto_utterance, cam.on_ahead = server._on_auto_utterance, server._on_ahead  # wired as server._make_camera does
rois = np.zeros((20, 96, 96), np.uint8)


def utterance(ahead, rois=rois, err=None):  # camera_proc emit()'s message
    cam.worker.send(("utterance", rois, 20, 20, 0.8, err, 0.002, "auto", None, None, None, ahead))


# ---------------------------------------------------------------- 1. confirmed after the reading
EVENTS.clear(); uid0 = server.STATE["utt_id"]
utterance(1)
time.sleep(ENCODE_S + 0.1)  # the reading is done; the hang has not run out yet
check("1. nothing shown before the confirmation", [], kinds())
t_confirm = time.time(); cam.worker.send(("ahead", 1, True))
wait_for(lambda: "result" in kinds())
res = next(e for e in EVENTS if e["type"] == "result")
check("1. status processing, raw, result, decision, idle", ["status:processing", "raw", "status:idle", "result", "decision"], kinds()[:5])
check("1. one utterance number used", uid0 + 1, res["utt_id"])
check("1. latency: lock and read (0: it was done), total from the confirmation", (["lock", "read", "total"], 0.0, True),
      (list(res["latency"]), res["latency"]["read"], res["latency"]["total"] < 0.05))
check("1. the reading's own stages under ahead, done before the end", (True, True),
      ({"crop", "encode", "phrase"} <= set(res["ahead"]), res["ahead"]["before_end"] > 0))
server.confirm_loop.cancel()

# ---------------------------------------------------------------- 2. dropped
EVENTS.clear(); uid0 = server.STATE["utt_id"]
utterance(2); time.sleep(0.02); cam.worker.send(("ahead", 2, False))
time.sleep(ENCODE_S + 0.2)
check("2. dropped: no event, no utterance number", ([], uid0), (kinds(), server.STATE["utt_id"]))

# ---------------------------------------------------------------- 3. confirmed while the reading runs
EVENTS.clear()
utterance(3); time.sleep(0.02); cam.worker.send(("ahead", 3, True))
wait_for(lambda: "result" in kinds())
res = next(e for e in EVENTS if e["type"] == "result")
check("3. confirmed mid-reading: the result follows it, read > 0 counted in the total", (True, True, True),
      (res["latency"]["read"] > ENCODE_S / 2, res["latency"]["total"] >= res["latency"]["read"], res["ahead"]["before_end"] < 0))
server.confirm_loop.cancel()

# ---------------------------------------------------------------- 4. a failed utterance
EVENTS.clear()
utterance(4, None, "face not tracked"); time.sleep(0.05)
check("4. failed utterance: nothing before the confirmation", [], kinds())
cam.worker.send(("ahead", 4, True))
wait_for(lambda: "error" in kinds())
check("4. then its error", True, any("Face not tracked" in e.get("message", "") for e in EVENTS if e["type"] == "error"))
EVENTS.clear()
utterance(5, None, "face not tracked"); time.sleep(0.02); cam.worker.send(("ahead", 5, False)); time.sleep(0.1)
check("4. dropped failed utterance: no event", [], kinds())

# ---------------------------------------------------------------- 5. the old way
EVENTS.clear(); uid0 = server.STATE["utt_id"]
utterance(None)
wait_for(lambda: "result" in kinds())
res = next((e for e in EVENTS if e["type"] == "result"), {})
check("5. not sent ahead: decoded at once, latency by stage", (uid0 + 1, ["lock", "crop", "encode", "phrase", "total"]),
      (res.get("utt_id"), list(res.get("latency", {}))))
check("5. no pending read-ahead left", {}, server.AHEAD)

print("\nCHECK_AHEAD " + ("PASSED" if ok else "FAILED"))
sys.exit(0 if ok else 1)
