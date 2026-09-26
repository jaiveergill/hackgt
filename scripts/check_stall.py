"""Check that a live camera that stops delivering frames is reported and reopened instead of freezing silently.

    .venv/bin/python scripts/check_stall.py

Runs the real capture worker (silent_running.camera_proc._worker) in a thread against a fake "webcam" that plays
data/samples/ted1_short.mp4, then returns no frames (as OpenCV does for an unplugged camera) until it is reopened.
The first reopen fails (still unplugged), the second succeeds (replugged).
Expected: a stall error, a failed-reopen error with backoff, no crash loop, then previews resuming. No server or VSR model needed.
"""
import os, sys, threading, time, multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from silent_running import sources, camera_proc

CLIP = os.path.join(sources.ROOT, "data/samples/ted1_short.mp4")


class UnpluggedCamera(sources.FileSource):
    """Looks like a webcam. 1 s after the first open, read() fails (unplugged); the 2nd open() fails too; the 3rd succeeds."""
    kind = "webcam"

    def __init__(self):
        super().__init__(CLIP)
        self.opens, self.t_unplug = 0, None

    def open(self):
        self.opens += 1
        if self.opens == 2:
            raise RuntimeError("camera index 0 did not deliver frames (still unplugged)")
        super().open()
        self.t_open = time.time()

    def read(self):
        if self.opens <= 2 and time.time() - self.t_open > 1.0:
            self.t_unplug = self.t_unplug or time.time()
            time.sleep(0.03)  # an unplugged camera: reads keep failing
            return False, None, None
        return super().read()


fake = UnpluggedCamera()
sources.make_source = lambda spec, *a, **k: fake  # the worker imports make_source from this module at call time

parent, child = mp.Pipe()
threading.Thread(target=camera_proc._worker, args=(child, "fake", 640, 480, 640, 5), daemon=True).start()

events, t0 = [], time.time()
while time.time() - t0 < 30:  # worker start (imports) ~2-15 s; unplug 1 s after open; stall detected 1.5 s later
    if parent.poll(0.1):
        m = parent.recv()
        events.append((time.time(), m[0], m[1] if m[0] in ("error", "fatal") else None))
parent.send(("quit",))

kinds = [k for _, k, _ in events]
errs = [msg for _, k, msg in events if k in ("error", "fatal")]
t_err = next((t for t, k, _ in events if k == "error"), None)
t_last_err = max((t for t, k, _ in events if k == "error"), default=None)
delay = t_err - fake.t_unplug if t_err and fake.t_unplug else None
resumed = t_last_err is not None and any(k == "preview" and t > t_last_err + 0.5 for t, k, _ in events)
print(f"previews: {kinds.count('preview')}, opens of the fake camera: {fake.opens}")
for e in errs:
    print("  error:", e)
print("expected: stall error 1.5-2.5 s after the unplug; failed reopen reported with backoff; 3rd open succeeds; previews resume")
print(f"actual:   first error after {delay and round(delay, 2)} s, opens={fake.opens}, errors={len(errs)}, previews_resumed={resumed}")
ok = (delay is not None and 1.4 <= delay <= 2.5 and fake.opens == 3 and resumed
      and any("failed" in e for e in errs) and len(errs) <= 4)
print("CHECK_STALL " + ("PASSED" if ok else "FAILED"))
sys.exit(0 if ok else 1)
