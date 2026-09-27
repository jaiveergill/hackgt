"""Check `stream:...?window=2x` (sources.StreamSource + ov2640_window) against a fake ESP32-CAM on this Mac.

Against scripts/fake_esp32.py, which behaves like the CameraWebServer sketch (stale buffered frames after a change, streams
ending on a sensor reconfigure). Its "sensor" sees a face in the centre, so the mouth-pixels readout
(camera_proc.MouthPixels) measures what the zoom gains on the frames actually received.

Checks: the commands and their order (settings first, then the window: a framesize push resets the window); the window
parameters are valid for the driver; frames arrive at the window's size; the mouth gets the expected extra pixels;
after a board reboot a reopen pushes everything again; firmware without /resolution, or one that ignores it, fails
loudly and releases the stream. What it cannot check: that the real board and its firmware accept these parameters
(step 2, on the hardware).

    python scripts/check_stream_window.py
"""
import os, sys, time
import numpy as np, cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from silent_running.sources import make_source
from silent_running.camera_proc import MouthPixels, VideoProcess
from silent_running.expression import FaceLandmarker
from fake_esp32 import FakeBoard

fails = 0
def check(ok, what):
    global fails
    fails += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {what}")


fl, mp, t_ms = FaceLandmarker(), MouthPixels(VideoProcess(convert_gray=True)), 0.0
def mouth(src, n=12):
    """Median mouth width (px) over the next n frames from the source, and their size."""
    global t_ms
    px, size = [], None
    for _ in range(n):
        ok, bgr, _ = src.read()
        if not ok: continue
        size = (bgr.shape[1], bgr.shape[0]); t_ms += 50
        lm, _ = fl(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), t_ms)
        if lm is not None: px.append(mp(lm))
    return (float(np.median(px)) if px else None), size


def open_source(board, port, query):
    src = make_source(f"stream:http://127.0.0.1:{port}/stream?{query}")
    src.board = f"http://127.0.0.1:{port}"  # the real board's control server is :80, which needs root to fake here
    board.log.clear()
    src.open()
    return src


def open_fails(board, port, query, expect):
    try:
        open_source(board, port, query).release(); check(False, f"{query}: open() should have failed")
    except (RuntimeError, ValueError) as e:
        check(expect in str(e), f"{query}: fails loudly: {e}")


print("1. no window (HVGA, as today)")
board = FakeBoard(); port = board.serve()
src = open_source(board, port, "framesize=9&quality=16")
base, size = mouth(src)
check(size == (480, 320), f"frames {size}, expected (480, 320)")
check([p for p, _ in board.log] == ["/control", "/control"], f"commands {board.log}")
print(f"     mouth {base:.1f} px")
src.release()

print("2. window=2x")
src = open_source(board, port, "framesize=9&quality=16&window=2x")
zoom, size = mouth(src)
res = [q for p, q in board.log if p == "/resolution"]
check([p for p, _ in board.log] == ["/control", "/control", "/resolution"], f"order: settings then window {[p for p, _ in board.log]}")
check(res and res[-1] == {"sx": "1", "offx": "200", "offy": "168", "tx": "400", "ty": "264", "ox": "400", "oy": "264"}, f"window {res}")
check(size == (400, 264), f"frames {size}, expected (400, 264), and no frame of the old view reached read()")
check(abs(zoom / base / (792 / 480) - 1) < 0.1, f"mouth {zoom:.1f} px = x{zoom / base:.2f} of {base:.1f} (expected x{792 / 480:.2f}: raw pixels vs HVGA)")
print(f"     info: {src.info()['window']}")

print("3. board reboots (settings and window lost), the stall recovery reopens")
board.reboot()
time.sleep(0.2)
board.log.clear()
src.reopen()
_, size = mouth(src, 4)
check([p for p, _ in board.log] == ["/control", "/control", "/resolution"], f"everything pushed again: {[p for p, _ in board.log]}")
check(size == (400, 264), f"frames {size} after reopen")
src.release()

for fw, expect in (("old", "flash the current CameraWebServer"), ("ignores", "did not apply the sensor window")):
    print(f"4. firmware that {'has no /resolution' if fw == 'old' else 'ignores /resolution'}")
    board = FakeBoard(fw); port = board.serve()
    src = make_source(f"stream:http://127.0.0.1:{port}/stream?framesize=9&quality=16&window=2x")
    src.board = f"http://127.0.0.1:{port}"
    try:
        src.open(); check(False, "open() should have failed")
    except RuntimeError as e:
        check(expect in str(e), f"fails loudly: {e}")
    check(src._thread is None or not src._thread.is_alive(), "stream released after the failure")

print("5. zooms that would only be scaled, and frame sizes outside the sensor's SVGA mode, are refused")
board = FakeBoard(); port = board.serve()
open_fails(board, port, "framesize=9&window=1.6x", "use more than 1.67x")
open_fails(board, port, "framesize=6&window=2x", "needs framesize HVGA")
open_fails(board, port, "window=0.5x", "zoom above 1x")
src = make_source("stream:http://cam.example:8080/video?token=abc")
check(src.url == "http://cam.example:8080/video?token=abc" and not src.settings, f"another camera's URL is used as given: {src.url} {src.settings}")

print("CHECK PASSED" if not fails else f"CHECK FAILED ({fails})")
sys.exit(1 if fails else 0)
