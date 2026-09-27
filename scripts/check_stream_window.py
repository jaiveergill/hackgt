"""Check `stream:...?window=2x` (sources.StreamSource + ov3660_window) against a fake ESP32-CAM on this Mac.

Against scripts/fake_esp32.py, which behaves like the CameraWebServer sketch (stale buffered frames after a change, streams
ending on a sensor reconfigure). Its "sensor" sees a face in the centre, so the mouth-pixels readout
(camera_proc.MouthPixels) measures what the zoom gains on the frames actually received.

Checks: the commands and their order (settings first, then the window: a framesize push resets the window); the window
parameters are the OV3660's (the board's sensor) and the ones measured on the board; the mouth gets 2x the pixels; after
a board reboot a reopen pushes everything again; the zoom changes live both ways, and a zoom the sensor can't do is refused
before the board is touched; firmware without /resolution fails loudly and releases the stream. What it cannot check:
the real board (its 2x window was measured on the board: 30.7 fps, frames the 1x view's centre at twice the size).

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

W2X = {"sx": "544", "sy": "450", "ex": "1535", "ey": "1097", "offx": "8", "offy": "2", "tx": "2172", "ty": "719", "ox": "480", "oy": "320",
       "scale": "0", "binning": "1"}  # measured on the board: 30.7 fps

print("2. window=2x")
src = open_source(board, port, "framesize=9&quality=16&window=2x")
zoom, size = mouth(src)
res = [q for p, q in board.log if p == "/resolution"]
check([p for p, _ in board.log] == ["/control", "/control", "/greg", "/greg", "/resolution"], f"order: settings, the sensor's PID, then the window {[p for p, _ in board.log]}")
check(res and res[-1] == W2X, f"window {res}")
check(size == (480, 320), f"frames {size}, expected (480, 320): the zoom keeps the frame size")
check(abs(zoom / base / 2 - 1) < 0.1, f"mouth {zoom:.1f} px = x{zoom / base:.2f} of {base:.1f} (expected x2: binned pixels 1:1 instead of scaled by 1/2)")
print(f"     info: {src.info()['window']}")

print("3. board reboots (settings and window lost), the stall recovery reopens")
board.reboot()
time.sleep(0.2)
board.log.clear()
src.reopen()
_, size = mouth(src, 4)
check([p for p, _ in board.log] == ["/control", "/control", "/resolution"], f"everything pushed again (the sensor is known): {[p for p, _ in board.log]}")
check(size == (480, 320), f"frames {size} after reopen")
src.release()

print("4. live zoom: 1x -> 2x -> 1x on an open stream; a zoom the sensor can't do is refused before the board is touched")
src = open_source(board, port, "framesize=9&quality=16")
board.log.clear()
src.set_zoom(2)
z2, _ = mouth(src)
check([p for p, _ in board.log] == ["/greg", "/greg", "/resolution"] and src.info()["zoom"] == 2, f"2x: {[p for p, _ in board.log]}, info zoom {src.info()['zoom']}")
check(abs(z2 / base / 2 - 1) < 0.1, f"mouth x{z2 / base:.2f} at 2x")
board.log.clear()
src.set_zoom(1)
z1, _ = mouth(src)
check(board.log == [("/control", {"var": "framesize", "val": "9"})] and src.info()["zoom"] == 1, f"1x pushes the framesize (the whole view): {board.log}")
check(abs(z1 / base - 1) < 0.1, f"mouth x{z1 / base:.2f} back at 1x")
board.log.clear()
try:
    src.set_zoom(3); check(False, "3x should be refused")
except ValueError as e:
    check(not board.log and "scaled up" in str(e), f"3x refused, board untouched: {e}")
src.release()

print("5. firmware without /resolution")
board = FakeBoard("old"); port = board.serve()
src = make_source(f"stream:http://127.0.0.1:{port}/stream?framesize=9&quality=16&window=2x")
src.board = f"http://127.0.0.1:{port}"
try:
    src.open(); check(False, "open() should have failed")
except RuntimeError as e:
    check("flash the current CameraWebServer" in str(e), f"fails loudly: {e}")
check(src._thread is None or not src._thread.is_alive(), "stream released after the failure")

print("6. zooms that would only be scaled up, and frame sizes the sensor can't zoom that far, are refused")
board = FakeBoard(); port = board.serve()
open_fails(board, port, "framesize=9&window=2.5x", "gives up to 2x")
open_fails(board, port, "framesize=10&window=2x", "gives up to 1.6x")
open_fails(board, port, "window=0.5x", "zoom above 1x")
src = make_source("stream:http://cam.example:8080/video?token=abc")
check(src.url == "http://cam.example:8080/video?token=abc" and not src.settings, f"another camera's URL is used as given: {src.url} {src.settings}")

print("CHECK PASSED" if not fails else f"CHECK FAILED ({fails})")
sys.exit(1 if fails else 0)
