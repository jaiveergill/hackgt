"""Check `--source serial` (sources.SerialSource, the ESP32-CAM over its USB cable with firmware/usb_cam) against
scripts/fake_esp32.py's FakeSerialBoard: firmware/usb_cam's protocol on a pseudo-terminal, paced to the cable's 1.5 Mbaud.

Checks: boot noise before the first message is skipped; settings are commanded and acknowledged; frames arrive at the
sensor's rate with the throughput the cable allows; window=2x gives 400x264 frames and ~1.65x the mouth pixels; bytes lost
mid-frame, or a bit flipped inside one (CRC-32), cost only that frame (the parser resyncs); a board that resets mid-stream ends the stream and the reopen restores
its settings; a board running the WiFi sketch, and a missing port, fail loudly; an unplugged cable ends the stream visibly.
What it cannot check: the real board (scripts/bench_link.py serial measures that).

    python scripts/check_serial_source.py
"""
import os, sys, time
import numpy as np, cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from silent_running.sources import make_source, serial_ports
from silent_running.camera_proc import MouthPixels, VideoProcess
from silent_running.expression import FaceLandmarker
from fake_esp32 import FakeSerialBoard

PTY_BAUD = 230400  # a pty accepts only standard rates; the fake paces its output to the real cable's 1.5 Mbaud itself
fails = 0
def check(ok, what):
    global fails
    fails += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {what}")


fl, mp, t_ms = FaceLandmarker(), MouthPixels(VideoProcess(convert_gray=True)), 0.0
def frames(src, seconds):
    """(fps, sizes, median mouth px) over `seconds` of reads."""
    global t_ms
    t0, n, sizes, px = time.time(), 0, set(), []
    while time.time() - t0 < seconds:
        ok, bgr, _ = src.read()
        if not ok: continue
        n += 1; sizes.add((bgr.shape[1], bgr.shape[0]))
        if n % 4 == 0:
            t_ms += 160
            lm, _ = fl(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), t_ms)
            if lm is not None: px.append(mp(lm))
    return n / seconds, sizes, (float(np.median(px)) if px else None)


print("1. HVGA over the cable")
board = FakeSerialBoard()
src = make_source(f"serial:{board.port}?baud={PTY_BAUD}")
src.open()
fps, sizes, base = frames(src, 4)
info = src.info()
check(board.log == ["set framesize 9", "set quality 16"], f"settings commanded and acknowledged: {board.log}")
check(sizes == {(480, 320)}, f"frames {sizes}")
check(fps > 20, f"{fps:.1f} fps, {info['mbps']} Mbit/s, {info['frame_kb']} KB/frame (sensor 25 fps; the cable carries 1.2 Mbit/s)")
print(f"     mouth {base:.1f} px")

print("2. window=2x")
src.release(); board.log.clear()
src = make_source(f"serial:{board.port}?baud={PTY_BAUD}&window=2x")
src.open()
fps, sizes, zoom = frames(src, 4)
check(board.log[:2] == ["set framesize 9", "set quality 16"] and board.log[2].startswith("win 1 200 168 400 264 400 264"), f"commands {board.log}")
check(sizes == {(400, 264)}, f"frames {sizes}, and no frame of the old view reached read()")
check(abs(zoom / base / (792 / 480) - 1) < 0.1, f"mouth {zoom:.1f} px = x{zoom / base:.2f} of {base:.1f} (expected x{792 / 480:.2f})")
src.release()

print("3. bytes lost mid-frame (every 4th frame cut short)")
board = FakeSerialBoard(drop_every=4)
src = make_source(f"serial:{board.port}?baud={PTY_BAUD}")
src.open()
fps, sizes, _ = frames(src, 4)
bad = src.info()["bad_frames"]
check(bad > 0 and fps > 14, f"{bad} bad frames dropped, stream kept going at {fps:.1f} fps (about 3 in 4 of 25 fps)")
src.release()

print("3b. one bit flipped inside every 5th frame (it would still decode, smeared)")
board = FakeSerialBoard(flip_every=5)
src = make_source(f"serial:{board.port}?baud={PTY_BAUD}")
src.open()
fps, _, _ = frames(src, 4)
bad = src.info()["bad_frames"]
check(bad > 0 and fps > 16, f"{bad} frames rejected by their CRC-32, stream kept going at {fps:.1f} fps (about 4 in 5 of 25 fps)")
src.release()

print("4. the WiFi sketch on the board, a missing port")
board = FakeSerialBoard(firmware="wifi")
src = make_source(f"serial:{board.port}?baud={PTY_BAUD}")
try:
    src.open(); check(False, "open() should have failed")
except RuntimeError as e:
    check("flash firmware/usb_cam" in str(e), f"fails loudly: {e}")
if not serial_ports():
    try:
        make_source("serial").open(); check(False, "open() should have failed")
    except RuntimeError as e:
        check("no USB-serial ports" in str(e), f"fails loudly: {e}")

print("5. the board resets mid-stream (brownout): settings and window lost")
board = FakeSerialBoard()
src = make_source(f"serial:{board.port}?baud={PTY_BAUD}&window=2x")
src.open(); frames(src, 1)
board.log.clear(); board.reset()
t0 = time.time()
while time.time() - t0 < 3 and src.read()[0]:
    pass
check(not src.read()[0] and "reset" in (src._err or ""), f"the stream ends with the reason, for the stall recovery: {src._err!r}")
src.reopen()
_, sizes, _ = frames(src, 2)
check(board.log[:2] == ["set framesize 9", "set quality 16"] and board.log[2].startswith("win ") and sizes == {(400, 264)},
      f"the reopen pushed everything again: {board.log}, frames {sizes}")
src.release()

print("6. cable unplugged")
board = FakeSerialBoard()
src = make_source(f"serial:{board.port}?baud={PTY_BAUD}")
src.open()
os.close(board.master)
t0 = time.time()
while time.time() - t0 < 3 and src.read()[0]:
    pass
ok, _, _ = src.read()
check(not ok and src._err, f"reads stop and the reason is kept for the stall report: {src._err!r}")
src.release()

print("CHECK PASSED" if not fails else f"CHECK FAILED ({fails})")
sys.exit(1 if fails else 0)
