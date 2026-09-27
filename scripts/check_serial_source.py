"""Check `--source serial` (sources.SerialSource, the ESP32-CAM over its USB cable with firmware/usb_cam) against
scripts/fake_esp32.py's FakeSerialBoard: firmware/usb_cam's protocol on a pseudo-terminal, paced to the cable's 1.5 Mbaud.

Checks: boot noise before the first message is skipped; settings are commanded and acknowledged; frames arrive at the
sensor's rate with the throughput the cable allows; window=2x keeps 480x320 frames with 2x the mouth pixels (the board's OV3660), and the zoom changes live; bytes lost
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
from silent_running.sources import serial_header

PTY_BAUD = 230400  # a pty accepts only standard rates; the fake paces its output to the real cable's 1.5 Mbaud itself
fails = 0
def check(ok, what):
    global fails
    fails += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {what}")


fl, mp, t_ms = FaceLandmarker(), MouthPixels(VideoProcess(convert_gray=True)), 0.0
def frames(src, seconds, board=None):
    """(fps, sizes, median mouth px) over `seconds` of reads; with the board, also the rate it sent at meanwhile."""
    global t_ms
    t0, n, sizes, px, sent0 = time.time(), 0, set(), [], board.frames if board else 0
    while time.time() - t0 < seconds:
        ok, bgr, _ = src.read()
        if not ok: continue
        n += 1; sizes.add((bgr.shape[1], bgr.shape[0]))
        if n % 4 == 0:
            t_ms += 160
            lm, _ = fl(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), t_ms)
            if lm is not None: px.append(mp(lm))
    out = n / seconds, sizes, (float(np.median(px)) if px else None)
    return out + ((board.frames - sent0) / seconds,) if board else out


print("1. HVGA over the cable")
board = FakeSerialBoard()
src = make_source(f"serial:{board.port}?baud={PTY_BAUD}")
src.open()
fps, sizes, base, sent = frames(src, 4, board)
info = src.info()
check(board.log == ["set framesize 9", "set quality 16"], f"settings commanded and acknowledged: {board.log}")
check(sizes == {(480, 320)}, f"frames {sizes}")
check(fps > 0.95 * sent, f"received {fps:.1f} of the {sent:.1f} frames/s sent, {info['mbps']} Mbit/s, {info['frame_kb']} KB/frame (1.5 Mbaud pace)")
print(f"     mouth {base:.1f} px")

print("2. window=2x")
src.release(); board.log.clear()
src = make_source(f"serial:{board.port}?baud={PTY_BAUD}&window=2x")
src.open()
fps, sizes, zoom = frames(src, 4)
check(board.log == ["set framesize 9", "set quality 16", "sensor", "win 544 450 1535 1097 8 2 2172 719 480 320 0 1"], f"commands {board.log}")
check(sizes == {(480, 320)}, f"frames {sizes}")
check(abs(zoom / base / 2 - 1) < 0.1, f"mouth {zoom:.1f} px = x{zoom / base:.2f} of {base:.1f} (expected x2), and no frame of the old view reached read()")
board.log.clear()
src.set_zoom(1)
_, _, back = frames(src, 2)
check(board.log == ["set framesize 9"] and abs(back / base - 1) < 0.1, f"live 1x: {board.log}, mouth x{back / base:.2f}")
src.release()

print("3. bytes lost mid-frame (every 4th frame cut short)")
board = FakeSerialBoard(drop_every=4)
src = make_source(f"serial:{board.port}?baud={PTY_BAUD}")
src.open()
fps, sizes, _, sent = frames(src, 4, board)
bad = src.info()["bad_frames"]
check(bad > 0 and fps > 0.95 * 0.75 * sent, f"{bad} bad frames dropped, received {fps:.1f} of {sent:.1f} sent/s (3 in 4 are intact)")
src.release()

print("3b. one bit flipped inside every 5th frame (it would still decode, smeared)")
board = FakeSerialBoard(flip_every=5)
src = make_source(f"serial:{board.port}?baud={PTY_BAUD}")
src.open()
fps, _, _, sent = frames(src, 4, board)
bad = src.info()["bad_frames"]
check(bad > 0 and fps > 0.95 * 0.8 * sent, f"{bad} frames rejected by their CRC-32, received {fps:.1f} of {sent:.1f} sent/s (4 in 5 are intact)")
src.release()

print("3c. a header whose length is garbage (4 GB), then normal frames: the parser must not wait for it")
board = FakeSerialBoard()
src = make_source(f"serial:{board.port}?baud={PTY_BAUD}")
src.open()
hdr = bytearray(serial_header(b"J", b"x")); hdr[4:8] = (0xFFFFFFF0).to_bytes(4, "little")
board.send_raw(bytes(hdr))
fps, _, _, sent = frames(src, 3, board)
check(fps > 0.95 * sent, f"received {fps:.1f} of {sent:.1f} sent/s right after it")

print("3d. an unsolicited board message while streaming: reported as the fault, not queued")
board.say("err frame capture failed")
time.sleep(0.5)
check(src.fault() == "err frame capture failed" and src._texts.qsize() == 0 and src.info()["board_msg"] == "err frame capture failed",
      f"fault() {src.fault()!r}, queued texts {src._texts.qsize()}")
src.release()

print("3e. a setting the board refuses: open fails and leaves nothing open")
board = FakeSerialBoard()
src = make_source(f"serial:{board.port}?baud={PTY_BAUD}&brightness=2")
try:
    src.open(); check(False, "open() should have failed")
except RuntimeError as e:
    check("refused" in str(e) and src._ser is None and not src._thread.is_alive(), f"fails loudly and releases the port: {e}")

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
check(board.log[:2] == ["set framesize 9", "set quality 16"] and board.log[2].startswith("win ") and sizes == {(480, 320)},
      f"the reopen pushed everything again: {board.log}, frames {sizes}")
src.release()

print("5b. every frame damaged for 2.5 s (bytes lost on the link): the stall names it, the reopen keeps the port")
board = FakeSerialBoard()
src = make_source(f"serial:{board.port}?baud={PTY_BAUD}&window=2x")
src.open(); frames(src, 1)
ser = src._ser
board.log.clear(); board.damage_until = time.time() + 2.5
t0 = time.time()
while time.time() - t0 < 2 and src.read()[0]:
    pass
time.sleep(0.5)
check("damaged" in (src.fault() or ""), f"stall reason: {src.fault()!r}")
src.reopen()
fps, sizes, _ = frames(src, 2)
check(src._ser is ser and board.log[:2] == ["set framesize 9", "set quality 16"] and board.log[2].startswith("win "),
      f"settings and zoom pushed again on the same open port (no board reset): {board.log}")
check(fps > 10 and sizes == {(480, 320)}, f"frames again after the burst: {fps:.1f} fps {sizes}")
src.release()

print("5c. nothing arrives for 3 s (a hung board or driver): the stall names it, the reopen reopens the port")
board = FakeSerialBoard()
src = make_source(f"serial:{board.port}?baud={PTY_BAUD}")
src.open(); frames(src, 1)
ser = src._ser
board.silent_until = time.time() + 3
t0 = time.time()
while time.time() - t0 < 2.5 and src.read()[0]:
    pass
time.sleep(1.2)
check("no bytes" in (src.fault() or ""), f"stall reason: {src.fault()!r}")
time.sleep(max(board.silent_until - time.time(), 0))
src.reopen()
fps, _, _ = frames(src, 2)
check(src._ser is not ser and fps > 10, f"the port was reopened and frames came back: {fps:.1f} fps")
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
