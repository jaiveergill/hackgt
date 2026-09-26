"""Check `stream:...?window=2x` (sources.StreamSource + ov2640_window) against a fake ESP32-CAM on this Mac.

The fake board serves the CameraWebServer endpoints our side uses: /control (framesize, quality) and /resolution (the
sensor window, applied like esp32-camera's ov2640 set_window: SVGA mode reads the 1600x1200 array binned to 800x600, the
window is cut from that and scaled to the output size; frame sizes use the driver's ratio_table view), plus the
:81-style MJPEG /stream. Like the real board it keeps sending BUFFERED frames of the old view after a change, and a sensor
reconfigure ends the streams that are open. Its "sensor" sees a face (a TED frame) in the centre, so the mouth-pixels
readout (camera_proc.MouthPixels) measures what the zoom gains on the frames actually received.

Checks: the commands and their order (settings first, then the window: a framesize push resets the window); the window
parameters are valid for the driver; frames arrive at the window's size; the mouth gets the expected extra pixels;
after a board reboot a reopen pushes everything again; firmware without /resolution, or one that ignores it, fails
loudly and releases the stream. What it cannot check: that the real board and its firmware accept these parameters
(step 2, on the hardware).

    python scripts/check_stream_window.py
"""
import os, sys, threading, time, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np, cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from silent_running.sources import make_source
from silent_running.camera_proc import MouthPixels, VideoProcess
from silent_running.expression import FaceLandmarker

FRAMESIZES = {6: (320, 240), 9: (480, 320), 10: (640, 480)}  # esp32-camera sensor.h (current enum)
VIEW = {(320, 240): (0, 0, 800, 600), (480, 320): (4, 36, 792, 528), (640, 480): (0, 0, 800, 600)}  # ratio_table / 2 (SVGA mode)
BOOT_FRAMESIZE = 6  # the CameraWebServer sketch drops to QVGA at boot
BOUNDARY = "123456789000000000000987654321"


class FakeBoard:
    def __init__(self, firmware="current"):
        self.firmware, self.log, self.gen = firmware, [], 0
        cap = cv2.VideoCapture(os.path.join(ROOT, "data", "samples", "ted_sample.mp4"))
        ok, im = cap.read()
        cx, cy = 470, 170  # the speaker's face in this frame
        pad = cv2.copyMakeBorder(im, 240, 240, 320, 320, cv2.BORDER_REPLICATE)
        self.sensor = cv2.resize(pad[cy:cy + 480, cx:cx + 640], (1600, 1200), interpolation=cv2.INTER_CUBIC)  # face centred
        self.svga = cv2.resize(self.sensor, (800, 600), interpolation=cv2.INTER_AREA)  # SVGA mode: 2x2 binned
        self.reboot()

    def reboot(self):
        self.framesize, self.window, self.gen, self.stale = BOOT_FRAMESIZE, None, self.gen + 1, []  # open streams end, settings reset

    def reconfigure(self, **state):
        """A sensor change: the frames already buffered still show the old view, and open streams end (fb_get fails)."""
        self.stale = [self.frame() for _ in range(2)]
        self.__dict__.update(state)
        self.gen += 1

    def next_frame(self):
        return self.stale.pop(0) if self.stale else self.frame()

    def frame(self):
        if self.window:
            w = self.window
            return cv2.resize(self.svga[w["offy"]:w["offy"] + w["ty"], w["offx"]:w["offx"] + w["tx"]], (w["ox"], w["oy"]), interpolation=cv2.INTER_AREA)
        ow, oh = FRAMESIZES[self.framesize]
        x, y, vw, vh = VIEW[(ow, oh)]
        return cv2.resize(self.svga[y:y + vh, x:x + vw], (ow, oh), interpolation=cv2.INTER_AREA)

    def control(self, path, q):
        self.log.append((path, q))
        if path == "/control":
            if q["var"] == "framesize":
                self.reconfigure(framesize=int(q["val"]), window=None)  # set_framesize rewrites the window: a zoom is lost
            return 200
        if path == "/resolution":
            if self.firmware == "old":
                return 404
            w = {k: int(q.get(k, 0)) for k in ("sx", "offx", "offy", "tx", "ty", "ox", "oy")}
            ok = (w["sx"] == 1 and w["offx"] + w["tx"] <= 800 and w["offy"] + w["ty"] <= 600 and 0 < w["ox"] <= w["tx"] and 0 < w["oy"] <= w["ty"]
                  and all(w[k] % 4 == 0 for k in ("tx", "ty")) and w["ox"] % 16 == 0 and w["oy"] % 8 == 0)  # sizes / 4; JPEG blocks
            if not ok:
                return 500
            if self.firmware != "ignores":
                self.reconfigure(window=w)
            return 200
        return 404

    def serve(self):
        board = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a): pass

            def do_GET(self):
                u = urllib.parse.urlsplit(self.path)
                if u.path != "/stream":
                    code = board.control(u.path, dict(urllib.parse.parse_qsl(u.query)))
                    self.send_response(code); self.send_header("Content-Length", "0"); self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", f"multipart/x-mixed-replace;boundary={BOUNDARY}")
                self.end_headers()
                gen = board.gen
                try:
                    while gen == board.gen:
                        jpg = cv2.imencode(".jpg", board.next_frame(), [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
                        self.wfile.write(f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpg)}\r\n\r\n".encode() + jpg + b"\r\n")
                        time.sleep(0.05)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self.httpd.server_address[1]


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
