"""A fake ESP32-CAM on this Mac: the CameraWebServer sketch over HTTP (FakeBoard) and firmware/usb_cam over a pseudo-terminal
(FakeSerialBoard), for scripts/check_stream_window.py, scripts/check_serial_source.py and scripts/bench_link.py.

It serves /control (framesize, quality), /resolution (the sensor window, applied like esp32-camera's ov2640 set_window: SVGA
mode reads the 1600x1200 array binned to 800x600, the window is cut from that and scaled to the output size; frame sizes use
the driver's ratio_table view) and the MJPEG /stream. Like the real board it keeps sending BUFFERED frames of the old view
after a change, a sensor reconfigure ends the streams that are open, and each frame is the newest the sensor has when the
previous send finishes (CAMERA_GRAB_LATEST). Its "sensor" sees a face (a TED frame) in the centre.
link(t) -> Mbit/s makes every send block like a TCP socket on a WiFi link of that capacity (None: unlimited). JPEG quality
is not modelled: frames are encoded at one fixed quality, so only sizes relative to each other are meaningful.
"""
import os, sys, threading, time, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from silent_running.sources import serial_header  # noqa: E402  (the same header firmware/usb_cam writes)
SENSOR_FPS = 25.0
JPEG_QUALITY = 60
FRAMESIZES = {6: (320, 240), 9: (480, 320), 10: (640, 480)}  # esp32-camera sensor.h (current enum)
VIEW = {(320, 240): (0, 0, 800, 600), (480, 320): (4, 36, 792, 528), (640, 480): (0, 0, 800, 600)}  # ratio_table / 2 (SVGA mode)
BOOT_FRAMESIZE = 6  # the CameraWebServer sketch drops to QVGA at boot
BOUNDARY = "123456789000000000000987654321"


class FakeBoard:
    def __init__(self, firmware="current", link=None):
        self.firmware, self.log, self.gen, self.link = firmware, [], 0, link
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
                gen, t0 = board.gen, time.time()
                try:
                    while gen == board.gen:
                        t_frame = time.time()
                        jpg = cv2.imencode(".jpg", board.next_frame(), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])[1].tobytes()
                        part = f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpg)}\r\n\r\n".encode() + jpg + b"\r\n"
                        for k in range(0, len(part), 1460):  # one TCP segment at a time, blocked for as long as the link needs
                            chunk = part[k:k + 1460]
                            if board.link:
                                time.sleep(len(chunk) * 8 / (board.link(time.time() - t0) * 1e6))
                            self.wfile.write(chunk)
                        time.sleep(max(t_frame + 1 / SENSOR_FPS - time.time(), 0))  # the sensor's own frame rate
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self.httpd.server_address[1]




class FakeSerialBoard:
    """firmware/usb_cam on a pseudo-terminal (`port` is its path): boot noise, then "SRF" messages; commands applied to a
    FakeBoard's sensor state. Output is paced to `baud` (the real cable's speed; a pty ignores the baud rate it is opened
    with). drop_every=N loses the middle of every Nth frame, as a full buffer would; flip_every=N flips one bit inside it. firmware="wifi" is the CameraWebServer
    sketch plugged in: boot and log text only, never a frame. reset() is a brownout: settings lost, boot noise, "ready"."""
    BOOT = b"ets Jul 29 2019 12:21:46\r\nrst:0x1 (POWERON_RESET),boot:0x13 (SPI_FAST_FLASH_BOOT)\r\nload:0x3fff0030,len:1344\r\n"

    def __init__(self, firmware="usb_cam", baud=1_500_000, drop_every=0, flip_every=0):
        self.board, self.firmware, self.baud, self.drop_every, self.flip_every = FakeBoard(), firmware, baud, drop_every, flip_every
        import tty
        self.master, slave = os.openpty()
        tty.setraw(slave)  # a UART has no line discipline: no echo of the board's output back to it, no line editing
        self.port = os.ttyname(slave)
        self.log, self.frames, self._reset = [], 0, False
        threading.Thread(target=self._run, daemon=True).start()

    def reset(self):
        self._reset = True

    def say(self, text):
        """An unsolicited message, as the firmware sends "err frame capture failed" while streaming."""
        self._msg(b"T", text.encode())

    def send_raw(self, data):
        self._write(data)

    def _write(self, data):
        time.sleep(len(data) * 10 / self.baud)  # 8N1: 10 bits per byte
        os.write(self.master, data)

    def _msg(self, kind, payload):
        self._write(serial_header(kind, payload) + payload)

    def _command(self, line):
        self.log.append(line)
        words = line.split()
        if words[:1] == ["set"] and len(words) == 3 and words[1] in ("framesize", "quality"):  # all firmware/usb_cam knows
            code = self.board.control("/control", {"var": words[1], "val": words[2]})
        elif words[:1] == ["win"] and len(words) == 8:
            code = self.board.control("/resolution", dict(zip(("sx", "offx", "offy", "tx", "ty", "ox", "oy"), words[1:])))
        else:
            code = 400
        if code == 200:
            self.board.stale = []  # firmware/usb_cam discards the frames buffered before the change, then answers "ok"
        self._msg(b"T", (("ok " if code == 200 else "err ") + line).encode())

    def _run(self):
        import select
        self._write(self.BOOT)
        if self.firmware == "wifi":
            while True:
                self._write(b"WiFi connecting...\r\n"); time.sleep(0.5)
        self._msg(b"T", b"ready")
        pending = b""
        while True:
            t_frame = time.time()
            if self._reset:
                self._reset = False
                self.board.reboot(); self._write(self.BOOT); self._msg(b"T", b"ready")
            if select.select([self.master], [], [], 0)[0]:
                pending += os.read(self.master, 1024)
            while b"\n" in pending:
                line, pending = pending.split(b"\n", 1)
                self._command(line.decode().strip())
            jpg = cv2.imencode(".jpg", self.board.next_frame(), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])[1].tobytes()
            self.frames += 1
            if self.flip_every and self.frames % self.flip_every == 0:
                bad = bytearray(jpg); bad[len(bad) // 2] ^= 0x10
                self._write(serial_header(b"J", jpg) + bytes(bad))  # sent intact by the board, one bit flipped on the way
            elif self.drop_every and self.frames % self.drop_every == 0:
                self._write(serial_header(b"J", jpg) + jpg[:len(jpg) // 3])  # the rest of this frame is lost
            else:
                self._msg(b"J", jpg)
            time.sleep(max(t_frame + 1 / SENSOR_FPS - time.time(), 0))
