"""A fake ESP32-CAM on this Mac: the CameraWebServer sketch over HTTP (FakeBoard) and firmware/usb_cam over a pseudo-terminal
(FakeSerialBoard), for scripts/check_stream_window.py, scripts/check_serial_source.py and scripts/bench_link.py.

Its sensor is an OV3660, like the real board's. It serves /control (framesize, quality), /resolution (the sensor window:
esp32-camera's ov3660 set_res_raw, i.e. an array window, 2x2 binning, the ISP offset, then the scaler or a crop to the output
size; a framesize applies the driver's ratio_table view, as set_framesize does), /greg (its PID registers) and the MJPEG
/stream. Like the real board it keeps sending BUFFERED frames of the old view after a change, a sensor reconfigure ends the
streams that are open, and each frame is the newest the sensor has when the previous send finishes (CAMERA_GRAB_LATEST). Its "sensor" sees a face (a TED frame) in the centre.
link(t) -> Mbit/s makes every send block like a TCP socket on a WiFi link of that capacity (None: unlimited). JPEG quality
is not modelled: frames are encoded at one fixed quality, so only sizes relative to each other are meaningful.
"""
import os, sys, threading, time, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from silent_running.sources import OV3660_VIEWS, serial_header  # noqa: E402  (the header firmware/usb_cam writes)
SENSOR_FPS = 25.0
JPEG_QUALITY = 60
FRAMESIZES = {6: (320, 240), 9: (480, 320), 10: (640, 480)}  # esp32-camera sensor.h (current enum)
WINDOW_KEYS = ("sx", "sy", "ex", "ey", "offx", "offy", "tx", "ty", "ox", "oy", "scale", "binning")  # set_res_raw's arguments
BOOT_FRAMESIZE = 6  # the CameraWebServer sketch drops to QVGA at boot
BOUNDARY = "123456789000000000000987654321"


class FakeBoard:
    def __init__(self, firmware="current", link=None):
        self.firmware, self.log, self.gen, self.link = firmware, [], 0, link
        cap = cv2.VideoCapture(os.path.join(ROOT, "data", "samples", "ted_sample.mp4"))
        ok, im = cap.read()
        cx, cy = 470, 170  # the speaker's face in this frame
        pad = cv2.copyMakeBorder(im, 240, 240, 320, 320, cv2.BORDER_REPLICATE)
        self.array = cv2.resize(pad[cy:cy + 480, cx:cx + 640], (2080, 1548), interpolation=cv2.INTER_CUBIC)  # face centred
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

    def framesize_window(self):
        """What ov3660 set_framesize writes for the current frame size."""
        w, h = FRAMESIZES[self.framesize]
        mw, mh, sx, sy, ex, ey, tx, ty = next(v for v in OV3660_VIEWS if v[0] * h == v[1] * w)
        binning = w <= mw // 2 and h <= mh // 2
        return {"sx": sx, "sy": sy, "ex": ex, "ey": ey, "offx": 8 if binning else 16, "offy": 2 if binning else 6, "tx": tx,
                "ty": ty // 2 + 1 if binning else ty, "ox": w, "oy": h, "scale": int((w, h) not in ((mw, mh), (mw // 2, mh // 2))), "binning": int(binning)}

    def isp_input(self, w):
        """The array window, binned if asked, minus the ISP offset on each side: what the scaler (or the crop) gets."""
        a = self.array[w["sy"]:w["ey"] + 1, w["sx"]:w["ex"] + 1]
        if w["binning"]:
            a = cv2.resize(a, (a.shape[1] // 2, a.shape[0] // 2), interpolation=cv2.INTER_AREA)
        return a[w["offy"]:a.shape[0] - w["offy"], w["offx"]:a.shape[1] - w["offx"]]

    def frame(self):
        w = self.window or self.framesize_window()
        a = self.isp_input(w)
        if w["scale"]:
            return cv2.resize(a, (w["ox"], w["oy"]), interpolation=cv2.INTER_AREA)
        return a[:w["oy"], :w["ox"]]

    def control(self, path, q):
        """-> (HTTP status, body)"""
        self.log.append((path, q))
        if path == "/control":
            if q["var"] == "framesize":
                self.reconfigure(framesize=int(q["val"]), window=None)  # set_framesize rewrites the window: a zoom is lost
            return 200, b""
        if path == "/greg":
            return 200, str({0x300A: 0x36, 0x300B: 0x60}.get(int(q["reg"]), 0) & int(q["mask"])).encode()
        if path == "/resolution":
            if self.firmware == "old":
                return 404, b""
            w = {k: int(q.get(k, 0)) for k in WINDOW_KEYS}
            ok = 0 <= w["sx"] < w["ex"] < self.array.shape[1] and 0 <= w["sy"] < w["ey"] < self.array.shape[0] and w["ox"] % 16 == 0 and w["oy"] % 8 == 0
            if ok:
                a = self.isp_input(w)
                ok = a.shape[1] >= w["ox"] and a.shape[0] >= w["oy"] if not w["scale"] else a.size > 0  # a crop needs the pixels
            if not ok:
                return 500, b""
            self.reconfigure(window=w)
            return 200, b""
        return 404, b""

    def serve(self):
        board = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a): pass

            def do_GET(self):
                u = urllib.parse.urlsplit(self.path)
                if u.path != "/stream":
                    code, body = board.control(u.path, dict(urllib.parse.parse_qsl(u.query)))
                    self.send_response(code); self.send_header("Content-Length", str(len(body))); self.end_headers()
                    self.wfile.write(body)
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
    sketch plugged in: boot and log text only, never a frame. reset() is a brownout: settings lost, boot noise, "ready".
    damage_until = time.time() + s damages every frame for s seconds (a burst of lost bytes on the USB-serial link);
    silent_until = time.time() + s sends and hears nothing for s seconds (a hung board or USB-serial driver)."""
    BOOT = b"ets Jul 29 2019 12:21:46\r\nrst:0x1 (POWERON_RESET),boot:0x13 (SPI_FAST_FLASH_BOOT)\r\nload:0x3fff0030,len:1344\r\n"

    def __init__(self, firmware="usb_cam", baud=1_500_000, drop_every=0, flip_every=0):
        self.board, self.firmware, self.baud, self.drop_every, self.flip_every = FakeBoard(), firmware, baud, drop_every, flip_every
        import tty
        self.master, slave = os.openpty()
        tty.setraw(slave)  # a UART has no line discipline: no echo of the board's output back to it, no line editing
        self.port = os.ttyname(slave)
        self.log, self.frames, self._reset, self.damage_until, self.silent_until = [], 0, False, 0.0, 0.0
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
        if words == ["sensor"]:
            self._msg(b"T", b"ok sensor 3660")
            return
        if words[:1] == ["set"] and len(words) == 3 and words[1] in ("framesize", "quality"):  # all firmware/usb_cam knows
            code, _ = self.board.control("/control", {"var": words[1], "val": words[2]})
        elif words[:1] == ["win"] and len(words) == 13:
            code, _ = self.board.control("/resolution", dict(zip(WINDOW_KEYS, words[1:])))
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
            if time.time() < self.silent_until:
                time.sleep(0.05); pending = b""; continue  # sends nothing and hears nothing
            jpg = cv2.imencode(".jpg", self.board.next_frame(), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])[1].tobytes()
            self.frames += 1
            if self.flip_every and self.frames % self.flip_every == 0:
                bad = bytearray(jpg); bad[len(bad) // 2] ^= 0x10
                self._write(serial_header(b"J", jpg) + bytes(bad))  # sent intact by the board, one bit flipped on the way
            elif (self.drop_every and self.frames % self.drop_every == 0) or time.time() < self.damage_until:
                self._write(serial_header(b"J", jpg) + jpg[:len(jpg) // 3])  # the rest of this frame is lost
            else:
                self._msg(b"J", jpg)
            time.sleep(max(t_frame + 1 / SENSOR_FPS - time.time(), 0))
