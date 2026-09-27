"""A fake ESP32-CAM (CameraWebServer sketch) on this Mac, for scripts/check_stream_window.py and scripts/bench_link.py.

It serves /control (framesize, quality), /resolution (the sensor window, applied like esp32-camera's ov2640 set_window: SVGA
mode reads the 1600x1200 array binned to 800x600, the window is cut from that and scaled to the output size; frame sizes use
the driver's ratio_table view) and the MJPEG /stream. Like the real board it keeps sending BUFFERED frames of the old view
after a change, a sensor reconfigure ends the streams that are open, and each frame is the newest the sensor has when the
previous send finishes (CAMERA_GRAB_LATEST). Its "sensor" sees a face (a TED frame) in the centre.
link(t) -> Mbit/s makes every send block like a TCP socket on a WiFi link of that capacity (None: unlimited). JPEG quality
is not modelled: frames are encoded at one fixed quality, so only sizes relative to each other are meaningful.
"""
import os, threading, time, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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


