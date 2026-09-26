"""Pluggable video sources. The camera process only ever talks to a VideoSource, so a laptop webcam, the USB camera
on the glasses clip and a recorded file all drive the exact same live pipeline (preview, auto-listen, decode).

Spec strings (server `--source`):
  webcam            auto: probe indices 0-2, prefer the one that sees a face
  webcam:1          OpenCV camera index 1 (preview is mirrored, selfie-style)
  usb               first USB (UVC) camera per system_profiler (never the FaceTime or an iPhone Continuity Camera)
  usb:1 | usb:Arducam   by index or by (substring of) the AVFoundation device name; preview not mirrored
  file:data/eval/x.mp4[?loop=0&realtime=0&gap=0]   plays at native fps, loops by default (judging backup); see FileSource
  stream:http://172.20.10.2:81/stream[?framesize=9&quality=16]   MJPEG over HTTP (ESP32-CAM on the glasses);
                    stream:172.20.10.2 = that host's :81/stream with framesize=9 (HVGA) and quality=16 applied on every (re)open
                    via the board's /control endpoint. HVGA q16 ~19 fps without dropouts on a phone hotspot; q10 saturated the
                    link (0.5-1 s gaps, 480 ms ping spikes); VGA runs at 6 fps on this board.
"""
import json, math, os, re, sys, time, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class VideoSource:
    """open() raises with the reason if the source can't deliver frames. read() -> (ok, bgr, ts): ts is strictly increasing,
    in wall-clock seconds (time.time() scale) so manual listen start/stop windows line up. `discontinuity` is True for the
    frame just returned if it jumps from the previous one (file loop), so motion trackers can reset."""
    kind = "base"
    mirror = False
    discontinuity = False
    finished = False  # a non-looping file reached its end
    stall_s = 1.5     # seconds without frames before the camera process reports a stall and reopens the source

    def open(self):
        raise NotImplementedError

    def read(self):
        raise NotImplementedError

    def release(self):
        pass

    def reopen(self):
        """Reopen after the source stopped delivering frames (e.g. a camera unplugged and plugged back in)."""
        self.release()
        self.open()

    def info(self):
        return {"kind": self.kind, "mirror": self.mirror, "finished": self.finished}


def _open_cv(index, width, height):
    import cv2
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    for _ in range(8):  # macOS cameras return a few black frames while warming up
        ok, f = cap.read()
        if ok and f is not None and f.mean() > 5:
            return cap
    cap.release()
    return None


class CameraSource(VideoSource):
    """A live OpenCV camera. index=-1 probes 0-2 and keeps the one where `face_fn(rgb)` finds a face most often."""
    kind = "webcam"
    mirror = True

    def __init__(self, index=-1, width=640, height=480, face_fn=None):
        self.index, self.width, self.height, self.face_fn = index, width, height, face_fn
        self.cap = None

    def open(self):
        if self.index >= 0:
            self.cap = _open_cv(self.index, self.width, self.height)
            if self.cap is None:
                raise RuntimeError(f"camera index {self.index} did not deliver frames (unplugged, or camera permission denied?)")
            return
        import cv2
        best, best_score = None, -1
        for idx in (0, 1, 2):
            c = _open_cv(idx, self.width, self.height)
            if c is None:
                continue
            faces = 0
            if self.face_fn is not None:
                for _ in range(6):
                    ok, f = c.read()
                    if ok and self.face_fn(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)):
                        faces += 1
            score = faces * 10 + (1 if abs(c.get(cv2.CAP_PROP_FRAME_WIDTH) / max(c.get(cv2.CAP_PROP_FRAME_HEIGHT), 1) - 4 / 3) < 0.05 else 0)
            print(f"[source] probe index={idx} faces={faces}/6 score={score}")
            if score > best_score:
                if best is not None:
                    best.release()
                best, best_score, self.index = c, score, idx
            else:
                c.release()
        if best is None:
            raise RuntimeError("no camera at index 0-2 delivered frames (camera permission denied?)")
        self.cap = best

    def read(self):
        if self.cap is None:  # released, or a reopen after an unplug failed: no frame yet
            return False, None, time.time()
        ok, bgr = self.cap.read()
        return ok, bgr, time.time()

    def reopen(self):
        """Reopen the same device after it stopped delivering frames (unplug/replug)."""
        self.release()
        CameraSource.open(self)

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def info(self):
        return {**super().info(), "index": self.index}


def uvc_camera_names():
    """Names of USB Video Class cameras, from `system_profiler SPCameraDataType` (macOS). Classifying by hardware type,
    not by name, matters: a Continuity Camera is named after the phone (e.g. "Hriday's Phone Camera"), so a name
    blacklist lets `usb` pick the iPhone. UVC devices report a model id like "UVC Camera VendorID_1133 ProductID_2085"; Apple's own (vendor 1452) are excluded."""
    r = subprocess.run(["system_profiler", "SPCameraDataType", "-json"], capture_output=True, text=True, timeout=15)
    if r.returncode != 0:
        raise RuntimeError(f"system_profiler failed ({r.returncode}): {r.stderr.strip()}")
    models = {c["_name"]: c.get("spcamera_model-id", "") for c in json.loads(r.stdout).get("SPCameraDataType", [])}
    # Intel-Mac FaceTime cameras are UVC too, with Apple's vendor id 1452 (0x05AC)
    return {name for name, model in models.items() if model.startswith("UVC Camera") and "VendorID_1452 " not in model + " "}


def list_devices():
    """[(index, name)] of video devices via ffmpeg/AVFoundation (macOS). Same order as OpenCV's AVFoundation backend."""
    out = subprocess.run(["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
                         capture_output=True, text=True, timeout=10).stderr
    devs, video = [], False
    for line in out.splitlines():
        if "video devices" in line:
            video = True
        elif "audio devices" in line:
            break
        elif video:
            m = re.search(r"\[(\d+)\] (.+)$", line)
            if m:
                devs.append((int(m.group(1)), m.group(2).strip()))
    return devs


class USBSource(CameraSource):
    """The camera on the glasses clip. Faces away from the wearer, so the preview is not mirrored."""
    kind = "usb"
    mirror = False

    def __init__(self, which=None, width=640, height=480):
        self.which, self.name = which, None
        super().__init__(index=0, width=width, height=height)

    def open(self):
        devs = [d for d in list_devices() if not d[1].lower().startswith("capture screen")]
        if str(self.which).isdigit():
            self.index = int(self.which)
            self.name = dict(devs).get(self.index)
        elif self.which is None:
            uvc = uvc_camera_names()
            ext = [d for d in devs if d[1] in uvc]
            if not ext:
                raise RuntimeError(f"no USB (UVC) camera found; devices: {devs}; UVC cameras per system_profiler: {sorted(uvc)}")
            self.index, self.name = ext[0]
        else:
            hit = [d for d in devs if self.which.lower() in d[1].lower()]
            if not hit:
                raise RuntimeError(f"no camera matching {self.which!r} (devices: {devs})")
            self.index, self.name = hit[0]
        print(f"[source] usb -> index={self.index} name={self.name!r}")
        super().open()

    def info(self):
        return {**super().info(), "name": self.name}


class FileSource(VideoSource):
    """Plays a recorded video as if it were a live camera, looping by default.
    realtime=True paces frames at the file's native fps against the wall clock, dropping frames if the consumer falls
    behind (like a real camera). realtime=False reads as fast as the consumer pulls, stamping frames with media time
    (t_play + i/fps). That is for offline auto-listen segmentation tests only: media time runs ahead of the wall clock,
    so manual listen and snapshot windows (which use time.time()) are meaningless in that mode.
    loop_gap > 0 inserts that many seconds of the frozen last frame between plays. A frozen frame has zero motion,
    unlike a real sensor, so the auto-listen noise floor decays toward 0 there; record real pauses for honest tests.
    Frames wider than max_width are downscaled."""
    kind = "file"
    mirror = False

    def __init__(self, path, realtime=True, loop=True, max_width=960, loop_gap=0.0):
        self.path = path if os.path.isabs(path) else os.path.join(ROOT, path)
        if not os.path.isfile(self.path):
            raise FileNotFoundError(f"video file not found: {self.path}")
        self.realtime, self.loop, self.max_width, self.loop_gap = realtime, loop, max_width, loop_gap
        self.cap, self.fps, self.n_frames, self.plays = None, 25.0, 0, 0

    def open(self):
        import cv2
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"OpenCV could not open {self.path}")
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if 1 <= fps <= 240 else 25.0
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._i = 0                  # frames consumed in the current play-through
        self._t0 = time.time()       # time of frame 0 of the current play-through
        self._last = None
        print(f"[source] file {os.path.relpath(self.path, ROOT)} {self.fps:.2f} fps, {self.n_frames} frames, realtime={self.realtime} loop={self.loop}")

    def read(self):
        import cv2
        if self.finished:
            time.sleep(0.02)
            return False, None, None
        if self.realtime:
            slot = math.floor((time.time() - self._t0) * self.fps) + 1  # next frame slot on this play's clock
            if slot < 0:  # inside the loop gap: hold the last frame, stamped on the same clock (before frame 0)
                time.sleep(max(self._t0 + slot / self.fps - time.time(), 0))
                self.discontinuity = False
                return True, self._last, self._t0 + slot / self.fps
            while self._i < slot - 1:  # fell behind: skip frames, as a live camera would
                if not self.cap.grab():
                    break
                self._i += 1
            time.sleep(max(self._t0 + self._i / self.fps - time.time(), 0))
        ok, bgr = self.cap.read()
        if not ok:
            if not self.loop or self._i == 0:
                self.finished = True
                print(f"[source] file finished after {self._i} frames")
                return False, None, None
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._t0 += self._i / self.fps + self.loop_gap
            self._i = 0
            self.plays += 1
            return self.read()
        ts = self._t0 + self._i / self.fps
        self.discontinuity = self._i == 0 and self.plays > 0
        self._i += 1
        if self.max_width and bgr.shape[1] > self.max_width:
            h, w = bgr.shape[:2]
            bgr = cv2.resize(bgr, (self.max_width, int(h * self.max_width / w)), interpolation=cv2.INTER_AREA)
        self._last = bgr
        return True, bgr, ts

    def release(self):
        if self.cap is not None:
            self.cap.release()

    def info(self):
        return {**super().info(), "path": os.path.relpath(self.path, ROOT), "fps": round(self.fps, 2), "plays": self.plays,
                "realtime": self.realtime, "loop": self.loop, "t_play": round(self._t0, 3)}  # time of frame 0 of this play


class StreamSource(VideoSource):
    """MJPEG-over-HTTP camera (ESP32-CAM `/stream`). Parses the multipart stream itself (no ffmpeg buffering) in a reader
    thread that keeps only the newest frame, so a slow consumer never accumulates lag. Reconnects when the stream stalls.
    Frames are stamped at arrival time. The head camera looks at the patient, so the preview is not mirrored."""
    kind = "stream"
    mirror = False
    stall_s = 8.0   # WiFi hiccups of 0.5-3 s are normal on a hotspot; reopening the TCP stream during one turns them into 10 s outages

    DEFAULT_SETTINGS = {"framesize": 9, "quality": 16}

    def __init__(self, url, timeout=5.0):
        url, _, query = url.partition("?")
        if not url.startswith("http"):
            url = f"http://{url}" if ("/" in url or ":" in url) else f"http://{url}:81/stream"   # bare host -> the ESP32 default
        self.settings = dict(self.DEFAULT_SETTINGS) if not url.startswith("http") or ":81/stream" in url else {}
        for kv in query.split("&"):
            if "=" in kv:
                k, v = kv.split("=", 1); self.settings[k] = v
        self.url, self.timeout = url, timeout
        self._frame, self._ts, self._seq, self._got = None, 0.0, 0, 0
        self._stop, self._thread, self._err, self._resp = False, None, None, None
        self.width = self.height = 0
        self.fps_est = 0.0
        self._configured = False

    def _configure(self):
        """ESP32-CAM: push frame size / JPEG quality through /control on the board's :80 server (they reset on reboot)."""
        import urllib.request, urllib.parse
        if not self.settings or self._configured:   # the board keeps settings until it reboots; don't spend 2 requests per reopen
            return
        self._configured = True
        u = urllib.parse.urlsplit(self.url)
        base = f"{u.scheme}://{u.hostname}"
        for k, v in self.settings.items():
            try:
                urllib.request.urlopen(f"{base}/control?var={k}&val={v}", timeout=2).read()
            except Exception as e:
                self._configured = False
                print(f"[source] could not set {k}={v} on {base}: {e}")
        time.sleep(0.3)

    def open(self):
        import urllib.request
        self._stop = False
        self._configure()
        req = urllib.request.Request(self.url, headers={"User-Agent": "silent-running"})
        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout)
        except Exception as e:
            raise RuntimeError(f"cannot open MJPEG stream {self.url}: {e}")
        ctype = resp.headers.get("Content-Type", "")
        if "multipart" not in ctype:
            raise RuntimeError(f"{self.url} is not an MJPEG stream (Content-Type {ctype!r})")
        self._resp = resp
        import threading
        self._thread = threading.Thread(target=self._reader, args=(resp,), daemon=True)
        self._thread.start()
        t0 = time.time()
        while self._frame is None and time.time() - t0 < self.timeout:
            if self._err: raise RuntimeError(self._err)
            time.sleep(0.02)
        if self._frame is None:
            raise RuntimeError(f"no frame from {self.url} within {self.timeout}s")
        print(f"[source] stream {self.url} {self.width}x{self.height}")

    def _reader(self, resp):
        import cv2, numpy as np
        buf = bytearray(); t_last, n = time.time(), 0
        try:
            while not self._stop:
                chunk = resp.read1(65536)   # whatever is available now; read(n) would block until n bytes = ~2 frames
                if not chunk:
                    self._err = "stream ended"; break
                buf += chunk
                while True:  # extract every complete JPEG (SOI..EOI); keep only the last one
                    a = buf.find(b"\xff\xd8")
                    if a < 0:
                        del buf[:]; break
                    b = buf.find(b"\xff\xd9", a + 2)
                    if b < 0:
                        if a > 0: del buf[:a]
                        break
                    jpg = bytes(buf[a:b + 2]); del buf[:b + 2]
                    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        self._frame, self._ts, self._seq = img, time.time(), self._seq + 1
                        self.height, self.width = img.shape[:2]
                        n += 1
                        if self._ts - t_last >= 2.0:
                            self.fps_est, t_last, n = n / (self._ts - t_last), self._ts, 0
        except Exception as e:
            self._err = f"stream read failed: {e}"
        finally:
            try: resp.close()
            except Exception: pass

    def read(self):
        t0 = time.time()
        while self._seq == self._got and time.time() - t0 < 1.0:  # wait for a new frame (like a blocking camera read)
            if self._err and self._thread and not self._thread.is_alive():
                return False, None, None
            time.sleep(0.003)
        if self._seq == self._got:
            return False, None, None
        self._got = self._seq
        return True, self._frame, self._ts

    def release(self):
        """Shut the socket down so the reader thread's blocking recv returns and the board sees the client go away at once.
        The ESP32 stream handler serves one client and only notices a dead socket when its write fails; a half-open old
        connection blocks the new one until then."""
        import socket
        self._stop = True
        resp, self._resp = self._resp, None
        if resp is not None:
            try:
                sock = resp.fp.raw._sock
                sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try: resp.close()
            except Exception: pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def reopen(self):
        self.release()
        self._frame, self._err, self._seq, self._got = None, None, 0, 0
        self.open()

    def info(self):
        return {**super().info(), "url": self.url, "fps": round(self.fps_est, 1), "width": self.width, "height": self.height, "settings": self.settings}


def _flag(v):
    return str(v).lower() not in ("0", "false", "no", "off")


def make_source(spec, width=640, height=480, face_fn=None):
    """Build a VideoSource from a spec string (see module docstring). An int is treated as a webcam index."""
    spec = str(spec if spec is not None else "webcam").strip()
    if spec.lstrip("-").isdigit():
        spec = f"webcam:{spec}"
    import re as _re
    if spec.startswith(("http://", "https://")) or _re.match(r"^\d{1,3}(\.\d{1,3}){3}(:\d+)?(/.*)?$", spec):
        spec = "stream:" + spec   # a bare URL or IP means the network camera
    kind, _, arg = spec.partition(":")
    kind = kind.lower()
    if kind == "webcam":
        return CameraSource(index=int(arg) if arg else -1, width=width, height=height, face_fn=face_fn)
    if kind == "usb":
        return USBSource(which=arg or None, width=width, height=height)
    if kind == "stream":
        return StreamSource(arg)
    if kind == "file":
        path, _, query = arg.partition("?")
        opts = dict(kv.split("=", 1) for kv in query.split("&") if "=" in kv)
        return FileSource(path, realtime=_flag(opts.get("realtime", 1)), loop=_flag(opts.get("loop", 1)),
                          loop_gap=float(opts.get("gap", 0)))
    raise ValueError(f"unknown video source {spec!r}; use webcam[:N] | usb[:N|name] | file:path.mp4 | stream:http://host:81/stream")


if __name__ == "__main__":  # quick check: python -m silent_running.sources file:data/samples/ted1_short.mp4
    if len(sys.argv) < 2:
        print("devices:", list_devices()); sys.exit()
    src = make_source(sys.argv[1])
    src.open()
    t, n, last = time.time(), 0, -1.0
    while time.time() - t < float(sys.argv[2] if len(sys.argv) > 2 else 5):
        ok, f, ts = src.read()
        if not ok:
            if src.finished: break
            continue
        assert ts > last, f"timestamps went backwards: {ts} after {last}"
        n, last = n + 1, ts
    print(f"{n} frames in {time.time()-t:.2f}s -> {n/(time.time()-t):.1f} fps; info={src.info()}")
    src.release()
