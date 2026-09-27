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
  stream:172.20.10.2?window=2x   the same, the sensor reading only the centre half of its view: 2x the pixels across the
                    mouth at the same frame rate (the board's sensor is an OV3660: see ov3660_window)
  serial[:/dev/cu.usbserial-XXXX][?baud=1500000&window=2x]   the ESP32-CAM over its USB cable (firmware/usb_cam): no WiFi,
                    so no freezes when the wearer's head shadows the antenna; the same settings and zoom
"""
import json, math, os, re, sys, threading, time, subprocess, zlib
import urllib.error, urllib.parse, urllib.request

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
    rotate = 0        # degrees clockwise the capture process turns every frame (a camera mounted sideways), see make_source

    def open(self):
        raise NotImplementedError

    def read(self):
        raise NotImplementedError

    def release(self):
        pass

    def fault(self):
        """Why the source stopped delivering frames, if it knows (shown in the stall report), else None."""
        return None

    def set_zoom(self, zoom):
        raise ValueError(f"a {self.kind} source has no sensor zoom (the ESP32-CAM's stream: and serial: sources do)")

    def reopen(self):
        """Reopen after the source stopped delivering frames (e.g. a camera unplugged and plugged back in)."""
        self.release()
        self.open()

    def info(self):
        return {"kind": self.kind, "mirror": self.mirror, "finished": self.finished, "rotate": self.rotate}


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


SENSORS = {0x26: "OV2640", 0x3660: "OV3660", 0x5640: "OV5640"}  # esp_camera sensor.h PIDs
# esp32-camera sensors/private_include/ov3660_settings.h ratio_table (the build's copy is identical): the view each aspect
# ratio reads, as max width/height, array window start/end and total size (HTS, VTS). Every row's ISP offset is 16,6.
OV3660_VIEWS = [(2048, 1536, 0, 0, 2079, 1547, 2300, 1564), (1920, 1280, 64, 128, 2015, 1419, 2172, 1436),
                (2048, 1280, 0, 128, 2079, 1419, 2300, 1436), (1920, 1152, 64, 192, 2015, 1355, 2172, 1372),
                (1920, 1080, 64, 242, 2015, 1333, 2172, 1322), (2048, 880, 0, 328, 2079, 1219, 2300, 1236),
                (1920, 1536, 64, 0, 2015, 1547, 2172, 1564), (1536, 1536, 256, 0, 1823, 1547, 2044, 1564),
                (864, 1536, 592, 0, 1487, 1547, 2044, 1564)]


def ov3660_window(zoom, w, h):
    """The board's set_res_raw parameters (its /resolution, or firmware/usb_cam's "win") that make its OV3660 read only the
    centre 1/zoom of the view a w x h frame size shows (esp32-camera sensors/ov3660.c: set_framesize, set_res_raw).
    Frame sizes up to half the array per side are read 2x2 binned, with the binned timing (VTS/2), and scaled down: HVGA
    reads 960x640 binned pixels for its 480x320. A zoom reads the centre 960/zoom x 640/zoom binned pixels with the same
    timing, so the frame rate stays; up to 2x for HVGA each zoom adds detail, and 2x sends the binned pixels 1:1, unscaled
    (2x the pixels across the mouth; measured on the board: 30.7 fps, as at 1x). Beyond that frames would only be scaled up.
    The frame size stays w x h, so frames can't show whether the board applied it: its answer does."""
    view = next((v for v in OV3660_VIEWS if v[0] * h == v[1] * w), None)
    if view is None:
        raise ValueError(f"the OV3660 has no view with the aspect ratio of {w}x{h}")
    mw, mh, sx, sy, ex, ey, tx, ty = view
    most = min(mw / 2 / w, mh / 2 / h)
    if most < 1:
        raise ValueError(f"window= needs a frame size the OV3660 reads binned (up to {mw // 2}x{mh // 2}); the board sends {w}x{h}")
    if zoom > most + 1e-6:
        raise ValueError(f"window={zoom:g}x of {w}x{h} would be scaled up, not more detail: the OV3660 gives up to {most:g}x")
    bw, bh = round(mw / 2 / zoom) // 2 * 2, round(mh / 2 / zoom) // 2 * 2  # binned pixels read, even: the Bayer order stays
    cx, cy = (sx + ex + 1) // 2, (sy + ey + 1) // 2  # centre of the view on the array
    hw, hh = bw + 16, bh + 4  # half the array window: the binned pixels plus the ISP offset (8, 2 binned) on each side
    return {"sx": cx - hw, "sy": cy - hh, "ex": cx + hw - 1, "ey": cy + hh - 1, "offx": 8, "offy": 2, "tx": tx, "ty": ty // 2 + 1,
            "ox": w, "oy": h, "scale": int((bw, bh) != (w, h)), "binning": 1}


def _zoom(window):
    """window=2x -> 2.0: a zoom above 1x (how far this board can go, ov3660_window says)."""
    try:
        zoom = float(str(window).lower().rstrip("x"))
    except ValueError:
        zoom = 0.0
    if not 1 < zoom <= 8:
        raise ValueError(f"window must be a zoom above 1x and up to 8x (e.g. window=2x), got {window!r}")
    return zoom


class Esp32Source(VideoSource):
    """An ESP32-CAM, whatever carries its frames (WiFi: StreamSource, the USB cable: SerialSource). Frames are decoded in a
    reader thread that keeps only the newest, so a slow consumer never accumulates lag; they are stamped at arrival.
    The head camera looks at the patient, so the preview is not mirrored.
    Every open pushes the settings (framesize, quality: a board reboot resets them), then window=2x zooms the sensor into
    the centre of its view (ov3660_window: the board says which sensor it has) after the settings (a framesize change
    resets it) and with no stream open (the WiFi sketch's stream handler can end on a sensor reconfigure). set_zoom changes
    the zoom live the same way. Frames count only after the ones the board buffered before a change, and must arrive at
    the expected size. Transports implement _set, _set_window, _sensor (the esp_camera PID), _connect (start receiving;
    returns at the first frame), _disconnect, release."""
    mirror = False
    BUFFERED = 2    # frames the board may hold from before a settings change (PSRAM: fb_count=2)
    DEFAULT_SETTINGS = {"framesize": 9, "quality": 16}

    def __init__(self, name, settings, timeout=5.0):
        self.name, self.settings, self.timeout = name, settings, timeout
        window = self.settings.pop("window", None)
        self.zoom = _zoom(window) if window is not None else None
        self._frame, self._ts, self._seq, self._seq0, self._got = None, 0.0, 0, 0, 0
        self._err, self._thread = None, None
        self.board_msg, self._t_board_msg = None, 0.0  # the board's last unsolicited message (e.g. "err frame capture failed")
        self.width = self.height = 0
        self.fps_est = self.mbps_est = 0.0
        self.frame_bytes = self.bad_frames = self._bad_at_frame = 0  # _bad_at_frame: bad_frames at the last good frame
        self._t_est, self._n_est, self._bytes_est = time.time(), 0, 0
        self.full = None     # frame size of the whole view, measured on every open: the window's geometry
        self.window = None   # the window's set_res_raw parameters (None: the whole view)
        self.sensor = None   # esp_camera PID, asked on the first zoom

    def open(self):
        try:
            self._configure()
        except Exception:  # whatever step failed, leave no port, socket or reader behind
            self.release()
            raise
        print(f"[source] {self.kind} {self.name} {self.width}x{self.height}" + (f", window {self.zoom:g}x of {SENSORS[self.sensor]}: {self.window}" if self.window else ""))

    def _configure(self):
        for k, v in self.settings.items():
            self._set(k, v)
        self._connect()
        self.full = self._size_after(self.BUFFERED + 1 if self.settings else 1)  # after the frames buffered before the settings
        self._got = self._seq - 1
        if self.zoom:
            try:
                self._zoom_to(self.zoom)
            except ValueError as e:
                raise RuntimeError(str(e))

    def _window_for(self, zoom):
        """zoom's window parameters (None at 1x); ValueError, before the board is touched, if this board can't zoom that far."""
        if "framesize" not in self.settings:
            raise ValueError(f"{self.name}: a sensor zoom needs the ESP32-CAM's framesize setting (pushing it is what resets the window)")
        if zoom is None:
            return None
        if self.sensor is None:
            self.sensor = self._sensor()
        if self.sensor != 0x3660:
            raise ValueError(f"{self.name}: the sensor zoom is worked out for the OV3660; this board has a {SENSORS.get(self.sensor, hex(self.sensor))}")
        return ov3660_window(zoom, *self.full)

    def _zoom_to(self, zoom):
        """Window the sensor for zoom (None: the whole view again) and start read() at the first frame that shows it."""
        window = self._window_for(zoom)
        self._disconnect()
        if window:
            self._set_window(window)
        else:
            self._set("framesize", self.settings["framesize"])  # set_framesize writes the whole view's window
        self._connect()
        size = self._size_after(self.BUFFERED + 1)
        if size != self.full:
            raise RuntimeError(f"{self.name}: frames are {size[0]}x{size[1]} after the {zoom or 1:g}x window, expected {self.full[0]}x{self.full[1]}")
        self._got = self._seq - 1
        self.zoom, self.window = zoom, window

    def set_zoom(self, zoom):
        """Change the sensor zoom live: 1 = the whole view, 2 = its centre half (ov3660_window). A zoom this board can't
        do is refused (ValueError) before the board is touched. If the board fails midway (it may have applied it, or be
        left with no stream), the source reopens at once at its previous zoom, then the RuntimeError is raised."""
        zoom = None if float(zoom) == 1 else _zoom(zoom)
        try:
            self._zoom_to(zoom)
        except RuntimeError:
            self.reopen()  # settings and the recorded zoom pushed again: board, stream and info() agree
            raise

    def _publish(self, jpg):
        """A received JPEG becomes the newest frame; a corrupt one (bytes lost on the way) is counted and dropped."""
        import cv2, numpy as np
        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            self.bad_frames += 1
            return
        self._frame, self._ts, self._seq, self._bad_at_frame = img, time.time(), self._seq + 1, self.bad_frames
        self.height, self.width, self.frame_bytes = img.shape[0], img.shape[1], len(jpg)
        self._n_est += 1; self._bytes_est += len(jpg)
        dt = self._ts - self._t_est
        if dt >= 2.0:  # what the link carried: a stream needing more than a link can carry freezes
            self.fps_est, self.mbps_est = self._n_est / dt, self._bytes_est * 8 / dt / 1e6
            self._t_est, self._n_est, self._bytes_est = self._ts, 0, 0

    def _size_after(self, n):
        """(width, height) of the n-th frame since the last _connect."""
        t0 = time.time()
        while self._seq - self._seq0 < n:
            if self._err or time.time() - t0 > self.timeout:
                raise RuntimeError(self._err or f"{self.name}: {self._seq - self._seq0} of {n} frames within {self.timeout}s")
            time.sleep(0.02)
        h, w = self._frame.shape[:2]
        return w, h

    def fault(self):
        recent = self.board_msg if time.time() - self._t_board_msg < 10 else None
        damaged = self.bad_frames - self._bad_at_frame
        return self._err or recent or (f"the last {damaged} frames all arrived damaged: bytes lost on the way" if damaged >= 3 else None)

    def _reset_estimates(self):
        self._t_est, self._n_est, self._bytes_est = time.time(), 0, 0

    def read(self):
        t0 = time.time()
        while self._seq == self._got and time.time() - t0 < min(self.stall_s, 1.0):  # wait for a new frame (like a blocking camera read)
            if self._err and self._thread and not self._thread.is_alive():
                return False, None, None
            time.sleep(0.003)
        if self._seq == self._got:
            return False, None, None
        self._got = self._seq
        return True, self._frame, self._ts

    def info(self):
        return {**super().info(), "fps": round(self.fps_est, 1), "mbps": round(self.mbps_est, 2), "frame_kb": round(self.frame_bytes / 1024, 1),
                "bad_frames": self.bad_frames, "board_msg": self.board_msg, "width": self.width, "height": self.height, "settings": self.settings,
                **({"zoom": self.zoom or 1, "sensor": SENSORS.get(self.sensor), "window": self.window} if "framesize" in self.settings else {})}


class StreamSource(Esp32Source):
    """MJPEG over HTTP: the ESP32-CAM CameraWebServer sketch over WiFi (a bare host, or http://host:81/stream; any .../stream
    URL), whose query holds /control settings and window=, or any other MJPEG URL, used as given (query included). The
    multipart stream is parsed here (no ffmpeg buffering). Over WiFi the board's antenna next to the wearer's head is the
    weak link (freezes of seconds); SerialSource (the USB cable) has none."""
    kind = "stream"
    stall_s = 8.0   # WiFi hiccups of 0.5-3 s are normal on a hotspot; reopening the TCP stream during one turns them into 10 s outages

    def __init__(self, url, timeout=5.0):
        base, _, query = url.partition("?")
        if not base.startswith("http"):
            base = f"http://{base}" if ("/" in base or ":" in base) else f"http://{base}:81/stream"   # bare host -> the ESP32 default
        settings = {}
        if urllib.parse.urlsplit(base).path == "/stream":  # the CameraWebServer sketch's stream (defaults on its usual port 81)
            settings = dict(self.DEFAULT_SETTINGS) if ":81/stream" in base else {}
            settings.update(kv.split("=", 1) for kv in query.split("&") if "=" in kv)
            query = ""
        elif "window=" in query:
            raise ValueError(f"window= needs the ESP32-CAM stream (its host, or http://host:81/stream), not {base}")
        self.url = base + (f"?{query}" if query else "")
        super().__init__(self.url, settings, timeout)
        u = urllib.parse.urlsplit(self.url)
        self.board = f"{u.scheme}://{u.hostname}"   # the ESP32 sketch's control server (:80) next to the :81 stream
        self._stop, self._resp = False, None

    def _get(self, path, params):
        q = urllib.parse.urlencode(params)
        try:
            return urllib.request.urlopen(f"{self.board}{path}?{q}", timeout=2).read()
        except urllib.error.HTTPError as e:
            hint = " (this firmware has no sensor window: flash the current CameraWebServer example)" if path == "/resolution" and e.code == 404 else ""
            raise RuntimeError(f"{self.board}{path}?{q} -> HTTP {e.code}{hint}")
        except Exception as e:
            raise RuntimeError(f"{self.board}{path}?{q} failed: {e}")

    def _set(self, var, val):
        self._get("/control", {"var": var, "val": val})

    def _set_window(self, w):
        self._get("/resolution", w)

    def _sensor(self):
        """The CameraWebServer sketch reads any sensor register with /greg: an OV3660/OV5640 has its PID at 0x300A-0x300B
        (an OV2640 reads other registers there, never 0x3660)."""
        hi, lo = (int(self._get("/greg", {"reg": r, "mask": 0xFF})) for r in (0x300A, 0x300B))
        return hi << 8 | lo

    def _connect(self):
        self._stop, self._err, self._seq0 = False, None, self._seq
        self._reset_estimates()
        req = urllib.request.Request(self.url, headers={"User-Agent": "silent-running"})
        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout)
        except Exception as e:
            raise RuntimeError(f"cannot open MJPEG stream {self.url}: {e}")
        ctype = resp.headers.get("Content-Type", "")
        if "multipart" not in ctype:
            resp.close()
            raise RuntimeError(f"{self.url} is not an MJPEG stream (Content-Type {ctype!r})")
        self._resp = resp
        self._thread = threading.Thread(target=self._reader, args=(resp,), daemon=True)
        self._thread.start()
        try:
            self._size_after(1)
        except RuntimeError:
            self.release()
            raise

    def _disconnect(self):
        self.release()

    def _reader(self, resp):
        buf = bytearray()
        try:
            while not self._stop:
                chunk = resp.read1(65536)   # whatever is available now; read(n) would block until n bytes = ~2 frames
                if not chunk:
                    self._err = "stream ended"; break
                buf += chunk
                while True:  # extract every complete JPEG (SOI..EOI)
                    a = buf.find(b"\xff\xd8")
                    if a < 0:
                        del buf[:]; break
                    b = buf.find(b"\xff\xd9", a + 2)
                    if b < 0:
                        if a > 0: del buf[:a]
                        break
                    jpg = bytes(buf[a:b + 2]); del buf[:b + 2]
                    self._publish(jpg)
        except Exception as e:
            self._err = f"stream read failed: {e}"
        finally:
            try: resp.close()
            except Exception: pass

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

    def info(self):
        return {**super().info(), "url": self.url}


def serial_ports():
    """USB-serial ports that can be the ESP32-CAM-MB board (CH340: macOS names it cu.usbserial-*, the WCH driver cu.wchusbserial*)."""
    import glob
    return sorted(glob.glob("/dev/cu.usbserial-*") + glob.glob("/dev/cu.wchusbserial*") + glob.glob("/dev/ttyUSB*"))


def serial_header(kind, payload):
    """firmware/usb_cam's message header: "SRF", the type (b"J" JPEG, b"T" text), the payload length (uint32 little-endian),
    a check byte of the length (a header that lost or corrupted bytes is rejected at once, instead of making the parser wait
    for a garbage length's worth of bytes) and the payload's CRC-32 (zlib's; the board's esp_rom_crc32_le(0, ...)): a byte
    flipped inside a JPEG would otherwise still decode, as a smeared frame the lip model reads without complaint."""
    n = len(payload)
    return b"SRF" + kind + n.to_bytes(4, "little") + bytes([_length_check(n)]) + zlib.crc32(payload).to_bytes(4, "little")


MAX_MESSAGE = 1600 * 1200 // 5  # firmware/usb_cam's JPEG buffers (sized for UXGA): no message is longer


def _length_check(n):
    return (n ^ n >> 8 ^ n >> 16 ^ n >> 24 ^ 0xA5) & 0xFF


class SerialSource(Esp32Source):
    """The ESP32-CAM over the USB cable (firmware/usb_cam): no radio, so nothing for the wearer's head to shadow. Spec:
    serial[:/dev/cu.usbserial-XXXX][?baud=1500000&framesize=9&quality=16&window=2x]; no port = the one USB-serial port.
    Settings: framesize, quality (firmware/usb_cam knows only these) and window=.
    Messages from the board (serial_header + payload): 'J' a JPEG, 'T' text: "ready" after boot, and "ok"/"err" + the
    command for each command. Bytes lost or flipped on the way cost only the message they hit: a header whose length check
    fails is skipped, a payload whose CRC-32 fails is dropped (bad_frames), and the parser resyncs at the next header.
    The board answers "ok" only after discarding the frames it buffered before the change, and the wire is ordered, so
    every frame after the last "ok" shows the new settings (BUFFERED = 0: nothing to skip on this side). A "ready" while streaming means the board reset (a brownout): it has lost the settings, so the stream ends and
    the camera process's stall recovery reopens, which pushes them again. At 1.5 Mbaud the cable carries ~150 KB/s: 30 fps
    of 5 KB HVGA frames (measured on the board (ESP32-CAM-MB, macOS): 2 Mbaud lost a byte in 60% of frames, 1.5 Mbaud 2%, 1 Mbaud 1%)."""
    kind = "serial"
    BUFFERED = 0
    # Bytes stream continuously (~150 KB/s, a frame every ~33 ms), so half a second without a frame is a burst of damaged
    # frames (reopen keeps the port) or a wedged link: macOS's CH340 driver (AppleUSBCHCOM) sometimes stops delivering
    # under load while the board keeps sending (its status stayed healthy; flushing, re-setting the baud rate or resetting
    # the board on the open port did not revive it, only reopening the port did). The fix at the source is WCH's driver.
    stall_s = 0.5

    def __init__(self, spec, timeout=5.0):
        port, _, query = spec.partition("?")
        settings = dict(self.DEFAULT_SETTINGS)
        settings.update(kv.split("=", 1) for kv in query.split("&") if "=" in kv)
        self.baud = int(settings.pop("baud", 1500000))
        self.port = port or None
        super().__init__(port or "serial", settings, timeout)
        self._ser, self._texts, self._streaming, self._awaiting, self._t_bytes = None, None, False, False, 0.0  # _t_bytes: when bytes last arrived

    def open(self):
        super().open()
        self._streaming = True

    def _open_port(self):
        """Open the port and wait until the board talks: frames, or "ready" after the reset that opening causes (the
        ESP32-CAM-MB wires DTR/RTS to EN/IO0 and opening the port pulses them: on the board every open reset it, 3/3 with
        DTR/RTS preset released and 3/3 releasing RTS before DTR after the open). They are then kept released, so the
        board runs instead of sitting in reset or its bootloader; the reset costs ~1.5 s per open."""
        if self._ser is not None:
            return
        import queue, serial
        if self.port is None:
            ports = serial_ports()
            if len(ports) != 1:
                raise RuntimeError(f"serial: {'no' if not ports else len(ports)} USB-serial ports ({ports}); name one: serial:/dev/cu.usbserial-XXXX")
            self.name = ports[0]
        else:
            self.name = self.port
        ser = serial.Serial()
        ser.port, ser.baudrate, ser.timeout = self.name, self.baud, 0.2
        ser.dtr = ser.rts = False
        try:
            ser.open()
        except serial.SerialException as e:
            raise RuntimeError(f"cannot open {self.name}: {e}")
        self._ser, self._texts, self._err, self._seq0, self._t_bytes = ser, queue.Queue(), None, self._seq, time.time()
        self._reset_estimates()
        self._thread = threading.Thread(target=self._reader, args=(ser,), daemon=True)
        self._thread.start()
        t0 = time.time()
        while self._seq == self._seq0 and self._texts.empty():
            if self._err or time.time() - t0 > self.timeout:
                err = self._err or (f"{self.name}: nothing from the board at {self.baud} baud in {self.timeout:g}s: flash firmware/usb_cam "
                                    f"(the WiFi sketch does not send frames over USB), and ?baud= must match its BAUD")
                self.release()
                raise RuntimeError(err)
            time.sleep(0.02)

    def _command(self, line):
        """Send one command; the board answers "ok <line>[ <result>]" or "err <line>". Returns the result."""
        import serial
        self._open_port()
        self._awaiting = True
        try:
            try:
                self._ser.write((line + "\n").encode())
            except serial.SerialException as e:
                raise RuntimeError(f"{self.name}: sending '{line}' failed: {e}")
            return self._await_answer(line)
        finally:
            self._awaiting = False

    def _await_answer(self, line):
        import queue
        t0 = time.time()
        while time.time() - t0 < self.timeout:
            if self._err:
                raise RuntimeError(self._err)
            try:
                text = self._texts.get(timeout=0.1)
            except queue.Empty:
                continue
            if text == f"ok {line}" or text.startswith(f"ok {line} "):
                return text[len(f"ok {line} "):]
            if text == f"err {line}":
                raise RuntimeError(f"{self.name}: the board refused '{line}'")
            print(f"[source] {self.name}: {text}")  # "ready" after the reset opening the port causes, or a capture error
        raise RuntimeError(f"{self.name}: no answer to '{line}' in {self.timeout:g}s (is firmware/usb_cam flashed?)")

    def _set(self, var, val):
        self._command(f"set {var} {val}")

    def _set_window(self, w):
        self._command("win " + " ".join(str(w[k]) for k in ("sx", "sy", "ex", "ey", "offx", "offy", "tx", "ty", "ox", "oy", "scale", "binning")))

    def _sensor(self):
        return int(self._command("sensor"), 16)

    def _connect(self):
        self._open_port()
        self._seq0 = self._seq
        try:
            self._size_after(1)
        except RuntimeError:
            self.release()
            raise

    def _disconnect(self):
        pass  # frames keep coming on the open port; _connect counts from its call

    def _reader(self, ser):
        buf = bytearray()
        try:
            while self._ser is ser:
                chunk = ser.read(max(1, ser.in_waiting))
                if chunk:
                    self._t_bytes = time.time()
                buf += chunk
                while True:
                    i = buf.find(b"SRF")
                    if i < 0:
                        del buf[:-2]; break   # boot messages or bytes of a lost message: keep a header's first 2 bytes
                    if len(buf) < i + 13:
                        del buf[:i]; break
                    kind, n = buf[i + 3], int.from_bytes(buf[i + 4:i + 8], "little")
                    if kind not in b"JT" or n > MAX_MESSAGE or buf[i + 8] != _length_check(n):  # "SRF" in other bytes, or a damaged header
                        del buf[:i + 3]; continue
                    if len(buf) < i + 13 + n:
                        del buf[:i]; break
                    payload = bytes(buf[i + 13:i + 13 + n])
                    if zlib.crc32(payload) != int.from_bytes(buf[i + 9:i + 13], "little"):
                        self.bad_frames += 1       # bytes of this message were lost or flipped; if lost, its length swallowed
                        del buf[:i + 3]; continue  # part of the next one, so resync at the next header, not after `n` bytes
                    del buf[:i + 13 + n]
                    if kind == ord("J"):
                        self._publish(payload)
                    elif payload == b"ready" and self._streaming:
                        self._err = f"{self.name}: the board reset (brownout?) and lost its settings; reopening pushes them again"
                        return
                    elif self._awaiting or not self._streaming:  # an answer to a command, or "ready" during open
                        self._texts.put(payload.decode())
                    else:  # unsolicited while streaming: keep the latest for fault() and info(), not a growing queue
                        self.board_msg, self._t_board_msg = payload.decode(), time.time()
                        print(f"[source] {self.name}: {self.board_msg}")
        except Exception as e:
            if self._ser is ser:
                self._err = f"{self.name} read failed: {e}"

    def fault(self):
        silent = time.time() - self._t_bytes
        return super().fault() or (f"no bytes from the board for {silent:.1f}s (the board, or the USB-serial driver, stopped)" if silent > 0.2 else None)

    def reopen(self):
        """After a stall. If bytes are still arriving (a burst of damaged frames, or a board reset whose "ready" was lost),
        the settings and zoom are pushed again on the open port: reopening it would reset the board (~1.5 s, see
        _open_port) and fix neither. The link streams a byte about every 7 us, so nothing new for 0.1 s means it is not
        arriving: macOS's CH340 driver wedged (or the board stopped), and only reopening the port revives it."""
        if self._ser is None or self._err or not self._thread.is_alive() or not self._bytes_arriving():
            return super().reopen()
        self._streaming = False
        try:
            self.open()
        except RuntimeError:  # the link wedged after all: open() released the port, so this opens it afresh
            super().reopen()

    def _bytes_arriving(self):
        t0 = time.time()
        while time.time() - t0 < 0.1:
            if self._ser.in_waiting or time.time() - self._t_bytes < 0.05:  # queued for the reader, or just read by it
                return True
            time.sleep(0.01)
        return False

    def release(self):
        self._streaming = False
        ser, self._ser = self._ser, None
        if ser is not None:
            if self._thread is not None:
                self._thread.join(timeout=1.0)
            ser.close()

    def info(self):
        return {**super().info(), "port": self.name, "baud": self.baud}


def with_param(spec, key, value):
    """The source spec with key=value set (None: removed): what a restarted capture process should open."""
    base, _, query = spec.partition("?")
    kv = [p for p in query.split("&") if p and not p.startswith(key + "=")]
    if value is not None:
        kv.append(f"{key}={value}")
    return base + ("?" + "&".join(kv) if kv else "")


def _flag(v):
    return str(v).lower() not in ("0", "false", "no", "off")


def rotate_frame(bgr, degrees):
    """Turn a frame clockwise by a multiple of 90 degrees (exact: no resampling, no cropping)."""
    import cv2
    return cv2.rotate(bgr, {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}[degrees])


def make_source(spec, width=640, height=480, face_fn=None):
    """Build a VideoSource from a spec string (see module docstring). An int is treated as a webcam index. Any spec takes
    rotate=90|180|270 (clockwise, for a camera mounted sideways): the capture process turns every frame before tracking."""
    spec = str(spec if spec is not None else "webcam").strip()
    base, _, query = spec.partition("?")
    rest = [kv for kv in query.split("&") if kv and not kv.startswith("rotate=")]
    rotate = int(next((kv.split("=", 1)[1] for kv in query.split("&") if kv.startswith("rotate=")), 0)) % 360
    if rotate not in (0, 90, 180, 270):
        raise ValueError(f"rotate must be 90, 180 or 270 (degrees clockwise), got {rotate}")
    src = _make_source(base + ("?" + "&".join(rest) if rest else ""), width, height, face_fn)
    src.rotate = rotate
    return src


def _make_source(spec, width, height, face_fn):
    if spec.lstrip("-").isdigit():
        spec = f"webcam:{spec}"
    if spec.startswith(("http://", "https://")) or re.match(r"^\d{1,3}(\.\d{1,3}){3}(:\d+)?(/[^?]*)?(\?.*)?$", spec):
        spec = "stream:" + spec   # a bare URL or IP means the network camera
    kind, arg = re.match(r"^([A-Za-z]*):?(.*)$", spec).groups()  # "serial?window=2x" works as well as "serial:?window=2x"
    kind = kind.lower()
    if kind == "webcam":
        return CameraSource(index=int(arg) if arg else -1, width=width, height=height, face_fn=face_fn)
    if kind == "usb":
        return USBSource(which=arg or None, width=width, height=height)
    if kind == "stream":
        return StreamSource(arg)
    if kind == "serial":
        return SerialSource(arg)
    if kind == "file":
        path, _, query = arg.partition("?")
        opts = dict(kv.split("=", 1) for kv in query.split("&") if "=" in kv)
        return FileSource(path, realtime=_flag(opts.get("realtime", 1)), loop=_flag(opts.get("loop", 1)),
                          loop_gap=float(opts.get("gap", 0)))
    raise ValueError(f"unknown video source {spec!r}; use webcam[:N] | usb[:N|name] | file:path.mp4 | stream:http://host:81/stream | serial[:port]")


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
