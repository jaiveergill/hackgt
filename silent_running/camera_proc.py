"""Webcam capture + face tracking + mouth cropping in a *separate process*.

Why: the VSR encoder issues thousands of tiny MPS/CPU kernels; sharing the GIL with a 30 fps capture loop
running MediaPipe made a 0.1 s encode take 1-15 s. The child process does all per-frame work and sends back
only 96x96 mouth crops for an utterance (~600 KB for 2.5 s) plus small preview JPEGs.
"""
import multiprocessing as mp
import threading, queue, time, collections, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_FPS = 25.0


def resample_25fps(items):
    ts = np.array([t for t, _, _ in items])
    t0, t1 = ts[0], ts[-1]
    duration = t1 - t0
    n = max(int(round(duration * MODEL_FPS)) + 1, 1)
    targets = t0 + np.arange(n) / MODEL_FPS
    idx = np.clip(np.searchsorted(ts, targets), 0, len(ts) - 1)
    prev = np.clip(idx - 1, 0, len(ts) - 1)
    idx = np.where(np.abs(ts[prev] - targets) < np.abs(ts[idx] - targets), prev, idx)
    frames = np.stack([items[i][1] for i in idx])
    lms = [items[i][2] for i in idx]
    return frames, lms, float(duration)


def _open(idx, width, height):
    import cv2
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    for _ in range(8):
        ok, f = cap.read()
        if ok and f is not None and f.mean() > 5:
            return cap
    cap.release()
    return None


class _MPDetector:
    """MediaPipe face detection -> [right eye, left eye, nose tip, mouth centre] pixel coords (or None). Mirrors
    third_party/chaplin/pipelines/detectors/mediapipe/detector.py so crops match what the VSR model was trained on."""
    def __init__(self):
        import mediapipe as mp
        self.fd = mp.solutions.face_detection
        self.short_range_detector = self.fd.FaceDetection(min_detection_confidence=0.5, model_selection=0)
        self.full_range_detector = self.fd.FaceDetection(min_detection_confidence=0.5, model_selection=1)

    def detect(self, frames, detector):
        out = []
        for frame in frames:
            res = detector.process(frame)
            if not res.detections:
                out.append(None); continue
            ih, iw = frame.shape[:2]
            best, best_size = None, -1
            for d in res.detections:
                b = d.location_data.relative_bounding_box
                size = b.width * iw + b.height * ih
                if size > best_size:
                    best, best_size = d, size
            kp = best.location_data.relative_keypoints
            out.append(np.array([[int(kp[i].x * iw), int(kp[i].y * ih)] for i in range(4)]))
        return out


def _worker(conn, index, width, height, preview_width, buffer_seconds):
    import cv2
    sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "third_party", "chaplin"))
    from pipelines.detectors.mediapipe.video_process import VideoProcess  # cv2/skimage only, no torch
    from silent_running.expression import FaceLandmarker, ExpressionTracker
    fl = FaceLandmarker()
    expr = ExpressionTracker()
    class _Det:  # adapter so the probing code below keeps working
        def detect(self, frames, _):
            return [fl(f)[0] for f in frames]
    detector = _Det(); det = None
    vp = VideoProcess(convert_gray=True)

    cap = None
    if index >= 0:
        cap = _open(index, width, height)
    else:
        # auto: among live cameras prefer the one that sees a face (a connected iPhone can enumerate before the FaceTime camera)
        best, best_score = None, -1
        for idx in [0, 1, 2]:
            c = _open(idx, width, height)
            if c is None:
                continue
            faces = 0
            for _ in range(6):
                ok, f = c.read()
                if ok and detector.detect([cv2.cvtColor(f, cv2.COLOR_BGR2RGB)], det)[0] is not None:
                    faces += 1
            score = faces * 10 + (1 if abs(c.get(cv2.CAP_PROP_FRAME_WIDTH) / max(c.get(cv2.CAP_PROP_FRAME_HEIGHT), 1) - 4 / 3) < 0.05 else 0)
            print(f"[camera] probe index={idx} faces={faces}/6 score={score}")
            if score > best_score:
                if best is not None: best.release()
                best, best_score, index = c, score, idx
            else:
                c.release()
        cap = best
    if cap is None:
        conn.send(("fatal", "no working camera"))
        return
    conn.send(("opened", index))

    buffer = collections.deque(maxlen=int(35 * buffer_seconds))
    t_origin = time.time()
    listen_start = None
    fps_t, fps_n, fps = time.time(), 0, 0.0
    frame_i = 0
    # auto-listen (mouth motion) state
    auto = False
    prev_patch = None
    energy = 0.0            # EMA of mouth-motion energy
    noise = None            # running estimate of the quiet-mouth energy floor
    active_since = None     # time mouth motion first exceeded the onset threshold
    quiet_since = None      # time motion dropped below the offset threshold while active
    auto_start = None       # utterance start time (with pre-roll)
    AUTO_ON, AUTO_OFF = 2.6, 1.6      # multiples of the noise floor
    MIN_ACTIVE, HANG, MIN_UTT, MAX_UTT, PRE_ROLL, POST_ROLL = 0.20, 0.60, 0.5, 6.0, 0.30, 0.15

    def emit(start, end, tag):
        expression = expr.finish()
        items = [it for it in buffer if it[0] >= start and it[0] <= end]
        if len(items) < 8:
            conn.send(("utterance", None, 0, len(items), 0.0, "too short", 0.0, tag)); return
        frames, lms, dur = resample_25fps(items)
        n_face = sum(l is not None for l in lms)
        if n_face < max(4, len(lms) // 4):
            conn.send(("utterance", None, n_face, len(lms), dur, "face not tracked", 0.0, tag)); return
        try:
            t0 = time.time()
            rois = vp(frames, list(lms))
            conn.send(("utterance", rois, n_face, len(lms), dur, None, time.time() - t0, tag, expression))
        except Exception as e:
            conn.send(("utterance", None, n_face, len(lms), dur, f"crop failed: {e}", 0.0, tag))

    while True:
      try:
        # commands
        while conn.poll():
            cmd = conn.recv()
            if cmd[0] == "start":
                listen_start = time.time(); expr.start()
            elif cmd[0] == "stop":
                start, listen_start = listen_start, None
                if start is None:
                    conn.send(("utterance", None, 0, 0, 0.0, "too short", 0.0, "manual")); continue
                emit(start, time.time(), "manual")
            elif cmd[0] == "auto":
                auto = bool(cmd[1]); active_since = quiet_since = auto_start = None
            elif cmd[0] == "snapshot":
                now = time.time()
                items = [it for it in buffer if it[0] >= now - cmd[1]]
                if len(items) < 8:
                    conn.send(("snapshot", None))
                else:
                    frames, lms, dur = resample_25fps(items)
                    conn.send(("snapshot", frames))
            elif cmd[0] == "quit":
                cap.release()
                return
        ok, bgr = cap.read()
        if not ok:
            time.sleep(0.005)
            continue
        ts = time.time()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        try:
            lm, bs = fl(rgb, (ts - t_origin) * 1000.0)
        except Exception as e:
            print("[camera] detect failed:", e); lm, bs = None, None
        buffer.append((ts, rgb, lm))
        expr.update(bs, listen_start is not None or auto_start is not None)
        # ---- mouth-motion energy (translation-compensated: patch is re-centred on the mouth every frame)
        patch = None
        if lm is not None:
            pts = lm.astype(int)
            eye = max(float(np.linalg.norm(pts[0] - pts[1])), 20.0)
            cx, cy = pts[3]
            hw, hh = int(eye * 0.55), int(eye * 0.4)
            y0, y1 = max(cy - hh, 0), min(cy + hh, rgb.shape[0]); x0, x1 = max(cx - hw, 0), min(cx + hw, rgb.shape[1])
            if y1 - y0 >= 4 and x1 - x0 >= 4:  # mouth may be partly outside the frame
                patch = cv2.resize(cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY), (32, 24)).astype(np.float32)
        if patch is not None:
            if prev_patch is not None:
                e = float(np.abs(patch - prev_patch).mean())
                energy = 0.6 * energy + 0.4 * e
                if noise is None: noise = e
                elif not (auto and auto_start): noise = 0.98 * noise + 0.02 * min(e, noise * 3)  # slow floor tracker, robust to speech bursts
            prev_patch = patch
        else:
            prev_patch = None
        if auto and noise is not None and listen_start is None:
            on_thr, off_thr = max(noise * AUTO_ON, 0.8), max(noise * AUTO_OFF, 0.5)
            if auto_start is None:
                if energy > on_thr:
                    active_since = active_since or ts
                    if ts - active_since >= MIN_ACTIVE:
                        auto_start = active_since - PRE_ROLL; quiet_since = None; expr.start()
                else:
                    active_since = None
            else:
                if energy < off_thr:
                    quiet_since = quiet_since or ts
                else:
                    quiet_since = None
                ended = (quiet_since is not None and ts - quiet_since >= HANG) or (ts - auto_start >= MAX_UTT)
                if ended:
                    end = (quiet_since or ts) + POST_ROLL
                    if end - auto_start - PRE_ROLL >= MIN_UTT:
                        emit(auto_start, min(end, ts), "auto")
                    auto_start = active_since = quiet_since = None
        fps_n += 1
        if ts - fps_t >= 1.0:
            fps, fps_t, fps_n = fps_n / (ts - fps_t), ts, 0
        frame_i += 1
        if frame_i % 2:  # preview at ~15 fps
            continue
        listening = listen_start is not None or auto_start is not None
        l0 = listen_start if listen_start is not None else auto_start
        n_listen = sum(1 for t, _, _ in buffer if listening and t >= l0)
        h, w = bgr.shape[:2]
        face = lm is not None
        bbox, face_frac = None, 0.0
        frame = bgr
        if face:
            pts = lm.astype(int)
            eye_dist = float(np.linalg.norm(pts[0] - pts[1]))
            face_frac = eye_dist / w
            cx, cy = pts[3]
            half = int(max(eye_dist * 0.9, 24))
            bbox = [int(cx - half), int(cy - half * 0.6), int(cx + half), int(cy + half * 0.6)]
            color = (80, 220, 120) if not listening else (60, 120, 255)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            for p in pts[:3]:
                cv2.circle(frame, tuple(p), 3, color, -1)
        if listening:
            cv2.circle(frame, (w - 28, 28), 12, (0, 0, 255), -1)
        if preview_width and w != preview_width:
            frame = cv2.resize(frame, (preview_width, int(h * preview_width / w)))
        frame = cv2.flip(frame, 1)
        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        meta = {"face": face, "fps": round(fps, 1), "frame_w": w, "frame_h": h, "bbox": bbox, "face_frac": round(face_frac, 3), "listening": listening, "n_frames": n_listen,
                "auto": auto, "energy": round(energy, 2), "noise": round(noise or 0.0, 2), "mouth_active": bool(auto_start is not None),
                "expression": expr.live_meta()}
        if ok:
            conn.send(("preview", jpg.tobytes(), meta))
      except (BrokenPipeError, EOFError):
        cap.release(); return  # parent went away
      except Exception as e:
        import traceback; traceback.print_exc()
        print("[camera] frame error (continuing):", e)
        prev_patch = None
        time.sleep(0.01)


class CameraProcess:
    def __init__(self, index=-1, width=640, height=480, buffer_seconds=20, preview_width=640):
        self._args = (index, width, height, preview_width, buffer_seconds)
        self.restarts = 0
        self._spawn()
        self.preview_jpeg = None
        self.meta = {"face": False, "fps": 0.0, "frame_w": width, "frame_h": height, "bbox": None, "face_frac": 0.0, "listening": False, "n_frames": 0}
        self.opened = False
        self.index = index
        self.responses = queue.Queue()
        self.listening = False
        self.on_auto_utterance = None
        self.on_signal = None
        self._quit = False
        threading.Thread(target=self._reader, daemon=True).start()

    def _spawn(self):
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        index, width, height, preview_width, buffer_seconds = self._args
        self.proc = ctx.Process(target=_worker, args=(child, index, width, height, preview_width, buffer_seconds), daemon=True)
        self.proc.start()

    def _restart(self):
        if self._quit:
            return
        self.restarts += 1
        print(f"[camera] process died; restarting (#{self.restarts})")
        self.opened = False
        self.meta = {**self.meta, "face": False, "listening": False}
        self.listening = False
        self._spawn()

    def _send(self, msg):
        try:
            self.conn.send(msg)
            return True
        except (BrokenPipeError, OSError):
            self._restart()
            return False

    def _reader(self):
        while not self._quit:
            try:
                msg = self.conn.recv()
            except (EOFError, OSError):
                self._restart()
                time.sleep(0.5)
                continue
            if msg[0] == "preview":
                self.preview_jpeg, self.meta = msg[1], msg[2]
            elif msg[0] == "opened":
                self.opened, self.index = True, msg[1]
                if getattr(self, "_auto", False):
                    self._send(("auto", True))
            elif msg[0] == "fatal":
                print("[camera]", msg[1])
            elif msg[0] == "signal":
                if self.on_signal:
                    self.on_signal(msg[1])
            elif msg[0] == "utterance" and len(msg) > 7 and msg[7] == "auto":
                if self.on_auto_utterance:
                    self.on_auto_utterance(self._utt(msg))
            else:
                self.responses.put(msg)

    @staticmethod
    def _utt(msg):
        return {"rois": msg[1], "n_face": msg[2], "n_total": msg[3], "duration": msg[4], "error": msg[5], "t_crop": msg[6] if len(msg) > 6 else 0.0,
                "expression": msg[8] if len(msg) > 8 else None}

    def set_auto(self, on):
        self._auto = bool(on)
        self._send(("auto", bool(on)))

    def start_listening(self):
        self.listening = True
        if not self._send(("start",)):
            raise RuntimeError("camera restarting, try again in a moment")

    def stop_listening(self, timeout=10.0):
        """Returns dict(rois, n_face, n_total, duration, error, t_crop)."""
        self.listening = False
        if not self._send(("stop",)):
            raise RuntimeError("camera restarting, try again in a moment")
        msg = self.responses.get(timeout=timeout)
        assert msg[0] == "utterance", msg
        return self._utt(msg)

    def snapshot_last(self, seconds, timeout=10.0):
        if not self._send(("snapshot", seconds)):
            return None
        msg = self.responses.get(timeout=timeout)
        return msg[1]

    def stop(self):
        self._quit = True
        try:
            self.conn.send(("quit",))
        except Exception:
            pass
