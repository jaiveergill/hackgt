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


class IncrementalCropper:
    """Streams the exact crops VideoProcess.crop_patch would produce, but frame by frame as they arrive.

    Frames are resampled to 25 fps against the utterance start time; the landmark smoother needs +-6 frames so
    each crop is produced 6 frames (~240 ms) after its frame; finish() crops the tail with the same truncated
    window the offline path uses. Missing landmarks hold the last valid ones (offline interpolates linearly)."""
    def __init__(self, vp, t_start):
        self.vp, self.t_start = vp, t_start
        self.items = []         # raw (ts, rgb, lm) since t_start
        self.frames, self.lms = [], []   # 25 fps resampled
        self.rois = []          # crops for frames[0:len(rois)]
        self.k = 0              # next 25 fps slot to fill
        self.last_lm = None
        self.pending_none = 0   # leading frames without landmarks, backfilled on first detection

    def feed(self, ts, rgb, lm):
        self.items.append((ts, rgb, lm))
        # fill 25 fps slots whose nearest frame is now decided (a frame at/after the target has arrived)
        while True:
            target = self.t_start + self.k / MODEL_FPS
            if self.items[-1][0] < target:
                break
            # nearest of the last frame before target and the first at/after
            j = len(self.items) - 1
            while j > 0 and self.items[j - 1][0] >= target:
                j -= 1
            cand = [j] + ([j - 1] if j > 0 else [])
            i = min(cand, key=lambda q: abs(self.items[q][0] - target))
            _, f, l = self.items[i]
            self._push(f, l)
            self.k += 1
        self._crop_ready(final=False)

    def _push(self, f, l):
        if l is None:
            if self.last_lm is None:
                self.pending_none += 1; l = None
            else:
                l = self.last_lm
        else:
            l = np.asarray(l, dtype=np.float64)
            if self.pending_none:
                for q in range(len(self.lms)):
                    if self.lms[q] is None: self.lms[q] = l
                self.pending_none = 0
            self.last_lm = l
        self.frames.append(f); self.lms.append(l)

    def _crop_one(self, i, n):
        w = min(self.vp.window_margin // 2, i, n - 1 - i)
        sm = np.mean([self.lms[x] for x in range(i - w, i + w + 1)], axis=0)
        sm += self.lms[i].mean(axis=0) - sm.mean(axis=0)
        tf, tl = self.vp.affine_transform(self.frames[i], sm, self.vp.reference, grayscale=self.vp.convert_gray)
        from pipelines.detectors.mediapipe.video_process import cut_patch
        try:
            return cut_patch(tf, tl[self.vp.start_idx:self.vp.stop_idx], self.vp.crop_height // 2, self.vp.crop_width // 2)
        except Exception:
            return self.rois[-1] if self.rois else np.zeros((self.vp.crop_height, self.vp.crop_width), dtype=np.uint8)

    def _crop_ready(self, final):
        n = len(self.frames)
        if self.pending_none:   # no landmarks yet at all
            return
        half = self.vp.window_margin // 2
        while len(self.rois) < n and (final or len(self.rois) + half < n):
            i = len(self.rois)
            self.rois.append(self._crop_one(i, n))

    def n_face(self):
        return sum(1 for _, _, l in self.items if l is not None)

    def finish(self):
        self._crop_ready(final=True)
        dur = (self.items[-1][0] - self.items[0][0]) if len(self.items) > 1 else 0.0
        return (np.stack(self.rois) if self.rois else None), dur

    def take_new(self, since):
        """Crops completed since index `since` (for streaming partials)."""
        return np.stack(self.rois[since:]) if len(self.rois) > since else None


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
    from silent_running.expression import FaceLandmarker
    fl = FaceLandmarker(blendshapes=False)   # no expression tracking on this branch: landmarks only, cheaper per frame
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
    MIN_ACTIVE, HANG, MIN_UTT, MAX_UTT, PRE_ROLL, POST_ROLL = 0.20, 0.35, 0.5, 6.0, 0.30, 0.10
    PARTIAL_EVERY = 0.20
    cropper = None; sent_rois = 0; last_partial = 0.0; just_committed = False; utt_serial = 0

    def begin(start):
        nonlocal cropper, sent_rois, last_partial, utt_serial
        cropper = IncrementalCropper(vp, start); sent_rois = 0; last_partial = time.time(); utt_serial += 1
        for it in buffer:
            if it[0] >= start: cropper.feed(*it)

    def emit(start, end, tag):
        nonlocal cropper, sent_rois
        expression = None
        c, cropper = cropper, None
        if c is None or len(c.items) < 8:
            conn.send(("utterance", None, 0, len(c.items) if c else 0, 0.0, "too short", 0.0, tag)); return
        n_face, n_total = c.n_face(), len(c.lms)
        if n_face < max(4, n_total // 4):
            conn.send(("utterance", None, n_face, n_total, 0.0, "face not tracked", 0.0, tag)); return
        try:
            t0 = time.time()
            rois, dur = c.finish()
            if rois is None:
                conn.send(("utterance", None, n_face, n_total, dur, "face not tracked", 0.0, tag)); return
            conn.send(("utterance", rois, n_face, n_total, dur, None, time.time() - t0, tag, expression))
        except Exception as e:
            conn.send(("utterance", None, n_face, n_total, 0.0, f"crop failed: {e}", 0.0, tag))

    while True:
      try:
        # commands
        while conn.poll():
            cmd = conn.recv()
            if cmd[0] == "start":
                listen_start = time.time(); just_committed = False; begin(listen_start)
            elif cmd[0] == "stop":
                start, listen_start = listen_start, None
                if start is None:
                    conn.send(("utterance", None, 0, 0, 0.0, "committed" if just_committed else "too short", 0.0, "manual")); just_committed = False; continue
                emit(start, time.time(), "manual")
            elif cmd[0] == "finish":   # parent committed early on the streamed crops: end the utterance now
                tag = "manual" if listen_start is not None else "auto"
                listen_start = None; auto_start = active_since = quiet_since = None; cropper = None
                just_committed = True
                conn.send(("finished", tag))
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
        if cropper is not None:
            try:
                cropper.feed(ts, rgb, lm)
                if ts - last_partial >= PARTIAL_EVERY:
                    chunk = cropper.take_new(sent_rois)
                    if chunk is not None:
                        sent_rois += len(chunk); last_partial = ts
                        conn.send(("partial", chunk, sent_rois, "manual" if listen_start is not None else "auto", ts - cropper.t_start, utt_serial))
            except Exception as e:
                print("[camera] crop error:", e)
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
                        auto_start = active_since - PRE_ROLL; quiet_since = None; begin(auto_start)
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
                    else:
                        cropper = None
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
                "auto": auto, "energy": round(energy, 2), "noise": round(noise or 0.0, 2), "mouth_active": bool(auto_start is not None)}
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
        self.on_partial = None   # (chunk (n,96,96), n_total_so_far, tag, seconds_since_start, utterance_serial)
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
            elif msg[0] == "partial":
                if self.on_partial:
                    try: self.on_partial(msg[1], msg[2], msg[3], msg[4], msg[5])
                    except Exception as e: print("[camera] on_partial:", e)
            elif msg[0] == "finished":
                self.listening = False
            elif msg[0] == "opened":
                self.opened, self.index = True, msg[1]
                if getattr(self, "_auto", False):
                    self._send(("auto", True))
            elif msg[0] == "fatal":
                print("[camera]", msg[1])
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

    def finish(self):
        """End the current utterance now (the server already committed on the streamed crops)."""
        self.listening = False
        self._send(("finish",))

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
