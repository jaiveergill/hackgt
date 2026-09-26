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


class _Outbox:
    """Sends to the parent from a thread so the capture loop never blocks on the pipe. The parent's reader thread can be
    starved of the GIL for over a second (model loading, decoding); a blocking send then froze capture itself, dropping the
    frames of whatever the patient was mouthing. Control messages (utterances, snapshots, errors) are queued and all
    delivered in order; previews are latest-wins: a newer preview replaces one the parent hasn't taken yet."""
    def __init__(self, conn):
        self.conn, self.dead, self.dropped = conn, False, 0
        self._q, self._preview = collections.deque(), None
        self._cv = threading.Condition()
        threading.Thread(target=self._run, daemon=True).start()

    def send(self, msg):
        with self._cv:
            self._q.append(msg); self._cv.notify()

    def preview(self, msg):
        with self._cv:
            if self._preview is not None:
                self.dropped += 1
            self._preview = msg; self._cv.notify()

    def _run(self):
        while True:
            with self._cv:
                while not self._q and self._preview is None:
                    self._cv.wait()
                if self._q:
                    msg = self._q.popleft()
                else:
                    msg, self._preview = self._preview, None
            try:
                self.conn.send(msg)
            except (BrokenPipeError, OSError):
                self.dead = True
                return
            except Exception as e:  # e.g. an unpicklable payload: a bug in that message, not in the pipe
                import traceback; traceback.print_exc()
                self.send(("error", f"camera process could not send {msg[0]!r}: {e!r}"))


def _worker(conn, spec, width, height, preview_width, buffer_seconds):
    import cv2
    sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "third_party", "chaplin"))
    from pipelines.detectors.mediapipe.video_process import VideoProcess  # cv2/skimage only, no torch
    from silent_running.expression import FaceLandmarker, ExpressionTracker
    from silent_running.sources import make_source
    from silent_running.signals.aggregate import Signals
    fl = FaceLandmarker()
    expr = ExpressionTracker()
    signals = Signals()
    vp = VideoProcess(convert_gray=True)

    t_origin = time.time()  # one clock for every landmarker call (VIDEO mode needs increasing timestamps), probe included
    try:
        src = make_source(spec, width, height, face_fn=lambda rgb: fl(rgb, (time.time() - t_origin) * 1000.0)[0] is not None)
        src.open()
    except Exception as e:  # reported to the UI as an error event via the parent's on_error
        import traceback; traceback.print_exc()
        conn.send(("fatal", f"could not open video source {spec!r}: {e}"))
        return
    conn.send(("opened", src.info()))
    out = _Outbox(conn)
    signals.attach(out.send)  # after "opened": its sends go through the outbox like every other worker message
    STALL_S, MAX_RETRY_S = 1.5, 8.0  # report + reopen a live camera after 1.5 s without frames; back off to 8 s while it stays gone
    last_frame = last_reopen = time.time()
    retry_s = STALL_S

    buffer = collections.deque(maxlen=int(35 * buffer_seconds))
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
            out.send(("utterance", None, 0, len(items), 0.0, "too short", 0.0, tag)); return
        frames, lms, dur = resample_25fps(items)
        n_face = sum(l is not None for l in lms)
        if n_face < max(4, len(lms) // 4):
            out.send(("utterance", None, n_face, len(lms), dur, "face not tracked", 0.0, tag)); return
        try:
            t0 = time.time()
            rois = vp(frames, list(lms))
            out.send(("utterance", rois, n_face, len(lms), dur, None, time.time() - t0, tag, expression, signals.summary(start, end, expression)))
        except Exception as e:
            out.send(("utterance", None, n_face, len(lms), dur, f"crop failed: {e}", 0.0, tag))

    while True:
      try:
        if out.dead:  # parent went away
            src.release(); return
        # commands
        while conn.poll():
            cmd = conn.recv()
            if cmd[0] == "start":
                listen_start = time.time(); expr.start()
            elif cmd[0] == "stop":
                start, listen_start = listen_start, None
                if start is None:
                    out.send(("utterance", None, 0, 0, 0.0, "too short", 0.0, "manual")); continue
                emit(start, time.time(), "manual")
            elif cmd[0] == "auto":
                auto = bool(cmd[1]); active_since = quiet_since = auto_start = None
            elif cmd[0] == "snapshot":
                now = time.time()
                items = [it for it in buffer if it[0] >= now - cmd[1]]
                if len(items) < 8:
                    out.send(("snapshot", None))
                else:
                    frames, lms, dur = resample_25fps(items)
                    out.send(("snapshot", frames))
            elif cmd[0] == "quit":
                src.release()
                return
        ok, bgr, ts = src.read()
        if not ok:
            if src.finished:
                continue
            now = time.time()
            if src.kind != "file" and now - last_frame >= STALL_S and now - last_reopen >= retry_s:
                # unplugged / dead camera: say so (UI error event) and try to reopen instead of freezing silently
                out.send(("error", f"{src.kind} camera delivered no frames for {now - last_frame:.1f}s; reopening"))
                try:
                    src.reopen()
                    print(f"[camera] reopened {src.info()}")
                except RuntimeError as e:
                    out.send(("error", f"reopening {src.kind} camera failed (next try in {min(retry_s * 2, MAX_RETRY_S):.0f}s): {e}"))
                    retry_s = min(retry_s * 2, MAX_RETRY_S)
                last_reopen = time.time()
            time.sleep(0.005)
            continue
        last_frame, retry_s = time.time(), STALL_S
        if src.discontinuity:  # file looped: don't count the jump as mouth motion
            prev_patch = None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        try:
            lm, bs = fl(rgb, (ts - t_origin) * 1000.0)
        except Exception as e:
            print("[camera] detect failed:", e); lm, bs = None, None
        buffer.append((ts, rgb, lm))
        expr.update(bs, listen_start is not None or auto_start is not None)
        signals.update(rgb, ts)
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
        signals.draw(frame)
        if listening:
            cv2.circle(frame, (w - 28, 28), 12, (0, 0, 255), -1)
        if preview_width and w != preview_width:
            frame = cv2.resize(frame, (preview_width, int(h * preview_width / w)))
        if src.mirror:
            frame = cv2.flip(frame, 1)
        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        meta = {"face": face, "fps": round(fps, 1), "frame_w": w, "frame_h": h, "bbox": bbox, "face_frac": round(face_frac, 3), "listening": listening, "n_frames": n_listen,
                "auto": auto, "energy": round(energy, 2), "noise": round(noise or 0.0, 2), "mouth_active": bool(auto_start is not None),
                "expression": expr.live_meta(), "source": src.info(), "preview_dropped": out.dropped, "hands": signals.status}
        if ok:
            out.preview(("preview", jpg.tobytes(), meta))
      except (BrokenPipeError, EOFError):
        src.release(); return  # parent went away
      except Exception as e:
        import traceback; traceback.print_exc()
        print("[camera] frame error (continuing):", e)
        prev_patch = None
        time.sleep(0.01)


class CameraProcess:
    def __init__(self, source="webcam", width=640, height=480, buffer_seconds=20, preview_width=640):
        """source: a spec string for silent_running.sources.make_source (webcam[:N] | usb[:N|name] | file:path.mp4)."""
        self._args = (source, width, height, preview_width, buffer_seconds)
        self.restarts = 0
        self.fatal = None
        self._lock = threading.RLock()  # guards conn/proc swaps against the reader applying a message from a replaced process
        self._switch_lock = threading.Lock()  # one switch_source at a time (it waits for the new source without holding _lock)
        self._spawn()
        self.preview_jpeg = None
        self.meta = {"face": False, "fps": 0.0, "frame_w": width, "frame_h": height, "bbox": None, "face_frac": 0.0, "listening": False, "n_frames": 0}
        self.opened = False
        self.source = source
        self.source_info = None
        self.responses = queue.Queue()
        self.listening = False
        self.on_auto_utterance = None
        self.on_error = None
        self.on_signal = None
        self._quit = False
        threading.Thread(target=self._reader, daemon=True).start()

    @staticmethod
    def _start_worker(args):
        ctx = mp.get_context("spawn")
        conn, child = ctx.Pipe()
        proc = ctx.Process(target=_worker, args=(child, *args), daemon=True)
        proc.start()
        return conn, proc

    def _spawn(self):
        self.conn, self.proc = self._start_worker(self._args)

    def _restart(self):
        with self._lock:
            if self._quit or self.fatal:  # a source that failed to open won't open on retry; wait for switch_source()
                return
            self.restarts += 1
            print(f"[camera] process died; restarting (#{self.restarts})")
            self.opened = False
            self.meta = {**self.meta, "face": False, "listening": False}
            self.listening = False
            self._spawn()

    def switch_source(self, source, timeout=60.0):
        """Swap the video source live (e.g. webcam -> recorded backup), make-before-break: the new capture process must
        report "opened" before the old one is stopped, so a source that fails to open leaves the working one running.
        Raises RuntimeError with the reason on failure."""
        with self._switch_lock:
            self._switch(source, timeout)

    def _switch(self, source, timeout):
        new_args = (source,) + self._args[1:]
        new_conn, new_proc = self._start_worker(new_args)
        opened, reason, t_end = None, f"did not open within {timeout:.0f}s", time.time() + timeout
        while opened is None and time.time() < t_end:
            if not new_conn.poll(0.2):
                if not new_proc.is_alive():
                    reason = f"capture process exited (code {new_proc.exitcode})"
                    break
                continue
            msg = new_conn.recv()
            if msg[0] == "opened":
                opened = msg[1]
            elif msg[0] == "fatal":
                reason = msg[1]
                break
        if opened is None:
            new_proc.terminate(); new_proc.join(); new_conn.close()
            raise RuntimeError(f"could not switch to {source!r}: {reason}; kept {self.source!r}")
        with self._lock:
            old_conn, old_proc = self.conn, self.proc
            self.conn, self.proc, self._args = new_conn, new_proc, new_args
            self.source, self.source_info, self.fatal = source, opened, None
            self.opened, self.listening = True, False
            self.preview_jpeg = None  # the old source's last frame and meta must not be served as the new source's
            self.meta = {**self.meta, "source": opened, "fps": 0.0, "face": False}
            while not self.responses.empty():  # replies queued by the old process belong to its source
                self.responses.get_nowait()
            if getattr(self, "_auto", False):
                self._send(("auto", True))
        try:
            old_conn.send(("quit",))
        except (BrokenPipeError, OSError):
            pass  # already exited
        # A healthy worker releases its source within ~0.1 s of "quit", but interpreter/MediaPipe teardown can take
        # over 2 s under load; one stuck reading a dead camera never sees "quit" at all. Either way, kill it.
        old_proc.join(2.0)
        if old_proc.is_alive():
            print("[camera] old capture process still alive 2 s after quit; terminating it")
            old_proc.terminate()
            old_proc.join()

    def _send(self, msg):
        try:
            self.conn.send(msg)
            return True
        except (BrokenPipeError, OSError):
            self._restart()
            return False

    def _reader(self):
        while not self._quit:
            conn = self.conn
            try:
                msg = conn.recv()
            except (EOFError, OSError):
                with self._lock:
                    if conn is not self.conn:  # old process after switch_source(); read from the new one
                        continue
                    self._restart()
                time.sleep(0.5)
                continue
            with self._lock:
                if conn is not self.conn:  # late message from the process we just replaced
                    continue
                self._apply(msg)

    def _apply(self, msg):
        if msg[0] == "preview":
            self.preview_jpeg, self.meta = msg[1], msg[2]
        elif msg[0] == "opened":
            self.opened, self.source_info = True, msg[1]
            if getattr(self, "_auto", False):
                self._send(("auto", True))
        elif msg[0] == "signal":
            if self.on_signal:
                self.on_signal(msg[1])
        elif msg[0] in ("fatal", "error"):
            print("[camera]", msg[1])
            if msg[0] == "fatal":
                self.fatal = msg[1]
            if self.on_error:
                self.on_error(msg[1])
        elif msg[0] == "utterance" and len(msg) > 7 and msg[7] == "auto":
            if self.on_auto_utterance:
                self.on_auto_utterance(self._utt(msg))
        elif msg[0] in ("utterance", "snapshot"):  # replies to stop_listening / snapshot_last
            self.responses.put(msg)
        else:  # a message kind with no handler here must not be taken as the reply to the next request
            print("[camera] unhandled message from the capture process:", msg[0])
            if self.on_error:
                self.on_error(f"unhandled message from the capture process: {msg[0]!r}")

    @staticmethod
    def _utt(msg):
        return {"rois": msg[1], "n_face": msg[2], "n_total": msg[3], "duration": msg[4], "error": msg[5], "t_crop": msg[6] if len(msg) > 6 else 0.0,
                "expression": msg[8] if len(msg) > 8 else None, "nonverbal": msg[9] if len(msg) > 9 else None}

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
