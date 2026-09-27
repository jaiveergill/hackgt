"""Ambient overview: the laptop's own camera as a second, preview-only feed for the dashboard (the glasses camera is the
bedside feed). No face tracking, no recognition: a thread reads frames at ~12 fps, scales them down and keeps the latest
JPEG for /ambient. Fails soft: if the camera is busy or denied, the dashboard shows the feed as unavailable."""
import threading, time


class AmbientCamera:
    def __init__(self, index=None, width=640, fps=12, quality=70):
        """index None: probe 0-2 and keep the first camera that delivers frames (on a Mac with an iPhone paired, index 0 can be a
        Continuity Camera that opens but never delivers)."""
        self.want, self.index, self.width, self.fps, self.quality = index, index, width, fps, quality
        self.jpeg = None
        self.opened = False
        self.error = None
        self.fps_measured = 0.0
        self._stop = False
        threading.Thread(target=self._run, daemon=True, name="ambient").start()

    def _open(self):
        import cv2
        errors = []
        for idx in ([self.want] if self.want is not None else [0, 1, 2]):  # re-probe on every open: indices reorder as a Continuity Camera comes and goes
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                errors.append(f"index {idx} did not open"); continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            for i in range(20):  # macOS cameras return all-black frames while warming up; a dark room is still a picture
                ok, f = cap.read()
                if ok and f is not None and (f.mean() > 1.0 or i >= 10):
                    self.index = idx
                    return cap
                time.sleep(0.05)
            cap.release()
            errors.append(f"index {idx} delivered no frames")
        raise RuntimeError("laptop camera: " + "; ".join(errors) + " (busy, or camera permission denied?)")

    def _run(self):
        import cv2
        cap = None
        n, t_win = 0, time.time()
        while not self._stop:
            if cap is None:
                try:
                    cap = self._open()
                    self.opened, self.error = True, None
                except Exception as e:
                    self.opened, self.error = False, str(e)
                    time.sleep(5.0)
                    continue
            t = time.time()
            ok, bgr = cap.read()
            if not ok or bgr is None:
                cap.release(); cap = None; self.opened = False; self.error = "camera stopped delivering frames"
                continue
            h, w = bgr.shape[:2]
            if w > self.width:
                bgr = cv2.resize(bgr, (self.width, int(h * self.width / w)), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            if ok:
                self.jpeg = buf.tobytes()
            n += 1
            if t - t_win >= 2.0:
                self.fps_measured, n, t_win = n / (t - t_win), 0, t
            time.sleep(max(0.0, 1.0 / self.fps - (time.time() - t)))
        if cap is not None:
            cap.release()

    def info(self):
        return {"opened": self.opened, "error": self.error, "index": self.index, "fps": round(self.fps_measured, 1)}

    def stop(self):
        self._stop = True
