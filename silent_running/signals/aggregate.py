"""All nonverbal trackers for the camera process: per-frame update -> live ("signal", dict) messages, and the utterance
`nonverbal` summary that rides on the utterance message (event contract in CLAUDE.local.md / AGENTS.md).

Head nod/shake, blink codes and pain are not implemented yet; their summary entries are always {"value": None, "confidence": 0.0}.
Disable hands with SR_HANDS=0. The hand landmarker costs ~15-25 ms, so it runs on every SR_HANDS_EVERY-th frame
(default 2: ~15 Hz at 30 fps, plenty for gestures held 0.5 s) to keep the preview above 20 fps.
If the hand tracker fails (missing model, MediaPipe error) hands are switched off and an ("error", message) goes to the
server, which broadcasts it; lip reading keeps running.
"""
import os, math, traceback
from silent_running.signals.hands import HandTracker

LOOKBACK = 1.0  # seconds before the utterance window that still count (a patient often shows the gesture, then mouths)
ABSENT = {"value": None, "confidence": 0.0}


class Signals:
    def __init__(self, send):
        self.send = send
        self.hands = None
        self.hands_every = int(os.environ.get("SR_HANDS_EVERY", "2"))
        if self.hands_every < 1:
            raise ValueError(f"SR_HANDS_EVERY must be >= 1, got {self.hands_every}")
        self.frame_i = 0
        if os.environ.get("SR_HANDS", "1") != "0":
            try:
                self.hands = HandTracker()
            except Exception as e:
                self._fail(e)
        print(f"[signals] hands {f'on, every {self.hands_every} frames' if self.hands else 'off'}")

    def _fail(self, e):
        traceback.print_exc()
        self.hands = None
        self.send(("error", f"hand signals disabled: {type(e).__name__}: {e}"))

    def update(self, rgb, ts, face_kp):
        """face_kp: the 4 face keypoints (right eye, left eye, nose, mouth) or None; the eye line gives the head roll."""
        self.frame_i += 1
        if not self.hands or self.frame_i % self.hands_every:
            return
        roll = 0.0
        if face_kp is not None:
            dx, dy = face_kp[1] - face_kp[0]
            roll = math.atan2(dy, dx)
        try:
            sig = self.hands.update(rgb, ts, roll)
        except Exception as e:
            self._fail(e); return
        if sig:
            self.send(("signal", sig))

    def summary(self, start, end, expression):
        hands = self.hands.summary(start - LOOKBACK, end) if self.hands else {k: ABSENT for k in ("fingers", "thumb", "point")}
        return {"head": ABSENT, "blink_code": ABSENT, "pain": ABSENT, **hands,
                "emotion": {"label": expression["emotion"], "intensity": expression["intensity"]}}

    def draw(self, bgr):
        if self.hands:
            self.hands.draw(bgr)
