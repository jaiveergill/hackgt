"""All nonverbal trackers for the camera process: per-frame update -> live ("signal", dict) messages, and the utterance
`nonverbal` summary that rides on the utterance message (event contract in CLAUDE.local.md / AGENTS.md).

Head nod/shake, blink codes and pain are not implemented yet; their summary entries are always {"value": None, "confidence": 0.0}.
Disable hands with SR_HANDS=0. The hand landmarker costs ~15-25 ms, so it runs on every SR_HANDS_EVERY-th frame
(default 2: ~15 Hz at 30 fps, plenty for gestures held 0.5 s) to keep the preview above 20 fps.
`status` ("on" | "off: <reason>") is the hands state; the camera process puts it in every preview meta, so the UI sees it
even when the tracker failed at startup, before any client was connected. If the tracker fails (missing model, bad
SR_HANDS_EVERY, MediaPipe error) hands are switched off, the reason goes into `status` and an ("error", message) goes to
the server; lip reading keeps running.
"""
import os, traceback
from silent_running.signals.hands import HandTracker

LOOKBACK = 1.0  # seconds before the utterance window that still count (a patient often shows the gesture, then mouths)
ABSENT = {"value": None, "confidence": 0.0}
HAND_KINDS = ("fingers", "thumb", "point")


def nonverbal_dict(expression, hands=None):
    """The contract's `nonverbal` dict. hands=None (no hand tracker, e.g. a decoded file) gives absent hand entries."""
    return {"head": ABSENT, "blink_code": ABSENT, "pain": ABSENT, **(hands or {k: ABSENT for k in HAND_KINDS}),
            "emotion": {"label": expression["emotion"], "intensity": expression["intensity"]}}


class Signals:
    def __init__(self, send):
        self.send = send
        self.hands = None
        self.frame_i = 0
        self.status = "off: disabled by SR_HANDS=0"
        if os.environ.get("SR_HANDS", "1") != "0":
            try:
                self.hands_every = int(os.environ.get("SR_HANDS_EVERY", "2"))
                if self.hands_every < 1:
                    raise ValueError(f"SR_HANDS_EVERY must be >= 1, got {self.hands_every}")
                self.hands = HandTracker()
                self.status = "on"
            except Exception as e:
                self._fail(e)
        print(f"[signals] hands {self.status}" + (f", every {self.hands_every} frames" if self.hands else ""))

    def _fail(self, e):
        traceback.print_exc()
        self.hands = None
        self.status = f"off: {type(e).__name__}: {e}"
        self.send(("error", f"hand signals {self.status}"))

    def update(self, rgb, ts):
        self.frame_i += 1
        if not self.hands or self.frame_i % self.hands_every:
            return
        try:
            sig = self.hands.update(rgb, ts)
        except Exception as e:
            self._fail(e); return
        if sig:
            self.send(("signal", sig))

    def summary(self, start, end, expression):
        return nonverbal_dict(expression, self.hands.summary(start - LOOKBACK, end) if self.hands else None)

    def draw(self, bgr):
        if self.hands:
            self.hands.draw(bgr)
