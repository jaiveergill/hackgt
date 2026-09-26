"""Hand signals -> finger count 1-10, thumbs up/down, pointing, from the MediaPipe Hand Landmarker (Apache-2.0).

Runs inside the camera process. One landmarker pass per frame gives up to two hands with 21 landmarks each (image +
metric 3D "world" coordinates). Each hand is first classified on its own:
  - thumb up / down: only the thumb straight, clearly (its own clarity is 1, see MARGIN), pointing up or down;
    a lone thumb held sideways gives no label (it is a tilted thumbs up/down, and would otherwise read as "1")
  - point: only the index finger straight and it points sideways or down (index straight up = counting 1)
  - count: number of straight fingers
A hand hanging down (wrist above the knuckles) is resting, not signalling, and is ignored unless only its index finger
is straight (pointing down): on HaGRID dev full frames 84% of the resting second hands hang, no gesturing hand does, and a
hanging relaxed hand otherwise reads as "thumb down" (1 of 31 hanging hands read as a point). A hand showing 0 (fist)
is ignored too. If two hands remain, two matching thumbs give that thumb, otherwise both are counted and summed
(5 + thumb = 6); a single remaining hand gives the label. So a thumbs-up next to a resting fist stays "thumb up", and a lone fist gives no
signal at all: a fist looks the same as a resting hand, and a false "0" answer to a pain question is the harmful error.
A finger is straight when its tip reaches well beyond its middle joint, measured on the metric world landmarks, so counting
does not depend on how the hand is rotated relative to the (head-mounted, moving) camera. Directions (thumb, point,
hanging) are measured on image axes: a thumbs-up points against gravity, and the head-mounted camera is upright while
the wearer's head is, whereas the patient's own head may lie at any roll (turned on their side every 2 h in the ICU), so
the patient's eye line is not a reference for "up". On HaGRID dev full photos, eye-line roll compensation (the first
version of this module) got no more thumbs right at +-20 deg camera roll (31 vs 35 of 74 without it) and only helped at
+-40 deg (35 vs 14 of 74); results/hands_hagrid_test_rot*.md has the test numbers. Counting is unaffected by roll.
Thresholds were chosen on the HaGRID dev split (scripts/eval_hands.py); test is held out.
Confidence (per frame, and the same number on live signals and in the utterance summary) is how clearly every finger is
straight or bent: 1 = all at least MARGIN from the threshold. It is calibrated on HaGRID full frames (see eval_hands.py).
Per frame the hands give one label ("fingers", n) / ("thumb", "up") / ("point", "left") / None. A label becomes a live
signal once it has been held for HOLD seconds; the utterance summary is, per kind, the held label seen on the most frames
during the utterance, else the last one held just before it (the same "held" rule as the live signal, so every summary
value was also sent live).
Left/right are from the patient's side, assuming an unmirrored camera image (webcam, glasses camera and file sources all
deliver raw frames); a clip recorded mirrored (e.g. Photo Booth) swaps them.
"""
import os, collections
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL = os.path.join(ROOT, "models", "hand_landmarker.task")
FINGERS = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}
FINGER_OUT = 1.2         # straight finger: tip at least this much farther from the wrist than its middle joint (PIP) is
THUMB_OUT = 1.1          # straight thumb: tip at least this much farther from the pinky knuckle than the thumb knuckle is
MIN_DETECTION, MIN_PRESENCE = 0.7, 0.8  # MediaPipe defaults (0.5) hallucinate hands on shirts/necklines in TED clips
MARGIN = 0.15            # relative distance from a finger threshold that counts as fully clear (confidence 1); a
                         # lone thumb must be this clear: on HaGRID dev (6 views) it cuts fist -> thumb from 10% to 1%
HOLD = 0.5               # seconds a label must be held before it becomes a live signal
AGREE = 0.8              # fraction of frames in the hold window that must agree
HISTORY = 20             # seconds of held poses kept for utterance summaries (the camera buffer is 20 s too)
KINDS = ("fingers", "thumb", "point")


def finger_ratios(w):
    """w: 21x3 world landmarks -> dict finger -> extension ratio over its threshold (> 1 = straight; thumb included)."""
    d = lambda a, b: np.linalg.norm(w[a] - w[b])
    out = {name: d(tip, 0) / (FINGER_OUT * d(pip, 0)) for name, (_, pip, _, tip) in FINGERS.items()}
    out["thumb"] = d(4, 17) / (THUMB_OUT * d(2, 17))
    return out


def finger_direction(img_lm, base, tip, aspect):
    """Direction base -> tip on image axes, from the patient's side (the camera faces the patient, so the patient's left is
    image right). img_lm: 21x2 normalised landmarks; aspect = width / height."""
    d = (img_lm[tip] - img_lm[base]) * np.array([aspect, 1.0])
    if abs(d[1]) >= abs(d[0]):
        return "up" if d[1] < 0 else "down"
    return "left" if d[0] > 0 else "right"


def straight_fingers(world):
    """-> (set of straight fingers, confidence). A lone thumb counts only when clearly straight (its own clarity is 1):
    just past the threshold is how a fist often reads."""
    r = finger_ratios(world)
    clarity = {k: float(np.clip(abs(v - 1.0) / MARGIN, 0.0, 1.0)) for k, v in r.items()}
    straight = {k for k, v in r.items() if v > 1.0}
    if straight == {"thumb"} and clarity["thumb"] < 1.0:
        straight = set()
    return straight, min(clarity.values())


def hand_pose(world, img_lm, aspect):
    """-> (kind, value, confidence) for one hand shown on its own: ("thumb", "up"|"down") | ("point", direction) |
    ("fingers", n >= 1), or None for a fist or a lone thumb held sideways (a tilted thumbs up/down, not a count)."""
    straight, conf = straight_fingers(world)
    if straight == {"thumb"}:
        direction = finger_direction(img_lm, 2, 4, aspect)
        return ("thumb", direction, conf) if direction in ("up", "down") else None
    if straight == {"index"}:
        direction = finger_direction(img_lm, 5, 8, aspect)
        if direction != "up":
            return "point", direction, conf
    return ("fingers", len(straight), conf) if straight else None


def frame_label(hands, aspect):
    """hands: list of (world 21x3, img 21x2) -> (kind, value, confidence) or None. Fists and hands hanging at rest are
    ignored (a hand pointing down is kept); two remaining hands give the same thumb if both show it (two thumbs up = yes,
    up + down = no label), otherwise their counts are summed (5 + thumb = 6)."""
    shown = []
    for w, i in hands:
        straight, conf = straight_fingers(w)
        if straight and (straight == {"index"} or finger_direction(i, 0, 9, aspect) != "down"):
            shown.append((w, i, len(straight), conf))
    if len(shown) == 2:
        conf = min(shown[0][3], shown[1][3])
        poses = [hand_pose(w, i, aspect) for w, i, _, _ in shown]
        if all(p and p[0] == "thumb" for p in poses):
            return ("thumb", poses[0][1], conf) if poses[0][1] == poses[1][1] else None
        return "fingers", shown[0][2] + shown[1][2], conf
    return hand_pose(shown[0][0], shown[0][1], aspect) if shown else None


def hands_from_result(res):
    """HandLandmarkerResult -> list of (world 21x3, img 21x2)"""
    return [(np.array([[p.x, p.y, p.z] for p in res.hand_world_landmarks[i]]), np.array([[p.x, p.y] for p in lms]))
            for i, lms in enumerate(res.hand_landmarks)]


def landmarker(video):
    """The MediaPipe Hand Landmarker with the live settings; video=False (IMAGE mode, no tracking) is for the stills eval."""
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision
    return vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=MODEL),
        running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE, num_hands=2,
        min_hand_detection_confidence=MIN_DETECTION, min_hand_presence_confidence=MIN_PRESENCE, min_tracking_confidence=0.5))


class HandTracker:
    def __init__(self):
        import mediapipe as mp
        self.rec = landmarker(True)
        self.mp = mp
        self.t0 = None
        self.t_ms = -1
        self.recent = collections.deque()  # (ts, label or None) over the last HOLD seconds
        self.held = collections.deque(maxlen=HISTORY * 30)  # (ts, (kind, value, confidence)) for frames where a pose was held
        self.emitted = None      # (kind, value) of the last live signal; reset when the hands drop the pose
        self.last_hands = []     # image landmarks of the last frame, for the preview overlay

    def detect(self, rgb, ts):
        self.t0 = ts if self.t0 is None else self.t0
        self.t_ms = max(int((ts - self.t0) * 1000), self.t_ms + 1)  # VIDEO mode needs strictly increasing timestamps
        return hands_from_result(self.rec.detect_for_video(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), self.t_ms))

    def update(self, rgb, ts):
        """Process one frame. Returns a live signal dict when a newly held pose is recognised, else None."""
        hands = self.detect(rgb, ts)
        self.last_hands = [h[1] for h in hands]
        self.recent.append((ts, frame_label(hands, rgb.shape[1] / rgb.shape[0])))
        full = self.recent[0][0] <= ts - HOLD * 0.8  # the window spans (almost) HOLD seconds
        while self.recent[0][0] < ts - HOLD:
            self.recent.popleft()
        if not full or len(self.recent) < 3:
            return None
        keys = [(lab[0], lab[1]) if lab else None for _, lab in self.recent]
        top, n = collections.Counter(keys).most_common(1)[0]
        if n / len(keys) < AGREE:
            return None
        if top is not None:
            conf = float(np.mean([lab[2] for _, lab in self.recent if lab and (lab[0], lab[1]) == top]))
            self.held.append((ts, (top[0], top[1], conf)))
        if top == self.emitted:
            return None
        self.emitted = top
        return {"kind": top[0], "value": top[1], "confidence": round(conf, 2), "ts": ts} if top else None

    def summary(self, start, end, lookback):
        """-> {"fingers": {...}, "thumb": {...}, "point": {...}}: per kind, the held value seen on the most frames in
        [start, end]; if none was held there, the last one held in the `lookback` seconds before start (a patient often
        shows the gesture, then mouths). Confidence is its mean per-frame confidence; value None if nothing was held."""
        out = {}
        for kind in KINDS:
            during = [(v, c) for t, (k, v, c) in self.held if k == kind and start <= t <= end]
            before = [(v, c) for t, (k, v, c) in self.held if k == kind and start - lookback <= t < start]
            frames = during or before
            if not frames:
                out[kind] = {"value": None, "confidence": 0.0}
                continue
            value = collections.Counter(v for v, _ in during).most_common(1)[0][0] if during else before[-1][0]
            out[kind] = {"value": value, "confidence": round(float(np.mean([c for v, c in frames if v == value])), 2)}
        return out

    def draw(self, bgr):
        import cv2
        h, w = bgr.shape[:2]
        for img in self.last_hands:
            pts = (img * [w, h]).astype(int)
            for a, b in ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
                         (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)):
                cv2.line(bgr, tuple(pts[a]), tuple(pts[b]), (255, 200, 60), 2)
