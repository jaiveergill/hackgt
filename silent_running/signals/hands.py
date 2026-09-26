"""Hand signals -> finger count 1-10, thumbs up/down, pointing, from the MediaPipe Hand Landmarker (Apache-2.0).

Runs inside the camera process. One landmarker pass per frame gives up to two hands with 21 landmarks each (image +
metric 3D "world" coordinates). With one hand we classify one pose:
  - thumb up / down: only the thumb straight and it points up or down
  - point: only the index finger straight and it points sideways or down (index straight up = counting 1)
  - count: number of straight fingers
With two hands both are counted and summed (5 + thumb = 6). A count of 0 is never reported: a fist looks the same as a
resting hand, and a false "0" answer to a pain question is the harmful error.
A finger is straight when its tip reaches well beyond its middle joint, measured on the metric world landmarks, so counting
does not depend on how the hand is rotated relative to the (head-mounted, moving) camera. Directions (thumb, point) are
measured relative to the patient's eye line when a face is tracked (image axes otherwise), so a tilted head or camera
does not turn "up" into "left". Thresholds were chosen on the HaGRID dev split (scripts/eval_hands.py); test is held out.
Per-frame confidence is how clearly every finger is straight or bent (1 = all far from the threshold).
Per frame the hands give one label ("fingers", n) / ("thumb", "up") / ("point", "left") / None. A label becomes a live
signal once it has been held for HOLD seconds; the utterance summary is, per kind, the held label seen on the most frames
inside a time window (the same "held" rule as the live signal, so the two never disagree).
"""
import os, collections
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL = os.path.join(ROOT, "models", "hand_landmarker.task")
FINGERS = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}
FINGER_OUT = 1.2         # straight finger: tip at least this much farther from the wrist than its middle joint (PIP) is
THUMB_OUT = 1.1          # straight thumb: tip at least this much farther from the pinky knuckle than the thumb knuckle is
MIN_DETECTION, MIN_PRESENCE = 0.7, 0.8  # MediaPipe defaults (0.5) hallucinate hands on shirts/necklines in TED clips
MARGIN = 0.15            # relative distance from a finger threshold that counts as fully clear (confidence 1)
HOLD = 0.5               # seconds a label must be held before it becomes a live signal
AGREE = 0.8              # fraction of frames in the hold window that must agree


def finger_ratios(w):
    """w: 21x3 world landmarks -> dict finger -> extension ratio over its threshold (> 1 = straight; thumb included)."""
    d = lambda a, b: np.linalg.norm(w[a] - w[b])
    out = {name: d(tip, 0) / (FINGER_OUT * d(pip, 0)) for name, (_, pip, _, tip) in FINGERS.items()}
    out["thumb"] = d(4, 17) / (THUMB_OUT * d(2, 17))
    return out


def finger_direction(img_lm, base, tip, aspect, roll):
    """Direction base -> tip from the patient's side (the camera faces the patient, so the patient's left is image right).
    img_lm: 21x2 normalised landmarks; aspect = width / height; roll: angle of the patient's eye line in the image (radians)."""
    d = (img_lm[tip] - img_lm[base]) * np.array([aspect, 1.0])
    c, s = np.cos(-roll), np.sin(-roll)
    d = np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])
    if abs(d[1]) >= abs(d[0]):
        return "up" if d[1] < 0 else "down"
    return "left" if d[0] > 0 else "right"


def hand_pose(world, img_lm, aspect, roll, count_only):
    """-> (kind, value, confidence) for one hand: ("thumb", "up"|"down") | ("point", direction) | ("fingers", n).
    count_only: always count (used when two hands are shown)."""
    r = finger_ratios(world)
    conf = float(min(np.clip(abs(v - 1.0) / MARGIN, 0.0, 1.0) for v in r.values()))
    straight = {k for k, v in r.items() if v > 1.0}
    if not count_only and straight == {"thumb"}:
        direction = finger_direction(img_lm, 2, 4, aspect, roll)
        if direction in ("up", "down"):
            return "thumb", direction, conf
    if not count_only and straight == {"index"}:
        direction = finger_direction(img_lm, 5, 8, aspect, roll)
        if direction != "up":
            return "point", direction, conf
    return "fingers", len(straight), conf


def frame_label(hands, aspect, roll):
    """hands: list of (world 21x3, img 21x2) -> (kind, value, confidence) or None. Two hands are counted and summed."""
    poses = [hand_pose(w, i, aspect, roll, len(hands) > 1) for w, i in hands]
    if len(poses) > 1:
        poses = [("fingers", sum(p[1] for p in poses), min(p[2] for p in poses))]
    if not poses or poses[0][:2] == ("fingers", 0):
        return None
    return poses[0]


def hands_from_result(res):
    """HandLandmarkerResult -> list of (world 21x3, img 21x2)"""
    return [(np.array([[p.x, p.y, p.z] for p in res.hand_world_landmarks[i]]), np.array([[p.x, p.y] for p in lms]))
            for i, lms in enumerate(res.hand_landmarks)]


class HandTracker:
    def __init__(self, model_path=MODEL, history_seconds=20.0):
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision
        opts = vision.HandLandmarkerOptions(base_options=mpp.BaseOptions(model_asset_path=model_path),
                                            running_mode=vision.RunningMode.VIDEO, num_hands=2,
                                            min_hand_detection_confidence=MIN_DETECTION, min_hand_presence_confidence=MIN_PRESENCE,
                                            min_tracking_confidence=0.5)
        self.rec = vision.HandLandmarker.create_from_options(opts)
        self.mp = mp
        self.t0 = None
        self.t_ms = -1
        self.recent = collections.deque()  # (ts, label or None) over the last HOLD seconds
        self.held = collections.deque(maxlen=int(history_seconds * 30))  # (ts, (kind, value, confidence)) for frames where a pose was held
        self.emitted = None      # (kind, value) of the last live signal; reset when the hands drop the pose
        self.last_hands = []     # image landmarks of the last frame, for the preview overlay

    def detect(self, rgb, ts):
        self.t0 = ts if self.t0 is None else self.t0
        self.t_ms = max(int((ts - self.t0) * 1000), self.t_ms + 1)  # VIDEO mode needs strictly increasing timestamps
        return hands_from_result(self.rec.detect_for_video(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), self.t_ms))

    def update(self, rgb, ts, roll):
        """Process one frame (roll: the patient's eye-line angle, 0 if no face). Returns a live signal dict when a newly
        held pose is recognised, else None."""
        hands = self.detect(rgb, ts)
        self.last_hands = [h[1] for h in hands]
        self.recent.append((ts, frame_label(hands, rgb.shape[1] / rgb.shape[0], roll)))
        full = self.recent[0][0] <= ts - HOLD * 0.8  # the window spans (almost) HOLD seconds
        while self.recent[0][0] < ts - HOLD:
            self.recent.popleft()
        if not full or len(self.recent) < 3:
            return None
        keys = [(lab[0], lab[1]) if lab else None for _, lab in self.recent]
        top, n = collections.Counter(keys).most_common(1)[0]
        agree = n / len(keys)
        if agree < AGREE:
            return None
        if top is not None:
            conf = float(np.mean([lab[2] for _, lab in self.recent if lab and (lab[0], lab[1]) == top])) * agree
            self.held.append((ts, (top[0], top[1], conf)))
        if top == self.emitted:
            return None
        self.emitted = top
        return {"kind": top[0], "value": top[1], "confidence": round(conf, 2), "ts": ts} if top else None

    def summary(self, start, end):
        """-> {"fingers": {...}, "thumb": {...}, "point": {...}}: per kind, the held value seen on the most frames in
        [start, end] with its mean live confidence (value None if nothing was held)."""
        held = [h for t, h in self.held if start <= t <= end]
        out = {k: {"value": None, "confidence": 0.0} for k in ("fingers", "thumb", "point")}
        for kind in out:
            counts = collections.Counter(v for k, v, _ in held if k == kind)
            if counts:
                value = counts.most_common(1)[0][0]
                out[kind] = {"value": value, "confidence": round(float(np.mean([c for k, v, c in held if (k, v) == (kind, value)])), 2)}
        return out

    def draw(self, bgr):
        import cv2
        h, w = bgr.shape[:2]
        for img in self.last_hands:
            pts = (img * [w, h]).astype(int)
            for a, b in ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
                         (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)):
                cv2.line(bgr, tuple(pts[a]), tuple(pts[b]), (255, 200, 60), 2)
