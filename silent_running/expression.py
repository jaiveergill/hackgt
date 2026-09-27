"""Facial expression -> emotion label + intensity, from MediaPipe Face Landmarker blendshapes (Plan 4, step A).

Runs inside the camera process. Per frame: 52 blendshape scores -> rule scores for angry / warm / sad / surprised.
Per utterance: robust aggregate (median of the top quartile) minus the user's resting baseline -> label + intensity.
Mouth-shape coefficients get half weight because the mouth is busy forming words during silent speech.
"""
import os, collections
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "models", "face_landmarker.task")
EMOTIONS = ["angry", "warm", "sad", "surprised"]
NEUTRAL_THRESHOLD = 0.22


def _m(bs, *names):
    return float(np.mean([bs.get(n, 0.0) for n in names]))


def frame_scores(bs):
    """bs: dict blendshape name -> score. Returns dict emotion -> 0..1 raw score."""
    brow_down = _m(bs, "browDownLeft", "browDownRight")
    squint = _m(bs, "eyeSquintLeft", "eyeSquintRight")
    sneer = _m(bs, "noseSneerLeft", "noseSneerRight")
    press = _m(bs, "mouthPressLeft", "mouthPressRight") * 0.5
    smile = _m(bs, "mouthSmileLeft", "mouthSmileRight") * 0.5
    cheek = _m(bs, "cheekSquintLeft", "cheekSquintRight")
    frown = _m(bs, "mouthFrownLeft", "mouthFrownRight") * 0.5
    inner_up = bs.get("browInnerUp", 0.0)
    outer_up = _m(bs, "browOuterUpLeft", "browOuterUpRight")
    wide = _m(bs, "eyeWideLeft", "eyeWideRight")
    angry = 0.5 * brow_down + 0.25 * squint + 0.25 * max(sneer, press)
    warm = max(0.0, 0.6 * smile * 2 + 0.4 * cheek - 0.5 * brow_down)
    sad = 0.5 * inner_up + 0.5 * frown * 2
    surprised = 0.5 * outer_up + 0.5 * wide
    return {"angry": min(angry, 1.0), "warm": min(warm, 1.0), "sad": min(sad, 1.0), "surprised": min(surprised, 1.0)}


class ExpressionTracker:
    """Keeps a rolling baseline of the resting face and aggregates scores over an utterance window."""
    def __init__(self, baseline_seconds=5.0, fps=30):
        self.baseline_buf = collections.deque(maxlen=int(baseline_seconds * fps))
        self.baseline = {e: 0.0 for e in EMOTIONS}
        self.live = {e: 0.0 for e in EMOTIONS}
        self.utt = []          # scores during the current listen window

    def update(self, bs, listening):
        sc = frame_scores(bs) if bs else None
        if sc is None:
            return
        # EMA for the live meter
        for e in EMOTIONS:
            self.live[e] = 0.7 * self.live[e] + 0.3 * sc[e]
        if listening:
            self.utt.append(sc)
        else:
            self.baseline_buf.append(sc)
            if len(self.baseline_buf) >= 15:
                for e in EMOTIONS:
                    self.baseline[e] = float(np.median([s[e] for s in self.baseline_buf]))

    def start(self):
        self.utt = []

    def finish(self, reset=True):
        """-> {emotion, intensity, scores, n_frames}. Call at the end of a listen window (reset=False: the window goes on)."""
        if len(self.utt) < 5:
            return {"emotion": "neutral", "intensity": 0.0, "scores": {}, "n_frames": len(self.utt)}
        agg = {}
        for e in EMOTIONS:
            v = np.sort([s[e] for s in self.utt])
            top = v[int(len(v) * 0.75):] if len(v) >= 8 else v
            agg[e] = float(np.median(top)) - self.baseline[e]
        best = max(agg, key=agg.get)
        inten = float(np.clip(agg[best] * 1.6, 0.0, 1.0))  # gain: resting-vs-expressive deltas are small
        out = {"emotion": best if inten >= NEUTRAL_THRESHOLD else "neutral", "intensity": round(inten if inten >= NEUTRAL_THRESHOLD else 0.0, 2),
               "scores": {e: round(max(agg[e], 0.0), 3) for e in EMOTIONS}, "baseline": {e: round(self.baseline[e], 3) for e in EMOTIONS}, "n_frames": len(self.utt)}
        if reset:
            self.utt = []
        return out

    def live_meta(self):
        return {e: round(max(self.live[e] - self.baseline[e], 0.0), 2) for e in EMOTIONS}


class FaceLandmarker:
    """MediaPipe tasks Face Landmarker: 478 landmarks + 52 blendshapes. Also yields the 4 keypoints
    (right eye, left eye, nose tip, mouth centre) the mouth-crop pipeline expects, in the same order as Chaplin's detector."""
    # landmark indices (MediaPipe face mesh)
    RIGHT_EYE = [33, 133, 160, 159, 158, 144, 145, 153]
    LEFT_EYE = [362, 263, 387, 386, 385, 373, 374, 380]
    NOSE_TIP = [1, 4]
    MOUTH = [61, 291, 0, 17, 13, 14, 78, 308]

    def __init__(self, model_path=MODEL):
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision
        opts = vision.FaceLandmarkerOptions(base_options=mpp.BaseOptions(model_asset_path=model_path),
                                            running_mode=vision.RunningMode.VIDEO, num_faces=1,
                                            output_face_blendshapes=True, output_facial_transformation_matrixes=False,
                                            min_face_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.lm = vision.FaceLandmarker.create_from_options(opts)
        self.mp = mp

    def __call__(self, rgb, ts_ms):
        """-> (keypoints 4x2 int array or None, blendshapes dict or None)"""
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        res = self.lm.detect_for_video(img, int(ts_ms))
        if not res.face_landmarks:
            return None, None
        h, w = rgb.shape[:2]
        pts = np.array([[l.x * w, l.y * h] for l in res.face_landmarks[0]])
        kp = np.array([pts[self.RIGHT_EYE].mean(0), pts[self.LEFT_EYE].mean(0), pts[self.NOSE_TIP].mean(0), pts[self.MOUTH].mean(0)]).astype(int)
        bs = {c.category_name: c.score for c in res.face_blendshapes[0]} if res.face_blendshapes else None
        return kp, bs
