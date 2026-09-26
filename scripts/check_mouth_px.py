"""Check the camera panel's mouth-pixels readout (camera_proc.MouthPixels) on a recorded clip.

The same video at half and double resolution must read half and double the mouth width (the number follows the camera's
pixels, not the face's share of the frame), and it must agree with the eye distance (reference face: 54.3 px apart when
the mouth is 45 px wide).

    python scripts/check_mouth_px.py [data/samples/ted1_short.mp4]
"""
import os, sys
import numpy as np, cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from silent_running.camera_proc import MouthPixels, VideoProcess
from silent_running.expression import FaceLandmarker

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "samples", "ted1_short.mp4")
cap, frames = cv2.VideoCapture(path), []
while len(frames) < 50:
    ok, f = cap.read()
    if not ok: break
    frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
mp = MouthPixels(VideoProcess(convert_gray=True))
fails = 0
med = {}
for scale in (1.0, 0.5, 2.0):
    fl = FaceLandmarker()  # VIDEO mode: one landmarker per pass, timestamps increasing
    px, eye = [], []
    for i, f in enumerate(frames):
        f = cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        lm, _ = fl(f, i * 40.0)
        if lm is not None:
            px.append(mp(lm)); eye.append(float(np.linalg.norm(lm[0] - lm[1])))
    med[scale] = np.median(px)
    from_eye = np.median(eye) * mp.need / 54.29
    ok = abs(med[scale] / from_eye - 1) < 0.15
    fails += not ok
    print(f"scale {scale:3.1f}: mouth {med[scale]:6.1f} px (model reads {mp.need:.0f}), from eye distance {from_eye:6.1f}  "
          f"{'ok' if ok else 'FAIL'} ({len(px)}/{len(frames)} frames tracked)")
for scale in (0.5, 2.0):
    ratio = med[scale] / med[1.0]
    ok = abs(ratio / scale - 1) < 0.08
    fails += not ok
    print(f"expected x{scale}: got x{ratio:.2f}  {'ok' if ok else 'FAIL'}")
print("CHECK PASSED" if not fails else f"CHECK FAILED ({fails})")
sys.exit(1 if fails else 0)
