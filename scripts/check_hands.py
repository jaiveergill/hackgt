"""Run the live nonverbal path (signals/aggregate.py Signals, exactly as the camera process calls it, including the
SR_HANDS_EVERY frame stride) over a video and print the signals it emits, expected vs actual, plus the per-frame cost.

  python scripts/check_hands.py clip.mp4 --expect fingers:3 thumb:up
  python scripts/check_hands.py --synthetic .context/hands_synth.mp4    # build + check a SYNTHETIC clip from HaGRID stills
  python scripts/check_hands.py clip.mp4 --rotate 50                     # simulate a rolled head-mounted camera

The synthetic clip holds one held-out HaGRID test image per gesture for 1 s, with 0.6 s of empty frames between them
(two-hand counts are two stills side by side; pointing is a "one" still rotated 90 degrees). It exercises the per-frame
rules, the hold/debounce logic and the utterance summary of each gesture; it is not a substitute for a real recording.
"""
import os, sys, json, time, argparse
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from silent_running.signals.aggregate import Signals
from silent_running.expression import FaceLandmarker
from scripts.eval_hands import crop, DATA

W, H, FPS = 640, 480, 30
# (classes shown side by side, rotate 90 deg clockwise?, expected signal)
SCRIPT = [(["palm"], False, "fingers:5"), (["three"], False, "fingers:3"), (["like"], False, "thumb:up"),
          (["dislike"], False, "thumb:down"), (["palm", "three2"], False, "fingers:8"), (["one"], True, "point:left")]


def still(cls, labels):
    name = next(k for k, v in sorted(labels.items()) if v["label"] == cls and v["split"] == "test")
    return crop(cv2.imread(os.path.join(DATA, name)), labels[name]["bbox"])[0], name


def build_synthetic(path):
    labels = json.load(open(os.path.join(DATA, "labels.json")))
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    blank = np.full((H, W, 3), 128, np.uint8)
    expected, used, segments, t = [], [], [], 0.0
    for classes, rotate, exp in SCRIPT:
        frame = blank.copy()
        slot = W // len(classes)
        for i, cls in enumerate(classes):
            img, name = still(cls, labels); used.append(name)
            if rotate:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            s = min(slot / img.shape[1], H / img.shape[0])
            img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)))
            x0, y0 = i * slot + (slot - img.shape[1]) // 2, (H - img.shape[0]) // 2
            frame[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
        for _ in range(int(0.6 * FPS)):
            out.write(blank)
        for _ in range(FPS):
            out.write(frame)
        t += 0.6 + 1.0
        expected.append(exp); segments.append((t - 1.0, t, exp))
    for _ in range(int(0.6 * FPS)):
        out.write(blank)
    out.release()
    print(f"SYNTHETIC clip {path} from HaGRID test stills: {', '.join(used)}")
    return expected, segments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?")
    ap.add_argument("--expect", nargs="*", default=None, help="expected live signals in order, e.g. fingers:3 thumb:up")
    ap.add_argument("--synthetic", help="write a synthetic clip here and check it")
    ap.add_argument("--rotate", type=float, default=0.0, help="rotate every frame by this many degrees (camera roll)")
    args = ap.parse_args()
    segments = []
    if args.synthetic:
        (args.expect, segments), args.video = build_synthetic(args.synthetic), args.synthetic
    if not args.video:
        ap.error("give a video or --synthetic")
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    sent = []
    signals = Signals(sent.append)
    face = FaceLandmarker()
    got, costs, i = [], [], 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        ts = i / fps; i += 1
        if args.rotate:
            h, w = bgr.shape[:2]
            bgr = cv2.warpAffine(bgr, cv2.getRotationMatrix2D((w / 2, h / 2), args.rotate, 1.0), (w, h))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        kp = face(rgb, ts * 1000)[0]  # face keypoints give the head roll, as in the camera process
        t = time.perf_counter()
        signals.update(rgb, ts, kp)
        costs.append(time.perf_counter() - t)
        while sent:
            kind, sig = sent.pop(0)
            if kind == "error":
                sys.exit(f"ERROR from signals: {sig}")
            got.append(f"{sig['kind']}:{sig['value']}")
            print(f"  t={ts:5.2f}s signal {got[-1]:12s} confidence {sig['confidence']}")
    print(f"frames {i}  per-frame cost (averaged over the stride) mean {np.mean(costs) * 1000:.1f} ms  max {np.max(costs) * 1000:.1f} ms  "
          f"load average {os.getloadavg()[0]:.1f} on {os.cpu_count()} cores")
    for start, end, exp in segments:
        summ = signals.hands.summary(start, end)  # the window itself, without aggregate.LOOKBACK
        print(f"  summary {start:4.1f}-{end:4.1f}s expected {exp:12s} got "
              f"{ {k: summ[k]['value'] for k in ('fingers', 'thumb', 'point') if summ[k]['value'] is not None} }")
    if args.expect is not None:
        print(f"expected: {args.expect}\nactual:   {got}")
        print("CHECK PASSED" if got == args.expect else "CHECK FAILED")
        sys.exit(0 if got == args.expect else 1)


if __name__ == "__main__":
    main()
