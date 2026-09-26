"""Run the live nonverbal path (signals/aggregate.py Signals, exactly as the camera process calls it, including the
SR_HANDS_EVERY frame stride and the utterance summary with its LOOKBACK) over a video and print the signals it emits,
expected vs actual, plus the per-frame cost.

  python scripts/check_hands.py clip.mp4 --expect fingers:3 thumb:up
  python scripts/check_hands.py --synthetic .context/hands_synth.mp4    # build + check a SYNTHETIC clip from HaGRID stills
  python scripts/check_hands.py clip.mp4 --rotate 20                     # simulate a rolled head-mounted camera

The synthetic clip holds one HaGRID test still per gesture for 1 s, with 0.6 s of empty frames between them; two-hand
cases are two stills side by side (a count summed over both hands, and a thumbs-up next to a fist, which must stay a
thumbs-up). Each segment uses the first test stills (in file order) whose composed frame the per-frame function
(IMAGE mode + frame_label) labels as expected, so the check tests the video path (VIDEO-mode tracking, stride,
hold/debounce, summary) against the per-frame result, not detector accuracy, which eval_hands.py measures. (The first
palm still, a backlit hand, is missed when it fills 75-90% of the frame height and found at 50-70%.) Pointing is a
"one" still rotated 90 degrees, so it passes by construction: pointing is NOT tested on real data. This is not a
substitute for a real recording.
"""
import os, sys, json, time, argparse, itertools
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from silent_running.signals import hands as H
from silent_running.signals.aggregate import Signals, HAND_KINDS
from scripts.eval_hands import crop, DATA

W, HEIGHT, FPS = 640, 480, 30
NEUTRAL = {"emotion": "neutral", "intensity": 0.0}
# (classes shown side by side, rotate 90 deg clockwise?, expected signal)
SCRIPT = [(["palm"], False, "fingers:5"), (["three"], False, "fingers:3"), (["like"], False, "thumb:up"),
          (["dislike"], False, "thumb:down"), (["palm", "three2"], False, "fingers:8"), (["like", "fist"], False, "thumb:up"),
          (["one"], True, "point:left")]


def compose(stills, rotate):
    """One 640x480 frame showing the stills side by side on grey."""
    frame = np.full((HEIGHT, W, 3), 128, np.uint8)
    slot = W // len(stills)
    for i, img in enumerate(stills):
        if rotate:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        s = min(slot / img.shape[1], HEIGHT / img.shape[0])
        img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)))
        x0, y0 = i * slot + (slot - img.shape[1]) // 2, (HEIGHT - img.shape[0]) // 2
        frame[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
    return frame


def segment_frame(classes, rotate, exp, labels, rec):
    """The first combination of test stills (in file order) whose composed frame the per-frame function labels `exp`."""
    import mediapipe as mp
    candidates = [[n for n, lab in sorted(labels.items()) if lab["label"] == cls and lab["split"] == "test"][:6] for cls in classes]
    for names in itertools.product(*candidates):
        frame = compose([crop(cv2.imread(os.path.join(DATA, n)), labels[n]["bbox"]) for n in names], rotate)
        hands = H.hands_from_result(rec.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))))
        got = H.frame_label(hands, W / HEIGHT)
        if got and f"{got[0]}:{got[1]}" == exp:
            return frame, names
    sys.exit(f"no test stills for {classes} give {exp} per frame")


def build_synthetic(path):
    labels = json.load(open(os.path.join(DATA, "labels.json")))
    rec = H.landmarker(False)
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, HEIGHT))
    blank = np.full((HEIGHT, W, 3), 128, np.uint8)
    expected, used, segments, t = [], [], [], 0.0
    for classes, rotate, exp in SCRIPT:
        frame, names = segment_frame(classes, rotate, exp, labels, rec); used += names
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
    signals = Signals()
    signals.attach(sent.append)
    if not signals.hands:
        sys.exit(f"hands {signals.status}")
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
        t = time.perf_counter()
        signals.update(rgb, ts)
        costs.append(time.perf_counter() - t)
        while sent:
            kind, sig = sent.pop(0)
            if kind == "error":
                sys.exit(f"ERROR from signals: {sig}")
            got.append(f"{sig['kind']}:{sig['value']}")
            print(f"  t={ts:5.2f}s signal {got[-1]:12s} confidence {sig['confidence']}")
    print(f"frames {i}  per-frame cost (averaged over the stride) mean {np.mean(costs) * 1000:.1f} ms  max {np.max(costs) * 1000:.1f} ms  "
          f"load average {os.getloadavg()[0]:.1f} on {os.cpu_count()} cores")
    summaries_ok = True
    for start, end, exp in segments:
        summ = signals.summary(start, end, NEUTRAL)  # the production summary, including aggregate.LOOKBACK
        held = {k: summ[k]["value"] for k in HAND_KINDS if summ[k]["value"] is not None}
        kind, value = exp.split(":")
        summaries_ok &= str(held.get(kind)) == value
        print(f"  summary {start:4.1f}-{end:4.1f}s expected {exp:12s} got {held}")
    if args.expect is not None:
        print(f"expected: {args.expect}\nactual:   {got}")
        passed = got == args.expect and summaries_ok
        print("CHECK PASSED" if passed else "CHECK FAILED")
        sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
