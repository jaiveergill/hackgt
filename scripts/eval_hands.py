"""Evaluate the live hand-signal function (silent_running/signals/hands.py frame_label) on labelled HaGRID photos.

Data: scripts/fetch_hagrid.py downloads 40 photos per gesture class into data/gestures/hagrid/ (labels.json is committed).
Each photo is scored the way the camera process scores a frame: the Hand Landmarker (live settings, IMAGE mode because
these are stills) runs on the whole photo, and hands.frame_label labels it from every detected hand (hanging hands and
fists ignored, two hands summed). A fist is expected to give no label; every other class its gesture, so a missed hand
counts as an error. Reports per-class accuracy, the precision of the labels reported at each confidence (the number the
fusion step thresholds on), and the harmful errors (false thumbs up/down, a fist read as anything) for the dev and test
splits (split by HaGRID user id) and writes results/hands_hagrid_<tag>.md.

--crop is an ORACLE view, not the live path: the photo is first cropped to CROP x the ground-truth hand box, as if the
hand filled more of the frame. --rotate DEG rotates the photo first (a rolled head-mounted camera).

  python scripts/eval_hands.py [--split test] [--crop] [--rotate 40]
"""
import os, sys, json, time, argparse, collections
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from silent_running.signals import hands as H

ROOT = H.ROOT
DATA = os.path.join(ROOT, "data", "gestures", "hagrid")
EXPECTED = {"fist": None, "one": ("fingers", 1), "peace": ("fingers", 2), "two_up": ("fingers", 2),
            "peace_inverted": ("fingers", 2), "call": ("fingers", 2), "three": ("fingers", 3), "three2": ("fingers", 3),
            "four": ("fingers", 4), "palm": ("fingers", 5), "stop": ("fingers", 5), "like": ("thumb", "up"), "dislike": ("thumb", "down")}
CROP = 3.0
BINS = (0.0, 0.25, 0.5, 0.75, 1.01)


def crop(img, box):
    h, w = img.shape[:2]
    x, y, bw, bh = box[0] * w, box[1] * h, box[2] * w, box[3] * h
    cx, cy, half = x + bw / 2, y + bh / 2, max(bw, bh) * CROP / 2
    x0, y0, x1, y1 = int(max(cx - half, 0)), int(max(cy - half, 0)), int(min(cx + half, w)), int(min(cy + half, h))
    return img[y0:y1, x0:x1]


def name(label):
    return f"{label[0]}:{label[1]}" if label else "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="all", choices=["dev", "test", "all"])
    ap.add_argument("--crop", action="store_true", help=f"ORACLE: crop {CROP}x around the ground-truth hand box first")
    ap.add_argument("--rotate", type=float, default=0.0, help="rotate each photo by this many degrees first (camera roll)")
    args = ap.parse_args()
    import mediapipe as mp
    rec = H.landmarker(False)
    labels = json.load(open(os.path.join(DATA, "labels.json")))
    per = collections.defaultdict(collections.Counter)  # class -> {n, correct}
    wrong = collections.defaultdict(collections.Counter)
    reported = []  # (confidence, correct?) for every photo that got a label
    times = []
    for fname, lab in sorted(labels.items()):
        if args.split != "all" and lab["split"] != args.split:
            continue
        img = cv2.imread(os.path.join(DATA, fname))
        if img is None:
            sys.exit(f"missing {fname}: run scripts/fetch_hagrid.py first")
        rgb, box = cv2.cvtColor(img, cv2.COLOR_BGR2RGB), lab["bbox"]
        if args.rotate:
            h, w = rgb.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), args.rotate, 1.0)
            rgb = cv2.warpAffine(rgb, M, (w, h))
            cx, cy = M @ [(box[0] + box[2] / 2) * w, (box[1] + box[3] / 2) * h, 1.0]
            box = [cx / w - box[2] / 2, cy / h - box[3] / 2, box[2], box[3]]
        view = np.ascontiguousarray(crop(rgb, box) if args.crop else rgb)
        t = time.time()
        hands = H.hands_from_result(rec.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=view)))
        times.append(time.time() - t)
        got = H.frame_label(hands, view.shape[1] / view.shape[0])
        exp, ok = EXPECTED[lab["label"]], (got[:2] if got else None) == EXPECTED[lab["label"]]
        c = per[lab["label"]]; c["n"] += 1; c["correct"] += ok
        if got:
            reported.append((got[2], ok, lab["label"], got[:2]))
        if not ok:
            wrong[lab["label"]][name(got)] += 1
    rows = ["| class | expected | n | correct | accuracy | errors |", "|---|---|---|---|---|---|"]
    tot = collections.Counter()
    for cls in EXPECTED:
        c = per.get(cls)
        if not c:
            continue
        tot.update(c)
        rows.append(f"| {cls} | {name(EXPECTED[cls])} | {c['n']} | {c['correct']} | {c['correct'] / c['n']:.0%} | "
                    f"{', '.join(f'{k} x{v}' for k, v in wrong[cls].most_common(3))} |")
    rows.append(f"| **all** | | {tot['n']} | {tot['correct']} | **{tot['correct'] / tot['n']:.1%}** | |")
    calib = ["| label confidence | labels reported | correct |", "|---|---|---|"]
    for lo, hi in zip(BINS, BINS[1:]):
        v = [ok for conf, ok, *_ in reported if lo <= conf < hi]
        calib.append(f"| {lo:.2f}-{min(hi, 1.0):.2f} | {len(v)} | {np.mean(v):.1%} |" if v else f"| {lo:.2f}-{min(hi, 1.0):.2f} | 0 | |")
    for thr in (0.25, 0.5, 0.75):
        v = [ok for conf, ok, *_ in reported if conf >= thr]
        calib.append(f"| **>= {thr}** | {len(v)} ({len(v) / tot['n']:.0%} of photos) | **{np.mean(v):.1%}** |")
    n_fist = per["fist"]["n"]
    harm = (f"harmful errors: false thumb:up (a \"yes\") on {sum(g == ('thumb', 'up') and not ok for _, ok, _, g in reported)} photos, "
            f"false thumb:down (a \"no\") on {sum(g == ('thumb', 'down') and not ok for _, ok, _, g in reported)}; "
            f"fist -> thumb:up {sum(c == 'fist' and g == ('thumb', 'up') for _, _, c, g in reported)}/{n_fist}, "
            f"fist -> any label {sum(c == 'fist' for _, _, c, _ in reported)}/{n_fist}")
    head = (f"split={args.split} view={f'ORACLE crop {CROP}x ground-truth hand box' if args.crop else 'full photo (live path)'}"
            f"{f', rotated {args.rotate:g} deg' if args.rotate else ''}; landmarker median {np.median(times) * 1000:.1f} ms/photo")
    table = head + "\n\n" + "\n".join(rows) + "\n\n" + "\n".join(calib) + "\n\n" + harm
    print(table)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    tag = f"{args.split}{'_crop' if args.crop else ''}{f'_rot{args.rotate:g}' if args.rotate else ''}"
    with open(os.path.join(ROOT, "results", f"hands_hagrid_{tag}.md"), "w") as f:
        f.write(f"# Hand signals on HaGRID ({tag})\n\n`python scripts/eval_hands.py {' '.join(sys.argv[1:])}`\n\n{table}\n")


if __name__ == "__main__":
    main()
