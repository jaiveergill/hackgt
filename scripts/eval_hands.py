"""Evaluate the hand-signal rules (silent_running/signals/hands.py) on labelled HaGRID images.

Data: scripts/fetch_hagrid.py downloads 40 images per gesture class into data/gestures/hagrid/ (labels.json is committed).
For each image the labelled hand is cropped with context (CROP x its box, as a head-mounted camera ~1 m away would see
it), the Hand Landmarker runs in IMAGE mode, and the detected hand closest to the label box is classified with
hands.hand_pose. Reports per-class accuracy for the dev and test splits (split by HaGRID user id) and writes
results/hands_hagrid_<split>.md.

With --rotate DEG the whole photo is rotated first (a rolled head-mounted camera) and directions are measured relative
to the eye line of the face in the photo (as the live path does); --no-roll measures them on raw image axes instead.
Scores are per hand pose; note the live path additionally never reports a lone fist (0).

  python scripts/eval_hands.py [--split test] [--full-frame] [--rotate 40 [--no-roll]]
"""
import os, sys, json, time, argparse, collections
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from silent_running.signals import hands as H

ROOT = H.ROOT
DATA = os.path.join(ROOT, "data", "gestures", "hagrid")
EXPECTED = {"fist": ("fingers", 0), "one": ("fingers", 1), "peace": ("fingers", 2), "two_up": ("fingers", 2),
            "peace_inverted": ("fingers", 2), "call": ("fingers", 2), "three": ("fingers", 3), "three2": ("fingers", 3),
            "four": ("fingers", 4), "palm": ("fingers", 5), "stop": ("fingers", 5), "like": ("thumb", "up"), "dislike": ("thumb", "down")}
CROP = 3.0


def crop(img, box):
    h, w = img.shape[:2]
    x, y, bw, bh = box[0] * w, box[1] * h, box[2] * w, box[3] * h
    cx, cy, half = x + bw / 2, y + bh / 2, max(bw, bh) * CROP / 2
    x0, y0, x1, y1 = int(max(cx - half, 0)), int(max(cy - half, 0)), int(min(cx + half, w)), int(min(cy + half, h))
    return img[y0:y1, x0:x1], (x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="all", choices=["dev", "test", "all"])
    ap.add_argument("--full-frame", action="store_true", help="run on the whole image instead of a crop around the hand")
    ap.add_argument("--rotate", type=float, default=0.0, help="rotate each photo by this many degrees first")
    ap.add_argument("--no-roll", action="store_true", help="with --rotate: ignore the face's roll")
    args = ap.parse_args()
    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision
    rec = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=H.MODEL), running_mode=vision.RunningMode.IMAGE, num_hands=2,
        min_hand_detection_confidence=H.MIN_DETECTION, min_hand_presence_confidence=H.MIN_PRESENCE))
    from silent_running.expression import FaceLandmarker
    face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=os.path.join(ROOT, "models", "face_landmarker.task")), running_mode=vision.RunningMode.IMAGE))
    no_face = 0
    labels = json.load(open(os.path.join(DATA, "labels.json")))
    per = collections.defaultdict(lambda: collections.Counter())  # class -> {n, found, correct}
    wrong = collections.defaultdict(collections.Counter)
    times = []
    by_conf = {True: [], False: []}  # per-hand confidence >= 0.5 -> [correct?]
    for name, lab in sorted(labels.items()):
        if args.split != "all" and lab["split"] != args.split:
            continue
        img = cv2.imread(os.path.join(DATA, name))
        if img is None:
            sys.exit(f"missing {name}: run scripts/fetch_hagrid.py first")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        box, roll = lab["bbox"], 0.0
        if args.rotate:
            h, w = rgb.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), args.rotate, 1.0)
            rgb = cv2.warpAffine(rgb, M, (w, h))
            cx, cy = M @ [(box[0] + box[2] / 2) * w, (box[1] + box[3] / 2) * h, 1.0]
            box = [cx / w - box[2] / 2, cy / h - box[3] / 2, box[2], box[3]]
            fres = face.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
            if fres.face_landmarks and not args.no_roll:
                pts = np.array([[p.x * w, p.y * h] for p in fres.face_landmarks[0]])
                dx, dy = pts[FaceLandmarker.LEFT_EYE].mean(0) - pts[FaceLandmarker.RIGHT_EYE].mean(0)
                roll = np.arctan2(dy, dx)
            no_face += not fres.face_landmarks
        if args.full_frame:
            view, (vx, vy, vw, vh) = rgb, (0.0, 0.0, 1.0, 1.0)
        else:
            view, (vx, vy, vw, vh) = crop(rgb, box)
        t = time.time()
        hands = H.hands_from_result(rec.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(view))))
        times.append(time.time() - t)
        c = per[lab["label"]]; c["n"] += 1
        # the labelled hand's centre in view coordinates; pick the detected hand nearest to it
        bx, by, bw, bh = box
        target = np.array([(bx + bw / 2 - vx) / vw, (by + bh / 2 - vy) / vh])
        if not hands:
            wrong[lab["label"]]["no hand"] += 1; continue
        best = min(hands, key=lambda hd: np.linalg.norm(hd[1].mean(0) - target))
        c["found"] += 1
        *got, conf = H.hand_pose(best[0], best[1], view.shape[1] / view.shape[0], roll, False)
        got = tuple(got)
        by_conf[conf >= 0.5].append(got == EXPECTED[lab["label"]])
        if got == EXPECTED[lab["label"]]:
            c["correct"] += 1
        else:
            wrong[lab["label"]][f"{got[0]}:{got[1]}"] += 1
    rows = ["| class | expected | n | hand found | correct | accuracy | errors |", "|---|---|---|---|---|---|---|"]
    tot = collections.Counter()
    for cls in EXPECTED:
        c = per.get(cls)
        if not c:
            continue
        tot.update(c)
        exp = f"{EXPECTED[cls][0]}:{EXPECTED[cls][1]}"
        rows.append(f"| {cls} | {exp} | {c['n']} | {c['found']} | {c['correct']} | {c['correct'] / c['n']:.0%} | "
                    f"{', '.join(f'{k} x{v}' for k, v in wrong[cls].most_common(3))} |")
    rows.append(f"| **all** | | {tot['n']} | {tot['found']} | {tot['correct']} | **{tot['correct'] / tot['n']:.1%}** | |")
    head = (f"split={args.split} view={'full frame' if args.full_frame else f'crop {CROP}x hand box'} "
            f"{f'rotate {args.rotate:g} deg, roll from face {not args.no_roll} (no face found in {no_face} photos) ' if args.rotate else ''}"
            f"landmarker latency median {np.median(times) * 1000:.1f} ms/image")
    calib = "  ".join(f"confidence {'>=' if hi else '<'} 0.5: {np.mean(v):.1%} correct of {len(v)} found hands" for hi, v in by_conf.items() if v)
    table = head + "\n\n" + "\n".join(rows) + "\n\n" + calib
    print(table)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    tag = f"{args.split}{'_full' if args.full_frame else ''}{f'_rot{args.rotate:g}' if args.rotate else ''}{'_noroll' if args.no_roll else ''}"
    with open(os.path.join(ROOT, "results", f"hands_hagrid_{tag}.md"), "w") as f:
        f.write(f"# Hand signals on HaGRID ({tag})\n\n`python scripts/eval_hands.py {' '.join(sys.argv[1:])}`\n\n{table}\n")


if __name__ == "__main__":
    main()
