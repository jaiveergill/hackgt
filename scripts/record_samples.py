"""Record silently-mouthed evaluation samples from the webcam.

Usage:
  python scripts/record_samples.py --speaker jaiveer [--phrases phrases.txt] [--reps 2]

Shows a preview window with the prompt phrase. Press SPACE to start recording, SPACE again to stop
(or hold for 'hold' mode), 'n' to skip, 'q' to quit. Each take is saved as
  data/eval/<speaker>/<phrase_slug>__<take>.mp4   (25 fps, 640x480, no audio)
and appended to data/eval/manifest.jsonl with the intended phrase.
"""
import argparse, os, re, json, time
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PHRASES = os.path.join(ROOT, "silent_running", "phrases.txt")

def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speaker", required=True)
    ap.add_argument("--phrases", default=DEFAULT_PHRASES)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--only", default=None, help="comma-separated substrings to select phrases")
    args = ap.parse_args()

    phrases = [l.strip() for l in open(args.phrases) if l.strip() and not l.startswith("#")]
    if args.only:
        keys = [k.strip().lower() for k in args.only.split(",")]
        phrases = [p for p in phrases if any(k in p.lower() for k in keys)]
    outdir = os.path.join(ROOT, "data", "eval", args.speaker)
    os.makedirs(outdir, exist_ok=True)
    manifest = os.path.join(ROOT, "data", "eval", "manifest.jsonl")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    w, h = int(cap.get(3)), int(cap.get(4))
    print(f"camera {w}x{h}")

    queue = [(p, r) for r in range(args.reps) for p in phrases]
    i = 0
    recording = False
    frames = []
    t_last = time.time()
    interval = 1.0 / args.fps
    while i < len(queue):
        phrase, rep = queue[i]
        ok, frame = cap.read()
        if not ok:
            continue
        now = time.time()
        if recording and now - t_last >= interval:
            frames.append(frame.copy())
            t_last = now
        disp = cv2.flip(frame, 1)
        color = (0, 0, 255) if recording else (0, 200, 0)
        cv2.rectangle(disp, (0, 0), (w, 70), (0, 0, 0), -1)
        cv2.putText(disp, f"[{i+1}/{len(queue)}] {phrase}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(disp, "REC" if recording else "SPACE=start/stop  n=skip  q=quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if recording:
            cv2.circle(disp, (w - 25, 35), 12, (0, 0, 255), -1)
        cv2.imshow("record", disp)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("n") and not recording:
            i += 1
        elif k == ord(" "):
            if not recording:
                recording, frames, t_last = True, [], time.time()
            else:
                recording = False
                if len(frames) < args.fps:  # < 1 s
                    print("too short, discarded")
                    continue
                take = 0
                while True:
                    fn = os.path.join(outdir, f"{slug(phrase)}__{take}.mp4")
                    if not os.path.exists(fn):
                        break
                    take += 1
                tmp = fn + ".raw.mp4"
                vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
                for f in frames:
                    vw.write(f)
                vw.release()
                os.system(f'ffmpeg -y -loglevel error -i "{tmp}" -an -r {args.fps} -c:v libx264 -crf 18 -pix_fmt yuv420p "{fn}" && rm "{tmp}"')
                rec = {"speaker": args.speaker, "phrase": phrase, "file": os.path.relpath(fn, ROOT), "n_frames": len(frames), "fps": args.fps, "ts": time.time()}
                with open(manifest, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"saved {rec['file']} ({len(frames)} frames, {len(frames)/args.fps:.1f}s)")
                i += 1
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
