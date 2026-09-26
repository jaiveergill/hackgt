"""Record one speaker session that feeds BOTH the ElevenLabs voice clone (Plan 3) and speaker adaptation (Plan 2).

  python scripts/record_session.py --speaker alice            # full session (~12 min)
  python scripts/record_session.py --speaker alice --pass voiced
  python scripts/record_session.py --speaker alice --pass silent --takes 2

Passes:
  voiced      read ~40 sentences ALOUD   -> audio (voice clone + Whisper labels) + video (adaptation)
  silent      mouth the 30 phrase-bank phrases silently, N takes -> adaptation train / held-out test
  silent_free mouth 10 arbitrary sentences silently            -> open-vocabulary test only
  expression  mouth 4 phrases with an ANGRY face, then with a WARM face -> calibrates face->emotion (Plan 4)

Each take is captured with ffmpeg (avfoundation): video 640x480 @25 fps H.264 + audio 16 kHz mono AAC in one
MP4, and a separate 16 kHz WAV for the voiced pass. Output: data/session/<speaker>/<pass>/<idx>__<take>.mp4
and a manifest at data/session/<speaker>/manifest.jsonl with the prompt text.

Controls: SPACE start / stop the take, r re-record last, n skip, q quit.
"""
import argparse, os, json, time, subprocess
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHRASES = [l.strip() for l in open(os.path.join(ROOT, "silent_running", "phrases.txt")) if l.strip() and not l.startswith("#")]

HARVARD = [
    "The birch canoe slid on the smooth planks", "Glue the sheet to the dark blue background", "It's easy to tell the depth of a well",
    "These days a chicken leg is a rare dish", "Rice is often served in round bowls", "The juice of lemons makes fine punch",
    "The box was thrown beside the parked truck", "The hogs were fed chopped corn and garbage", "Four hours of steady work faced us",
    "A large size in stockings is hard to sell", "The boy was there when the sun rose", "A rod is used to catch pink salmon",
    "The source of the huge river is the clear spring", "Kick the ball straight and follow through", "Help the woman get back to her feet",
    "A pot of tea helps to pass the evening", "Smoky fires lack flame and heat", "The soft cushion broke the man's fall",
    "The salt breeze came across from the sea", "The girl at the booth sold fifty bonds",
]
HOSPITAL = [
    "I would like to see my daughter today", "Can you please raise the head of the bed", "The pain is worse when I breathe in",
    "I did not sleep well last night", "Please tell the doctor I feel better", "My mouth is very dry right now",
    "I want to try eating something soft", "When can I go home", "The light is too bright in here", "I need to change position",
    "Please close the window", "My back is hurting a lot", "Thank you for taking care of me", "I am worried about the surgery",
    "Can I have some ice chips", "What did the test results say", "I feel much better than yesterday", "Please bring my phone",
    "I cannot feel my left hand", "Is my family coming today",
]
SILENT_FREE = [
    "Good morning everyone", "I would like a cup of coffee", "Please open the door", "The weather is nice today",
    "My name is on the chart", "Can you help me stand up", "I want to watch the news", "Turn the volume down please",
    "I am ready to go now", "See you tomorrow morning",
]
EXPRESSION = [f"{how}: {ph}" for how in ("ANGRY face", "WARM face") for ph in ("I need the nurse", "Call my family", "I am in pain", "Thank you")]
PASSES = {"voiced": HARVARD + HOSPITAL, "silent": PHRASES, "silent_free": SILENT_FREE, "expression": EXPRESSION}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speaker", required=True)
    ap.add_argument("--pass", dest="passes", default="voiced,silent,silent_free,expression")
    ap.add_argument("--takes", type=int, default=1, help="takes per prompt (use 2 for the silent pass: 1 train + 1 test)")
    ap.add_argument("--video-dev", default="0", help="avfoundation video index (FaceTime HD Camera is usually 0)")
    ap.add_argument("--audio-dev", default="0", help="avfoundation audio index (MacBook Pro Microphone is usually 0)")
    ap.add_argument("--preview-cam", type=int, default=0, help="OpenCV index for the on-screen preview")
    args = ap.parse_args()

    out_root = os.path.join(ROOT, "data", "session", args.speaker)
    manifest = os.path.join(out_root, "manifest.jsonl")
    os.makedirs(out_root, exist_ok=True)
    queue = []
    for p in args.passes.split(","):
        for i, text in enumerate(PASSES[p]):
            for t in range(args.takes if p == "silent" else 1):
                queue.append((p, i, t, text))

    cap = cv2.VideoCapture(args.preview_cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    proc = None; cur = None; t_start = 0; qi = 0
    print(f"{len(queue)} takes queued. SPACE start/stop, r redo, n skip, q quit.")
    while qi < len(queue):
        pss, idx, take, text = queue[qi]
        ok, frame = cap.read()
        if not ok:
            continue
        disp = cv2.flip(frame, 1)
        h, w = disp.shape[:2]
        cv2.rectangle(disp, (0, 0), (w, 92), (0, 0, 0), -1)
        mode = "READ ALOUD" if pss == "voiced" else ("MOUTH SILENTLY with the face shown" if pss == "expression" else "MOUTH SILENTLY (no sound)")
        cv2.putText(disp, f"[{qi+1}/{len(queue)}] {mode}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255) if pss == "voiced" else (120, 255, 120), 2)
        cv2.putText(disp, text, (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        if proc is not None:
            cv2.circle(disp, (w - 25, 30), 12, (0, 0, 255), -1)
            cv2.putText(disp, f"REC {time.time()-t_start:.1f}s  SPACE=stop", (10, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        else:
            cv2.putText(disp, "SPACE=start  r=redo  n=skip  q=quit", (10, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)
        cv2.imshow("record_session", disp)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("n") and proc is None:
            qi += 1
        elif k == ord("r") and proc is None and qi > 0:
            qi -= 1
        elif k == ord(" "):
            if proc is None:
                d = os.path.join(out_root, pss); os.makedirs(d, exist_ok=True)
                cur = os.path.join(d, f"{idx:03d}__{take}.mp4")
                cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "avfoundation", "-framerate", "30", "-video_size", "640x480",
                       "-i", f"{args.video_dev}:{args.audio_dev}", "-r", "25", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
                       "-pix_fmt", "yuv420p", "-ar", "16000", "-ac", "1", "-c:a", "aac", cur]
                cap.release()  # avfoundation needs exclusive access to the camera
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
                t_start = time.time()
            else:
                dur = time.time() - t_start
                proc.communicate(input=b"q", timeout=15); proc = None
                cap = cv2.VideoCapture(args.preview_cam); cap.set(3, 640); cap.set(4, 480)
                if dur < 0.8 or not os.path.exists(cur):
                    print("too short, discarded"); continue
                wav = None
                if pss == "voiced":
                    wav = cur[:-4] + ".wav"
                    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", cur, "-vn", "-ar", "16000", "-ac", "1", wav])
                label_text = text.split(": ", 1)[1] if pss == "expression" else text
                rec = {"speaker": args.speaker, "pass": pss, "idx": idx, "take": take, "text": label_text, "prompt": text,
                       "expected_emotion": (text.split(" ")[0].lower() if pss == "expression" else None), "file": os.path.relpath(cur, ROOT),
                       "wav": os.path.relpath(wav, ROOT) if wav else None, "duration": round(dur, 2), "silent": pss != "voiced", "ts": time.time()}
                with open(manifest, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"saved {rec['file']} ({dur:.1f}s) '{text}'")
                qi += 1
    if proc is not None:
        proc.communicate(input=b"q", timeout=15)
    cap.release(); cv2.destroyAllWindows()
    print("session written to", manifest)


if __name__ == "__main__":
    main()
