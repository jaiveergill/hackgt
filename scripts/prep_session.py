"""Turn a recorded session (scripts/record_session.py) into adaptation data.

For every take in data/session/<speaker>/manifest.jsonl:
  * mouth crops via the exact inference pipeline (MediaPipe -> affine align -> 96x96 gray @25 fps) -> <take>.npy (T,96,96) uint8
  * label: prompt text for silent takes; for voiced takes, OpenAI transcription of the WAV, kept only if it agrees with the
    prompt within --max-wer (default 0.34) — otherwise the prompt is used and the take is flagged.
Writes data/session/<speaker>/prep.jsonl and a train/test split:
  train = voiced + silent take 0        test = silent take 1 (+ silent_free)

  python scripts/prep_session.py --speaker alice [--no-asr]
Runs in its own process on purpose: the inference pipeline (Chaplin's espnet) and the trainer (Auto-AVSR's espnet) clash.
"""
import os, sys, json, argparse, re
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def norm(s):
    return re.sub(r"[^a-z' ]", "", s.lower().replace("-", " ")).split()


def wer(h, r):
    import torchaudio
    h, r = norm(h), norm(r)
    return torchaudio.functional.edit_distance(h, r) / max(len(r), 1)


def transcribe(wav):
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
    from openai import OpenAI
    c = OpenAI()
    with open(wav, "rb") as f:
        r = c.audio.transcriptions.create(model="gpt-4o-transcribe", file=f, language="en")
    return r.text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speaker", required=True)
    ap.add_argument("--no-asr", action="store_true", help="use prompt text for voiced takes too")
    ap.add_argument("--max-wer", type=float, default=0.34)
    args = ap.parse_args()
    from silent_running.vsr import VSREngine
    eng = VSREngine(device="cpu", decode_device="cpu")
    sdir = os.path.join(ROOT, "data", "session", args.speaker)
    recs = [json.loads(l) for l in open(os.path.join(sdir, "manifest.jsonl")) if l.strip()]
    out = []
    for r in recs:
        path = os.path.join(ROOT, r["file"])
        try:
            x, lm, rois = eng.preprocess_video(path)
        except Exception as e:
            print("SKIP (crop failed)", r["file"], e); continue
        n_face = sum(l is not None for l in lm)
        if n_face < len(lm) // 2:
            print("SKIP (face)", r["file"], f"{n_face}/{len(lm)}"); continue
        npy = path[:-4] + ".npy"
        np.save(npy, rois.astype(np.uint8))
        label, asr, flag = r["text"], None, None
        if r["pass"] == "voiced" and r.get("wav") and not args.no_asr:
            try:
                asr = transcribe(os.path.join(ROOT, r["wav"]))
                w = wer(asr, r["text"])
                if w <= args.max_wer:
                    label = asr
                else:
                    flag = f"asr disagrees (wer {w:.2f}); using prompt"
            except Exception as e:
                flag = f"asr failed: {e}"
        split = "test" if (r["pass"] == "silent_free" or (r["pass"] == "silent" and r["take"] >= 1)) else "train"
        rec = {**r, "npy": os.path.relpath(npy, ROOT), "n_frames": int(rois.shape[0]), "label": label.upper(), "asr": asr, "flag": flag, "split": split}
        out.append(rec)
        print(f"{split:5s} {r['pass']:11s} {rois.shape[0]:3d}f  '{label}'" + (f"   [{flag}]" if flag else ""))
    with open(os.path.join(sdir, "prep.jsonl"), "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    n_tr = sum(r["split"] == "train" for r in out); n_te = len(out) - n_tr
    print(f"\nprepared {len(out)} takes: {n_tr} train, {n_te} test  -> {os.path.join(sdir, 'prep.jsonl')}")


if __name__ == "__main__":
    main()
