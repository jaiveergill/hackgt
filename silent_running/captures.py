"""Shared, git-friendly log of every live utterance: what the camera saw, what the system decided, and what the person
actually said (corrections from the UI). It is the dataset the project was missing: silent mouthing, from our cameras.
Not to be confused with silent_running/sessionlog.py (logs/, gitignored): that one diagnoses lag and stalls per run; this
one is curated recognition data (clips + labels) that the team commits and shares.

Layout (commit it: `git add data/captures && git push`; teammates' sessions never conflict):
  data/captures/<session>.jsonl            one file per server run; one JSON object per line, append-only
  data/captures/clips/<session>/<utt>.mkv  the 96x96 grayscale mouth crops the model read, 25 fps, lossless (bit-exact)

Line types: "session" (first line: host, code version, camera, inventory, scoring settings), "utterance" (clip, source,
frames), and the UI events as broadcast: "result" (ranking trimmed to the top 10), "decision", "confirm", then "label"
(the corrected text). scripts/captures.py reads them back, re-scores labelled clips with the current code, and reports.
Only camera sources (webcam/usb/stream/serial, e.g. the ESP32-CAM) are captured by default; SR_CAPTURE=all adds live file playback (e.g. a phone recording
replayed with --source file:...); SR_CAPTURE=0 turns capture off. /api/decode_file is never captured.
"""
import datetime, json, os, socket, subprocess, threading

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(ROOT, "data", "captures")
EVENT_TYPES = {"result", "decision", "confirm"}
TOP = 10  # ranking rows kept per result (the inventory can have 200+)


def _git_commit():
    r = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() or None


class CaptureLog:
    def __init__(self, meta):
        host = socket.gethostname().split(".")[0].lower().replace(" ", "-")
        self.session = f"{datetime.datetime.now():%Y%m%d-%H%M%S}_{host}"
        self.path = os.path.join(DIR, self.session + ".jsonl")
        self.clip_dir = os.path.join(DIR, "clips", self.session)
        self.lock = threading.Lock()
        self.utts = set()  # captured utt_ids: events of other utterances (decode_file, tests) are not logged
        # written with the first utterance: a run that captures nothing (smoke tests, decode_file) leaves no file to commit
        self.header = {"type": "session", "session": self.session, "host": host, "commit": _git_commit(), **meta}

    def _write(self, rec):
        rec = {"ts": round(datetime.datetime.now().timestamp(), 3), **rec}
        with self.lock, open(self.path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    def utterance(self, utt_id, rois, source):
        """Records the crops the model is about to read. The clip is encoded on a thread: capture adds no decode latency."""
        if not self.utts:
            os.makedirs(self.clip_dir, exist_ok=True)
            self._write(self.header)
        self.utts.add(utt_id)
        rel = os.path.relpath(os.path.join(self.clip_dir, f"{utt_id:05d}.mkv"), DIR)
        self._write({"type": "utterance", "utt_id": utt_id, "clip": rel, "frames": int(len(rois)), "source": source})
        threading.Thread(target=self._save_clip, args=(os.path.join(DIR, rel), np.ascontiguousarray(rois, dtype=np.uint8)), daemon=True).start()

    def _save_clip(self, path, rois):
        t, h, w = rois.shape
        r = subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}", "-r", "25",
                            "-i", "-", "-c:v", "libx264", "-qp", "0", "-pix_fmt", "gray", path], input=rois.tobytes(), capture_output=True)
        if r.returncode:
            self._write({"type": "error", "message": f"clip not saved ({path}): {r.stderr.decode(errors='replace').strip()}"})

    def event(self, msg):
        if msg.get("type") not in EVENT_TYPES or msg.get("utt_id") not in self.utts:
            return
        if msg["type"] == "result" and msg.get("ranking"):
            msg = {**msg, "ranking": msg["ranking"][:TOP]}
        self._write(msg)

    def label(self, utt_id, text, by):
        if int(utt_id) not in self.utts:
            raise KeyError(f"utterance {utt_id} was not captured in this session")
        self._write({"type": "label", "utt_id": int(utt_id), "text": text, "by": by})


def read_clip(path):
    """Mouth crops (T, 96, 96) uint8 from a captured clip, bit-exact."""
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=width,height", "-of", "csv=p=0", path],
                       capture_output=True, text=True, check=True)
    w, h = map(int, r.stdout.strip().split(","))
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w)


def load_sessions(root=DIR):
    """{session: {"meta": {...}, "utts": {utt_id: {"utterance", "result", "decision", "confirm": [...], "label"}}}}"""
    out = {}
    for fn in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        if not fn.endswith(".jsonl"):
            continue
        sess = {"meta": None, "utts": {}}
        for line in open(os.path.join(root, fn)):
            rec = json.loads(line)
            if rec["type"] == "session":
                sess["meta"] = rec
                continue
            if "utt_id" not in rec:
                continue
            u = sess["utts"].setdefault(rec["utt_id"], {"confirm": []})
            if rec["type"] == "confirm":
                u["confirm"].append(rec)
            else:
                u[rec["type"]] = rec  # a later label replaces an earlier one
        out[fn[:-len(".jsonl")]] = sess
    return out
