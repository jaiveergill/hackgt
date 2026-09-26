"""How long must the mouth stay still before hands-free listening ends the utterance (camera_proc.AUTO_HANG)? A short hang
answers sooner, but a pause inside a phrase longer than the hang ends the utterance there. For each hang this runs the
real capture worker (camera_proc._worker: face landmarker, mouth-motion energy, auto-listen, cropper) over recorded clips,
each played as one utterance between stretches of still face, and counts per clip (phrase = first to last clip frame):

  split      2+ utterances overlap the phrase: a pause inside it ended one (pieces shorter than MIN_UTT, which are dropped,
             count too)
  tail lost  one utterance overlaps it, and it ended more than 0.2 s before the phrase's last frame
  missed     no utterance overlaps it
  ok         otherwise
  outside    utterances in the still face around the phrase (false triggers; the jump from the previous clip's face
             to this one's, and the landmarker settling after it, set them off)
plus the delay from the phrase's last frame to the moment its utterance ended (the hang's share of the latency), and the
longest pause inside the phrase: the mouth went still (motion below the off threshold) and moved again before the
utterance ended. Pauses under POST_ROLL (0.15 s) are not visible; a pause longer than the hang ends the utterance, so run
the longest hang to see them all.

  python scripts/eval_hang.py --miracl ~/.cache/silent_running/miracl/full --hangs 0.35 0.45 0.6 [--clips data/samples/ted*.mp4]

MIRACL clips are 15 fps and resampled to 25 (repeated frames, as the camera buffer would), the others are played at 25 fps.
The still stretches are the clip's first / last frame repeated (1.5 s before, 3 s after): no sensor noise, but the face
landmarker's smoothing still settles for a few frames after motion stops, as it does live.
No VSR model or lock needed. SR_HANDS=0 (hand signals don't take part in segmentation).
"""
import argparse, glob, json, os, sys, threading, time, statistics as st
import multiprocessing as mp
import numpy as np, av, cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["SR_HANDS"] = "0"
from silent_running import camera_proc, sources

FPS, PRE, POST = 25.0, 1.5, 3.0
PRE_ROLL, POST_ROLL = 0.30, 0.15  # camera_proc._worker: an utterance starts PRE_ROLL before the motion, ends POST_ROLL after it
MIRACL = ["Stop navigation", "Excuse me", "I am sorry", "Thank you", "Good bye", "I love this game", "Nice to meet you",
          "You are welcome", "How are you", "Have a good time"]


def frames_25(path):
    with av.open(path) as c:
        fps = float(c.streams.video[0].average_rate)
        fr = [f.to_ndarray(format="bgr24") for f in c.decode(video=0)]
    idx = np.round(np.arange(int(round(len(fr) * FPS / fps))) * fps / FPS).astype(int).clip(0, len(fr) - 1)
    return [fr[i] for i in idx]


class Playlist(sources.VideoSource):
    """Clips back to back, each between still stretches, stamped with media time (like a file source with realtime=0).
    Records the time of each clip's first and last frame."""
    kind = "file"

    def __init__(self, paths):
        self.paths, self.clips, self.done = paths, [], threading.Event()

    def open(self):
        self.gen, self.i, self.t0 = self._frames(), 0, time.time()  # the landmarker needs times on the worker's (wall) clock

    def _frames(self):
        for p in self.paths:
            fr = frames_25(p)
            h, w = fr[0].shape[:2]
            if w > 960:  # FileSource's max_width
                fr = [cv2.resize(f, (960, int(h * 960 / w)), interpolation=cv2.INTER_AREA) for f in fr]
            c = {"path": p}
            self.clips.append(c)
            yield from [fr[0]] * int(PRE * FPS)
            for k, f in enumerate(fr):
                if k == 0:
                    c["t_first"] = self.t
                c["t_last"] = self.t
                yield f
            yield from [fr[-1]] * int(POST * FPS)

    @property
    def t(self):
        return self.t0 + self.i / FPS

    def read(self):
        f = next(self.gen, None)
        if f is None:
            self.finished = True
            self.done.set()
            time.sleep(0.01)
            return False, None, None
        ts = self.t
        self.i += 1
        return True, f, ts


class Recorder(camera_proc.IncrementalCropper):
    """The worker's cropper, recording each utterance it starts: start, end (None = dropped as shorter than MIN_UTT),
    the time it was ended (the newest frame when finish() ran) and its error."""
    log = []

    def __init__(self, vp, start, max_backlog):
        super().__init__(vp, start, max_backlog)
        self.end = self.error_msg = None
        self.trace = []  # (frame time, safe): the worker passes safe = min(quiet_since + POST_ROLL, ts) while the mouth is still
        Recorder.log.append(self)

    def advance(self, safe):
        self.trace.append((self.ts[-1], safe))
        super().advance(safe)

    def finish(self, end):
        r = super().finish(end)
        self.end, self.error_msg = end, r[4]
        return r


def run(paths, hang):
    camera_proc.AUTO_HANG = hang
    camera_proc.IncrementalCropper = Recorder
    Recorder.log = []
    src = Playlist(paths)
    sources.make_source = lambda spec, *a, **k: src  # the worker imports make_source from this module at call time
    parent, child = mp.Pipe()
    threading.Thread(target=camera_proc._worker, args=(child, "playlist", 640, 480, 320, 20), daemon=True).start()
    assert parent.recv()[0] == "opened"
    parent.send(("auto", True))
    while not src.done.is_set():
        while parent.poll(0.05):
            m = parent.recv()
            if m[0] in ("error", "fatal"):
                print("  worker:", m[1], flush=True)
    time.sleep(0.5)
    while parent.poll(0.05):
        parent.recv()
    parent.send(("quit",))
    return src.clips, list(Recorder.log)


def classify(clips, log, hang):
    segs = []
    for u in log:  # u.ts[-1]: the newest frame when the utterance ended (the hang ran out) or was dropped
        end = u.end if u.end is not None else u.ts[-1] - hang + POST_ROLL
        pauses, quiet = [], None
        for ts, safe in u.trace:  # a still stretch that ended with the mouth moving again (not the one that ended the utterance)
            if safe < ts:
                quiet = safe - POST_ROLL
            elif quiet is not None:
                pauses.append((quiet, ts - quiet)); quiet = None
        segs.append({"start": u.start, "end": end, "t_ended": u.ts[-1], "emitted": u.end is not None and u.error_msg is None,
                     "dropped": u.end is None, "error": u.error_msg, "pauses": pauses})
    out = []
    for i, c in enumerate(clips):
        lo = c["t_first"] - PRE + 0.1
        hi = clips[i + 1]["t_first"] - PRE + 0.1 if i + 1 < len(clips) else 1e18
        mine = [u for u in segs if lo <= u["start"] < hi]
        over = [u for u in mine if u["start"] + PRE_ROLL <= c["t_last"] and u["end"] > c["t_first"] + 0.1]
        emitted = [u for u in over if u["emitted"]]
        if not emitted:
            cls = "missed"
        elif len(over) > 1:
            cls = "split"
        elif emitted[0]["end"] < c["t_last"] - 0.2:
            cls = "tail lost"
        else:
            cls = "ok"
        rel = lambda t: round(t - c["t_last"], 3)
        pause = max([d for u in over for t, d in u["pauses"] if c["t_first"] <= t <= c["t_last"]], default=0.0)
        out.append({"path": c["path"], "class": cls, "outside": len(mine) - len(over), "first": rel(c["t_first"]), "pause": round(pause, 3),
                    "delay": emitted[-1]["t_ended"] - c["t_last"] if emitted else None,
                    "errors": [u["error"] for u in mine if u["error"]],
                    "utterances": [(rel(u["start"]), rel(u["end"]), "dropped" if u["dropped"] else "emitted") for u in mine]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--miracl")
    ap.add_argument("--clips", nargs="*", default=[])
    ap.add_argument("--hangs", nargs="+", type=float, default=[0.35, 0.45, 0.6])
    ap.add_argument("--json", default=None, help="write per-clip results here")
    a = ap.parse_args()
    sets = {}
    if a.miracl:
        sets["MIRACL"] = sorted(glob.glob(os.path.join(os.path.expanduser(a.miracl), "*/*/*.mp4")))
    if a.clips:
        sets["samples"] = a.clips
    hangs = sorted(a.hangs)
    res = {}
    for name, paths in sets.items():
        for h in hangs:
            t = time.time()
            clips, log = run(paths, h)
            res[(name, h)] = classify(clips, log, h)
            print(f"{name} hang {h:.2f}: {len(log)} utterances started over {len(clips)} clips ({time.time() - t:.0f} s)", flush=True)
    print("\n| clips | hang (s) | ok | split | tail lost | missed | utterances outside the phrase | errors | phrase end -> utterance end, median (p90) | longest pause inside a phrase |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    dump = {}
    for name in sets:
        for h in hangs:
            rows = res[(name, h)]
            n = {k: sum(r["class"] == k for r in rows) for k in ("ok", "split", "tail lost", "missed")}
            q = sorted(r["delay"] for r in rows if r["delay"] is not None)
            print(f"| {name} ({len(rows)}) | {h:.2f} | {n['ok']} | {n['split']} | {n['tail lost']} | {n['missed']} | {sum(r['outside'] for r in rows)} | "
                  f"{sum(len(r['errors']) for r in rows)} | {st.median(q):.2f} s ({q[int(0.9 * len(q))]:.2f} s) | "
                  f"{max(r['pause'] for r in rows):.2f} s ({sum(r['pause'] > 0 for r in rows)} phrases paused >= {POST_ROLL} s) |")
            dump[f"{name}@{h}"] = [{**r, "path": os.path.relpath(r["path"], ROOT) if r["path"].startswith(ROOT) else r["path"]} for r in rows]
    for name in sets:  # what a shorter hang breaks that the longest one does not
        for h in hangs[:-1]:
            for d, d0 in zip(dump[f"{name}@{h}"], dump[f"{name}@{hangs[-1]}"]):
                if d["class"] != "ok" and d0["class"] == "ok":
                    print(f"  {name} hang {h:.2f} {d['class']}: {'/'.join(d['path'].split('/')[-3:])} phrase from {d['first']} s; utterances "
                          f"(start, end) s rel. to its last frame {d['utterances']}; at {hangs[-1]:.2f}: {d0['utterances']}")
    if a.json:
        json.dump(dump, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
