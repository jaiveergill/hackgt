"""Check that the capture process's IncrementalCropper gives exactly the crops emit() used to compute in one batch when an
utterance ended (resample_25fps + VideoProcess over the frames in [start, end]), and time both. No VSR model or lock.

  python scripts/check_cropper.py --miracl ~/.cache/silent_running/miracl/full [--clips data/samples/*.mp4]

Every clip is fed frame by frame at its native frame rate (MIRACL is 15 fps, a webcam 30: the 25 fps resampling is part of
what is compared), with a few ms of timestamp jitter like a live camera, and the camera's own landmarker. Each clip is
one utterance, cropped five ways:
  manual      starts after the first 2 frames and ends at the last frame (safe = the newest frame, as while holding Listen)
  auto        ends POST_ROLL after a mouth that went still HANG before the last frame: the cropper has already seen and
              assigned frames past the end, as when the hands-free hang runs out
  gaps        manual with the landmarks of frames 0-1, 6-10 and the last 3 removed: leading fill, interpolation, trailing fill
  no face     manual with no landmarks at all: the same "face not tracked" error
  too short   manual over the last 7 frames: the same "too short" error
Expected: identical crops (np.array_equal) or the same error message, the same n_face / n_total / duration.
"""
import argparse, glob, os, sys, time
import numpy as np, av

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from silent_running.camera_proc import IncrementalCropper, VideoProcess, resample_25fps, AUTO_HANG
from silent_running.expression import FaceLandmarker

POST_ROLL = 0.15  # camera_proc._worker


def batch_emit(vp, items, start, end):
    """What emit() computed before the cropper (origin/main camera_proc.py), as (rois, n_face, n_total, duration, error)."""
    items = [it for it in items if start <= it[0] <= end]
    if len(items) < 8:
        return None, 0, len(items), 0.0, "too short"
    frames, lms, dur = resample_25fps(items)
    n_face = sum(l is not None for l in lms)
    if n_face < max(4, len(lms) // 4):
        return None, n_face, len(lms), dur, "face not tracked"
    try:
        return vp(frames, list(lms)), n_face, len(lms), dur, None
    except Exception as e:
        return None, n_face, len(lms), dur, f"crop failed: {e}"


def incremental(vp, items, start, end, safe_fn):
    c = IncrementalCropper(vp, start, max_backlog=700)
    t_feed = []
    for ts, rgb, lm in items:
        t = time.perf_counter()
        c.feed(ts, rgb, lm)
        c.advance(safe_fn(ts))
        t_feed.append(time.perf_counter() - t)
    t = time.perf_counter()
    r = c.finish(end)
    return r, time.perf_counter() - t, t_feed


def clip_items(path, fl, rng):
    with av.open(path) as c:
        fps = float(c.streams.video[0].average_rate)
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    t0 = 1000.0 + rng.uniform(0, 1)
    ts = t0 + np.arange(len(frames)) / fps + rng.uniform(-0.003, 0.003, len(frames))
    fl.t_ms = 0
    lms = [fl(np.ascontiguousarray(f), (t - t0) * 1000.0 + 1)[0] for f, t in zip(frames, ts)]
    return [(float(t), f, l) for t, f, l in zip(ts, frames, lms)], fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--miracl")
    ap.add_argument("--clips", nargs="*", default=[])
    a = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(os.path.expanduser(a.miracl), "*/*/*.mp4"))) if a.miracl else []
    paths += a.clips
    if not paths:
        ap.error("no clips: pass --miracl DIR and/or --clips FILE...")
    vp = VideoProcess(convert_gray=True)
    rng = np.random.default_rng(0)
    tally, fails, t_batch, t_finish, t_frame = {}, [], [], [], []
    for p in paths:
        fl = FaceLandmarker()  # VIDEO mode: a fresh landmarker per clip (timestamps restart)
        items, fps = clip_items(p, fl, rng)
        first, last = items[0][0], items[-1][0]
        q = last - AUTO_HANG  # the mouth went still here; the hang runs out at the last frame
        gaps = [(t, f, None if k < 2 or 6 <= k <= 10 or k >= len(items) - 3 else l) for k, (t, f, l) in enumerate(items)]
        cases = {"manual": (items, items[2][0], last + 0.001, lambda ts: ts),
                 "auto": (items, first, q + POST_ROLL, lambda ts: ts if ts < q else min(q + POST_ROLL, ts)),
                 "gaps": (gaps, first, last + 0.001, lambda ts: ts),
                 "no face": ([(t, f, None) for t, f, _ in items], first, last + 0.001, lambda ts: ts),
                 "too short": (items, items[max(len(items) - 7, 0)][0], last + 0.001, lambda ts: ts)}
        for name, (its, start, end, safe_fn) in cases.items():
            t = time.perf_counter(); ref = batch_emit(vp, its, start, end); tb = time.perf_counter() - t
            got, tf, tfeed = incremental(vp, its, start, end, safe_fn)
            same = ref[1:] == got[1:] and (ref[0] is None) == (got[0] is None) and (ref[0] is None or np.array_equal(ref[0], got[0]))
            k = tally.setdefault(name, [0, 0, 0])
            k[0] += 1; k[1] += same; k[2] += ref[0] is not None
            if not same:
                diff = float(np.abs(ref[0].astype(int) - got[0].astype(int)).max()) if ref[0] is not None and got[0] is not None and ref[0].shape == got[0].shape else None
                fails.append(f"{'/'.join(p.split('/')[-3:])} [{name}] expected {ref[1:]} shape {None if ref[0] is None else ref[0].shape}, got {got[1:]} "
                             f"shape {None if got[0] is None else got[0].shape}, max pixel diff {diff}")
            if name == "manual" and ref[0] is not None:
                t_batch.append(tb); t_finish.append(tf); t_frame += tfeed
        print(f"{'/'.join(p.split('/')[-3:])} {fps:.0f} fps, identical so far: " + " ".join(f"{n}={v[1]}/{v[0]}" for n, v in tally.items()), flush=True)
    print()
    for f in fails:
        print("MISMATCH", f)
    for name, (n, same, cropped) in tally.items():
        print(f"{'PASS' if same == n else 'FAIL'} {name}: expected {n}/{n} identical, got {same}/{n} ({cropped} cropped, {n - cropped} errors)")
    tb, tf, tfr = np.array(t_batch) * 1000, np.array(t_finish) * 1000, np.array(t_frame) * 1000
    print(f"batch crop at the end (old emit): median {np.median(tb):.1f} ms, p90 {np.percentile(tb, 90):.1f}, max {tb.max():.1f}")
    print(f"cropper finish() at the end:      median {np.median(tf):.1f} ms, p90 {np.percentile(tf, 90):.1f}, max {tf.max():.1f}")
    print(f"cropper per frame while mouthing: mean {tfr.mean():.2f} ms, p99 {np.percentile(tfr, 99):.2f}, max {tfr.max():.1f}")
    ok = not fails
    print("CHECK_CROPPER " + ("PASSED" if ok else "FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
