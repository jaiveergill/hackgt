"""Check that the GPU phrase scorer gives the CPU scorer's answers, and time both. Holds the machine-wide lock (loads the model).

  python scripts/check_scorer.py --miracl ~/.cache/silent_running/miracl/full [--enroll 2]

vsr.VSREngine scores the phrase shortlist with a copy of the attention decoder on the GPU (score_device), padded to a few
shapes that warmup() compiled. The CPU decoder it replaces is the same model, so on every MIRACL clip (15 fps, resampled
to 25; inventory = phrases.txt + the MIRACL phrases) this runs PhraseDecoder.decode twice on the same encoding, CPU scorer
then GPU scorer, and compares:
  1. no profile: the selected phrase, the shortlist, max |difference| of the VSR scores and of the posteriors;
  2. with a profile per speaker (enrollment evidence, PR #5: takes 1..K of every phrase enrolled, the later takes decoded),
     so enrolled phrases outside the shortlist (`always=`) go through the GPU scorer too.
Timing: PhraseDecoder.decode per clip on each scorer (median / p90 / max), and first-call stalls after warmup (a first
call on a new shape that takes over 40 ms longer than repeating it). Prints expected vs actual; exit 1 on a failure.
"""
import argparse, fcntl, os, sys, tempfile, time, shutil
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from silent_running.vsr import VSREngine, load_phrases
from silent_running import enroll
from silent_running.enroll import Profile
from silent_running.decoder import PhraseDecoder
from silent_running.context import ContextStore
from scripts.eval_enroll import miracl_clips, encode_clip

TOL = 1e-3  # nats; float rounding between devices and padded shapes, far below any score gap that decides a phrase
failures = []


def check(ok, what, expected, actual):
    print(f"{'PASS' if ok else 'FAIL'} {what}: expected {expected}, got {actual}", flush=True)
    if not ok:
        failures.append(what)


class Scorer:
    """Switch the engine's phrase scorer between the CPU decoder (what main used) and the GPU copy."""
    def __init__(self, engine):
        self.engine, self.gpu = engine, (engine.score_device, engine.score_decoder)

    def use(self, which):
        e = self.engine
        e.score_device, e.score_decoder = self.gpu if which == "gpu" else (e.decode_device, e.model.decoder)


def decode_both(dec, scorer, enc):
    out, dt = {}, {}
    for which in ("cpu", "gpu"):
        scorer.use(which)
        t = time.time(); out[which] = dec.decode(enc); dt[which] = time.time() - t
    scorer.use("gpu")
    return out, dt


def compare(res, what, n_same, diffs):
    c, g = res["cpu"], res["gpu"]
    rc = {r["phrase"]: r for r in c["ranking"]}
    rg = {r["phrase"]: r for r in g["ranking"]}
    short_c = {p for p, r in rc.items() if r["att"] is not None}
    short_g = {p for p, r in rg.items() if r["att"] is not None}
    n_same[0] += c["selected"] == g["selected"]
    n_same[1] += short_c == short_g
    diffs["score"].append(max(abs(rc[p]["vsr_score"] - rg[p]["vsr_score"]) for p in short_c & short_g))
    diffs["prob"].append(max(abs(rc[p]["final_prob"] - rg[p]["final_prob"]) for p in rc))
    if c["selected"] != g["selected"]:
        print(f"  {what}: CPU {c['selected']!r} {c['confidence']:.3f} vs GPU {g['selected']!r} {g['confidence']:.3f}")


def report(label, n, n_same, diffs, times):
    check(n_same[0] == n, f"{label}: same phrase selected", f"{n}/{n}", f"{n_same[0]}/{n}")
    check(n_same[1] == n, f"{label}: same attention shortlist", f"{n}/{n}", f"{n_same[1]}/{n}")
    check(max(diffs["score"]) < TOL, f"{label}: max |VSR score difference| on the shortlist", f"< {TOL} nats", f"{max(diffs['score']):.2e}")
    print(f"  max |posterior difference| {max(diffs['prob']):.2e}")
    for which in ("cpu", "gpu"):
        t = np.array(times[which]) * 1000
        print(f"  decode on {which.upper()} scorer: median {np.median(t):.0f} ms, p90 {np.percentile(t, 90):.0f}, max {t.max():.0f} (n={len(t)})")


def stalls(engine, inventory):
    """First call vs repeat on encode lengths and phrase sets warmup() never used."""
    import torch, random
    random.seed(0)
    words = "WE ARE GOING TO THE STORE TODAY AND THEN I DON'T KNOW WHAT TO DO ABOUT IT BECAUSE THE WEATHER IS BAD".split()
    bad, n = [], 0
    for T in range(11, 170, 5):
        enc = torch.randn(T, 768)
        for c in ([" ".join(words[:random.randint(1, len(words))])], random.sample(inventory, 48), random.sample(inventory, 50)):
            a = time.time(); engine._score_phrases_full(enc, c); a = time.time() - a
            b = time.time(); engine._score_phrases_full(enc, c); b = time.time() - b
            n += 1
            if a > b + 0.04:
                bad.append((T, len(c), round(a * 1000), round(b * 1000)))
    check(not bad, "no first-call stall after warmup (first call > repeat + 40 ms)", f"0/{n}", f"{len(bad)}/{n} {bad}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--miracl", required=True)
    ap.add_argument("--enroll", type=int, default=2, help="enrolled takes per phrase for the profile part")
    a = ap.parse_args()
    lock = open(os.path.expanduser("~/.cache/silent_running/smoke.lock"), "w")  # same machine-wide lock as smoke.py (8 GB RAM)
    print("waiting for the machine-wide lock ...", flush=True)
    fcntl.flock(lock, fcntl.LOCK_EX)
    print(f"lock acquired, load {os.getloadavg()[0]:.1f}", flush=True)
    enroll.PROFILE_DIR = tempfile.mkdtemp()
    clips, miracl = miracl_clips(os.path.expanduser(a.miracl))
    inv204 = load_phrases()
    inventory = inv204 + [p for p in miracl if p.lower() not in {q.lower() for q in inv204}]
    engine = VSREngine()
    t = time.time(); engine.warmup(); print(f"warmup {time.time() - t:.1f}s, score device {engine.score_device}", flush=True)
    check(engine.score_device.type == "mps", "phrase scorer on the GPU", "mps", engine.score_device.type)
    scorer = Scorer(engine)
    dec = PhraseDecoder(engine, inventory, ContextStore())
    encs = {c["path"]: encode_clip(engine, c["path"]) for c in clips}
    clips = [c for c in clips if encs[c["path"]] is not None]

    print(f"\n1. no profile, {len(clips)} clips, {len(inventory)} phrases")
    n_same, diffs, times, top1 = [0, 0], {"score": [], "prob": []}, {"cpu": [], "gpu": []}, {"cpu": 0, "gpu": 0}
    for c in clips:
        res, dt = decode_both(dec, scorer, encs[c["path"]])
        compare(res, os.path.relpath(c["path"], os.path.expanduser(a.miracl)), n_same, diffs)
        for w in dt:
            times[w].append(dt[w]); top1[w] += res[w]["selected"] == c["phrase"]
    report("no profile", len(clips), n_same, diffs, times)
    print(f"  top-1: CPU {top1['cpu']}/{len(clips)}, GPU {top1['gpu']}/{len(clips)}")

    print(f"\n2. with a profile per speaker (takes 1-{a.enroll} of every phrase enrolled), later takes decoded")
    n_same, diffs, times, n, n_always = [0, 0], {"score": [], "prob": []}, {"cpu": [], "gpu": []}, 0, 0
    for spk in sorted({c["speaker"] for c in clips}):
        prof = Profile(f"check_scorer_{spk}")
        for c in clips:
            if c["speaker"] == spk and c["take"] <= a.enroll:
                prof.add(c["phrase"], encs[c["path"]])
        dec.profile = prof
        for c in clips:
            if c["speaker"] != spk or c["take"] <= a.enroll:
                continue
            res, dt = decode_both(dec, scorer, encs[c["path"]])
            compare(res, os.path.relpath(c["path"], os.path.expanduser(a.miracl)), n_same, diffs)
            short = [r for r in res["gpu"]["ranking"] if r["att"] is not None]
            n_always += len(short) > 48
            for w in dt:
                times[w].append(dt[w])
            n += 1
    dec.profile = None
    report("with a profile", n, n_same, diffs, times)
    check(n_always > 0, "enrolled phrases outside the CTC shortlist were rescored (always=)", "> 0 clips", f"{n_always}/{n} clips")

    print("\n3. first-call stalls")
    stalls(engine, inventory)
    shutil.rmtree(enroll.PROFILE_DIR)
    print(f"\nload {os.getloadavg()[0]:.1f}")
    print("CHECK_SCORER " + ("PASSED" if not failures else f"FAILED: {failures}"))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
