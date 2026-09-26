"""Check patient enrollment on one speaker's recorded clips: enroll their first K takes of every phrase, then print,
for each later take, the expected phrase vs the generic top-1 vs the enrolled top-1 (real engine.score_phrases path).

  python scripts/check_enroll.py --miracl ~/.cache/silent_running/miracl/full --speaker F01 --enroll 2
"""
import argparse, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from silent_running.vsr import VSREngine
from silent_running.enroll import Profile, WEIGHT
from scripts.eval_enroll import miracl_clips, manifest_clips, encode_clip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--miracl")
    ap.add_argument("--manifest")
    ap.add_argument("--speaker", required=True)
    ap.add_argument("--enroll", type=int, default=2)
    ap.add_argument("--weight", type=float, default=WEIGHT)
    a = ap.parse_args()
    clips, inventory = miracl_clips(os.path.expanduser(a.miracl)) if a.miracl else manifest_clips(a.manifest)
    clips = [c for c in clips if c["speaker"] == a.speaker]
    engine = VSREngine()
    prof = Profile(f"check_{a.speaker}", weight=a.weight)
    for c in clips:
        if c["take"] <= a.enroll:
            enc = encode_clip(engine, c["path"])
            if enc is None:
                print("skip enroll (face not tracked):", c["path"]); continue
            prof.add(c["phrase"], enc)
    print(f"enrolled {sum(prof.counts().values())} takes of {len(prof.counts())} phrases, weight {a.weight}")
    n = g_ok = e_ok = 0
    for c in clips:
        if c["take"] <= a.enroll:
            continue
        enc = encode_clip(engine, c["path"])
        if enc is None:
            print("skip test (face not tracked):", c["path"]); continue
        engine.profile = None
        generic = engine.score_phrases(enc, inventory)[0]["phrase"]
        engine.profile = prof
        ranked = engine.score_phrases(enc, inventory)
        enrolled = ranked[0]["phrase"]
        bonus = next(r["proto"] for r in ranked if r["phrase"] == c["phrase"])
        n += 1; g_ok += generic == c["phrase"]; e_ok += enrolled == c["phrase"]
        print(f"take {c['take']}  expected {c['phrase']!r:22} generic {generic!r:22} {'OK ' if generic == c['phrase'] else 'BAD'}  "
              f"enrolled {enrolled!r:22} {'OK ' if enrolled == c['phrase'] else 'BAD'}  bonus(expected) {bonus:+.2f}")
    print(f"top-1 generic {g_ok}/{n}, enrolled {e_ok}/{n}")


if __name__ == "__main__":
    main()
