"""The shared capture log (data/captures, see silent_running/captures.py): statistics, and re-scoring every labelled clip
with the current code, so a change is judged on our own silent mouthing from our own cameras before it merges.

    python scripts/captures.py stats              sessions, utterances, labels; accuracy at the time; top confusions
    python scripts/captures.py rescore            re-read labelled clips with this checkout's code: fixed / broken / same

rescore loads the lip-reading model, so it holds the machine-wide lock like smoke.py.
"""
import argparse, collections, fcntl, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from silent_running import captures


def labelled(sessions):
    """[(session, utt_id, label, what the system said then, clip, the lips' reading then)] per corrected utterance."""
    out = []
    for name, s in sessions.items():
        for uid, u in sorted(s["utts"].items()):
            if "label" in u and "utterance" in u:
                r = u.get("result") or {}
                said = (u.get("llm") or {}).get("corrected") or r.get("selected")
                out.append((name, uid, u["label"]["text"], said, os.path.join(captures.DIR, u["utterance"]["clip"]), r.get("selected")))
    return out


def stats(sessions):
    n_utt = sum(len(s["utts"]) for s in sessions.values())
    rows = labelled(sessions)
    right = sum(said is not None and said.lower() == lab.lower() for _, _, lab, said, *_ in rows)
    print(f"{len(sessions)} sessions, {n_utt} utterances, {len(rows)} labelled; right at the time: {right}/{len(rows)}")
    for name, s in sessions.items():
        m = s["meta"] or {}
        print(f"  {name}: {len(s['utts'])} utterances, commit {m.get('commit')}, source {m.get('source')}")
    conf = collections.Counter((lab, said) for _, _, lab, said, *_ in rows if (said or "").lower() != lab.lower())
    if conf:
        print("top confusions (said -> actually):")
        for (lab, said), n in conf.most_common(10):
            print(f"  {n:3d}  {said!r} -> {lab!r}")


def rescore(sessions):
    """Re-read every labelled clip with this checkout (the beam's top reading, before the LLM) and compare with the label."""
    from silent_running.vsr import VSREngine
    rows = labelled(sessions)
    if not rows:
        sys.exit("no labelled utterances in data/captures yet: correct some results in the UI first")
    lock = open(os.path.expanduser("~/.cache/silent_running/smoke.lock"), "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    engine = VSREngine(beam_size=10)
    engine.warmup()
    same_as = lambda a, b: norm(a) == norm(b)
    tally = collections.Counter()
    same = total = 0
    for name, uid, lab, said, clip, lips_then in rows:
        nbest = engine.beam_search(engine.encode(engine.to_model_input(captures.read_clip(clip))), 5)
        pick = nbest[0]["text"] if nbest else ""
        if lips_then is not None:  # same code => same reading: proves the clip round-trips bit-exact
            total += 1; same += same_as(pick, lips_then)
        # judged against the lips' reading then (before the LLM), so an LLM correction is not mistaken for a code change
        was, now = same_as(lips_then or said or "", lab), same_as(pick, lab)
        kind = {(False, True): "fixed", (True, False): "BROKEN", (True, True): "right both", (False, False): "wrong both"}[(was, now)]
        tally[kind] += 1
        if was != now:
            print(f"  {kind:6s} {name} #{uid}: actually {lab!r}; then {said!r}, now {pick!r}")
    n = sum(tally.values())
    print(f"rescored {n} labelled utterances: right now {tally['fixed'] + tally['right both']}/{n}, then {tally['BROKEN'] + tally['right both']}/{n} | {dict(tally)}")
    print(f"same reading as logged: {same}/{total} (all, when this checkout matches the logging commit)")


def norm(s):
    return " ".join("".join(c for c in s.lower() if c.isalnum() or c in " '").split())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["stats", "rescore"])
    a = ap.parse_args()
    s = captures.load_sessions()
    stats(s) if a.command == "stats" else rescore(s)
