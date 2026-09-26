"""The shared capture log (data/captures, see silent_running/captures.py): statistics, and re-scoring every labelled clip
with the current code, so a change is judged on our own silent mouthing from our own cameras before it merges.

    python scripts/captures.py stats              sessions, utterances, labels; accuracy at the time; top confusions
    python scripts/captures.py rescore            re-read labelled clips with this checkout's code: fixed / broken / same
    SR_ILM_WEIGHT=0.3 python scripts/captures.py rescore     the same with other scoring settings

rescore loads the lip-reading model, so it holds the machine-wide lock like smoke.py.
"""
import argparse, collections, fcntl, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from silent_running import captures


def labelled(sessions):
    """[(session, utt_id, label, what the system said then, clip, its lips-only pick then)] per corrected utterance."""
    out = []
    for name, s in sessions.items():
        for uid, u in sorted(s["utts"].items()):
            if "label" in u and "utterance" in u:
                r = u.get("result") or {}
                said = (u.get("decision") or {}).get("text") or r.get("selected")
                out.append((name, uid, u["label"]["text"], said, os.path.join(captures.DIR, u["utterance"]["clip"]), r.get("visual_top")))
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
    from silent_running.vsr import VSREngine, load_phrases
    from silent_running.decoder import PhraseDecoder
    from silent_running.context import ContextStore
    rows = labelled(sessions)
    if not rows:
        sys.exit("no labelled utterances in data/captures yet: correct some results in the UI first")
    lock = open(os.path.expanduser("~/.cache/silent_running/smoke.lock"), "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    phrases = load_phrases()
    engine = VSREngine()
    engine.warmup(phrases)
    dec = PhraseDecoder(engine, phrases, ContextStore())  # no context: judges the lips (and scoring) alone
    inv = {p.lower() for p in phrases}
    tally = collections.Counter()
    same = total = 0
    for name, uid, lab, said, clip, lips_then in rows:
        pick = dec.decode(engine.encode(engine.to_model_input(captures.read_clip(clip))))["selected"]
        if lips_then is not None:  # same code and inventory => same lips-only pick: proves the clip round-trips bit-exact
            total += 1; same += pick == lips_then
        if lab.lower() not in inv:
            tally["label not in the phrase list"] += 1
            continue
        # judged against the lips-only pick then (before context), so a context boost is not mistaken for a code change
        was, now = (lips_then or said or "").lower() == lab.lower(), pick.lower() == lab.lower()
        kind = {(False, True): "fixed", (True, False): "BROKEN", (True, True): "right both", (False, False): "wrong both"}[(was, now)]
        tally[kind] += 1
        if was != now:
            print(f"  {kind:6s} {name} #{uid}: actually {lab!r}; then {said!r}, now {pick!r}")
    n = sum(v for k, v in tally.items() if k != "label not in the phrase list")
    right_now = tally["fixed"] + tally["right both"]
    print(f"rescored {n} labelled utterances: right now {right_now}/{n}, then {tally['BROKEN'] + tally['right both']}/{n} | {dict(tally)}")
    print(f"same lips-only pick as logged: {same}/{total} (all, when this checkout matches the logging commit and settings)")
    print("scoring settings:", {k: v for k, v in os.environ.items() if k.startswith("SR_")} or "defaults")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["stats", "rescore"])
    a = ap.parse_args()
    s = captures.load_sessions()
    stats(s) if a.command == "stats" else rescore(s)
