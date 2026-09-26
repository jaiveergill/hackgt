"""Check patient enrollment end to end on one speaker's recorded clips. Holds the machine-wide lock (loads the model).

  python scripts/check_enroll.py --miracl ~/.cache/silent_running/miracl/full --speaker F01 --enroll 2

Part 1, accuracy through the server's path (PhraseDecoder + a real Profile): enroll takes 1..K of every phrase, then of
half the phrases, and print for each later take the expected phrase vs the generic and the enrolled top-1. Expect
enrolled >= generic overall, and on the phrases that were NOT enrolled (a phrase without takes must not be penalised).
Part 2, the enrollment flow through the real server.py functions (events captured in-process):
  continuing a complete profile or reps=0 is refused and live utterances are still decoded; the UI state shows the
  prompt; a take that contradicts the phrase's earlier take is rejected; undo re-prompts; starting enrollment cancels a
  pending confirmation and takes never produce a decision or confirmation; a profile switch requested mid-decode waits
  for that decode. Prints PASS/FAIL with expected vs actual; exit 1 on any failure.
"""
import argparse, fcntl, os, sys, threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from silent_running.vsr import VSREngine
from silent_running.enroll import Profile
from silent_running.decoder import PhraseDecoder
from silent_running.context import ContextStore
from silent_running import confirm as confirm_mod
from scripts.eval_enroll import miracl_clips, manifest_clips, encode_clip, frames_25fps

failures = []


def check(ok, what, expected, actual):
    print(f"{'PASS' if ok else 'FAIL'} {what}: expected {expected}, got {actual}")
    if not ok:
        failures.append(what)


def accuracy(engine, clips, inventory, k):
    dec = PhraseDecoder(engine, inventory, ContextStore())
    for label, enrolled in (("all phrases enrolled", inventory), ("half the phrases enrolled", inventory[:len(inventory) // 2])):
        prof = Profile("check_accuracy")
        for c in clips:
            if c["take"] <= k and c["phrase"] in enrolled:
                prof.add(c["phrase"], encode_clip(engine, c["path"]))
        print(f"\n{label}: {len(prof.phrases)} takes of {len(prof.counts())} phrases, impostor mean {prof.mu:.3f}")
        tally = {True: [0, 0, 0], False: [0, 0, 0]}  # enrolled phrase? -> [n, generic ok, enrolled ok]
        for c in clips:
            if c["take"] <= k:
                continue
            enc = encode_clip(engine, c["path"])
            dec.profile = None
            generic = dec.decode(enc)["selected"]
            dec.profile = prof
            pd = dec.decode(enc)
            ev = next(r["proto"] for r in pd["ranking"] if r["phrase"] == c["phrase"])
            t = tally[c["phrase"] in enrolled]
            t[0] += 1; t[1] += generic == c["phrase"]; t[2] += pd["selected"] == c["phrase"]
            print(f"take {c['take']}  expected {c['phrase']!r:20} generic {generic!r:20} {'OK ' if generic == c['phrase'] else 'BAD'}  "
                  f"enrolled {pd['selected']!r:20} {'OK ' if pd['selected'] == c['phrase'] else 'BAD'} conf {pd['confidence']:.2f}  evidence(expected) {ev:+.1f}")
        n, g, e = (sum(tally[b][i] for b in tally) for i in range(3))
        print(f"top-1 generic {g}/{n}, enrolled {e}/{n}")
        check(e >= g, f"{label}: enrolled top-1 >= generic", f">= {g}/{n}", f"{e}/{n}")
        if tally[False][0]:
            n, g, e = tally[False]
            check(e >= g, f"{label}: phrases without takes are not penalised", f">= {g}/{n} (generic)", f"{e}/{n}")


def utterance(engine, path):
    frames = frames_25fps(path)
    lms = engine.landmarks_for_frames(frames)
    return {"rois": engine.mouth_rois(frames, lms), "n_face": sum(l is not None for l in lms), "n_total": len(lms),
            "duration": len(frames) / 25.0, "t_crop": 0.0}


def server_flow(engine, clips, inventory):
    import silent_running.server as S
    events = []
    S.broadcast = events.append
    S.engine, S.phrases, S.context = engine, inventory, ContextStore()
    S.phrase_decoder = PhraseDecoder(engine, inventory, S.context)
    S.confirm_loop = confirm_mod.ConfirmLoop(S._confirm_emit, S._on_confirmed)
    names = ("check_enroll_full", "check_enroll_new")
    for n in names:
        if os.path.exists(Profile(n).path):
            os.remove(Profile(n).path)
    take = lambda phrase, t: next(c["path"] for c in clips if c["phrase"] == phrase and c["take"] == t)
    types = lambda: [e["type"] for e in events]

    # B3: nothing left to enroll -> refused, and live utterances keep being decoded
    full = Profile(names[0])
    for p in inventory:
        for t in (1, 2):
            full.add(p, encode_clip(engine, take(p, t)))
    full.save()
    r = S.api_enroll_start(profile=names[0], reps=2)
    check(getattr(r, "status_code", 200) == 400 and S.ENROLL["session"] is None, "continue a complete profile",
          "400, no session", f"{getattr(r, 'status_code', 200)} {r.body.decode() if hasattr(r, 'body') else r}, session {S.ENROLL['session']}")
    r = S.api_enroll_start(profile=names[1], reps=0)
    check(getattr(r, "status_code", 200) == 400 and S.ENROLL["session"] is None, "reps=0", "400, no session", f"{getattr(r, 'status_code', 200)}")
    events.clear()
    S._decode_utterance(utterance(engine, take(inventory[0], 5)))
    check("result" in types(), "live utterance after the refused starts", "a result event", types())

    # a pending confirmation is cancelled by starting enrollment
    S.confirm_loop.start({"utt_id": 999, "alternatives": [{"text": inventory[3], "confidence": 0.4}]})
    events.clear()
    r = S.api_enroll_start(profile=names[1], reps=2)
    check(not S.confirm_loop.pending() and any(e["type"] == "confirm" and e["state"] == "rejected" for e in events),
          "starting enrollment cancels the open question", "no pending question, a rejected confirm event", [(e["type"], e.get("state")) for e in events])
    check(S.STATE["status"] == "enrolling" and S.STATE["enroll"]["prompt"] == inventory[0], "UI state while enrolling",
          f"status enrolling, prompt {inventory[0]!r}", f"{S.STATE['status']}, {S.STATE['enroll'] and S.STATE['enroll']['prompt']!r}")

    # round 1: one take of every phrase, in prompt order, through the live-utterance path. A take the server rejects
    # is mouthed again (the next recorded take), as the patient would after seeing the error.
    events.clear()
    used = {}
    for p in inventory:
        for t in range(1, 6):
            before = S.ENROLL["session"].done
            S._decode_utterance(utterance(engine, take(p, t)))
            if S.ENROLL["session"].done > before:
                used[p] = t
                break
    stored = [e["stored"] for e in events if e["type"] == "enroll" and e["state"] == "take"]
    errs = [e["message"] for e in events if e["type"] == "error"]
    check(stored == list(inventory) and all("no mouth movement" in m for m in errs), "round 1: one take per phrase, in prompt order",
          "every phrase stored; every retry explained by an error event", f"stored {len(stored)}, takes used {used}, errors {errs}")
    check(not {"result", "decision", "confirm"} & set(types()), "enrollment takes never decode, decide or confirm",
          "no result/decision/confirm events", sorted(set(types())))
    S._on_signal({"kind": "nod", "value": None, "confidence": 0.9})
    check("confirm" not in types(), "a nod while enrolling answers nothing", "no confirm event", sorted(set(types())))

    # round 2: prompt = inventory[0], which has one take. Unrelated speech and another phrase's take are rejected.
    sess = S.ENROLL["session"]
    other = next(t for t in range(1, 6) if t != used[inventory[1]])
    for label, path in (("unrelated speech (TED clip)", os.path.join(ROOT, "data/samples/ted1_short.mp4")), (f"a take of {inventory[1]!r}", take(inventory[1], other))):
        events.clear()
        S._decode_utterance(utterance(engine, path))
        errs = [e["message"] for e in events if e["type"] == "error"]
        check(bool(errs) and "does not match" in errs[0] and sess.profile.counts()[inventory[0]] == 1 and sess.prompt == inventory[0],
              f"{label} as a take of {inventory[0]!r}", "rejected, not stored, same prompt", f"{errs[:1]} counts {sess.profile.counts()[inventory[0]]} prompt {sess.prompt!r}")
    events.clear()
    genuine = next(t for t in range(used[inventory[0]] + 1, 6))
    S._decode_utterance(utterance(engine, take(inventory[0], genuine)))
    check(sess.profile.counts()[inventory[0]] == 2, f"genuine take {genuine} of {inventory[0]!r}", "stored (2 takes)",
          f"{sess.profile.counts()[inventory[0]]} {[e.get('message') for e in events if e['type'] == 'error']}")
    r = S.api_enroll_undo()
    check(r["removed"] == inventory[0] and sess.profile.counts()[inventory[0]] == 1 and sess.prompt == inventory[0], "undo",
          f"removes the take of {inventory[0]!r} and prompts it again", f"removed {r['removed']!r}, counts {sess.profile.counts()[inventory[0]]}, prompt {sess.prompt!r}")
    r = S.api_enroll_stop()
    check(r["activated"] and S.STATE["profile"] == names[1] and S.STATE["enroll"] is None and S.STATE["status"] == "idle",
          "stop activates the profile", f"active {names[1]!r}, idle", f"active {S.STATE['profile']!r}, {S.STATE['status']}")

    # a profile switch requested in the middle of a decode waits for it
    orig, seen = S.phrase_decoder.decode, {}
    def decode_with_switch(enc):
        th = threading.Thread(target=lambda: seen.update(resp=S.api_profile(name="")))
        th.start(); th.join(1.0)
        seen["blocked"], seen["thread"] = th.is_alive(), th
        return orig(enc)
    S.phrase_decoder.decode = decode_with_switch
    res = S.run_decode(**utterance(engine, take(inventory[2], 5)))
    seen["thread"].join()
    S.phrase_decoder.decode = orig
    used = any(r["proto"] != 0.0 for r in res["ranking"])
    check(seen["blocked"] and res["profile"] == names[1] and used and S.STATE["profile"] is None, "profile switch during a decode",
          f"switch waits; result labelled {names[1]!r} and scored with it; generic afterwards",
          f"waited {seen['blocked']}, labelled {res['profile']!r}, evidence used {used}, active after {S.STATE['profile']!r}")
    for n in names:
        os.remove(Profile(n).path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--miracl")
    ap.add_argument("--manifest")
    ap.add_argument("--speaker", required=True)
    ap.add_argument("--enroll", type=int, default=2, help="enrollment takes per phrase in part 1 (tests use the later takes)")
    a = ap.parse_args()
    lock = open(os.path.expanduser("~/.cache/silent_running/smoke.lock"), "w")  # same machine-wide lock as smoke.py (8 GB RAM)
    print("waiting for the machine-wide lock ...", flush=True)
    fcntl.flock(lock, fcntl.LOCK_EX)
    clips, inventory = miracl_clips(os.path.expanduser(a.miracl)) if a.miracl else manifest_clips(a.manifest)
    clips = [c for c in clips if c["speaker"] == a.speaker]
    engine = VSREngine()
    engine.warmup()
    accuracy(engine, clips, inventory, a.enroll)
    print()
    server_flow(engine, clips, inventory)
    print("\nCHECK_ENROLL " + ("PASSED" if not failures else f"FAILED: {failures}"))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
