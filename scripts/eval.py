"""Evaluate Silent Running on recorded silent-speech samples.

Reads data/eval/manifest.jsonl (written by scripts/record_samples.py), runs the full pipeline on each
clip and writes one JSON line per clip to results/<tag>.jsonl with:
  raw (beam top-1, what the app reads before the LLM), nbest, top1_correct, top3_correct (in the beam's top 3), per-stage latency.
Also prints a summary table. Never fakes outputs: whatever the model says is what gets logged.

  python scripts/eval.py --tag baseline --device mps [--speaker name] [--ctc-weight 0.1]
"""
import os, sys, json, time, argparse, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from silent_running.vsr import VSREngine, ROOT
import torch

def norm(s):
    return "".join(c for c in s.lower() if c.isalnum() or c == " ").strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--speaker", default=None)
    ap.add_argument("--manifest", default=os.path.join(ROOT, "data", "eval", "manifest.jsonl"))
    ap.add_argument("--ctc-weight", type=float, default=0.1)
    ap.add_argument("--beam", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.manifest) if l.strip()]
    if args.speaker:
        recs = [r for r in recs if r["speaker"] == args.speaker]
    if args.limit:
        recs = recs[: args.limit]
    eng = VSREngine(device=args.device, beam_size=args.beam, ctc_weight=args.ctc_weight)
    eng.warmup()  # note: live server latency is lower still (GPU keep-warm thread); eval numbers include cold-shape costs
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    out_path = os.path.join(ROOT, "results", f"{args.tag}.jsonl")
    out = open(out_path, "w")
    n = top1 = top3 = 0
    lat = collections.defaultdict(list)
    per_phrase = collections.defaultdict(lambda: [0, 0])
    for r in recs:
        path = os.path.join(ROOT, r["file"])
        if not os.path.exists(path):
            print("missing", path); continue
        t0 = time.time()
        try:
            x, lm, rois = eng.preprocess_video(path)
        except Exception as e:
            print(f"PREPROCESS FAIL {r['file']}: {e}")
            out.write(json.dumps({**r, "error": str(e)}) + "\n"); continue
        t1 = time.time()
        enc = eng.encode(x)
        if args.device != "cpu": torch.mps.synchronize()
        t2 = time.time()
        greedy = eng.ctc_greedy(enc)
        no_speech = not greedy.strip()  # same guard the server uses: empty CTC = mouth did not move like speech
        nbest = eng.beam_search(enc, 5)
        t3 = time.time()
        truth = norm(r["phrase"])
        ranked = [norm(h["text"]) for h in nbest]
        c1 = ranked[0] == truth
        c3 = truth in ranked[:3]
        n += 1; top1 += c1; top3 += c3
        per_phrase[r["phrase"]][0] += c1; per_phrase[r["phrase"]][1] += 1
        lat["preprocess"].append(t1 - t0); lat["encode"].append(t2 - t1); lat["beam"].append(t3 - t2)
        rec = {**r, "n_frames_detected": sum(l is not None for l in lm), "ctc_greedy": greedy, "no_speech": no_speech, "raw": nbest[0]["text"], "nbest": nbest,
               "top1_correct": c1, "top3_correct": c3, "latency": {"preprocess": t1 - t0, "encode": t2 - t1, "beam": t3 - t2}}
        out.write(json.dumps(rec) + "\n"); out.flush()
        mark = ("OK " if c1 else ("t3 " if c3 else "XX ")) + ("[no-speech] " if no_speech else "")
        print(f"{mark} [{r['speaker']}] '{r['phrase']}'  raw='{nbest[0]['text']}' ({nbest[0]['score']:.1f} vs {nbest[1]['score']:.1f} {nbest[1]['text']!r})" if len(nbest) > 1 else f"{mark} [{r['speaker']}] '{r['phrase']}'  raw='{nbest[0]['text']}'")
    out.close()
    if n:
        print(f"\n== {args.tag}: n={n} top1={top1/n:.2%} top3={top3/n:.2%}")
        print("   latency mean s: " + "  ".join(f"{k}={sum(v)/len(v):.2f}" for k, v in lat.items()))
        worst = sorted(per_phrase.items(), key=lambda kv: kv[1][0] / kv[1][1])[:8]
        print("   weakest phrases: " + ", ".join(f"{p} {c}/{t}" for p, (c, t) in worst))
        summary = {"tag": args.tag, "n": n, "top1": top1 / n, "top3": top3 / n,
                   "latency_mean": {k: sum(v) / len(v) for k, v in lat.items()}, "device": args.device, "ctc_weight": args.ctc_weight,
                   "per_phrase": {p: {"correct": c, "total": t} for p, (c, t) in per_phrase.items()}}
        with open(os.path.join(ROOT, "results", "summary.jsonl"), "a") as f:
            f.write(json.dumps(summary) + "\n")
    print("wrote", out_path)

if __name__ == "__main__":
    main()
