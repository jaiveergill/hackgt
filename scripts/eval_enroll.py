"""Patient enrollment eval: Phrase Mode top-1 with the generic model vs with each speaker's own enrolled prototypes.

  python scripts/eval_enroll.py --miracl ~/.cache/silent_running/miracl/full --enroll 1,2,3 --tag miracl
  python scripts/eval_enroll.py --manifest data/eval/manifest.jsonl --enroll 1,2 --tag ours

Protocol, per speaker: the first K takes of every phrase are enrolled, the takes after the largest K are the test set
(identical for every K). The prototype weight is chosen by 2-fold cross-validation over speakers (tuned on one half,
scored on the other), so the reported "enrolled" number never saw its own test speakers during tuning.
A final pass runs the real path (engine.profile = Profile, engine.score_phrases) and checks it agrees with the sweep.
Writes results/enroll_<tag>.md.
"""
import argparse, hashlib, json, os, sys, time
from collections import defaultdict
import av
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from silent_running.vsr import VSREngine, load_phrases, _read_video
from silent_running.enroll import Profile, dtw_similarity

MIRACL_PHRASES = ["Stop navigation", "Excuse me", "I am sorry", "Thank you", "Good bye", "I love this game",
                  "Nice to meet you", "You are welcome", "How are you", "Have a good time"]
CACHE = os.path.expanduser("~/.cache/silent_running/enroll_eval")
WEIGHTS = [0, 2, 5, 10, 15, 20, 30, 50, 80, 150, 1000]


def miracl_clips(d):
    out = []
    for spk in sorted(os.listdir(d)):
        for ph in sorted(os.listdir(os.path.join(d, spk))):
            for f in sorted(os.listdir(os.path.join(d, spk, ph))):
                out.append({"speaker": spk, "phrase": MIRACL_PHRASES[int(ph) - 1], "take": int(f[:-4]), "path": os.path.join(d, spk, ph, f)})
    return out, MIRACL_PHRASES


def manifest_clips(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    out, n = [], defaultdict(int)
    for r in sorted(rows, key=lambda r: r["file"]):
        key = (r["speaker"], r["phrase"])
        n[key] += 1
        out.append({"speaker": r["speaker"], "phrase": r["phrase"], "take": n[key], "path": os.path.join(ROOT, r["file"])})
    return out, load_phrases()


def frames_25fps(path):
    with av.open(path) as c:
        fps = float(c.streams.video[0].average_rate)
    frames = _read_video(path)[0].numpy()
    idx = np.round(np.arange(int(round(len(frames) * 25.0 / fps))) * fps / 25.0).astype(int).clip(0, len(frames) - 1)
    return frames[idx]


def encode_clip(engine, path):
    key = os.path.join(CACHE, hashlib.sha1(f"{os.path.abspath(path)}:{os.path.getmtime(path)}".encode()).hexdigest() + ".pt")
    if os.path.exists(key):
        return torch.load(key, weights_only=True)
    frames = frames_25fps(path)
    lms = engine.landmarks_for_frames(frames)
    n_face = sum(l is not None for l in lms)
    if n_face < max(4, len(lms) // 4):
        return None
    enc = engine.encode(engine.to_model_input(engine.mouth_rois(frames, lms)))
    os.makedirs(CACHE, exist_ok=True)
    torch.save(enc, key)
    return enc


def mean_pool_similarity(query, templates):
    q = torch.nn.functional.normalize(query.float().mean(0), dim=0)
    return torch.stack([torch.nn.functional.normalize(t.float().mean(0), dim=0) @ q for t in templates])


def top1(base, sims, w):
    """base: {phrase: model log-lik}; sims: {phrase: similarity}; same centring as Profile.bonus."""
    mean = sum(sims.values()) / len(sims)
    return max(base, key=lambda p: base[p] + w * (sims[p] - mean))


def confidence(base, sims, w):
    """Top-1 softmax probability, as PhraseDecoder reports it (no context prior)."""
    mean = sum(sims.values()) / len(sims)
    x = np.array([base[p] + w * (sims[p] - mean) for p in base])
    e = np.exp(x - x.max())
    return float(e.max() / e.sum())


def best_sims(enc, takes, sim_fn):
    flat = [(p, t) for p, ts in takes.items() for t in ts]
    out = {}
    for (p, _), s in zip(flat, sim_fn(enc, [t for _, t in flat]).tolist()):
        out[p] = max(out.get(p, float("-inf")), s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--miracl")
    ap.add_argument("--manifest")
    ap.add_argument("--enroll", default="1,2,3", help="numbers of enrollment takes per phrase to evaluate")
    ap.add_argument("--tag", default="miracl")
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    clips, inventory = miracl_clips(os.path.expanduser(a.miracl)) if a.miracl else manifest_clips(a.manifest)
    ks = [int(k) for k in a.enroll.split(",")]
    engine = VSREngine(device=a.device)
    engine.warmup()

    t0 = time.time()
    dropped = []
    for i, c in enumerate(clips):
        c["enc"] = encode_clip(engine, c["path"])
        if c["enc"] is None:
            dropped.append(c["path"]); continue
        c["base"] = {r["phrase"]: r["score"] for r in engine.score_phrases(c["enc"], inventory)}
        if i % 50 == 0:
            print(f"[{i}/{len(clips)}] {time.time() - t0:.0f}s", flush=True)
    clips = [c for c in clips if c["enc"] is not None]
    print(f"encoded {len(clips)} clips, dropped {len(dropped)} (face not tracked): {dropped[:5]}")

    speakers = sorted({c["speaker"] for c in clips})
    test = [c for c in clips if c["take"] > max(ks)]
    lines = [f"# Enrollment eval: {a.tag}", "",
             f"{len(speakers)} speakers, {len(inventory)}-phrase inventory, test = takes > {max(ks)} of every phrase "
             f"({len(test)} clips, identical for every K). Generic = VSR log-likelihood only.", ""]

    # similarities for every (K, test clip) under both similarity functions
    sims = {}
    for method, fn in (("dtw", dtw_similarity), ("meanpool", mean_pool_similarity)):
        for k in ks:
            for s in speakers:
                takes = defaultdict(list)
                for c in clips:
                    if c["speaker"] == s and c["take"] <= k:
                        takes[c["phrase"]].append(c["enc"])
                for c in test:
                    if c["speaker"] == s:
                        sims[method, k, id(c)] = best_sims(c["enc"], takes, fn)

    def acc(method, k, w, spk):
        cs = [c for c in test if c["speaker"] in spk]
        return sum(top1(c["base"], sims[method, k, id(c)], w) == c["phrase"] for c in cs) / max(len(cs), 1)

    folds = [speakers[0::2], speakers[1::2]]
    chosen = {}
    lines += ["| similarity | K takes | generic top-1 | enrolled top-1 (CV weight) | prototype only | weights chosen per fold |", "|---|---|---|---|---|---|"]
    for method in ("dtw", "meanpool"):
        for k in ks:
            correct, ws = 0, []
            for i, f in enumerate(folds):
                other = folds[1 - i]
                w = max(WEIGHTS, key=lambda w: acc(method, k, w, other))
                ws.append(w)
                correct += acc(method, k, w, f) * sum(c["speaker"] in f for c in test)
            chosen[method, k] = ws
            lines.append(f"| {method} | {k} | {acc(method, k, 0, speakers):.3f} | {correct / len(test):.3f} | "
                         f"{acc(method, k, 1e6, speakers):.3f} | {ws} |")
    lines += ["", "Top-1 over all speakers by weight (DTW):", "", "| K | " + " | ".join(f"w={w}" for w in WEIGHTS) + " |",
              "|---|" + "---|" * len(WEIGHTS)]
    for k in ks:
        lines.append(f"| {k} | " + " | ".join(f"{acc('dtw', k, w, speakers):.3f}" for w in WEIGHTS) + " |")
    k = max(ks)
    w = max(WEIGHTS, key=lambda w: acc("dtw", k, w, speakers))
    lines += ["", f"Per speaker (DTW, K = {k}, w = {w}: the best weight over ALL speakers, i.e. in-sample; the CV column above is the honest number):", "",
              "| speaker | test clips | generic | enrolled |", "|---|---|---|---|"]
    for s in speakers:
        lines.append(f"| {s} | {sum(c['speaker'] == s for c in test)} | {acc('dtw', k, 0, [s]):.3f} | {acc('dtw', k, w, [s]):.3f} |")

    lines += ["", f"Confidence (top-1 softmax, K = {k}): server thresholds are 0.5 for a critical alert and 0.6 for history.", "",
              "| | mean conf when right | mean conf when wrong | wrong with conf >= 0.5 | wrong with conf >= 0.6 |", "|---|---|---|---|---|"]
    for label, wt in (("generic", 0), (f"enrolled w={w}", w)):
        rows = [(top1(c["base"], sims["dtw", k, id(c)], wt) == c["phrase"], confidence(c["base"], sims["dtw", k, id(c)], wt)) for c in test]
        right, wrong = [cf for ok, cf in rows if ok], [cf for ok, cf in rows if not ok]
        lines.append(f"| {label} | {np.mean(right):.2f} ({len(right)}) | {np.mean(wrong) if wrong else float('nan'):.2f} ({len(wrong)}) | "
                     f"{sum(cf >= 0.5 for cf in wrong)} | {sum(cf >= 0.6 for cf in wrong)} |")

    # real path check: engine.score_phrases with an active Profile must reproduce the sweep's choice
    mism, t_bonus = 0, []
    for s in speakers:
        prof = Profile(f"eval_{s}", weight=w)
        for c in clips:
            if c["speaker"] == s and c["take"] <= k:
                prof.add(c["phrase"], c["enc"])
        engine.profile = prof
        for c in test:
            if c["speaker"] != s:
                continue
            t = time.time(); prof.bonus(c["enc"], inventory); t_bonus.append(time.time() - t)
            real = engine.score_phrases(c["enc"], inventory)[0]["phrase"]
            mism += real != top1(c["base"], sims["dtw", k, id(c)], w)
    engine.profile = None
    lines += ["", f"Real-path check (engine.score_phrases with an active Profile, K={k}, w={w}): {mism} disagreements with the sweep "
              f"over {len(test)} clips. Prototype scoring cost: median {1000 * np.median(t_bonus):.1f} ms per utterance "
              f"({len(inventory)} phrases x {k} takes)."]

    for n in (40 * 3, 204 * 2):
        q, big = torch.randn(60, 768), [torch.randn(60, 768) for _ in range(n)]
        t = time.time(); dtw_similarity(q, big); dt = time.time() - t
        lines.append(f"Scaling (synthetic features): {n} templates, 60-frame (2.4 s) utterance and takes: {1000 * dt:.1f} ms.")
    out = os.path.join(ROOT, "results", f"enroll_{a.tag}.md")
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", out)


if __name__ == "__main__":
    main()
