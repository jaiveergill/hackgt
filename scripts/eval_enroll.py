"""Patient enrollment eval: generic Phrase Mode vs the calibrated template evidence of enroll.py, held out by speaker.

  python scripts/eval_enroll.py --miracl ~/.cache/silent_running/miracl/full --tag miracl
  python scripts/eval_enroll.py --manifest data/eval/manifest.jsonl --tag ours

Every clip is encoded once (cached), scored by the generic model over the inventory, and compared with every other clip
by DTW. Protocols (test = the last two takes of every phrase; profiles are built from earlier takes of the SAME clips set):
  all K=k         the speaker's own profile, takes 1..k of every phrase
  half            takes 1..2 of half the phrases (both halves in turn); accuracy on enrolled and unenrolled phrases
  one not enrolled  takes 1..2 of all phrases but the tested one
  other speaker   another speaker's profile (takes 1..3) is active
The evidence parameters (SLOPE, OFFSET) are fitted by minimising the log-loss of the decoder's posterior on the other
speakers' "all K" and "half" instances (leave-one-speaker-out); every number in the tables is on the held-out speaker.
Then: reliability of the confidence, how often a nurse question can change a decision, the enrollment take gate, a
real-path check (PhraseDecoder with a real Profile, the server's code path), the same clips inside the 204-phrase ICU
inventory (CTC prefilter path, only these phrases enrolled) and latency on real features.
Holds the machine-wide lock (the model needs ~2 GB). Writes results/enroll_<tag>.md.
"""
import argparse, fcntl, hashlib, json, os, shutil, sys, tempfile, time
from collections import defaultdict
import av
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from silent_running.vsr import VSREngine, load_phrases, _read_video
from silent_running import enroll
from silent_running.enroll import Profile, dtw_similarity, impostor_mean, best_per_phrase, SLOPE, OFFSET
from silent_running.decoder import PhraseDecoder
from silent_running.context import ContextStore

MIRACL_PHRASES = ["Stop navigation", "Excuse me", "I am sorry", "Thank you", "Good bye", "I love this game",
                  "Nice to meet you", "You are welcome", "How are you", "Have a good time"]
CACHE = os.path.expanduser("~/.cache/silent_running/enroll_eval")
UNRELATED = ["data/samples/ted1_short.mp4", "data/samples/ted1_12s.mp4"]  # speech that is none of the phrases
BINS = [(0.0, 0.5), (0.5, 0.6), (0.6, 0.9), (0.9, 1.01)]  # server thresholds: critical alert 0.5, speak without asking 0.6
TRAIN = ("all K=1", "all K=2", "all K=3", "half: enrolled phrases", "half: unenrolled phrases")


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


class Sweep:
    """Generic scores + the DTW matrix; builds evidence inputs for (query, profile takes) without re-running DTW."""
    def __init__(self, clips, inventory):
        self.c, self.inv = clips, inventory
        self.base = np.array([[c["base"][p] for p in inventory] for c in clips])
        self.y = np.array([inventory.index(c["phrase"]) for c in clips])
        tpl = [c["enc"].to(torch.float16) for c in clips]  # profiles store float16 takes
        self.S = torch.stack([dtw_similarity(c["enc"], tpl) for c in clips])
        self._mu = {}

    def idx(self, speaker, takes, phrases=None):
        return [i for i, c in enumerate(self.c) if c["speaker"] == speaker and c["take"] in takes and (phrases is None or c["phrase"] in phrases)]

    def mu(self, T):
        k = tuple(T)
        if k not in self._mu:
            self._mu[k] = impostor_mean(self.S[T][:, T], [self.c[j]["phrase"] for j in T])
        return self._mu[k]

    def inputs(self, row, T):
        """-> (mask, x): which phrases are enrolled and their best similarity minus the profile's impostor mean."""
        best = best_per_phrase(row[T].tolist(), [self.c[j]["phrase"] for j in T])
        mask, x, mu = np.zeros(len(self.inv)), np.zeros(len(self.inv)), self.mu(T)
        for p, s in best.items():
            mask[self.inv.index(p)] = 1.0; x[self.inv.index(p)] = s - mu
        return mask, x


def protocols(sw, speakers, s, k_max):
    """[(query index, profile take indices, protocol)] with speaker s as the test speaker."""
    inv, test = sw.inv, [i for i, c in enumerate(sw.c) if c["speaker"] == s and c["take"] > k_max]
    out = []
    for k in range(1, k_max + 1):
        T = sw.idx(s, range(1, k + 1))
        out += [(q, T, f"all K={k}") for q in test]
    for half in (inv[:len(inv) // 2], inv[len(inv) // 2:]):
        T = sw.idx(s, (1, 2), half)
        out += [(q, T, "half: enrolled phrases" if sw.c[q]["phrase"] in half else "half: unenrolled phrases") for q in test]
    for q in test:
        out.append((q, sw.idx(s, (1, 2), [p for p in inv if p != sw.c[q]["phrase"]]), "one phrase not enrolled"))
    for a in speakers:
        if a != s:
            out += [(q, sw.idx(a, range(1, k_max + 1)), "other speaker's profile") for q in test]
    return out


def fit(insts):
    """(base, mask, x, label) instances -> (slope, offset) minimising the mean log-loss of the decoder's posterior
    softmax(base + mask * (slope * x - offset)). Convex in (slope, offset), so L-BFGS finds the optimum."""
    B, M, X = (torch.tensor(np.stack([i[k] for i in insts])) for k in range(3))
    Y = torch.tensor([i[3] for i in insts])
    th = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([th], max_iter=500, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(B + M * (th[0] * X - th[1]), Y)
        loss.backward()
        return loss
    opt.step(closure)
    return tuple(th.detach().tolist())


def scores(base, mask, x, slope, offset):
    return base + mask * (slope * x - offset)


def posterior(sc):
    e = np.exp(sc - sc.max())
    return e / e.sum()


def calibration(ok, conf):
    """Correct flags and top-1 confidences -> accuracy, expected calibration error (10 bins), confident errors."""
    ok, conf = np.array(ok), np.array(conf)
    b = np.minimum((conf * 10).astype(int), 9)
    ece = sum(abs(ok[b == k].mean() - conf[b == k].mean()) * (b == k).sum() for k in range(10) if (b == k).any()) / len(ok)
    return {"n": len(ok), "acc": ok.mean(), "ece": ece, "wrong": int((~ok).sum()), "w6": int((~ok & (conf >= 0.6)).sum()),
            "w9": int((~ok & (conf >= 0.9)).sum()), "ok": ok, "conf": conf}


def summary(rows):
    """rows: [(scores, label)] -> calibration() plus the share of close decisions (top-2 margin < 2 nats)."""
    ok, conf, margin = [], [], []
    for sc, y in rows:
        pr = posterior(sc); o = np.sort(sc)
        ok.append(pr.argmax() == y); conf.append(pr.max()); margin.append(o[-1] - o[-2])
    return {**calibration(ok, conf), "close": (np.array(margin) < 2.0).mean()}


def context_flips(rows, inv):
    """How often a nurse question naming the runner-up flips the decision, and how often one naming the true phrase
    fixes a wrong decision (the real ContextStore prior, gamma 1)."""
    ctx = ContextStore()
    def prior_for(q):
        ctx.update(last_prompt=q)
        return np.array([r["prior"] for r in ctx.log_prior(inv)])
    flips = fixes = wrong = 0
    for sc, y in rows:
        o = np.argsort(-sc)
        flips += (sc + prior_for(f"Is it {inv[o[1]].lower()}?")).argmax() != o[0]
        if o[0] != y:
            wrong += 1
            fixes += (sc + prior_for(f"Is it {inv[y].lower()}?")).argmax() == y
    return flips / len(rows), fixes, wrong


def median_ms(f, n=5):
    ts = []
    for _ in range(n):
        t = time.time(); f(); ts.append(time.time() - t)
    return 1000 * float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--miracl")
    ap.add_argument("--manifest")
    ap.add_argument("--kmax", type=int, default=3, help="largest number of enrollment takes per phrase (tests use later takes)")
    ap.add_argument("--tag", default="miracl")
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    lock = open(os.path.expanduser("~/.cache/silent_running/smoke.lock"), "w")  # same machine-wide lock as smoke.py (8 GB RAM)
    print("waiting for the machine-wide lock ...", flush=True)
    fcntl.flock(lock, fcntl.LOCK_EX)
    print(f"lock acquired, load {os.getloadavg()[0]:.1f}", flush=True)
    enroll.PROFILE_DIR = tempfile.mkdtemp()  # the eval's profiles never touch data/profiles
    clips, inventory = miracl_clips(os.path.expanduser(a.miracl)) if a.miracl else manifest_clips(a.manifest)
    engine = VSREngine(device=a.device)
    engine.warmup()

    t0 = time.time()
    dropped = []
    for c in clips:
        c["enc"] = encode_clip(engine, c["path"])
        if c["enc"] is None:
            dropped.append(c["path"]); continue
        c["base"] = {r["phrase"]: r["score"] for r in engine.score_phrases(c["enc"], inventory)}
        c["no_speech"] = not engine.ctc_greedy(c["enc"]).strip()  # the server's "no mouth movement" gate
    clips = [c for c in clips if c["enc"] is not None]
    print(f"encoded + scored {len(clips)} clips in {time.time() - t0:.0f}s, dropped {len(dropped)} (face not tracked): {dropped[:5]}", flush=True)
    t0 = time.time()
    sw = Sweep(clips, inventory)
    print(f"DTW matrix {tuple(sw.S.shape)} in {time.time() - t0:.0f}s", flush=True)
    speakers = sorted({c["speaker"] for c in clips})
    inst = lambda q, T: (sw.base[q], *sw.inputs(sw.S[q], T), sw.y[q])

    # ---- leave-one-speaker-out fit and held-out scores
    per_spk = {s: [(inst(q, T), tag) for q, T, tag in protocols(sw, speakers, s, a.kmax)] for s in speakers}
    held, folds = defaultdict(list), []
    for s in speakers:
        th = fit([i for o in speakers if o != s for i, tag in per_spk[o] if tag in TRAIN])
        folds.append(th)
        for i, tag in per_spk[s]:
            held[tag].append((i, th))
    final = fit([i for s in speakers for i, tag in per_spk[s] if tag in TRAIN])

    n_takes = max(c["take"] for c in clips)
    L = [f"# Enrollment eval: {a.tag}, {len(speakers)} speakers x {len(inventory)} phrases x {n_takes} takes", "",
         f"Data: {'the Kaggle MIRACL-VC1 subset (blueguydeez8974/miracl-vc1)' if a.miracl else a.manifest}, {len(clips)} clips, every take of a speaker from one recording session (so cross-session "
         f"drift is not measured). Test = takes {a.kmax + 1}-{n_takes} of every phrase; profiles use earlier takes.",
         "", "Evidence for an enrolled phrase = SLOPE * (best DTW similarity - the profile's impostor mean) - OFFSET, added to the VSR "
         "log-likelihood; unenrolled phrases get 0. (SLOPE, OFFSET) are fitted by minimising the log-loss of the decoder's posterior "
         f"on the other {len(speakers) - 1} speakers' {', '.join(TRAIN)} instances; every number below is on the held-out speaker.", "",
         "Fitted (SLOPE, OFFSET) per held-out speaker: " + ", ".join(f"{s} ({p:.1f}, {o:.1f})" for s, (p, o) in zip(speakers, folds)),
         f"Fitted on all speakers (the defaults in enroll.py): SLOPE {final[0]:.1f}, OFFSET {final[1]:.1f} "
         f"(enroll.py has SLOPE {SLOPE}, OFFSET {OFFSET}).", "",
         "| protocol (held out) | n | generic top-1 | enrolled top-1 | ECE generic / enrolled | wrong with conf >= 0.6: generic / enrolled | "
         "wrong with conf >= 0.9: generic / enrolled | decisions within 2 nats: generic / enrolled |", "|---|---|---|---|---|---|---|---|"]
    own_g, own_e = [], []
    for tag, rows in held.items():
        g = summary([(b, y) for (b, m, x, y), th in rows])
        e = summary([(scores(b, m, x, *th), y) for (b, m, x, y), th in rows])
        L.append(f"| {tag} | {e['n']} | {g['acc']:.3f} | **{e['acc']:.3f}** | {g['ece']:.3f} / {e['ece']:.3f} | {g['w6']}/{g['wrong']} / {e['w6']}/{e['wrong']} | "
                 f"{g['w9']} / {e['w9']} | {g['close']:.2f} / {e['close']:.2f} |")
        if tag != "other speaker's profile":
            own_g.append(g); own_e.append(e)

    L += ["", "Reliability of the confidence (held out, own-profile protocols pooled):", "",
          "| confidence | generic: n, accuracy, mean conf | enrolled: n, accuracy, mean conf |", "|---|---|---|"]
    for lo, hi in BINS:
        cells = []
        for grp in (own_g, own_e):
            ok = np.concatenate([g["ok"] for g in grp]); cf = np.concatenate([g["conf"] for g in grp]); m = (cf >= lo) & (cf < hi)
            cells.append(f"{m.sum()}, {ok[m].mean():.3f}, {cf[m].mean():.2f}" if m.any() else "0")
        L.append(f"| {lo:.1f}-{min(hi, 1.0):.1f} | {cells[0]} | {cells[1]} |")

    L += ["", "Nurse question (real ContextStore prior, gamma 1), held out, own-profile protocols: \"Is it <runner-up>?\" flips the "
          "decision / \"Is it <true phrase>?\" fixes a wrong decision:", ""]
    own = [(i, th) for tag, rows in held.items() if tag != "other speaker's profile" for i, th in rows]
    for label, rows in (("generic", [(b, y) for (b, m, x, y), th in own]), ("enrolled", [(scores(b, m, x, *th), y) for (b, m, x, y), th in own])):
        f, fx, w = context_flips(rows, inventory)
        L.append(f"- {label}: flips {f:.1%} of {len(rows)} decisions; fixes {fx} of {w} wrong decisions")

    # ---- enrollment take gate (Enrollment.check): reject a take whose evidence for its own label is < 0
    L += ["", f"Take gate (a new take of phrase X is rejected if the profile's evidence for X is < 0; needs an earlier take of X). "
          f"Profile = take 1 of every phrase, candidate = take 2, final parameters:", ""]
    unrel = []
    for p in UNRELATED:
        e = encode_clip(engine, os.path.join(ROOT, p))
        if e is None:
            L.append(f"- (skipped {p}: face not tracked)"); continue
        unrel.append(dtw_similarity(e, [c["enc"].to(torch.float16) for c in clips]))
    gen = junk = other = 0; n_gen = n_junk = n_other = 0
    for s in speakers:
        T = sw.idx(s, (1,))
        for q in sw.idx(s, (2,)):
            b, m, x, y = inst(q, T); ev = m * (final[0] * x - final[1])
            gen += ev[y] < 0; n_gen += 1
            junk += sum(ev[k] < 0 for k in range(len(inventory)) if k != y); n_junk += len(inventory) - 1
        for row in unrel:
            m, x = sw.inputs(row, T); ev = final[0] * x - final[1]
            other += int(((ev < 0) & (m > 0)).sum()); n_other += int(m.sum())
    L += [f"- genuine take 2 rejected: {gen}/{n_gen}", f"- take of another phrase, labelled as X: rejected {junk}/{n_junk}",
          f"- unrelated speech ({len(unrel)} TED clips x every phrase label): rejected {other}/{n_other}",
          f"- the \"no mouth movement\" gate (empty CTC greedy transcript), which every take passes first: rejects {sum(c['no_speech'] for c in clips)}/{len(clips)} genuine clips",
          "- a first take of a phrase has nothing to compare with and is only checked for mouth movement (use /api/enroll/undo)."]

    # ---- real path: PhraseDecoder + Profile (what the server runs), final parameters from enroll.py
    dec = PhraseDecoder(engine, inventory, ContextStore())
    mism, dconf, n = 0, 0.0, 0
    for s in speakers:
        for label, T in (("all", sw.idx(s, range(1, a.kmax + 1))), ("half", sw.idx(s, (1, 2), inventory[:len(inventory) // 2]))):
            prof = Profile(f"eval_{s}_{label}")
            for j in T:
                prof.add(clips[j]["phrase"], clips[j]["enc"])
            dec.profile = prof
            for q in sw.idx(s, range(a.kmax + 1, n_takes + 1)):
                b, m, x, y = inst(q, T)
                pr = posterior(scores(b, m, x, SLOPE, OFFSET))
                pd = dec.decode(clips[q]["enc"])
                mism += pd["selected"] != inventory[int(pr.argmax())]; dconf = max(dconf, abs(pd["confidence"] - pr.max())); n += 1
    dec.profile = None
    L += ["", f"Real-path check (PhraseDecoder.decode with a real Profile, all-enrolled and half-enrolled, enroll.py parameters): "
          f"{mism} disagreements with the sweep over {n} decisions, max confidence difference {dconf:.4f}."]

    # ---- large inventory: the ICU phrases + these phrases, through the CTC prefilter; only these phrases have takes
    icu = load_phrases()
    big = icu + [p for p in inventory if p not in icu]
    dec = PhraseDecoder(engine, big, ContextStore())
    res = {"generic": ([], []), "enrolled": ([], [])}
    for s in speakers:
        prof = Profile(f"eval_{s}_big")
        for j in sw.idx(s, range(1, a.kmax + 1)):
            prof.add(clips[j]["phrase"], clips[j]["enc"])
        for q in sw.idx(s, range(a.kmax + 1, n_takes + 1)):
            for label, pf in (("generic", None), ("enrolled", prof)):
                dec.profile = pf
                pd = dec.decode(clips[q]["enc"])
                res[label][0].append(pd["selected"] == clips[q]["phrase"]); res[label][1].append(pd["confidence"])
    g, e = (calibration(*res[k]) for k in ("generic", "enrolled"))
    L += ["", f"Large inventory ({len(big)} phrases = the {len(icu)} ICU phrases + these; CTC prefilter + attention on the top 48 "
          f"and on every phrase with positive evidence), profile = takes 1-{a.kmax} of these {len(inventory)} phrases only, test takes "
          f"{a.kmax + 1}-{n_takes}, real PhraseDecoder, enroll.py parameters (fitted on all speakers, so not held out):", "",
          "| | n | top-1 | ECE | wrong with conf >= 0.6 | wrong with conf >= 0.9 |", "|---|---|---|---|---|---|",
          f"| generic | {g['n']} | {g['acc']:.3f} | {g['ece']:.3f} | {g['w6']}/{g['wrong']} | {g['w9']} |",
          f"| enrolled | {e['n']} | **{e['acc']:.3f}** | {e['ece']:.3f} | {e['w6']}/{e['wrong']} | {e['w9']} |"]

    # ---- latency on real features (evidence and the whole PhraseDecoder.decode), under this script's lock
    L += ["", f"Latency, real features (MIRACL takes as templates, slices of a TED encoding as queries), median of 5, load {os.getloadavg()[0]:.1f} at start:", "",
          "| inventory | templates | query | evidence ms | decode ms without profile | decode ms with profile |", "|---|---|---|---|---|---|"]
    ted = encode_clip(engine, os.path.join(ROOT, "data/samples/ted1_12s.mp4"))
    icu = load_phrases()
    rng = np.random.default_rng(0)
    for n_ph, reps in ((40, 3), (len(icu), 2)):
        inv = icu[:n_ph]
        prof = Profile("latency")
        for p in inv:
            for _ in range(reps):
                prof.add(p, clips[int(rng.integers(len(clips)))]["enc"])
        dec = PhraseDecoder(engine, inv, ContextStore())
        for secs in (1, 2, 4):
            q = ted[:25 * secs]
            t_ev = median_ms(lambda: prof.evidence(q, inv))
            dec.profile = None; t0d = median_ms(lambda: dec.decode(q))
            dec.profile = prof; t1d = median_ms(lambda: dec.decode(q))
            L.append(f"| {n_ph} phrases | {len(prof.phrases)} | {secs} s ({q.shape[0]} frames) | {t_ev:.0f} | {t0d:.0f} | {t1d:.0f} |")
        t = time.time(); prof.save(); t_save = time.time() - t
        L.append(f"| {n_ph} phrases | profile save (per take) | {os.path.getsize(prof.path) / 1e6:.0f} MB | {1000 * t_save:.0f} ms | | |")
    shutil.rmtree(enroll.PROFILE_DIR)
    L.append(f"\nLoad at end: {os.getloadavg()[0]:.1f}.")

    out = os.path.join(ROOT, "results", f"enroll_{a.tag}.md")
    open(out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("wrote", out)


if __name__ == "__main__":
    main()
