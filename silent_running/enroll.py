"""Patient enrollment without training: template evidence from the VSR encoder features.

The patient mouths each phrase 1-3 times. Each take's encoder output (T x 768, the same `enc` the decoder reads) is
stored in the patient's profile, together with the DTW similarity between every pair of takes. At recognition time
every enrolled phrase gets its best DTW similarity s to the utterance (over its takes), and PhraseDecoder adds

    evidence(phrase) = SLOPE * (s - mu) - OFFSET

to the VSR log-likelihood. mu is the profile's own impostor similarity: the mean, over its takes, of a take's best
similarity to each *other* enrolled phrase. The evidence is a log-likelihood ratio "this phrase vs not this phrase":
positive when the utterance matches the phrase's takes better than takes of other phrases usually do, negative when
it does not. It is absolute (it does not depend on which other phrases are scored), and a phrase with no takes gets 0
(no evidence either way), so enrolling some phrases never buries the others. SLOPE and OFFSET are fitted by minimising
the held-out log-loss of the decoder's posterior (leave-one-speaker-out, scripts/eval_enroll.py,
results/enroll_miracl.md), so the reported confidence stays calibrated. No active profile = the generic model.

Profiles live in data/profiles/<name>.pt as {"phrases": [str], "encs": [float16 (T, 768)], "sim": float32 (N, N)}.
"""
import os, re, time
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILE_DIR = os.path.join(ROOT, "data", "profiles")
NAME = re.compile(r"[A-Za-z0-9_-]+")
# log-loss fit on all 8 MIRACL speakers; the held-out (leave-one-speaker-out) numbers are in results/enroll_miracl.md
SLOPE = float(os.environ.get("ENROLL_SLOPE", "85.1"))   # nats per unit of DTW cosine similarity
OFFSET = float(os.environ.get("ENROLL_OFFSET", "12.4"))  # nats; the evidence is 0 at s = mu + OFFSET / SLOPE
MIN_PHRASES = 5  # a profile gives evidence once this many phrases have takes: the smallest profile the held-out eval covers


def dtw_similarity(query, templates):
    """query (Tq, D), templates list of (Tr, D) -> tensor (K,) of cosine similarity along the best DTW path, in [-1, 1].

    Symmetric DTW (Sakoe & Chiba 1978): steps (1,0) and (0,1) cost c, the diagonal step costs 2c, total normalised by
    Tq + Tr, with c = 1 - cosine. Any two lengths have an alignment. Cells on one anti-diagonal are independent, so the
    recursion is vectorised over anti-diagonals and templates. Templates are zero-padded; the end cell (Tq, Tr_k) only
    depends on cells inside template k."""
    q = F.normalize(query.float().cpu(), dim=-1)
    lens = torch.tensor([t.shape[0] for t in templates])
    R = torch.zeros(len(templates), int(lens.max()), q.shape[1])
    for k, t in enumerate(templates):
        R[k, :t.shape[0]] = F.normalize(t.float(), dim=-1)
    cost = 1 - torch.einsum("qd,ktd->kqt", q, R)  # (K, Tq, Tr)
    K, Tq, Tr = cost.shape
    D = torch.full((K, Tq + 1, Tr + 1), float("inf"))
    D[:, 0, 0] = 0
    for d in range(2, Tq + Tr + 1):
        i = torch.arange(max(1, d - Tr), min(Tq, d - 1) + 1)
        j = d - i
        c = cost[:, i - 1, j - 1]
        D[:, i, j] = torch.minimum(torch.minimum(D[:, i - 1, j], D[:, i, j - 1]) + c, D[:, i - 1, j - 1] + 2 * c)
    return 1 - D[torch.arange(K), Tq, lens] / (Tq + lens)


def best_per_phrase(sims, labels):
    """Similarities to takes labelled `labels` -> {phrase: best similarity over its takes}."""
    best = {}
    for p, s in zip(labels, sims):
        best[p] = max(best.get(p, float("-inf")), float(s))
    return best


def impostor_mean(sim, labels):
    """(N, N) similarities between takes and their phrases -> mean over takes i and phrases q != labels[i] of
    max_{j: labels[j] = q} sim[i, j]: how similar a take typically is to the best take of a phrase it is not.
    None with fewer than two phrases (nothing to compare against)."""
    names = sorted(set(labels))
    if len(names) < 2:
        return None
    col = torch.tensor([names.index(p) for p in labels])
    best = torch.full((len(labels), len(names)), float("-inf")).scatter_reduce(1, col.expand(len(labels), -1), sim, "amax")
    return float(best[col[:, None] != torch.arange(len(names))].mean())


def template_evidence(best, mu, slope=SLOPE, offset=OFFSET):
    """{phrase: best similarity} and the profile's impostor mean -> {phrase: log-likelihood ratio in nats}."""
    return {p: slope * (s - mu) - offset for p, s in best.items()}


class Profile:
    def __init__(self, name):
        if not NAME.fullmatch(name):
            raise ValueError(f"profile name must be letters, digits, '_' or '-': {name!r}")
        self.name = name
        self.phrases = []            # phrase of each take, in recording order
        self.encs = []               # float16 (T, 768) per take
        self.sim = torch.zeros(0, 0)  # DTW similarity between takes
        self.mu = None               # impostor mean similarity; None until two phrases have takes

    @property
    def path(self):
        return os.path.join(PROFILE_DIR, self.name + ".pt")

    @classmethod
    def load(cls, name):
        p = cls(name)
        if os.path.exists(p.path):
            d = torch.load(p.path, weights_only=True)
            p.phrases, p.encs, p.sim = d["phrases"], d["encs"], d["sim"]
            p.mu = impostor_mean(p.sim, p.phrases)
        return p

    def save(self):
        os.makedirs(PROFILE_DIR, exist_ok=True)
        torch.save({"phrases": self.phrases, "encs": self.encs, "sim": self.sim}, self.path + ".tmp")
        os.replace(self.path + ".tmp", self.path)  # atomic: a concurrent load never sees a half-written file

    @property
    def ready(self):
        """Evidence needs a calibrated impostor mean: takes of at least MIN_PHRASES phrases."""
        return len(set(self.phrases)) >= MIN_PHRASES

    def add(self, phrase, enc):
        enc = enc.detach().to("cpu", torch.float16).clone()
        n = len(self.encs)
        sim = torch.ones(n + 1, n + 1)
        sim[:n, :n] = self.sim
        if n:
            sim[n, :n] = sim[:n, n] = dtw_similarity(enc, self.encs)
        self.phrases.append(phrase); self.encs.append(enc); self.sim = sim
        self.mu = impostor_mean(self.sim, self.phrases)

    def remove_last(self, phrase=None):
        """Drop the most recent take (of `phrase`, if given). Returns the phrase of the removed take."""
        i = len(self.phrases) - 1 if phrase is None else max(k for k, p in enumerate(self.phrases) if p == phrase)
        keep = [k for k in range(len(self.phrases)) if k != i]
        removed = self.phrases[i]
        self.phrases = [self.phrases[k] for k in keep]; self.encs = [self.encs[k] for k in keep]
        self.sim = self.sim[keep][:, keep]
        self.mu = impostor_mean(self.sim, self.phrases)
        return removed

    def counts(self):
        out = {}
        for p in self.phrases:
            out[p] = out.get(p, 0) + 1
        return out

    def evidence(self, enc, phrases):
        """-> {phrase: template evidence in nats} for the enrolled phrases among `phrases` (unenrolled phrases: absent = 0)."""
        if not self.ready:
            raise ValueError(f"profile {self.name!r} needs takes of at least {MIN_PHRASES} phrases")
        wanted = set(phrases)
        idx = [k for k, p in enumerate(self.phrases) if p in wanted]
        if not idx:
            return {}
        sims = dtw_similarity(enc, [self.encs[k] for k in idx])
        return template_evidence(best_per_phrase(sims.tolist(), [self.phrases[k] for k in idx]), self.mu)


def list_profiles():
    if not os.path.isdir(PROFILE_DIR):
        return []
    return sorted(f[:-3] for f in os.listdir(PROFILE_DIR) if f.endswith(".pt") and NAME.fullmatch(f[:-3]))


class Enrollment:
    """Server-side enrollment session: a queue of prompts (each phrase up to `reps` takes, counting takes the profile
    already has, cycling through the list so takes of one phrase are spread out), filled one captured utterance at a time."""
    def __init__(self, profile, phrases, reps=2):
        if reps < 1:
            raise ValueError(f"reps must be at least 1, got {reps}")
        have = profile.counts()
        self.queue = [p for r in range(reps) for p in phrases if have.get(p, 0) <= r]
        if not self.queue:
            raise ValueError(f"nothing left to enroll for {profile.name!r}: every phrase already has {reps} take(s)")
        self.profile = profile
        self.done = 0
        self.t_start = time.time()

    @property
    def prompt(self):
        return self.queue[self.done] if self.done < len(self.queue) else None

    def check(self, enc, phrase):
        """Reject a take that the profile's own evidence says is not `phrase` (it contradicts the earlier takes of it)."""
        if self.profile.ready and phrase in self.profile.counts():
            e = self.profile.evidence(enc, [phrase])[phrase]
            if e < 0:
                raise ValueError(f"take does not match the earlier take(s) of {phrase!r} (template evidence {e:+.1f} nats); not stored. "
                                 f"Mouth it again, or discard the earlier take (POST /api/enroll/undo, phrase={phrase!r}) if that one was wrong.")

    def add(self, enc, phrase=None):
        """Store `enc` as a take of `phrase` (default: the current prompt) and advance. Returns the phrase stored."""
        phrase = phrase or self.prompt
        self.check(enc, phrase)
        self.profile.add(phrase, enc)
        if phrase == self.prompt:
            self.done += 1
        return phrase

    def undo(self, phrase=None):
        """Discard the last take (of `phrase`, if given) and prompt its phrase again next. Returns the phrase."""
        if phrase is not None and phrase not in self.profile.counts():
            raise ValueError(f"no take of {phrase!r} to discard")
        if not self.profile.phrases:
            raise ValueError("no take to discard")
        removed = self.profile.remove_last(phrase)
        self.queue.insert(self.done, removed)
        return removed

    def snapshot(self):
        return {"profile": self.profile.name, "prompt": self.prompt, "done": self.done, "total": len(self.queue),
                "elapsed": round(time.time() - self.t_start, 1), "counts": self.profile.counts()}
