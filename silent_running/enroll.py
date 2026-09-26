"""Patient enrollment without training: nearest-prototype scoring on the VSR encoder features.

The patient mouths each phrase 2-3 times. Each take's encoder output (T x 768, the same `enc` the decoder reads) is
stored per phrase in the patient's profile. At recognition time every enrolled phrase gets a similarity to the
utterance (best DTW match over its takes), and VSREngine.score_phrases adds `weight * (similarity - mean similarity)`
to the model log-likelihood. Phrases that were not enrolled get no bonus. No active profile = the generic model.

Profiles live in data/profiles/<name>.pt as {phrase: [float16 tensor (T, 768), ...]}.
"""
import os, re, time
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILE_DIR = os.path.join(ROOT, "data", "profiles")
WEIGHT = float(os.environ.get("ENROLL_WEIGHT", "150.0"))  # nats per unit of cosine similarity; 2-fold CV pick on MIRACL-VC1 (results/enroll_miracl.md)


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


class Profile:
    def __init__(self, name, takes=None, weight=WEIGHT):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError(f"profile name must be letters, digits, '_' or '-': {name!r}")
        self.name = name
        self.takes = takes or {}  # phrase -> [tensor (T, 768) float16]
        self.weight = weight

    @property
    def path(self):
        return os.path.join(PROFILE_DIR, self.name + ".pt")

    @classmethod
    def load(cls, name):
        p = cls(name)
        if os.path.exists(p.path):
            p.takes = torch.load(p.path, weights_only=True)
        return p

    def save(self):
        os.makedirs(PROFILE_DIR, exist_ok=True)
        torch.save(self.takes, self.path + ".tmp")
        os.replace(self.path + ".tmp", self.path)  # atomic: a concurrent load never sees a half-written file

    def add(self, phrase, enc):
        self.takes.setdefault(phrase, []).append(enc.detach().to("cpu", torch.float16).clone())

    def counts(self):
        return {p: len(t) for p, t in self.takes.items()}

    def similarities(self, enc, phrases):
        """-> {phrase: best DTW similarity over its takes} for the enrolled phrases among `phrases`."""
        enrolled = [p for p in phrases if self.takes.get(p)]
        if not enrolled:
            return {}
        flat = [(p, t) for p in enrolled for t in self.takes[p]]
        sims = dtw_similarity(enc, [t for _, t in flat])
        best = {}
        for (p, _), s in zip(flat, sims.tolist()):
            best[p] = max(best.get(p, float("-inf")), s)
        return best

    def bonus(self, enc, phrases):
        """-> {phrase: additive log-score bonus}, centred on the mean over enrolled phrases (unenrolled phrases get 0)."""
        sims = self.similarities(enc, phrases)
        if not sims:
            return {}
        mean = sum(sims.values()) / len(sims)
        return {p: self.weight * (s - mean) for p, s in sims.items()}


def list_profiles():
    if not os.path.isdir(PROFILE_DIR):
        return []
    return sorted(f[:-3] for f in os.listdir(PROFILE_DIR) if f.endswith(".pt"))


class Enrollment:
    """Server-side enrollment session: a queue of prompts (each phrase up to `reps` takes, counting takes the profile
    already has, cycling through the list so takes of one phrase are spread out), filled one captured utterance at a time."""
    def __init__(self, profile, phrases, reps=2):
        self.profile = profile
        have = profile.counts()
        self.queue = [p for r in range(reps) for p in phrases if have.get(p, 0) <= r]
        self.done = 0
        self.t_start = time.time()

    @property
    def prompt(self):
        return self.queue[self.done] if self.done < len(self.queue) else None

    def add(self, enc, phrase=None):
        """Store `enc` as a take of `phrase` (default: the current prompt) and advance. Returns the phrase stored."""
        phrase = phrase or self.prompt
        if phrase is None:
            raise ValueError("enrollment queue is finished")
        self.profile.add(phrase, enc)
        if phrase == self.prompt:
            self.done += 1
        return phrase

    def snapshot(self):
        return {"profile": self.profile.name, "prompt": self.prompt, "done": self.done, "total": len(self.queue),
                "elapsed": round(time.time() - self.t_start, 1), "counts": self.profile.counts()}
