"""Phrase-constrained and open decoding on top of the VSR engine, with context-aware reranking."""
import math, time
import numpy as np


def softmax(xs, temp=1.0):
    xs = np.array(xs, dtype=np.float64) / temp
    xs = xs - xs.max()
    e = np.exp(xs)
    return (e / e.sum()).tolist()


class PhraseDecoder:
    """score(phrase) = VSR log-likelihood (authoritative) + enrolled-template evidence + gamma * contextual log-prior."""
    def __init__(self, engine, phrases, context, gamma=1.0, temp=1.0):
        self.engine, self.phrases, self.context = engine, phrases, context
        self.gamma, self.temp = gamma, temp
        self.profile = None  # active enroll.Profile (patient enrollment); None = generic model

    def decode(self, enc):
        t0 = time.time()
        vsr = self.engine.score_phrases(enc, self.phrases)  # sorted by VSR score
        prof = self.profile  # read once: a profile switch mid-decode must not mix two profiles
        proto = prof.evidence(enc, self.phrases) if prof else {}  # template log-likelihood ratios; unenrolled phrases get 0
        t1 = time.time()
        prior = {p["phrase"]: p for p in self.context.log_prior(self.phrases)}
        vsr_probs = softmax([r["score"] + proto.get(r["phrase"], 0.0) for r in vsr], self.temp)
        rows = []
        for r, vp in zip(vsr, vsr_probs):
            pr = prior[r["phrase"]]
            rows.append({"phrase": r["phrase"], "vsr_score": r["score"], "att": r["att"], "ctc": r["ctc"], "proto": proto.get(r["phrase"], 0.0),
                         "vsr_prob": vp, "prior": pr["prior"], "reasons": pr["reasons"],
                         "final_score": r["score"] + proto.get(r["phrase"], 0.0) + self.gamma * pr["prior"]})
        fp = softmax([r["final_score"] for r in rows], self.temp)
        for r, p in zip(rows, fp):
            r["final_prob"] = p
        rows.sort(key=lambda r: -r["final_score"])
        margin = rows[0]["final_score"] - rows[1]["final_score"] if len(rows) > 1 else 99.0
        visual_top = max(vsr, key=lambda r: r["score"] + proto.get(r["phrase"], 0.0))["phrase"]  # lips + templates, before context
        return {"ranking": rows, "selected": rows[0]["phrase"], "confidence": rows[0]["final_prob"], "margin": margin,
                "visual_top": visual_top, "context_changed_choice": rows[0]["phrase"] != visual_top,
                "t_score": t1 - t0}
