"""Phrase-constrained and open decoding on top of the VSR engine, with context-aware reranking."""
import math, time
import numpy as np


def softmax(xs, temp=1.0):
    xs = np.array(xs, dtype=np.float64) / temp
    xs = xs - xs.max()
    e = np.exp(xs)
    return (e / e.sum()).tolist()


class PhraseDecoder:
    """score(phrase) = VSR log-likelihood (authoritative) + gamma * contextual log-prior."""
    def __init__(self, engine, phrases, context, gamma=1.0, temp=1.0):
        self.engine, self.phrases, self.context = engine, phrases, context
        self.gamma, self.temp = gamma, temp

    def decode(self, enc, vsr=None):
        """vsr: optional precomputed rows [{phrase, att, ctc, score, n_tok}] (e.g. CTC-only scores from the streaming path)."""
        t0 = time.time()
        if vsr is None:
            vsr = self.engine.score_phrases(enc, self.phrases)  # sorted by VSR score
        else:
            vsr = sorted(vsr, key=lambda r: -r["score"])
        t1 = time.time()
        prior = {p["phrase"]: p for p in self.context.log_prior(self.phrases)}
        vsr_probs = softmax([r["score"] for r in vsr], self.temp)
        rows = []
        for r, vp in zip(vsr, vsr_probs):
            pr = prior[r["phrase"]]
            rows.append({"phrase": r["phrase"], "vsr_score": r["score"], "att": r["att"], "ctc": r["ctc"],
                         "vsr_prob": vp, "prior": pr["prior"], "reasons": pr["reasons"],
                         "final_score": r["score"] + self.gamma * pr["prior"]})
        fp = softmax([r["final_score"] for r in rows], self.temp)
        for r, p in zip(rows, fp):
            r["final_prob"] = p
        rows.sort(key=lambda r: -r["final_score"])
        margin = rows[0]["final_score"] - rows[1]["final_score"] if len(rows) > 1 else 99.0
        return {"ranking": rows, "selected": rows[0]["phrase"], "confidence": rows[0]["final_prob"], "margin": margin,
                "visual_top": vsr[0]["phrase"], "context_changed_choice": rows[0]["phrase"] != vsr[0]["phrase"],
                "t_score": t1 - t0}
