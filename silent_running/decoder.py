"""Phrase-constrained and open decoding on top of the VSR engine, with context-aware reranking."""
import math, os, time
import numpy as np

# The lip score of a phrase is the model's log P(phrase | video). Its attention decoder was trained on TED transcripts, so that
# score carries a TED language-model prior: with ambiguous lips the phrases TED says most ("I don't know", "Thank you",
# "Sorry", "Okay") win, whatever was mouthed. Two corrections, both in nats so enrollment evidence and the context prior
# still add on the same scale:
#   - subtract ILM_WEIGHT x the decoder's internal-LM log-prob of the phrase (VSREngine.internal_lm): removes the TED prior
#   - add TOKEN_BONUS per token (EOS included): the summed log-likelihood still penalises every extra token
# Chosen by leave-one-speaker-out CV on 160 MIRACL clips against the 204-phrase inventory (+10 MIRACL phrases): top-1
# 50.6% -> 70.0% held out, ECE 0.094 -> 0.066, no critical phrase picked wrongly; bonus 1.0 in 15/16 folds over two
# inventory sizes, ILM weight 0.2 (204 phrases) to 0.3 (40 phrases). Spoken clips: re-tune on silent mouthing when recorded.
ILM_WEIGHT = float(os.environ.get("SR_ILM_WEIGHT", "0.2"))
TOKEN_BONUS = float(os.environ.get("SR_TOKEN_BONUS", "1.0"))


def softmax(xs, temp=1.0):
    xs = np.array(xs, dtype=np.float64) / temp
    xs = xs - xs.max()
    e = np.exp(xs)
    return (e / e.sum()).tolist()


class PhraseDecoder:
    """score(phrase) = VSR log-likelihood (authoritative) + prior correction (token bonus - weighted internal LM)
    + enrolled-template evidence + gamma * contextual log-prior."""
    def __init__(self, engine, phrases, context, gamma=1.0, temp=1.0, token_bonus=TOKEN_BONUS, ilm_weight=ILM_WEIGHT):
        self.engine, self.phrases, self.context = engine, phrases, context
        self.gamma, self.temp = gamma, temp
        # per phrase, not from score_phrases rows (prefiltered-out rows carry n_tok 0); one decoder pass at startup
        ilm = engine.internal_lm(phrases) if ilm_weight else {}
        self.correction = {p: token_bonus * (len(engine.tokenize(p)) + 1) - ilm_weight * ilm.get(p, 0.0) for p in phrases}
        self.profile = None  # active enroll.Profile (patient enrollment); None = generic model

    def decode(self, enc):
        t0 = time.time()
        prof = self.profile  # read once: a profile switch mid-decode must not mix two profiles
        proto = prof.evidence(enc, self.phrases) if prof else {}  # template log-likelihood ratios; unenrolled phrases get 0
        # sorted by VSR score; a phrase the templates support is scored in full even outside the CTC prefilter
        vsr = self.engine.score_phrases(enc, self.phrases, always=[p for p, e in proto.items() if e > 0])
        t1 = time.time()
        prior = {p["phrase"]: p for p in self.context.log_prior(self.phrases)}
        lips = {r["phrase"]: r["score"] + self.correction[r["phrase"]] + proto.get(r["phrase"], 0.0) for r in vsr}
        # a prefiltered-out phrase has only a CTC estimate below every rescored one: its correction must not lift it above them
        kept = [lips[r["phrase"]] for r in vsr if not r.get("prefiltered_out")]
        for r in vsr:
            if r.get("prefiltered_out"):
                lips[r["phrase"]] = min(lips[r["phrase"]], min(kept)) - 1.0
        vsr_probs = softmax([lips[r["phrase"]] for r in vsr], self.temp)
        rows = []
        for r, vp in zip(vsr, vsr_probs):
            pr = prior[r["phrase"]]
            rows.append({"phrase": r["phrase"], "vsr_score": r["score"], "att": r["att"], "ctc": r["ctc"], "proto": proto.get(r["phrase"], 0.0),
                         "correction": self.correction[r["phrase"]], "vsr_prob": vp, "prior": pr["prior"], "reasons": pr["reasons"],
                         "final_score": lips[r["phrase"]] + self.gamma * pr["prior"]})
        fp = softmax([r["final_score"] for r in rows], self.temp)
        for r, p in zip(rows, fp):
            r["final_prob"] = p
        rows.sort(key=lambda r: -r["final_score"])
        margin = rows[0]["final_score"] - rows[1]["final_score"] if len(rows) > 1 else 99.0
        visual_top = max(lips, key=lips.get)  # lips + templates, before context
        return {"ranking": rows, "selected": rows[0]["phrase"], "confidence": rows[0]["final_prob"], "margin": margin,
                "visual_top": visual_top, "context_changed_choice": rows[0]["phrase"] != visual_top,
                "t_score": t1 - t0}
