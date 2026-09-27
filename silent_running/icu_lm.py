"""ICU patient language model: a subword n-gram over the VSR model's own 5000-piece vocabulary, trained on data/icu/corpus.txt,
plugged into the ESPnet beam search as the "lm" scorer (shallow fusion) and used to rescore phrase/bank candidates.

Why: the shipped RNN LM was trained on TED talks, so the decoder happily produces "hand gestures" and "band together"
for an ICU patient. A prior over what patients actually say has to act *inside* decoding: an LLM that only sees the
n-best afterwards cannot recover words the decoder never proposed. Witten-Bell interpolated n-gram, pure torch, no kenlm.
"""
import os, math, pickle, collections, time
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "data", "icu", "corpus.txt")
CACHE = os.path.join(ROOT, "models", "icu_lm")

try:
    from espnet.nets.scorer_interface import BatchScorerInterface
except ImportError:  # third_party path not set up yet (scripts import us first)
    import sys
    sys.path.insert(0, os.path.join(ROOT, "third_party", "chaplin")); sys.path.insert(0, os.path.join(ROOT, "third_party", "auto_avsr"))
    from espnet.nets.scorer_interface import BatchScorerInterface


class IcuNgramLM(BatchScorerInterface):
    """order-n subword LM with Witten-Bell interpolation down to an add-k unigram over the full vocabulary (so every token keeps
    a small floor probability and the LM weight, not -inf, decides how strongly the prior pulls)."""

    def __init__(self, tokenize, vocab_size, eos, corpus=CORPUS, order=3, unigram_k=0.5, max_cache=6000, verbose=True):
        self.tokenize, self.V, self.eos, self.order, self.k, self.max_cache = tokenize, vocab_size, eos, order, unigram_k, max_cache
        self.counts = [collections.defaultdict(collections.Counter) for _ in range(order)]  # counts[n][context tuple of len n] -> Counter(next)
        self.n_sent = 0
        self._cache = collections.OrderedDict()
        t0 = time.time()
        with open(corpus) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                ids = tokenize(line) + [eos]
                hist = [eos] * (order - 1)  # sos == eos id in this model
                for w in ids:
                    for n in range(order):
                        ctx = tuple(hist[len(hist) - n:]) if n else ()
                        self.counts[n][ctx][w] += 1
                    hist = (hist + [w])[-(order - 1):] if order > 1 else []
                self.n_sent += 1
        self._uni = None
        if verbose:
            print(f"[icu_lm] {self.n_sent} sentences, {sum(self.counts[0][()].values())} tokens, {len(self.counts[order-1])} {order}-gram contexts, built in {time.time()-t0:.2f}s")

    # ------------------------------------------------------------------ probabilities
    def _unigram(self):
        if self._uni is None:
            c = torch.full((self.V,), self.k)
            for w, n in self.counts[0][()].items():
                c[w] += n
            self._uni = c / c.sum()
        return self._uni

    def _probs(self, ctx):
        """Dense P(. | ctx) as a (V,) tensor, Witten-Bell interpolated with the shorter context."""
        key = tuple(ctx)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        n = len(key)
        if n == 0:
            p = self._unigram()
        else:
            lower = self._probs(key[1:])
            cnt = self.counts[n].get(key)
            if not cnt:
                p = lower
            else:
                N, T = sum(cnt.values()), len(cnt)
                p = lower * (T / (N + T))
                idx = torch.tensor(list(cnt.keys())); val = torch.tensor([float(v) for v in cnt.values()])
                p = p.clone(); p[idx] += val / (N + T)
        self._cache[key] = p
        if len(self._cache) > self.max_cache:
            self._cache.popitem(last=False)
        return p

    def logprobs(self, ctx):
        return torch.log(self._probs(tuple(ctx)[-(self.order - 1):] if self.order > 1 else ()))

    def sentence_logprob(self, text_or_ids):
        ids = self.tokenize(text_or_ids) if isinstance(text_or_ids, str) else list(text_or_ids)
        ids = ids + [self.eos]
        hist, lp = [self.eos] * (self.order - 1), 0.0
        for w in ids:
            lp += float(self.logprobs(hist)[w])
            hist = (hist + [w])[-(self.order - 1):] if self.order > 1 else []
        return lp

    def perplexity(self, text):
        ids = self.tokenize(text)
        return math.exp(-self.sentence_logprob(ids) / (len(ids) + 1))

    # ------------------------------------------------------------------ ESPnet scorer interface (beam search shallow fusion)
    def init_state(self, x):
        return None

    def score(self, y, state, x):
        ctx = y.tolist()[-(self.order - 1):] if self.order > 1 else []
        return self.logprobs(ctx).to(y.device), None

    def batch_score(self, ys, states, xs):
        rows = [self.logprobs(y.tolist()[-(self.order - 1):] if self.order > 1 else []) for y in ys]
        return torch.stack(rows).to(ys.device), [None] * len(rows)

    def select_state(self, state, i, new_id=None):
        return None


def load(engine, corpus=CORPUS, order=3):
    """Build (or load the pickled counts of) the LM for this engine's tokenizer."""
    os.makedirs(CACHE, exist_ok=True)
    stamp = f"{os.path.getmtime(corpus):.0f}_{order}"
    pk = os.path.join(CACHE, f"ngram_{stamp}.pkl")
    if os.path.exists(pk):
        lm = pickle.load(open(pk, "rb"))
        lm.tokenize = engine.tokenize
        lm._cache = collections.OrderedDict()
        return lm
    lm = IcuNgramLM(engine.tokenize, len(engine.token_list), engine.eos, corpus=corpus, order=order)
    tok, lm.tokenize = lm.tokenize, None  # sentencepiece handles don't pickle
    cache, lm._cache = lm._cache, collections.OrderedDict()
    pickle.dump(lm, open(pk, "wb"))
    lm.tokenize, lm._cache = tok, cache
    return lm
