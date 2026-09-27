"""Context store + transparent contextual prior + optional LLM reranker.

The visual model is authoritative. Context only *adjusts* scores of candidates the VSR model already
supports; it never introduces phrases. Every adjustment is returned so the UI can show it.
"""
import os, json, time, re, threading

def _load_categories():
    from silent_running.vsr import load_phrase_table
    cats = {}
    for r in load_phrase_table():
        cats.setdefault(r["category"], []).append(r["phrase"])
    return cats
CATEGORIES = _load_categories()
PHRASE_CATEGORY = {p.lower(): c for c, ps in CATEGORIES.items() for p in ps}
STOP = set("i am a the to my me is it of and please need want".split())


def _words(s):
    return [w for w in re.findall(r"[a-z']+", s.lower()) if w not in STOP]


class ContextStore:
    def __init__(self, w_category=1.5, w_keyword=1.0, w_recent=0.5, w_question=1.5):
        self.notes = ""            # free text typed by staff, e.g. "post-op day 1, complained of chest pain"
        self.category = None       # patient-selected category (or None)
        self.history = []          # [(ts, phrase)] of confirmed utterances
        self.last_prompt = ""      # what the nurse just said/asked (typed in UI), e.g. "Do you want water?"
        self.w = dict(category=w_category, keyword=w_keyword, recent=w_recent, question=w_question)
        self.lock = threading.Lock()

    def update(self, notes=None, category=None, last_prompt=None):
        with self.lock:
            if notes is not None: self.notes = notes
            if category is not None: self.category = category or None
            if last_prompt is not None: self.last_prompt = last_prompt

    def add_history(self, phrase):
        with self.lock:
            self.history.append((time.time(), phrase))
            self.history = self.history[-20:]

    def snapshot(self):
        with self.lock:
            return {"notes": self.notes, "category": self.category, "last_prompt": self.last_prompt,
                    "history": [p for _, p in self.history[-6:]]}

    def log_prior(self, phrases):
        """Additive log-prior adjustment per phrase, plus a human-readable list of reasons."""
        snap = self.snapshot()
        note_words = set(_words(snap["notes"])) | set(_words(snap["last_prompt"]))
        is_question = snap["last_prompt"].strip().endswith("?") or bool(re.match(r"^(do|did|are|is|can|could|would|will|have|has|should)\b", snap["last_prompt"].strip().lower()))
        recent = [p.lower() for p in snap["history"][-3:]]
        out = []
        for p in phrases:
            adj, reasons = 0.0, []
            pl = p.lower()
            if snap["category"] and PHRASE_CATEGORY.get(pl) == snap["category"]:
                adj += self.w["category"]; reasons.append(f"category:{snap['category']}")
            ov = note_words & set(_words(p))
            if ov:
                adj += self.w["keyword"] * min(len(ov), 2); reasons.append("keywords:" + ",".join(sorted(ov)))
            if is_question and pl in ("yes", "no"):
                adj += self.w["question"]; reasons.append("yes/no question asked")
            if pl in recent:
                adj += self.w["recent"]; reasons.append("repeated recently")
            out.append({"phrase": p, "prior": adj, "reasons": reasons})
        return out


# LLM providers, all through the OpenAI-compatible chat API: (key variable, base URL, default model). Grok default: the fastest
# of the account's models on this task (1.5-1.9 s per call; grok-4.3 took 4.7-11.5 s, grok-4.7 8.8-10.4 s), and Open Mode
# speaks only after the verdict, so this latency is added to every sentence.
LLM_PROVIDERS = {"grok": ("XAI_API_KEY", "https://api.x.ai/v1", "grok-4.20-0309-non-reasoning"),
                 "openai": ("OPENAI_API_KEY", None, "gpt-4o-mini")}


class LLMInterpreter:
    """Open Mode's interpreter (server._bg_llm): proposes what the patient meant; the visual model verifies each proposal.
    provider: grok (XAI_API_KEY) | openai (OPENAI_API_KEY), read from the environment / project .env."""
    def __init__(self, provider="grok", model=None, timeout=20):
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
        self.key_var, base_url, default_model = LLM_PROVIDERS[provider]
        self.provider, self.model = provider, model or default_model
        self.key = os.environ.get(self.key_var, "")
        self.client = None
        if self.key:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.key, base_url=base_url, timeout=timeout)

    def available(self):
        return self.client is not None

    def propose(self, candidates, context):
        """Up to 3 sentences the patient most plausibly meant, most likely first: {"sentences", "reason"} or {"error"}. The
        caller must verify each against the visual model before using it."""
        out = {}
        for kind, value in self.propose_stream(candidates, context):
            out[kind] = value
        return {"error": out["error"]} if "error" in out else out

    def propose_stream(self, candidates, context):
        """propose(), streamed: yields ("sentences", [...]) as soon as the reply's sentence list is complete, then
        ("reason", str); or ("error", str). The sentences come first in the reply and the reason after them took about as
        long again (Grok: sentences at 0.64 s, whole reply 1.22 s, median of 10), so the caller can act on them at once."""
        if self.client is None:
            yield "error", f"{self.key_var} not set"
            return
        sys_p = ("You interpret the output of a silent lip-reading model for a voiceless ICU patient. You get the model's n-best "
                 "hypotheses (higher score = more visual support) plus any context. They are often garbled or ungrammatical: lip "
                 "reading confuses sounds that look alike on the lips (p/b/m, t/d/n, k/g, f/v, s/z, most vowels) and drops or merges "
                 "short words. Write up to 3 natural sentences the patient most plausibly meant, most likely first, with similar mouth "
                 "shapes; if the top hypothesis is not a sentence a person would say, do not repeat it. The lip model checks every "
                 "one against the video. Output plain text in upper case without punctuation.")
        lines = "\n".join(f"{i}. {c['text']} (score {c['score']:.1f})" for i, c in enumerate(candidates))
        ctx = {k: (v[:200] if k == "notes" else v) for k, v in context.items() if v}  # only what exists; notes capped
        user = (f"Context: {json.dumps(ctx)}\n" if ctx else "") + f"Hypotheses:\n{lines}"
        schema = {"type": "object", "additionalProperties": False,
                  "properties": {"sentences": {"type": "array", "items": {"type": "string"}}, "reason": {"type": "string"}},
                  "required": ["sentences", "reason"]}
        try:
            stream = self.client.chat.completions.create(model=self.model, temperature=0, stream=True,
                messages=[{"role": "system", "content": sys_p}, {"role": "user", "content": user}],
                response_format={"type": "json_schema", "json_schema": {"name": "proposal", "strict": True, "schema": schema}})
            buf, sents = "", None
            for chunk in stream:
                buf += (chunk.choices[0].delta.content or "") if chunk.choices else ""
                if sents is None:
                    sents = _sentences_so_far(buf)
                    if sents is not None:
                        if not sents:
                            yield "error", "empty proposal"
                            return
                        yield "sentences", sents
            out = json.loads(buf)
            if sents is None:  # the reply put the reason first: the sentences are known only now
                sents = _clean(out.get("sentences", []))
                if not sents:
                    yield "error", "empty proposal"
                    return
                yield "sentences", sents
            yield "reason", out.get("reason", "")
        except Exception as e:
            yield "error", str(e)


def _clean(sentences):
    sents = [re.sub(r"[^A-Z' ]", "", t.upper()).strip() for t in sentences]
    return list(dict.fromkeys(t for t in sents if t))[:3]


def _sentences_so_far(buf):
    """The cleaned "sentences" list of a partial JSON reply once that list is complete, else None."""
    m = re.search(r'"sentences"\s*:\s*', buf)
    if not m:
        return None
    try:
        return _clean(json.JSONDecoder().raw_decode(buf, m.end())[0])
    except json.JSONDecodeError:
        return None
