"""What the patient's words are read against: the context the nurse and the UI set (ContextStore), and the LLM that
interprets the lip reading with it (LLMInterpreter). The visual model is authoritative: it verifies every LLM proposal."""
import os, json, time, re, threading


class ContextStore:
    def __init__(self):
        self.notes = ""            # free text typed by staff, e.g. "post-op day 1, complained of chest pain"
        self.category = None       # the topic picked in the UI (a phrases.txt category), or None
        self.history = []          # [(ts, text)] of what the patient said
        self.last_prompt = ""      # what the nurse just said/asked, e.g. "Do you want water?"
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


# LLM providers, all through the OpenAI-compatible chat API: (key variable, base URL, default model). Grok default: the fastest
# of the account's models on this task (1.5-1.9 s per call; grok-4.3 took 4.7-11.5 s, grok-4.7 8.8-10.4 s), and the app
# speaks only after the verdict, so this latency is added to every sentence.
LLM_PROVIDERS = {"grok": ("XAI_API_KEY", "https://api.x.ai/v1", "grok-4.20-0309-non-reasoning"),
                 "openai": ("OPENAI_API_KEY", None, "gpt-4o-mini")}


class LLMInterpreter:
    """The interpreter (server._bg_llm): proposes what the patient meant; the visual model verifies each proposal.
    provider: grok (XAI_API_KEY) | openai (OPENAI_API_KEY), read from the environment / project .env."""
    def __init__(self, provider="grok", model=None, timeout=4.0):
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
        self.key_var, base_url, default_model = LLM_PROVIDERS[provider]
        self.provider, self.model = provider, model or default_model
        self.key = os.environ.get(self.key_var, "")
        self.client = None
        if self.key:
            from openai import OpenAI
            # One attempt, bounded: the app speaks after the verdict, and a hung call used to take 60 s (20 s x 3 attempts)
            # before the raw reading was spoken. The default model answers in 1.3-1.7 s.
            self.client = OpenAI(api_key=self.key, base_url=base_url, timeout=timeout, max_retries=0)

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
                 "short words. The patient is in an ICU bed: what they say is almost always about their health, care, body, needs, "
                 "feelings or family, so prefer that meaning when the mouth shapes allow it, but never invent one they do not "
                 "support. Write up to 3 natural sentences the patient most plausibly meant, most likely first, with similar mouth "
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
