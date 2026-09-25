"""Context store + transparent contextual prior + optional LLM reranker.

The visual model is authoritative. Context only *adjusts* scores of candidates the VSR model already
supports; it never introduces phrases. Every adjustment is returned so the UI can show it.
"""
import os, json, time, re, threading

CATEGORIES = {
    "pain": ["I am in pain", "My chest hurts", "My head hurts", "I need medication"],
    "breathing": ["I can't breathe", "Please suction"],
    "comfort": ["I am cold", "I am hot", "Turn me over", "I want to sit up", "I want to lie down", "Please turn off the light", "Please turn on the TV"],
    "people": ["Call my family", "I need the nurse", "Please call the doctor", "I need help"],
    "needs": ["I need water", "I need to use the bathroom", "I am hungry", "I need my glasses"],
    "answers": ["Yes", "No", "Thank you"],
    "status": ["I feel sick", "I feel dizzy", "I am scared", "I am tired", "Where am I", "What time is it"],
}
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


class OpenAIChooser:
    """OpenAI-backed chooser. Used asynchronously (never on the primary Phrase Mode path).
    Reads OPENAI_API_KEY from the environment / project .env."""
    def __init__(self, model="gpt-4o-mini", timeout=20):
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
        self.model, self.timeout = model, timeout
        self.key = os.environ.get("OPENAI_API_KEY", "")
        self.client = None
        if self.key:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.key, timeout=timeout)

    def available(self):
        return self.client is not None

    def prewarm(self):
        return self.available()

    def choose(self, candidates, context, mode="phrase"):
        """Pick among VSR-supported candidates. Returns {index, reason, corrected} or {"error": ...}."""
        if self.client is None:
            return {"error": "OPENAI_API_KEY not set"}
        sys_p = ("You help decode silent lip-reading for a hospital patient who cannot speak. You receive candidate "
                 "transcripts from a visual speech model (higher score = more visual support) and context. Choose the index of the "
                 "candidate the patient most plausibly said. Prefer higher-scored candidates unless context clearly favors another. "
                 "Never invent content that is not in the candidates.")
        if mode == "open":
            sys_p += (" Also give a minimally corrected, naturally-cased version of the chosen candidate in 'corrected' "
                      "(fix at most one obvious lip-reading confusion; keep the same words otherwise).")
        lines = "\n".join(f"{i}. {c['text']} (score {c['score']:.1f})" for i, c in enumerate(candidates))
        user = f"Context: {json.dumps(context)}\nCandidates:\n{lines}"
        schema = {"type": "object", "additionalProperties": False,
                  "properties": {"index": {"type": "integer"}, "reason": {"type": "string"}, "corrected": {"type": "string"}},
                  "required": ["index", "reason", "corrected"]}
        try:
            r = self.client.chat.completions.create(
                model=self.model, temperature=0,
                messages=[{"role": "system", "content": sys_p}, {"role": "user", "content": user}],
                response_format={"type": "json_schema", "json_schema": {"name": "choice", "strict": True, "schema": schema}})
            out = json.loads(r.choices[0].message.content)
            i = int(out.get("index", 0))
            if not (0 <= i < len(candidates)):
                return {"error": f"index {i} out of range"}
            return {"index": i, "reason": out.get("reason", ""), "corrected": out.get("corrected") or candidates[i]["text"]}
        except Exception as e:
            return {"error": str(e)}

    def propose(self, candidates, context):
        """Generative error correction: propose ONE corrected sentence built from the visual hypotheses.
        The caller must verify the proposal against the visual model before using it."""
        if self.client is None:
            return {"error": "OPENAI_API_KEY not set"}
        sys_p = ("You are correcting the output of a silent lip-reading model for a hospital patient. You get the model's n-best "
                 "hypotheses (higher score = more visual support; they usually share the correct skeleton and differ in confusable words) "
                 "plus context. Lip reading confuses sounds that look alike on the lips: p/b/m, t/d/n, k/g, f/v, s/z, and most vowels. "
                 "Write the single most plausible sentence the person actually said. Keep the word count and rhythm of the top hypotheses; "
                 "only replace words with visually similar alternatives that make the sentence coherent. Do not add new ideas. "
                 "Output plain text in upper case without punctuation.")
        lines = "\n".join(f"{i}. {c['text']} (score {c['score']:.1f})" for i, c in enumerate(candidates))
        user = f"Context: {json.dumps(context)}\nHypotheses:\n{lines}"
        schema = {"type": "object", "additionalProperties": False, "properties": {"sentence": {"type": "string"}, "reason": {"type": "string"}}, "required": ["sentence", "reason"]}
        try:
            r = self.client.chat.completions.create(model=self.model, temperature=0,
                messages=[{"role": "system", "content": sys_p}, {"role": "user", "content": user}],
                response_format={"type": "json_schema", "json_schema": {"name": "proposal", "strict": True, "schema": schema}})
            out = json.loads(r.choices[0].message.content)
            sent = re.sub(r"[^A-Z' ]", "", out.get("sentence", "").upper()).strip()
            if not sent:
                return {"error": "empty proposal"}
            return {"sentence": sent, "reason": out.get("reason", "")}
        except Exception as e:
            return {"error": str(e)}
