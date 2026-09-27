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


class OpenAIChooser:
    """OpenAI proposer for Open Mode's verified correction (server._bg_llm). Never on the Phrase Mode path.
    Reads OPENAI_API_KEY from the environment / project .env."""
    def __init__(self, model="gpt-4o-mini", timeout=20):
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
        self.model = model
        self.key = os.environ.get("OPENAI_API_KEY", "")
        self.client = None
        if self.key:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.key, timeout=timeout)

    def available(self):
        return self.client is not None

    def pick_phrase(self, transcript, candidates, context):
        """Re-rank in-domain candidates for a Phrase Mode utterance. candidates: [{"text", "prob", "rank"}], best lip match first.
        Returns {"index": int (-1 = none fits), "confidence": high|medium|low, "reason"} or {"error"}. The caller verifies the
        pick against the visual scores and keeps the confirm loop: this only re-ranks and grades, it never speaks on its own.
        Measured on simulated lip noise (scratch harness, 120 trials): plain picking is not better than the lip scorer's top-1,
        and the model rarely abstains, hence the explicit abstain rules and the visual-margin verification in the server."""
        if self.client is None:
            return {"error": "OPENAI_API_KEY not set"}
        sys_p = ("You are the safety layer of a silent-speech communicator for a voiceless ICU patient. Input: a NOISY lip-reading transcript, "
                 "a ranked list of candidate phrases with the lip model's posteriors (rank 1 = best lip match), and context (what the nurse just "
                 "asked, patient category, notes, recent phrases).\n"
                 "Lip reading cannot distinguish sounds made with the same mouth shape: p/b/m, t/d/n/s/z/l, k/g/h, f/v, ch/sh/j, and most vowels; "
                 "words are often dropped or merged. Judge candidates by mouth-shape similarity to the transcript and word rhythm, then by fit to the context.\n"
                 "Rules: (1) prefer higher-ranked candidates unless the transcript or context clearly points elsewhere; (2) a wrong phrase is worse "
                 "than no phrase: if two candidates fit about equally and mean different things, answer 'low'; (3) never choose an urgent phrase "
                 "(breathing, chest, choking, bleeding, emergency) on context alone; (4) the phrase actually said is often NOT in the list: return -1 "
                 "whenever the transcript does not contain at least one clear mouth-shape match to the chosen phrase's key words; (5) 'high' only when "
                 "the transcript covers most of the phrase's words in order; a one- or two-word transcript never justifies 'high'.")
        lines = "\n".join(f"{i}. {c['text']}  (lips {c.get('prob', 0)*100:.0f}%)" for i, c in enumerate(candidates))
        user = f"Lip transcript (noisy): {transcript or '(nothing)'}\nContext: {json.dumps(context)}\nCandidates:\n{lines}\nAnswer with index (or -1), confidence, reason."
        schema = {"type": "object", "additionalProperties": False,
                  "properties": {"index": {"type": "integer"}, "confidence": {"type": "string", "enum": ["high", "medium", "low"]}, "reason": {"type": "string"}},
                  "required": ["index", "confidence", "reason"]}
        try:
            r = self.client.chat.completions.create(model=self.model, temperature=0,
                messages=[{"role": "system", "content": sys_p}, {"role": "user", "content": user}],
                response_format={"type": "json_schema", "json_schema": {"name": "pick", "strict": True, "schema": schema}})
            out = json.loads(r.choices[0].message.content)
            i = int(out.get("index", -1))
            return {"index": i if 0 <= i < len(candidates) else -1, "confidence": out.get("confidence", "low"), "reason": out.get("reason", "")}
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
