"""Low-confidence confirmation loop.

A confident result is spoken directly. A weak one becomes a question in the nurse's ear, "Sounds like: X?", and the
patient answers: nod = yes (speak X), shake = no (ask about the next alternative). No answer within TIMEOUT s of the
prompt finishing playing ends the loop without speaking anything, because an unconfirmed guess is never put in the
patient's mouth. Whoever plays the prompt (the UI, or the server's `say`) reports playback end via played(); if nobody
does, the question still times out PROMPT_WAIT + TIMEOUT s after it was asked.

Answers come from nonverbal `signal` events (nod/shake, blink code, thumbs), a mouthed yes/no, or the nurse's Y / N
keys. The loop only sees "yes" / "no" plus where it came from.
"""
import os, threading, time

SPEAK_CONF = float(os.environ.get("CONFIRM_SPEAK_CONF", 0.6))  # at or above this (and in-inventory): speak directly
MIN_ALT_CONF = 0.05          # alternatives below this probability are not offered
MAX_ATTEMPTS = 3             # candidates offered before giving up
TIMEOUT = float(os.environ.get("CONFIRM_TIMEOUT", 4.0))  # s the patient has to answer once the prompt has played
# s allowed for the prompt to play before the answer window starts anyway (nobody reported playback end). Prompts use the
# local system voice (no network): over the phrase bank "Sounds like: X?" lasts 2.2 s median, 3.2 s max (say -v Daniel),
# and a played() report arriving within PROMPT_WAIT + TIMEOUT still restarts the window.
PROMPT_WAIT = 4.0
SIGNAL_CONF = 0.5            # a gesture must be at least this confident to count as an answer


def answer_from_signal(sig):
    """Map a `signal` event to "yes" / "no" / None."""
    if sig["confidence"] < SIGNAL_CONF:  # validated by the server (server._signal_problem)
        return None
    kind, value = sig.get("kind"), sig.get("value")
    if kind == "nod":
        return "yes"
    if kind == "shake":
        return "no"
    if kind == "thumb" and value in ("up", "down"):
        return "yes" if value == "up" else "no"
    if kind == "blink_code" and value in ("yes", "no"):
        return value
    return None


def plan(result):
    """Phrase Mode result -> `decision` event: speak it, confirm it, or do nothing."""
    rows = [r for r in result.get("ranking", []) if not r.get("prefiltered_out")]
    alts = [{"text": r["phrase"], "confidence": round(r["final_prob"], 3)} for r in rows[:MAX_ATTEMPTS]
            if r is rows[0] or r["final_prob"] >= MIN_ALT_CONF]
    conf = result["confidence"]
    if result.get("critical") or (conf >= SPEAK_CONF and result.get("in_inventory", True)):
        action, reason = "speak", f"confidence {conf:.0%}" + (" (critical: escalated)" if result.get("critical") else "")
    elif alts:
        action = "confirm"
        reason = f"confidence {conf:.0%} < {SPEAK_CONF:.0%}" if conf < SPEAK_CONF else "weak match: free transcript fits better than any phrase"
    else:
        action, reason = "none", "no candidate"
    return {"type": "decision", "utt_id": result["utt_id"], "text": result["selected"], "confidence": round(conf, 3),
            "source": "lips", "reason": reason, "provider": "vsr", "alternatives": alts, "action": action}


class ConfirmLoop:
    """One pending question at a time. `emit(event)` broadcasts; `on_confirmed(text, info)` runs when the patient says yes."""

    def __init__(self, emit, on_confirmed, timeout=TIMEOUT):
        self.emit, self.on_confirmed, self.timeout = emit, on_confirmed, timeout
        self.lock = threading.Lock()
        self.cur = None   # {utt_id, alts, i, t0 (end of utterance), t_ask, token}
        self.timer = None
        self.token = 0    # invalidates stale timers

    def pending(self):
        return self.cur is not None

    def start(self, decision, t0=None):
        """Begin asking about decision["alternatives"] in order. Supersedes any pending question."""
        with self.lock:
            if self.cur:
                self._event("rejected", reason="superseded by a new utterance")
            self.cur = {"utt_id": decision["utt_id"], "alts": decision["alternatives"], "i": 0, "t0": t0 or time.time()}
            self._ask()

    def answer(self, ans, source="nurse", ts=None):
        """Apply a yes/no. Returns True if it was used. Gestures made before the question was asked are ignored."""
        with self.lock:
            c = self.cur
            if c is None or ans not in ("yes", "no") or (ts is not None and ts < c["t_ask"]):
                return False
            cand = c["alts"][c["i"]]["text"]
            now = time.time()
            lat = {"answer": round(now - c["t_ask"], 3), "since_utterance": round(now - c["t0"], 3)}
            if ans == "yes":
                self.cur = None
                self._event("confirmed", c=c, candidate=cand, by=source, say=cand, latency=lat)
                info = {"utt_id": c["utt_id"], "attempt": c["i"] + 1, "by": source, "latency": lat}
                threading.Thread(target=self.on_confirmed, args=(cand, info), daemon=True).start()
                return True
            last = c["i"] + 1 >= len(c["alts"])
            self._event("rejected", c=c, candidate=cand, by=source, latency=lat,
                        say="Okay. Please try again." if last else None)
            c["i"] += 1
            if last:
                self.cur = None
            else:
                self._ask()
            return True

    def played(self, utt_id, attempt):
        """The prompt for (utt_id, attempt) finished playing: the patient now has TIMEOUT s to answer."""
        with self.lock:
            c = self.cur
            if c is None or c["utt_id"] != utt_id or c["i"] + 1 != attempt or "t_played" in c:
                return False  # stale, or a second UI tab reporting the same prompt: only the first report starts the window
            c["t_played"] = time.time()
            self._arm(self.timeout)
            return True

    def cancel(self, reason="cancelled", ts=None):
        """End the pending question. With `ts` (when the superseding utterance ended), only a question asked before then."""
        with self.lock:
            if self.cur and (ts is None or ts >= self.cur["t_ask"]):
                self._event("rejected", reason=reason)
                self.cur = None

    # ---- internals (call with self.lock held)
    def _ask(self):
        c = self.cur
        cand = c["alts"][c["i"]]["text"]
        say = f"Sounds like: {cand}?"
        c["t_ask"] = time.time()
        c.pop("t_played", None)
        self._arm(PROMPT_WAIT + self.timeout)
        self._event("asking", candidate=cand, confidence=c["alts"][c["i"]]["confidence"], say=say, timeout=self.timeout,
                    latency={"since_utterance": round(c["t_ask"] - c["t0"], 3)}, remaining=[a["text"] for a in c["alts"][c["i"] + 1:]])

    def _arm(self, seconds):
        """(Re)start the timeout for the current question; any earlier timer becomes stale."""
        if self.timer:
            self.timer.cancel()
        self.token += 1
        self.cur["token"] = self.token
        self.timer = threading.Timer(seconds, self._on_timeout, args=(self.token,))
        self.timer.daemon = True
        self.timer.start()

    def _on_timeout(self, token):
        with self.lock:
            c = self.cur
            if c is None or c["token"] != token:
                return
            self.cur = None
            self._event("timeout", c=c, candidate=c["alts"][c["i"]]["text"], heard=c.get("t_played") is not None)

    def _event(self, state, c=None, **kw):
        c = c or self.cur
        self.emit({"type": "confirm", "utt_id": c["utt_id"], "candidate": kw.pop("candidate", c["alts"][c["i"]]["text"]),
                   "attempt": c["i"] + 1, "state": state, "ts": time.time(), **kw})
