"""The unit: what the charge nurse's dashboard shows. One bed is the live patient in front of the camera; the rest are
simulated so the unit looks like a unit (the dashboard says so). Every patient utterance becomes a transcript line and, when
it is a request or an emergency, an alert the nurse acknowledges; time-to-acknowledge is measured on every alert.

Events are appended to data/unit/<YYYY-MM-DD>.jsonl and replayed at startup, so "today" survives a restart.
"""
import json, os, random, threading, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNIT_DIR = os.path.join(ROOT, "data", "unit")

# Phrase category -> how the bed shows. `urgent` (critical phrases) is red; a request is yellow; everything else keeps the
# bed green: a calm patient talking is not a task for the nurse.
LEVELS = ("calm", "request", "urgent")
REQUEST_CATEGORIES = {"pain", "breathing", "needs", "comfort", "care", "people"}


def level_of(category, critical):
    if critical:
        return "urgent"
    return "request" if (category or "").lower() in REQUEST_CATEGORIES else "calm"


# ------------------------------------------------------------------------------------------------ simulated patients
SIM_BEDS = [  # (bed, initials, one-line reason for admission shown on the detail view)
    ("1", "M.R.", "post-op day 2, tracheostomy"),
    ("2", "A.K.", "ventilator-dependent, C4 spinal cord injury"),
    ("3", "D.L.", "ALS, non-verbal, mouth free"),
    ("5", "S.P.", "post-op day 1, laryngectomy"),
    ("6", "R.N.", "tracheostomy, weaning"),
    ("7", "E.W.", "ventilator-dependent, Guillain-Barré"),
]
SIM_NURSES = ["R. Okafor", "T. Nguyen", "L. Alvarez", "J. Park"]
SIM_LINES = {  # (text, category, critical). Weighted below.
    "calm": [("Yes", "answers", False), ("No", "answers", False), ("Thank you", "feelings", False), ("I am okay", "feelings", False),
             ("I am tired", "feelings", False), ("What time is it", "questions", False), ("I want to sleep", "feelings", False),
             ("I feel better today", "feelings", False), ("When is the doctor coming", "questions", False)],
    "request": [("I am thirsty", "needs", False), ("I need to be turned", "care", False), ("My back hurts", "pain", False),
                ("I need suction", "care", False), ("I am cold", "comfort", False), ("I want to see my wife", "people", False),
                ("I need the bedpan", "needs", False), ("The pain is getting worse", "pain", False), ("Please turn the light off", "comfort", False),
                ("I want my daughter", "people", False), ("I need my medication", "care", False), ("My mouth is dry", "needs", False)],
    "urgent": [("I can't breathe", "urgent", True), ("I am in a lot of pain", "urgent", True), ("I need help right now", "urgent", True),
               ("The tube is hurting me", "urgent", True), ("My chest hurts", "urgent", True)],
}
SIM_NURSE_REPLIES = {"calm": ["Okay, I'm here if you need anything.", "Rest now, I'll check back soon."],
                     "request": ["On my way.", "Got it, give me two minutes.", "I'll bring that right now."],
                     "urgent": ["I'm coming now.", "Stay with me, help is on the way."]}


def _pick_level(rng):
    r = rng.random()
    return "urgent" if r < 0.08 else "request" if r < 0.42 else "calm"


class Unit:
    def __init__(self, real_bed="4", real_initials="J.G.", real_note="tracheostomy, day 3 · live camera", emit=None, simulate=True, seed=None):
        self.emit = emit or (lambda msg: None)
        self.lock = threading.RLock()
        self.real_bed = real_bed
        self.beds = {}
        self.alerts = {}
        self.next_alert = 1
        self._add_bed(real_bed, real_initials, real_note, real=True)
        for bed, ini, note in SIM_BEDS:
            self._add_bed(bed, ini, note, real=False)
        self.day = time.strftime("%Y-%m-%d")
        self.path = os.path.join(UNIT_DIR, f"{self.day}.jsonl")
        self._fh = None
        self.rng = random.Random(seed)
        self._replay()
        if simulate:
            if not any(len(b["transcript"]) for b in self.beds.values() if not b["real"]):
                self._seed_history()
            threading.Thread(target=self._simulate, daemon=True).start()

    # ---------------------------------------------------------------------------------------- model
    def _add_bed(self, bed, initials, note, real):
        self.beds[bed] = {"bed": bed, "initials": initials, "note": note, "real": real, "status": "calm", "last_text": None, "last_ts": None,
                          "last_who": None, "patient_text": None, "patient_ts": None, "transcript": [], "open_alert": None}

    def _bed_status(self, b):
        opens = [self.alerts[a] for a in (b["open_alert"],) if a is not None]
        b["status"] = opens[0]["level"] if opens else "calm"

    def _line(self, bed, who, text, ts=None, **kw):
        b = self.beds[bed]
        rec = {"ts": ts or time.time(), "who": who, "text": text, **kw}
        b["transcript"].append(rec)
        b["last_text"], b["last_ts"], b["last_who"] = text, rec["ts"], who
        if who == "patient":
            b["patient_text"], b["patient_ts"] = text, rec["ts"]
        return rec

    def patient_said(self, bed, text, category=None, critical=False, confidence=None, ts=None, _replay=False):
        """A patient's words: transcript line; a request or emergency opens (or escalates) the bed's alert."""
        with self.lock:
            level = level_of(category, critical)
            rec = self._line(bed, "patient", text, ts, level=level, category=category, confidence=confidence)
            b = self.beds[bed]
            alert = None
            if level != "calm":
                cur = self.alerts.get(b["open_alert"]) if b["open_alert"] is not None else None
                if cur is None:
                    alert = {"id": self.next_alert, "bed": bed, "text": text, "level": level, "ts": rec["ts"], "ack_ts": None, "ack_by": None}
                    self.next_alert += 1
                    self.alerts[alert["id"]] = alert
                    b["open_alert"] = alert["id"]
                else:  # the bed already has an open alert: it now says the latest thing, at the higher of the two levels
                    alert = cur
                    alert["text"] = text
                    if LEVELS.index(level) > LEVELS.index(alert["level"]):
                        alert["level"], alert["ts"] = level, rec["ts"]  # escalation restarts the clock: the emergency is new
                self._bed_status(b)
            if not _replay:
                self._persist({"ev": "patient", "bed": bed, "text": text, "category": category, "critical": bool(critical), "confidence": confidence, "ts": rec["ts"]})
                self._emit("patient", bed, alert)
            return rec

    def nurse_said(self, bed, text, by=None, ts=None, _replay=False):
        """What the nurse said at the bedside (the glasses' microphone, or typed): logged as the conversation's other half."""
        with self.lock:
            rec = self._line(bed, "nurse", text, ts, by=by)
            if not _replay:
                self._persist({"ev": "nurse", "bed": bed, "text": text, "by": by, "ts": rec["ts"]})
                self._emit("nurse", bed, None)
            return rec

    def ack(self, alert_id, by="charge nurse", ts=None, _replay=False):
        with self.lock:
            a = self.alerts.get(int(alert_id))
            if a is None:
                raise KeyError(f"no alert {alert_id}")
            if a["ack_ts"] is None:
                a["ack_ts"], a["ack_by"] = ts or time.time(), by
                b = self.beds[a["bed"]]
                if b["open_alert"] == a["id"]:
                    b["open_alert"] = None
                self._bed_status(b)
                if not _replay:
                    self._persist({"ev": "ack", "alert": a["id"], "by": by, "ts": a["ack_ts"]})
                    self._emit("ack", a["bed"], a)
            return a

    # ---------------------------------------------------------------------------------------- views
    def _day_start(self):
        t = time.localtime()
        return time.mktime((t.tm_year, t.tm_mon, t.tm_mday, 0, 0, 0, 0, 0, -1))

    def metrics(self):
        with self.lock:
            day0 = self._day_start()
            today = [a for a in self.alerts.values() if a["ts"] >= day0]
            acked = [a for a in today if a["ack_ts"] is not None]
            open_ = [a for a in self.alerts.values() if a["ack_ts"] is None]
            resp = [max(a["ack_ts"] - a["ts"], 0.0) for a in acked]
            return {"open": len(open_), "open_urgent": sum(a["level"] == "urgent" for a in open_),
                    "avg_response_s": (sum(resp) / len(resp)) if resp else None,
                    "requests_today": len(today), "beds": len(self.beds),
                    "requests_per_bed": len(today) / max(len(self.beds), 1)}

    def bed_view(self, bed, transcript=False):
        b = self.beds[bed]
        v = {k: v for k, v in b.items() if k != "transcript"}
        v["alert"] = self.alerts.get(b["open_alert"]) if b["open_alert"] is not None else None
        if transcript:
            day0 = self._day_start()
            v["transcript"] = [r for r in b["transcript"] if r["ts"] >= day0][-400:]
        return v

    def snapshot(self):
        with self.lock:
            beds = sorted(self.beds.values(), key=lambda b: int(b["bed"]) if b["bed"].isdigit() else 99)
            alerts = sorted(self.alerts.values(), key=lambda a: (a["ack_ts"] is not None, -LEVELS.index(a["level"]), a["ts"]))
            return {"real_bed": self.real_bed, "beds": [self.bed_view(b["bed"]) for b in beds], "alerts": alerts[:60], "metrics": self.metrics(),
                    "now": time.time()}

    def _emit(self, ev, bed, alert):
        with self.lock:
            self.emit({"type": "unit", "event": ev, "bed": self.bed_view(bed), "line": self.beds[bed]["transcript"][-1] if ev != "ack" else None,
                       "alert": alert, "metrics": self.metrics(), "now": time.time()})

    # ---------------------------------------------------------------------------------------- persistence
    def _persist(self, rec):
        if self._fh is None:
            os.makedirs(UNIT_DIR, exist_ok=True)
            self._fh = open(self.path, "a", buffering=1)
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _replay(self):
        if not os.path.exists(self.path):
            return
        for line in open(self.path):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("bed") is not None and r["bed"] not in self.beds:
                continue
            if r["ev"] == "patient":
                self.patient_said(r["bed"], r["text"], r.get("category"), r.get("critical"), r.get("confidence"), ts=r["ts"], _replay=True)
            elif r["ev"] == "nurse":
                self.nurse_said(r["bed"], r["text"], r.get("by"), ts=r["ts"], _replay=True)
            elif r["ev"] == "ack" and r["alert"] in self.alerts:
                self.ack(r["alert"], r.get("by"), ts=r["ts"], _replay=True)

    # ---------------------------------------------------------------------------------------- simulation
    def _sim_event(self, bed, level, ts, ack_after):
        text, cat, crit = self.rng.choice(SIM_LINES[level])
        self.patient_said(bed, text, cat, crit, confidence=round(self.rng.uniform(0.7, 0.98), 2), ts=ts)
        if level != "calm":
            nurse = self.rng.choice(SIM_NURSES)
            a = self.beds[bed]["open_alert"]
            return (a, nurse, ts + ack_after)
        return None

    def _seed_history(self):
        """A day's worth of quiet activity on the simulated beds (backdated 6 hours), so the metrics and transcripts are not
        empty when the demo starts."""
        now = time.time()
        for bed, _, _ in SIM_BEDS:
            t = now - 6 * 3600 + self.rng.uniform(0, 900)
            while t < now - 300:
                level = _pick_level(self.rng)
                p = self._sim_event(bed, level, t, self.rng.uniform(45, 240) if level == "request" else self.rng.uniform(20, 90))
                if p:  # acknowledged before the bed's next line (the next line is >= 10 min later)
                    self.ack(p[0], by=p[1], ts=p[2])
                    self.nurse_said(bed, self.rng.choice(SIM_NURSE_REPLIES[level]), by=p[1], ts=p[2] + self.rng.uniform(5, 40))
                t += self.rng.uniform(600, 1800)

    def _simulate(self):
        """Live: every 20-60 s one simulated bed says something; its requests are acknowledged by unit staff after 30-120 s
        unless the charge nurse gets there first. The live bed is never touched."""
        due = []  # (t_ack, alert_id, nurse, bed, level)
        time.sleep(8)
        while True:
            now = time.time()
            for item in list(due):
                if item[0] <= now:
                    due.remove(item)
                    a = self.alerts.get(item[1])
                    if a and a["ack_ts"] is None:
                        self.ack(a["id"], by=item[2])
                        self.nurse_said(item[3], self.rng.choice(SIM_NURSE_REPLIES[item[4]]), by=item[2])
            bed = self.rng.choice(SIM_BEDS)[0]
            if self.beds[bed]["open_alert"] is None or self.rng.random() < 0.3:
                level = _pick_level(self.rng)
                p = self._sim_event(bed, level, time.time(), self.rng.uniform(30, 120))
                if p:
                    due.append((p[2], p[0], p[1], bed, level))
            time.sleep(self.rng.uniform(20, 60))
