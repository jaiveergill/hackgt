"""The Impiricus seam: what the unit board hands to Impiricus Ascend, and what Ascend hands back to the nurse.

Impiricus has no public API, so this side is SIMULATED and every screen says so. The shapes follow Impiricus's own vocabulary:
  * Spark runs engagement journeys triggered by real-world events (a first-time prescription, ...). The board contributes a new
    trigger: a bedside request, de-identified (unit, bed, category, urgency, time-to-acknowledge; never a name or PHI).
  * Ascend connects HCPs to pharma resources in real time (dosing calculators, treatment information, patient resources, Wallet
    cards). The board surfaces one resource to the bedside nurse, a care-team audience DocUpdate (prescriber-only) cannot reach.
  * ION picks the next best action from engagement and clinical signals. Here it is a transparent rule over today's requests.
Engagement (a resource opened, a medical science liaison asked) is journaled: that is the metric Impiricus sells on.
"""
import json, os, threading, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# One Ascend-style resource per request category. `kind` uses Impiricus's resource types. Medical-affairs material only:
# never promotion, never in the alert path. Content is placeholder text for the demo and is labelled simulated in the UI.
RESOURCES = {
    "pain": {"kind": "Treatment information", "title": "ICU pain reassessment after a non-verbal report",
             "body": "CPOT / BPS reassessment within 30 min of an intervention; opioid-sparing adjuncts checklist; when to escalate to the intensivist.",
             "cta": "Open reassessment guide"},
    "breathing": {"kind": "Dosing calculator", "title": "Tracheostomy cuff pressure and suction reference",
                  "body": "Target cuff pressure 20-30 cmH2O; suction depth and catheter sizing by tube ID; humidification check.",
                  "cta": "Open calculator"},
    "care": {"kind": "Dosing calculator", "title": "PRN medication timing check",
             "body": "Last-dose interval against the order set; renal and hepatic adjustment prompts; what to document.",
             "cta": "Open calculator"},
    "needs": {"kind": "Patient resource", "title": "Oral care and hydration for the tracheostomy patient",
              "body": "Swab and moisturize protocol; NPO-safe comfort measures; family handout in 20+ languages.",
              "cta": "Open patient resource"},
    "comfort": {"kind": "Patient resource", "title": "Repositioning and pressure-injury prevention",
                "body": "Two-hourly turn schedule; Braden reassessment; heel offloading.",
                "cta": "Open patient resource"},
    "people": {"kind": "Wallet card", "title": "Family update card",
               "body": "A Wallet card the nurse forwards to the family: visiting hours, how to reach the unit, what the patient can communicate today.",
               "cta": "Send Wallet card"},
    "urgent": {"kind": "Treatment information", "title": "Airway emergency: displaced or blocked tracheostomy",
               "body": "Green/red algorithm; capnography; when to remove the tube and ventilate via the stoma or the mouth.",
               "cta": "Open algorithm"},
}
RESOURCE_FOR_LEVEL = {"urgent": "urgent"}


class AscendBridge:
    def __init__(self, emit=None, webhook=None, unit_name="ICU 4B"):
        self.emit = emit or (lambda m: None)
        self.webhook = webhook  # a real endpoint if Impiricus ever offers one; None = simulated delivery
        self.unit_name = unit_name
        self.lock = threading.RLock()  # snapshot() calls summary() under it
        self.journal = []       # everything handed to / back from Ascend today, newest last
        self.next_id = 1
        self.ask_counts = {}    # category -> requests today, for the ION-style next best action

    # ------------------------------------------------------------------ unit board -> Ascend (Spark triggers)
    def on_unit_event(self, m):
        """Called with every `unit` broadcast. Requests become Spark triggers; acknowledgements close the journey step."""
        ev = m.get("event")
        if ev == "patient":
            line = m.get("line") or {}
            level = line.get("level", "calm")
            cat = line.get("category")
            if cat:
                self.ask_counts[cat] = self.ask_counts.get(cat, 0) + 1
            if level == "calm":
                return
            a = m.get("alert") or {}
            self._send("spark.trigger", {"trigger": "bedside_request", "unit": self.unit_name, "bed": m["bed"]["bed"], "category": cat,
                                         "urgency": level, "alert_id": a.get("id"), "ts": line.get("ts")}, bed=m["bed"]["bed"], ts=line.get("ts"))
        elif ev == "ack":
            a = m.get("alert") or {}
            self._send("spark.trigger", {"trigger": "bedside_request_acknowledged", "unit": self.unit_name, "bed": a.get("bed"),
                                         "alert_id": a.get("id"), "time_to_acknowledge_s": round((a.get("ack_ts") or 0) - (a.get("ts") or 0), 1),
                                         "by": "care team"}, bed=a.get("bed"), ts=a.get("ack_ts"))

    def _send(self, kind, payload, bed=None, direction="out", ts=None):
        rec = {"id": self.next_id, "ts": ts or time.time(), "kind": kind, "direction": direction, "bed": bed, "payload": payload,
               "status": "simulated" if not self.webhook else "pending"}
        with self.lock:
            self.next_id += 1
            self.journal.append(rec); del self.journal[:-500]
        if self.webhook:
            threading.Thread(target=self._deliver, args=(rec,), daemon=True).start()
        self.emit({"type": "ascend", "event": rec, "summary": self.summary()})
        return rec

    def _deliver(self, rec):
        try:
            req = urllib.request.Request(self.webhook, data=json.dumps({"kind": rec["kind"], **rec["payload"]}).encode(),
                                         headers={"Content-Type": "application/json"}, method="POST")
            urllib.request.urlopen(req, timeout=5).read()
            rec["status"] = "delivered"
        except Exception as e:
            rec["status"] = f"failed: {e}"
        self.emit({"type": "ascend", "event": rec, "summary": self.summary()})

    # ------------------------------------------------------------------ Ascend -> nurse (resources, next best action)
    def resource_for(self, bed_view):
        """The one Ascend resource to show on a bed's detail: for the open alert if any, else the last request category."""
        a = bed_view.get("alert")
        if a:
            cat = RESOURCE_FOR_LEVEL.get(a["level"]) or self._cat_of(a["text"]) or "care"
            reason = f"open {a['level']} request: “{a['text']}”"
        else:
            cat, reason = self._last_request(bed_view)
        if cat not in RESOURCES:
            return None
        r = dict(RESOURCES[cat])
        r.update({"category": cat, "reason": reason, "source": "Impiricus Ascend (simulated)"})
        return r

    def _cat_of(self, text):
        try:
            from silent_running.context import PHRASE_CATEGORY
            return PHRASE_CATEGORY.get((text or "").lower())
        except Exception:
            return None

    def _last_request(self, bed_view):
        for line in reversed(bed_view.get("transcript") or []):
            if line.get("who") == "patient" and line.get("level") in ("request", "urgent") and line.get("category") in RESOURCES:
                return line["category"], f"last request today: “{line['text']}”"
        return None, None

    def next_best_action(self, bed_view):
        """ION-style: a transparent rule over today's requests at this bed. Returns None when there is nothing to say."""
        lines = [l for l in (bed_view.get("transcript") or []) if l.get("who") == "patient" and l.get("level") in ("request", "urgent")]
        if not lines:
            return None
        counts = {}
        for l in lines:
            counts[l.get("category")] = counts.get(l.get("category"), 0) + 1
        cat, n = max(counts.items(), key=lambda kv: kv[1])
        if n >= 2 and cat in RESOURCES:
            return {"category": cat, "count": n, "text": f"{n} {cat} requests at this bed today. Suggested: review with the intensivist at rounds; "
                                                         f"resource “{RESOURCES[cat]['title']}” is ready.", "basis": "rule: most frequent request category today, at least 2"}
        return None

    def engaged(self, action, bed, resource=None, by="charge nurse"):
        """The nurse acted on an Ascend resource: this is the engagement Impiricus measures."""
        return self._send(f"engagement.{action}", {"unit": self.unit_name, "bed": bed, "resource": resource, "by": by}, bed=bed, direction="in")

    # ------------------------------------------------------------------ views
    def summary(self):
        with self.lock:
            day0 = time.mktime(time.localtime()[:3] + (0, 0, 0, 0, 0, -1))
            today = [r for r in self.journal if r["ts"] >= day0]
            return {"triggers": sum(r["kind"] == "spark.trigger" for r in today),
                    "resources_opened": sum(r["kind"] == "engagement.opened" for r in today),
                    "msl_asks": sum(r["kind"] == "engagement.msl" for r in today),
                    "wallet_sent": sum(r["kind"] == "engagement.wallet" for r in today),
                    "delivery": "webhook " + self.webhook if self.webhook else "simulated (no public Impiricus API)"}

    def snapshot(self, limit=40):
        with self.lock:
            return {"summary": self.summary(), "events": self.journal[-limit:]}
