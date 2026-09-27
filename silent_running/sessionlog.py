"""Per-session JSONL log so a run can be diagnosed after the fact (lag, stalls, decode latency, voice timings, errors).

One file per server start: logs/session_<YYYYmmdd_HHMMSS>.jsonl, and logs/latest.jsonl points at it. Every line is
{"t": unix seconds, "rel": seconds since start, "ev": event name, ...fields}. Read it with scripts/session_report.py.
"""
import json, os, threading, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, "logs")
_lock = threading.Lock()
_fh = None
_t0 = None


def start(**fields):
    global _fh, _t0
    os.makedirs(LOG_DIR, exist_ok=True)
    _t0 = time.time()
    path = os.path.join(LOG_DIR, time.strftime("session_%Y%m%d_%H%M%S.jsonl", time.localtime(_t0)))
    _fh = open(path, "a", buffering=1)
    latest = os.path.join(LOG_DIR, "latest.jsonl")
    try:
        if os.path.islink(latest) or os.path.exists(latest):
            os.remove(latest)
        os.symlink(os.path.basename(path), latest)
    except OSError:
        pass
    log("start", **fields)
    return path


def log(ev, **fields):
    if _fh is None:
        return
    now = time.time()
    rec = {"t": round(now, 3), "rel": round(now - _t0, 3), "ev": ev}
    for k, v in fields.items():
        rec[k] = _jsonable(v)
    with _lock:
        _fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _jsonable(v):
    try:
        json.dumps(v)
        return v
    except (TypeError, ValueError):
        if isinstance(v, dict):
            return {str(k): _jsonable(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [_jsonable(x) for x in v]
        return repr(v)


# fields worth keeping from each broadcast event type (everything else in the event is UI payload)
_KEEP = {
    "feeds": None, "unit": None, "ascend": None,  # the nurse board's own events: they have their own log (data/unit/)
    "result": ("utt_id", "mode", "selected", "confidence", "margin", "n_frames", "duration", "source", "latency", "ahead", "in_inventory", "phrase_gap", "raw_greedy", "critical", "mouth_px"),
    "delivery": ("utt_id", "cached", "streamed", "t_synth", "total", "model"),
    "alert": ("utt_id", "text", "confidence"),
    "error": ("utt_id", "message"),
    "status": ("status", "stage", "utt_id"),
    "state": ("state",),
    "nbest": ("utt_id", "error"),
    "llm": ("utt_id", "accepted", "changed", "latency", "error"),
    "llm_pick": ("utt_id", "latency", "transcript", "visual_top", "visual_conf", "pick", "confidence", "gap", "verified", "applied", "new_confidence", "note", "error"),
    "confirm": None, "signal": None, "enroll": None, "log": None,
}


def from_broadcast(msg):
    """Log a UI broadcast, compacted. Unknown types are logged with all their scalar fields."""
    t = msg.get("type")
    keep = _KEEP.get(t, "scalars")
    if keep is None:
        return
    if keep == "scalars":
        fields = {k: v for k, v in msg.items() if k != "type" and isinstance(v, (str, int, float, bool, type(None)))}
    else:
        fields = {k: msg[k] for k in keep if k in msg}
        if t == "result" and "ranking" in msg and msg["ranking"]:
            fields["top"] = [[r.get("phrase"), round(r.get("final_prob", 0), 3)] for r in msg["ranking"][:3]]
    log(t, **fields)
