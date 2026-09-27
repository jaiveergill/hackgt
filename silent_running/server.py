"""Silent Running server: webcam -> face tracking -> pretrained VSR -> constrained/contextual decoding -> UI.

  python -m silent_running.server [--camera 0] [--port 8000] [--device mps] [--llm-provider grok|openai] [--llm MODEL]
"""
import os, sys, json, time, asyncio, threading, argparse, subprocess, re, queue, traceback
sys.setswitchinterval(0.0005)  # many tiny torch ops on MPS/CPU must not wait 5 ms behind the camera thread each
import av
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from silent_running.vsr import VSREngine, load_phrases, load_phrase_table, _read_video
from silent_running.camera_proc import CameraProcess
from silent_running.signals.aggregate import nonverbal_dict
from silent_running.context import ContextStore, LLMInterpreter
from silent_running import tts as eltts
from silent_running import prosody
from silent_running import sessionlog
from silent_running import captures
from silent_running.unit import Unit
from silent_running.ambient import AmbientCamera
from silent_running.ascend import AscendBridge

STATIC = os.path.join(ROOT, "silent_running", "static")
app = FastAPI(title="Silent Running")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

LLM_MARGIN = 3.0  # nats; an LLM proposal is accepted only if the visual model scores it within this of the raw top hypothesis
STATE = {"auto_listen": False, "voice": None, "expressive": True, "emotion_override": None, "llm_enabled": True, "status": "idle", "utt_id": 0, "warm": False, "pace": False}
clients = set()
loop = None
engine = camera = context = llm = capture = None
unit = ambient = ascend = None  # the charge nurse's unit (silent_running/unit.py) and the laptop's ambient camera (ambient.py)
phrases = load_phrases()
PHRASE_TABLE = load_phrase_table()
LOG = []  # conversation log: [{ts, who, text, ...}]
work_lock = threading.Lock()
SIGNAL_KINDS = ("nod", "shake", "blink_code", "fingers", "thumb", "point", "pain", "mouthing")  # shared event contract
SIGNAL_CLOCK_SKEW = 60.0  # s; signal.ts must be time.time(), not a monotonic/media clock
signals = queue.Queue()  # camera signals, handled in arrival order off the camera reader thread


def broadcast(msg):
    if capture:
        capture.event(msg)
    try:
        sessionlog.from_broadcast(msg)
    except Exception as e:
        print("[sessionlog]", e)
    data = json.dumps(msg)
    async def _send():
        dead = []
        for ws in list(clients):
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            clients.discard(ws)
    if loop is not None:
        asyncio.run_coroutine_threadsafe(_send(), loop)


def set_status(s, **kw):
    STATE["status"] = s
    broadcast({"type": "status", "status": s, **kw})


# ----------------------------------------------------------------------------- decoding pipeline
def _read(rois, on_greedy=None):
    """The model's reading of an utterance's mouth crops (work_lock held). No events and no state: it can run before the
    utterance is confirmed (run_decode's `ahead`). on_greedy(read) runs once the free transcript is known, before the beam
    search. -> {"t": [start, input, encode, greedy, beam], "error" | "enc", "greedy", "nbest", ...}"""
    r = {"t": [time.time()]}
    try:
        x = engine.to_model_input(rois)
    except Exception as e:
        r["error"] = f"Mouth crop failed: {e}"
        return r
    r["n_frames"] = int(x.shape[1])
    r["t"].append(time.time())
    r["enc"] = engine.encode(x)
    r["t"].append(time.time())
    r["greedy"] = engine.ctc_greedy(r["enc"])
    r["t"].append(time.time())
    if on_greedy:
        on_greedy(r)
    if not r["greedy"].strip():  # no speech: nothing more to read
        return r
    r["nbest"] = engine.beam_search(r["enc"], 5)
    r["t"].append(time.time())
    r["timing"] = _safe_timing(r["enc"], r["nbest"][0]["text"])
    return r


def _stages(r, t_crop):
    """Seconds per stage of a read; crop includes the capture process's (t_crop)."""
    t = r["t"]
    st = {"crop": t_crop + t[1] - t[0], "encode": t[2] - t[1], "greedy": t[3] - t[2]}
    if len(t) > 4:
        st["beam"] = t[4] - t[3]
    return st


class _Ahead:
    """A hands-free utterance the capture process sent before the hang confirmed its end (camera_proc `ahead`)."""
    TIMEOUT = 30.0  # s; the confirmation comes ~0.2 s later, or once frames flow again after a stall; a dead capture process never sends it

    def __init__(self):
        self.ev, self.confirmed, self.t = threading.Event(), False, None

    def settle(self, confirmed):
        self.confirmed, self.t = confirmed, time.time()
        self.ev.set()

    def wait(self):
        """True once the end is confirmed; False if the mouth moved again (or nothing came)."""
        return self.ev.wait(self.TIMEOUT) and self.confirmed


AHEAD = {}  # id -> _Ahead, from the utterance sent ahead until its confirmation (both arrive on the camera reader thread, in order)


def run_decode(rois, duration, source="webcam", label=None, t_crop=0.0, expression=None, nonverbal=None, mouth_px=None, ahead=None):
    """Full pipeline on an utterance's mouth crops: the beam n-best, then (with the LLM on) its interpretation (_bg_llm).
    Emits incremental events so the UI can show progress.
    ahead: an _Ahead. The model reads the utterance at once without showing anything, and it is published only once its end
    is confirmed, with latencies from the confirmation (the reading's own stages go under "ahead"); dropped if not."""
    read = None
    if ahead is not None:
        with work_lock:
            read = _read(rois)
        if not ahead.wait():
            return None
    t_queued = time.time()
    t0 = ahead.t if ahead else t_queued - t_crop  # end of the utterance: every latency below includes the wait for work_lock
    with work_lock:
        wait = time.time() - t_queued
        STATE["utt_id"] += 1
        uid = STATE["utt_id"]
        # live camera only by default: file playback and decode_file are replays of known clips (SR_CAPTURE=all: file playback too)
        if capture and (source.split("-")[0] in ("webcam", "usb", "stream", "serial") or (os.environ.get("SR_CAPTURE") == "all" and source.startswith("file"))):
            capture.utterance(uid, rois, {"kind": source, **((camera.source_info or {}) if camera else {})})
        def show_raw(r):  # the free transcript; while the beam search runs unless the reading ran ahead
            st = _stages(r, t_crop)
            broadcast({"type": "raw", "utt_id": uid, "stage": "greedy", "text": r["greedy"], "n_frames": r["n_frames"], "duration": duration,
                       "latency": {"lock": wait, "crop": st["crop"], "encode": st["encode"], "greedy": st["greedy"]}})
            if r["greedy"].strip() and ahead is None:
                set_status("processing", utt_id=uid, stage="beam")
        if read is None:
            set_status("processing", utt_id=uid, stage="encode")
            read = _read(rois, show_raw)
        elif "error" not in read:
            show_raw(read)
        if "error" in read:
            set_status("idle")
            broadcast({"type": "error", "utt_id": uid, "message": read["error"]})
            return None
        enc, greedy, st = read["enc"], read["greedy"], _stages(read, t_crop)
        t_done = max(read["t"][-1], t_queued + wait)
        if ahead is None:
            latency = {"lock": wait, **{k: v for k, v in st.items() if k != "greedy"}, "total": t_done - t0}
        else:  # the reading ran during the hang: what is left is the part of it after the confirmation, and the lock
            latency = {"lock": wait, "read": max(read["t"][-1] - t0, 0.0), "total": t_done - t0}
        result = {"utt_id": uid, "raw_greedy": greedy, "n_frames": read["n_frames"], "duration": duration, "source": source, "label": label,
                  "mouth_px": mouth_px,  # median mouth width (camera px) while mouthing; the crop reads 45 (camera_proc.MouthPixels)
                  "expression": expression or {"emotion": "neutral", "intensity": 0.0}}
        if ahead is not None:
            result["ahead"] = {**st, "before_end": round(t0 - read["t"][-1], 3)}  # > 0: the reading was done before the end was confirmed
        result["nonverbal"] = nonverbal or nonverbal_dict(result["expression"])  # no camera (decode_file): every entry absent
        LAST["enc"] = enc; LAST["utt_id"] = uid
        if not greedy.strip():
            # CTC saw no speech-like mouth movement at all. The attention decoder would hallucinate fluent text here.
            set_status("idle")
            broadcast({"type": "error", "utt_id": uid, "message": "No speech detected (mouth did not move enough). Mouth the phrase clearly while holding Listen."})
            result.update({"selected": None, "no_speech": True})
            return result
        result["timing"] = LAST["timing"] = read["timing"]
        nbest = read["nbest"]
        probs = _softmax([h["score"] for h in nbest])
        for h, p in zip(nbest, probs):
            h["prob"] = p
        result.update({"nbest": nbest, "selected": _pretty(nbest[0]["text"]), "confidence": probs[0], "context": context.snapshot(), "latency_total": latency["total"]})
        _prefetch_voice(result["selected"], result["expression"])
        interpret = STATE["llm_enabled"] and llm is not None
        if not interpret:  # the reading is final: an alert before the result, whose speech it replaces
            _patient_final(result["selected"], uid, probs[0])
        set_status("idle")
        broadcast({"type": "result", **result, "latency": latency})
        _log("patient", result["selected"], confidence=probs[0], emotion=(expression or {}).get("emotion"))
        if interpret:
            threading.Thread(target=_bg_llm, args=(nbest, uid, enc, probs[0]), daemon=True).start()
        return result


def _bg_llm(nbest, uid, enc, confidence):
    """The LLM PROPOSES up to 3 sentences the patient most plausibly meant (n-best + context), the visual model VERIFIES
    them in one batch against the video. The LLM's most likely proposal within LLM_MARGIN nats of the raw top hypothesis
    wins; none -> the raw top stands. Every proposal and the verdict are broadcast so the UI shows exactly what happened.
    The verdict goes out as soon as the reply's sentences are in; its reason follows (`llm_reason`)."""
    t0 = time.time()
    ctx = context.snapshot()
    ctx["recent_utterances"] = ctx.pop("history", [])
    top = nbest[0]["text"].strip().upper()
    verdict = False
    for kind, value in llm.propose_stream([{"text": h["text"], "score": h["score"]} for h in nbest], ctx):
        if kind == "error":
            if not verdict:  # the raw reading is what the UI says
                _patient_final(_pretty(top), uid, confidence)
            broadcast({"type": "llm_reason" if verdict else "llm", "utt_id": uid, "error": value, "latency": time.time() - t0})
            return
        if kind == "reason":
            broadcast({"type": "llm_reason", "utt_id": uid, "reason": value, "latency": time.time() - t0})
            continue
        props, verdict = value, True
        with work_lock:
            sc = {r["phrase"]: r["score"] for r in engine.score_phrases(enc, list(dict.fromkeys([top] + props)))}
        gaps = {p: sc[p] - sc[top] for p in props}
        chosen = next((p for p in props if gaps[p] >= -LLM_MARGIN), None)
        proposal = chosen or props[0]
        _patient_final(_pretty(chosen or top), uid, confidence)
        broadcast({"type": "llm", "utt_id": uid, "proposal": _pretty(proposal), "gap": gaps[proposal],
                   "accepted": chosen is not None, "changed": proposal != top, "corrected": _pretty(chosen or top),
                   "alternatives": [{"text": _pretty(p), "gap": gaps[p], "fits": gaps[p] >= -LLM_MARGIN} for p in props],
                   "latency": time.time() - t0, "model": f"{llm.provider} {llm.model}"})


LAST = {"enc": None, "utt_id": 0}


def _norm(text):
    return " ".join(re.findall(r"[a-z']+", text.lower()))


CRITICAL = {_norm(r["phrase"]) for r in PHRASE_TABLE if r["critical"]}  # phrases.txt lines marked " !"
CATEGORY_OF = {_norm(r["phrase"]): r["category"] for r in PHRASE_TABLE}


def _patient_final(text, uid, confidence):
    """What the patient is about to be heard saying is settled: into the history (the next utterance's context; never the
    utterance still being interpreted, or the LLM is told the raw reading was already said and repeats it), onto the nurse
    board's bed transcript, and the full-screen alert (the UI announces it twice, urgently) if it is a critical phrase. Sent
    before the event whose speech it replaces."""
    context.add_history(text)
    critical = _norm(text) in CRITICAL
    if unit:
        unit.patient_said(unit.real_bed, text, CATEGORY_OF.get(_norm(text)), critical, confidence)
    if critical:
        broadcast({"type": "alert", "utt_id": uid, "text": text, "confidence": confidence, "ts": time.time()})


def _on_camera_signal(sig):
    """Called on the camera reader thread, which must stay a pure dispatcher (it also carries previews and utterances)."""
    signals.put(sig)


def _signal_worker():
    """Handles camera signals in arrival order."""
    while True:
        sig = signals.get()
        try:
            _on_signal(sig)
        except Exception as e:  # a bug here must not silently stop every later signal: log it and show it
            traceback.print_exc()
            broadcast({"type": "error", "message": f"signal handler failed on {sig!r}: {e!r}"})


def _json_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)  # numpy float32 / bool would break json.dumps or mean nothing


def _signal_problem(sig):
    """What is wrong with a `signal` message under the shared contract, or None if it is well-formed."""
    if not isinstance(sig, dict):
        return f"signal must be a dict, got {type(sig).__name__}"
    if sig.get("kind") not in SIGNAL_KINDS:
        return f"signal.kind {sig.get('kind')!r} is not one of {', '.join(SIGNAL_KINDS)}"
    c = sig.get("confidence")
    if not _json_number(c) or not 0.0 <= c <= 1.0:
        return f"signal.confidence {c!r} ({type(c).__name__}) must be a number in 0-1"
    v = sig.get("value")
    if v is not None and not isinstance(v, (str, bool, int, float)):
        return f"signal.value {v!r} ({type(v).__name__}) must be a string, number, bool or null"
    if "ts" in sig and not (_json_number(sig["ts"]) and abs(sig["ts"] - time.time()) < SIGNAL_CLOCK_SKEW):
        return f"signal.ts {sig['ts']!r} must be a time.time() float (wall clock, within {SIGNAL_CLOCK_SKEW:.0f} s of the server's)"
    return None


def _on_signal(sig):
    """Nonverbal signal from the camera process: forward to the UI."""
    problem = _signal_problem(sig)
    if problem:
        print("[signal] rejected:", problem)
        broadcast({"type": "error", "message": f"malformed camera signal: {problem}"})
        return
    broadcast({"type": "signal", "ts": time.time(), **sig})


def _log(who, text, **kw):
    rec = {"ts": time.time(), "who": who, "text": text, **kw}
    LOG.append(rec); del LOG[:-200]
    broadcast({"type": "log", "entry": rec})
    if who == "nurse" and unit:  # the bedside conversation's other half, documented on the bed
        unit.nurse_said(unit.real_bed, text, by="bedside nurse")


NURSE_PAUSE = 0.7  # s of silence after speech that ends the nurse's question
NURSE_SILENCE_DB = -35  # dB; quieter than this is silence (a louder room never goes quiet: the recording runs to its limit)


def record_until_silence(wav, seconds, source=("-f", "avfoundation", "-i", ":0")):
    """Record `source` (the laptop mic) to a 16 kHz mono wav until NURSE_PAUSE s of silence follow speech, at most `seconds`.
    A fixed 5 s recording made even a 1 s question wait 5 s before transcription began. Returns the seconds recorded."""
    t0 = time.time()
    p = subprocess.Popen(["ffmpeg", "-y", "-hide_banner", "-nostats", *source, "-t", str(seconds), "-af", f"silencedetect=noise={NURSE_SILENCE_DB}dB:d={NURSE_PAUSE}",
                          "-ar", "16000", "-ac", "1", wav], stdin=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    heard = False  # speech so far: a silence that ended, or sound before the first silence
    for line in p.stderr:
        m = re.search(r"silence_(start|end): (-?[\d.]+)", line)
        if not m:
            continue
        if m.group(1) == "end" or float(m.group(2)) > 0.3:
            heard = True
        if m.group(1) == "start" and heard:
            p.stdin.write("q"); p.stdin.flush()  # ffmpeg's own stop: the wav is finalized
            break
    p.communicate()
    return time.time() - t0


def nurse_listen(seconds=5.0):
    """Record the laptop mic (the caregiver speaking) until they pause, transcribe with OpenAI, and drop it into context as
    the nurse prompt."""
    import tempfile
    set_status("nurse_listening")
    wav = tempfile.mktemp(suffix=".wav")
    sessionlog.log("nurse_recorded", seconds=round(record_until_silence(wav, seconds), 2))
    set_status("idle")
    if not os.path.exists(wav):
        broadcast({"type": "error", "message": "microphone capture failed"}); return
    try:
        from openai import OpenAI
        with open(wav, "rb") as f:
            text = OpenAI().audio.transcriptions.create(model="gpt-4o-transcribe", file=f, language="en").text.strip()
    except Exception as e:
        broadcast({"type": "error", "message": f"transcription failed: {e}"}); return
    finally:
        try: os.remove(wav)
        except Exception: pass
    if not text:
        broadcast({"type": "error", "message": "nurse: nothing heard"}); return
    context.update(last_prompt=text)
    _log("nurse", text)
    broadcast({"type": "context", "context": context.snapshot()})


def _safe_timing(enc, text):
    try:
        return prosody.video_word_timing(engine, enc, text)
    except Exception as e:
        print("[timing]", e); return None


def _link_monitor():
    """Every 5 s: the capture process's frame stats (fps, gaps) and, for a network camera, one ping to the board.
    Lag that the user sees is diagnosable afterwards from these two lines alone."""
    while True:
        time.sleep(5.0)
        try:
            cam = camera
            if cam is None:
                continue
            st = cam.stats
            if st:
                sessionlog.log("camera_stats", **st)
            info = cam.source_info or {}
            host = None
            if info.get("kind") == "stream":
                m = re.match(r"https?://([^/:]+)", info.get("url", ""))
                host = m.group(1) if m else None
            if host:
                p = subprocess.run(["ping", "-c", "3", "-i", "0.2", "-W", "1000", host], capture_output=True, text=True, timeout=6)
                rtts = [float(x) for x in re.findall(r"time=([\d.]+)", p.stdout)]
                loss = re.search(r"([\d.]+)% packet loss", p.stdout)
                sessionlog.log("ping", host=host, rtt_ms=[round(r, 1) for r in rtts], max_ms=round(max(rtts), 1) if rtts else None,
                               loss_pct=float(loss.group(1)) if loss else None)
            cpu = os.getloadavg()[0]
            sessionlog.log("host", load1=round(cpu, 2), busy=work_lock.locked(), status=STATE.get("status"), warm=STATE.get("warm"))
        except Exception as e:
            sessionlog.log("monitor_error", error=repr(e))


def _load_aligner():
    """The MMS forced aligner behind "match my pace" (prosody.align_audio_words): +1.5 GB resident and 2.6-25 s to load,
    measured. It loads when pace matching is switched on, so a server that never uses it never pays for it; the UI shows
    "warming up" meanwhile, since the load competes with decodes for CPU."""
    if prosody._mms is not None:
        return
    STATE["warm"] = False; broadcast({"type": "state", "state": STATE})
    try:
        t = time.time(); prosody._aligner(); print(f"[prosody] MMS aligner ready in {time.time()-t:.1f}s")
    except Exception as e:
        STATE["pace"] = False
        broadcast({"type": "error", "message": f"match my pace switched off: the voice aligner failed to load: {e}"})
    STATE["warm"] = True; broadcast({"type": "state", "state": STATE})


def _keep_warm():
    """Apple GPU clocks down after ~1 s idle and the next encode costs 0.5-1.5 s instead of 0.05 s.
    A tiny dummy encode (24 frames) every 0.4 s (0.2 s while listening) keeps the demo path fast. It holds the GPU for
    60-100 ms, so it skips a tick while a hands-free utterance may be ending (its mouth is still): that utterance's
    reading would wait behind it."""
    dummy = torch.zeros(1, 24, 88, 88)
    while True:
        try:
            if not work_lock.locked() and not ((camera.meta if camera else None) or {}).get("mouth_still"):
                engine.encode(dummy)
        except Exception as e:
            print("[keepwarm]", e)
        time.sleep(0.2 if (camera and camera.listening) else 0.4)


def _softmax(xs):
    xs = np.array(xs, dtype=np.float64)
    e = np.exp(xs - xs.max())
    return (e / e.sum()).tolist()


def _pretty(s):
    s = s.strip().lower()
    s = re.sub(r"\bi\b", "I", s)
    s = re.sub(r"\bi'", "I'", s)
    return s[:1].upper() + s[1:] if s else s


# ----------------------------------------------------------------------------- HTTP
@app.get("/")
def index():
    return HTMLResponse(open(os.path.join(STATIC, "index.html")).read())


async def mjpeg():
    # Async so the stream is cancelled when the client disconnects or the server shuts down. As a sync generator it ran
    # in a worker thread stuck in time.sleep: a closed tab left it looping forever and SIGTERM never finished (1 GB leaked).
    boundary = b"--frame"
    last = None
    while True:
        jpg = camera.preview_jpeg if camera else None
        if jpg is not None and jpg is not last:
            last = jpg
            yield boundary + b"\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n"
        await asyncio.sleep(1 / 20)


@app.get("/stream")
def stream():
    return StreamingResponse(mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")


# ----------------------------------------------------------------------------- charge nurse dashboard
@app.get("/dashboard")
def dashboard():
    return HTMLResponse(open(os.path.join(STATIC, "dashboard.html")).read())


async def _ambient_mjpeg():
    boundary = b"--frame"
    last = None
    while True:
        jpg = ambient.jpeg if ambient else None
        if jpg is not None and jpg is not last:
            last = jpg
            yield boundary + b"\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n"
        await asyncio.sleep(1 / 12)


@app.get("/ambient")
def ambient_stream():
    """The laptop's own camera (ambient overview of the room), preview only."""
    if ambient is None:
        return JSONResponse({"error": "ambient camera off (--ambient none)"}, status_code=404)
    return StreamingResponse(_ambient_mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")


FEED_LABELS = {"stream": "Glasses camera", "serial": "Glasses camera (USB)", "webcam": "Laptop webcam", "usb": "USB camera", "file": "Recording"}


def feeds_snapshot():
    """What the board shows about its two feeds, from the real sources: the bedside camera (the glasses, or whatever stands
    in for them right now) and the laptop's ambient camera."""
    info = (camera.source_info or {}) if camera else {}
    meta = camera.meta if camera else {}
    kind = info.get("kind")
    bedside = {"kind": kind, "opened": bool(camera and camera.opened), "fatal": camera.fatal if camera else "no camera",
               "label": FEED_LABELS.get(kind, "Camera"), "glasses": kind in ("stream", "serial"),
               "detail": info.get("url") or info.get("port") or (os.path.basename(info["path"]) if info.get("path") else None) or (f"index {info['index']}" if info.get("index") is not None else None),
               "fps": meta.get("fps"), "face": meta.get("face"), "mouth_px": meta.get("mouth_px"),
               "stats": camera.stats if camera else None}
    amb = ambient.info() if ambient else None
    if amb:
        amb["label"] = "Laptop camera"
        amb["same_as_bedside"] = kind == "webcam" and amb.get("opened") and info.get("index") == amb.get("index")
    return {"bedside": bedside, "ambient": amb}


def _board_monitor():
    """Every 2 s: if the bedside camera's source, its liveness, or the ambient camera's state changed, log it on the unit (the
    board's log) and push the new feeds to the board. A camera that stalls or reconnects shows up within 2 s."""
    last = None
    while True:
        time.sleep(2.0)
        try:
            if unit is None:
                continue
            f = feeds_snapshot()
            b, a = f["bedside"], f["ambient"] or {}
            key = (b["kind"], b["opened"], b["detail"], bool(a.get("opened")), a.get("index"), (a.get("error") or "")[:40], bool(b["stats"] and b["stats"].get("fps", 1) == 0))
            if key != last:
                txt = (f"Bedside feed: {b['label']}" + (f" ({b['detail']})" if b["detail"] else "") + (", live" if b["opened"] else f", not delivering ({b['fatal'] or 'stalled'})"))
                txt += " · Ambient: " + ("laptop camera live" if a.get("opened") else f"unavailable ({a.get('error') or 'starting'})" if a else "off")
                unit.system(txt, bedside=b["kind"], bedside_live=b["opened"], ambient_live=bool(a.get("opened")))
                last = key
            broadcast({"type": "feeds", "feeds": f})
        except Exception as e:
            print("[board]", repr(e))


@app.get("/api/unit")
def api_unit():
    if unit is None:
        return JSONResponse({"error": "no unit"}, status_code=404)
    return {**unit.snapshot(), "feeds": feeds_snapshot(), "ambient": ambient.info() if ambient else None, "camera": camera.source_info if camera else None}


@app.get("/api/unit/log")
def api_unit_log(format: str = "json", kinds: str = None, limit: int = 2000):
    """Today's board log, flat: patient and nurse lines on every bed, acknowledgements, system events, Ascend events.
    format=json | jsonl | csv (download). kinds=patient,nurse,ack,system,ascend"""
    if unit is None:
        return JSONResponse({"error": "no unit"}, status_code=404)
    want = set(kinds.split(",")) if kinds else None
    rows = unit.log_rows(kinds=(want - {"ascend"}) if want else None) if (want is None or want - {"ascend"}) else []
    if ascend and (want is None or "ascend" in want):
        for e in ascend.snapshot(limit=500)["events"]:
            p = e.get("payload") or {}
            rows.append({"ts": e["ts"], "kind": "ascend", "bed": e.get("bed"), "text": e["kind"] + (f" · {p.get('trigger')}" if p.get("trigger") else f" · {p.get('resource')}" if p.get("resource") else ""),
                         "detail": {"direction": e["direction"], "status": e["status"], **{k: v for k, v in p.items() if k not in ("unit",)}}})
    rows.sort(key=lambda r: r["ts"])
    rows = rows[-limit:]
    stamp = time.strftime("%Y-%m-%d")
    if format == "jsonl":
        body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        return Response(body, media_type="application/x-ndjson", headers={"Content-Disposition": f"attachment; filename=unit-log-{stamp}.jsonl"})
    if format == "csv":
        import csv, io
        buf = io.StringIO(); w = csv.writer(buf)
        w.writerow(["time", "kind", "bed", "text", "level", "category", "confidence", "by", "time_to_ack_s", "detail"])
        for r in rows:
            w.writerow([time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["ts"])), r["kind"], r.get("bed") or "", r["text"], r.get("level") or "", r.get("category") or "",
                        r.get("confidence") if r.get("confidence") is not None else "", r.get("by") or "", r.get("time_to_ack_s") if r.get("time_to_ack_s") is not None else "",
                        json.dumps(r["detail"], ensure_ascii=False) if r.get("detail") else ""])
        return Response(buf.getvalue(), media_type="text/csv", headers={"Content-Disposition": f"attachment; filename=unit-log-{stamp}.csv"})
    return {"rows": rows, "now": time.time()}


@app.get("/api/unit/bed")
def api_unit_bed(bed: str):
    if unit is None or bed not in unit.beds:
        return JSONResponse({"error": f"no bed {bed}"}, status_code=404)
    with unit.lock:
        return unit.bed_view(bed, transcript=True)


@app.post("/api/unit/ack")
def api_unit_ack(alert: int, by: str = "charge nurse"):
    try:
        a = unit.ack(alert, by=by)
    except KeyError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    return {"ok": True, "alert": a, "metrics": unit.metrics()}


@app.get("/api/ascend")
def api_ascend():
    """What the board handed to Ascend today (Spark triggers) and what came back (engagements). Simulated: no public API."""
    if ascend is None:
        return JSONResponse({"error": "no ascend bridge"}, status_code=404)
    return ascend.snapshot()


@app.get("/api/ascend/bed")
def api_ascend_bed(bed: str):
    """The Ascend resource and ION-style next best action for one bed's detail view."""
    if unit is None or ascend is None or bed not in unit.beds:
        return JSONResponse({"error": f"no bed {bed}"}, status_code=404)
    with unit.lock:
        v = unit.bed_view(bed, transcript=True)
    return {"resource": ascend.resource_for(v), "next_best_action": ascend.next_best_action(v)}


@app.post("/api/ascend/engage")
def api_ascend_engage(bed: str, action: str, resource: str = None, by: str = "charge nurse"):
    """The nurse opened a resource, asked a medical science liaison, or sent a Wallet card: the engagement Ascend measures."""
    if ascend is None or action not in ("opened", "msl", "wallet"):
        return JSONResponse({"error": "bad action"}, status_code=400)
    rec = ascend.engaged(action, bed, resource=resource, by=by)
    if unit and bed in unit.beds:  # documented on the bed like any other line at the bedside
        note = {"opened": f"Opened Ascend resource: {resource}", "msl": f"Asked a medical science liaison about: {resource}", "wallet": f"Sent Wallet card: {resource}"}[action]
        unit.nurse_said(bed, note, by=by)
    return {"ok": True, "event": rec, "summary": ascend.summary()}


@app.post("/api/unit/nurse")
def api_unit_nurse(bed: str, text: str, by: str = "charge nurse"):
    """A note the charge nurse types on a bed's detail view (documented like a bedside line)."""
    if unit is None or bed not in unit.beds:
        return JSONResponse({"error": f"no bed {bed}"}, status_code=404)
    return {"ok": True, "line": unit.nurse_said(bed, text.strip(), by=by)}


def _voice_id(voice):
    """The ElevenLabs voice id of `voice` (default: the selected voice), or None without a voice or an API key."""
    speaker = voice or STATE.get("voice")
    vid = eltts.voice_for(speaker) if speaker else None
    return vid if vid and eltts.available() else None


def _emotion(emotion, intensity):
    """The emotion the voice delivers for the face's (emotion, intensity): the UI's forced emotion or "match my face" off win."""
    if STATE.get("emotion_override"):
        emotion, intensity = STATE["emotion_override"], max(intensity, 0.7)
    if not STATE.get("expressive", True):
        emotion, intensity = "neutral", 0.0
    return emotion, intensity


def _prefetch_voice(text, expression):
    """Start the voice of `text` as the UI will ask for it (the face's emotion, the natural pace), so its request joins a
    stream that is already running: the raw reading while the LLM decides, which it keeps most of the time."""
    vid = _voice_id(None)
    if vid and not STATE.get("pace"):  # match my pace retimes the whole clip: nothing to stream early
        emotion, intensity = _emotion(expression.get("emotion", "neutral"), expression.get("intensity", 0.0))
        try:
            eltts.prefetch(text, vid, emotion=emotion, intensity=intensity)
        except Exception as e:  # the UI's own request then reports it (and falls back to the browser voice)
            print("[tts] prefetch failed:", e)


def _mp3(audio, cached, **headers):
    from fastapi.responses import Response
    return Response(content=audio, media_type="audio/mpeg", headers={"X-TTS-Cached": str(cached), **headers})


@app.get("/api/tts")
def api_tts(text: str, voice: str = None, cached_only: int = 0):
    """Cloned-voice speech (Plan 3). Returns MP3; 503 if no ElevenLabs key or voice so the UI falls back to the browser voice.
    cached_only=1: 404 instead of synthesizing (the UI's voice bank loads what the server already has, at no ElevenLabs cost)."""
    vid = _voice_id(voice)
    if not vid:
        return JSONResponse({"error": "no cloned voice available"}, status_code=503)
    if cached_only:
        audio = eltts.cached(text, vid)
        return _mp3(audio, True) if audio is not None else JSONResponse({"error": "not cached"}, status_code=404)
    try:
        audio, cached, secs = eltts.synth(text, vid)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=502)
    return _mp3(audio, cached, **{"X-TTS-Seconds": f"{secs:.2f}"})


@app.get("/api/tts_stream")
def api_tts_stream(text: str, voice: str = None):
    """/api/tts for text that is not cached, streamed: the browser starts playing on the first chunk instead of after the
    whole clip. Cached text is served whole."""
    vid = _voice_id(voice)
    if not vid:
        return JSONResponse({"error": "no cloned voice available"}, status_code=503)
    audio = eltts.cached(text, vid)
    if audio is not None:
        return _mp3(audio, True)
    return _stream_mp3(text, vid)


def _stream_mp3(text, vid, on_first=None, on_done=None, **delivery):
    """The ElevenLabs stream of a request that is not cached (tts.synth_stream; delivery = emotion, intensity, rate).
    on_first() runs at its first chunk, on_done() once it completed. An upstream failure is logged and shown as an error;
    the stream then aborts: before any audio the UI falls back to the browser voice, after it the phrase is cut short."""
    def gen():
        first = True
        try:
            for chunk in eltts.synth_stream(text, vid, **delivery):
                if first and on_first:
                    on_first()
                first = False
                yield chunk
        except Exception as e:  # not GeneratorExit: a client that stops listening is not an error
            traceback.print_exc()
            broadcast({"type": "error", "message": f"ElevenLabs stream failed for {text!r}: {e}"})
            raise
        if on_done:
            on_done()
    return StreamingResponse(gen(), media_type="audio/mpeg", headers={"X-TTS-Cached": "False", "Cache-Control": "no-store"})


@app.get("/api/say")
def api_say(text: str, emotion: str = "neutral", intensity: float = 0.0, rate: float = 1.0, voice: str = None, retime: int = 1, utt_id: int = 0):
    """Expressive delivery: ElevenLabs (tag/stability/speed from emotion+intensity+rate) then per-word retiming to the
    mouthed word durations of utterance `utt_id` (if it is the last one). Returns WAV + X-Delivery header with the report;
    with nothing to retime, the MP3 (streamed as it is generated if not cached: v3 synthesized whole took 1.1-3.1 s in
    the session logs before a sound played)."""
    from fastapi.responses import Response
    vid = _voice_id(voice)
    if not vid:
        return JSONResponse({"error": "no ElevenLabs voice available"}, status_code=503)
    emotion, intensity = _emotion(emotion, intensity)
    t0 = time.time()
    plan = eltts.plan_delivery(text, emotion, intensity, rate)
    report = {"emotion": emotion, "intensity": intensity, "rate": rate, "tag": plan["tag"], "model": plan["model"], "stability": plan["settings"]["stability"],
              "retime": {"applied": False}}
    timing = LAST.get("timing") if utt_id and utt_id == LAST.get("utt_id") else None
    if retime and timing and prosody._mms is None:  # never a 2.6-25 s aligner load inside a spoken reply (_load_aligner)
        report["retime"] = {"applied": False, "reason": "voice aligner not loaded yet: match my pace was just switched on, or failed to load"}
        retime = 0
    delivery = {"emotion": emotion, "intensity": intensity, "rate": rate}
    if not (retime and timing and timing.get("words")):  # nothing to retime: the clip as synthesized, no mp3 -> wav decode
        mp3 = eltts.cached(text, vid, **delivery)
        if mp3 is None:
            def first():
                report.update(cached=False, streamed=True, t_synth=round(time.time() - t0, 2))  # t_synth: to the first chunk
            def done():
                report["total"] = round(time.time() - t0, 2)
                broadcast({"type": "delivery", "utt_id": utt_id, **report})
            return _stream_mp3(text, vid, first, done, **delivery)
        report.update(cached=True, t_synth=0.0, total=round(time.time() - t0, 2))
        broadcast({"type": "delivery", "utt_id": utt_id, **report})
        return _mp3(mp3, True, **{"X-Delivery": json.dumps({k: v for k, v in report.items() if k != "retime"})})
    try:
        mp3, cached, secs = eltts.synth(text, vid, **delivery)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=502)
    report.update(cached=cached, t_synth=round(time.time() - t0, 2))
    wav, sr = prosody.mp3_to_wav(mp3)
    t1 = time.time()
    try:
        aw = prosody.align_audio_words(wav, sr, [w["word"] for w in timing["words"]])
        wav, rep = prosody.retime(wav, sr, aw, timing["words"], timing["pauses"])
        rep["audio_words"] = aw; rep["t"] = round(time.time() - t1, 2)
        report["retime"] = rep
    except Exception as e:
        report["retime"] = {"applied": False, "reason": str(e)}
    report["total"] = round(time.time() - t0, 2)
    broadcast({"type": "delivery", "utt_id": utt_id, **report})
    return Response(content=prosody.wav_bytes(wav, sr), media_type="audio/wav", headers={"X-Delivery": json.dumps({k: v for k, v in report.items() if k != "retime"})})


@app.post("/api/voice/clone_upload")
async def api_voice_clone_upload(speaker: str, request: Request):
    """Clone from audio recorded in the browser (webm/opus or wav body). Converts with ffmpeg, clones, prewarms in background."""
    body = await request.body()
    if len(body) < 20000:
        return JSONResponse({"error": "recording too short"}, status_code=400)
    d = os.path.join(ROOT, "data", "session", speaker, "voiced"); os.makedirs(d, exist_ok=True)
    raw = os.path.join(d, f"browser_{int(time.time())}.webm"); wav = raw[:-5] + ".wav"
    open(raw, "wb").write(body)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-ar", "16000", "-ac", "1", wav])
    if not os.path.exists(wav):
        return JSONResponse({"error": "could not decode recording"}, status_code=400)
    try:
        rec = eltts.create_voice(speaker, [wav])
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=502)
    STATE["voice"] = speaker
    broadcast({"type": "state", "state": STATE})
    def _pw():
        n = eltts.prewarm_phrase_bank(rec["voice_id"], phrases)
        broadcast({"type": "prewarmed", "speaker": speaker, "n": n})
    threading.Thread(target=_pw, daemon=True).start()
    return {**rec, "prewarming": True}


@app.get("/api/voices")
def api_voices():
    return {"available": eltts.available(), "voices": eltts.list_voices(),
            "stock": eltts.stock_voices()[:12], "selected": STATE.get("voice")}


@app.post("/api/voice/clone")
def api_voice_clone(speaker: str):
    """Clone from the voiced pass of data/session/<speaker> and prewarm the phrase bank."""
    sdir = os.path.join(ROOT, "data", "session", speaker, "manifest.jsonl")
    if not os.path.exists(sdir):
        return JSONResponse({"error": "no session for speaker"}, status_code=404)
    wavs = [os.path.join(ROOT, r["wav"]) for r in map(json.loads, open(sdir)) if r.get("wav")]
    if not wavs:
        return JSONResponse({"error": "no voiced takes"}, status_code=400)
    try:
        rec = eltts.create_voice(speaker, wavs)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=502)
    STATE["voice"] = speaker
    broadcast({"type": "state", "state": STATE})
    threading.Thread(target=lambda: print("[tts] prewarmed", eltts.prewarm_phrase_bank(rec["voice_id"], phrases)), daemon=True).start()
    return {**rec, "prewarming": True}


@app.get("/api/state")
def api_state():
    return {"state": STATE, "phrases": phrases, "phrase_table": PHRASE_TABLE, "log": LOG[-50:], "context": context.snapshot(), "camera": camera.meta if camera else None,
            "engine": {"device": str(engine.device), "decode_device": str(engine.decode_device), "model": "Auto-AVSR " + os.path.basename(STATE.get("model_dir", "LRS3_V_WER19.1"))},
            "llm": {"model": llm.model if llm else None, "available": bool(llm and llm.available())}}


@app.post("/api/signal")
def api_signal(kind: str, value: str = None, confidence: float = 1.0):
    """MOCK: inject a nonverbal signal as if the camera had detected it (e.g. kind=nod). Only with MOCK_SIGNALS=1."""
    if os.environ.get("MOCK_SIGNALS") != "1":
        return JSONResponse({"error": "mock signals disabled; start the server with MOCK_SIGNALS=1"}, status_code=403)
    _on_signal({"kind": kind, "value": value, "confidence": confidence, "mock": True})
    return {"ok": True}


@app.post("/api/source")
def api_source(spec: str):
    """Switch the live video source without restarting (e.g. to a recorded backup): webcam[:N] | usb[:N|name] | file:path.mp4"""
    global camera
    from silent_running.sources import make_source
    try:
        make_source(spec)  # bad spec or missing file: refuse before tearing down the working source
    except (ValueError, OSError) as e:
        broadcast({"type": "error", "message": f"video source: {e}"})
        return JSONResponse({"error": str(e)}, status_code=400)
    if camera is None:
        camera = _make_camera(spec)
        for _ in range(300):
            if camera.opened or camera.fatal: break
            time.sleep(0.1)
        if not camera.opened:
            return JSONResponse({"error": camera.fatal or "source did not open in time", "source": spec}, status_code=400)
    else:
        try:
            camera.switch_source(spec)  # make-before-break: on failure the current source keeps running
        except RuntimeError as e:
            broadcast({"type": "error", "message": f"video source: {e}"})
            return JSONResponse({"error": str(e), "source": spec}, status_code=400)
    return {"ok": True, "source": camera.source_info}


@app.post("/api/zoom")
def api_zoom(zoom: float):
    """The ESP32-CAM's sensor zoom, live: 1 = the whole view, 2 = its centre half at 2x the pixels (sources.ov3660_window)."""
    if camera is None or not camera.opened:
        return JSONResponse({"error": "no camera"}, status_code=400)
    try:
        info = camera.set_zoom(zoom)
    except (RuntimeError, queue.Empty) as e:
        msg = str(e) or "the capture process did not answer"
        broadcast({"type": "error", "message": f"zoom {zoom:g}x: {msg}"})
        return JSONResponse({"error": msg, "source": camera.source_info}, status_code=400)
    sessionlog.log("zoom", zoom=zoom, source=info)
    return {"ok": True, "source": info}


@app.post("/api/rotate")
def api_rotate(degrees: int):
    """Turn the camera image clockwise by 0/90/180/270 degrees, live (a camera mounted sideways on the glasses)."""
    if degrees % 90 or not 0 <= degrees < 360:
        return JSONResponse({"error": "degrees must be 0, 90, 180 or 270"}, status_code=400)
    if camera is None or not camera.opened:
        return JSONResponse({"error": "no camera"}, status_code=400)
    try:
        info = camera.set_rotate(degrees)
    except (RuntimeError, queue.Empty) as e:
        msg = str(e) or "the capture process did not answer"
        broadcast({"type": "error", "message": f"rotate {degrees}: {msg}"})
        return JSONResponse({"error": msg, "source": camera.source_info}, status_code=400)
    sessionlog.log("rotate", degrees=degrees, source=info)
    return {"ok": True, "source": info}


def _file_rois(path):
    """Video file -> (abs path, mouth crops, n_frames, crop seconds, fps), or a JSONResponse error."""
    p = path if os.path.isabs(path) else os.path.join(ROOT, path)
    if not os.path.exists(p):
        return JSONResponse({"error": "not found"}, status_code=404)
    frames = _read_video(p)[0].numpy()
    t0 = time.time()
    lms = engine.landmarks_for_frames(frames)
    n_face = sum(l is not None for l in lms)
    if n_face < max(4, len(lms) // 4):
        return JSONResponse({"error": f"face not tracked ({n_face}/{len(lms)} frames)"}, status_code=400)
    try:
        rois = engine.mouth_rois(frames, lms)
    except Exception as e:
        return JSONResponse({"error": f"crop failed: {e}"}, status_code=400)
    with av.open(p) as c:
        fps = float(c.streams.video[0].average_rate)
    return p, rois, len(frames), time.time() - t0, fps


@app.post("/api/decode_file")
def api_decode_file(path: str, label: str = None):
    """Run the identical pipeline on a video file (for evaluation / fallback demos)."""
    r = _file_rois(path)
    if isinstance(r, JSONResponse):
        return r
    p, rois, n, t_crop, _ = r
    res = run_decode(rois, n / 25.0, source=os.path.relpath(p, ROOT), label=label, t_crop=t_crop)
    return res or {"error": "decode failed"}


# ----------------------------------------------------------------------------- WebSocket
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    try:
        await ws.send_text(json.dumps({"type": "hello", "state": STATE, "phrases": phrases, "phrase_table": PHRASE_TABLE, "context": context.snapshot(), "log": LOG[-30:]}))
        while True:
            msg = json.loads(await ws.receive_text())
            cmd = msg.get("cmd")
            if cmd == "start":
                try:
                    camera.start_listening()
                    set_status("listening")
                except Exception as e:
                    set_status("idle")
                    await ws.send_text(json.dumps({"type": "error", "message": f"camera error: {e}"}))
            elif cmd == "stop":
                set_status("processing", stage="crop")
                threading.Thread(target=_stop_and_decode, daemon=True).start()
            elif cmd == "settings":
                for k in ("llm_enabled", "voice", "expressive", "emotion_override"):
                    if k in msg: STATE[k] = msg[k]
                if "pace" in msg:  # "match my pace": the voice aligner it needs loads now, not at every startup
                    STATE["pace"] = bool(msg["pace"])
                    if STATE["pace"]:
                        threading.Thread(target=_load_aligner, daemon=True).start()
                if "auto_listen" in msg and camera:
                    STATE["auto_listen"] = bool(msg["auto_listen"]); camera.set_auto(STATE["auto_listen"])
                broadcast({"type": "state", "state": STATE})
            elif cmd == "context":
                context.update(notes=msg.get("notes"), category=msg.get("category"), last_prompt=msg.get("last_prompt"))
                broadcast({"type": "context", "context": context.snapshot()})
            elif cmd == "nurse":
                threading.Thread(target=nurse_listen, args=(float(msg.get("seconds", 5)),), daemon=True).start()
            elif cmd == "nurse_text":
                t = (msg.get("text") or "").strip()
                if t:
                    context.update(last_prompt=t); _log("nurse", t); broadcast({"type": "context", "context": context.snapshot()})
            elif cmd == "confirm":  # the nurse picked a reading for the patient (adds to history)
                context.add_history(msg.get("text", ""))
                broadcast({"type": "context", "context": context.snapshot()})
            elif cmd == "label":  # "what was actually said" for a captured utterance (data/captures, scripts/captures.py)
                if capture is None:
                    await ws.send_text(json.dumps({"type": "error", "message": "label not saved: capture is off (SR_CAPTURE=0)"}))
                    continue
                try:
                    capture.label(msg["utt_id"], msg["text"].strip(), msg.get("by", "ui"))
                except KeyError as e:  # an utterance this session did not capture (e.g. a file replay)
                    await ws.send_text(json.dumps({"type": "error", "message": f"label not saved: {e}"}))
                    continue
                broadcast({"type": "labeled", "utt_id": msg["utt_id"], "text": msg["text"].strip()})
            elif cmd == "save_sample":
                _save_sample(msg.get("phrase", ""), msg.get("speaker", "unknown"))
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(ws)


def _stop_and_decode():
    try:
        u = camera.stop_listening()
    except Exception as e:
        set_status("idle"); broadcast({"type": "error", "message": f"camera error: {e}"}); return
    _decode_utterance(u, _source_kind())


def _source_kind():
    return (camera.source_info or {}).get("kind", "camera") if camera else "camera"


def _make_camera(spec):
    cam = CameraProcess(source=spec)
    cam.on_auto_utterance = _on_auto_utterance
    cam.on_error = lambda m: (sessionlog.log("camera_error", message=m), broadcast({"type": "error", "message": f"camera: {m}"}))  # video source and hand-signal errors
    cam.on_signal = _on_camera_signal
    cam.on_ahead = _on_ahead
    return cam


def _on_auto_utterance(u):
    source = f"{_source_kind()}-auto"
    if u.get("ahead") is None:
        set_status("processing", stage="auto")
        threading.Thread(target=_decode_utterance, args=(u, source), daemon=True).start()
        return
    a = AHEAD[u["ahead"]] = _Ahead()
    threading.Thread(target=_decode_ahead, args=(u, source, a), daemon=True).start()


def _on_ahead(aid, confirmed):
    a = AHEAD.pop(aid, None)
    if a is None:
        return
    if confirmed:
        set_status("processing", stage="auto")
    a.settle(confirmed)


def _decode_ahead(u, source, a):
    """An auto utterance sent ahead of its confirmed end: read now, shown once confirmed (run_decode)."""
    if u["rois"] is None:  # an error to report: only once confirmed
        if a.wait():
            _decode_utterance(u, source)
        return
    run_decode(u["rois"], u["duration"], source=source, t_crop=u["t_crop"], expression=u.get("expression"), nonverbal=u.get("nonverbal"),
               mouth_px=u.get("mouth_px"), ahead=a)


def _decode_utterance(u, source="webcam"):
    if u["rois"] is None:
        set_status("idle")
        msgs = {"too short": "Utterance too short. Hold Listen while you mouth the phrase.", "face not tracked": f"Face not tracked well enough ({u['n_face']}/{u['n_total']} frames). Face the camera and try again."}
        broadcast({"type": "error", "message": msgs.get(u["error"], u["error"] or "capture failed")})
        return
    run_decode(u["rois"], u["duration"], source=source, t_crop=u["t_crop"], expression=u.get("expression"), nonverbal=u.get("nonverbal"), mouth_px=u.get("mouth_px"))


def _save_sample(phrase, speaker):
    """Save the last utterance window as an eval sample (same format as scripts/record_samples.py)."""
    import cv2
    frames = camera.snapshot_last(6.0)
    if frames is None:
        return
    d = os.path.join(ROOT, "data", "eval", speaker)
    os.makedirs(d, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "_", phrase.lower()).strip("_") or "sample"
    k = 0
    while os.path.exists(os.path.join(d, f"{slug}__{k}.mp4")):
        k += 1
    fn = os.path.join(d, f"{slug}__{k}.mp4")
    tmp = fn + ".raw.mp4"
    h, w = frames.shape[1:3]
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()
    os.system(f'ffmpeg -y -loglevel error -i "{tmp}" -an -r 25 -c:v libx264 -crf 18 -pix_fmt yuv420p "{fn}" && rm "{tmp}"')
    with open(os.path.join(ROOT, "data", "eval", "manifest.jsonl"), "a") as f:
        f.write(json.dumps({"speaker": speaker, "phrase": phrase, "file": os.path.relpath(fn, ROOT), "n_frames": len(frames), "fps": 25, "ts": time.time(), "source": "ui"}) + "\n")
    broadcast({"type": "saved", "file": os.path.relpath(fn, ROOT), "phrase": phrase})


@app.websocket("/ws_meta")
async def ws_meta(ws: WebSocket):
    """Lightweight 10 Hz tracking metadata stream."""
    await ws.accept()
    try:
        while True:
            await ws.send_text(json.dumps({"type": "meta", **(camera.meta if camera else {}), "status": STATE["status"]}))
            await asyncio.sleep(0.1)
    except Exception:
        pass


@app.on_event("startup")
async def _startup():
    global loop
    loop = asyncio.get_event_loop()


def main():
    global engine, camera, context, llm, capture, unit, ambient, ascend
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None, help="video source: webcam[:N] | usb[:N|name] | file:path.mp4[?loop=0&realtime=0] | stream:<ESP32 host>[?window=2x] | serial[?window=2x] (ESP32-CAM over USB) (default: webcam auto-detect)")
    ap.add_argument("--camera", type=int, default=-1, help="shorthand for --source webcam:N; -1 = auto-detect first live camera")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--decode-device", default="cpu")
    ap.add_argument("--ctc-weight", type=float, default=0.1)
    ap.add_argument("--llm-provider", default="grok", choices=["grok", "openai"], help="the interpreter: grok (XAI_API_KEY) | openai (OPENAI_API_KEY)")
    ap.add_argument("--llm", default=None, help="model name (default: the provider's, see context.LLM_PROVIDERS)")
    ap.add_argument("--model-dir", default=None, help="VSR checkpoint dir (default models/LRS3_V_WER19.1; use models/adapted_<name> after scripts/adapt.py)")
    ap.add_argument("--voice", default=None, help="default TTS voice: cloned speaker name or stock ElevenLabs voice name (e.g. Bella)")
    ap.add_argument("--no-camera", action="store_true")
    ap.add_argument("--ambient", default="auto", help="ambient overview camera for the dashboard: auto (the first laptop camera that delivers frames), an index, or none")
    ap.add_argument("--bed", default=os.environ.get("SR_BED", "4"), help="the live patient's bed number on the dashboard")
    ap.add_argument("--initials", default=os.environ.get("SR_INITIALS", "J.G."), help="the live patient's initials on the dashboard")
    ap.add_argument("--ascend-webhook", default=os.environ.get("ASCEND_WEBHOOK", ""), help="POST Spark triggers here (an Ascend endpoint, if one ever exists); empty = simulated delivery")
    ap.add_argument("--no-simulate", action="store_true", help="dashboard: no simulated beds' activity")
    args = ap.parse_args()
    if (args.source or "").startswith("serial"):
        # The ESP32-CAM's CH340 buffers 32 bytes (~0.2 ms at 1.5 Mbaud) and macOS drains it from a user-space driver
        # (com.apple.DriverKit-AppleUSBCHCOM) that competes for the CPU: when it runs late, bytes are lost and it can wedge.
        # Our processes (the capture process inherits this) yield to it. 3-min soaks under load: 1 wedge, 27.1 fps, 3.2%
        # damaged frames at normal priority; none, 30.2 fps, 2.0% with the load at lower priority.
        os.nice(10)
    t0 = time.time()
    engine = VSREngine(device=args.device, decode_device=args.decode_device, beam_size=10, ctc_weight=args.ctc_weight, **({"model_dir": args.model_dir} if args.model_dir else {}))
    STATE["model_dir"] = os.path.relpath(args.model_dir, ROOT) if args.model_dir else "models/LRS3_V_WER19.1"
    if args.voice:
        STATE["voice"] = args.voice
    engine.warmup()
    print(f"[engine] loaded + warmed in {time.time()-t0:.1f}s (encoder on {engine.device}, decoder on {engine.decode_device}, verification on {engine.score_device})")
    context = ContextStore()
    if os.environ.get("SR_CAPTURE", "1") != "0":  # data/captures: every live utterance + corrections (silent_running/captures.py)
        capture = captures.CaptureLog({"source": args.source or (f"webcam:{args.camera}" if args.camera >= 0 else "webcam"),
                                       "model_dir": STATE["model_dir"],
                                       "scoring": {k: v for k, v in os.environ.items() if k.startswith("SR_")}})
        print(f"[capture] logging live utterances to {os.path.relpath(capture.path, ROOT)}")
    ascend = AscendBridge(emit=broadcast, webhook=args.ascend_webhook or None)
    def _unit_emit(m):  # every unit event reaches the board, and the Ascend seam sees it too
        broadcast(m)
        try:
            ascend.on_unit_event(m)
        except Exception as e:
            print("[ascend]", repr(e))
    unit = Unit(real_bed=args.bed, real_initials=args.initials, emit=_unit_emit, simulate=not args.no_simulate)
    print(f"[unit] {len(unit.beds)} beds (live: bed {unit.real_bed}), {len(unit.alerts)} alerts on file for today")
    if args.ambient != "none":
        def _bedside_index():  # the laptop webcam the bedside feed holds, if that is what it is right now
            info = (camera.source_info or {}) if camera else {}
            return info.get("index") if info.get("kind") == "webcam" else None
        ambient = AmbientCamera(index=None if args.ambient == "auto" else int(args.ambient), avoid=_bedside_index)
    threading.Thread(target=_signal_worker, daemon=True).start()  # camera signals, including cameras created later via /api/source
    llm = LLMInterpreter(provider=args.llm_provider, model=args.llm)
    print(f"[llm] {llm.provider} {llm.model} available={llm.available()}" + ("" if llm.available() else f" (set {llm.key_var} in .env)"))
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    except Exception:
        sha = None
    logp = sessionlog.start(args=vars(args), git=sha, model_dir=STATE.get("model_dir"), pid=os.getpid())
    print(f"[sessionlog] {os.path.relpath(logp, ROOT)}")
    if not args.no_camera:
        camera = _make_camera(args.source or (f"webcam:{args.camera}" if args.camera >= 0 else "webcam"))
        for _ in range(400):  # child imports torch/mediapipe first (~10-20 s)
            if camera.opened or camera.fatal: break
            time.sleep(0.1)
        print(f"[camera] opened={camera.opened} source={camera.source_info or camera.source}")
        sessionlog.log("camera_opened", opened=camera.opened, fatal=camera.fatal, source=camera.source_info or str(camera.source))
    threading.Thread(target=_keep_warm, daemon=True).start()
    threading.Thread(target=_link_monitor, daemon=True).start()
    unit.system(f"Server started ({sha or 'no git'}), bedside source {args.source or 'webcam'}", git=sha, source=args.source or "webcam")
    threading.Thread(target=_board_monitor, daemon=True).start()  # the board's feeds and its log follow the real cameras
    STATE["warm"] = True  # engine warmed up and the camera started (or failed, reported): latency from now on is steady state
    def _prewarm_tts():  # the ElevenLabs SDK import and client, and the stock voice list the UI asks for first
        t = time.time()
        if eltts.available():
            eltts.stock_voices()
        print(f"[tts] ElevenLabs client ready in {time.time() - t:.1f}s (available={eltts.available()})")
    threading.Thread(target=_prewarm_tts, daemon=True).start()
    import uvicorn
    # an open preview (/stream) never ends on its own: give open connections 3 s on shutdown, then close them
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning", timeout_graceful_shutdown=3)


if __name__ == "__main__":
    main()
